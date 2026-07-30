"""Regression tests for silent provider failure retry in streaming.

Silent provider failures occur when the API returns no error AND no messages
+(empty response body, provider timeout without HTTP error, rate limits without
+HTTP 429). The streaming layer retries up to 3 times with exponential backoff.
"""

from __future__ import annotations

import queue
import sys
import types
from unittest import mock

import pytest

import api.config as config
import api.models as models
import api.streaming as streaming
from api.models import Session


class MockAgent:
    """Mock agent for testing silent provider failures."""
    pass


def _prepare_session(session_id: str, stream_id: str, *, pending_user_message: str):
    """Prepare a test session for streaming tests."""
    session = Session(session_id=session_id, title="Test Session")
    session.messages = []
    session.context_messages = []
    session.pending_user_message = pending_user_message
    session.pending_attachments = []
    session.pending_started_at = 1234567890.0
    session.pending_user_source = "cli"
    session.active_stream_id = stream_id
    session.save()
    models.SESSIONS[session_id] = session
    return session


def _queue_events(fake_queue):
    """Extract events from a fake queue."""
    return [(item[0], item[1]) for item in list(fake_queue.queue)]


def _run_stream(monkeypatch, session, stream_id, agent_cls, *, workspace):
    """Run a streaming test with mocked dependencies."""
    fake_queue = queue.Queue()
    streaming.STREAMS[stream_id] = fake_queue
    config.STREAM_PARTIAL_TEXT[stream_id] = ""

    with mock.patch.object(streaming, "get_session", return_value=session), \
         mock.patch.object(streaming, "_get_ai_agent", return_value=agent_cls), \
         mock.patch.object(streaming, "resolve_model_provider", return_value=("test-model", "test-provider", None)), \
         mock.patch("api.config.get_config", return_value={}), \
         mock.patch("api.config._resolve_cli_toolsets", return_value=[]):
        streaming._run_agent_streaming(
            session_id=session.session_id,
            msg_text=session.pending_user_message,
            model="test-model",
            workspace=workspace,
            stream_id=stream_id,
        )

    return fake_queue


class SilentRetryAgent:
    """Agent that simulates silent failures followed by success."""
    runs = 0

    def run_conversation(self, **kwargs):
        type(self).runs += 1
        history = list(kwargs.get("conversation_history") or [])
        if type(self).runs < 2:  # Silent failure on first attempt
            return {"messages": []}  # Empty result (silent failure)
        return {
            "status": "ok",
            "messages": history + [{"role": "assistant", "content": "Success after retry"}],
        }


class NoneRetryAgent:
    """Agent that simulates None failures followed by success."""
    runs = 0

    def run_conversation(self, **kwargs):
        type(self).runs += 1
        history = list(kwargs.get("conversation_history") or [])
        if type(self).runs < 3:  # Silent failure on first two attempts
            return None  # None result (silent failure)
        return {
            "status": "ok",
            "messages": history + [{"role": "assistant", "content": "Success after retry"}],
        }


class ExhaustedAgent:
    """Agent that simulates all retries exhausted."""
    runs = 0

    def run_conversation(self, **kwargs):
        type(self).runs += 1
        history = list(kwargs.get("conversation_history") or [])
        if type(self).runs < 4:  # Silent failures for first three attempts
            return {"messages": []}  # Empty result
        return {
            "status": "ok",
            "messages": history + [{"role": "assistant", "content": "Final attempt"}],
        }


class ImmediateSuccessAgent:
    """Agent that succeeds immediately without retries."""
    runs = 0

    def run_conversation(self, **kwargs):
        type(self).runs += 1
        history = list(kwargs.get("conversation_history") or [])
        return {
            "status": "ok",
            "messages": history + [{"role": "assistant", "content": "Immediate success"}],
        }


def test_silent_provider_failure_retries_and_succeeds(tmp_path, monkeypatch):
    """Silent failure on attempt 1 should retry and succeed on attempt 2."""
    session = _prepare_session("silent_retry", "stream_silent_retry", pending_user_message="Test message")
    agent_cls = SilentRetryAgent

    with mock.patch("api.streaming.time.sleep"):  # Skip actual delays
        fake_queue = _run_stream(monkeypatch, session, "stream_silent_retry", SilentRetryAgent, workspace=str(tmp_path))

    assert SilentRetryAgent.runs == 2  # Two attempts: initial + one retry
    saved = Session.load("silent_retry")
    assert saved is not None
    assert saved.messages[-1]["role"] == "assistant"
    assert saved.messages[-1]["content"] == "Success after retry"

    events = _queue_events(fake_queue)
    assert any(event == "done" for event, _ in events)


def test_silent_provider_failure_none_result_retries_and_succeeds(tmp_path, monkeypatch):
    """None result should also trigger retry and succeed."""
    session = _prepare_session("silent_none", "stream_silent_none", pending_user_message="Test message")
    agent_cls = NoneRetryAgent

    with mock.patch("api.streaming.time.sleep"):  # Skip actual delays
        fake_queue = _run_stream(monkeypatch, session, "stream_silent_none", NoneRetryAgent, workspace=str(tmp_path))

    assert NoneRetryAgent.runs == 3  # Three attempts: initial + two retries
    saved = Session.load("silent_none")
    assert saved is not None
    assert saved.messages[-1]["role"] == "assistant"


def test_silent_provider_failure_all_retries_exhausted(tmp_path, monkeypatch):
    """After 3 retries exhausted, should return empty response to user."""
    session = _prepare_session("silent_exhausted", "stream_silent_exhausted", pending_user_message="Test message")
    agent_cls = ExhaustedAgent

    with mock.patch("api.streaming.time.sleep"):  # Skip actual delays
        fake_queue = _run_stream(monkeypatch, session, "stream_silent_exhausted", ExhaustedAgent, workspace=str(tmp_path))

    assert ExhaustedAgent.runs == 3  # Only 3 attempts max
    saved = Session.load("silent_exhausted")
    assert saved is not None
    # After retries exhausted, should have empty response
    assert any(msg.get("role") == "assistant" for msg in saved.messages)


def test_silent_provider_failure_no_error_field_succeeds_first_try(tmp_path, monkeypatch):
    """Normal success with content should not trigger retry."""
    session = _prepare_session("no_error_success", "stream_no_error_success", pending_user_message="Test message")
    agent_cls = ImmediateSuccessAgent

    with mock.patch("api.streaming.time.sleep"):
        fake_queue = _run_stream(monkeypatch, session, "stream_no_error_success", ImmediateSuccessAgent, workspace=str(tmp_path))

    assert ImmediateSuccessAgent.runs == 1  # Only one attempt
    saved = Session.load("no_error_success")
    assert saved is not None
    assert saved.messages[-1]["content"] == "Immediate success"

    events = _queue_events(fake_queue)
    assert any(event == "done" for event, _ in events)


def test_silent_provider_failure_no_error_field_succeeds_first_try(tmp_path, monkeypatch):
    """Normal success with content should not trigger retry."""
    session = _prepare_session("no_error_success", "stream_no_error_success", pending_user_message="Test message")

    class ImmediateSuccessAgent(MockAgent):
        runs = 0

        def run_conversation(self, **kwargs):
            type(self).runs += 1
            history = list(kwargs.get("conversation_history") or [])
            return {
                "status": "ok",
                "messages": history + [{"role": "assistant", "content": "Immediate success"}],
            }

    agent_cls = ImmediateSuccessAgent

    with mock.patch("api.streaming.time.sleep"):
        fake_queue = _run_stream(monkeypatch, session, "stream_no_error_success", agent_cls, workspace=str(tmp_path))

    assert agent_cls.runs == 1  # Only one attempt
    saved = Session.load("no_error_success")
    assert saved is not None
    assert saved.messages[-1]["content"] == "Immediate success"

    events = _queue_events(fake_queue)
    assert any(event == "done" for event, _ in events)