import pytest

from api import streaming


class _FakeAgent:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []
        self._status_callback = None
        self._last_error = None

    def run_conversation(self, **kwargs):
        self.calls.append(kwargs)
        if self.results:
            result = self.results.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        return {"messages": []}


class _FakeCancelEvent:
    def __init__(self):
        self._set = False

    def is_set(self):
        return self._set

    def set(self):
        self._set = True


@pytest.mark.parametrize(
    ("result_a", "result_b", "expected_calls", "expected_messages"),
    [
        ({"messages": []}, {"messages": [{"role": "assistant", "content": "ok"}]}, 2, [{"role": "assistant", "content": "ok"}]),
        ({"messages": [{"role": "assistant", "content": "ok"}]}, {"messages": [{"role": "assistant", "content": "ok"}]}, 1, [{"role": "assistant", "content": "ok"}]),
    ],
)
def test_silent_provider_retry_retries_only_when_turn_is_silent(result_a, result_b, expected_calls, expected_messages):
    agent = _FakeAgent([result_a, result_b])
    cancel_event = _FakeCancelEvent()
    sleep_calls = []

    result = streaming._run_conversation_with_silent_provider_retry(
        agent,
        run_kwargs={},
        previous_context_messages=[{"role": "user", "content": "hello"}],
        msg_text="hello",
        cancel_event=cancel_event,
        flush_reasoning_buffer=lambda: None,
        sleep_fn=lambda delay: sleep_calls.append(delay),
    )

    assert len(agent.calls) == expected_calls
    assert result.get("messages") == expected_messages
