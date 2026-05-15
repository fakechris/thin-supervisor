from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from supervisor.loop import SupervisorLoop
from supervisor.plan.loader import load_spec
from supervisor.config import RuntimeConfig
from supervisor.runtime_recovery import (
    RuntimeRecoveryPolicy,
    detect_runtime_recovery,
    next_allowed_recovery_at,
    policy_with_profile,
)
from supervisor.storage.state_store import StateStore


class _Terminal:
    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.injected: list[str] = []
        self._read_done = False
        self.read_count = 0

    def read(self, lines: int = 100) -> str:
        self._read_done = True
        self.read_count += 1
        if self.outputs:
            return self.outputs.pop(0)
        return ""

    def inject(self, text: str) -> None:
        assert self._read_done
        self.injected.append(text)
        self._read_done = False


class _StopAfterRetry:
    def __init__(self, terminal: _Terminal):
        self.terminal = terminal

    def is_set(self) -> bool:
        return any(text.startswith("retry") for text in self.terminal.injected)


class _StopAfterReads:
    def __init__(self, terminal: _Terminal, count: int):
        self.terminal = terminal
        self.count = count

    def is_set(self) -> bool:
        return self.terminal.read_count >= self.count


def _checkpoint(status: str = "working") -> str:
    return (
        "<checkpoint>\n"
        f"status: {status}\n"
        "current_node: write_test\n"
        "summary: recovered with real work\n"
        "evidence:\n"
        "  - command: pytest\n"
        "candidate_next_actions:\n"
        "  - continue\n"
        "needs:\n"
        "  - none\n"
        "question_for_supervisor:\n"
        "  - none\n"
        "</checkpoint>\n"
    )


def test_detects_claude_429_reset_as_utc8_and_adds_retry_grace() -> None:
    text = (
        "API Error: Request rejected (429) · 已达到 5 小时的使用上限。"
        "您的限额将在 2026-05-15 04:07:12 重置。"
    )
    now = datetime(2026, 5, 14, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    policy = RuntimeRecoveryPolicy(reset_timezone="Asia/Shanghai", reset_grace_seconds=60)

    retry = detect_runtime_recovery(text, now=now, policy=policy)

    assert retry is not None
    assert retry.kind == "rate_limit"
    assert retry.retry_at == datetime(2026, 5, 15, 4, 8, 12, tzinfo=ZoneInfo("Asia/Shanghai"))


def test_retry_policy_skips_configured_utc8_high_watermark() -> None:
    policy = RuntimeRecoveryPolicy(quiet_windows=("UTC+8 12:00-18:00",))
    candidate = datetime(2026, 5, 15, 4, 30, tzinfo=timezone.utc)

    retry_at = next_allowed_recovery_at(candidate, policy)

    assert retry_at == datetime(2026, 5, 15, 10, 0, tzinfo=timezone.utc)


def test_detects_malformed_200_gateway_error_for_short_retry() -> None:
    text = (
        "API Error: API returned an empty or malformed response (HTTP 200) "
        "— check for a proxy or gateway intercepting the request"
    )
    now = datetime(2026, 5, 14, 12, 0, tzinfo=timezone.utc)
    policy = RuntimeRecoveryPolicy(transient_delay_seconds=90)

    retry = detect_runtime_recovery(text, now=now, policy=policy)

    assert retry is not None
    assert retry.kind == "transient_connectivity"
    assert retry.retry_at == datetime(2026, 5, 14, 12, 1, 30, tzinfo=timezone.utc)


def test_detects_english_expires_at_reset_text() -> None:
    text = "API Error: Request rejected (429). Limit expires at 2026-05-15 04:07:12."
    now = datetime(2026, 5, 14, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    policy = RuntimeRecoveryPolicy(reset_timezone="Asia/Shanghai", reset_grace_seconds=60)

    retry = detect_runtime_recovery(text, now=now, policy=policy)

    assert retry is not None
    assert retry.retry_at == datetime(2026, 5, 15, 4, 8, 12, tzinfo=ZoneInfo("Asia/Shanghai"))


def test_detects_timestamp_only_reset_text() -> None:
    text = "Limit expires at 2026-05-15 04:07:12."
    now = datetime(2026, 5, 14, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    policy = RuntimeRecoveryPolicy(reset_timezone="Asia/Shanghai", reset_grace_seconds=60)

    retry = detect_runtime_recovery(text, now=now, policy=policy)

    assert retry is not None
    assert retry.kind == "rate_limit"
    assert retry.retry_at == datetime(2026, 5, 15, 4, 8, 12, tzinfo=ZoneInfo("Asia/Shanghai"))


def test_glm_profile_reserves_midday_and_uses_five_hour_fallback() -> None:
    policy = policy_with_profile(RuntimeRecoveryPolicy(profile="glm"))
    now = datetime(2026, 5, 15, 11, 30, tzinfo=ZoneInfo("Asia/Shanghai"))

    retry = detect_runtime_recovery(
        "API Error: Request rejected (429)",
        now=now,
        policy=policy,
    )

    assert retry is not None
    assert retry.retry_at == datetime(2026, 5, 15, 18, 0, tzinfo=ZoneInfo("Asia/Shanghai"))


def test_invalid_timezone_config_falls_back_to_utc() -> None:
    policy = RuntimeRecoveryPolicy(reset_timezone="Not/AZone")

    retry = detect_runtime_recovery(
        "API Error: Request rejected (429). Limit expires at 2026-05-15 04:07:12.",
        now=datetime(2026, 5, 14, 20, 0, tzinfo=timezone.utc),
        policy=policy,
    )

    assert retry is not None
    assert retry.retry_at == datetime(2026, 5, 15, 4, 8, 12, tzinfo=timezone.utc)


def test_sidecar_injects_retry_for_malformed_http_200_error(tmp_path, monkeypatch) -> None:
    spec = load_spec("specs/examples/linear_plan.example.yaml")
    store = StateStore(str(tmp_path / "runtime"))
    state = store.load_or_init(spec)
    policy = RuntimeRecoveryPolicy(enabled=True, transient_delay_seconds=0)
    loop = SupervisorLoop(store, runtime_recovery_policy=policy)
    terminal = _Terminal([
        "",
        (
            "⎿  API Error: API returned an empty or malformed response (HTTP 200) "
            "— check for a proxy or gateway intercepting the request\n"
            "✻ Cooked for 23m 57s\n"
        ),
    ])

    monkeypatch.setattr("supervisor.loop.time.sleep", lambda seconds: None)

    final = loop.run_sidecar(
        spec,
        state,
        terminal,
        poll_interval=0,
        read_lines=50,
        stop_event=_StopAfterRetry(terminal),
    )

    retry_injections = [text for text in terminal.injected if text.startswith("retry")]
    assert retry_injections
    assert "empty or malformed response" in retry_injections[-1]
    assert final.top_state.value in {"ATTACHED", "RUNNING"}


def test_sidecar_reuses_original_fallback_deadline_for_repeated_error(tmp_path, monkeypatch) -> None:
    spec = load_spec("specs/examples/linear_plan.example.yaml")
    store = StateStore(str(tmp_path / "runtime"))
    state = store.load_or_init(spec)
    policy = RuntimeRecoveryPolicy(enabled=True, rate_limit_fallback_delay_seconds=120)
    loop = SupervisorLoop(store, runtime_recovery_policy=policy)
    error = "API Error: Request rejected (429)"
    terminal = _Terminal(["", error, error, error])
    times = iter([
        datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc),
        datetime(2026, 5, 15, 12, 1, tzinfo=timezone.utc),
        datetime(2026, 5, 15, 12, 2, tzinfo=timezone.utc),
        datetime(2026, 5, 15, 12, 3, tzinfo=timezone.utc),
    ])

    class _FakeDateTime:
        @staticmethod
        def now(tz=None):
            value = next(times)
            return value.astimezone(tz) if tz else value.replace(tzinfo=None)

    monkeypatch.setattr("supervisor.loop.datetime", _FakeDateTime)
    monkeypatch.setattr("supervisor.loop.time.sleep", lambda seconds: None)

    final = loop.run_sidecar(
        spec,
        state,
        terminal,
        poll_interval=0,
        read_lines=50,
        stop_event=_StopAfterRetry(terminal),
    )

    retry_injections = [text for text in terminal.injected if text.startswith("retry")]
    assert retry_injections
    assert final.top_state.value in {"ATTACHED", "RUNNING"}


def test_sidecar_clears_scheduled_recovery_after_real_checkpoint(tmp_path, monkeypatch) -> None:
    spec = load_spec("specs/examples/linear_plan.example.yaml")
    store = StateStore(str(tmp_path / "runtime"))
    state = store.load_or_init(spec)
    policy = RuntimeRecoveryPolicy(enabled=True, rate_limit_fallback_delay_seconds=120)
    loop = SupervisorLoop(store, runtime_recovery_policy=policy)
    terminal = _Terminal(["", "API Error: Request rejected (429)", _checkpoint(), ""])
    times = iter([
        datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc),
        datetime(2026, 5, 15, 12, 3, tzinfo=timezone.utc),
    ])

    class _FakeDateTime:
        @staticmethod
        def now(tz=None):
            value = next(times)
            return value.astimezone(tz) if tz else value.replace(tzinfo=None)

    monkeypatch.setattr("supervisor.loop.datetime", _FakeDateTime)
    monkeypatch.setattr("supervisor.loop.time.sleep", lambda seconds: None)

    loop.run_sidecar(
        spec,
        state,
        terminal,
        poll_interval=0,
        read_lines=50,
        stop_event=_StopAfterReads(terminal, 4),
    )

    assert not any(text.startswith("retry") for text in terminal.injected)


def test_runtime_recovery_config_fields_override_provider_aliases() -> None:
    cfg = RuntimeConfig(
        runtime_recovery_enabled=True,
        provider_retry_enabled=False,
        runtime_recovery_reset_timezone="UTC",
        provider_retry_reset_timezone="Asia/Shanghai",
    )

    policy = cfg.runtime_recovery_policy()

    assert policy.enabled is True
    assert policy.reset_timezone == "UTC"


def test_provider_aliases_are_used_when_runtime_recovery_fields_are_unset() -> None:
    cfg = RuntimeConfig(
        runtime_recovery_enabled=None,
        runtime_recovery_reset_timezone=None,
        runtime_recovery_transient_delay_seconds=None,
        runtime_recovery_max_attempts_per_run=None,
        provider_retry_enabled=False,
        provider_retry_reset_timezone="UTC",
        provider_retry_transient_delay_seconds=42,
        provider_retry_max_attempts_per_run=7,
    )

    policy = cfg.runtime_recovery_policy()

    assert policy.enabled is False
    assert policy.reset_timezone == "UTC"
    assert policy.transient_delay_seconds == 42
    assert policy.max_attempts_per_run == 7
