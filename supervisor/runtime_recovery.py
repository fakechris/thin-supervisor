from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class RuntimeRecoveryPolicy:
    enabled: bool | None = True
    profile: str = ""
    reset_timezone: str | None = "Asia/Shanghai"
    reset_grace_seconds: int | None = 60
    transient_delay_seconds: int | None = 60
    rate_limit_fallback_delay_seconds: int | None = 300
    quiet_windows: tuple[str, ...] = ()
    max_attempts_per_run: int | None = 3


@dataclass(frozen=True)
class RuntimeRecoveryObservation:
    kind: str
    reason: str
    retry_at: datetime
    signature: str


PROFILE_DEFAULTS: dict[str, RuntimeRecoveryPolicy] = {
    "glm": RuntimeRecoveryPolicy(
        profile="glm",
        reset_timezone="Asia/Shanghai",
        reset_grace_seconds=60,
        transient_delay_seconds=300,
        rate_limit_fallback_delay_seconds=5 * 60 * 60,
        quiet_windows=("UTC+8 12:00-18:00",),
        max_attempts_per_run=3,
    ),
}


_RESET_PATTERNS = (
    re.compile(r"限额将在\s*(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})\s*重置"),
    re.compile(r"reset(?:s|ting)?(?:\s+at|\s+on|:)?\s*(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})", re.I),
    re.compile(r"expires?(?:\s+at|\s+on|:)?\s*(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})", re.I),
    re.compile(r"try again(?:\s+at|\s+after|:)?\s*(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})", re.I),
    re.compile(r"until\s*(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})", re.I),
)

_TRANSIENT_PATTERNS = (
    re.compile(r"empty or malformed response\s*\(HTTP 200\)", re.I),
    re.compile(r"proxy or gateway intercepting the request", re.I),
    re.compile(r"\bnetwork (?:is )?(?:down|unavailable|disconnected|connection lost)\b", re.I),
    re.compile(r"\b(?:connection|network) (?:reset|refused|timed out|timeout)\b", re.I),
)


def policy_with_profile(policy: RuntimeRecoveryPolicy) -> RuntimeRecoveryPolicy:
    profile = (policy.profile or "").strip().lower()
    base = PROFILE_DEFAULTS.get(profile)
    if base is None:
        return policy
    defaults = RuntimeRecoveryPolicy()
    return replace(
        base,
        enabled=policy.enabled if policy.enabled is not None else base.enabled,
        profile=profile,
        reset_timezone=(
            policy.reset_timezone
            if policy.reset_timezone is not None and policy.reset_timezone != defaults.reset_timezone
            else base.reset_timezone
        ),
        reset_grace_seconds=(
            policy.reset_grace_seconds
            if policy.reset_grace_seconds is not None and policy.reset_grace_seconds != defaults.reset_grace_seconds
            else base.reset_grace_seconds
        ),
        transient_delay_seconds=(
            policy.transient_delay_seconds
            if policy.transient_delay_seconds is not None and policy.transient_delay_seconds != defaults.transient_delay_seconds
            else base.transient_delay_seconds
        ),
        rate_limit_fallback_delay_seconds=(
            policy.rate_limit_fallback_delay_seconds
            if (
                policy.rate_limit_fallback_delay_seconds is not None
                and policy.rate_limit_fallback_delay_seconds != defaults.rate_limit_fallback_delay_seconds
            )
            else base.rate_limit_fallback_delay_seconds
        ),
        quiet_windows=policy.quiet_windows or base.quiet_windows,
        max_attempts_per_run=(
            policy.max_attempts_per_run
            if policy.max_attempts_per_run is not None and policy.max_attempts_per_run != defaults.max_attempts_per_run
            else base.max_attempts_per_run
        ),
    )


def detect_runtime_recovery(
    text: str,
    *,
    now: datetime | None = None,
    policy: RuntimeRecoveryPolicy | None = None,
) -> RuntimeRecoveryObservation | None:
    policy = policy_with_profile(policy or RuntimeRecoveryPolicy())
    if not policy.enabled or not text:
        return None

    now = _aware(now or datetime.now(timezone.utc))
    lowered = text.lower()
    reset_retry_at = _extract_reset_time(text, policy=policy)

    if reset_retry_at is not None:
        retry_at = next_allowed_recovery_at(reset_retry_at, policy)
        return RuntimeRecoveryObservation(
            kind="rate_limit",
            reason=_compact_reason(text),
            retry_at=retry_at,
            signature=_signature("rate_limit", text),
        )

    if "429" in lowered or "rate limit" in lowered or "使用上限" in text:
        retry_at = now + timedelta(seconds=policy.rate_limit_fallback_delay_seconds or 0)
        retry_at = next_allowed_recovery_at(retry_at, policy)
        return RuntimeRecoveryObservation(
            kind="rate_limit",
            reason=_compact_reason(text),
            retry_at=retry_at,
            signature=_signature("rate_limit", text),
        )

    if any(pattern.search(text) for pattern in _TRANSIENT_PATTERNS):
        retry_at = now + timedelta(seconds=policy.transient_delay_seconds or 0)
        retry_at = next_allowed_recovery_at(retry_at, policy)
        return RuntimeRecoveryObservation(
            kind="transient_connectivity",
            reason=_compact_reason(text),
            retry_at=retry_at,
            signature=_signature("transient_connectivity", text),
        )

    return None


def next_allowed_recovery_at(candidate: datetime, policy: RuntimeRecoveryPolicy) -> datetime:
    policy = policy_with_profile(policy)
    result = _aware(candidate)
    for _ in range(max(1, len(policy.quiet_windows) + 1)):
        moved = False
        for raw in policy.quiet_windows:
            window = _parse_quiet_window(raw, default_tz=policy.reset_timezone)
            if window is None:
                continue
            tz, start, end = window
            local = result.astimezone(tz)
            if not _time_in_window(local.timetz().replace(tzinfo=None), start, end):
                continue
            local_end = _window_end(local, start, end)
            result = local_end.astimezone(result.tzinfo or timezone.utc)
            moved = True
        if not moved:
            return result
    return result


def seconds_until_recovery(retry_at: datetime, *, now: datetime | None = None) -> float:
    now = _aware(now or datetime.now(timezone.utc))
    return max(0.0, (_aware(retry_at).astimezone(timezone.utc) - now.astimezone(timezone.utc)).total_seconds())


def _extract_reset_time(text: str, *, policy: RuntimeRecoveryPolicy) -> datetime | None:
    for pattern in _RESET_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        parsed = datetime.fromisoformat(match.group("ts").replace(" ", "T"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_timezone(policy.reset_timezone))
        return parsed + timedelta(seconds=policy.reset_grace_seconds)
    return None


def _parse_quiet_window(raw: str, *, default_tz: str) -> tuple[timezone | ZoneInfo, time, time] | None:
    parts = raw.strip().split()
    if not parts:
        return None
    if len(parts) == 1:
        tz_name = default_tz
        span = parts[0]
    else:
        tz_name = parts[0]
        span = parts[1]
    match = re.fullmatch(r"(?P<start>\d{1,2}:\d{2})-(?P<end>\d{1,2}:\d{2})", span)
    if not match:
        return None
    return (
        _timezone(tz_name),
        time.fromisoformat(match.group("start")),
        time.fromisoformat(match.group("end")),
    )


def _timezone(name: str) -> timezone | ZoneInfo:
    normalized = (name or "UTC").strip()
    match = re.fullmatch(r"UTC([+-])(\d{1,2})(?::?(\d{2}))?", normalized, re.I)
    if match:
        sign = 1 if match.group(1) == "+" else -1
        hours = int(match.group(2))
        minutes = int(match.group(3) or "0")
        return timezone(sign * timedelta(hours=hours, minutes=minutes))
    if normalized.upper() == "UTC":
        return timezone.utc
    try:
        return ZoneInfo(normalized)
    except Exception:
        return timezone.utc


def _time_in_window(value: time, start: time, end: time) -> bool:
    if start < end:
        return start <= value < end
    return value >= start or value < end


def _window_end(local_dt: datetime, start: time, end: time) -> datetime:
    end_dt = local_dt.replace(hour=end.hour, minute=end.minute, second=0, microsecond=0)
    if start >= end and local_dt.timetz().replace(tzinfo=None) >= start:
        end_dt += timedelta(days=1)
    if end_dt <= local_dt:
        end_dt += timedelta(days=1)
    return end_dt


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _compact_reason(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    joined = " ".join(lines)
    return joined[:300]


def _signature(kind: str, text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    normalized = re.sub(r"\d{4}-\d{2}-\d{2}[ t]\d{2}:\d{2}:\d{2}", "<ts>", normalized)
    return f"{kind}:{normalized[:180]}"
