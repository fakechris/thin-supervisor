"""Runtime configuration with file / env / defaults layering."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


# Global config directory (overridable for testing)
_GLOBAL_CONFIG_ENV = "THIN_SUPERVISOR_GLOBAL_CONFIG"

# Fields safe to inherit from global config into any project
_GLOBAL_INHERITABLE = frozenset({
    "worker_provider", "worker_model", "judge_model",
    "judge_temperature", "judge_max_tokens", "worker_trust_level",
    "notification_channels", "pause_handling_mode", "max_auto_interventions",
    "poll_interval_sec", "read_lines",
    "runtime_recovery_enabled", "runtime_recovery_profile",
    "runtime_recovery_reset_timezone", "runtime_recovery_reset_grace_seconds",
    "runtime_recovery_transient_delay_seconds",
    "runtime_recovery_rate_limit_fallback_delay_seconds",
    "runtime_recovery_quiet_windows", "runtime_recovery_max_attempts_per_run",
    "provider_retry_enabled", "provider_retry_reset_timezone",
    "provider_retry_reset_grace_seconds", "provider_retry_transient_delay_seconds",
    "provider_retry_rate_limit_fallback_delay_seconds",
    "provider_retry_skip_windows", "provider_retry_max_attempts_per_run",
    "explainer_model", "explainer_temperature", "explainer_max_tokens",
    "deep_explainer_model", "deep_explainer_temperature", "deep_explainer_max_tokens",
    "clarification_escalation_confidence",
})


def global_config_path() -> Path:
    """Return the global defaults config path."""
    env = os.environ.get(_GLOBAL_CONFIG_ENV, "").strip()
    if env:
        return Path(env)
    return Path.home() / ".config" / "thin-supervisor" / "defaults.yaml"


def _update_config_file(path: Path, key: str, value) -> Path:
    """Read-modify-write a single key in a YAML config file with file locking."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / f".{path.name}.lock"
    lock_fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        data = {}
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        data[key] = value
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".yaml")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, default_flow_style=False)
            os.replace(tmp_path, str(path))
        except Exception:
            os.unlink(tmp_path)
            raise
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    return path


def save_global_config(key: str, value) -> Path:
    """Write a single key to the global config, creating it if needed."""
    return _update_config_file(global_config_path(), key, value)


def save_project_config(key: str, value, project_dir: str | Path = ".") -> Path:
    """Write a single key to the project config."""
    path = Path(project_dir) / ".supervisor" / "config.yaml"
    return _update_config_file(path, key, value)


def coerce_config_value(key: str, value: str):
    """Coerce a string value to the correct type for a RuntimeConfig field."""
    known = {f.name: f for f in fields(RuntimeConfig)}
    if key not in known:
        return value
    ftype = known[key].type
    if value.lower() in ("null", "none", "~"):
        return None
    if _field_accepts(ftype, "float", float):
        return float(value)
    if _field_accepts(ftype, "int", int):
        return int(value)
    if _field_accepts(ftype, "bool", bool):
        return value.lower() in ("1", "true", "yes", "on")
    return value


def _field_accepts(ftype, type_name: str, pytype) -> bool:
    text = str(ftype)
    return ftype in (type_name, pytype) or type_name in text


@dataclass
class RuntimeConfig:
    # -- Execution Surface --
    surface_type: str = "tmux"    # "tmux" | "open_relay"
    surface_target: str = ""     # pane label/%id (tmux) or session id (open_relay)
    pane_target: str = ""        # legacy alias for surface_target (tmux compat)
    poll_interval_sec: float = 2.0
    read_lines: int = 100

    # -- Worker Profile --
    worker_provider: str = "unknown"    # anthropic | openai | minimax | ...
    worker_model: str = ""              # claude-opus-4-6 | gpt-5.4 | ...
    worker_trust_level: str = "standard"  # low | standard | high

    # -- LLM Judge --
    judge_model: str | None = None  # None = stub mode
    judge_temperature: float = 0.1
    judge_max_tokens: int = 512

    # -- LLM Explainer (operator-facing, separate from judge) --
    #
    # Two tiers: routine `explainer_model` is the cheap/fast default used for
    # explain_run / explain_exchange / request_clarification. Optional
    # `deep_explainer_model` is used only for drift/codebase-heavy analysis
    # (assess_drift) — set to the stronger model you're willing to spend on
    # when operators ask "is this run still on track?" If `deep_explainer_model`
    # is None, drift assessment falls back to the routine explainer.
    explainer_model: str | None = None  # None = stub mode (cheap/fast default)
    explainer_temperature: float = 0.3
    explainer_max_tokens: int = 1024
    deep_explainer_model: str | None = None  # None = reuse explainer_model
    deep_explainer_temperature: float = 0.2
    deep_explainer_max_tokens: int = 2048

    # Clarification routing: when the explainer's self-reported confidence
    # is below this threshold, the channel surfaces `escalation_recommended`
    # so the operator can explicitly request a worker follow-up. Escalation
    # is never automatic — the operator remains in the loop.
    clarification_escalation_confidence: float = 0.4

    # -- Runtime paths --
    runtime_dir: str = ".supervisor/runtime"
    state_file: str = ".supervisor/runtime/state.json"
    event_log_file: str = ".supervisor/runtime/event_log.jsonl"
    decision_log_file: str = ".supervisor/runtime/decision_log.jsonl"

    # -- Retry (overridable, spec values take precedence) --
    max_retries_per_node: int = 3
    max_retries_global: int = 12

    # -- Gate --
    branch_confidence_threshold: float = 0.75
    default_agent_timeout_sec: int = 300

    # -- Runtime recovery --
    runtime_recovery_enabled: bool | None = None
    runtime_recovery_profile: str | None = None
    runtime_recovery_reset_timezone: str | None = None
    runtime_recovery_reset_grace_seconds: int | None = None
    runtime_recovery_transient_delay_seconds: int | None = None
    runtime_recovery_rate_limit_fallback_delay_seconds: int | None = None
    runtime_recovery_quiet_windows: list[str] | None = None
    runtime_recovery_max_attempts_per_run: int | None = None

    # Back-compat aliases for configs written during the initial provider-retry rollout.
    provider_retry_enabled: bool = True
    provider_retry_reset_timezone: str = "Asia/Shanghai"
    provider_retry_reset_grace_seconds: int = 60
    provider_retry_transient_delay_seconds: int = 60
    provider_retry_rate_limit_fallback_delay_seconds: int = 300
    provider_retry_skip_windows: list[str] = field(default_factory=list)
    provider_retry_max_attempts_per_run: int = 3

    # -- Notifications --
    notification_channels: list[dict] = field(default_factory=lambda: [
        {"kind": "tmux_display"},
        {"kind": "jsonl"},
    ])
    pause_handling_mode: str = "notify_then_ai"  # notify_only | notify_then_ai
    max_auto_interventions: int = 2

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> "RuntimeConfig":
        """Load config from a YAML file, ignoring unknown keys."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @classmethod
    def from_env(cls, prefix: str = "SUPERVISOR_") -> "RuntimeConfig":
        """Build config from ``SUPERVISOR_*`` environment variables."""
        data: dict = {}
        known = {f.name: f for f in fields(cls)}
        for key, val in os.environ.items():
            if not key.startswith(prefix):
                continue
            field_name = key[len(prefix):].lower()
            if field_name not in known:
                continue
            ftype = known[field_name].type
            if _field_accepts(ftype, "float", float):
                data[field_name] = float(val)
            elif _field_accepts(ftype, "int", int):
                data[field_name] = int(val)
            elif _field_accepts(ftype, "bool", bool):
                data[field_name] = val.lower() in ("1", "true", "yes", "on")
            else:
                data[field_name] = val
        return cls(**data)

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "RuntimeConfig":
        """Load with priority: defaults → global (inheritable) → project → env.

        Global config applies only inheritable fields. Project config applies
        all fields. Environment variables override everything.
        """
        base = cls()
        known = {f.name: f for f in fields(cls)}

        # 1. Global config — inheritable fields only
        gpath = global_config_path()
        if gpath.exists():
            try:
                gdata = yaml.safe_load(gpath.read_text(encoding="utf-8")) or {}
                for k, v in gdata.items():
                    if k in known and k in _GLOBAL_INHERITABLE:
                        setattr(base, k, v)
            except Exception:
                pass  # corrupt global config — skip silently

        # 2. Project config — all fields
        if config_path and Path(config_path).exists():
            pdata = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
            for k, v in pdata.items():
                if k in known:
                    setattr(base, k, v)

        # 3. Env vars override everything
        prefix = "SUPERVISOR_"
        for key, val in os.environ.items():
            if not key.startswith(prefix):
                continue
            field_name = key[len(prefix):].lower()
            if field_name not in known:
                continue
            ftype = known[field_name].type
            if _field_accepts(ftype, "float", float):
                setattr(base, field_name, float(val))
            elif _field_accepts(ftype, "int", int):
                setattr(base, field_name, int(val))
            elif _field_accepts(ftype, "bool", bool):
                setattr(base, field_name, val.lower() in ("1", "true", "yes", "on"))
            else:
                setattr(base, field_name, val)
        return base

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def effective_target(self) -> str:
        """Resolve the effective surface target (surface_target > pane_target)."""
        return self.surface_target or self.pane_target

    def runtime_recovery_policy(self):
        from supervisor.runtime_recovery import RuntimeRecoveryPolicy, policy_with_profile

        defaults = RuntimeConfig()
        windows = (
            self.runtime_recovery_quiet_windows
            if self.runtime_recovery_quiet_windows is not None
            else self.provider_retry_skip_windows
        )
        if isinstance(windows, str):
            windows = [item.strip() for item in windows.split(",") if item.strip()]
        return policy_with_profile(RuntimeRecoveryPolicy(
            enabled=(
                self.runtime_recovery_enabled
                if self.runtime_recovery_enabled is not None
                else self.provider_retry_enabled
            ),
            profile=self.runtime_recovery_profile or "",
            reset_timezone=(
                self.runtime_recovery_reset_timezone
                if self.runtime_recovery_reset_timezone is not None
                else self.provider_retry_reset_timezone
            ),
            reset_grace_seconds=(
                self.provider_retry_reset_grace_seconds
                if self.runtime_recovery_reset_grace_seconds is None
                else self.runtime_recovery_reset_grace_seconds
            ),
            transient_delay_seconds=(
                self.provider_retry_transient_delay_seconds
                if self.runtime_recovery_transient_delay_seconds is None
                else self.runtime_recovery_transient_delay_seconds
            ),
            rate_limit_fallback_delay_seconds=(
                self.provider_retry_rate_limit_fallback_delay_seconds
                if self.runtime_recovery_rate_limit_fallback_delay_seconds is None
                else self.runtime_recovery_rate_limit_fallback_delay_seconds
            ),
            quiet_windows=tuple(windows or ()),
            max_attempts_per_run=(
                self.provider_retry_max_attempts_per_run
                if self.runtime_recovery_max_attempts_per_run is None
                else self.runtime_recovery_max_attempts_per_run
            ),
        ))

    def default_config_yaml(self) -> str:
        """Render a commented YAML template suitable for ``init``."""
        return (
            "# thin-supervisor config\n"
            "\n"
            "# Execution surface: tmux | open_relay\n"
            f"surface_type: \"{self.surface_type}\"\n"
            "# Surface target: pane label/%id (tmux) or session id (open_relay)\n"
            f"surface_target: \"\"\n"
            f"poll_interval_sec: {self.poll_interval_sec}\n"
            f"read_lines: {self.read_lines}\n"
            "\n"
            "# Worker profile (affects supervision intensity)\n"
            "# provider: anthropic | openai | minimax | ...\n"
            f"worker_provider: \"{self.worker_provider}\"\n"
            "# model: claude-opus-4-6 | gpt-5.4 | ...\n"
            f"worker_model: \"{self.worker_model}\"\n"
            "# trust: low | standard | high (high = minimal supervision)\n"
            f"worker_trust_level: \"{self.worker_trust_level}\"\n"
            "\n"
            "# Runtime recovery: orthogonal to workflow steps; retries provider/connectivity failures.\n"
            f"runtime_recovery_enabled: {str(self.runtime_recovery_enabled if self.runtime_recovery_enabled is not None else True).lower()}\n"
            "# Optional preset profile. Example: \"glm\" reserves UTC+8 12:00-18:00 and uses 5h fallback.\n"
            f"runtime_recovery_profile: \"{self.runtime_recovery_profile or ''}\"\n"
            "# Naive reset timestamps in Chinese Claude/GLM output are usually UTC+8.\n"
            f"runtime_recovery_reset_timezone: \"{self.runtime_recovery_reset_timezone or self.provider_retry_reset_timezone}\"\n"
            f"runtime_recovery_reset_grace_seconds: {self.runtime_recovery_reset_grace_seconds if self.runtime_recovery_reset_grace_seconds is not None else self.provider_retry_reset_grace_seconds}\n"
            f"runtime_recovery_transient_delay_seconds: {self.runtime_recovery_transient_delay_seconds if self.runtime_recovery_transient_delay_seconds is not None else self.provider_retry_transient_delay_seconds}\n"
            f"runtime_recovery_rate_limit_fallback_delay_seconds: {self.runtime_recovery_rate_limit_fallback_delay_seconds if self.runtime_recovery_rate_limit_fallback_delay_seconds is not None else self.provider_retry_rate_limit_fallback_delay_seconds}\n"
            "# Optional quiet windows, e.g. [\"UTC+8 12:00-18:00\"]\n"
            "runtime_recovery_quiet_windows: []\n"
            f"runtime_recovery_max_attempts_per_run: {self.runtime_recovery_max_attempts_per_run if self.runtime_recovery_max_attempts_per_run is not None else self.provider_retry_max_attempts_per_run}\n"
            "\n"
            "# LLM judge (set to null for stub/offline mode)\n"
            "# Examples: anthropic/claude-haiku-4-5-20251001, openai/gpt-4o-mini\n"
            f"judge_model: null\n"
            f"judge_temperature: {self.judge_temperature}\n"
            f"judge_max_tokens: {self.judge_max_tokens}\n"
            "\n"
            "# LLM explainer (operator-facing, separate from judge)\n"
            "# Tolerates approximation; defaults to cheaper/faster model\n"
            f"explainer_model: null\n"
            f"explainer_temperature: {self.explainer_temperature}\n"
            f"explainer_max_tokens: {self.explainer_max_tokens}\n"
            "\n"
            "# Optional heavier explainer for drift/codebase analysis (assess_drift).\n"
            "# Leave null to reuse explainer_model.\n"
            f"deep_explainer_model: null\n"
            f"deep_explainer_temperature: {self.deep_explainer_temperature}\n"
            f"deep_explainer_max_tokens: {self.deep_explainer_max_tokens}\n"
            "\n"
            "# Confidence below which clarification surfaces an escalation hint\n"
            "# (does not auto-escalate — the operator decides).\n"
            f"clarification_escalation_confidence: {self.clarification_escalation_confidence}\n"
            "\n"
            "# Runtime\n"
            f"runtime_dir: \"{self.runtime_dir}\"\n"
            "\n"
            "# Notification channels used when a run pauses for human.\n"
            "# Built-ins today: tmux_display, jsonl\n"
            "notification_channels:\n"
            "  - kind: \"tmux_display\"\n"
            "  - kind: \"jsonl\"\n"
            "\n"
            "# Pause handling strategy.\n"
            "# notify_only: pause and wait\n"
            "# notify_then_ai: notify, then let the agent try an automatic recovery first\n"
            f"pause_handling_mode: \"{self.pause_handling_mode}\"\n"
            f"max_auto_interventions: {self.max_auto_interventions}\n"
        )
