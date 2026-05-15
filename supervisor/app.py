"""CLI entry point for thin-supervisor."""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import sys
import time as _time
from datetime import datetime, timezone
from pathlib import Path

from supervisor.storage.state_store import StateStore
from supervisor.loop import SupervisorLoop
from supervisor.config import RuntimeConfig
from supervisor.adapters.transcript_adapter import TranscriptAdapter
from supervisor.global_registry import find_pane_owner, list_daemons, list_pane_owners
from supervisor.interventions import AutoInterventionManager
from supervisor.notifications import NotificationManager
from supervisor.pause_summary import summarize_state
from supervisor.spec_approval import approve_spec, load_runnable_spec
from supervisor.learning import (
    append_friction_event,
    list_friction_events,
    load_user_preferences,
    save_user_preferences,
    summarize_friction_events,
)


SUPERVISOR_DIR = ".supervisor"
CONFIG_FILE = ".supervisor/config.yaml"
RUNTIME_DIR = ".supervisor/runtime"
SPECS_DIR = ".supervisor/specs"
OPS_LOG_FILE = ".supervisor/runtime/ops_log.jsonl"


# ------------------------------------------------------------------
# foreground PID persistence
# ------------------------------------------------------------------


def _persist_foreground_pid(run_dir: str, pid: int) -> None:
    """Write foreground controller PID into state.json for liveness detection."""
    state_path = Path(run_dir) / "state.json"
    if state_path.exists():
        data = json.loads(state_path.read_text())
        data["_foreground_pid"] = pid
        state_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


# ------------------------------------------------------------------
# init / deinit
# ------------------------------------------------------------------


def cmd_init(args):
    """Create .supervisor/ directory with default config."""
    base = Path(SUPERVISOR_DIR)
    repair = getattr(args, "repair", False)
    if base.exists() and not args.force and not repair:
        print(f"{SUPERVISOR_DIR}/ already exists. Use --force to overwrite.")
        return 1

    base_preexisted = base.exists()
    base.mkdir(parents=True, exist_ok=True)
    Path(RUNTIME_DIR).mkdir(parents=True, exist_ok=True)
    Path(SPECS_DIR).mkdir(parents=True, exist_ok=True)
    Path(f"{SUPERVISOR_DIR}/clarify").mkdir(parents=True, exist_ok=True)
    Path(f"{SUPERVISOR_DIR}/plans").mkdir(parents=True, exist_ok=True)

    config = RuntimeConfig()
    config_path = Path(CONFIG_FILE)
    created_config = False
    if args.force or not config_path.exists():
        config_path.write_text(config.default_config_yaml(), encoding="utf-8")
        created_config = True

    gitignore = Path(".gitignore")
    if gitignore.exists():
        content = gitignore.read_text()
        if RUNTIME_DIR not in content:
            with gitignore.open("a") as f:
                f.write(f"\n{RUNTIME_DIR}/\n")

    if repair:
        _append_ops_event(
            "init_repair",
            {
                "supervisor_dir_preexisted": base_preexisted,
                "created_config": created_config,
                "config_path": CONFIG_FILE,
            },
        )
        print(f"Repaired {SUPERVISOR_DIR}/")
    else:
        print(f"Initialized {SUPERVISOR_DIR}/")
    return 0


def _append_ops_event(event_type: str, payload: dict) -> None:
    ops_log_path = Path(OPS_LOG_FILE)
    ops_log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }
    with ops_log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def cmd_deinit(args):
    """Remove .supervisor/ directory."""
    base = Path(SUPERVISOR_DIR)
    if not base.exists():
        print(f"{SUPERVISOR_DIR}/ does not exist.")
        return 1

    if not args.force:
        # Check for active daemon
        from supervisor.daemon.client import DaemonClient
        client = DaemonClient()
        if client.is_running():
            print("Daemon is running. Use 'thin-supervisor daemon stop' first, or --force.")
            return 1

    shutil.rmtree(base)
    print(f"Removed {SUPERVISOR_DIR}/")
    return 0


# ------------------------------------------------------------------
# daemon start / stop
# ------------------------------------------------------------------


def _fork_daemon(config: RuntimeConfig) -> int:
    """Fork a daemon process. Returns child PID to parent, 0 to child (never returns)."""
    from supervisor.daemon.server import DaemonServer

    pid = os.fork()
    if pid > 0:
        return pid  # parent

    # Child — detach
    os.setsid()
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)

    log_path = Path(RUNTIME_DIR) / "daemon.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(log_path), level=logging.INFO, force=True,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    server = DaemonServer(config)
    server.start()
    sys.exit(0)


def cmd_daemon_start(args):
    """Start the supervisor daemon (single process, multi-run)."""
    from supervisor.daemon.client import DaemonClient

    client = DaemonClient()
    if client.is_running():
        pid = client.daemon_pid()
        print(f"Daemon already running (PID {pid}).")
        return 1

    config = RuntimeConfig.load(args.config or CONFIG_FILE)
    pid = _fork_daemon(config)
    print(f"Daemon started (PID {pid})")
    return 0


def cmd_daemon_stop(args):
    """Stop the supervisor daemon."""
    from supervisor.daemon.client import DaemonClient, PID_PATH

    client = DaemonClient()
    pid = client.daemon_pid()
    if not pid:
        print("No daemon PID file found.")
        return 1

    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Sent SIGTERM to daemon (PID {pid})")
        exited = False
        for _ in range(50):
            try:
                os.kill(pid, 0)
                _time.sleep(0.1)
            except ProcessLookupError:
                exited = True
                break
        if exited:
            Path(PID_PATH).unlink(missing_ok=True)
            print("Daemon stopped.")
        else:
            print(f"Warning: PID {pid} did not exit within 5s.")
            return 1
    except ProcessLookupError:
        print(f"Process {pid} not found (already stopped?).")
        Path(PID_PATH).unlink(missing_ok=True)
    except PermissionError:
        print(f"Error: no permission to signal PID {pid}.")
        return 1
    return 0


# ------------------------------------------------------------------
# run register / foreground / stop
# ------------------------------------------------------------------


def _ensure_daemon(config_path: str | None = None) -> "DaemonClient":
    """Ensure daemon is running, auto-start if needed. Returns client."""
    from supervisor.daemon.client import DaemonClient

    client = DaemonClient()
    if client.is_running():
        return client

    print("Daemon not running. Starting...")
    config = RuntimeConfig.load(config_path or CONFIG_FILE)
    _fork_daemon(config)

    for _ in range(30):
        _time.sleep(0.2)
        if client.is_running():
            return client
    print("Error: daemon did not start within 6s.")
    sys.exit(1)


def _resolve_target_and_surface(args, config):
    """Resolve target and surface_type from args + config, with validation."""
    target = getattr(args, "target", None) or getattr(args, "pane", None) or ""
    surface = getattr(args, "surface", None) or getattr(config, "surface_type", "tmux")

    if not target:
        print("Error: --pane or --target is required.")
        return None, None

    # Validate target format for surface type
    if surface == "jsonl" and not (target.endswith(".jsonl") or Path(target).exists()):
        print(f"Warning: jsonl surface expects a .jsonl file path, got '{target}'")
    elif surface == "tmux" and target.endswith(".jsonl"):
        print(f"Warning: tmux surface got a .jsonl path — did you mean --surface jsonl?")

    return target, surface


def cmd_run_register(args):
    """Register a new run with the daemon."""
    config = RuntimeConfig.load(getattr(args, "config", None) or CONFIG_FILE)
    target, surface = _resolve_target_and_surface(args, config)
    if not target:
        return 1
    pane_target = target

    spec_path = os.path.abspath(args.spec)
    try:
        load_runnable_spec(spec_path)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    client = _ensure_daemon(args.config)
    workspace_root = os.getcwd()

    result = client.register(spec_path, pane_target, workspace_root=workspace_root, surface_type=surface)
    if result.get("ok"):
        print(f"Run registered: {result['run_id']}")
        print(f"  spec: {spec_path}")
        print(f"  target: {pane_target} ({surface})")
    else:
        print(f"Error: {result.get('error', 'unknown')}")
        return 1
    return 0


def cmd_run_foreground(args):
    """Run a single sidecar in foreground (debug only, not for normal use)."""
    from supervisor.global_registry import acquire_pane_lock, release_pane_lock

    spec_path = os.path.abspath(args.spec)
    try:
        spec = load_runnable_spec(spec_path)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    config = RuntimeConfig.load(args.config or CONFIG_FILE)

    target, surface_type = _resolve_target_and_surface(args, config)
    if not target:
        return 1
    pane_target = target

    # Per-run isolated directory
    import uuid
    run_id = f"run_{uuid.uuid4().hex[:12]}"

    # Acquire pane lock to prevent conflicts with daemon-owned runs
    pane_owner = {
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "socket": "",
        "run_id": run_id,
        "spec_path": spec_path,
        "controller_mode": "foreground",
    }
    acquired, existing = acquire_pane_lock(pane_target, pane_owner)
    if not acquired:
        mode = existing.get("controller_mode", "unknown") if existing else "unknown"
        owner_run = existing.get("run_id", "?") if existing else "?"
        print(f"Error: pane {pane_target} is locked by {mode} run {owner_run}.")
        if mode == "daemon":
            print("Use 'thin-supervisor run register' for normal execution.")
        else:
            print("Stop the existing run first.")
        return 1

    # Command channels are per-credential-set singletons with
    # cross-process advisory locks.
    from supervisor.operator.channel_host import OperatorChannelHost
    channel_host = OperatorChannelHost.from_config(config)
    channel_host.start()

    try:
        run_dir = str(Path(RUNTIME_DIR) / "runs" / run_id)
        store = StateStore(run_dir)
        state = store.load_or_init(
            spec,
            spec_path=os.path.abspath(args.spec),
            pane_target=pane_target,
            surface_type=surface_type,
            workspace_root=os.getcwd(),
            controller_mode="foreground",
        )
        # Persist PID so status/dashboard can detect alive foreground runs
        store.save(state)
        _persist_foreground_pid(run_dir, os.getpid())

        from supervisor.adapters.surface_factory import create_surface
        terminal = create_surface(surface_type, pane_target)

        diag = terminal.doctor()
        if not diag["ok"]:
            print(f"Surface issues: {diag['issues']}")
            return 1

        from supervisor.domain.models import WorkerProfile
        worker = WorkerProfile(
            provider=config.worker_provider,
            model_name=config.worker_model,
            trust_level=config.worker_trust_level,
        )
        loop = SupervisorLoop(
            store,
            judge_model=config.judge_model,
            judge_temperature=config.judge_temperature,
            judge_max_tokens=config.judge_max_tokens,
            worker_profile=worker,
            notification_manager=NotificationManager.from_config(
                config,
                runtime_root=store.runtime_root,
                command_channels=channel_host.channels,
            ),
            auto_intervention_manager=AutoInterventionManager(
                mode=config.pause_handling_mode,
                max_auto_interventions=config.max_auto_interventions,
            ),
            runtime_recovery_policy=config.runtime_recovery_policy(),
        )

        print(f"[DEBUG MODE] Foreground controller — for debugging only")
        print(f"  run={run_id} pane={pane_target} spec={spec.id}")

        try:
            final_state = loop.run_sidecar(
                spec, state, terminal,
                poll_interval=config.poll_interval_sec,
                read_lines=config.read_lines,
                idle_timeout_sec=config.default_agent_timeout_sec,
            )
            print(f"\nRun finished: {final_state.top_state.value}")
        except KeyboardInterrupt:
            store.save(state)
            print("\nInterrupted. State saved.")
    finally:
        channel_host.stop()
        release_pane_lock(pane_target, run_id)

    return 0


def cmd_run_resume(args):
    """Resume a paused or crashed run."""
    config = RuntimeConfig.load(getattr(args, "config", None) or CONFIG_FILE)
    target, surface = _resolve_target_and_surface(args, config)
    if not target:
        return 1

    spec_path = os.path.abspath(args.spec)
    try:
        load_runnable_spec(spec_path)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    client = _ensure_daemon(getattr(args, "config", None))

    result = client.resume(spec_path, target, surface_type=surface)
    if result.get("ok"):
        print(f"Run resumed: {result['run_id']} (from {result.get('resumed_from', '?')})")
    else:
        print(f"Error: {result.get('error', 'unknown')}")
        return 1
    return 0


def cmd_run_review(args):
    """Record reviewer acknowledgement for a paused run."""
    from supervisor.daemon.client import DaemonClient

    client = DaemonClient()
    if not client.is_running():
        print("Daemon not running.")
        return 1

    result = client.ack_review(args.run_id, reviewer=args.by)
    if result.get("ok"):
        print(f"Review recorded for {result['run_id']} by {args.by}")
        print(f"  top_state: {result.get('top_state', 'UNKNOWN')}")
    else:
        print(f"Error: {result.get('error', 'unknown')}")
        return 1
    return 0


def cmd_run_export(args):
    """Export a run's durable history as stable JSON."""
    from supervisor.history import export_run

    runtime_dir = RuntimeConfig.load(getattr(args, "config", None) or CONFIG_FILE).runtime_dir
    try:
        payload = export_run(args.run_id, runtime_dir=runtime_dir)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(str(target))
        return 0
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_run_summarize(args):
    """Summarize a historical run."""
    from supervisor.history import export_run, summarize_run

    runtime_dir = RuntimeConfig.load(getattr(args, "config", None) or CONFIG_FILE).runtime_dir
    try:
        payload = summarize_run(export_run(args.run_id, runtime_dir=runtime_dir))
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print(f"Run: {payload['run_id']}")
    print(f"State: {payload['top_state']}")
    print(f"Checkpoints: {payload['counts']['checkpoints']}")
    print(f"Verifications ok: {payload['counts']['verifications_ok']}")
    print(f"Routing events: {payload['counts']['routing_events']}")
    print(f"Friction events: {payload['friction_summary']['total_events']}")
    print(f"Friction kinds: {', '.join(payload['friction_kinds']) or '(none)'}")
    print(f"Oracle consultations: {', '.join(payload['oracle_consultation_ids']) or '(none)'}")
    return 0


def cmd_run_replay(args):
    """Replay historical gate decisions without execution."""
    from supervisor.history import export_run, replay_run

    runtime_dir = RuntimeConfig.load(getattr(args, "config", None) or CONFIG_FILE).runtime_dir
    try:
        payload = replay_run(export_run(args.run_id, runtime_dir=runtime_dir))
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print(f"Run: {payload['run_id']}")
    print(f"Matched decisions: {payload['matched_count']}/{payload['decision_count']}")
    print(f"Mismatches: {len(payload['mismatches'])}")
    return 0


def cmd_run_postmortem(args):
    """Write a markdown postmortem for a historical run."""
    from supervisor.history import export_run, render_postmortem

    runtime_dir = RuntimeConfig.load(getattr(args, "config", None) or CONFIG_FILE).runtime_dir
    try:
        markdown = render_postmortem(export_run(args.run_id, runtime_dir=runtime_dir))
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    target = Path(args.output) if args.output else Path(".supervisor") / "reports" / f"{args.run_id}.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(markdown, encoding="utf-8")
    print(str(target))
    return 0


def cmd_run_stop(args):
    """Stop a specific run."""
    from supervisor.daemon.client import DaemonClient

    client = DaemonClient()
    if not client.is_running():
        print("Daemon not running.")
        return 1

    result = client.stop_run(args.run_id)
    if result.get("ok"):
        print(f"Run {args.run_id} stopped.")
    else:
        print(f"Error: {result.get('error', 'unknown')}")
        return 1
    return 0


# ------------------------------------------------------------------
# status
# ------------------------------------------------------------------


def _find_local_run_summaries() -> list[dict]:
    """Read persisted local run state outside the daemon's active registry."""
    runtime_dir = Path(RUNTIME_DIR)
    states: list[dict] = []

    legacy_state = runtime_dir / "state.json"
    if legacy_state.exists():
        try:
            states.append(json.loads(legacy_state.read_text()))
        except json.JSONDecodeError:
            pass

    runs_dir = runtime_dir / "runs"
    if runs_dir.exists():
        for run_dir in sorted(runs_dir.iterdir()):
            state_path = run_dir / "state.json"
            if not state_path.exists():
                continue
            try:
                states.append(json.loads(state_path.read_text()))
            except json.JSONDecodeError:
                continue

    return states


def _summarize_local_state_for_hint(state: dict) -> dict:
    summary = summarize_state(state)
    if summary.get("top_state") in {"RUNNING", "GATING", "VERIFYING"}:
        is_orphaned = False
        controller = state.get("controller_mode", "daemon")
        if controller == "foreground":
            # Foreground is orphaned if the owning process is dead
            import signal
            pid = state.get("_foreground_pid", 0)
            if pid:
                try:
                    os.kill(pid, 0)
                except (OSError, ProcessLookupError):
                    is_orphaned = True
            else:
                # No PID recorded — assume orphaned (legacy state)
                is_orphaned = True
        else:
            # Daemon-owned — always orphaned when found in local scan
            # (if daemon were managing it, it wouldn't be in local-only state)
            is_orphaned = True

        if is_orphaned:
            reason = (
                "foreground process no longer running"
                if controller == "foreground"
                else "persisted run was left in progress without an active daemon worker"
            )
            paused_view = dict(state)
            paused_view["top_state"] = "PAUSED_FOR_HUMAN"
            paused_view["human_escalations"] = list(paused_view.get("human_escalations", [])) + [
                {"reason": reason}
            ]
            summary = summarize_state(paused_view)
            summary["orphaned_local_state"] = True
            summary["orphaned_from"] = state.get("top_state", "")
    return summary


def _print_local_state_hint() -> None:
    summaries = _find_local_run_summaries()
    if not summaries:
        return

    print("Local state found:")
    for state in summaries:
        display = _summarize_local_state_for_hint(state)
        mode = display.get("controller_mode", "daemon")
        mode_tag = "[foreground]" if mode == "foreground" else "[daemon]"
        if display.get("orphaned_local_state"):
            mode_tag = "[orphaned]"
        print(
            "  "
            f"{mode_tag} "
            f"{display.get('run_id', '?')} "
            f"{display.get('top_state', '?')} "
            f"node={display.get('current_node_id', '') or '?'} "
            f"pane={display.get('pane_target', '?')}"
        )
        if display.get("status_reason"):
            print(f"    status: {display['status_reason']}")
        if display.get("pause_reason"):
            print(f"    reason: {display['pause_reason']}")
        if display.get("next_action"):
            print(f"    next:   {display['next_action']}")
    print("  These are persisted local state files, not daemon-managed active runs.")


def cmd_list(args):
    """List all active runs with detailed state."""
    from supervisor.daemon.client import DaemonClient

    client = DaemonClient()
    if not client.is_running():
        print("Daemon not running.")
        return 1

    result = client.list_runs()
    if not result.get("ok"):
        print(f"Error: {result.get('error', 'unknown')}")
        return 1

    runs = result.get("runs", [])
    if not runs:
        print("No active runs.")
        _print_local_state_hint()
        return 0

    print(f"{'RUN_ID':<20} {'PANE':<18} {'STATE':<15} {'NODE':<20} {'DONE'}")
    for r in runs:
        done = ", ".join(r.get("done_nodes", [])) or "(none)"
        print(f"{r['run_id']:<20} {r['pane_target']:<18} {r['top_state']:<15} {r.get('current_node', ''):<20} {done}")
        if r.get("status_reason"):
            print(f"  status: {r['status_reason']}")
        if r.get("pause_reason"):
            print(f"  reason: {r['pause_reason']}")
        if r.get("next_action"):
            print(f"  next:   {r['next_action']}")
    return 0


def _list_global_daemons() -> list[dict]:
    return list_daemons()


def _compute_idle_display(daemon: dict) -> str:
    """Format idle duration from registry (kept fresh by daemon refresh loop)."""
    idle = daemon.get("idle_for_sec", 0)
    if idle >= 3600:
        return f"{idle // 3600}h"
    if idle >= 60:
        return f"{idle // 60}m"
    return f"{idle}s" if idle > 0 else "-"


def _find_global_pane_owner(pane_target: str) -> dict | None:
    return find_pane_owner(pane_target)


def cmd_ps(args):
    """List all globally registered daemon processes and foreground runs."""
    daemons = _list_global_daemons()
    foreground_runs = [
        p for p in list_pane_owners()
        if p.get("controller_mode") == "foreground"
    ]

    if not daemons and not foreground_runs:
        print("No registered daemons or foreground runs.")
        return 0

    if daemons:
        print("Daemons:")
        print(f"  {'PID':<8} {'STATE':<8} {'RUNS':<6} {'IDLE':<8} {'CWD'}")
        for daemon in daemons:
            state = daemon.get("state", "active")
            idle_str = "-"
            if state == "idle" and daemon.get("active_runs", 0) == 0:
                # Compute real-time idle from registry snapshot timestamp
                idle_str = _compute_idle_display(daemon)
            print(
                f"  {daemon.get('pid', '?'):<8} "
                f"{state:<8} "
                f"{daemon.get('active_runs', 0):<6} "
                f"{idle_str:<8} "
                f"{daemon.get('cwd', '')}"
            )

    if foreground_runs:
        print("Foreground debug runs:")
        print(f"  {'PID':<8} {'RUN_ID':<20} {'PANE':<10} {'SPEC'}")
        for run in foreground_runs:
            print(
                f"  {run.get('pid', '?'):<8} "
                f"{run.get('run_id', '?'):<20} "
                f"{run.get('pane_target', '?'):<10} "
                f"{run.get('spec_path', '?')}"
            )

    return 0


def cmd_pane_owner(args):
    """Show global owner metadata for a pane lock."""
    owner = _find_global_pane_owner(args.pane)
    if not owner:
        print(f"No owner found for pane {args.pane}.")
        return 1

    mode = owner.get("controller_mode", "unknown")
    print(f"Pane:       {owner.get('pane_target', args.pane)}")
    print(f"Run:        {owner.get('run_id', '?')}")
    print(f"Controller: {mode}")
    print(f"PID:        {owner.get('pid', '?')}")
    print(f"CWD:        {owner.get('cwd', '?')}")
    print(f"Socket:     {owner.get('socket', '?')}")
    print(f"Spec:       {owner.get('spec_path', '?')}")
    print(f"Attached:   {owner.get('acquired_at', '?')}")
    return 0


def cmd_observe(args):
    """Read-only observation of a specific run.

    Resolves the run against the global session index so observe works
    from any cwd, across worktrees, and falls back to on-disk state +
    session log when no live daemon owns the run.
    """
    from supervisor.operator.actions import ActionUnavailable, do_inspect
    from supervisor.operator.run_context import RunContext
    from supervisor.operator.session_index import find_session

    rec = find_session(args.run_id)
    if rec is None:
        print(f"Error: run not found: {args.run_id}")
        return 1

    ctx = RunContext.from_run_dict({
        "run_id": rec.run_id,
        "worktree": rec.worktree_root,
        "tag": rec.tag or "local",
        "top_state": rec.top_state,
        "pane_target": rec.pane_target,
        "socket": rec.daemon_socket,
    })

    try:
        data = do_inspect(ctx)
    except ActionUnavailable as e:
        print(f"Error: observe unavailable: {e}")
        return 1
    except (ConnectionRefusedError, FileNotFoundError, OSError) as e:
        # Narrow race: list_daemons() reaps dead PIDs, but a daemon can
        # die between registry read and socket connect.  Fail cleanly
        # with the run_id and error rather than dumping a traceback.
        print(f"Error: daemon unreachable for {rec.run_id}: {e}")
        return 1

    snap = data.get("snapshot", {}) or {}
    timeline = data.get("timeline", []) or []

    print(f"Run:     {rec.run_id}")
    print(f"Spec:    {snap.get('spec_id', '?')}")
    print(f"State:   {snap.get('top_state', '?')}")
    print(f"Node:    {snap.get('current_node', '?')}")
    print(f"Attempt: {snap.get('current_attempt', 0)}")
    done = snap.get("done_nodes") or []
    print(f"Done:    {', '.join(done) if done else '(none)'}")
    if rec.worktree_root:
        print(f"Worktree: {rec.worktree_root}")
    if snap.get("pause_reason"):
        print(f"Pause:   {snap['pause_reason']}")
    if snap.get("next_action"):
        print(f"Next:    {snap['next_action']}")

    ep = snap.get("event_plane") or {}
    if ep:
        parts = []
        if ep.get("waits_open"):
            parts.append(f"waits_open={ep['waits_open']}")
        if ep.get("mailbox_new"):
            parts.append(f"mailbox_new={ep['mailbox_new']}")
        if ep.get("mailbox_acknowledged"):
            parts.append(f"mailbox_ack={ep['mailbox_acknowledged']}")
        if parts:
            print(f"Event plane: {', '.join(parts)}")
        if ep.get("latest_wake_decision"):
            print(f"Last wake:   {ep['latest_wake_decision']}")

    if timeline:
        print(f"\nRecent events ({len(timeline)}):")
        for e in timeline:
            ts = e.get("occurred_at") or e.get("timestamp", "")
            print(f"  [{e.get('event_type', '?')}] {ts[:19]}")
    return 0


def cmd_clarify(args):
    """Ask the explainer a free-form question about a run, optionally escalate.

    The answer is served by ``ExplainerClient.request_clarification`` — same
    async job that powers TUI/IM clarification. When ``--escalate`` is set,
    records a ``clarification_escalated_to_worker`` session event for
    operator audit (actual worker transport ships in 0.3.8).
    """
    from supervisor.operator.actions import (
        ActionUnavailable,
        do_escalate_clarification,
        poll_job,
        submit_clarification,
    )
    from supervisor.operator.run_context import RunContext
    from supervisor.operator.session_index import find_session

    rec = find_session(args.run_id)
    if rec is None:
        print(f"Error: run not found: {args.run_id}")
        return 1

    ctx = RunContext.from_run_dict({
        "run_id": rec.run_id,
        "worktree": rec.worktree_root,
        "tag": rec.tag or "local",
        "top_state": rec.top_state,
        "pane_target": rec.pane_target,
        "socket": rec.daemon_socket,
    })

    question = " ".join(args.question).strip()
    if not question:
        print("Error: question is required")
        return 1

    try:
        job = submit_clarification(ctx, question, language=args.language)
    except ActionUnavailable as e:
        print(f"Error: clarify unavailable: {e}")
        return 1
    except (ConnectionRefusedError, FileNotFoundError, OSError) as e:
        # Daemon may die between run lookup and socket connect; same
        # narrow race handled in cmd_observe.
        print(f"Error: daemon unreachable for {rec.run_id}: {e}")
        return 1

    deadline = _time.time() + max(1, args.timeout)
    result: dict = {}
    while _time.time() < deadline:
        try:
            status = poll_job(ctx, job)
        except (ConnectionRefusedError, FileNotFoundError, OSError) as e:
            print(f"Error: daemon unreachable for {rec.run_id}: {e}")
            return 1
        s = status.get("status", "")
        if s == "completed":
            result = status.get("result", {}) or {}
            break
        if s == "failed":
            print(f"Error: clarify failed: {status.get('error', 'unknown')}")
            return 1
        _time.sleep(0.2)
    else:
        print(f"Error: clarify timed out after {args.timeout}s")
        return 1

    answer = result.get("answer", "")
    confidence = result.get("confidence")
    escalation_recommended = result.get("escalation_recommended", False)

    print(f"Run:       {rec.run_id}")
    print(f"Q:         {question}")
    print(f"A:         {answer}")
    if confidence is not None:
        print(f"Confidence: {confidence}")
    print(f"Escalate?  {'yes (recommended)' if escalation_recommended else 'no'}")

    if args.escalate:
        try:
            esc = do_escalate_clarification(
                ctx, question,
                language=args.language,
                reason=(
                    "low_confidence" if escalation_recommended
                    else "operator_initiated"
                ),
                operator=args.operator,
                confidence=confidence,
            )
        except ActionUnavailable as e:
            print(f"Error: escalate unavailable: {e}")
            return 1
        except (ConnectionRefusedError, FileNotFoundError, OSError) as e:
            print(f"Error: daemon unreachable for {rec.run_id}: {e}")
            return 1
        print(f"Escalated: id={esc['escalation_id']} via={esc['source']}")
        print("  (audit-only in 0.3.7; worker-side routing ships in 0.3.8)")
    return 0


def cmd_note(args):
    """Shared notes for cross-run collaboration."""
    from supervisor.daemon.client import DaemonClient

    client = DaemonClient()
    if not client.is_running():
        print("Daemon not running. Start with: thin-supervisor daemon start")
        return 1

    if args.note_action == "add":
        content = " ".join(args.content) if args.content else ""
        if not content:
            print("Error: note content required.")
            return 1
        result = client.note_add(
            content,
            note_type=args.type or "context",
            author_run_id=args.run or "human",
            metadata={},
        )
        if result.get("ok"):
            print(f"Note added: {result['note_id']}")
        else:
            print(f"Error: {result.get('error', 'unknown')}")
            return 1

    elif args.note_action == "list":
        result = client.note_list(
            note_type=args.type or "",
            run_id=args.run or "",
        )
        if not result.get("ok"):
            print(f"Error: {result.get('error', 'unknown')}")
            return 1
        notes = result.get("notes", [])
        if not notes:
            print("No notes.")
            return 0
        for n in notes:
            print(f"[{n['note_id']}] ({n['note_type']}) {n['timestamp'][:19]}")
            print(f"  by: {n['author_run_id']}")
            print(f"  {n['content'][:120]}")
            print()

    return 0


# ------------------------------------------------------------------
# event-plane surfaces (mailbox + waits)
# ------------------------------------------------------------------


def cmd_mailbox(args):
    """List mailbox items for a session (event-plane surface)."""
    from supervisor.daemon.client import DaemonClient

    client = DaemonClient()
    if not client.is_running():
        print("Daemon not running. Start with: thin-supervisor daemon start")
        return 1

    session_id = getattr(args, "session_id", "") or ""
    delivery_status = getattr(args, "delivery_status", "") or ""
    result = client.mailbox_list(session_id=session_id, delivery_status=delivery_status)
    if not result.get("ok"):
        print(f"Error: {result.get('error', 'unknown')}")
        return 1

    items = result.get("items", [])
    if not items:
        print("No mailbox items.")
        return 0

    for it in items:
        created = (it.get("created_at") or "")[:19]
        print(
            f"[{it.get('mailbox_item_id','')}] {created} "
            f"{it.get('source_kind','')}/{it.get('wake_decision','')} "
            f"status={it.get('delivery_status','')}"
        )
        summary = it.get("summary") or ""
        if summary:
            print(f"  {summary[:200]}")
    return 0


def cmd_review_request(args):
    """Register an external review request (works for plan or execute phase)."""
    from supervisor.daemon.client import DaemonClient

    client = DaemonClient()
    if not client.is_running():
        print("Daemon not running. Start with: thin-supervisor daemon start")
        return 1

    result = client.external_task_create(
        session_id=args.session_id,
        provider=args.provider,
        target_ref=args.target_ref,
        run_id=getattr(args, "run_id", "") or "",
        phase=getattr(args, "phase", "execute") or "execute",
        task_kind=getattr(args, "task_kind", "review") or "review",
        blocking_policy=getattr(args, "blocking_policy", "notify_only") or "notify_only",
    )
    if not result.get("ok"):
        print(f"Error: {result.get('error', 'unknown')}")
        return 1
    print(f"request_id: {result['request_id']}")
    print(f"wait_id: {result['wait_id']}")
    print(f"session_id: {result['session_id']}")
    return 0


def cmd_review_result(args):
    """Ingest an external review result for a pending request."""
    from supervisor.daemon.client import DaemonClient

    client = DaemonClient()
    if not client.is_running():
        print("Daemon not running. Start with: thin-supervisor daemon start")
        return 1

    payload: dict = {}
    raw = getattr(args, "payload_json", "") or ""
    if raw:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"Error: --payload-json must be valid JSON: {exc}")
            return 1

    result = client.external_result_ingest(
        request_id=args.request_id,
        provider=args.provider,
        result_kind=args.result_kind,
        summary=getattr(args, "summary", "") or "",
        payload=payload,
        idempotency_key=getattr(args, "idempotency_key", "") or "",
    )
    if not result.get("ok"):
        print(f"Error: {result.get('error', 'unknown')}")
        return 1
    print(f"result_id: {result.get('result_id', '')}")
    print(f"mailbox_item_id: {result.get('mailbox_item_id', '')}")
    if result.get("wake_decision"):
        print(f"wake_decision: {result['wake_decision']}")
    return 0


def cmd_waits(args):
    """List open session waits (event-plane surface)."""
    from supervisor.daemon.client import DaemonClient

    client = DaemonClient()
    if not client.is_running():
        print("Daemon not running. Start with: thin-supervisor daemon start")
        return 1

    session_id = getattr(args, "session_id", "") or ""
    result = client.waits_list(session_id=session_id)
    if not result.get("ok"):
        print(f"Error: {result.get('error', 'unknown')}")
        return 1

    waits = result.get("waits", [])
    if not waits:
        print("No open waits.")
        return 0

    for w in waits:
        entered = (w.get("entered_at") or "")[:19]
        deadline = (w.get("deadline_at") or "").strip() or "-"
        print(
            f"[{w.get('wait_id','')}] {entered} "
            f"{w.get('wait_kind','')} status={w.get('status','')} "
            f"session={w.get('session_id','')} deadline={deadline}"
        )
    return 0


# ------------------------------------------------------------------
# oracle
# ------------------------------------------------------------------


def cmd_oracle(args):
    """Consult an external or fallback oracle for a second opinion."""
    from supervisor.oracle.client import OracleClient

    if args.oracle_action != "consult":
        print("Usage: thin-supervisor-dev oracle consult --question <text> [--file path ...]")
        return 1
    if not args.question:
        print("Error: --question is required.")
        return 1

    client = OracleClient()
    try:
        opinion = client.consult(
            question=args.question,
            file_paths=args.file or [],
            mode=args.mode,
            provider=args.provider,
        )
    except Exception as exc:
        logging.getLogger(__name__).exception("oracle consultation failed")
        print(f"Error: oracle consultation failed: {exc}")
        return 1
    if hasattr(opinion, "to_dict"):
        payload = opinion.to_dict()
    else:
        payload = opinion

    if args.run:
        from supervisor.daemon.client import DaemonClient

        daemon = DaemonClient()
        if not daemon.is_running():
            print("Daemon not running. Start it before saving oracle notes to a run.")
            return 1
        note_lines = [
            f"Oracle consultation: {payload.get('consultation_id', '?')}",
            f"provider: {payload.get('provider', '?')}/{payload.get('model_name', '?')}",
            f"mode: {payload.get('mode', '?')}",
            f"question: {payload.get('question', '')}",
            "",
            payload.get("response_text", ""),
        ]
        try:
            result = daemon.note_add(
                "\n".join(note_lines).strip(),
                note_type="oracle",
                author_run_id=args.run,
                title=f"oracle: {payload.get('question', '')[:60]}",
                metadata=payload,
            )
        except Exception as exc:
            logging.getLogger(__name__).exception("failed to persist oracle note")
            print(f"Error: failed to persist oracle note: {exc}")
            return 1
        if not result.get("ok"):
            print(f"Error: {result.get('error', 'unknown')}")
            return 1

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"Consultation: {payload.get('consultation_id', '?')}")
        print(f"Provider:     {payload.get('provider', '?')}/{payload.get('model_name', '?')}")
        print(f"Mode:         {payload.get('mode', '?')}")
        print(f"Files:        {', '.join(payload.get('files', [])) or '(none)'}")
        print(f"Question:     {payload.get('question', '')}")
        print("\nResponse:\n")
        print(payload.get("response_text", ""))
    return 0


def cmd_spec(args):
    """Manage spec lifecycle operations."""
    if args.spec_action == "approve":
        try:
            approval = approve_spec(args.spec, approved_by=args.by)
        except Exception as exc:
            print(f"Error: {exc}")
            return 1
        print(f"Spec approved: {os.path.abspath(args.spec)}")
        print(f"  by: {approval.get('approved_by', '') or args.by}")
        print(f"  at: {approval.get('approved_at', '')}")
        return 0

    print("Usage: thin-supervisor spec approve --spec <path> [--by human]")
    return 1


def cmd_learn(args):
    """Manage friction logs and user preference memory."""
    runtime_dir = RuntimeConfig.load(getattr(args, "config", None) or CONFIG_FILE).runtime_dir

    if args.learn_action == "friction":
        if args.friction_action == "add":
            try:
                event = append_friction_event(
                    runtime_dir,
                    kind=args.kind,
                    message=args.message,
                    run_id=args.run_id or "",
                    user_id=args.user_id or "default",
                    signals=list(getattr(args, "signal", []) or []),
                )
            except Exception as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
            if getattr(args, "json", False):
                print(json.dumps(event, ensure_ascii=False))
            else:
                print(f"Friction event recorded: {event['event_id']}")
                print(f"  kind: {event['kind']}")
                print(f"  run_id: {event['run_id'] or '-'}")
            return 0

        if args.friction_action == "list":
            try:
                events = list_friction_events(
                    runtime_dir,
                    run_id=args.run_id or "",
                    kind=args.kind or "",
                    user_id=args.user_id or "",
                )
            except Exception as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
            if getattr(args, "json", False):
                print(json.dumps(events, ensure_ascii=False))
            else:
                for event in events:
                    print(f"{event['event_id']} {event['kind']} run={event.get('run_id', '') or '-'}")
            return 0

        if args.friction_action == "summarize":
            try:
                summary = summarize_friction_events(
                    runtime_dir,
                    run_id=args.run_id or "",
                    kind=args.kind or "",
                    user_id=args.user_id or "",
                )
            except Exception as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
            if getattr(args, "json", False):
                print(json.dumps(summary, ensure_ascii=False))
            else:
                print(f"Total events: {summary['total_events']}")
                print(f"By kind: {json.dumps(summary['by_kind'], ensure_ascii=False)}")
                print(f"By signal: {json.dumps(summary['by_signal'], ensure_ascii=False)}")
            return 0

    if args.learn_action == "prefs":
        if args.prefs_action == "set":
            try:
                prefs = save_user_preferences(
                    runtime_dir,
                    {args.key: args.value},
                    user_id=args.user_id or "default",
                )
            except Exception as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
            if getattr(args, "json", False):
                print(json.dumps(prefs, ensure_ascii=False))
            else:
                print(f"Preference saved: {args.key}={args.value}")
            return 0

        if args.prefs_action == "show":
            try:
                prefs = load_user_preferences(runtime_dir, user_id=args.user_id or "default")
            except Exception as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
            if getattr(args, "json", False):
                print(json.dumps(prefs, ensure_ascii=False))
            else:
                for key, value in sorted(prefs.items()):
                    print(f"{key}: {value}")
            return 0

    print("Usage: thin-supervisor-dev learn {friction,prefs} ...")
    return 1


def cmd_eval(args):
    """Run deterministic eval suites for skill and policy behavior."""
    from supervisor.eval import (
        build_candidate_dossier,
        compare_eval_policies,
        current_promotions,
        current_rollouts,
        default_report_dir,
        evaluate_candidate_gate,
        expand_eval_suite,
        list_bundled_suites,
        load_candidate_manifest,
        load_eval_suite,
        list_promotions,
        list_rollouts,
        propose_candidate_policy,
        promote_candidate,
        record_rollout,
        review_candidate_manifest,
        run_eval_suite,
        run_canary_eval,
        run_replay_eval,
        save_candidate_manifest,
        save_eval_report,
        save_eval_suite,
    )

    runtime_dir = RuntimeConfig.load(getattr(args, "config", None) or CONFIG_FILE).runtime_dir

    if args.eval_action == "run":
        try:
            suite_ref = args.suite_file or args.suite
            suite = load_eval_suite(suite_ref)
            report = run_eval_suite(suite, policy=args.policy)
            if getattr(args, "save_report", False) or getattr(args, "output", ""):
                report_path = save_eval_report(
                    report,
                    report_kind="run",
                    runtime_dir=runtime_dir,
                    output_path=getattr(args, "output", ""),
                )
                report["report_path"] = str(report_path)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        if getattr(args, "json", False):
            print(json.dumps(report, ensure_ascii=False))
        else:
            counts = report["counts"]
            print(f"Suite:     {report['suite']}")
            print(f"Policy:    {report['policy']}")
            print(f"Pass rate: {counts['passed']}/{counts['total']} ({counts['pass_rate']:.0%})")
            for item in report["results"]:
                status = "PASS" if item["passed"] else "FAIL"
                print(f"{status} {item['case_id']}")
        return 0

    if args.eval_action == "replay":
        try:
            report = run_replay_eval(args.run_id, runtime_dir=runtime_dir)
            if getattr(args, "save_report", False) or getattr(args, "output", ""):
                report_path = save_eval_report(
                    report,
                    report_kind="replay",
                    runtime_dir=runtime_dir,
                    output_path=getattr(args, "output", ""),
                )
                report["report_path"] = str(report_path)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        if getattr(args, "json", False):
            print(json.dumps(report, ensure_ascii=False))
        else:
            summary = report["summary"]
            print(f"Run:       {report['run_id']}")
            print(f"Decisions: {summary['matched_count']}/{summary['decision_count']} matched")
            print(f"Mismatches:{summary['mismatch_count']}")
            print(f"Pass rate: {summary['pass_rate']:.0%}")
        return 0

    if args.eval_action == "compare":
        try:
            suite_ref = args.suite_file or args.suite
            suite = load_eval_suite(suite_ref)
            report = compare_eval_policies(
                suite,
                baseline_policy=args.baseline_policy,
                candidate_policy=args.candidate_policy,
            )
            if getattr(args, "save_report", False) or getattr(args, "output", ""):
                report_path = save_eval_report(
                    report,
                    report_kind="compare",
                    runtime_dir=runtime_dir,
                    output_path=getattr(args, "output", ""),
                )
                report["report_path"] = str(report_path)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        if getattr(args, "json", False):
            print(json.dumps(report, ensure_ascii=False))
        else:
            wins = report["summary"]["wins"]
            print(f"Suite:      {report['suite']}")
            print(f"Baseline:   {report['baseline_policy']}")
            print(f"Candidate:  {report['candidate_policy']}")
            print(f"Wins:       baseline={wins['baseline']} candidate={wins['candidate']} tie={wins['tie']}")
        return 0

    if args.eval_action == "canary":
        try:
            candidate_id = str(getattr(args, "candidate_id", "") or "").strip()
            phase = str(getattr(args, "phase", None) or "").strip()
            if phase and phase != "shadow" and not candidate_id:
                print("Error: --phase requires --candidate-id", file=sys.stderr)
                return 1
            if candidate_id and not phase:
                phase = "shadow"
            report = run_canary_eval(
                args.run_id,
                runtime_dir=runtime_dir,
                max_mismatch_rate=args.max_mismatch_rate,
                max_friction_events=args.max_friction_events,
            )
            if candidate_id:
                report["rollout_record"] = record_rollout(
                    candidate_id=candidate_id,
                    phase=phase,
                    canary_report=report,
                    runtime_dir=runtime_dir,
                )
            if getattr(args, "save_report", False) or getattr(args, "output", ""):
                report_path = save_eval_report(
                    report,
                    report_kind="canary",
                    runtime_dir=runtime_dir,
                    output_path=getattr(args, "output", ""),
                )
                report["report_path"] = str(report_path)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        if getattr(args, "json", False):
            print(json.dumps(report, ensure_ascii=False))
        else:
            summary = report["summary"]
            print(f"Runs:       {summary['run_count']}")
            print(f"Decision:   {report['decision']}")
            print(f"Pass rate:  {summary['avg_pass_rate']:.0%}")
            print(f"Mismatch:   {summary['mismatch_count']}/{summary['decision_count']} ({summary['mismatch_rate']:.0%})")
            print(f"Friction:   {summary['friction']['total_events']}")
        return 0

    if args.eval_action == "rollout-history":
        try:
            history = list_rollouts(
                runtime_dir=runtime_dir,
                candidate_id=getattr(args, "candidate_id", ""),
            )
            payload = {
                "history": history,
                "current": current_rollouts(history),
            }
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        if getattr(args, "json", False):
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print(f"Rollouts:     {len(payload['history'])}")
            for candidate_id, item in payload["current"].items():
                print(f"{candidate_id}: {item.get('phase', '?')} ({item.get('decision', '?')})")
        return 0

    if args.eval_action == "expand":
        try:
            suite_ref = args.suite_file or args.suite
            suite = load_eval_suite(suite_ref)
            expanded = expand_eval_suite(suite, variants_per_case=args.variants_per_case)
            output = save_eval_suite(expanded, args.output)
            payload = {
                "suite": suite.name,
                "output": str(output),
                "generated_cases": len(expanded.cases),
                "variants_per_case": args.variants_per_case,
            }
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        if getattr(args, "json", False):
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print(f"Expanded suite: {suite.name}")
            print(f"Output:         {output}")
            print(f"Generated:      {len(expanded.cases)} cases")
        return 0

    if args.eval_action == "propose":
        try:
            suite_ref = args.suite_file or args.suite
            suite = load_eval_suite(suite_ref)
            proposal = propose_candidate_policy(
                suite,
                objective=args.objective,
                baseline_policy=args.baseline_policy,
            )
            if getattr(args, "save_report", False) or getattr(args, "output", ""):
                report_path = save_eval_report(
                    proposal,
                    report_kind="proposal",
                    runtime_dir=runtime_dir,
                    output_path=getattr(args, "output", ""),
                )
                proposal["report_path"] = str(report_path)
                manifest_path = save_candidate_manifest(proposal, runtime_dir=runtime_dir)
                proposal["candidate_manifest_path"] = str(manifest_path)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        if getattr(args, "json", False):
            print(json.dumps(proposal, ensure_ascii=False))
        else:
            print(f"Suite:        {proposal['suite']}")
            print(f"Objective:    {proposal['objective']}")
            print(f"Baseline:     {proposal['baseline_policy']}")
            print(f"Recommended:  {proposal['recommended_candidate_policy']}")
            print(f"Rationale:    {proposal['rationale']}")
        return 0

    if args.eval_action == "review-candidate":
        try:
            manifest = load_candidate_manifest(
                candidate_id=getattr(args, "candidate_id", ""),
                manifest_path=getattr(args, "manifest", ""),
                runtime_dir=runtime_dir,
            )
            review = review_candidate_manifest(manifest)
            if getattr(args, "save_report", False) or getattr(args, "output", ""):
                report_path = save_eval_report(
                    {
                        **review,
                        "manifest_path": getattr(args, "manifest", ""),
                    },
                    report_kind="review",
                    runtime_dir=runtime_dir,
                    output_path=getattr(args, "output", ""),
                )
                review["report_path"] = str(report_path)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        if getattr(args, "json", False):
            print(json.dumps(review, ensure_ascii=False))
        else:
            print(f"Candidate:    {review['candidate_id']}")
            print(f"Policy:       {review['candidate_policy']}")
            print(f"Objective:    {review['objective']}")
            print(f"Review:       {review['review_status']}")
            print(f"Next action:  {review['next_action']}")
        return 0

    if args.eval_action == "candidate-status":
        try:
            dossier = build_candidate_dossier(
                candidate_id=getattr(args, "candidate_id", ""),
                manifest_path=getattr(args, "manifest", ""),
                runtime_dir=runtime_dir,
            )
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        if getattr(args, "json", False):
            print(json.dumps(dossier, ensure_ascii=False))
        else:
            print(f"Candidate:    {dossier['candidate']['candidate_id']}")
            print(f"Policy:       {dossier['candidate']['candidate_policy']}")
            print(f"Objective:    {dossier['proposal'].get('objective', '')}")
            print(f"Promoted:     {'yes' if dossier['promotion']['is_current'] else 'no'}")
            print(f"Next action:  {dossier['next_action']}")
        return 0

    if args.eval_action == "gate-candidate":
        try:
            review, gate = _prepare_candidate_gate(
                args,
                runtime_dir=runtime_dir,
                load_candidate_manifest=load_candidate_manifest,
                review_candidate_manifest=review_candidate_manifest,
                load_eval_suite=load_eval_suite,
                run_canary_eval=run_canary_eval,
                evaluate_candidate_gate=evaluate_candidate_gate,
            )
            if getattr(args, "save_report", False) or getattr(args, "output", ""):
                report_path = save_eval_report(
                    gate,
                    report_kind="gate",
                    runtime_dir=runtime_dir,
                    output_path=getattr(args, "output", ""),
                )
                gate["report_path"] = str(report_path)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        if getattr(args, "json", False):
            print(json.dumps(gate, ensure_ascii=False))
        else:
            print(f"Candidate:    {gate['candidate_id']}")
            print(f"Decision:     {gate['decision']}")
            print(f"Review:       {gate['review_status']}")
            print(f"Next action:  {gate['next_action']}")
        return 0

    if args.eval_action == "promote-candidate":
        try:
            review, gate = _prepare_candidate_gate(
                args,
                runtime_dir=runtime_dir,
                load_candidate_manifest=load_candidate_manifest,
                review_candidate_manifest=review_candidate_manifest,
                load_eval_suite=load_eval_suite,
                run_canary_eval=run_canary_eval,
                evaluate_candidate_gate=evaluate_candidate_gate,
            )
            gate["objective"] = review.get("objective", "")
            gate["touched_fragments"] = list(review.get("touched_fragments", []) or [])
            record = promote_candidate(
                gate,
                runtime_dir=runtime_dir,
                approved_by=args.approved_by,
                force=args.force,
            )
            if getattr(args, "save_report", False) or getattr(args, "output", ""):
                report_path = save_eval_report(
                    record,
                    report_kind="promotion",
                    runtime_dir=runtime_dir,
                    output_path=getattr(args, "output", ""),
                )
                record["report_path"] = str(report_path)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        if getattr(args, "json", False):
            print(json.dumps(record, ensure_ascii=False))
        else:
            print(f"Candidate:    {record['candidate_id']}")
            print(f"Status:       {record['status']}")
            print(f"Approved by:  {record['approved_by']}")
        return 0

    if args.eval_action == "improve":
        try:
            payload = _run_eval_improve(
                args,
                runtime_dir=runtime_dir,
                load_eval_suite=load_eval_suite,
                propose_candidate_policy=propose_candidate_policy,
                save_candidate_manifest=save_candidate_manifest,
                review_candidate_manifest=review_candidate_manifest,
                build_candidate_dossier=build_candidate_dossier,
                run_canary_eval=run_canary_eval,
                evaluate_candidate_gate=evaluate_candidate_gate,
                promote_candidate=promote_candidate,
                save_eval_report=save_eval_report,
            )
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        if getattr(args, "json", False):
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print(f"Stage:        {payload['stage']}")
            print(f"Candidate:    {payload['candidate_id']}")
            print(f"Next action:  {payload['next_action']}")
            if payload.get("promotion"):
                print(f"Promotion:    {payload['promotion'].get('status', '?')}")
        return 0

    if args.eval_action == "promotion-history":
        try:
            history = list_promotions(runtime_dir=runtime_dir)
            payload = {
                "history": history,
                "current": current_promotions(history),
            }
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        if getattr(args, "json", False):
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print(f"Promotions:   {len(payload['history'])}")
            for suite, item in payload["current"].items():
                print(f"{suite}: {item.get('candidate_id', '?')} ({item.get('status', '?')})")
        return 0

    if args.eval_action == "list":
        for name in list_bundled_suites():
            print(name)
        return 0

    print("Usage: thin-supervisor-dev eval {list,run,replay,compare,canary,rollout-history,expand,propose,review-candidate,candidate-status,gate-candidate,promote-candidate,improve,promotion-history} ...")
    return 1


def _prepare_candidate_gate(
    args,
    *,
    runtime_dir,
    load_candidate_manifest,
    review_candidate_manifest,
    load_eval_suite,
    run_canary_eval,
    evaluate_candidate_gate,
):
    manifest = load_candidate_manifest(
        candidate_id=getattr(args, "candidate_id", ""),
        manifest_path=getattr(args, "manifest", ""),
        runtime_dir=runtime_dir,
    )
    review = review_candidate_manifest(manifest)
    suite = load_eval_suite(review["suite"])
    canary_report = None
    if getattr(args, "run_id", []):
        canary_report = run_canary_eval(
            args.run_id,
            runtime_dir=runtime_dir,
            max_mismatch_rate=getattr(args, "max_mismatch_rate", 0.25),
            max_friction_events=getattr(args, "max_friction_events", 0),
        )
    gate = evaluate_candidate_gate(review, suite=suite, canary_report=canary_report)
    gate["manifest_path"] = getattr(args, "manifest", "")
    return review, gate


def _run_eval_improve(
    args,
    *,
    runtime_dir,
    load_eval_suite,
    propose_candidate_policy,
    save_candidate_manifest,
    review_candidate_manifest,
    build_candidate_dossier,
    run_canary_eval,
    evaluate_candidate_gate,
    promote_candidate,
    save_eval_report,
):
    suite_ref = args.suite_file or args.suite
    suite = load_eval_suite(suite_ref)
    proposal = propose_candidate_policy(
        suite,
        objective=args.objective,
        baseline_policy=args.baseline_policy,
    )
    manifest_path = save_candidate_manifest(proposal, runtime_dir=runtime_dir)
    manifest = _manifest_from_proposal(proposal)
    review = review_candidate_manifest(manifest)

    if getattr(args, "save_report", False):
        proposal["report_path"] = str(
            save_eval_report(proposal, report_kind="proposal", runtime_dir=runtime_dir)
        )
        review["report_path"] = str(
            save_eval_report(
                {**review, "manifest_path": str(manifest_path)},
                report_kind="review",
                runtime_dir=runtime_dir,
            )
        )

    dossier = build_candidate_dossier(manifest_path=str(manifest_path), runtime_dir=runtime_dir)
    candidate_id = review["candidate_id"]
    payload = {
        "stage": "proposed",
        "candidate_id": candidate_id,
        "candidate_manifest_path": str(manifest_path),
        "next_action": dossier.get("next_action", review.get("next_action", "")),
        "proposal": proposal,
        "review": review,
        "dossier": dossier,
    }

    if getattr(args, "dry_run", False):
        _maybe_save_improve_report(args, payload, runtime_dir=runtime_dir, save_eval_report=save_eval_report)
        return payload

    canary_report = None
    if getattr(args, "run_id", []):
        canary_report = run_canary_eval(
            args.run_id,
            runtime_dir=runtime_dir,
            max_mismatch_rate=getattr(args, "max_mismatch_rate", 0.25),
            max_friction_events=getattr(args, "max_friction_events", 0),
        )
    gate = evaluate_candidate_gate(review, suite=suite, canary_report=canary_report)
    gate["manifest_path"] = str(manifest_path)
    if getattr(args, "save_report", False):
        gate["report_path"] = str(
            save_eval_report(gate, report_kind="gate", runtime_dir=runtime_dir)
        )

    dossier = build_candidate_dossier(manifest_path=str(manifest_path), runtime_dir=runtime_dir)
    payload.update(
        {
            "stage": "gated",
            "next_action": gate.get("next_action", dossier.get("next_action", "")),
            "gate": gate,
            "dossier": dossier,
        }
    )

    if not getattr(args, "approved_by", ""):
        _maybe_save_improve_report(args, payload, runtime_dir=runtime_dir, save_eval_report=save_eval_report)
        return payload

    if gate.get("decision") != "promote" and not getattr(args, "force", False):
        payload["promotion_skipped_reason"] = f"gate decision={gate.get('decision', '')}"
        _maybe_save_improve_report(args, payload, runtime_dir=runtime_dir, save_eval_report=save_eval_report)
        return payload

    gate["objective"] = review.get("objective", "")
    gate["touched_fragments"] = list(review.get("touched_fragments", []) or [])
    promotion = promote_candidate(
        gate,
        runtime_dir=runtime_dir,
        approved_by=args.approved_by,
        force=args.force,
    )
    if getattr(args, "save_report", False):
        promotion["report_path"] = str(
            save_eval_report(promotion, report_kind="promotion", runtime_dir=runtime_dir)
        )

    dossier = build_candidate_dossier(manifest_path=str(manifest_path), runtime_dir=runtime_dir)
    payload.update(
        {
            "stage": "promoted",
            "next_action": dossier.get("next_action", payload.get("next_action", "")),
            "promotion": promotion,
            "dossier": dossier,
        }
    )
    _maybe_save_improve_report(args, payload, runtime_dir=runtime_dir, save_eval_report=save_eval_report)
    return payload


def _manifest_from_proposal(proposal: dict) -> dict:
    candidate = dict(proposal.get("candidate") or {})
    candidate_id = str(candidate.get("candidate_id") or "").strip()
    return {
        "candidate_id": candidate_id,
        "proposal": {
            "suite": proposal.get("suite", ""),
            "objective": proposal.get("objective", ""),
            "baseline_policy": proposal.get("baseline_policy", ""),
            "recommended_candidate_policy": proposal.get("recommended_candidate_policy", ""),
            "rationale": proposal.get("rationale", ""),
        },
        "candidate": candidate,
    }


def _maybe_save_improve_report(args, payload: dict, *, runtime_dir: str, save_eval_report) -> None:
    if not (getattr(args, "save_report", False) or getattr(args, "output", "")):
        return
    payload["report_path"] = str(
        save_eval_report(
            payload,
            report_kind="improve",
            runtime_dir=runtime_dir,
            output_path=getattr(args, "output", ""),
        )
    )


# ------------------------------------------------------------------
# skill install
# ------------------------------------------------------------------


def _read_skill_frontmatter_name(skill_dir: Path) -> str:
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.exists():
        return ""
    for line in skill_file.read_text(encoding="utf-8").splitlines()[:20]:
        if line.startswith("name:"):
            return line.split(":", 1)[1].strip()
    return ""


def _remove_duplicate_skill_entries(
    skill_root: Path,
    *,
    canonical_dirname: str,
    canonical_skill_name: str,
) -> list[Path]:
    removed: list[Path] = []
    if not skill_root.exists():
        return removed

    for child in skill_root.iterdir():
        if child.name == canonical_dirname:
            continue
        if not child.is_dir():
            continue
        if _read_skill_frontmatter_name(child) != canonical_skill_name:
            continue
        if child.is_symlink():
            child.unlink()
        else:
            shutil.rmtree(child)
        removed.append(child)
    return removed


def cmd_skill_install(args):
    """Auto-detect agent and install appropriate skill."""
    import shutil

    # Try editable install path first, then pip install path
    skill_src = Path(__file__).resolve().parent.parent / "skills"
    packaged_skill_src = Path(__file__).resolve().parent.parent / "packaging"
    if not skill_src.exists():
        # pip install: skills may be in the package data or repo checkout
        # Fall back to downloading from GitHub
        print("Skills not found locally. Install from repo:")
        print("  git clone https://github.com/fakechris/thin-supervisor")
        print("  cp -r thin-supervisor/packaging/thin-supervisor-codex ~/.codex/skills/thin-supervisor")
        print("  cp -r thin-supervisor/skills/thin-supervisor ~/.claude/skills/thin-supervisor")
        return 1
    installed = []

    install_roots: dict[Path, set[str]] = {}

    codex_home = Path.home() / ".codex"
    if codex_home.exists():
        install_roots.setdefault((codex_home / "skills").resolve(), set()).add("codex")

    claude_home = Path.home() / ".claude"
    if claude_home.exists():
        install_roots.setdefault((claude_home / "skills").resolve(), set()).add("claude")

    for skill_root, agents in sorted(install_roots.items(), key=lambda item: str(item[0])):
        skill_root.mkdir(parents=True, exist_ok=True)
        dest = skill_root / "thin-supervisor"

        # If both agents share the same underlying skill root, install one
        # canonical visible skill to avoid duplicate /thin-supervisor entries.
        if agents == {"codex"}:
            src = packaged_skill_src / "thin-supervisor-codex"
            label = "Codex"
        else:
            src = skill_src / "thin-supervisor"
            label = "Claude Code" if agents == {"claude"} else "Codex + Claude (shared skill root)"

        if src.exists():
            shutil.copytree(str(src), str(dest), dirs_exist_ok=True)
            removed = _remove_duplicate_skill_entries(
                skill_root,
                canonical_dirname="thin-supervisor",
                canonical_skill_name="thin-supervisor",
            )
            installed.append(f"{label}: {dest}")
            for path in removed:
                installed.append(f"Removed duplicate alias: {path}")

    if installed:
        print("Skills installed:")
        for i in installed:
            print(f"  ✅ {i}")
        print("\nInvoke with /thin-supervisor in your agent.")
    else:
        print("No agent detected (~/.codex or ~/.claude not found).")
        print("Install manually: cp -r packaging/thin-supervisor-codex ~/.codex/skills/thin-supervisor")
        return 1
    return 0


# ------------------------------------------------------------------
# session detect / jsonl / sessions
# ------------------------------------------------------------------


def cmd_session(args):
    """Session detection commands."""
    from supervisor.session_detect import (
        detect_agent,
        detect_session_id,
        find_jsonl_for_session,
        find_latest_jsonl,
        list_sessions,
    )

    if args.session_action == "detect":
        agent = detect_agent()
        sid = detect_session_id(agent)
        if sid:
            print(sid)
        else:
            print(f"error: could not detect session ID (agent={agent})", file=sys.stderr)
            return 1

    elif args.session_action == "jsonl":
        agent = detect_agent()
        sid = detect_session_id(agent)
        path = find_jsonl_for_session(sid, agent) if sid else None
        if path is None:
            path = find_latest_jsonl(agent)
        if path:
            print(str(path))
        else:
            print(f"error: no JSONL transcript found (agent={agent})", file=sys.stderr)
            return 1

    elif args.session_action == "list":
        sessions = list_sessions()
        if not sessions:
            print("No sessions found.")
            return 0
        print(f"{'AGENT':<8} {'CWD':<40} {'MODIFIED'}")
        for s in sessions:
            cwd = s.get("cwd", "") or "(unknown)"
            mod = s.get("modified", "")[:19]
            print(f"{s['agent']:<8} {cwd:<40} {mod}")

    return 0


def _display_view(record) -> dict:
    """Build the display dict for a SessionRecord.

    For orphaned non-paused runs (RUNNING/GATING/VERIFYING without a live
    controller), rewrite the view to PAUSED_FOR_HUMAN with an injected
    escalation reason so the operator sees a concrete next action — this
    preserves the long-standing 'persisted run was left in progress without
    an active daemon worker' UX that operator tooling depends on.
    """
    ep = getattr(record, "event_plane", None) or {}
    ep_tags = []
    if int(ep.get("waits_open", 0) or 0) > 0:
        ep_tags.append("awaiting-review")
    if int(ep.get("mailbox_new", 0) or 0) > 0:
        ep_tags.append(f"mailbox:{int(ep['mailbox_new'])}")

    view = {
        "run_id": record.run_id,
        "worktree_root": record.worktree_root,
        "top_state": record.top_state,
        "current_node": record.current_node,
        "pane_target": record.pane_target,
        "controller_mode": record.controller_mode,
        "pause_reason": record.pause_reason,
        "pause_class": record.pause_class,
        "next_action": record.next_action,
        "status_reason": record.last_checkpoint_summary,
        "daemon_socket": record.daemon_socket,
        "tag": record.tag,
        "event_plane_tags": ep_tags,
    }
    orphan_non_paused = (
        record.is_orphaned
        and record.top_state in {"RUNNING", "GATING", "VERIFYING"}
    )
    if orphan_non_paused:
        reason = (
            "foreground process no longer running"
            if record.controller_mode == "foreground"
            else "persisted run was left in progress without an active daemon worker"
        )
        synthetic = {
            "run_id": record.run_id,
            "top_state": "PAUSED_FOR_HUMAN",
            "current_node_id": record.current_node,
            "pane_target": record.pane_target,
            "spec_path": record.spec_path,
            # Preserve the persisted surface so the resume hint renders
            # the correct `--surface …` suffix for non-tmux runs (jsonl,
            # open_relay).  Empty means summarize_state omits the flag,
            # matching the old dict(state) passthrough.
            "surface_type": record.surface_type,
            "human_escalations": [{"reason": reason}],
        }
        rewritten = summarize_state(synthetic)
        view["top_state"] = "PAUSED_FOR_HUMAN"
        view["pause_reason"] = rewritten.get("pause_reason", reason)
        view["next_action"] = rewritten.get("next_action", "")
    return view


def cmd_status(args):
    """Show all run states, bucketed by liveness tag.

    Global-first by default: scans every discoverable worktree (cwd,
    known_worktrees, live daemon cwds, pane-owner cwds, git worktrees).
    `--local` narrows to the current cwd only.
    """
    from supervisor.operator.session_index import collect_sessions

    local_only = bool(getattr(args, "local", False))
    sessions = collect_sessions(local_only=local_only)

    daemon_runs: list[dict] = []
    foreground_runs: list[dict] = []
    orphaned_runs: list[dict] = []
    completed_runs: list[dict] = []

    for rec in sessions:
        view = _display_view(rec)
        if rec.is_completed:
            completed_runs.append(view)
            continue
        if rec.is_orphaned:
            orphaned_runs.append(view)
            continue
        if rec.is_live and rec.controller_mode == "foreground":
            foreground_runs.append(view)
            continue
        if rec.is_live:
            daemon_runs.append(view)
            continue
        # Persisted local state that isn't actionable (e.g. ABORTED bucket
        # already handled above). Fall through to orphaned bucket so nothing
        # silently disappears from operator view.
        orphaned_runs.append(view)

    if not daemon_runs and not foreground_runs and not orphaned_runs and not completed_runs:
        # Global-first: any live daemon anywhere means "something is up";
        # don't fall back to a cwd-local DaemonClient probe, which would
        # lie when a daemon is running in a different worktree.
        if _list_global_daemons():
            print("Daemons running, no active runs. (see `thin-supervisor ps`)")
        else:
            print("No runs found. No daemons running.")
        return 0

    cwd = os.getcwd()

    def _wt_suffix(view: dict) -> str:
        wt = view.get("worktree_root", "")
        if not wt:
            return ""
        try:
            if Path(wt).resolve() == Path(cwd).resolve():
                return ""
        except (OSError, RuntimeError):
            pass
        return f"  worktree={wt}"

    def _state_label(r: dict) -> str:
        # Decorate paused runs with their pause_class, so operators see the
        # business/safety/review/recovery split at a glance without reading
        # the reason line.  ATTACHED and RECOVERY_NEEDED render as-is — the
        # top_state is already specific.
        state = r.get("top_state", "")
        pclass = r.get("pause_class", "")
        base = f"{state}[{pclass}]" if state == "PAUSED_FOR_HUMAN" and pclass else state
        tags = r.get("event_plane_tags") or []
        return f"{base} [{' '.join(tags)}]" if tags else base

    if daemon_runs:
        print("Active runs:")
        for r in daemon_runs:
            print(
                f"  [daemon]  {r['run_id']}  {_state_label(r)}  "
                f"node={r.get('current_node', '')}  "
                f"pane={r.get('pane_target', '?')}{_wt_suffix(r)}"
            )
            if r.get("status_reason"):
                print(f"    status: {r['status_reason']}")
            if r.get("pause_reason"):
                print(f"    reason: {r['pause_reason']}")
            if r.get("next_action"):
                print(f"    next:   {r['next_action']}")

    if foreground_runs:
        print("Debug foreground runs:")
        for r in foreground_runs:
            print(
                f"  [foreground]  {r.get('run_id', '?')}  "
                f"{_state_label(r)}  "
                f"node={r.get('current_node', '')}  "
                f"pane={r.get('pane_target', '?')}{_wt_suffix(r)}"
            )
            if r.get("status_reason"):
                print(f"    status: {r['status_reason']}")

    if orphaned_runs:
        print("Orphaned local state:")
        for r in orphaned_runs:
            print(
                f"  [orphaned]  {r.get('run_id', '?')}  "
                f"{_state_label(r)}  "
                f"node={r.get('current_node', '')}  "
                f"pane={r.get('pane_target', '?')}{_wt_suffix(r)}"
            )
            if r.get("pause_reason"):
                print(f"    reason: {r['pause_reason']}")
            if r.get("next_action"):
                print(f"    next:   {r['next_action']}")

    if completed_runs:
        print("Recently completed:")
        for r in completed_runs:
            print(
                f"  [done]  {r.get('run_id', '?')}  "
                f"node={r.get('current_node', '')}{_wt_suffix(r)}"
            )
            if r.get("next_action"):
                print(f"    next: {r['next_action']}")

    return 0


# ------------------------------------------------------------------
# bridge (unchanged)
# ------------------------------------------------------------------


def cmd_dashboard(args):
    """Interactive dashboard: numbered list of all runs, press a key to inspect.

    Uses the canonical session_index so `dashboard` sees exactly the same
    run universe as `status` and `tui`. No direct daemon fan-out or local
    scan — collect_sessions() is the single source of truth.
    """
    from supervisor.operator.session_index import collect_sessions

    items: list[dict] = []
    for rec in collect_sessions():
        items.append({
            "run_id": rec.run_id,
            "tag": rec.tag or "local",
            "state": rec.top_state,
            "node": rec.current_node,
            "pane": rec.pane_target or "?",
            "worktree": rec.worktree_root,
        })

    daemons = _list_global_daemons()
    if daemons:
        print("Daemons:")
        for d in daemons:
            state = d.get("state", "active")
            idle = d.get("idle_for_sec", 0)
            idle_str = f"idle {idle}s" if idle > 0 else "active"
            print(f"  PID={d.get('pid', '?')}  {idle_str}  runs={d.get('active_runs', 0)}  {d.get('cwd', '')}")
        print()

    if not items:
        print("No runs found.")
        return 0

    # Print numbered list
    print("Runs:")
    for i, item in enumerate(items, 1):
        wt = f"  ({item['worktree']})" if item.get("worktree") else ""
        print(f"  {i}. [{item['tag']}]  {item['run_id']}  {item['state']}  node={item['node']}  pane={item['pane']}{wt}")

    print(f"\n[1-{len(items)}] inspect  [q] quit")

    # Interactive loop
    while True:
        try:
            choice = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if choice in ("q", "quit", ""):
            break
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(items):
                _inspect_run(items[idx], args)
                # Re-show the list
                print()
                for i, item in enumerate(items, 1):
                    wt = f"  ({item['worktree']})" if item.get("worktree") else ""
                    print(f"  {i}. [{item['tag']}]  {item['run_id']}  {item['state']}  node={item['node']}  pane={item['pane']}{wt}")
                print(f"\n[1-{len(items)}] inspect  [q] quit")
            else:
                print(f"  Invalid: choose 1-{len(items)}")
        except ValueError:
            print(f"  Invalid: choose 1-{len(items)} or q")

    return 0


def _inspect_run(item: dict, args) -> None:
    """Print detailed info for a single run."""
    run_id = item["run_id"]
    tag = item["tag"]
    print(f"\n{'='*60}")
    print(f"Run:        {run_id}")
    print(f"Controller: {tag}")
    print(f"State:      {item['state']}")
    print(f"Node:       {item['node']}")
    print(f"Pane:       {item['pane']}")
    if item.get("worktree"):
        print(f"Worktree:   {item['worktree']}")

    # Load state and session events via StateStore
    # Use the worktree root to find the correct state directory
    worktree = item.get("worktree", "").strip()
    if worktree:
        run_dir = str(Path(worktree) / ".supervisor" / "runtime" / "runs" / run_id)
    else:
        run_dir = str(Path(RUNTIME_DIR) / "runs" / run_id)
    try:
        store = StateStore(run_dir)
        state_data = store.load_raw()
        if state_data:
            print(f"Delivery:   {state_data.get('delivery_state', 'IDLE')}")
            print(f"Spec:       {state_data.get('spec_path', '?')}")
            print(f"Checkpoint: seq={state_data.get('checkpoint_seq', 0)}")
            escalations = state_data.get("human_escalations", [])
            if escalations:
                last = escalations[-1] if isinstance(escalations[-1], dict) else {"reason": str(escalations[-1])}
                print(f"Last pause: {last.get('reason', '?')}")

        events = store.read_recent_session_events(count=5)
        if events:
            total = store.session_event_count()
            print(f"\nRecent events ({total} total):")
            for evt in events:
                print(f"  [{evt.get('event_type', '?')}] {evt.get('timestamp', '')[:19]}")
    except Exception:
        pass

    print(f"{'='*60}")


def cmd_bootstrap(args):
    """Run full zero-setup bootstrap and print step results."""
    from supervisor.bootstrap import bootstrap

    result = bootstrap()
    for step in result.steps:
        icon = "ok" if step["status"] == "ok" else ("skip" if step["status"] == "skipped" else "FAIL")
        print(f"  [{icon}] {step['name']}: {step['message']}")
    if result.missing_credentials:
        print(f"\n  Optional: {len(result.missing_credentials)} credential(s) not configured")
        for cred in result.missing_credentials:
            print(f"    - {cred['key']}: {cred['description']}")
            print(f"      Set with: thin-supervisor config set --key {cred['key']} --value <value>")
    if result.ok:
        print(f"\nReady. pane={result.pane_target} surface={result.surface_type}")
    else:
        print(f"\nBootstrap failed: {result.error}")
        if result.conflict:
            mode = result.conflict.get("controller_mode", "unknown")
            run_id = result.conflict.get("run_id", "?")
            print(f"\n  Existing run: {run_id} (controller: {mode})")
            if mode == "daemon":
                print(f"  Observe:  thin-supervisor observe {run_id}")
                print(f"  Stop:     thin-supervisor run stop {run_id}")
            elif mode == "foreground":
                pid = result.conflict.get("pid", "?")
                print(f"  Stop:     kill {pid}  # foreground debug process")
    return 0 if result.ok else 1


def _render_overview_text(snapshot, *, cwd: str) -> str:
    """Render a ``SystemSnapshot`` as compact plain text.

    Sections (in order):
      1. headline counts
      2. alerts (only when non-empty)
      3. hottest sessions (non-completed first, newest first)
      4. recent system events
    """
    from pathlib import Path as _Path

    lines: list[str] = []
    c = snapshot.counts
    lines.append(
        f"System: daemons={c.daemons}  live={c.live_sessions}  "
        f"foreground={c.foreground_runs}  orphaned={c.orphaned_sessions}  "
        f"completed={c.completed_sessions}"
    )
    lines.append(
        f"Event plane: waits_open={c.waits_open}  mailbox_new={c.mailbox_new}  "
        f"mailbox_acknowledged={c.mailbox_acknowledged}"
    )

    if snapshot.alerts:
        lines.append("")
        lines.append("Alerts:")
        for alert in snapshot.alerts:
            lines.append(f"  [{alert.kind}] {alert.summary}")

    actionable = [s for s in snapshot.sessions if not s.is_completed]
    if actionable:
        lines.append("")
        lines.append("Hottest sessions:")
        for s in actionable[:8]:
            wt_suffix = ""
            try:
                if _Path(s.worktree_root).resolve() != _Path(cwd).resolve():
                    wt_suffix = f"  worktree={s.worktree_root}"
            except (OSError, RuntimeError):
                pass
            tag = s.tag or ("live" if s.is_live else "local")
            ep = s.event_plane or {}
            ep_suffix = ""
            if ep.get("waits_open") or ep.get("mailbox_new"):
                ep_suffix = (
                    f"  waits={ep.get('waits_open', 0)}"
                    f"  mailbox_new={ep.get('mailbox_new', 0)}"
                )
            lines.append(
                f"  [{tag}]  {s.run_id}  {s.top_state}  "
                f"node={s.current_node}{ep_suffix}{wt_suffix}"
            )
            if s.pause_reason:
                lines.append(f"    reason: {s.pause_reason}")
    elif not snapshot.sessions:
        lines.append("")
        lines.append("No sessions found under this worktree scope.")

    if snapshot.recent_timeline:
        lines.append("")
        lines.append("Recent system events:")
        for ev in snapshot.recent_timeline[:10]:
            who = ev.run_id or ev.session_id or "system"
            lines.append(f"  {ev.occurred_at}  [{who}]  {ev.summary}")

    return "\n".join(lines)


def cmd_hook(args):
    """Agent-side hook handler + installer for observation-only surfaces."""
    action = getattr(args, "hook_action", None)
    if action == "stop":
        return _cmd_hook_stop(args)
    if action == "install":
        return _cmd_hook_install(args)
    if action == "uninstall":
        return _cmd_hook_uninstall(args)
    print("Usage: thin-supervisor hook {stop|install|uninstall}")
    return 1


def _cmd_hook_stop(args):
    from supervisor import hook as hook_mod
    from supervisor.session_detect import detect_agent, detect_session_id

    sid = (getattr(args, "session_id", "") or "").strip()
    if not sid:
        sid = detect_session_id(detect_agent()) or ""

    result = hook_mod.run_stop_hook(sid)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return int(result.exit_code)


def _cmd_hook_install(args):
    from supervisor.hook_install import install_stop_hook, resolve_targets

    agent = str(getattr(args, "agent", "both") or "both")
    targets = resolve_targets(agent)
    if not targets:
        print(f"No agent home directories found for --agent {agent}.")
        print("Create ~/.claude and/or ~/.codex first, or install the agent CLI.")
        return 1

    any_changed = False
    for t in targets:
        changed, msg = install_stop_hook(t.settings_path)
        print(f"[{t.label}] {msg}")
        any_changed = any_changed or changed
    if not any_changed:
        print("Nothing to do — hooks already installed.")
    return 0


def _cmd_hook_uninstall(args):
    from supervisor.hook_install import resolve_targets, uninstall_stop_hook

    agent = str(getattr(args, "agent", "both") or "both")
    targets = resolve_targets(agent)
    if not targets:
        print(f"No agent home directories found for --agent {agent}.")
        return 1

    for t in targets:
        changed, msg = uninstall_stop_hook(t.settings_path)
        print(f"[{t.label}] {msg}")
    return 0


def cmd_a2a(args):
    """Run the A2A HTTP adapter.

    Durable state lives in ``.supervisor/runtime/shared/`` (same event-plane
    store the daemon writes to).  A2A runs as a standalone process — it
    does not require the supervisor daemon to be running; it only needs
    write access to the shared runtime root.
    """
    action = getattr(args, "a2a_action", None)
    if action != "serve":
        print("Usage: thin-supervisor a2a serve [--host H] [--port P] [--token-env VAR]")
        return 1

    from supervisor.adapters.a2a.server import A2AServer
    from supervisor.boundary.models import InboundGuardConfig
    from supervisor.event_plane.ingest import EventPlaneIngest
    from supervisor.event_plane.store import EventPlaneStore
    from supervisor.learning import _shared_dir
    from supervisor.storage.system_events import append_system_event

    host = str(getattr(args, "host", "127.0.0.1"))
    port = int(getattr(args, "port", 8081))
    token_env = str(getattr(args, "token_env", "SUPERVISOR_A2A_TOKEN"))
    rate_limit = int(getattr(args, "rate_limit", 20))

    runtime_root = Path(RUNTIME_DIR)
    runtime_root.mkdir(parents=True, exist_ok=True)
    shared = _shared_dir(runtime_root)
    audit_path = shared / "inbound_audit.jsonl"

    token = os.environ.get(token_env, "").strip()
    cfg = InboundGuardConfig(
        auth_token=token,
        rate_limit_per_minute=rate_limit,
        audit_path=audit_path,
    )

    store = EventPlaneStore(str(runtime_root))
    ingest = EventPlaneIngest(store)
    server = A2AServer(host=host, port=port, ingest=ingest, guard_config=cfg)

    # Advertise the listener to overview via system_events.
    append_system_event(
        runtime_root,
        "a2a_started",
        {"host": host, "port": server.port, "auth_required": bool(token)},
    )

    print(
        f"thin-supervisor a2a listening on http://{host}:{server.port} "
        f"(auth {'required' if token else 'off (localhost only)'}, "
        f"rate_limit={rate_limit}/min)"
    )

    # ``BaseServer.shutdown()`` deadlocks if called from the thread
    # currently executing ``serve_forever()`` (it sets a flag and then
    # waits for the serve loop to notice, but that loop is the very
    # thread that got interrupted by the signal).  Run the serve loop
    # on a daemon worker thread and keep the main thread parked on an
    # ``Event`` so the signal handler can cleanly tear the server down.
    import threading

    stop = threading.Event()
    serve_thread = threading.Thread(
        target=server.serve_forever, name="a2a-serve", daemon=True
    )
    serve_thread.start()

    def _graceful(*_a):
        stop.set()

    signal.signal(signal.SIGTERM, _graceful)
    signal.signal(signal.SIGINT, _graceful)
    try:
        try:
            stop.wait()
        except KeyboardInterrupt:
            stop.set()
    finally:
        server.shutdown()
        server.server_close()
        serve_thread.join(timeout=5.0)
        append_system_event(
            runtime_root,
            "a2a_stopped",
            {"host": host, "port": server.port},
        )
    return 0


def cmd_overview(args):
    """System-level snapshot: daemons, sessions, alerts, recent events.

    Flags (via argparse ``Namespace``):
      - ``local``: restrict to the current worktree (same semantics as
        ``status --local``).
      - ``json``: emit machine-readable JSON instead of text.
      - ``watch``: re-render until the user interrupts.  Tests pass
        ``max_iterations`` and ``interval`` to keep the loop bounded.
    """
    import json as _json
    import time as _time

    from supervisor.operator.system_overview import load_system_snapshot

    local_only = bool(getattr(args, "local", False))
    want_json = bool(getattr(args, "json", False))
    watch = bool(getattr(args, "watch", False))
    interval = float(getattr(args, "interval", 2.0) or 0.0)
    max_iterations = int(getattr(args, "max_iterations", 0) or 0)
    # Floor the watch interval so `--interval 0` (or negative) does not
    # degrade into a CPU-bound busy loop rendering the snapshot as fast
    # as it can be computed.
    WATCH_MIN_INTERVAL = 0.5

    def _once() -> None:
        try:
            snapshot = load_system_snapshot(local_only=local_only)
        except Exception as exc:  # noqa: BLE001 — watch must survive transient errors
            # Under --watch, a transient failure (missing registry file
            # mid-rotation, permission blip, partially-written JSON)
            # must not kill the render loop.  Print one diagnostic line
            # and let the next tick retry.
            if watch:
                print(f"(overview: transient error — {exc})")
                return
            raise
        if want_json:
            print(_json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(_render_overview_text(snapshot, cwd=os.getcwd()))

    if not watch:
        _once()
        return 0

    sleep_for = max(interval, WATCH_MIN_INTERVAL)
    iterations = 0
    try:
        while True:
            _once()
            iterations += 1
            if max_iterations and iterations >= max_iterations:
                break
            _time.sleep(sleep_for)
    except KeyboardInterrupt:
        pass
    return 0


def cmd_config_set(args):
    """Set a config value in global or project scope."""
    from dataclasses import fields as dc_fields
    from supervisor.config import RuntimeConfig, coerce_config_value
    from supervisor.credentials import persist_credential

    known = {f.name for f in dc_fields(RuntimeConfig)}
    if args.key not in known:
        print(f"Error: unknown config key '{args.key}'")
        return 1

    value = coerce_config_value(args.key, args.value)
    persist_credential(args.key, value, scope=args.scope)
    print(f"Saved {args.key} to {args.scope} config.")
    return 0


def cmd_bridge(args):
    """Tmux pane operations (read, type, keys, list, id, doctor)."""
    from supervisor.terminal.adapter import TerminalAdapter, TerminalAdapterError

    action = args.bridge_action

    if action == "id":
        pane_id = os.environ.get("TMUX_PANE", "")
        if pane_id:
            print(pane_id)
        else:
            print("error: not inside a tmux pane", file=sys.stderr)
            return 1
        return 0

    if action == "doctor":
        adapter = TerminalAdapter(args.target or "%0")
        info = adapter.doctor()
        print(f"Socket:  {info['socket']}")
        print(f"Panes:   {info['pane_count']}")
        if info["issues"]:
            for issue in info["issues"]:
                print(f"  Issue: {issue}")
        print(f"Status:  {'OK' if info['ok'] else 'ISSUES FOUND'}")
        return 0 if info["ok"] else 1

    if action == "list":
        adapter = TerminalAdapter("%0")
        try:
            panes = adapter.list_panes()
        except TerminalAdapterError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"{'TARGET':<8} {'SESSION:WIN':<15} {'SIZE':<12} {'PROCESS':<12} {'LABEL':<12} {'CWD'}")
        for p in panes:
            print(f"{p.pane_id:<8} {p.session_window:<15} {p.size:<12} {p.process:<12} {p.label or '-':<12} {p.cwd}")
        return 0

    if not args.target:
        print("error: target pane required (label or %id)", file=sys.stderr)
        return 1

    adapter = TerminalAdapter(args.target)

    try:
        if action == "read":
            if args.extra:
                try:
                    lines = int(args.extra[0])
                    if lines <= 0:
                        raise ValueError
                except ValueError:
                    print(f"error: lines must be a positive integer, got '{args.extra[0]}'", file=sys.stderr)
                    return 1
            else:
                lines = 50
            text = adapter.read(lines=lines)
            print(text, end="")
        elif action == "type":
            if not args.extra:
                print("error: text argument required", file=sys.stderr)
                return 1
            adapter.read()
            adapter.type_text(" ".join(args.extra))
        elif action == "keys":
            if not args.extra:
                print("error: key argument(s) required", file=sys.stderr)
                return 1
            adapter.read()
            adapter.send_keys(*args.extra)
        elif action == "name":
            if not args.extra:
                print("error: label argument required", file=sys.stderr)
                return 1
            adapter.name_pane(args.extra[0])
        else:
            print(f"error: unknown bridge action '{action}'", file=sys.stderr)
            return 1
    except TerminalAdapterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


# ------------------------------------------------------------------
# Migration shim for removed legacy syntax: thin-supervisor run <spec> --pane <p> [--daemon]
# ------------------------------------------------------------------


def cmd_run_legacy(args):
    """Legacy entrypoint kept only to print a migration error."""
    print("Legacy run syntax has been removed.")
    print("Use one of:")
    print("  thin-supervisor run register --spec <spec> --pane <pane>")
    print("  thin-supervisor run foreground --spec <spec> --pane <pane>")
    return 1


def _run_event_file(event_file, spec, state, store, loop):
    event = json.loads(Path(event_file).read_text())
    if event.get("type") == "agent_output" and "text" in event.get("payload", {}):
        adapter = TranscriptAdapter()
        cp = adapter.parse_checkpoint(event["payload"]["text"])
        event["payload"]["checkpoint"] = cp.to_dict() if cp else {}
    store.append_event(event)
    loop.handle_event(state, event)
    from supervisor.domain.enums import TopState as TS
    if state.top_state == TS.GATING:
        decision = loop.gate(spec, state)
        store.append_decision(decision.to_dict())
        loop.apply_decision(spec, state, decision)
    if state.top_state == TS.VERIFYING:
        verification = loop.verify_current_node(spec, state)
        store.append_event({"type": "verification_finished", "payload": verification})
        loop.apply_verification(spec, state, verification)
    store.save(state)
    print(json.dumps(state.to_dict(), ensure_ascii=False, indent=2))
    return 0


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def _add_oracle_parser(sub) -> None:
    p_oracle = sub.add_parser("oracle", help="Consult an external or fallback oracle")
    oracle_sub = p_oracle.add_subparsers(dest="oracle_action")
    p_oracle_consult = oracle_sub.add_parser("consult", help="Get an advisory second opinion")
    p_oracle_consult.add_argument("--question", required=True, help="Question for the oracle")
    p_oracle_consult.add_argument("--file", action="append", default=[], help="Relevant file path (repeatable)")
    p_oracle_consult.add_argument("--mode", default="review", choices=["review", "plan", "debug"])
    p_oracle_consult.add_argument("--provider", default="auto", choices=["auto", "openai", "deepseek", "anthropic"])
    p_oracle_consult.add_argument("--run", default="", help="Optional run ID to persist as a shared oracle note")
    p_oracle_consult.add_argument("--json", action="store_true", help="Print JSON output")


def _add_learn_parser(sub) -> None:
    p_learn = sub.add_parser("learn", help="Persist learning signals for future skill evolution")
    learn_sub = p_learn.add_subparsers(dest="learn_action")

    p_learn_friction = learn_sub.add_parser("friction", help="Add or list friction events")
    friction_sub = p_learn_friction.add_subparsers(dest="friction_action")
    p_friction_add = friction_sub.add_parser("add", help="Record a friction event")
    p_friction_add.add_argument("--kind", required=True, help="Event kind")
    p_friction_add.add_argument("--message", required=True, help="Human-readable event summary")
    p_friction_add.add_argument("--run-id", default="", help="Related run id")
    p_friction_add.add_argument("--user-id", default="default", help="Preference owner / user id")
    p_friction_add.add_argument("--signal", action="append", default=[], help="Structured signal(s)")
    p_friction_add.add_argument("--json", action="store_true", help="Print JSON output")
    p_friction_add.add_argument("--config", default=None, help="Config YAML path")

    p_friction_list = friction_sub.add_parser("list", help="List friction events")
    p_friction_list.add_argument("--kind", default="", help="Filter by kind")
    p_friction_list.add_argument("--run-id", default="", help="Filter by run id")
    p_friction_list.add_argument("--user-id", default="", help="Filter by user id")
    p_friction_list.add_argument("--json", action="store_true", help="Print JSON output")
    p_friction_list.add_argument("--config", default=None, help="Config YAML path")

    p_friction_summarize = friction_sub.add_parser("summarize", help="Summarize friction events")
    p_friction_summarize.add_argument("--kind", default="", help="Filter by kind")
    p_friction_summarize.add_argument("--run-id", default="", help="Filter by run id")
    p_friction_summarize.add_argument("--user-id", default="", help="Filter by user id")
    p_friction_summarize.add_argument("--json", action="store_true", help="Print JSON output")
    p_friction_summarize.add_argument("--config", default=None, help="Config YAML path")

    p_learn_prefs = learn_sub.add_parser("prefs", help="Set or show user preference memory")
    prefs_sub = p_learn_prefs.add_subparsers(dest="prefs_action")
    p_prefs_set = prefs_sub.add_parser("set", help="Persist one preference")
    p_prefs_set.add_argument("--key", required=True, help="Preference key")
    p_prefs_set.add_argument("--value", required=True, help="Preference value")
    p_prefs_set.add_argument("--user-id", default="default", help="Preference owner / user id")
    p_prefs_set.add_argument("--json", action="store_true", help="Print JSON output")
    p_prefs_set.add_argument("--config", default=None, help="Config YAML path")

    p_prefs_show = prefs_sub.add_parser("show", help="Show user preferences")
    p_prefs_show.add_argument("--user-id", default="default", help="Preference owner / user id")
    p_prefs_show.add_argument("--json", action="store_true", help="Print JSON output")
    p_prefs_show.add_argument("--config", default=None, help="Config YAML path")


def _add_eval_parser(sub) -> None:
    p_eval = sub.add_parser("eval", help="Run deterministic skill/policy eval suites")
    eval_sub = p_eval.add_subparsers(dest="eval_action")
    eval_sub.add_parser("list", help="List bundled eval suites")
    p_eval_run = eval_sub.add_parser("run", help="Run a bundled or explicit eval suite")
    p_eval_run.add_argument("--suite", default="approval-core", help="Bundled suite name")
    p_eval_run.add_argument("--suite-file", default=None, help="Path to a JSONL eval suite")
    p_eval_run.add_argument("--policy", default="builtin-approval-v1", help="Policy adapter to evaluate")
    p_eval_run.add_argument("--output", default="", help="Optional report output path")
    p_eval_run.add_argument("--save-report", action="store_true", help="Persist report under .supervisor/evals/reports/")
    p_eval_run.add_argument("--config", default=None, help="Config YAML path")
    p_eval_run.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    p_eval_replay = eval_sub.add_parser("replay", help="Replay a historical run as an eval report")
    p_eval_replay.add_argument("--run-id", required=True, help="Historical run id")
    p_eval_replay.add_argument("--output", default="", help="Optional report output path")
    p_eval_replay.add_argument("--save-report", action="store_true", help="Persist report under .supervisor/evals/reports/")
    p_eval_replay.add_argument("--config", default=None, help="Config YAML path")
    p_eval_replay.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    p_eval_compare = eval_sub.add_parser("compare", help="Blind-compare baseline vs candidate policy on a suite")
    p_eval_compare.add_argument("--suite", default="approval-core", help="Bundled suite name")
    p_eval_compare.add_argument("--suite-file", default=None, help="Path to a JSONL eval suite")
    p_eval_compare.add_argument("--baseline-policy", default="builtin-approval-v1", help="Baseline policy id")
    p_eval_compare.add_argument("--candidate-policy", required=True, help="Candidate policy id")
    p_eval_compare.add_argument("--output", default="", help="Optional report output path")
    p_eval_compare.add_argument("--save-report", action="store_true", help="Persist report under .supervisor/evals/reports/")
    p_eval_compare.add_argument("--config", default=None, help="Config YAML path")
    p_eval_compare.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    p_eval_canary = eval_sub.add_parser("canary", help="Aggregate replay/friction over recent real runs")
    p_eval_canary.add_argument("--run-id", action="append", required=True, help="Run id to include (repeatable)")
    p_eval_canary.add_argument("--candidate-id", default="", help="Optional candidate id to bind rollout bookkeeping")
    p_eval_canary.add_argument("--phase", choices=["shadow", "limited"], default=None, help="Rollout phase when candidate-id is provided (defaults to shadow)")
    p_eval_canary.add_argument("--max-mismatch-rate", type=float, default=0.25, help="Promotion hold threshold for mismatch rate")
    p_eval_canary.add_argument("--max-friction-events", type=int, default=0, help="Promotion hold threshold for friction events")
    p_eval_canary.add_argument("--output", default="", help="Optional report output path")
    p_eval_canary.add_argument("--save-report", action="store_true", help="Persist report under .supervisor/evals/reports/")
    p_eval_canary.add_argument("--config", default=None, help="Config YAML path")
    p_eval_canary.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    p_eval_rollouts = eval_sub.add_parser("rollout-history", help="Show candidate rollout history")
    p_eval_rollouts.add_argument("--candidate-id", default="", help="Optional candidate id filter")
    p_eval_rollouts.add_argument("--config", default=None, help="Config YAML path")
    p_eval_rollouts.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    p_eval_expand = eval_sub.add_parser("expand", help="Generate synthetic variants from a suite")
    p_eval_expand.add_argument("--suite", default="approval-core", help="Bundled suite name")
    p_eval_expand.add_argument("--suite-file", default=None, help="Path to a JSONL eval suite")
    p_eval_expand.add_argument("--output", required=True, help="Output JSONL path")
    p_eval_expand.add_argument("--variants-per-case", type=int, default=2, help="Synthetic variants to generate per case")
    p_eval_expand.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    p_eval_propose = eval_sub.add_parser("propose", help="Propose a constrained candidate policy for an objective")
    p_eval_propose.add_argument("--suite", default="approval-core", help="Bundled suite name")
    p_eval_propose.add_argument("--suite-file", default=None, help="Path to a JSONL eval suite")
    p_eval_propose.add_argument("--baseline-policy", default="builtin-approval-v1", help="Baseline policy id")
    p_eval_propose.add_argument("--objective", required=True, choices=["reduce_repeated_confirmation", "reduce_false_approval"], help="Optimization objective")
    p_eval_propose.add_argument("--output", default="", help="Optional report output path")
    p_eval_propose.add_argument("--save-report", action="store_true", help="Persist report under .supervisor/evals/reports/")
    p_eval_propose.add_argument("--config", default=None, help="Config YAML path")
    p_eval_propose.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    p_eval_review = eval_sub.add_parser("review-candidate", help="Review a persisted candidate manifest")
    p_eval_review.add_argument("--candidate-id", default="", help="Candidate id to load from .supervisor/evals/candidates/")
    p_eval_review.add_argument("--manifest", default="", help="Explicit candidate manifest path")
    p_eval_review.add_argument("--output", default="", help="Optional report output path")
    p_eval_review.add_argument("--save-report", action="store_true", help="Persist report under .supervisor/evals/reports/")
    p_eval_review.add_argument("--config", default=None, help="Config YAML path")
    p_eval_review.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    p_eval_status = eval_sub.add_parser("candidate-status", help="Show a unified lifecycle dossier for a candidate")
    p_eval_status.add_argument("--candidate-id", default="", help="Candidate id to load from .supervisor/evals/candidates/")
    p_eval_status.add_argument("--manifest", default="", help="Explicit candidate manifest path")
    p_eval_status.add_argument("--config", default=None, help="Config YAML path")
    p_eval_status.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    p_eval_gate = eval_sub.add_parser("gate-candidate", help="Run the bounded promotion gate for a candidate")
    p_eval_gate.add_argument("--candidate-id", default="", help="Candidate id to load from .supervisor/evals/candidates/")
    p_eval_gate.add_argument("--manifest", default="", help="Explicit candidate manifest path")
    p_eval_gate.add_argument("--run-id", action="append", default=[], help="Optional canary run id (repeatable)")
    p_eval_gate.add_argument("--max-mismatch-rate", type=float, default=0.25, help="Canary hold threshold for mismatch rate")
    p_eval_gate.add_argument("--max-friction-events", type=int, default=0, help="Canary hold threshold for friction events")
    p_eval_gate.add_argument("--output", default="", help="Optional report output path")
    p_eval_gate.add_argument("--save-report", action="store_true", help="Persist report under .supervisor/evals/reports/")
    p_eval_gate.add_argument("--config", default=None, help="Config YAML path")
    p_eval_gate.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    p_eval_promote = eval_sub.add_parser("promote-candidate", help="Record a manually approved promotion decision")
    p_eval_promote.add_argument("--candidate-id", default="", help="Candidate id to load from .supervisor/evals/candidates/")
    p_eval_promote.add_argument("--manifest", default="", help="Explicit candidate manifest path")
    p_eval_promote.add_argument("--approved-by", required=True, help="Approver identity, e.g. human")
    p_eval_promote.add_argument("--force", action="store_true", help="Allow promotion even if gate says hold/rollback")
    p_eval_promote.add_argument("--run-id", action="append", default=[], help="Optional canary run id (repeatable)")
    p_eval_promote.add_argument("--max-mismatch-rate", type=float, default=0.25, help="Canary hold threshold for mismatch rate")
    p_eval_promote.add_argument("--max-friction-events", type=int, default=0, help="Canary hold threshold for friction events")
    p_eval_promote.add_argument("--output", default="", help="Optional report output path")
    p_eval_promote.add_argument("--save-report", action="store_true", help="Persist report under .supervisor/evals/reports/")
    p_eval_promote.add_argument("--config", default=None, help="Config YAML path")
    p_eval_promote.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    p_eval_improve = eval_sub.add_parser("improve", help="Run propose -> review/status -> gate -> optional promote as one orchestration command")
    p_eval_improve.add_argument("--suite", default="approval-core", help="Bundled suite name")
    p_eval_improve.add_argument("--suite-file", default=None, help="Path to a JSONL eval suite")
    p_eval_improve.add_argument("--baseline-policy", default="builtin-approval-v1", help="Baseline policy id")
    p_eval_improve.add_argument("--objective", required=True, choices=["reduce_repeated_confirmation", "reduce_false_approval"], help="Optimization objective")
    p_eval_improve.add_argument("--run-id", action="append", default=[], help="Optional canary run id (repeatable)")
    p_eval_improve.add_argument("--max-mismatch-rate", type=float, default=0.25, help="Canary hold threshold for mismatch rate")
    p_eval_improve.add_argument("--max-friction-events", type=int, default=0, help="Canary hold threshold for friction events")
    p_eval_improve.add_argument("--approved-by", default="", help="Approver identity; omitted means stop before promotion")
    p_eval_improve.add_argument("--force", action="store_true", help="Allow promotion even if gate is not yet promote")
    p_eval_improve.add_argument("--dry-run", action="store_true", help="Stop after proposal/review without gating or promotion")
    p_eval_improve.add_argument("--output", default="", help="Optional final improve report output path")
    p_eval_improve.add_argument("--save-report", action="store_true", help="Persist stage reports under .supervisor/evals/reports/")
    p_eval_improve.add_argument("--config", default=None, help="Config YAML path")
    p_eval_improve.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    p_eval_history = eval_sub.add_parser("promotion-history", help="Show promotion registry history")
    p_eval_history.add_argument("--config", default=None, help="Config YAML path")
    p_eval_history.add_argument("--json", action="store_true", help="Print machine-readable JSON")


def build_runtime_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thin-supervisor",
        description="Thin tmux sidecar supervisor for AI coding agent workflows",
    )
    sub = parser.add_subparsers(dest="command")

    p_init = sub.add_parser("init", help="Initialize .supervisor/ in current project")
    p_init.add_argument("--force", action="store_true")
    p_init.add_argument("--repair", action="store_true", help="Repair a partial .supervisor/ scaffold without overwriting config")

    p_deinit = sub.add_parser("deinit", help="Remove .supervisor/ directory")
    p_deinit.add_argument("--force", action="store_true")

    p_daemon = sub.add_parser("daemon", help="Manage the supervisor daemon")
    daemon_sub = p_daemon.add_subparsers(dest="daemon_action")
    p_daemon_start = daemon_sub.add_parser("start", help="Start daemon")
    p_daemon_start.add_argument("--config", default=None)
    daemon_sub.add_parser("stop", help="Stop daemon")

    p_run = sub.add_parser("run", help="Manage supervisor runs")
    run_sub = p_run.add_subparsers(dest="run_action")
    p_register = run_sub.add_parser("register", help="Register a new run with the daemon")
    p_register.add_argument("--spec", required=True, help="Path to spec YAML")
    p_register.add_argument("--pane", default=None, help="Surface target (tmux pane, oly session, or jsonl path)")
    p_register.add_argument("--target", default=None, help="Alias for --pane")
    p_register.add_argument("--surface", default=None, help="Override surface type (tmux|open_relay|jsonl)")
    p_register.add_argument("--config", default=None)

    p_foreground = run_sub.add_parser("foreground", help="Run sidecar in foreground (debug only)")
    p_foreground.add_argument("--spec", required=True, help="Path to spec YAML")
    p_foreground.add_argument("--pane", default=None, help="Surface target")
    p_foreground.add_argument("--target", default=None, help="Alias for --pane")
    p_foreground.add_argument("--surface", default=None, help="Override surface type")
    p_foreground.add_argument("--config", default=None)

    p_run_stop = run_sub.add_parser("stop", help="Stop a specific run")
    p_run_stop.add_argument("run_id", help="Run ID to stop")

    p_resume = run_sub.add_parser("resume", help="Resume a paused/crashed run")
    p_resume.add_argument("--spec", required=True, help="Path to spec YAML")
    p_resume.add_argument("--pane", default=None, help="Surface target")
    p_resume.add_argument("--target", default=None, help="Alias for --pane")
    p_resume.add_argument("--surface", default=None, help="Surface type override")
    p_resume.add_argument("--config", default=None)

    p_review = run_sub.add_parser("review", help="Record reviewer acknowledgement")
    p_review.add_argument("run_id", help="Run ID to mark as reviewed")
    p_review.add_argument("--by", required=True, choices=["human", "stronger_reviewer"])

    p_export = run_sub.add_parser("export", help="Export a run's durable history as JSON")
    p_export.add_argument("run_id", help="Run ID to export")
    p_export.add_argument("--output", default="", help="Optional file path for exported JSON")
    p_export.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    p_export.add_argument("--config", default=None, help="Optional config file")

    p_summarize = run_sub.add_parser("summarize", help="Summarize a historical run")
    p_summarize.add_argument("run_id", help="Run ID to summarize")
    p_summarize.add_argument("--json", action="store_true", help="Print JSON output")
    p_summarize.add_argument("--config", default=None, help="Optional config file")

    p_replay = run_sub.add_parser("replay", help="Replay historical gate decisions without execution")
    p_replay.add_argument("run_id", help="Run ID to replay")
    p_replay.add_argument("--json", action="store_true", help="Print JSON output")
    p_replay.add_argument("--config", default=None, help="Optional config file")

    p_postmortem = run_sub.add_parser("postmortem", help="Write a markdown postmortem for a run")
    p_postmortem.add_argument("run_id", help="Run ID to analyze")
    p_postmortem.add_argument("--output", default="", help="Optional markdown output path")
    p_postmortem.add_argument("--config", default=None, help="Optional config file")

    p_run.add_argument("spec_path", nargs="?", default=None, help=argparse.SUPPRESS)
    p_run.add_argument("--pane", default=None, help=argparse.SUPPRESS)
    p_run.add_argument("--config", default=None, help=argparse.SUPPRESS)
    p_run.add_argument("--event-file", default=None, help=argparse.SUPPRESS)
    p_run.add_argument("--dry-run", action="store_true", help=argparse.SUPPRESS)
    p_run.add_argument("--daemon", "-d", action="store_true", help=argparse.SUPPRESS)

    sub.add_parser("list", help="List all active runs (detailed)")
    sub.add_parser("ps", help="List all registered daemons across worktrees")

    p_pane_owner = sub.add_parser("pane-owner", help="Show which run owns a pane")
    p_pane_owner.add_argument("pane", help="tmux pane target")

    p_observe = sub.add_parser("observe", help="Read-only observation of a run")
    p_observe.add_argument("run_id", help="Run ID to observe")

    p_clarify = sub.add_parser(
        "clarify",
        help="Ask the explainer about a run; optionally escalate to the worker",
    )
    p_clarify.add_argument("run_id", help="Run ID")
    p_clarify.add_argument("question", nargs="+", help="Free-form question")
    p_clarify.add_argument(
        "--language", default="en", choices=["en", "zh"],
        help="Answer language",
    )
    p_clarify.add_argument(
        "--escalate", action="store_true",
        help="Record an operator decision to escalate this to the worker "
             "(audit-only in 0.3.7; worker transport in 0.3.8)",
    )
    p_clarify.add_argument(
        "--operator", default="",
        help="Optional operator identifier recorded in the escalation event",
    )
    p_clarify.add_argument(
        "--timeout", type=int, default=30,
        help="Clarify poll timeout (seconds)",
    )

    p_note = sub.add_parser("note", help="Shared notes for cross-run collaboration")
    note_sub = p_note.add_subparsers(dest="note_action")
    p_note_add = note_sub.add_parser("add", help="Add a note")
    p_note_add.add_argument("content", nargs="*", help="Note content")
    p_note_add.add_argument("--type", default="context", help="Note type: context|finding|handoff|warning|question")
    p_note_add.add_argument("--run", default="", help="Author run ID")
    p_note_list = note_sub.add_parser("list", help="List notes")
    p_note_list.add_argument("--type", default="", help="Filter by type")
    p_note_list.add_argument("--run", default="", help="Filter by author run ID")

    p_review = sub.add_parser("review", help="Async external review (event-plane request/result)")
    review_sub = p_review.add_subparsers(dest="review_action")
    p_review_req = review_sub.add_parser("request", help="Register an external review request")
    p_review_req.add_argument("--session-id", dest="session_id", required=True)
    p_review_req.add_argument("--provider", required=True)
    p_review_req.add_argument("--target-ref", dest="target_ref", required=True)
    p_review_req.add_argument("--run-id", dest="run_id", default="")
    p_review_req.add_argument("--phase", default="execute", choices=["execute", "post_implement", "finish", "plan"])
    p_review_req.add_argument("--task-kind", dest="task_kind", default="review")
    p_review_req.add_argument(
        "--blocking-policy",
        dest="blocking_policy",
        default="notify_only",
        choices=["block_session", "notify_only", "advisory_only"],
    )
    p_review_res = review_sub.add_parser("result", help="Ingest an external review result")
    p_review_res.add_argument("--request-id", dest="request_id", required=True)
    p_review_res.add_argument("--provider", required=True)
    p_review_res.add_argument("--result-kind", dest="result_kind", required=True)
    p_review_res.add_argument("--summary", default="")
    p_review_res.add_argument("--payload-json", dest="payload_json", default="")
    p_review_res.add_argument("--idempotency-key", dest="idempotency_key", default="")

    p_mailbox = sub.add_parser("mailbox", help="List event-plane mailbox items for a session")
    p_mailbox.add_argument("--session-id", dest="session_id", required=True, help="Session ID")
    p_mailbox.add_argument(
        "--delivery-status",
        dest="delivery_status",
        default="",
        help="Filter by delivery_status (new|surfaced|acknowledged|consumed)",
    )

    p_waits = sub.add_parser("waits", help="List open event-plane session waits")
    p_waits.add_argument("--session-id", dest="session_id", default="", help="Filter by session ID")

    p_skill = sub.add_parser("skill", help="Skill management")
    skill_sub = p_skill.add_subparsers(dest="skill_action")
    skill_sub.add_parser("install", help="Auto-detect agent and install skill")

    p_spec = sub.add_parser("spec", help="Manage spec lifecycle state")
    spec_sub = p_spec.add_subparsers(dest="spec_action")
    p_spec_approve = spec_sub.add_parser("approve", help="Mark a draft spec approved for execution")
    p_spec_approve.add_argument("--spec", required=True, help="Path to spec YAML")
    p_spec_approve.add_argument("--by", default="human", help="Approver label")

    p_session = sub.add_parser("session", help="Session detection")
    session_sub = p_session.add_subparsers(dest="session_action")
    session_sub.add_parser("detect", help="Detect current session ID")
    session_sub.add_parser("jsonl", help="Find current session JSONL path")
    session_sub.add_parser("list", help="List all discoverable sessions")

    p_status = sub.add_parser("status", help="Show all run states")
    p_status.add_argument("--config", default=None)
    p_status.add_argument(
        "--local",
        action="store_true",
        help="Restrict to the current worktree (skip other known worktrees)",
    )

    p_overview = sub.add_parser(
        "overview",
        help="System-level snapshot: daemons, sessions, alerts, recent events",
    )
    p_overview.add_argument("--config", default=None)
    p_overview.add_argument(
        "--local",
        action="store_true",
        help="Restrict to the current worktree (skip other known worktrees)",
    )
    p_overview.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON",
    )
    p_overview.add_argument(
        "--watch", action="store_true", help="Re-render until interrupted",
    )
    p_overview.add_argument(
        "--interval", type=float, default=2.0,
        help="Seconds between renders under --watch (default 2.0)",
    )
    p_overview.add_argument(
        "--max-iterations", dest="max_iterations", type=int, default=0,
        help=argparse.SUPPRESS,  # test hook to bound the watch loop
    )

    sub.add_parser("stop", help="Stop the supervisor daemon (alias for daemon stop)")

    p_bridge = sub.add_parser("bridge", help="Tmux pane operations")
    p_bridge.add_argument("bridge_action", choices=["read", "type", "keys", "list", "id", "doctor", "name"])
    p_bridge.add_argument("target", nargs="?", default=None)
    p_bridge.add_argument("extra", nargs="*")

    sub.add_parser("bootstrap", help="Auto-detect, init, start daemon, and validate surface")
    sub.add_parser("dashboard", help="Interactive run dashboard — numbered list with drill-in")
    sub.add_parser("tui", help="Operator TUI — three-pane view with explain/drift/pause")

    p_hook = sub.add_parser(
        "hook",
        help="Agent-side hook handlers (Stop hook for observation-only surfaces)",
    )
    hook_sub = p_hook.add_subparsers(dest="hook_action")
    p_hook_stop = hook_sub.add_parser(
        "stop",
        help="Deliver the pending supervisor instruction to the agent on stop",
    )
    p_hook_stop.add_argument(
        "--session-id",
        dest="session_id",
        default="",
        help="Override session ID (defaults to auto-detect)",
    )
    p_hook_install = hook_sub.add_parser(
        "install",
        help="Register the Stop hook in Claude Code / Codex settings",
    )
    p_hook_install.add_argument(
        "--agent",
        choices=("claude", "codex", "both"),
        default="both",
        help="Which agent(s) to configure (default both)",
    )
    p_hook_uninstall = hook_sub.add_parser(
        "uninstall",
        help="Remove the Stop hook from Claude Code / Codex settings",
    )
    p_hook_uninstall.add_argument(
        "--agent",
        choices=("claude", "codex", "both"),
        default="both",
        help="Which agent(s) to clean up (default both)",
    )

    p_a2a = sub.add_parser("a2a", help="A2A (Agent-to-Agent) inbound protocol adapter")
    a2a_sub = p_a2a.add_subparsers(dest="a2a_action")
    p_a2a_serve = a2a_sub.add_parser("serve", help="Run the A2A HTTP server")
    p_a2a_serve.add_argument("--host", default="127.0.0.1", help="Bind address (default 127.0.0.1)")
    p_a2a_serve.add_argument("--port", type=int, default=8081, help="Bind port (default 8081)")
    p_a2a_serve.add_argument(
        "--token-env",
        default="SUPERVISOR_A2A_TOKEN",
        help="Env var holding the bearer token; if unset, localhost-only",
    )
    p_a2a_serve.add_argument(
        "--rate-limit",
        type=int,
        default=20,
        help="Requests-per-minute per client IP (default 20)",
    )

    p_config = sub.add_parser("config", help="Read or write config values")
    config_sub = p_config.add_subparsers(dest="config_action")
    p_config_set = config_sub.add_parser("set", help="Set a config value")
    p_config_set.add_argument("--key", required=True, help="Config key name")
    p_config_set.add_argument("--value", required=True, help="Value to set")
    p_config_set.add_argument("--scope", choices=["global", "project"], default="global", help="Config scope")

    return parser


def _parse_runtime_argv(argv: list[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    legacy_run_actions = {
        "register",
        "foreground",
        "stop",
        "resume",
        "review",
        "export",
        "summarize",
        "replay",
        "postmortem",
    }
    if (
        len(argv) >= 2
        and argv[0] == "run"
        and argv[1] not in legacy_run_actions
        and not argv[1].startswith("-")
    ):
        legacy = argparse.ArgumentParser(prog="thin-supervisor run", add_help=False)
        legacy.add_argument("spec_path")
        legacy.add_argument("--pane", default=None)
        legacy.add_argument("--config", default=None)
        legacy.add_argument("--event-file", default=None)
        legacy.add_argument("--dry-run", action="store_true")
        legacy.add_argument("--daemon", "-d", action="store_true")
        parsed = legacy.parse_args(argv[1:])
        setattr(parsed, "command", "run")
        setattr(parsed, "run_action", None)
        return parsed
    parser = build_runtime_parser()
    return parser.parse_args(argv)


def main():
    parser = build_runtime_parser()
    args = _parse_runtime_argv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.command == "init":
        sys.exit(cmd_init(args))
    elif args.command == "deinit":
        sys.exit(cmd_deinit(args))
    elif args.command == "daemon":
        if args.daemon_action == "start":
            sys.exit(cmd_daemon_start(args))
        elif args.daemon_action == "stop":
            sys.exit(cmd_daemon_stop(args))
        else:
            print("Usage: thin-supervisor daemon {start|stop}")
            sys.exit(1)
    elif args.command == "run":
        if args.run_action == "register":
            sys.exit(cmd_run_register(args))
        elif args.run_action == "foreground":
            sys.exit(cmd_run_foreground(args))
        elif args.run_action == "stop":
            sys.exit(cmd_run_stop(args))
        elif args.run_action == "resume":
            sys.exit(cmd_run_resume(args))
        elif args.run_action == "review":
            sys.exit(cmd_run_review(args))
        elif args.run_action == "export":
            sys.exit(cmd_run_export(args))
        elif args.run_action == "summarize":
            sys.exit(cmd_run_summarize(args))
        elif args.run_action == "replay":
            sys.exit(cmd_run_replay(args))
        elif args.run_action == "postmortem":
            sys.exit(cmd_run_postmortem(args))
        elif args.spec_path:
            sys.exit(cmd_run_legacy(args))
        else:
            print("Usage: thin-supervisor run {register|foreground|stop|resume|review|export|summarize|replay|postmortem}")
            sys.exit(1)
    elif args.command == "stop":
        sys.exit(cmd_daemon_stop(args))
    elif args.command == "list":
        sys.exit(cmd_list(args))
    elif args.command == "ps":
        sys.exit(cmd_ps(args))
    elif args.command == "pane-owner":
        sys.exit(cmd_pane_owner(args))
    elif args.command == "observe":
        sys.exit(cmd_observe(args))
    elif args.command == "clarify":
        sys.exit(cmd_clarify(args))
    elif args.command == "note":
        if args.note_action in ("add", "list"):
            sys.exit(cmd_note(args))
        else:
            print("Usage: thin-supervisor note {add|list}")
            sys.exit(1)
    elif args.command == "review":
        if args.review_action == "request":
            sys.exit(cmd_review_request(args))
        elif args.review_action == "result":
            sys.exit(cmd_review_result(args))
        else:
            print("Usage: thin-supervisor review {request|result}")
            sys.exit(1)
    elif args.command == "mailbox":
        sys.exit(cmd_mailbox(args))
    elif args.command == "waits":
        sys.exit(cmd_waits(args))
    elif args.command == "skill":
        if args.skill_action == "install":
            sys.exit(cmd_skill_install(args))
        else:
            print("Usage: thin-supervisor skill {install}")
            sys.exit(1)
    elif args.command == "spec":
        sys.exit(cmd_spec(args))
    elif args.command == "session":
        if args.session_action in ("detect", "jsonl", "list"):
            sys.exit(cmd_session(args))
        else:
            print("Usage: thin-supervisor session {detect|jsonl|list}")
            sys.exit(1)
    elif args.command == "status":
        sys.exit(cmd_status(args))
    elif args.command == "overview":
        sys.exit(cmd_overview(args))
    elif args.command == "bridge":
        sys.exit(cmd_bridge(args))
    elif args.command == "bootstrap":
        sys.exit(cmd_bootstrap(args))
    elif args.command == "dashboard":
        sys.exit(cmd_dashboard(args))
    elif args.command == "tui":
        from supervisor.operator.tui import run_tui
        run_tui()
        sys.exit(0)
    elif args.command == "hook":
        sys.exit(cmd_hook(args))
    elif args.command == "a2a":
        sys.exit(cmd_a2a(args))
    elif args.command == "config":
        if args.config_action == "set":
            sys.exit(cmd_config_set(args))
        else:
            print("Usage: thin-supervisor config {set}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
