"""Single daemon that manages multiple concurrent supervisor runs.

Listens on a Unix domain socket for register/stop/status requests.
Each run executes in its own thread via run_sidecar().
"""
from __future__ import annotations

import json
import hashlib
import logging
import os
import signal
import socket
import sys
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from supervisor.domain.enums import DeliveryState, TopState
from supervisor.domain.models import SupervisorState
from supervisor.event_plane.ingest import EventPlaneIngest
from supervisor.event_plane.models import SessionMailboxItem
from supervisor.event_plane.store import EventPlaneStore
from supervisor.event_plane.surface import summarize_for_session
from supervisor.event_plane.wake_policy import evaluate as evaluate_wake
from supervisor.plan.loader import load_spec
from supervisor.storage.state_store import StateStore
from supervisor.gates.finish_gate import FinishGate
from supervisor.domain.state_machine import FINAL_STATES, transition_top_state
from supervisor.loop import SupervisorLoop
from supervisor.adapters.surface_factory import create_surface
from supervisor.config import RuntimeConfig
from supervisor.global_registry import (
    acquire_pane_lock,
    register_daemon,
    register_worktree,
    release_pane_lock,
    unregister_daemon,
    update_daemon,
)
from supervisor.interventions import AutoInterventionManager
from supervisor.notifications import NotificationManager
from supervisor.llm.explainer_client import ExplainerClient
from supervisor.operator.actions import build_explainer_context_from_state
from supervisor.operator.api import (
    recent_exchange,
    snapshot_from_state,
    timeline_from_session_log,
)
from supervisor.operator.models import RunEventPlaneSummary
from supervisor.operator.jobs import JobTracker
from supervisor.pause_summary import summarize_state
from supervisor.spec_approval import load_runnable_spec
from supervisor.storage.system_events import append_system_event

logger = logging.getLogger(__name__)

# Default paths — overridden by config in production
DEFAULT_SOCK_PATH = ".supervisor/daemon.sock"
DEFAULT_PID_PATH = ".supervisor/daemon.pid"
DEFAULT_RUNS_DIR = ".supervisor/runtime/runs"

MAX_REQUEST_SIZE = 64 * 1024  # 64KB max request
RECOVERABLE_ORPHANED_STATES = {
    TopState.RUNNING,
    TopState.GATING,
    TopState.VERIFYING,
    # A daemon crash mid-attach leaves the run in ATTACHED with no owner;
    # the operator can still resume it.  Symmetric treatment for
    # RECOVERY_NEEDED — the supervisor was mid auto-intervention and
    # the next loop iteration will re-run the recipe.
    TopState.ATTACHED,
    TopState.RECOVERY_NEEDED,
}


class RunEntry:
    """Registry entry for one active run."""

    def __init__(self, run_id: str, spec_path: str, pane_target: str,
                 workspace_root: str, surface_type: str, thread: threading.Thread, store: StateStore):
        self.run_id = run_id
        self.spec_path = spec_path
        self.pane_target = pane_target
        self.workspace_root = workspace_root
        self.surface_type = surface_type
        self.thread = thread
        self.store = store
        self.stop_event = threading.Event()

    def to_dict(self) -> dict:
        state = self._read_state()
        summary = summarize_state(state or {}) if state else {}
        return {
            "run_id": self.run_id,
            "session_id": state.get("session_id", "") if state else "",
            "spec_path": self.spec_path,
            "pane_target": self.pane_target,
            "alive": self.thread.is_alive() if self.thread else False,
            "top_state": state.get("top_state", "UNKNOWN") if state else "UNKNOWN",
            "current_node": state.get("current_node_id", "") if state else "",
            "status_reason": summary.get("status_reason", ""),
            "pause_reason": summary.get("pause_reason", ""),
            "next_action": summary.get("next_action", ""),
        }

    def _read_state(self) -> dict | None:
        try:
            return json.loads(self.store.state_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None


DEFAULT_IDLE_SHUTDOWN_SEC = 600  # 10 minutes
UNIX_SOCKET_MAX_PATH_BYTES = 107 if sys.platform.startswith("linux") else 103


def _resolve_runtime_path(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def _bindable_socket_path(path: str) -> str:
    resolved = _resolve_runtime_path(path)
    if len(resolved.encode("utf-8")) <= UNIX_SOCKET_MAX_PATH_BYTES:
        return resolved

    # AF_UNIX bind paths are short on macOS/BSD/Linux; keep the registry key
    # unique by deriving a deterministic hashed socket path under /tmp.
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
    fallback = f"/tmp/thin-supervisor-{digest}.sock"
    logger.warning(
        "socket path %s exceeds AF_UNIX limit; using fallback %s",
        resolved,
        fallback,
    )
    return fallback


class DaemonServer:
    """Single-process daemon managing multiple supervisor runs."""

    def __init__(self, config: RuntimeConfig | None = None, *,
                 sock_path: str = "", pid_path: str = "", runs_dir: str = "",
                 idle_shutdown_sec: int | None = None):
        self.config = config or RuntimeConfig()
        self.sock_path = _bindable_socket_path(sock_path or DEFAULT_SOCK_PATH)
        self.pid_path = _resolve_runtime_path(pid_path or DEFAULT_PID_PATH)
        self.runs_dir = _resolve_runtime_path(runs_dir or DEFAULT_RUNS_DIR)
        self.idle_shutdown_sec = idle_shutdown_sec if idle_shutdown_sec is not None else DEFAULT_IDLE_SHUTDOWN_SEC
        self._runs: dict[str, RunEntry] = {}
        self._lock = threading.Lock()
        self._shutdown = threading.Event()
        self._sock: socket.socket | None = None
        self._started_at = time.time()
        self._last_run_finished_at: float = 0
        self._last_client_contact_at: float = time.time()
        self._explainer = ExplainerClient(
            model=self.config.explainer_model,
            temperature=self.config.explainer_temperature,
            max_tokens=self.config.explainer_max_tokens,
            deep_model=self.config.deep_explainer_model,
            deep_temperature=self.config.deep_explainer_temperature,
            deep_max_tokens=self.config.deep_explainer_max_tokens,
        )
        self._job_tracker = JobTracker()
        # Command channels are per-credential-set singletons with
        # cross-process advisory locks.  Built here but started only after
        # the daemon is ready to accept IPC in start() — otherwise a
        # command arriving during startup would hit an unbound socket.
        from supervisor.operator.channel_host import OperatorChannelHost
        self._channel_host = OperatorChannelHost.from_config(self.config)
        # Event plane (Task 3): request/result/mailbox substrate lives
        # under runs_dir's parent/shared so it co-locates with sessions.jsonl.
        event_plane_root = Path(self.runs_dir).parent
        self._event_plane_store = EventPlaneStore(event_plane_root)
        self._event_plane_ingest = EventPlaneIngest(self._event_plane_store)
        # Bounded LRU of StateStore instances for reaped runs so concurrent
        # event-plane appends share the same seq lock/counter. Without this,
        # two threads hitting `_find_store_for_run` for the same reaped run
        # would create separate StateStores and race on seq allocation. The
        # cap prevents unbounded growth over long-running daemons that
        # process many runs.
        self._reaped_stores: OrderedDict[str, StateStore] = OrderedDict()
        self._reaped_stores_lock = threading.Lock()
        self._reaped_stores_max = 128

    def start(self) -> None:
        """Start the daemon: bind socket, write PID, accept connections."""
        Path(self.runs_dir).mkdir(parents=True, exist_ok=True)

        sock_path = Path(self.sock_path)
        sock_path.unlink(missing_ok=True)

        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(str(sock_path))
        self._sock.listen(5)
        self._sock.settimeout(1.0)

        Path(self.pid_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.pid_path).write_text(str(os.getpid()))
        register_daemon(self._daemon_metadata())
        register_worktree(os.getcwd())
        append_system_event(
            Path(self.runs_dir).parent,
            "daemon_started",
            {
                "pid": os.getpid(),
                "socket": self.sock_path,
                "cwd": os.getcwd(),
            },
        )
        self._channel_host.start()
        recovered = self._recover_orphaned_runs()
        if recovered:
            logger.warning("recovered %d orphaned persisted run(s) into PAUSED_FOR_HUMAN", recovered)

        try:
            signal.signal(signal.SIGTERM, self._handle_sigterm)
        except ValueError:
            pass  # not main thread

        logger.info("daemon started (PID %d, socket %s)", os.getpid(), self.sock_path)

        try:
            self._accept_loop()
        finally:
            self._cleanup()

    def _recover_orphaned_runs(self) -> int:
        runs_dir = Path(self.runs_dir)
        if not runs_dir.exists():
            return 0

        recovered = 0
        for run_dir in sorted(runs_dir.iterdir()):
            state_path = run_dir / "state.json"
            if not state_path.exists():
                continue
            try:
                state_data = json.loads(state_path.read_text())
                state = SupervisorState.from_dict(state_data)
            except (OSError, json.JSONDecodeError, KeyError, ValueError):
                continue

            if state.top_state not in RECOVERABLE_ORPHANED_STATES:
                continue
            if getattr(state, "controller_mode", "daemon") == "foreground":
                continue

            previous_top_state = state_data.get("top_state", state.top_state.value)
            delivery = state_data.get("delivery_state", DeliveryState.IDLE)
            store = StateStore(str(run_dir))
            store._session_seq = store._read_last_seq()
            store.transition_and_record(
                state,
                TopState.PAUSED_FOR_HUMAN,
                reason="daemon startup orphan recovery",
                source="daemon._recover_orphaned_runs",
            )
            if delivery in (DeliveryState.INJECTED, DeliveryState.SUBMITTED):
                recovery_detail = (
                    f"daemon restarted while delivery was in progress "
                    f"(delivery_state={delivery}, top_state={previous_top_state}); "
                    "instruction may not have been processed — explicit resume required"
                )
            elif delivery == DeliveryState.TIMED_OUT:
                recovery_detail = (
                    f"daemon restarted after delivery timeout "
                    f"(top_state={previous_top_state}); "
                    "check agent state before resuming"
                )
            else:
                recovery_detail = (
                    f"daemon restarted while the run was in progress "
                    f"({previous_top_state}); "
                    "explicit resume is required"
                )
            state.delivery_state = DeliveryState.IDLE
            state.human_escalations.append({
                "reason": recovery_detail,
                "orphaned_from": previous_top_state,
                "delivery_state_at_crash": delivery,
                "pause_class": "recovery",
            })
            store.append_session_event(
                state.run_id,
                "orphaned_run_recovered",
                {
                    "previous_top_state": previous_top_state,
                    "delivery_state_at_crash": delivery,
                    "reason": state.human_escalations[-1]["reason"],
                },
            )
            store.save(state)
            recovered += 1

        return recovered

    def _accept_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                self._reap_finished()
                self._refresh_idle_state()
                self._check_idle_shutdown()
                continue
            except OSError:
                break
            self._last_client_contact_at = time.time()
            try:
                conn.settimeout(10)
                self._handle_connection(conn)
            except Exception:
                logger.exception("error handling connection")
            finally:
                conn.close()

    def _handle_connection(self, conn: socket.socket) -> None:
        data = b""
        while len(data) < MAX_REQUEST_SIZE:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break

        if len(data) >= MAX_REQUEST_SIZE:
            self._send(conn, {"ok": False, "error": "request too large"})
            return

        try:
            request = json.loads(data.decode("utf-8").strip())
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send(conn, {"ok": False, "error": "invalid JSON"})
            return

        action = request.get("action", "")
        if action == "register":
            response = self._do_register(request)
        elif action == "status":
            response = self._do_status()
        elif action == "stop":
            response = self._do_stop(request.get("run_id", ""))
        elif action == "stop_all":
            response = self._do_stop_all()
        elif action == "resume":
            response = self._do_resume(request)
        elif action == "ack_review":
            response = self._do_ack_review(request)
        elif action == "list_runs":
            response = self._do_list_runs()
        elif action == "observe":
            response = self._do_observe(request.get("run_id", ""))
        elif action == "note_add":
            response = self._do_note_add(request)
        elif action == "note_list":
            response = self._do_note_list(request)
        elif action == "get_snapshot":
            response = self._do_get_snapshot(request.get("run_id", ""))
        elif action == "get_timeline":
            response = self._do_get_timeline(request)
        elif action == "get_exchange":
            response = self._do_get_exchange(request.get("run_id", ""))
        elif action == "explain_run":
            response = self._do_explain_run(request)
        elif action == "explain_exchange":
            response = self._do_explain_exchange(request)
        elif action == "assess_drift":
            response = self._do_assess_drift(request)
        elif action == "request_clarification":
            response = self._do_request_clarification(request)
        elif action == "escalate_clarification":
            response = self._do_escalate_clarification(request)
        elif action == "get_job":
            response = self._do_get_job(request.get("job_id", ""))
        elif action == "external_task_create":
            response = self._do_external_task_create(request)
        elif action == "external_result_ingest":
            response = self._do_external_result_ingest(request)
        elif action == "mailbox_list":
            response = self._do_mailbox_list(request)
        elif action == "mailbox_ack":
            response = self._do_mailbox_ack(request)
        elif action == "waits_list":
            response = self._do_waits_list(request)
        elif action == "ping":
            response = {"ok": True, "pong": True}
        else:
            response = {"ok": False, "error": f"unknown action: {action}"}

        self._send(conn, response)

    def _do_register(self, request: dict) -> dict:
        spec_path = request.get("spec_path", "")
        pane_target = request.get("pane_target", "")
        workspace_root = request.get("workspace_root", os.getcwd())

        if not spec_path or not pane_target:
            return {"ok": False, "error": "spec_path and pane_target required"}

        try:
            spec = load_runnable_spec(spec_path)
        except Exception as e:
            return {"ok": False, "error": f"spec load failed: {e}"}

        run_id = f"run_{uuid.uuid4().hex[:12]}"
        run_dir = str(Path(self.runs_dir) / run_id)
        store = StateStore(run_dir)
        surface_type = request.get("surface_type") or getattr(self.config, "surface_type", "tmux")
        state = store.load_or_init(
            spec, spec_path=spec_path, pane_target=pane_target,
            surface_type=surface_type,
            workspace_root=workspace_root,
            controller_mode="daemon",
        )

        if state.run_id != run_id:
            state.run_id = run_id
            store.save(state)

        entry = RunEntry(run_id, spec_path, pane_target, workspace_root,
                         surface_type=surface_type, thread=None, store=store)
        pane_owner = self._pane_owner_metadata(run_id, spec_path, pane_target, workspace_root)

        with self._lock:
            for existing in self._runs.values():
                if existing.pane_target == pane_target and existing.thread and existing.thread.is_alive():
                    return {"ok": False, "error": f"pane {pane_target} already has active run {existing.run_id}"}
            acquired, existing_owner = acquire_pane_lock(pane_target, pane_owner)
            if not acquired:
                return {
                    "ok": False,
                    "error": (
                        f"pane {pane_target} already owned by "
                        f"{existing_owner.get('run_id', '?')} in {existing_owner.get('cwd', '?')}"
                    ),
                }

            thread = threading.Thread(
                target=self._run_worker,
                args=(entry, spec, state),
                name=f"run-{run_id}",
                daemon=True,
            )
            entry.thread = thread
            self._runs[run_id] = entry
            self._update_daemon_record_locked()

        thread.start()
        logger.info("registered run %s: spec=%s pane=%s cwd=%s", run_id, spec_path, pane_target, workspace_root)
        return {"ok": True, "run_id": run_id}

    def _run_worker(self, entry: RunEntry, spec, state) -> None:
        """Worker thread: runs run_sidecar for one run."""
        try:
            terminal = create_surface(entry.surface_type, entry.pane_target)
            from supervisor.domain.models import WorkerProfile
            worker = WorkerProfile(
                provider=getattr(self.config, "worker_provider", "unknown"),
                model_name=getattr(self.config, "worker_model", ""),
                trust_level=getattr(self.config, "worker_trust_level", "standard"),
            )
            loop = SupervisorLoop(
                entry.store,
                judge_model=self.config.judge_model,
                judge_temperature=self.config.judge_temperature,
                judge_max_tokens=self.config.judge_max_tokens,
                worker_profile=worker,
                notification_manager=NotificationManager.from_config(
                    self.config,
                    runtime_root=entry.store.runtime_root,
                    command_channels=self._channel_host.channels,
                ),
                auto_intervention_manager=AutoInterventionManager(
                    mode=self.config.pause_handling_mode,
                    max_auto_interventions=self.config.max_auto_interventions,
                ),
                runtime_recovery_policy=self.config.runtime_recovery_policy(),
            )
            loop.run_sidecar(
                spec, state, terminal,
                poll_interval=self.config.poll_interval_sec,
                read_lines=self.config.read_lines,
                stop_event=entry.stop_event,
                idle_timeout_sec=self.config.default_agent_timeout_sec,
            )
            logger.info("run %s finished: %s", entry.run_id, state.top_state.value)
        except Exception:
            logger.exception("run %s crashed", entry.run_id)

    def _do_status(self) -> dict:
        with self._lock:
            runs = [e.to_dict() for e in self._runs.values()]
        for run in runs:
            run["event_plane"] = summarize_for_session(
                self._event_plane_store, run.get("session_id", "")
            )
        return {"ok": True, "runs": runs}

    def _do_stop(self, run_id: str) -> dict:
        with self._lock:
            entry = self._runs.get(run_id)
        if not entry:
            return {"ok": False, "error": f"run {run_id} not found"}
        entry.stop_event.set()
        # Non-blocking: don't wait in the IPC handler thread.
        # Reaper will clean up after thread exits.
        logger.info("stop signal sent to run %s", run_id)
        return {"ok": True}

    def _do_stop_all(self) -> dict:
        with self._lock:
            entries = list(self._runs.values())
        for entry in entries:
            entry.stop_event.set()
        return {"ok": True, "stopped": len(entries)}

    def _do_resume(self, request: dict) -> dict:
        """Resume a paused or crashed run by scanning existing run dirs."""
        spec_path = request.get("spec_path", "")
        pane_target = request.get("pane_target", "")

        if not spec_path or not pane_target:
            return {"ok": False, "error": "spec_path and pane_target required"}

        # Scan run dirs for matching state
        runs_dir = Path(self.runs_dir)
        if not runs_dir.exists():
            return {"ok": False, "error": "no runs directory"}

        # Load spec to get spec_id for matching
        try:
            target_spec = load_runnable_spec(spec_path)
        except Exception as e:
            return {"ok": False, "error": f"spec load failed: {e}"}

        for run_dir in sorted(runs_dir.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True):
            state_path = run_dir / "state.json"
            if not state_path.exists():
                continue
            try:
                state_data = json.loads(state_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            # Match by spec_id + pane_target (not just pane)
            if (state_data.get("spec_id") == target_spec.id
                    and state_data.get("pane_target", "") == pane_target
                    and state_data.get("top_state") in (
                        "PAUSED_FOR_HUMAN", "RUNNING", "READY",
                        "GATING", "VERIFYING",
                        # New states from the state-machine redesign: a run
                        # stranded mid-attach or mid-recovery is still the
                        # right target for resume.
                        "ATTACHED", "RECOVERY_NEEDED",
                    )):
                run_id = state_data["run_id"]
                resumable_state = state_data.get("top_state", "")
                if resumable_state in {"GATING", "VERIFYING"}:
                    return {
                        "ok": False,
                        "error": (
                            f"run {run_id} is in {resumable_state} and cannot safely resume. "
                            "Stop or observe the run before restarting it."
                        ),
                    }
                current_spec_hash = StateStore._hash_spec(spec_path)
                saved_spec_hash = state_data.get("spec_hash", "")
                if current_spec_hash and not saved_spec_hash:
                    return {
                        "ok": False,
                        "error": (
                            f"run {run_id} has no persisted spec hash. "
                            "Legacy runs must be re-registered instead of resumed."
                        ),
                    }
                if current_spec_hash and saved_spec_hash and current_spec_hash != saved_spec_hash:
                    return {
                        "ok": False,
                        "error": (
                            "spec was modified since the run was created. "
                            "Use register to start a new run, or revert the spec to resume."
                        ),
                    }

                store = StateStore(str(run_dir))
                surface_type = request.get("surface_type") or state_data.get("surface_type", "tmux")
                state = store.load_or_init(
                    target_spec, spec_path=spec_path, pane_target=pane_target,
                    surface_type=surface_type,
                    workspace_root=state_data.get("workspace_root", os.getcwd()),
                    controller_mode="daemon",
                )
                resumed_from = state.top_state.value
                entry = RunEntry(run_id, spec_path, pane_target,
                                 state_data.get("workspace_root", ""),
                                 surface_type=surface_type, thread=None, store=store)

                # Acquire pane lock before starting
                pane_owner = self._pane_owner_metadata(run_id, spec_path, pane_target,
                                                       state_data.get("workspace_root", ""))
                with self._lock:
                    if run_id in self._runs:
                        return {"ok": False, "error": f"run {run_id} is already active"}
                    for existing in self._runs.values():
                        if existing.pane_target == pane_target and existing.thread and existing.thread.is_alive():
                            return {"ok": False, "error": f"pane {pane_target} owned by run {existing.run_id}"}
                    acquired, existing_owner = acquire_pane_lock(pane_target, pane_owner)
                    if not acquired:
                        return {"ok": False, "error": f"pane {pane_target} locked by {existing_owner}"}
                    if state.top_state == TopState.PAUSED_FOR_HUMAN:
                        # Restore ATTACHED when the pause originated on the
                        # attach boundary (captured in `pre_pause_top_state`
                        # by `_pause_for_human`). Otherwise default to
                        # RUNNING. This preserves the first-execution-
                        # evidence gate across a resume: a run paused from
                        # ATTACHED (e.g. RE_INJECT cap exhausted) must still
                        # require real execution evidence on the next
                        # checkpoint, not slip into RUNNING and silently
                        # CONTINUE on admin-only evidence.
                        if state.pre_pause_top_state == TopState.ATTACHED.value:
                            store.transition_and_record(
                                state,
                                TopState.ATTACHED,
                                reason="resume requested (restoring attach boundary)",
                                source="daemon._do_resume",
                            )
                            # Re-arm the re-inject budget so the resumed run
                            # gets a fresh attempt at proving execution
                            # evidence, rather than immediately re-exhausting.
                            state.re_inject_count = 0
                        else:
                            store.transition_and_record(
                                state,
                                TopState.RUNNING,
                                reason="resume requested",
                                source="daemon._do_resume",
                            )
                        state.delivery_state = DeliveryState.IDLE
                        state.auto_intervention_count = 0
                        state.node_mismatch_count = 0
                        state.last_mismatch_node_id = ""
                        state.human_escalations = []
                        state.pre_pause_top_state = ""
                        store.append_session_event(
                            run_id,
                            "resume_requested",
                            {"resumed_from": resumed_from},
                        )
                        store.save(state)
                    elif state.top_state == TopState.RECOVERY_NEEDED:
                        # A persisted RECOVERY_NEEDED means the prior sidecar
                        # died between `_enter_recovery` and its follow-up
                        # transition. Do NOT silently flip to RUNNING here —
                        # the stalled auto-intervention can't be safely
                        # replayed without knowing what the recipe was
                        # supposed to inject. The sidecar's boot-time
                        # fail-safe in `_run_sidecar_inner` will surface this
                        # as a `rec.crash_during_recovery` pause for an
                        # operator, so we leave the state untouched and just
                        # record the resume attempt.
                        store.append_session_event(
                            run_id,
                            "resume_requested",
                            {"resumed_from": resumed_from},
                        )
                    thread = threading.Thread(
                        target=self._run_worker, args=(entry, target_spec, state),
                        name=f"run-{run_id}", daemon=True,
                    )
                    entry.thread = thread
                    self._runs[run_id] = entry
                    self._update_daemon_record_locked()
                thread.start()
                logger.info("resumed run %s from %s", run_id, resumed_from)
                return {"ok": True, "run_id": run_id, "resumed_from": resumed_from}

        return {"ok": False, "error": "no resumable run found for this spec + pane"}

    def _do_ack_review(self, request: dict) -> dict:
        """Record review acknowledgement and re-check finish gate."""
        run_id = request.get("run_id", "")
        reviewer = request.get("reviewer", "")
        if not run_id or not reviewer:
            return {"ok": False, "error": "run_id and reviewer required"}
        if reviewer not in {"human", "stronger_reviewer"}:
            return {"ok": False, "error": f"invalid reviewer: {reviewer}"}

        with self._lock:
            active_entry = self._runs.get(run_id)
            if active_entry and active_entry.thread and active_entry.thread.is_alive():
                return {
                    "ok": False,
                    "error": (
                        f"run {run_id} is currently active; "
                        "stop it or wait for it to pause before acknowledging review"
                    ),
                }
            if active_entry is not None:
                store = active_entry.store
                try:
                    state_data = json.loads(store.state_path.read_text())
                except (OSError, json.JSONDecodeError):
                    return {"ok": False, "error": f"could not read state for {run_id}"}
            else:
                store = None
                state_data = None
                for run_dir in sorted(Path(self.runs_dir).iterdir()):
                    state_path = run_dir / "state.json"
                    if not state_path.exists():
                        continue
                    try:
                        candidate = json.loads(state_path.read_text())
                    except (OSError, json.JSONDecodeError):
                        continue
                    if candidate.get("run_id") == run_id:
                        store = StateStore(str(run_dir))
                        state_data = candidate
                        break
                if store is None or state_data is None:
                    return {"ok": False, "error": f"run {run_id} not found"}

            spec_path = state_data.get("spec_path", "")
            if not spec_path:
                return {"ok": False, "error": f"run {run_id} has no spec_path"}
            current_spec_hash = StateStore._hash_spec(spec_path)
            saved_spec_hash = state_data.get("spec_hash", "")
            if current_spec_hash and saved_spec_hash and current_spec_hash != saved_spec_hash:
                return {
                    "ok": False,
                    "error": (
                        "spec was modified since the run was created. "
                        "Revert the spec to acknowledge the review for this run."
                    ),
                }
            try:
                spec = load_spec(spec_path)
            except Exception as e:
                return {"ok": False, "error": f"spec load failed: {e}"}

            state = store.load_or_init(
                spec,
                spec_path=spec_path,
                pane_target=state_data.get("pane_target", ""),
                surface_type=state_data.get("surface_type", "tmux"),
                workspace_root=state_data.get("workspace_root", os.getcwd()),
                controller_mode="daemon",
            )
            completed_reviews = set(getattr(state, "completed_reviews", []) or [])
            completed_reviews.add(reviewer)
            state.completed_reviews = sorted(completed_reviews)
            store.append_session_event(run_id, "review_acknowledged", {"reviewer": reviewer})

            finish = FinishGate().evaluate(spec, state, cwd=state.workspace_root)
            if finish["ok"]:
                if state.top_state not in FINAL_STATES:
                    store.transition_and_record(
                        state,
                        TopState.COMPLETED,
                        reason="review acknowledged and finish gate passed",
                        source="daemon._do_ack_review",
                    )
                    store.append_session_event(run_id, "completed_after_review", {"reviewer": reviewer})
            store.save(state)
            return {"ok": True, "run_id": run_id, "top_state": state.top_state.value}

    # ------------------------------------------------------------------
    # P0: list + observe
    # ------------------------------------------------------------------

    def _do_list_runs(self) -> dict:
        """List all active runs with detailed state."""
        with self._lock:
            runs = []
            for e in self._runs.values():
                state = e._read_state() or {}
                summary = summarize_state(state)
                runs.append({
                    "run_id": e.run_id,
                    "session_id": state.get("session_id", ""),
                    "spec_id": state.get("spec_id", ""),
                    "spec_path": e.spec_path,
                    "pane_target": e.pane_target,
                    "workspace": e.workspace_root,
                    "alive": e.thread.is_alive() if e.thread else False,
                    "top_state": state.get("top_state", "UNKNOWN"),
                    "current_node": state.get("current_node_id", ""),
                    "done_nodes": state.get("done_node_ids", []),
                    "current_attempt": state.get("current_attempt", 0),
                    "pause_reason": summary.get("pause_reason", ""),
                    "next_action": summary.get("next_action", ""),
                    "event_plane": summarize_for_session(
                        self._event_plane_store, state.get("session_id", "")
                    ),
                })
        return {"ok": True, "runs": runs}

    def _do_observe(self, run_id: str) -> dict:
        """Read-only observation of a specific run's state + recent events."""
        with self._lock:
            entry = self._runs.get(run_id)
        if not entry:
            return {"ok": False, "error": f"run {run_id} not found"}

        state = entry._read_state() or {}
        recent: list[dict] = []
        try:
            if entry.store.session_log_path.exists():
                lines = entry.store.session_log_path.read_text().strip().splitlines()[-5:]
                for line in lines:
                    try:
                        recent.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass

        return {
            "ok": True,
            "run_id": run_id,
            "state": state,
            "recent_events": recent,
            "event_plane": summarize_for_session(
                self._event_plane_store, state.get("session_id", "")
            ),
        }

    # ------------------------------------------------------------------
    # Operator channel: canonical read APIs
    # ------------------------------------------------------------------

    def _resolve_run_store(self, run_id: str) -> tuple[dict | None, Path | None]:
        """Resolve state dict and session_log path for a run.

        Checks active runs first, falls back to on-disk state for
        completed/reaped runs so operator reads work after reaping.
        """
        with self._lock:
            entry = self._runs.get(run_id)
        if entry:
            return entry._read_state() or {}, entry.store.session_log_path

        # Fallback: scan on-disk run directories
        run_dir = Path(self.runs_dir) / run_id
        state_path = run_dir / "state.json"
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                return state, run_dir / "session_log.jsonl"
            except (json.JSONDecodeError, OSError):
                pass
        return None, None

    def _do_get_snapshot(self, run_id: str) -> dict:
        """Return a RunSnapshot for a single run."""
        state, session_log = self._resolve_run_store(run_id)
        if state is None:
            return {"ok": False, "error": f"run {run_id} not found"}
        event_plane = self._event_plane_summary_for_state(state)
        snap = snapshot_from_state(state, session_log, event_plane=event_plane)
        return {"ok": True, **snap.to_dict()}

    def _event_plane_summary_for_state(self, state: dict) -> "RunEventPlaneSummary | None":
        """Build a typed RunEventPlaneSummary from this daemon's event
        plane store for the session on *state*.  Returns None when the
        state has no session_id (older runs pre Task 3).
        """
        session_id = state.get("session_id", "") if state else ""
        if not session_id:
            return None
        ep = summarize_for_session(self._event_plane_store, session_id)
        return RunEventPlaneSummary(
            waits_open=int(ep.get("waits_open", 0)),
            mailbox_new=int(ep.get("mailbox_new", 0)),
            mailbox_acknowledged=int(ep.get("mailbox_acknowledged", 0)),
            requests_total=int(ep.get("requests_total", 0)),
            latest_mailbox_item_id=str(ep.get("latest_mailbox_item_id", "") or ""),
            latest_wake_decision=str(ep.get("latest_wake_decision", "") or ""),
        )

    def _do_get_timeline(self, request: dict) -> dict:
        """Return RunTimelineEvents for a run."""
        run_id = request.get("run_id", "")
        state, session_log = self._resolve_run_store(run_id)
        if state is None:
            return {"ok": False, "error": f"run {run_id} not found"}
        limit = request.get("limit", 20)
        since_seq = request.get("since_seq", 0)
        events = timeline_from_session_log(
            session_log,
            limit=limit,
            since_seq=since_seq,
        )
        return {"ok": True, "run_id": run_id, "events": [e.to_dict() for e in events]}

    def _do_get_exchange(self, run_id: str) -> dict:
        """Return recent exchange summary for a run."""
        state, session_log = self._resolve_run_store(run_id)
        if state is None:
            return {"ok": False, "error": f"run {run_id} not found"}
        exchange = recent_exchange(state, session_log)
        return {"ok": True, **exchange}

    # ------------------------------------------------------------------
    # Operator channel: async explainer jobs
    # ------------------------------------------------------------------

    def _build_explainer_context(self, run_id: str, **extra) -> dict:
        """Build context dict for explainer calls.

        Delegates to the shared ``build_explainer_context_from_state``
        after resolving state via ``_resolve_run_store``.
        """
        state, session_log = self._resolve_run_store(run_id)
        return build_explainer_context_from_state(
            state or {}, session_log, **extra,
        )

    def _do_explain_run(self, request: dict) -> dict:
        run_id = request.get("run_id", "")
        language = request.get("language", "en")
        state, _ = self._resolve_run_store(run_id)
        if state is None:
            return {"ok": False, "error": f"run {run_id} not found"}

        def _job():
            ctx = self._build_explainer_context(run_id, language=language)
            return self._explainer.explain_run(ctx)

        job_id = self._job_tracker.submit("explain_run", _job)
        return {"ok": True, "job_id": job_id}

    def _do_explain_exchange(self, request: dict) -> dict:
        run_id = request.get("run_id", "")
        language = request.get("language", "en")
        state, _ = self._resolve_run_store(run_id)
        if state is None:
            return {"ok": False, "error": f"run {run_id} not found"}

        def _job():
            ctx = self._build_explainer_context(run_id, language=language)
            state2, session_log2 = self._resolve_run_store(run_id)
            exchange = recent_exchange(state2 or {}, session_log2)
            ctx["exchange"] = exchange
            return self._explainer.explain_exchange(ctx)

        job_id = self._job_tracker.submit("explain_exchange", _job)
        return {"ok": True, "job_id": job_id}

    def _do_assess_drift(self, request: dict) -> dict:
        run_id = request.get("run_id", "")
        language = request.get("language", "en")
        state, _ = self._resolve_run_store(run_id)
        if state is None:
            return {"ok": False, "error": f"run {run_id} not found"}

        def _job():
            ctx = self._build_explainer_context(run_id, language=language)
            return self._explainer.assess_drift(ctx)

        job_id = self._job_tracker.submit("assess_drift", _job)
        return {"ok": True, "job_id": job_id}

    def _do_request_clarification(self, request: dict) -> dict:
        run_id = request.get("run_id", "")
        question = request.get("question", "")
        language = request.get("language", "en")
        state, session_log = self._resolve_run_store(run_id)
        if state is None:
            return {"ok": False, "error": f"run {run_id} not found"}
        if not question:
            return {"ok": False, "error": "question is required"}

        # Resolve store for active runs (thread-safe event writing)
        with self._lock:
            entry = self._runs.get(run_id)
        store = entry.store if entry else None

        def _write_event(event_type: str, payload: dict) -> None:
            if store:
                store.append_session_event(run_id, event_type, payload)
            elif session_log:
                from supervisor.operator.api import append_timeline_event
                append_timeline_event(session_log, run_id, event_type, payload)

        escalation_threshold = self.config.clarification_escalation_confidence

        def _job():
            from supervisor.operator.clarification import finalize_clarification

            _write_event("clarification_request", {
                "question": question, "language": language,
            })

            ctx = self._build_explainer_context(run_id, language=language)
            ctx["question"] = question
            result = self._explainer.request_clarification(ctx)

            return finalize_clarification(
                result,
                question=question,
                escalation_threshold=escalation_threshold,
                write_event=_write_event,
            )

        job_id = self._job_tracker.submit("clarification", _job)
        return {"ok": True, "job_id": job_id}

    def _do_escalate_clarification(self, request: dict) -> dict:
        """Record an operator's decision to escalate a clarification to the worker.

        Audit-only in 0.3.7 — actual transport of the question to the worker
        (via a side-instruction queue drained by SupervisorLoop) and
        worker-reply capture ship in 0.3.8. The event written here is the
        durable record that an operator chose to escalate.
        """
        run_id = request.get("run_id", "")
        question = request.get("question", "")
        language = request.get("language", "en")
        reason = request.get("reason", "operator_initiated")
        operator = request.get("operator", "")
        confidence = request.get("confidence")

        state, session_log = self._resolve_run_store(run_id)
        if state is None:
            return {"ok": False, "error": f"run {run_id} not found"}
        if not question:
            return {"ok": False, "error": "question is required"}

        escalation_id = uuid.uuid4().hex[:16]
        payload = {
            "escalation_id": escalation_id,
            "question": question,
            "language": language,
            "reason": reason,
            "operator": operator,
            "confidence": confidence,
            "transport": "pending_0_3_8",
        }

        with self._lock:
            entry = self._runs.get(run_id)
        store = entry.store if entry else None
        if store:
            store.append_session_event(
                run_id, "clarification_escalated_to_worker", payload,
            )
        elif session_log:
            from supervisor.operator.api import append_timeline_event
            append_timeline_event(
                session_log, run_id, "clarification_escalated_to_worker", payload,
            )
        return {"ok": True, "escalation_id": escalation_id}

    def _do_get_job(self, job_id: str) -> dict:
        job = self._job_tracker.get(job_id)
        if not job:
            return {"ok": False, "error": f"job {job_id} not found"}
        return {"ok": True, **job.to_dict()}

    # ------------------------------------------------------------------
    # P1: shared notes
    # ------------------------------------------------------------------

    def _shared_notes_path(self) -> Path:
        p = Path(self.runs_dir).parent / "shared"
        p.mkdir(parents=True, exist_ok=True)
        return p / "notes.jsonl"

    def _do_note_add(self, request: dict) -> dict:
        """Add a shared note."""
        content = request.get("content", "")
        if not content:
            return {"ok": False, "error": "content required"}
        raw_metadata = request.get("metadata", {})
        if raw_metadata is None:
            metadata = {}
        elif isinstance(raw_metadata, dict):
            metadata = raw_metadata
        else:
            return {"ok": False, "error": "metadata must be an object"}

        note = {
            "note_id": f"note_{uuid.uuid4().hex[:12]}",
            "timestamp": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
            "author_run_id": request.get("author_run_id", "human"),
            "target_run_id": request.get("target_run_id", ""),
            "note_type": request.get("note_type", "context"),
            "title": request.get("title", content[:80]),
            "content": content,
            "metadata": metadata,
        }

        path = self._shared_notes_path()
        with self._lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(note, ensure_ascii=False) + "\n")

        logger.info("note added: %s by %s", note["note_id"], note["author_run_id"])
        return {"ok": True, "note_id": note["note_id"]}

    def _do_note_list(self, request: dict) -> dict:
        """List shared notes, optionally filtered by type or run."""
        path = self._shared_notes_path()
        if not path.exists():
            return {"ok": True, "notes": []}

        filter_type = request.get("note_type")
        filter_run = request.get("run_id")
        filter_target = request.get("target_run_id")
        limit = request.get("limit", 20)

        notes: list[dict] = []
        try:
            for line in path.read_text().strip().splitlines():
                try:
                    note = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if filter_type and note.get("note_type") != filter_type:
                    continue
                if filter_run and note.get("author_run_id") != filter_run:
                    continue
                if filter_target and note.get("target_run_id") != filter_target:
                    continue
                notes.append(note)
        except OSError:
            pass

        # Return most recent first, up to limit
        return {"ok": True, "notes": notes[-limit:][::-1]}

    # ------------------------------------------------------------------
    # Event plane (Task 3): external task request / result / mailbox
    # ------------------------------------------------------------------

    def _find_store_for_run(self, run_id: str) -> StateStore | None:
        if not run_id:
            return None
        with self._lock:
            entry = self._runs.get(run_id)
        if entry is not None:
            return entry.store
        run_dir = Path(self.runs_dir) / run_id
        if not (run_dir / "state.json").exists():
            return None
        # Reuse a cached StateStore per run_id so concurrent callers share
        # the seq lock/counter. This closes a race where two threads would
        # each build their own StateStore, read the same last seq, and both
        # emit seq=N+1 for different events.
        with self._reaped_stores_lock:
            cached = self._reaped_stores.get(run_id)
            if cached is not None:
                self._reaped_stores.move_to_end(run_id)
                return cached
            store = StateStore(str(run_dir))
            # Freshly-instantiated StateStore starts at seq=0; align with the
            # run's existing session_log so events appended via this path
            # don't collide with earlier seqs written while the run was live.
            store._session_seq = store._read_last_seq()
            self._reaped_stores[run_id] = store
            while len(self._reaped_stores) > self._reaped_stores_max:
                self._reaped_stores.popitem(last=False)
            return store

    def _append_run_session_event(self, run_id: str, event_type: str, payload: dict) -> None:
        store = self._find_store_for_run(run_id)
        if store is None:
            return
        try:
            store.append_session_event(run_id, event_type, payload)
        except OSError:
            logger.warning("failed to append session event %s for %s", event_type, run_id)

    def _do_external_task_create(self, request: dict) -> dict:
        resp = self._event_plane_ingest.register_request(
            session_id=request.get("session_id", ""),
            provider=request.get("provider", ""),
            target_ref=request.get("target_ref", ""),
            run_id=request.get("run_id") or None,
            phase=request.get("phase", "execute"),
            task_kind=request.get("task_kind", "review"),
            blocking_policy=request.get("blocking_policy", "notify_only"),
            deadline_at=request.get("deadline_at", ""),
            resume_policy=request.get("resume_policy", ""),
        )
        if resp.get("ok"):
            run_id = request.get("run_id") or ""
            if run_id:
                self._append_run_session_event(
                    run_id,
                    "external_task_requested",
                    {
                        "request_id": resp["request_id"],
                        "wait_id": resp["wait_id"],
                        "session_id": resp["session_id"],
                        "provider": request.get("provider", ""),
                        "target_ref": request.get("target_ref", ""),
                        "task_kind": request.get("task_kind", "review"),
                    },
                )
        return resp

    def _do_external_result_ingest(self, request: dict) -> dict:
        resp = self._event_plane_ingest.ingest_result(
            request_id=request.get("request_id", ""),
            provider=request.get("provider", ""),
            result_kind=request.get("result_kind", ""),
            summary=request.get("summary", ""),
            payload=request.get("payload") or {},
            run_id=request.get("run_id") if request.get("run_id") is not None else None,
            idempotency_key=request.get("idempotency_key", ""),
        )
        if resp.get("ok") and not resp.get("deduped"):
            # Best-effort session event. If run_id is known on either side we
            # append to that run's session_log; correlation to session is
            # handled durably in the event plane itself.
            req_run_id = request.get("run_id") or ""
            req = self._event_plane_store.latest_request(request.get("request_id", ""))
            if not req_run_id and req is not None and req.run_id:
                req_run_id = req.run_id
            if req_run_id:
                self._append_run_session_event(
                    req_run_id,
                    "external_result_ingested",
                    {
                        "request_id": request.get("request_id", ""),
                        "result_id": resp.get("result_id", ""),
                        "mailbox_item_id": resp.get("mailbox_item_id", ""),
                        "session_id": resp.get("session_id", ""),
                        "provider": request.get("provider", ""),
                        "result_kind": request.get("result_kind", ""),
                    },
                )

            # Apply wake policy to the just-landed mailbox item. This is the
            # only legitimate place the decision gets made (Rule 4).
            decision = self._apply_wake_policy(
                request_id=request.get("request_id", ""),
                mailbox_item_id=resp.get("mailbox_item_id", ""),
                run_id_hint=req_run_id,
            )
            if decision:
                resp["wake_decision"] = decision["decision"]
                resp["wake_reason"] = decision.get("reason", "")
        return resp

    def _apply_wake_policy(
        self,
        *,
        request_id: str,
        mailbox_item_id: str,
        run_id_hint: str,
    ) -> dict | None:
        """Run wake policy on a freshly-created mailbox item.

        Persists the decision onto the mailbox item (append-only, latest
        wins) and emits a ``wake_decision_applied`` session event when a
        run is known. Returns the decision dict (or None on error).
        """
        req = self._event_plane_store.latest_request(request_id)
        if req is None:
            return None
        item = self._event_plane_store.latest_mailbox_item(mailbox_item_id)
        if item is None:
            return None

        run_state = None
        run_id = run_id_hint or req.run_id or ""
        if run_id:
            store = self._find_store_for_run(run_id)
            if store is not None:
                try:
                    data = json.loads(store.state_path.read_text())
                    run_state = {"top_state": data.get("top_state", "")}
                except (OSError, ValueError):
                    run_state = None

        decision = evaluate_wake(request=req, mailbox_item=item, run_state=run_state)

        updated = SessionMailboxItem.from_dict(item.to_dict())
        updated.wake_decision = decision.decision
        self._event_plane_store.append_mailbox_item(updated)

        # v1 scope: the wake_policy decision is durably recorded on the
        # mailbox item and in the session_log, and the mailbox item is the
        # operator-observable notification. Auto-resume for ``wake_worker``
        # (via `_do_resume`) is intentionally deferred because the resume
        # path needs spec_path / pane_target / surface_type that the event-
        # plane path does not currently carry. Log a clear warning so the
        # gap is visible rather than silent.
        if decision.decision == "wake_worker":
            logger.warning(
                "wake_policy decided wake_worker for run=%s session=%s mailbox_item=%s; "
                "auto-resume not yet wired — operator must resume explicitly",
                run_id or "?",
                req.session_id,
                mailbox_item_id,
            )
        # ``defer`` means the run is busy (RUNNING/GATING/VERIFYING/ATTACHED)
        # when the mailbox item arrives.  v1 only evaluates wake policy
        # once at ingest time — no state-transition callback re-runs this
        # when the run later pauses, so a deferred decision is effectively
        # stranded on the mailbox item until an operator acts.  Match the
        # ``wake_worker`` warning so the v1 scope limitation is visible
        # in logs rather than silent.  (The mailbox item still surfaces
        # through ``overview`` / ``status`` via ``mailbox_new``, so the
        # operator is not blind to it — just not auto-woken.)
        if decision.decision == "defer":
            logger.warning(
                "wake_policy deferred wake for run=%s session=%s mailbox_item=%s "
                "(%s); v1 does not re-evaluate on later state changes — "
                "operator must resume explicitly",
                run_id or "?",
                req.session_id,
                mailbox_item_id,
                decision.reason,
            )

        if run_id:
            self._append_run_session_event(
                run_id,
                "wake_decision_applied",
                {
                    "mailbox_item_id": mailbox_item_id,
                    "request_id": request_id,
                    "decision": decision.decision,
                    "reason": decision.reason,
                    "session_id": req.session_id,
                },
            )

        # Always promote the wake decision to the shared system log —
        # ``run_id`` can be empty when the request's originating run has
        # already ended, but the operator still needs to see the
        # decision at the system level.
        append_system_event(
            Path(self.runs_dir).parent,
            "wake_decision_applied",
            {
                "mailbox_item_id": mailbox_item_id,
                "request_id": request_id,
                "decision": decision.decision,
                "reason": decision.reason,
                "session_id": req.session_id,
                "run_id": run_id,
            },
        )

        return {"decision": decision.decision, "reason": decision.reason}

    def _do_mailbox_list(self, request: dict) -> dict:
        return self._event_plane_ingest.list_mailbox(
            session_id=request.get("session_id", ""),
            delivery_status=request.get("delivery_status", ""),
        )

    def _do_mailbox_ack(self, request: dict) -> dict:
        return self._event_plane_ingest.ack_mailbox_item(
            mailbox_item_id=request.get("mailbox_item_id", ""),
            delivery_status=request.get("delivery_status", "acknowledged"),
        )

    def _do_waits_list(self, request: dict) -> dict:
        return self._event_plane_ingest.list_waits(
            session_id=request.get("session_id", ""),
        )

    def _reap_finished(self) -> None:
        """Remove completed/stopped runs from registry.

        Two-phase: collect candidates under lock, then join outside lock
        to avoid blocking IPC while waiting for threads.
        """
        # Phase 1: identify candidates (under lock, fast)
        with self._lock:
            candidates = [
                (rid, e) for rid, e in self._runs.items()
                if not e.thread.is_alive() or e.stop_event.is_set()
            ]

        if not candidates:
            return

        # Phase 2: join threads outside lock (may block briefly)
        reaped = []
        for rid, e in candidates:
            if e.thread.is_alive():
                e.thread.join(timeout=2)
                if e.thread.is_alive():
                    continue  # still alive — skip, don't create zombie
            release_pane_lock(e.pane_target, e.run_id)
            reaped.append(rid)

        # Phase 3: remove from registry (under lock)
        if reaped:
            self._last_run_finished_at = time.time()
            with self._lock:
                for rid in reaped:
                    self._runs.pop(rid, None)
                self._update_daemon_record_locked()
            logger.info("reaped %d finished run(s)", len(reaped))
            # Phase 4: close sessions attached to runs that finished in a
            # terminal state and have no remaining active run. Without this,
            # find_session_by_attachment would forever return the same
            # session for a given (workspace_root, spec_id), and every
            # future run would inherit stale mailbox / waits.
            for rid in reaped:
                self._maybe_close_session_for(rid)

    def _maybe_close_session_for(self, run_id: str) -> None:
        """Close the Session attached to a reaped run if safe.

        A session is safe to close when:
        - the reaped run finished in a terminal top_state
          (COMPLETED / FAILED / ABORTED), and
        - no other run currently in the in-memory registry shares the
          same session_id (so we don't close a session that still has
          an active run).
        """
        store = self._find_store_for_run(run_id)
        if store is None:
            return
        try:
            data = json.loads(store.state_path.read_text())
        except (OSError, ValueError):
            return
        top_state = data.get("top_state", "")
        if top_state not in {"COMPLETED", "FAILED", "ABORTED"}:
            return
        session_id = data.get("session_id", "")
        if not session_id:
            return
        # Snapshot active stores under the lock, then do disk reads outside
        # of it — the per-entry state.json reads can't be held under
        # self._lock because that lock guards every IPC handler.
        with self._lock:
            active_stores = [entry.store for entry in self._runs.values()]
        for entry_store in active_stores:
            try:
                entry_state = json.loads(entry_store.state_path.read_text())
            except (OSError, ValueError):
                continue
            if entry_state.get("session_id") == session_id:
                return  # another run still holds this session
        try:
            store.close_session(session_id)
        except OSError:
            logger.warning("failed to close session %s on reap of %s", session_id, run_id)

    def _refresh_idle_state(self) -> None:
        """Update registry with current idle duration (called every ~1s from accept loop)."""
        with self._lock:
            active = len(self._runs)
        if active > 0:
            return
        now = time.time()
        last_activity = max(self._last_run_finished_at, self._last_client_contact_at, self._started_at)
        update_daemon(
            self.sock_path,
            state="idle",
            active_runs=0,
            idle_for_sec=int(now - last_activity),
        )

    def _check_idle_shutdown(self) -> None:
        """Auto-shutdown if idle for too long with zero active runs."""
        if self.idle_shutdown_sec <= 0:
            return
        with self._lock:
            active = len(self._runs)
        if active > 0:
            return
        now = time.time()
        last_activity = max(self._last_run_finished_at, self._last_client_contact_at, self._started_at)
        idle_for = now - last_activity
        if idle_for >= self.idle_shutdown_sec:
            logger.info("idle shutdown: no active runs for %.0fs (threshold=%ds)", idle_for, self.idle_shutdown_sec)
            update_daemon(self.sock_path, state="shutting_down", idle_for_sec=int(idle_for))
            self._shutdown.set()

    def _handle_sigterm(self, signum, frame):
        logger.info("SIGTERM received, shutting down")
        self._shutdown.set()

    def _cleanup(self) -> None:
        self._do_stop_all()
        # Wait for threads to finish (with timeout)
        with self._lock:
            entries = list(self._runs.values())
            threads = [e.thread for e in entries if e.thread]
        for t in threads:
            t.join(timeout=5)
        for entry in entries:
            release_pane_lock(entry.pane_target, entry.run_id)
        self._channel_host.stop()
        if self._sock:
            self._sock.close()
        Path(self.sock_path).unlink(missing_ok=True)
        Path(self.pid_path).unlink(missing_ok=True)
        unregister_daemon(self.sock_path)
        append_system_event(
            Path(self.runs_dir).parent,
            "daemon_stopped",
            {
                "pid": os.getpid(),
                "socket": self.sock_path,
                "cwd": os.getcwd(),
            },
        )
        logger.info("daemon stopped")

    @staticmethod
    def _send(conn: socket.socket, data: dict) -> None:
        conn.sendall((json.dumps(data) + "\n").encode("utf-8"))

    def _daemon_metadata(self) -> dict:
        from datetime import datetime, timezone
        return {
            "pid": os.getpid(),
            "cwd": os.getcwd(),
            "socket": self.sock_path,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "active_runs": len(self._runs),
            "state": "active" if self._runs else "idle",
            "idle_shutdown_sec": self.idle_shutdown_sec,
        }

    def _pane_owner_metadata(self, run_id: str, spec_path: str,
                             pane_target: str, workspace_root: str) -> dict:
        return {
            "pid": os.getpid(),
            "cwd": workspace_root or os.getcwd(),
            "socket": self.sock_path,
            "run_id": run_id,
            "pane_target": pane_target,
            "spec_path": spec_path,
            "controller_mode": "daemon",
        }

    def _update_daemon_record_locked(self) -> None:
        now = time.time()
        last_activity = max(self._last_run_finished_at, self._last_client_contact_at, self._started_at)
        update_daemon(
            self.sock_path,
            pid=os.getpid(),
            cwd=os.getcwd(),
            active_runs=len(self._runs),
            state="active" if self._runs else "idle",
            idle_for_sec=int(now - last_activity) if not self._runs else 0,
        )
