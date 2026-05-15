from __future__ import annotations

import logging
import signal
import time
from datetime import datetime, timezone

from supervisor.domain.enums import DeliveryState, TopState, DecisionType
from supervisor.domain.models import (
    Checkpoint, SupervisorDecision, HandoffInstruction,
    WorkerProfile, SupervisionPolicy, RoutingDecision, AcceptanceContract,
)
from supervisor.domain.state_machine import FINAL_STATES, transition_top_state
from supervisor.gates.continue_gate import ContinueGate
from supervisor.gates.branch_gate import BranchGate
from supervisor.gates.finish_gate import FinishGate
from supervisor.gates.contradictions import detect_contradiction
from supervisor.gates.escalation import classify_for_escalation, escalation_decision
from supervisor.gates.rules import is_admin_only_evidence
from supervisor.protocol.normalizer import normalize_checkpoint
from supervisor.llm.judge_client import JudgeClient
from supervisor.verifiers.suite import VerifierSuite
from supervisor.adapters.transcript_adapter import TranscriptAdapter
from supervisor.instructions.composer import InstructionComposer
from supervisor.gates.supervision_policy import SupervisionPolicyEngine
from supervisor.history import latest_oracle_consultation_id_for_run
from supervisor.interventions import AutoInterventionManager
from supervisor.notifications import NotificationEvent, NotificationManager
from supervisor.pause_summary import PAUSE_CLASSES, latest_human_escalation, summarize_state
from supervisor.runtime_recovery import (
    RuntimeRecoveryObservation,
    RuntimeRecoveryPolicy,
    detect_runtime_recovery,
    seconds_until_recovery,
)
from supervisor.progress import write_progress
from supervisor.protocol.reason_code import (
    ESC_AUTHORIZATION_REQUIRED,
    ESC_BLOCKED_GENUINE,
    ESC_MISSING_EXTERNAL_INPUT,
    ESC_REVIEW_REQUIRED,
    REC_CRASH_DURING_RECOVERY,
    REC_DELIVERY_TIMEOUT,
    REC_IDLE_TIMEOUT,
    REC_INJECT_FAILED,
    REC_NODE_MISMATCH_PERSISTED,
    REC_REINJECTION_EXHAUSTED,
    REC_RETRY_BUDGET_EXHAUSTED,
    REC_VERIFICATION_RETRY_EXHAUSTED,
)

logger = logging.getLogger(__name__)

MIN_POLL_SLEEP_SEC = 0.01
ZERO_POLL_IDLE_TIMEOUT_SEC = 0.5

# How long to wait for a Stop-hook ACK on an observation-only (JSONL) surface
# before falling back to the pause-for-human path. The hook only fires when the
# agent chooses to stop, so this window needs to tolerate a long-running turn.
OBSERVATION_HOOK_ACK_TIMEOUT_SEC = 600  # 10 minutes
OBSERVATION_HOOK_POLL_INTERVAL_SEC = 1.0


def build_context(spec, state) -> dict:
    return {
        "spec_id": spec.id,
        "current_node_id": state.current_node_id,
        "top_state": state.top_state.value,
        "last_agent_question": state.last_event.get("payload", {}).get("question", ""),
        "last_agent_checkpoint": state.last_agent_checkpoint,
        "done_node_ids": state.done_node_ids,
        "retry_budget": {
            "per_node": state.retry_budget.per_node,
            "global_limit": state.retry_budget.global_limit,
            "used_global": state.retry_budget.used_global,
        },
    }


# Upper bound on attach-boundary re-injections before falling through to a
# recovery pause. If the agent keeps emitting admin-only checkpoints after
# this many re-injects, the supervisor stops retrying and surfaces a recovery
# pause so an operator can look at the pane.
MAX_RE_INJECTS = 3


class SupervisorLoop:
    def __init__(self, store, judge_model: str | None = None,
                 judge_temperature: float = 0.1, judge_max_tokens: int = 512,
                 worker_profile: WorkerProfile | None = None,
                 notification_manager: NotificationManager | None = None,
                 auto_intervention_manager: AutoInterventionManager | None = None,
                 runtime_recovery_policy: RuntimeRecoveryPolicy | None = None):
        self.store = store
        self.judge_client = JudgeClient(
            model=judge_model,
            temperature=judge_temperature,
            max_tokens=judge_max_tokens,
        )
        self.continue_gate = ContinueGate(self.judge_client)
        self.branch_gate = BranchGate(self.judge_client)
        self.finish_gate = FinishGate()
        self.verifier_suite = VerifierSuite()
        self.composer = InstructionComposer()
        self.policy_engine = SupervisionPolicyEngine()
        self.worker_profile = worker_profile or WorkerProfile()
        self.notification_manager = notification_manager or NotificationManager()
        self.auto_intervention_manager = auto_intervention_manager or AutoInterventionManager(mode="notify_only")
        self.runtime_recovery_policy = runtime_recovery_policy or RuntimeRecoveryPolicy()
        # Set while a sidecar loop is active; consulted by helpers that need
        # to cooperate with daemon stop_event / SIGTERM.
        self._interrupted_ref = None

    def handle_event(self, state, event):
        state.last_event = event
        # States where an arriving checkpoint must NOT force a transition to
        # GATING. RECOVERY_NEEDED belongs here: its allowed transitions are
        # only RUNNING / PAUSED_FOR_HUMAN / terminals (see
        # `supervisor/domain/state_machine.py`), so re-entering GATING would
        # raise InvalidTopStateTransition. In practice a run only lands in
        # RECOVERY_NEEDED persistently if the sidecar crashed between
        # `_enter_recovery()` and its follow-up transition — without this
        # guard a resume in that condition becomes a permanent crash loop.
        preserve_state = {
            TopState.ATTACHED,
            TopState.VERIFYING,
            TopState.RECOVERY_NEEDED,
            TopState.PAUSED_FOR_HUMAN,
            TopState.COMPLETED,
            TopState.FAILED,
            TopState.ABORTED,
        }
        if event["type"] == "agent_output":
            cp = event.get("payload", {}).get("checkpoint")
            if cp:
                if isinstance(cp, Checkpoint):
                    state.last_agent_checkpoint = cp.to_dict()
                else:
                    state.last_agent_checkpoint = cp
                if state.top_state not in preserve_state:
                    self.store.transition_and_record(
                        state, TopState.GATING,
                        reason="agent checkpoint arrived",
                        source="loop.handle_event",
                    )
        elif event["type"] == "agent_ask":
            if state.top_state not in preserve_state:
                self.store.transition_and_record(
                    state, TopState.GATING,
                    reason="agent asked question",
                    source="loop.handle_event",
                )
        elif event["type"] in {"agent_stop", "timeout"}:
            if state.top_state not in preserve_state:
                self.store.transition_and_record(
                    state, TopState.GATING,
                    reason=f"agent event: {event['type']}",
                    source="loop.handle_event",
                )

    def gate(self, spec, state, *, triggered_by_seq: int = 0,
             triggered_by_checkpoint_id: str = "") -> SupervisorDecision:
        node = spec.get_node(state.current_node_id)
        if node.type == "decision":
            return self.branch_gate.decide(spec, state, node, triggered_by_seq=triggered_by_seq)

        cp = state.last_agent_checkpoint or {}
        cp_status = cp.get("status", "")
        question = (state.last_event or {}).get("payload", {}).get("question", "") or ""

        # Canonical normalization — per Section C of the repartitioning
        # doc, raw payloads are normalized exactly once at the gate entry
        # so every downstream decision is reasoning about typed fields,
        # not raw dict keys. `normalized is None` for malformed payloads
        # (e.g. missing status); v1 payloads come back with schema_version=1
        # and the v2 semantic fields left as None / ().
        normalized = normalize_checkpoint(cp) if cp else None

        # Structured fast-paths — all three branches below require a
        # normalized v2-aware payload. Kept inside one guard block so the
        # precondition is stated once.
        if normalized is not None:
            # 1. Worker declared requires_authorization=true. Per Section D
            #    of the repartitioning doc, this is a worker-declared
            #    authorization request and maps to
            #    esc.authorization_required — distinct from
            #    esc.dangerous_irreversible which the harness emits when
            #    the dangerous-action classifier fires on its own.
            if normalized.requires_authorization is True:
                return SupervisorDecision.make(
                    decision=DecisionType.ESCALATE_TO_HUMAN.value,
                    reason="worker declared requires_authorization=true",
                    gate_type="checkpoint_status",
                    confidence=1.0,
                    needs_human=True,
                    triggered_by_seq=triggered_by_seq,
                    triggered_by_checkpoint_id=triggered_by_checkpoint_id,
                    reason_code=ESC_AUTHORIZATION_REQUIRED,
                )

            # 2. Section E contradiction routing runs before the heuristic
            #    ATTACHED guard. A payload with contradicting v2 semantics
            #    is inherently unsafe to trust via the fast path.
            if normalized.schema_version == 2:
                contradiction = detect_contradiction(normalized, question=question)
                if contradiction is not None:
                    structured = self._route_contradiction(
                        contradiction,
                        state=state,
                        cp_status=cp_status,
                        triggered_by_seq=triggered_by_seq,
                        triggered_by_checkpoint_id=triggered_by_checkpoint_id,
                    )
                    if structured is not None:
                        return structured

            # 3. Worker declared business escalation with blocking_inputs
            #    listed. Route straight to ESCALATE_TO_HUMAN, carrying
            #    esc.missing_external_input so the escalation reason code
            #    matches a classifier-driven hit.
            if (
                normalized.escalation_class == "business"
                and normalized.blocking_inputs
            ):
                return SupervisorDecision.make(
                    decision=DecisionType.ESCALATE_TO_HUMAN.value,
                    reason=(
                        "worker declared business escalation with blocking_inputs="
                        f"{list(normalized.blocking_inputs)}"
                    ),
                    gate_type="checkpoint_status",
                    confidence=1.0,
                    needs_human=True,
                    triggered_by_seq=triggered_by_seq,
                    triggered_by_checkpoint_id=triggered_by_checkpoint_id,
                    reason_code=ESC_MISSING_EXTERNAL_INPUT,
                )

            # 4. Worker declared escalation_class=review. Per the protocol
            #    prompt ("completion proof is ready and a human must sign
            #    off"), this is a legitimate request for human review —
            #    route to ESCALATE_TO_HUMAN with esc.review_required.
            if normalized.escalation_class == "review":
                return SupervisorDecision.make(
                    decision=DecisionType.ESCALATE_TO_HUMAN.value,
                    reason="worker declared escalation_class=review",
                    gate_type="checkpoint_status",
                    confidence=1.0,
                    needs_human=True,
                    triggered_by_seq=triggered_by_seq,
                    triggered_by_checkpoint_id=triggered_by_checkpoint_id,
                    reason_code=ESC_REVIEW_REQUIRED,
                )

        # `blocked` always wins: an agent asking for external input is a
        # legitimate human pause regardless of attach state.
        if cp_status == "blocked":
            return SupervisorDecision.make(
                decision=DecisionType.ESCALATE_TO_HUMAN.value,
                reason="checkpoint says blocked",
                gate_type="checkpoint_status",
                confidence=1.0,
                needs_human=True,
                triggered_by_seq=triggered_by_seq,
                triggered_by_checkpoint_id=triggered_by_checkpoint_id,
                reason_code=ESC_BLOCKED_GENUINE,
            )

        # ATTACHED-boundary guard runs BEFORE the step_done / workflow_done
        # short-circuits.  A first checkpoint claiming step_done/workflow_done
        # with admin-only evidence would otherwise skip straight to
        # VERIFY_STEP, bypassing the first-execution gate entirely — exactly
        # the escape hatch the gate is meant to close. Limited to statuses
        # that make affirmative progress claims; empty/partial statuses fall
        # through to ContinueGate so their missing-status problem isn't
        # hidden behind a RE_INJECT reason string.
        #
        # Business escalation wins over RE_INJECT: if the first checkpoint
        # cites admin-only evidence AND also carries a MISSING_EXTERNAL_INPUT /
        # DANGEROUS_ACTION / BLOCKED signal in needs / question_for_supervisor
        # OR in the agent's question payload, the correct move is
        # ESCALATE_TO_HUMAN, not a re-inject.  ContinueGate enforces the same
        # ordering internally — the shared `escalation` helper is the single
        # source of truth so the two layers cannot drift again.
        #
        # Note: cp_status is guaranteed to be in ("working", "step_done",
        # "workflow_done") here — the status == "blocked" case is handled above
        # at the top of gate() with its own reason/confidence.  The classifier's
        # BLOCKED class can still fire via pattern matches in
        # summary/needs/evidence even when the explicit status field is not
        # "blocked"; that shape belongs here.
        if (
            state.top_state == TopState.ATTACHED
            and cp_status in ("working", "step_done", "workflow_done")
            and is_admin_only_evidence(cp.get("evidence"))
        ):
            esc_hit = classify_for_escalation(cp, question)
            if esc_hit is not None:
                return escalation_decision(
                    esc_hit,
                    gate_type="checkpoint_status",
                    triggered_by_seq=triggered_by_seq,
                    triggered_by_checkpoint_id=triggered_by_checkpoint_id,
                )
            return SupervisorDecision.make(
                decision=DecisionType.RE_INJECT.value,
                reason=(
                    f"attached: first checkpoint claims {cp_status!r} with no "
                    f"execution evidence on current_node"
                ),
                gate_type="checkpoint_status",
                confidence=0.95,
                needs_human=False,
                triggered_by_seq=triggered_by_seq,
                triggered_by_checkpoint_id=triggered_by_checkpoint_id,
            )

        if cp_status == "step_done":
            return SupervisorDecision.make(
                decision=DecisionType.VERIFY_STEP.value,
                reason="checkpoint says step_done",
                gate_type="checkpoint_status",
                confidence=1.0,
                triggered_by_seq=triggered_by_seq,
                triggered_by_checkpoint_id=triggered_by_checkpoint_id,
            )
        if cp_status == "workflow_done":
            return SupervisorDecision.make(
                decision=DecisionType.VERIFY_STEP.value,
                reason="checkpoint says workflow_done",
                gate_type="checkpoint_status",
                confidence=1.0,
                triggered_by_seq=triggered_by_seq,
                triggered_by_checkpoint_id=triggered_by_checkpoint_id,
            )
        return self.continue_gate.decide(build_context(spec, state), triggered_by_seq=triggered_by_seq)

    def _route_contradiction(
        self,
        contradiction,
        *,
        state,
        cp_status: str,
        triggered_by_seq: int,
        triggered_by_checkpoint_id: str,
    ) -> SupervisorDecision | None:
        """Section E — map a detected contradiction to a gate decision.

        Returns None for the "runtime-owned field conflict" route: runtime
        state wins but the caller falls through to its non-contradicted
        routing. The tag is preserved as a session event so operators /
        eval can observe the drift without changing the decision.
        """
        route = contradiction.route
        if route == "runtime_owned_conflict":
            self.store.append_session_event(
                state.run_id,
                "contradiction_demoted",
                {
                    "route": route,
                    "reason_code": contradiction.reason_code,
                    "detail": contradiction.detail,
                },
            )
            return None

        if route == "safety_contradiction":
            return SupervisorDecision.make(
                decision=DecisionType.ESCALATE_TO_HUMAN.value,
                reason=f"safety contradiction: {contradiction.detail}",
                gate_type="checkpoint_status",
                confidence=1.0,
                needs_human=True,
                triggered_by_seq=triggered_by_seq,
                triggered_by_checkpoint_id=triggered_by_checkpoint_id,
                reason_code=contradiction.reason_code,
            )

        if route == "business_contradiction":
            return SupervisorDecision.make(
                decision=DecisionType.ESCALATE_TO_HUMAN.value,
                reason=f"business contradiction: {contradiction.detail}",
                gate_type="checkpoint_status",
                confidence=0.98,
                needs_human=True,
                triggered_by_seq=triggered_by_seq,
                triggered_by_checkpoint_id=triggered_by_checkpoint_id,
                reason_code=contradiction.reason_code,
            )

        if route == "execution_semantic_contradiction":
            # Attach-boundary re-inject does NOT charge the retry budget.
            # The cap on MAX_RE_INJECTS still applies in `apply_decision`.
            return SupervisorDecision.make(
                decision=DecisionType.RE_INJECT.value,
                reason=(
                    f"execution-semantic contradiction ({cp_status!r}): "
                    f"{contradiction.detail}"
                ),
                gate_type="checkpoint_status",
                confidence=0.95,
                needs_human=False,
                triggered_by_seq=triggered_by_seq,
                triggered_by_checkpoint_id=triggered_by_checkpoint_id,
                reason_code=contradiction.reason_code,
            )

        raise ValueError(f"unknown contradiction route: {route!r}")

    def verify_current_node(self, spec, state, *, cwd: str | None = None) -> dict:
        node = spec.get_node(state.current_node_id)
        # Node is "done" if already in done list OR if checkpoint says step_done
        # (the node gets added to done_node_ids after verification passes)
        cp_status = (state.last_agent_checkpoint or {}).get("status", "")
        node_done = (
            state.current_node_id in state.done_node_ids
            or cp_status in ("step_done", "workflow_done")
        )
        context = {"current_node_done": node_done}
        return self.verifier_suite.run(node.verify, context, cwd=cwd)

    def apply_decision(self, spec, state, decision: SupervisorDecision | dict):
        if isinstance(decision, dict):
            state.last_decision = decision
            kind = decision["decision"].upper()
        else:
            state.last_decision = decision.to_dict()
            kind = decision.decision.upper()

        if kind == DecisionType.CONTINUE.value:
            state.re_inject_count = 0
            self.store.transition_and_record(
                state, TopState.RUNNING,
                reason="continue decision",
                source="loop.apply_decision",
            )
            return
        if kind == DecisionType.VERIFY_STEP.value:
            state.re_inject_count = 0
            self.store.transition_and_record(
                state, TopState.VERIFYING,
                reason="verify_step decision",
                source="loop.apply_decision",
            )
            return
        if kind == DecisionType.RE_INJECT.value:
            # Attach-boundary re-inject: MUST NOT touch current_attempt or the
            # retry budget — this is the whole point of having RE_INJECT as a
            # separate decision from RETRY.  The run stays in ATTACHED so the
            # next checkpoint also gets first-execution-evidence scrutiny.
            state.re_inject_count += 1
            # Cap unbounded loops: if the agent keeps emitting admin-only
            # checkpoints after MAX_RE_INJECTS re-injections, stop trying and
            # surface a recovery pause so an operator can intervene. Without
            # this cap, RE_INJECT can loop forever (the agent is responsive,
            # so the delivery-ack timeout never fires).
            if state.re_inject_count > MAX_RE_INJECTS:
                self._pause_for_human(state, {
                    "reason": (
                        f"re-inject exhausted after {MAX_RE_INJECTS} attempts: "
                        f"agent keeps returning admin-only checkpoints for "
                        f"node {state.current_node_id}"
                    ),
                    "node_id": state.current_node_id,
                    "re_inject_count": state.re_inject_count,
                    "pause_class": "recovery",
                    "reason_code": REC_REINJECTION_EXHAUSTED,
                })
            return
        if kind == DecisionType.RETRY.value:
            state.re_inject_count = 0
            state.current_attempt += 1
            state.retry_budget.used_global += 1
            if (state.current_attempt >= state.retry_budget.per_node
                    or state.retry_budget.used_global >= state.retry_budget.global_limit):
                self._pause_for_human(state, {
                    "reason": (
                        f"retry budget exhausted for node {state.current_node_id} "
                        f"(attempt {state.current_attempt}/{state.retry_budget.per_node}, "
                        f"global {state.retry_budget.used_global}/{state.retry_budget.global_limit})"
                    ),
                    "node_id": state.current_node_id,
                    "current_attempt": state.current_attempt,
                    "pause_class": "recovery",
                    "reason_code": REC_RETRY_BUDGET_EXHAUSTED,
                })
            else:
                self.store.transition_and_record(
                    state, TopState.RUNNING,
                    reason="retry decision",
                    source="loop.apply_decision",
                )
            return
        if kind == DecisionType.BRANCH.value:
            _get = decision.get if isinstance(decision, dict) else lambda k, d=None: getattr(decision, k, d)
            state.branch_history.append({
                "node_id": state.current_node_id,
                "selected_branch": _get("selected_branch"),
                "next_node_id": _get("next_node_id"),
                "reason": _get("reason"),
            })
            state.current_node_id = _get("next_node_id")
            state.current_attempt = 0
            self.store.transition_and_record(
                state, TopState.RUNNING,
                reason="branch decision",
                source="loop.apply_decision",
            )
            return
        if kind == DecisionType.ESCALATE_TO_HUMAN.value:
            pause_payload = decision.to_dict() if hasattr(decision, "to_dict") else dict(decision)
            pause_payload["pause_class"] = self._classify_gate_escalation(decision)
            self._pause_for_human(state, pause_payload)
            # Create RoutingDecision for audit trail
            decision_id = (
                decision.decision_id if hasattr(decision, "decision_id")
                else decision.get("decision_id", "") if isinstance(decision, dict)
                else ""
            )
            routing = RoutingDecision(
                target_type="human",
                scope="single_question",
                reason=decision.reason if hasattr(decision, "reason") else str(decision.get("reason", "")),
                triggered_by_decision_id=decision_id,
                consultation_id=latest_oracle_consultation_id_for_run(
                    state.run_id if hasattr(state, "run_id") else "",
                    str(getattr(self.store, "runtime_root", ".supervisor/runtime")),
                ),
            )
            self.store.append_session_event(
                state.run_id if hasattr(state, "run_id") else "",
                "routing", routing.to_dict(),
            )
            return
        if kind == DecisionType.ABORT.value:
            self.store.transition_and_record(
                state, TopState.ABORTED,
                reason="abort decision",
                source="loop.apply_decision",
            )
            return
        if kind == DecisionType.FINISH.value:
            finish = self.finish_gate.evaluate(spec, state, cwd=state.workspace_root or None)
            if finish["ok"]:
                self.store.transition_and_record(
                    state, TopState.COMPLETED,
                    reason="finish gate satisfied",
                    source="loop.apply_decision",
                )
                self._notify_transition(
                    state,
                    event_type="run_completed",
                    reason="workflow completed",
                    next_action=f"thin-supervisor run summarize {state.run_id}",
                )
            else:
                # Finish gate rejected — evidence insufficient for completion.
                # Operator reviews the finish proof, not the session health.
                self._pause_for_human(state, {**finish, "pause_class": "review"})
            return
        raise ValueError(f"unsupported decision: {kind}")

    def apply_verification(self, spec, state, verification: dict, *, cwd: str | None = None):
        state.verification = verification
        if verification["ok"]:
            completed_node_id = state.current_node_id
            if state.current_node_id not in state.done_node_ids:
                state.done_node_ids.append(state.current_node_id)
            next_id = spec.next_node_id(state.current_node_id)
            if next_id is None:
                finish = self.finish_gate.evaluate(spec, state, cwd=cwd)
                if finish["ok"]:
                    self.store.transition_and_record(
                        state, TopState.COMPLETED,
                        reason="final verification passed",
                        source="loop.apply_verification",
                    )
                    self._notify_transition(
                        state,
                        event_type="run_completed",
                        reason="workflow completed",
                        next_action=f"thin-supervisor run summarize {state.run_id}",
                    )
                else:
                    self._pause_for_human(state, {**finish, "pause_class": "review"})
            else:
                state.current_node_id = next_id
                state.current_attempt = 0
                self.store.transition_and_record(
                    state, TopState.RUNNING,
                    reason="verification advanced to next node",
                    source="loop.apply_verification",
                )
                self._notify_transition(
                    state,
                    event_type="step_verified",
                    reason=f"verified {completed_node_id}; advanced to {next_id}",
                    next_action=f"continue current_node {next_id}",
                )
            return
        state.current_attempt += 1
        state.retry_budget.used_global += 1
        if (state.current_attempt >= state.retry_budget.per_node
                or state.retry_budget.used_global >= state.retry_budget.global_limit):
            self._pause_for_human(state, {
                "reason": (
                    f"verification retry budget exhausted for node {state.current_node_id} "
                    f"(attempt {state.current_attempt}/{state.retry_budget.per_node}, "
                    f"global {state.retry_budget.used_global}/{state.retry_budget.global_limit})"
                ),
                "node_id": state.current_node_id,
                "verification": verification,
                "pause_class": "recovery",
                "reason_code": REC_VERIFICATION_RETRY_EXHAUSTED,
            })
        else:
            self.store.transition_and_record(
                state, TopState.RUNNING,
                reason="verification failed but retry budget remains",
                source="loop.apply_verification",
            )

    def _set_delivery_state(self, state, new_state, *, reason: str = "") -> None:
        value = new_state.value if isinstance(new_state, DeliveryState) else str(new_state)
        old = state.delivery_state
        if old == value:
            return
        state.delivery_state = value
        self.store.append_session_event(
            state.run_id, "delivery_state_change",
            {"from": old, "to": value, "reason": reason},
        )

    def _pause_for_human(self, state, payload: dict | None = None) -> dict:
        details = dict(payload or {})
        pclass = str(details.get("pause_class", "")).strip().lower()
        if pclass not in PAUSE_CLASSES:
            # Every pause path must declare its class. Silent defaults here
            # would re-hide the business-vs-recovery split we are explicitly
            # trying to surface. If this raises in production, the caller
            # forgot to tag a new pause site — tag it, don't widen the
            # fallback.
            raise ValueError(
                f"_pause_for_human requires payload['pause_class'] ∈ {PAUSE_CLASSES}; "
                f"got {details.get('pause_class')!r}"
            )
        details["pause_class"] = pclass
        # Capture the source state so resume can restore ATTACHED
        # semantics when the pause originated on the attach boundary
        # (e.g. re-inject cap exhausted). Without this, a PAUSED_FOR_HUMAN
        # → RUNNING resume would skip the first-execution-evidence gate
        # and let the agent slip through with admin-only evidence.
        #
        # Only capture when empty: if the run reached pause via
        # `_enter_recovery` (ATTACHED → RECOVERY_NEEDED → auto-intervention
        # exhausted → _pause_for_human), that path already recorded the
        # pre-recovery state. Overwriting here would clobber "ATTACHED"
        # with "RECOVERY_NEEDED" and let resume silently default to RUNNING.
        if not state.pre_pause_top_state:
            state.pre_pause_top_state = state.top_state.value
        self.store.transition_and_record(
            state, TopState.PAUSED_FOR_HUMAN,
            reason="paused for human",
            source="loop._pause_for_human",
        )
        state.human_escalations.append(details)
        summary = summarize_state(state.to_dict())
        event_payload = dict(details)
        event_payload["pause_reason"] = summary.get("pause_reason", "")
        event_payload["next_action"] = summary.get("next_action", "")
        event_payload["is_waiting_for_review"] = summary.get("is_waiting_for_review", False)
        self.store.append_session_event(state.run_id, "human_pause", event_payload)
        self.notification_manager.notify(NotificationEvent(
            event_type="human_pause",
            run_id=state.run_id,
            top_state=state.top_state.value,
            reason=event_payload["pause_reason"],
            next_action=event_payload["next_action"],
            pane_target=state.pane_target,
            spec_path=state.spec_path,
            workspace_root=state.workspace_root,
            surface_type=state.surface_type,
            delivery_state=state.delivery_state,
            pause_class=pclass,
        ))
        return event_payload

    def _enter_recovery(self, state, payload: dict | None = None) -> dict:
        """Record a recovery-needed transition.

        Mirrors `_pause_for_human` but targets `RECOVERY_NEEDED` — the
        supervisor owns the run from here and a live auto-intervention recipe
        will be attempted. If that recipe fails, the caller is responsible
        for falling through to `_pause_for_human(pause_class='recovery')`.

        Deliberately does NOT fire an operator notification. The whole point
        of RECOVERY_NEEDED is "do not wake a human yet" — observability
        surfaces read the session event log directly.
        """
        details = dict(payload or {})
        pclass = str(details.get("pause_class", "")).strip().lower()
        if pclass != "recovery":
            raise ValueError(
                f"_enter_recovery requires pause_class='recovery'; got {details.get('pause_class')!r}"
            )
        details["pause_class"] = pclass
        # Capture the pre-recovery state before transitioning. If the
        # recovery recipe later fails and falls through to
        # `_pause_for_human`, this preserves the original source state
        # (e.g. ATTACHED) so resume can restore the attach boundary
        # instead of defaulting to RUNNING.
        if not state.pre_pause_top_state:
            state.pre_pause_top_state = state.top_state.value
        self.store.transition_and_record(
            state, TopState.RECOVERY_NEEDED,
            reason="recovery needed",
            source="loop._enter_recovery",
        )
        self.store.append_session_event(state.run_id, "recovery_needed", details)
        return details

    @staticmethod
    def _classify_gate_escalation(decision) -> str:
        """Derive pause_class for an ESCALATE_TO_HUMAN gate decision.

        The gate issues one enum value but the human reason behind it can be
        business, safety, or review. We inspect the reason text; fall back to
        `business` (the most common non-recovery escalate trigger per
        `skills/thin-supervisor/references/escalation-rules.md`).
        """
        if hasattr(decision, "reason"):
            reason = str(decision.reason or "")
        elif isinstance(decision, dict):
            reason = str(decision.get("reason", "") or "")
        else:
            reason = ""
        lowered = reason.lower()
        if reason.startswith("requires review by:") or "insufficient evidence" in lowered:
            return "review"
        if any(
            word in lowered
            for word in ("dangerous", "destructive", "irreversible", "authorization")
        ):
            return "safety"
        return "business"

    def _notify_transition(self, state, *, event_type: str, reason: str, next_action: str) -> None:
        payload = {
            "reason": reason,
            "next_action": next_action,
            "top_state": state.top_state.value,
            "current_node": state.current_node_id,
        }
        self.store.append_session_event(state.run_id, event_type, payload)
        self.notification_manager.notify(NotificationEvent(
            event_type=event_type,
            run_id=state.run_id,
            top_state=state.top_state.value,
            reason=reason,
            next_action=next_action,
            pane_target=state.pane_target,
            spec_path=state.spec_path,
            workspace_root=state.workspace_root,
            surface_type=state.surface_type,
            delivery_state=state.delivery_state,
        ))

    def _attempt_auto_intervention(self, spec, state, terminal, payload: dict | None) -> bool:
        plan = self.auto_intervention_manager.maybe_plan(spec, state, payload or {}, terminal)
        if not plan:
            return False
        ready, reason = self._wait_for_injection_window(
            state, terminal, instruction_id="auto-intervention"
        )
        if not ready:
            self._set_delivery_state(state, DeliveryState.FAILED, reason=reason)
            return False
        self._set_delivery_state(state, DeliveryState.INJECTED, reason="auto intervention")
        try:
            terminal.inject(plan.instruction)
        except Exception:
            logger.exception("auto intervention injection failed")
            self._set_delivery_state(state, DeliveryState.FAILED, reason="auto intervention inject failed")
            return False

        adapter_state = getattr(terminal, "last_delivery_state", None) or DeliveryState.SUBMITTED
        self._set_delivery_state(state, adapter_state, reason="auto intervention confirmed")
        state.auto_intervention_count += 1
        self.store.transition_and_record(
            state, TopState.RUNNING,
            reason="auto intervention requested",
            source="loop._attempt_auto_intervention",
        )
        # Recovery succeeded. Clear any pre-recovery state snapshot — a
        # subsequent natural pause from RUNNING should capture RUNNING,
        # not inherit the stale pre-recovery (e.g. ATTACHED) value.
        state.pre_pause_top_state = ""
        record = {
            "reason": plan.reason,
            "instruction": plan.instruction,
            "count": state.auto_intervention_count,
            "resumed_node": state.current_node_id,
        }
        self.store.append_session_event(state.run_id, "auto_intervention", record)
        self.notification_manager.notify(NotificationEvent(
            event_type="auto_intervention",
            run_id=state.run_id,
            top_state=state.top_state.value,
            reason=plan.reason,
            next_action=f"continue current_node {state.current_node_id}",
            pane_target=state.pane_target,
            spec_path=state.spec_path,
            workspace_root=state.workspace_root,
            surface_type=state.surface_type,
            delivery_state=state.delivery_state,
        ))
        self.store.save(state)
        return True

    @staticmethod
    def _reset_recovery_tracking(state, *, clear_escalations: bool = False) -> None:
        state.auto_intervention_count = 0
        state.node_mismatch_count = 0
        state.last_mismatch_node_id = ""
        if clear_escalations:
            state.human_escalations = []

    def is_final(self, state) -> bool:
        return state.top_state in FINAL_STATES

    # ------------------------------------------------------------------
    # Sidecar loop
    # ------------------------------------------------------------------

    def run_sidecar(self, spec, state, terminal, *, poll_interval: float = 2.0,
                    read_lines: int = 100, stop_event=None,
                    idle_timeout_sec: float | None = None):
        """Main sidecar event loop with full causality chain.

        Checkpoint → SupervisorDecision → HandoffInstruction
        Each object carries IDs linking back to its trigger.

        Parameters
        ----------
        stop_event : threading.Event | None
            External stop signal (used by daemon to stop individual runs).
            If None, SIGTERM handler is installed (foreground mode).
        """
        adapter = TranscriptAdapter()
        max_node_mismatch = 5
        interrupted = False

        surface_id = ""
        if hasattr(terminal, "session_id"):
            try:
                surface_id = terminal.session_id()
            except Exception:
                pass

        # Interrupt mechanism: external stop_event OR SIGTERM handler
        if stop_event is not None:
            interrupted_ref = stop_event.is_set
        else:
            def _sigterm_handler(signum, frame):
                nonlocal interrupted
                interrupted = True
                logger.info("SIGTERM received, saving state and exiting")

            try:
                prev_handler = signal.getsignal(signal.SIGTERM)
                signal.signal(signal.SIGTERM, _sigterm_handler)
            except ValueError:
                prev_handler = None  # not main thread
            interrupted_ref = lambda: interrupted

        # Expose the interrupt predicate to helpers (e.g. hook-handoff poll
        # loop) that need to cooperate with stop_event / SIGTERM without
        # threading the ref through every call site.
        self._interrupted_ref = interrupted_ref
        try:
            self._run_sidecar_inner(
                spec, state, terminal, adapter, surface_id,
                poll_interval=poll_interval, read_lines=read_lines,
                max_node_mismatch=max_node_mismatch,
                interrupted_ref=interrupted_ref,
                idle_timeout_sec=idle_timeout_sec,
            )
        except Exception:
            logger.exception("sidecar loop error")
            self.store.save(state)
            raise
        finally:
            self._interrupted_ref = None
            if stop_event is None and prev_handler is not None:
                try:
                    signal.signal(signal.SIGTERM, prev_handler)
                except ValueError:
                    pass  # not main thread

        return state

    def _run_sidecar_inner(
        self, spec, state, terminal, adapter, surface_id, *,
        poll_interval, read_lines,
        max_node_mismatch, interrupted_ref, idle_timeout_sec,
    ):
        if poll_interval is None or poll_interval <= 0:
            effective_poll_interval = MIN_POLL_SLEEP_SEC
        else:
            effective_poll_interval = max(poll_interval, MIN_POLL_SLEEP_SEC)
        effective_idle_timeout_sec = idle_timeout_sec
        if effective_idle_timeout_sec is None and (poll_interval is None or poll_interval <= 0):
            effective_idle_timeout_sec = ZERO_POLL_IDLE_TIMEOUT_SEC
        pending_text = None
        last_activity_at = time.monotonic()
        recovery_attempts: dict[str, int] = {}
        recovery_total_attempts = 0
        scheduled_recovery: RuntimeRecoveryObservation | None = None
        scheduled_recovery_signature = ""
        exhausted_recovery_signatures: set[str] = set()
        # Delivery ack: transient monotonic time of last injection (not persisted)
        delivery_ack_deadline = 0.0  # 0 = not awaiting ack
        DELIVERY_ACK_TIMEOUT = 60  # seconds

        # Compute supervision policy based on worker + contract + state
        contract = spec.acceptance or AcceptanceContract.from_finish_policy(spec.finish_policy, goal=spec.goal)
        policy = self.policy_engine.determine(self.worker_profile, contract, state)
        logger.info("supervision policy: %s (%s)", policy.mode, policy.reason)

        # Boot-time fail-safe: the sidecar should never *start* with
        # top_state=RECOVERY_NEEDED. That state is transient — `_enter_recovery`
        # is always immediately followed by `_attempt_auto_intervention` (which
        # transitions to RUNNING on success) or `_pause_for_human`. If it
        # persisted across a process boundary, the only explanation is that
        # the prior sidecar crashed between those two calls. Without this
        # fail-safe the main loop would spin reading the pane forever with
        # no delivery_ack_deadline armed and no pending checkpoint to drive
        # a transition — a permanent hang. Surface to an operator instead.
        if state.top_state == TopState.RECOVERY_NEEDED:
            logger.warning(
                "run %s booted in RECOVERY_NEEDED — prior sidecar crashed mid-recovery",
                state.run_id,
            )
            self._pause_for_human(state, {
                "reason": (
                    "sidecar crashed during recovery; the stalled auto-intervention "
                    "cannot be safely replayed without operator review"
                ),
                "node_id": state.current_node_id,
                "pause_class": "recovery",
                "reason_code": REC_CRASH_DURING_RECOVERY,
            })
            self.store.save(state)
            return

        # READY → ATTACHED: inject first instruction. ATTACHED gates the first
        # checkpoint — a CONTINUE requires real execution evidence on the
        # current node, not clarify/plan/attach artifacts. Fresh register only;
        # resume paths skip this branch and keep their existing top_state.
        if state.top_state == TopState.READY:
            self.store.transition_and_record(
                state, TopState.ATTACHED,
                reason="initial handoff",
                source="loop.run_sidecar",
            )
            self.store.save(state)
            pending_text = terminal.read(lines=read_lines)
            # Only parse the LAST checkpoint in the pane to avoid stale ones
            # from previous runs that would falsely suppress init inject.
            # Find the last <checkpoint> block rather than truncating by line count,
            # since checkpoint blocks can vary in size.
            init_parse_text = ""
            if pending_text:
                last_cp_start = pending_text.rfind("<checkpoint>")
                if last_cp_start >= 0:
                    init_parse_text = pending_text[last_cp_start:]
            cp = adapter.parse_checkpoint(init_parse_text, run_id=state.run_id, surface_id=surface_id)
            if cp and cp.run_id and cp.run_id != state.run_id:
                cp = None  # checkpoint from a different run
            if cp:
                # The pane/session is already emitting progress for the current
                # node, so avoid re-injecting the same instruction on the first
                # working checkpoint.
                state.last_injected_node_id = state.current_node_id
                state.last_injected_attempt = state.current_attempt
                state.last_injection_seq = state.checkpoint_seq
                delivery_ack_deadline = time.monotonic() + DELIVERY_ACK_TIMEOUT
            if not cp:
                # Skip init inject for observation-only surfaces (agent already running)
                if not getattr(terminal, "is_observation_only", False):
                    node = spec.get_node(state.current_node_id)
                    instruction = self.composer.build(
                        node, state,
                        triggered_by_decision_id="",
                        trigger_type="init",
                        policy=policy,
                        first_node_delivery=True,
                    )
                    state.last_injected_node_id = state.current_node_id
                    state.last_injected_attempt = 0
                    state.last_injection_seq = state.checkpoint_seq
                    self.store.save(state)
                    if not self._inject_or_pause(state, terminal, instruction, spec=spec):
                        return
                    delivery_ack_deadline = time.monotonic() + DELIVERY_ACK_TIMEOUT
                pending_text = None

        while not self.is_final(state) and state.top_state != TopState.PAUSED_FOR_HUMAN:
            if interrupted_ref():
                self.store.save(state)
                return

            # Cache monotonic time for this iteration
            now = time.monotonic()

            # Delivery ack timeout: if we injected but no checkpoint arrived
            if (delivery_ack_deadline > 0
                    and now > delivery_ack_deadline
                    and state.checkpoint_seq <= state.last_injection_seq):
                self._set_delivery_state(state, DeliveryState.TIMED_OUT, reason="no checkpoint after injection")
                payload = {
                    "reason": "no checkpoint received within delivery timeout after injection",
                    "node_id": state.current_node_id,
                    "timeout_sec": DELIVERY_ACK_TIMEOUT,
                    "injection_seq": state.last_injection_seq,
                    "delivery_state": state.delivery_state,
                    "pause_class": "recovery",
                    "reason_code": REC_DELIVERY_TIMEOUT,
                }
                self.store.append_session_event(state.run_id, "delivery_ack_timeout", payload)
                logger.warning("delivery ack timeout: no checkpoint after injection for %ds", DELIVERY_ACK_TIMEOUT)
                recovery_payload = self._enter_recovery(state, payload)
                if self._attempt_auto_intervention(spec, state, terminal, recovery_payload):
                    # Auto-intervention injected a new instruction — reset tracking
                    state.last_injection_seq = state.checkpoint_seq
                    delivery_ack_deadline = now + DELIVERY_ACK_TIMEOUT
                    self.store.save(state)
                    continue
                # Recipe exhausted — surface to a human with recovery class
                self._pause_for_human(state, payload)
                self.store.save(state)
                return

            # 1. Read pane
            try:
                text = pending_text if pending_text is not None else terminal.read(lines=read_lines)
            except Exception as e:
                logger.warning("terminal read failed: %s", e)
                time.sleep(effective_poll_interval)
                continue
            pending_text = None
            if text:
                last_activity_at = now

            # 2. Parse checkpoint with identity
            checkpoints = adapter.parse_checkpoints(text, run_id=state.run_id, surface_id=surface_id)
            if not checkpoints:
                wall_clock_now = datetime.now(timezone.utc)
                recovery = detect_runtime_recovery(
                    text,
                    now=wall_clock_now,
                    policy=self.runtime_recovery_policy,
                )
                if recovery is not None:
                    if recovery_total_attempts < self.runtime_recovery_policy.max_attempts_per_run:
                        if scheduled_recovery is None or scheduled_recovery_signature != recovery.signature:
                            scheduled_recovery = recovery
                            scheduled_recovery_signature = recovery.signature
                            self.store.append_session_event(
                                state.run_id,
                                "runtime_recovery_scheduled",
                                {
                                    "kind": recovery.kind,
                                    "reason": recovery.reason,
                                    "retry_at": recovery.retry_at.isoformat(),
                                    "attempt": recovery_total_attempts + 1,
                                    "max_attempts": self.runtime_recovery_policy.max_attempts_per_run,
                                },
                            )
                            self.store.save(state)
                    elif recovery.signature not in exhausted_recovery_signatures:
                        exhausted_recovery_signatures.add(recovery.signature)
                        self.store.append_session_event(
                            state.run_id,
                            "runtime_recovery_exhausted",
                            {
                                "kind": recovery.kind,
                                "reason": recovery.reason,
                                "attempts": recovery_total_attempts,
                                "max_attempts": self.runtime_recovery_policy.max_attempts_per_run,
                            },
                        )
                        self.store.save(state)

                if (
                    scheduled_recovery is not None
                    and seconds_until_recovery(scheduled_recovery.retry_at, now=wall_clock_now) <= 0
                ):
                    attempts = recovery_attempts.get(scheduled_recovery.signature, 0)
                    instruction = self._build_runtime_recovery_instruction(
                        state, scheduled_recovery
                    )
                    state.last_injected_node_id = state.current_node_id
                    state.last_injected_attempt = state.current_attempt
                    state.last_injection_seq = state.checkpoint_seq
                    self.store.save(state)
                    if not self._inject_or_pause(state, terminal, instruction, spec=spec):
                        return
                    recovery_attempts[scheduled_recovery.signature] = attempts + 1
                    recovery_total_attempts += 1
                    self.store.append_session_event(
                        state.run_id,
                        "runtime_recovery_injected",
                        {
                            "kind": scheduled_recovery.kind,
                            "reason": scheduled_recovery.reason,
                            "attempt": recovery_total_attempts,
                            "max_attempts": self.runtime_recovery_policy.max_attempts_per_run,
                        },
                    )
                    scheduled_recovery = None
                    scheduled_recovery_signature = ""
                    delivery_ack_deadline = time.monotonic() + DELIVERY_ACK_TIMEOUT
                    time.sleep(effective_poll_interval)
                    continue

                if scheduled_recovery is not None:
                    time.sleep(effective_poll_interval)
                    continue

                if effective_idle_timeout_sec and effective_idle_timeout_sec > 0:
                    idle_for = now - last_activity_at
                    if idle_for >= effective_idle_timeout_sec:
                        payload = {
                            "reason": f"agent idle timeout after {int(idle_for)}s without checkpoint or visible output",
                            "idle_timeout_sec": effective_idle_timeout_sec,
                            "node_id": state.current_node_id,
                            "pause_class": "recovery",
                            "reason_code": REC_IDLE_TIMEOUT,
                        }
                        self.store.append_event({"type": "timeout", "payload": payload})
                        self.store.append_session_event(state.run_id, "agent_idle_timeout", payload)
                        self.handle_event(state, {"type": "timeout", "payload": payload})
                        recovery_payload = self._enter_recovery(state, payload)
                        if self._attempt_auto_intervention(spec, state, terminal, recovery_payload):
                            last_activity_at = now
                            self.store.save(state)
                            time.sleep(effective_poll_interval)
                            continue
                        # Recipe exhausted — surface to a human with recovery class
                        self._pause_for_human(state, payload)
                        self.store.save(state)
                        return
                time.sleep(effective_poll_interval)
                continue

            restart_loop = False
            deferred_continue = None
            preserve_checkpoint_buffer = False
            for checkpoint in checkpoints:
                # #2: reject checkpoints from wrong run
                if checkpoint.run_id and checkpoint.run_id != state.run_id:
                    continue

                # #7: seq-based dedup with reset tolerance
                if checkpoint.checkpoint_seq > 0:
                    if checkpoint.checkpoint_seq <= state.checkpoint_seq:
                        seq_reset = checkpoint.checkpoint_seq == 1 and state.checkpoint_seq > 0
                        large_gap_reset = state.checkpoint_seq - checkpoint.checkpoint_seq >= 100
                        if not seq_reset and not large_gap_reset:
                            continue
                # Content-based dedup
                last_cp = state.last_agent_checkpoint
                if (last_cp
                        and checkpoint.status == last_cp.get("status")
                        and checkpoint.current_node == last_cp.get("current_node")
                        and checkpoint.summary == last_cp.get("summary")
                        and checkpoint.checkpoint_seq == last_cp.get("checkpoint_seq", 0)):
                    continue

                # #5: node mismatch — observation-only surfaces cannot rely on
                # injected instructions, so bind unknown nodes to the current spec
                # node but escalate if the agent is still reporting an already-done
                # node (delivery clearly stalled).
                if checkpoint.current_node != state.current_node_id:
                    if getattr(terminal, "is_observation_only", False):
                        if checkpoint.current_node in state.done_node_ids:
                            reason = (
                                "observation-only surface is still reporting a completed node; "
                                "supervisor instruction was likely not delivered"
                            )
                            logger.warning("%s: cp=%s state=%s",
                                           reason, checkpoint.current_node, state.current_node_id)
                            payload = {
                                "reason": reason,
                                "checkpoint_node": checkpoint.current_node,
                                "state_node": state.current_node_id,
                                "pause_class": "recovery",
                                "reason_code": REC_DELIVERY_TIMEOUT,
                            }
                            self.store.append_session_event(
                                state.run_id, "observation_delivery_stalled", payload
                            )
                            pause_payload = self._pause_for_human(state, payload)
                            self._attempt_auto_intervention(spec, state, terminal, pause_payload)
                            self.store.save(state)
                            return
                        logger.info(
                            "observation-only: rebinding checkpoint node cp=%s -> state=%s",
                            checkpoint.current_node, state.current_node_id,
                        )
                        checkpoint.current_node = state.current_node_id
                    else:
                        state.node_mismatch_count += 1
                        state.last_mismatch_node_id = checkpoint.current_node
                        logger.warning("checkpoint node mismatch (%d/%d): cp=%s state=%s",
                                       state.node_mismatch_count, max_node_mismatch,
                                       checkpoint.current_node, state.current_node_id)
                        self.store.append_session_event(
                            state.run_id, "checkpoint_mismatch",
                            {"checkpoint_node": checkpoint.current_node, "state_node": state.current_node_id,
                             "count": state.node_mismatch_count},
                        )
                        self.store.save(state)
                        if state.node_mismatch_count >= max_node_mismatch:
                            mismatch_payload = {
                                "reason": f"node mismatch persisted for {state.node_mismatch_count} checkpoints",
                                "checkpoint_node": checkpoint.current_node,
                                "state_node": state.current_node_id,
                                "pause_class": "recovery",
                                "reason_code": REC_NODE_MISMATCH_PERSISTED,
                            }
                            recovery_payload = self._enter_recovery(state, mismatch_payload)
                            if self._attempt_auto_intervention(spec, state, terminal, recovery_payload):
                                restart_loop = True
                                break
                            # Recipe exhausted — surface to a human with recovery class
                            self._pause_for_human(state, mismatch_payload)
                            self.store.save(state)
                            return
                        preserve_checkpoint_buffer = True
                        continue
                state.node_mismatch_count = 0
                state.last_mismatch_node_id = ""
                preserve_checkpoint_buffer = False

                # Accepted checkpoint — clear delivery deadline (post-dedup)
                if delivery_ack_deadline > 0:
                    delivery_ack_deadline = 0
                if checkpoint.checkpoint_seq > 0:
                    state.checkpoint_seq = checkpoint.checkpoint_seq
                if (state.checkpoint_seq > state.last_injection_seq
                        and state.delivery_state not in (DeliveryState.IDLE, DeliveryState.STARTED_PROCESSING)):
                    self._set_delivery_state(state, DeliveryState.STARTED_PROCESSING, reason="checkpoint received")
                scheduled_recovery = None
                scheduled_recovery_signature = ""
                logger.info("checkpoint: %s (id=%s)", checkpoint.summary, checkpoint.checkpoint_id)
                if checkpoint.status in {"working", "step_done", "workflow_done"}:
                    self._reset_recovery_tracking(state, clear_escalations=False)

                # 3. Event
                cp_dict = checkpoint.to_dict()
                event = {"type": "agent_output", "payload": {"checkpoint": cp_dict}}
                self.store.append_event(event)
                self.store.append_session_event(state.run_id, "checkpoint", cp_dict)
                self.handle_event(state, event)

                # 4. Gate → SupervisorDecision
                #    ATTACHED uses the same gate as GATING so the first
                #    checkpoint after register is gated for execution evidence
                #    (not admin-only artifacts) before CONTINUE can advance.
                decision: SupervisorDecision | None = None
                if state.top_state in (TopState.GATING, TopState.ATTACHED):
                    decision = self.gate(
                        spec, state,
                        triggered_by_seq=checkpoint.checkpoint_seq,
                        triggered_by_checkpoint_id=checkpoint.checkpoint_id,
                    )
                    self.store.append_decision(decision.to_dict())
                    self.store.append_session_event(state.run_id, "gate_decision", decision.to_dict())
                    self.apply_decision(spec, state, decision)
                    logger.info("decision: %s (id=%s)", decision.decision, decision.decision_id)
                    if state.top_state == TopState.PAUSED_FOR_HUMAN:
                        pause_payload = latest_human_escalation(state.to_dict())
                        if self._attempt_auto_intervention(spec, state, terminal, pause_payload):
                            restart_loop = True
                            break

                # 4b. RE_INJECT — attach-boundary re-inject of a focused
                # first-execution prompt.  Dedicated branch so retry-counter
                # semantics can never leak into this path. Skipped when
                # apply_decision already paused the run (re-inject cap
                # exhausted) — we do not want to inject again after the
                # recovery pause fires.
                if (
                    decision
                    and decision.decision.upper() == DecisionType.RE_INJECT.value
                    and state.top_state == TopState.ATTACHED
                ):
                    node = spec.get_node(state.current_node_id)
                    re_instruction = self.composer.build(
                        node, state,
                        triggered_by_decision_id=decision.decision_id,
                        trigger_type="re_inject",
                        policy=policy,
                        first_node_delivery=True,
                    )
                    state.last_injected_node_id = state.current_node_id
                    state.last_injected_attempt = state.current_attempt
                    state.last_injection_seq = state.checkpoint_seq
                    self.store.save(state)
                    if not self._inject_or_pause(state, terminal, re_instruction, spec=spec):
                        return
                    delivery_ack_deadline = time.monotonic() + DELIVERY_ACK_TIMEOUT
                    logger.info(
                        "re-injected: %s (id=%s, decision=%s)",
                        node.id, re_instruction.instruction_id, decision.decision_id,
                    )
                    restart_loop = True
                    break

                # 5. Verify
                if state.top_state == TopState.VERIFYING:
                    cwd = self._get_cwd(terminal, state)
                    try:
                        verification = self.verify_current_node(spec, state, cwd=cwd)
                    except Exception as e:
                        logger.error("verification error: %s", e)
                        verification = {"ok": False, "results": [{"type": "error", "ok": False, "reason": str(e)}]}
                    self.store.append_event({"type": "verification_finished", "payload": verification})
                    self.store.append_session_event(state.run_id, "verification", verification)
                    self.apply_verification(spec, state, verification, cwd=cwd)
                    logger.info("verification ok=%s, state=%s", verification.get("ok"), state.top_state.value)
                    if state.top_state == TopState.PAUSED_FOR_HUMAN:
                        pause_payload = latest_human_escalation(state.to_dict())
                        if self._attempt_auto_intervention(spec, state, terminal, pause_payload):
                            restart_loop = True
                            break

                # 6. Inject — #11: save BEFORE inject
                if state.top_state == TopState.RUNNING:
                    node_changed = state.current_node_id != state.last_injected_node_id
                    new_retry = state.current_attempt > 0 and state.current_attempt != state.last_injected_attempt
                    continue_guidance = bool(
                        decision
                        and decision.decision.upper() == DecisionType.CONTINUE.value
                        and getattr(decision, "next_instruction", None)
                    )
                    if node_changed or new_retry or continue_guidance:
                        node = spec.get_node(state.current_node_id)
                        decision_id = decision.decision_id if decision else ""
                        trigger = (
                            "retry"
                            if new_retry
                            else (
                                "continue"
                                if continue_guidance
                                else ("branch" if decision and decision.decision.upper() == "BRANCH" else "node_advance")
                            )
                        )
                        # Re-evaluate policy (node advance resets attempt, failures escalate)
                        policy = self.policy_engine.determine(self.worker_profile, contract, state)
                        instruction = self.composer.build(
                            node, state,
                            triggered_by_decision_id=decision_id,
                            trigger_type=trigger,
                            policy=policy,
                            first_node_delivery=(trigger in ("node_advance", "branch")),
                        )
                        if continue_guidance:
                            deferred_continue = (instruction, state.current_node_id, state.current_attempt)
                            continue
                        # Persist the selected next node before delivery so a
                        # crash cannot replay the previous instruction.
                        state.last_injected_node_id = state.current_node_id
                        state.last_injected_attempt = state.current_attempt
                        state.last_injection_seq = state.checkpoint_seq
                        self.store.save(state)
                        if not self._inject_or_pause(state, terminal, instruction, spec=spec):
                            return
                        delivery_ack_deadline = time.monotonic() + DELIVERY_ACK_TIMEOUT
                        logger.info("injected: %s (id=%s, trigger=%s)", node.id, instruction.instruction_id, trigger)
                        if state.top_state == TopState.PAUSED_FOR_HUMAN:
                            return
                        if continue_guidance:
                            continue
                        restart_loop = True
                        break

            # 7. Persist + progress
            if deferred_continue is not None and state.top_state == TopState.RUNNING and not restart_loop:
                deferred_continue_instruction, expected_node_id, expected_attempt = deferred_continue
                if (
                    state.current_node_id != expected_node_id
                    or state.current_attempt != expected_attempt
                ):
                    deferred_continue = None
            if deferred_continue is not None and state.top_state == TopState.RUNNING and not restart_loop:
                deferred_continue_instruction, _expected_node_id, _expected_attempt = deferred_continue
                state.last_injected_node_id = state.current_node_id
                state.last_injected_attempt = state.current_attempt
                state.last_injection_seq = state.checkpoint_seq
                self.store.save(state)
                if not self._inject_or_pause(state, terminal, deferred_continue_instruction, spec=spec):
                    return
                delivery_ack_deadline = time.monotonic() + DELIVERY_ACK_TIMEOUT
                logger.info(
                    "injected: %s (id=%s, trigger=%s)",
                    state.current_node_id,
                    deferred_continue_instruction.instruction_id,
                    deferred_continue_instruction.trigger_type,
                )
            self.store.save(state)
            if hasattr(terminal, "consume_checkpoint") and not preserve_checkpoint_buffer:
                try:
                    terminal.consume_checkpoint()
                except Exception as exc:
                    logger.debug("consume_checkpoint failed: %s", exc)
            try:
                write_progress(state, spec, str(self.store.runtime_dir))
            except Exception:
                pass  # progress is best-effort
            if restart_loop:
                continue

    def _get_cwd(self, terminal, state=None) -> str | None:
        if hasattr(terminal, "current_cwd"):
            try:
                cwd = terminal.current_cwd()
                if cwd:
                    return cwd
            except Exception:
                pass
        # Fallback to persisted workspace_root
        if state and state.workspace_root:
            return state.workspace_root
        return None

    def _build_runtime_recovery_instruction(self, state, recovery: RuntimeRecoveryObservation) -> HandoffInstruction:
        content = (
            "retry\n\n"
            "Runtime recovery detected a provider/connectivity failure outside the task logic. "
            f"Retry the last interrupted action and continue current_node={state.current_node_id}. "
            f"Observed {recovery.kind}: {recovery.reason}"
        )
        return HandoffInstruction.make(
            content=content,
            node_id=state.current_node_id,
            current_attempt=state.current_attempt,
            triggered_by_decision_id="",
            trigger_type="runtime_recovery",
        )

    def _wait_for_injection_window(self, state, terminal, *, instruction_id: str) -> tuple[bool, str]:
        readiness_fn = getattr(terminal, "injection_readiness", None)
        if not callable(readiness_fn):
            return True, ""

        max_defers = 3
        for attempt in range(1, max_defers + 1):
            outcome, reason = readiness_fn()
            if outcome == "inject":
                return True, reason

            payload = {
                "instruction_id": instruction_id,
                "node_id": state.current_node_id,
                "attempt": attempt,
                "reason": reason,
                "outcome": outcome,
            }
            self.store.append_session_event(state.run_id, "injection_deferred", payload)
            if outcome != "defer":
                return False, f"terminal unavailable before injection: {reason}"
            if attempt >= max_defers:
                return False, f"terminal stayed busy before injection: {reason}"
            time.sleep(MIN_POLL_SLEEP_SEC)

        return True, ""

    def _inject_via_hook_handoff(self, state, terminal, instruction) -> bool:
        """Write instruction to the hook-handoff file and wait for the Stop-hook ACK.

        Returns True when the ACK was observed within the timeout; False when
        the run was paused for human attention.
        """
        inject_with_id = getattr(terminal, "inject_with_id", None)
        poll_delivery = getattr(terminal, "poll_delivery", None)
        self._set_delivery_state(state, DeliveryState.INJECTED, reason="hook handoff written")

        try:
            if callable(inject_with_id):
                inject_with_id(
                    instruction.content,
                    instruction_id=instruction.instruction_id,
                    run_id=state.run_id,
                    node_id=state.current_node_id,
                )
            else:
                terminal.inject(instruction.content)
        except Exception as exc:
            logger.warning("observation-only inject failed: %s", exc)
            self._set_delivery_state(
                state, DeliveryState.FAILED, reason=f"inject failed: {exc}"
            )
            self._pause_for_human(state, {
                "reason": f"failed to write instruction handoff file: {exc}",
                "node_id": state.current_node_id,
                "instruction_id": instruction.instruction_id,
                "pause_class": "recovery",
                "reason_code": REC_INJECT_FAILED,
            })
            self.store.save(state)
            return False

        self.store.append_session_event(
            state.run_id, "injection_hook_handoff", instruction.to_dict()
        )

        if not callable(poll_delivery):
            # Adapter doesn't expose a delivery poll — same degraded behaviour
            # as before: pause so the operator knows delivery is unconfirmed.
            self._set_delivery_state(
                state, DeliveryState.FAILED, reason="surface lacks poll_delivery"
            )
            self._pause_for_human(state, {
                "reason": (
                    "observation-only surface cannot confirm instruction delivery; "
                    "resume on an interactive surface or wire delivery hooks"
                ),
                "node_id": state.current_node_id,
                "instruction_id": instruction.instruction_id,
                "pause_class": "recovery",
                "reason_code": REC_INJECT_FAILED,
            })
            self.store.save(state)
            return False

        interrupted_ref = getattr(self, "_interrupted_ref", None)
        deadline = time.monotonic() + OBSERVATION_HOOK_ACK_TIMEOUT_SEC
        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 5
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            # Respect daemon stop / SIGTERM without waiting out the 10-min ACK
            # window. Persist the current state and bail cleanly.
            if interrupted_ref is not None and interrupted_ref():
                self.store.save(state)
                return False
            try:
                delivered = poll_delivery(instruction.instruction_id)
            except Exception as exc:  # noqa: BLE001 — adapter contract is loose
                consecutive_errors += 1
                last_error = exc
                logger.warning(
                    "poll_delivery raised (%d/%d): %s",
                    consecutive_errors, MAX_CONSECUTIVE_ERRORS, exc,
                )
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    break
            else:
                consecutive_errors = 0
                if delivered:
                    self._set_delivery_state(
                        state, DeliveryState.ACKNOWLEDGED, reason="stop hook ACK"
                    )
                    self.store.append_session_event(
                        state.run_id,
                        "injection_hook_ack",
                        {
                            "instruction_id": instruction.instruction_id,
                            "node_id": state.current_node_id,
                        },
                    )
                    self.store.save(state)
                    return True
            time.sleep(OBSERVATION_HOOK_POLL_INTERVAL_SEC)

        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            self._set_delivery_state(
                state, DeliveryState.FAILED,
                reason=f"poll_delivery error cap: {last_error}",
            )
            payload = {
                "reason": (
                    f"observation-only surface poll_delivery failed "
                    f"{consecutive_errors} times in a row: {last_error}"
                ),
                "node_id": state.current_node_id,
                "instruction_id": instruction.instruction_id,
                "pause_class": "recovery",
                "reason_code": REC_INJECT_FAILED,
            }
            self.store.append_session_event(
                state.run_id, "injection_hook_poll_errors", payload
            )
            self._pause_for_human(state, payload)
            self.store.save(state)
            return False

        # No ACK within the window.
        self._set_delivery_state(
            state, DeliveryState.TIMED_OUT, reason="stop hook ACK timeout"
        )
        payload = {
            "reason": (
                "observation-only surface waited for the Stop-hook ACK but it "
                f"did not arrive within {OBSERVATION_HOOK_ACK_TIMEOUT_SEC}s; "
                "confirm the agent has the `thin-supervisor hook stop` hook wired"
            ),
            "node_id": state.current_node_id,
            "instruction_id": instruction.instruction_id,
            "pause_class": "recovery",
            "reason_code": REC_INJECT_FAILED,
        }
        self.store.append_session_event(
            state.run_id, "injection_hook_ack_timeout", payload
        )
        self._pause_for_human(state, payload)
        self.store.save(state)
        return False

    def _inject_or_pause(self, state, terminal, instruction, *, spec=None) -> bool:
        # Observation-only surfaces (e.g., JSONL): write the instruction to the
        # hook-handoff file and poll for the Stop-hook ACK before proceeding.
        # If no ACK arrives within the window, fall through to pause-for-human
        # so an operator can move the run to an interactive surface or wire
        # the hook.
        if getattr(terminal, "is_observation_only", False):
            return self._inject_via_hook_handoff(state, terminal, instruction)

        ready, readiness_reason = self._wait_for_injection_window(
            state, terminal, instruction_id=instruction.instruction_id
        )
        if not ready:
            self._set_delivery_state(state, DeliveryState.FAILED, reason=readiness_reason)
            payload = {
                "instruction_id": instruction.instruction_id,
                "node_id": state.current_node_id,
                "error": readiness_reason,
            }
            self.store.append_session_event(state.run_id, "injection_failed", payload)
            inject_fail_payload = {
                "reason": readiness_reason,
                "node_id": state.current_node_id,
                "instruction_id": instruction.instruction_id,
                "pause_class": "recovery",
            }
            if spec is not None:
                recovery_payload = self._enter_recovery(state, inject_fail_payload)
                if self._attempt_auto_intervention(spec, state, terminal, recovery_payload):
                    self.store.save(state)
                    return True
            self._pause_for_human(state, inject_fail_payload)
            self.store.save(state)
            return False

        self._set_delivery_state(state, DeliveryState.INJECTED, reason="sending to terminal")
        try:
            terminal.inject(instruction.content)
        except Exception as exc:
            self._set_delivery_state(state, DeliveryState.FAILED, reason=str(exc))
            payload = {
                "instruction_id": instruction.instruction_id,
                "node_id": state.current_node_id,
                "error": str(exc),
            }
            self.store.append_session_event(state.run_id, "injection_failed", payload)
            inject_fail_payload = {
                "reason": f"injection failed: {exc}",
                "node_id": state.current_node_id,
                "instruction_id": instruction.instruction_id,
                "pause_class": "recovery",
                "reason_code": REC_INJECT_FAILED,
            }
            # If caller threaded the spec through, try an auto-intervention
            # re-inject; otherwise fall through to a recovery-flavored human
            # pause (AutoInterventionManager short-circuits on spec=None).
            if spec is not None:
                recovery_payload = self._enter_recovery(state, inject_fail_payload)
                if self._attempt_auto_intervention(spec, state, terminal, recovery_payload):
                    self.store.save(state)
                    return True
            self._pause_for_human(state, inject_fail_payload)
            self.store.save(state)
            return False

        # Read adapter-reported delivery state (tmux: SUBMITTED or ACKNOWLEDGED)
        adapter_state = getattr(terminal, "last_delivery_state", None) or DeliveryState.SUBMITTED
        self._set_delivery_state(state, adapter_state, reason="terminal confirmed")
        self.store.append_session_event(
            state.run_id, "injection", instruction.to_dict()
        )
        return True
