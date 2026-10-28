"""
Orchestrator: Main agent control loop.
Routes steps to workers, applies actions, saves state, and emits events.
"""

import os
import json
import uuid
import time
import traceback
import logging
import asyncio
import threading
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional
from genai_wrapper import Client
from google import genai

from events import event_bus
from tools import tools, validate_path, RateLimiter
from workers import Planner, Coder, Reviewer, validate_action

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("Orchestrator")


# -------------------------
# Config & State
# -------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    model: str = "gemini-2.5-flash"
    state_path: str = field(default_factory=lambda: os.path.join(_SCRIPT_DIR, "state.json"))
    memory_path: str = field(default_factory=lambda: os.path.join(_SCRIPT_DIR, "memory.json"))
    max_steps: int = 50
    workspace_root: str = field(default_factory=lambda: os.path.join(_SCRIPT_DIR, "workspace"))
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    llm_max_requests_per_minute: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_REQUESTS_PER_MINUTE", "5")))
    llm_burst: int = field(default_factory=lambda: int(os.getenv("LLM_BURST", "1")))


@dataclass
class OrchestratorState:
    """Saved state of an orchestrator run."""
    run_id: str
    goal: str
    plan: Dict[str, Any]
    current_step: int = 0
    total_steps: int = 0
    status: str = "idle"  # idle, running, completed, error, waiting_user, waiting_quota
    user_response: Optional[str] = None
    completed_steps: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)  # Step summaries and execution log
    last_error: str = ""  # Most recent error message for UI display
    last_summary: str = ""  # Last step summary for final output
    final_output: Optional[str] = None  # Final result to display to user
    last_run_at: Optional[float] = None  # Timestamp when run started
    updated_at: float = field(default_factory=time.time)  # Last update timestamp
    artifacts: List[Dict[str, Any]] = field(default_factory=list)  # Files created during run
    last_worker: str = ""  # Last worker type that processed a step

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class JsonMemory:
    """Simple JSON-based memory for facts and notes."""

    def __init__(self, path: str):
        self.path = path
        self.state = self._load()

    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {"facts": [], "notes": []}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {"facts": [], "notes": []}

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    def add_fact(self, text: str) -> None:
        self.state.setdefault("facts", []).append(text)
        self.save()

    def add_note(self, text: str) -> None:
        self.state.setdefault("notes", []).append(text)
        self.save()

    def snapshot(self) -> str:
        """Get a snapshot for context (last 6 facts/notes)."""
        facts = self.state.get("facts", [])[-6:]
        notes = self.state.get("notes", [])[-6:]
        return (
            "MEMORY\nFacts: " + str(facts) + "\nNotes: " + str(notes)
        )


# -------------------------
# Orchestrator
# -------------------------

# Global store for cancellation flags per run (thread-safe)
_cancel_flags_lock = threading.Lock()
_cancel_flags: Dict[str, threading.Event] = {}

def get_cancel_flag(run_id: str) -> threading.Event:
    """Get or create a cancel flag for a run."""
    with _cancel_flags_lock:
        if run_id not in _cancel_flags:
            _cancel_flags[run_id] = threading.Event()
        return _cancel_flags[run_id]

def clear_cancel_flag(run_id: str) -> None:
    """Clear cancel flag for a run (cleanup after completion)."""
    with _cancel_flags_lock:
        if run_id in _cancel_flags:
            del _cancel_flags[run_id]

class Orchestrator:
    """Main orchestration loop."""

    def __init__(self, cfg: OrchestratorConfig):
        if not cfg.api_key:
            raise RuntimeError("Missing GEMINI_API_KEY env var.")
        
        self.cfg = cfg
        self.client = genai.Client(api_key=cfg.api_key)
        self.memory = JsonMemory(cfg.memory_path)
        self.planner = Planner(self.client, cfg.model)
        self.coder = Coder(self.client, cfg.model)
        self.reviewer = Reviewer(self.client, cfg.model)
        
        # Ensure workspace root exists and set in tools registry
        os.makedirs(cfg.workspace_root, exist_ok=True)
        tools.set_workspace_root(cfg.workspace_root)
        
        # Initialize rate limiter for LLM calls
        self.rate_limiter = RateLimiter(
            max_requests_per_minute=cfg.llm_max_requests_per_minute,
            burst=cfg.llm_burst
        )
        
        self.state: Optional[OrchestratorState] = None
        self.run_id = str(uuid.uuid4())
        
        # Initialize per-run directory
        self.runs_root = os.path.join(_SCRIPT_DIR, "runs")
        os.makedirs(self.runs_root, exist_ok=True)
        
        # Initialize cancel flag for this run
        self.cancel_flag = get_cancel_flag(self.run_id)

    def _save_state(self) -> None:
        """Save orchestrator state to disk (both global and per-run)."""
        if self.state:
            # Save to global state.json
            os.makedirs(os.path.dirname(self.cfg.state_path) or ".", exist_ok=True)
            with open(self.cfg.state_path, "w", encoding="utf-8") as f:
                json.dump(self.state.to_dict(), f, indent=2)
            
            # Save to per-run directory
            run_dir = os.path.join(self.runs_root, self.run_id)
            os.makedirs(run_dir, exist_ok=True)
            run_state_path = os.path.join(run_dir, "state.json")
            with open(run_state_path, "w", encoding="utf-8") as f:
                json.dump(self.state.to_dict(), f, indent=2)
            
            event_bus.emit("state_saved", {"step": self.state.current_step}, self.run_id)

    def _save_memory_snapshot(self) -> None:
        """Save memory snapshot for this run to per-run directory."""
        run_dir = os.path.join(self.runs_root, self.run_id)
        os.makedirs(run_dir, exist_ok=True)
        snapshot_path = os.path.join(run_dir, "memory_snapshot.json")
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(self.memory.state, f, indent=2)

    def _get_state_dict(self) -> Dict[str, Any]:
        """Get current state as dict (for API responses)."""
        if self.state:
            return self.state.to_dict()
        return {}

    def _append_log(self, message: str) -> None:
        """Append a log message to state.logs and save."""
        if self.state:
            self.state.logs.append(message)
            # Cap logs to last 200 lines
            if len(self.state.logs) > 200:
                self.state.logs = self.state.logs[-200:]
            self.state.updated_at = time.time()
            self._save_state()
            logger.info(f"[Log] {message}")

    def _set_error(self, error_msg: str) -> None:
        """Set error status and message."""
        if self.state:
            self.state.status = "error"
            self.state.last_error = error_msg
            self.state.errors.append(error_msg)
            self.state.updated_at = time.time()
            # Append to logs
            trace = traceback.format_exc()
            if "Traceback" in trace:
                self.state.logs.append(f"ERROR: {error_msg}")
                self.state.logs.append(trace)
            self._save_state()
            logger.error(f"Orchestrator error: {error_msg}")

    def _set_state(self, **patch) -> None:
        """Update state fields and track update time."""
        if self.state:
            for key, val in patch.items():
                if hasattr(self.state, key):
                    setattr(self.state, key, val)
            self.state.updated_at = time.time()
            self._save_state()

    def run(self, goal: str) -> None:
        """
        Main orchestration loop.
        1. Emit run_started
        2. Generate plan
        3. Execute steps: planner -> router -> worker -> apply action -> next step
        4. Emit run_finished
        """
        try:
            # Only regenerate run_id if run() is being called again on the same instance
            # (indicated by state already being set)
            if self.state is not None:
                self.run_id = str(uuid.uuid4())
            self.state = OrchestratorState(
                run_id=self.run_id,
                goal=goal,
                plan={},
                status="running",
                last_run_at=time.time(),
                updated_at=time.time(),
                logs=[]
            )
            self._save_state()
            
            event_bus.emit(
                "run_started",
                {"goal": goal},
                self.run_id,
            )

            # Step 1: Generate plan
            log_msg = f"Goal: {goal}"
            self._append_log(log_msg)
            print(f"\n[Orchestrator] Run {self.run_id[:8]}...")
            print(f"[Orchestrator] {log_msg}")
            print(f"[Orchestrator] Generating plan...")
            
            event_bus.emit("plan_started", {}, self.run_id)
            
            plan_result = self.planner.generate_plan(goal, self.memory.snapshot())
            steps_list = plan_result.get("steps", [])
            
            # PHASE 2: Enforce max_steps
            if len(steps_list) > self.cfg.max_steps:
                self._append_log(f"Plan truncated from {len(steps_list)} to {self.cfg.max_steps} steps")
                steps_list = steps_list[:self.cfg.max_steps]
                plan_result["steps"] = steps_list
                logger.warning(f"Plan truncated: {len(plan_result.get('steps', []))} -> {self.cfg.max_steps}")
            
            total_steps = len(steps_list)
            self._set_state(plan=plan_result, total_steps=total_steps)
            self._append_log(f"Plan created with {total_steps} steps")
            self._save_memory_snapshot()
            
            event_bus.emit(
                "plan_created",
                {
                    "num_steps": total_steps,
                    "steps": plan_result.get("steps", []),
                },
                self.run_id,
            )

            # Step 2: Execute plan steps
            steps = plan_result.get("steps", [])
            for idx, step in enumerate(steps, 1):
                # Check for cancellation
                if self.cancel_flag.is_set():
                    self._append_log("Run cancelled by user")
                    event_bus.emit("run_cancelled", {}, self.run_id)
                    self._set_state(status="cancelled")
                    self._save_state()
                    return
                
                if not self.state or self.state.status not in ("running", "waiting_user"):
                    break
                
                step_id = f"step_{idx}"
                self._set_state(current_step=idx)
                
                step_desc = step.get("description", "")
                log_msg = f"Step {idx}/{total_steps}: {step_desc}"
                self._append_log(log_msg)
                print(f"\n[Step {idx}/{total_steps}] {step_desc}")
                
                event_bus.emit("step_started", {"step_num": idx, "description": step_desc}, self.run_id, step_id)
                
                # Route to worker
                worker_route = self._route_step(step_desc)
                print(f"  -> routed to: {worker_route}")
                self._append_log(f"  Routed to: {worker_route}")
                
                # Track last worker for final output promotion
                if self.state:
                    self.state.last_worker = worker_route
                
                event_bus.emit("worker_selected", {"worker": worker_route}, self.run_id, step_id)
                
                # Call worker
                context = self._build_context(step_desc)
                
                # Rate limiting: acquire token before LLM call
                acquired, wait_seconds = self.rate_limiter.acquire()
                while not acquired:
                    # Check for cancellation before sleeping
                    if self.cancel_flag.is_set():
                        self._append_log("Run cancelled while waiting for quota")
                        event_bus.emit("run_cancelled", {}, self.run_id)
                        self._set_state(status="cancelled")
                        self._save_state()
                        return
                    
                    # Emit rate_limited event
                    event_bus.emit(
                        "rate_limited",
                        {"wait_seconds": wait_seconds},
                        self.run_id,
                        step_id
                    )
                    # Set status to waiting_quota
                    self._set_state(status="waiting_quota")
                    log_msg = f"Rate limited by LLM quota, waiting {wait_seconds:.1f}s"
                    self._append_log(log_msg)
                    print(f"[Rate Limited] {log_msg}")
                    
                    # Sleep and retry (check cancel flag periodically)
                    sleep_remaining = wait_seconds
                    while sleep_remaining > 0:
                        if self.cancel_flag.is_set():
                            self._append_log("Run cancelled while waiting for quota")
                            event_bus.emit("run_cancelled", {}, self.run_id)
                            self._set_state(status="cancelled")
                            self._save_state()
                            return
                        sleep_duration = min(0.5, sleep_remaining)  # Check cancel every 500ms
                        time.sleep(sleep_duration)
                        sleep_remaining -= sleep_duration
                    
                    acquired, wait_seconds = self.rate_limiter.acquire()
                
                # Resume normal operation
                if self.state and self.state.status == "waiting_quota":
                    self._set_state(status="running")
                    self._append_log("Quota available, resuming")
                
                event_bus.emit("llm_request", {"worker": worker_route}, self.run_id, step_id)
                
                # PHASE 2: Per-step timeout (60 seconds)
                action = None
                try:
                    if worker_route == "planner":
                        action = self.planner.generate_plan(step_desc, context)
                    elif worker_route == "coder":
                        action = self.coder.process_step(step_desc, context, self.memory.snapshot())
                    else:  # reviewer
                        action = self.reviewer.process_step(step_desc, context, self.memory.snapshot())
                except json.JSONDecodeError as e:
                    # JSON parse failure - emit INVALID_WORKER_OUTPUT event
                    error_msg = f"Invalid JSON from {worker_route} worker: {str(e)}"
                    self._append_log(f"❌ {error_msg}")
                    self._set_error(error_msg)
                    logger.exception(f"JSON parse failed: {e}")
                    event_bus.emit("error", {
                        "reason": "INVALID_WORKER_OUTPUT",
                        "message": error_msg,
                        "type": "json_parse_error"
                    }, self.run_id, step_id)
                    break
                except Exception as e:
                    error_msg = f"Step {idx} failed: {str(e)}"
                    self._append_log(f"❌ {error_msg}")
                    self._set_error(error_msg)
                    logger.exception(f"Worker failed: {e}")
                    event_bus.emit("step_failed", {"reason": str(e)}, self.run_id, step_id)
                    break
                
                # Validate worker response
                is_valid, error_reason = validate_action(action)
                if not is_valid:
                    error_msg = f"Invalid worker output: {error_reason}"
                    self._append_log(f"❌ {error_msg}")
                    self._set_error(error_msg)
                    event_bus.emit("error", {"reason": "INVALID_WORKER_OUTPUT", "message": error_msg}, self.run_id, step_id)
                    break
                
                event_bus.emit("llm_response", {"worker": worker_route, "action_type": action.get("type")}, self.run_id, step_id)
                
                action_type = action.get("type")
                print(f"  -> action: {action_type}")
                self._append_log(f"  Action: {action_type}")
                
                # Apply action
                self._apply_action(action, step_id)
                
                # Check for user input needed - wait for response
                if self.state.status == "waiting_user":
                    self._append_log("Waiting for user input")
                    print(f"[Orchestrator] Waiting for user input: {self.state.notes}")
                    
                    # Wait for user response (with timeout of 1 hour = 3600 seconds)
                    self._wait_for_user_input(timeout_seconds=3600)
                    
                    # If response was provided, continue to next step
                    if self.state.status == "running" and self.state.user_response:
                        self._append_log(f"Received user input: {self.state.user_response[:100]}")
                        self.state.user_response = None  # Mark as consumed
                        self._save_state()
                        # Continue to next step (loop continues naturally)
                    else:
                        # Timeout or cancelled
                        break

            # Finish
            if self.state:
                if self.state.status == "running":
                    self.state.status = "completed"
                
                # Set final output based on status
                if self.state.final_output is None:
                    if self.state.status == "completed":
                        # If there's a meaningful summary (>20 chars), use it; else fallback
                        if self.state.last_summary and len(self.state.last_summary.strip()) > 20:
                            self.state.final_output = self.state.last_summary.strip()
                        else:
                            self.state.final_output = "Run completed"
                    elif self.state.status == "error":
                        self.state.final_output = self.state.last_error or "Run failed"
                    elif self.state.status == "cancelled":
                        self.state.final_output = "Cancelled by user"
                
                self._append_log("Run completed")
                self._save_state()
                self._save_memory_snapshot()
            
            print(f"\n[Orchestrator] Run finished.")
            event_bus.emit(
                "run_finished",
                {"status": self.state.status if self.state else "unknown"},
                self.run_id,
            )
        except Exception as e:
            logger.exception(f"Orchestrator failed: {e}")
            if self.state:
                # Set error final output
                if self.state.final_output is None:
                    self.state.final_output = str(e) or "Run failed"
                self._set_error(str(e))
                self._save_memory_snapshot()
            event_bus.emit("error", {"message": str(e), "traceback": traceback.format_exc()}, self.run_id)
        finally:
            # Cleanup: clear cancel flag
            clear_cancel_flag(self.run_id)

    def _route_step(self, step_description: str) -> str:
        """
        Route step to appropriate worker based on content.
        - Planner: generate, plan, outline, strategy
        - Coder: code, implement, file, function, write, create, modify
        - Reviewer: verify, check, test, edge-case, review, quality
        """
        lower = step_description.lower()
        
        if any(word in lower for word in ["code", "implement", "file", "function", "write", "create", "modify", "script"]):
            return "coder"
        elif any(word in lower for word in ["verify", "check", "test", "edge-case", "edge case", "review", "quality", "validate"]):
            return "reviewer"
        else:
            return "planner"

    def _build_context(self, step_description: str) -> str:
        """Build context info for the worker."""
        lines = [
            "CONTEXT",
            f"Current goal: {self.state.goal if self.state else '?'}",
            f"Step: {step_description}",
            f"Memory: {self.memory.snapshot()}",
        ]
        return "\n".join(lines)

    def _apply_action(self, action: Dict[str, Any], step_id: str) -> None:
        """Apply the action returned by the worker."""
        action_type = action.get("type")
        
        if action_type == "tool_call":
            self._apply_tool_call(action, step_id)
        elif action_type == "write_file":
            self._apply_write_file(action, step_id)
        elif action_type == "step_complete":
            self._apply_step_complete(action, step_id)
        elif action_type == "need_user":
            self._apply_need_user(action, step_id)
        elif action_type == "plan":
            self._apply_plan(action, step_id)
        else:
            event_bus.emit(
                "error",
                {"message": f"Unknown action type: {action_type}"},
                self.run_id,
                step_id,
            )

    def _apply_tool_call(self, action: Dict[str, Any], step_id: str) -> None:
        """Apply a tool call."""
        tool_name = action.get("name", "")
        tool_args = action.get("args", {})
        
        print(f"    [tool_call] {tool_name} {tool_args}")
        
        event_bus.emit("tool_started", {"name": tool_name}, self.run_id, step_id)
        
        result = tools.run(tool_name, tool_args)
        
        event_bus.emit("tool_finished", {"name": tool_name, "result": result[:200]}, self.run_id, step_id)
        
        # Special handling for "remember" tool
        if tool_name == "remember":
            kind = tool_args.get("kind", "note").lower()
            text = tool_args.get("text", "")
            if kind == "fact":
                self.memory.add_fact(text)
            else:
                self.memory.add_note(text)
            
            event_bus.emit("memory_saved", {"kind": kind, "text": text[:100]}, self.run_id, step_id)
        
        event_bus.emit("worker_action", {"type": "tool_call", "tool": tool_name}, self.run_id, step_id)

    def _apply_write_file(self, action: Dict[str, Any], step_id: str) -> None:
        """Apply a file write."""
        from tools import validate_path
        from datetime import datetime
        
        path = action.get("path", "")
        content = action.get("content", "")
        
        print(f"    [write_file] {path} ({len(content)} bytes)")
        
        # Validate path
        is_valid, error_reason = validate_path(path, self.cfg.workspace_root)
        if not is_valid:
            event_bus.emit(
                "error",
                {"message": f"PATH_VIOLATION: {error_reason}", "reason": "PATH_VIOLATION"},
                self.run_id,
                step_id,
            )
            return
        
        full_path = os.path.join(self.cfg.workspace_root, path)
        
        try:
            os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            event_bus.emit("file_written", {"path": path, "size": len(content)}, self.run_id, step_id)
            event_bus.emit("worker_action", {"type": "write_file", "path": path}, self.run_id, step_id)
            
            # Add artifact to state
            if self.state:
                ts_iso = datetime.utcnow().isoformat() + "Z"
                file_size = len(content)
                artifact = {"path": path, "size": file_size, "ts_iso": ts_iso, "kind": "file"}
                
                # Avoid duplicates: update existing or append new
                existing = next((a for a in self.state.artifacts if a["path"] == path), None)
                if existing:
                    existing["size"] = file_size
                    existing["ts_iso"] = ts_iso
                else:
                    self.state.artifacts.append(artifact)
            
            self.memory.add_note(f"Wrote file: {path}")
        except Exception as e:
            event_bus.emit(
                "error",
                {"message": f"Failed to write {path}: {e}"},
                self.run_id,
                step_id,
            )

    def _apply_step_complete(self, action: Dict[str, Any], step_id: str) -> None:
        """Apply step completion."""
        summary = action.get("summary", "")
        
        print(f"    [step_complete] {summary}")
        
        event_bus.emit("step_completed", {"summary": summary}, self.run_id, step_id)
        event_bus.emit("worker_action", {"type": "step_complete", "summary": summary}, self.run_id, step_id)
        
        # Store summary for final output
        if summary and self.state:
            self.state.last_summary = summary
        
        if summary:
            self.memory.add_note(summary)

    def _apply_need_user(self, action: Dict[str, Any], step_id: str) -> None:
        """Apply need_user (ask for clarification)."""
        question = action.get("question", "")
        why = action.get("why", "")
        
        print(f"    [need_user] {question}")
        
        if self.state:
            self.state.status = "waiting_user"
            self.state.notes.append(f"User input needed: {question}")
        
        event_bus.emit(
            "need_user",
            {"question": question, "why": why},
            self.run_id,
            step_id,
        )
        event_bus.emit("worker_action", {"type": "need_user", "question": question}, self.run_id, step_id)

    def _apply_plan(self, action: Dict[str, Any], step_id: str) -> None:
        """Apply plan action (when planner returns a plan during step execution)."""
        steps = action.get("steps", [])
        reasoning = action.get("reasoning", "")
        
        print(f"    [plan] {len(steps)} sub-steps, reasoning: {reasoning[:50]}")
        
        # Save plan details to memory
        plan_summary = f"Generated plan with {len(steps)} steps"
        if reasoning:
            plan_summary += f": {reasoning[:100]}"
        self.memory.add_note(plan_summary)
        
        # Emit event for UI visibility
        event_bus.emit("plan_generated", {"num_steps": len(steps)}, self.run_id, step_id)
        event_bus.emit("worker_action", {"type": "plan", "num_steps": len(steps)}, self.run_id, step_id)

    def provide_user_input(self, response: str) -> None:
        """Resume orchestration with user input."""
        if self.state:
            self.state.user_response = response
            self.state.status = "running"
            self._save_state()
            
            # Emit event with truncated response (max 200 chars)
            truncated = response[:200] if len(response) > 200 else response
            event_bus.emit("user_input", {"response": truncated}, self.run_id)
            
            # Record in memory
            self.memory.add_note(f"User responded: {response[:100]}")
    
    def _wait_for_user_input(self, timeout_seconds: int = 3600) -> None:
        """
        Wait for user input to be provided. Polls every 1 second.
        Timeout: 1 hour by default (prevents infinite wait).
        """
        import time
        elapsed = 0
        poll_interval = 1  # Check every 1 second
        
        while elapsed < timeout_seconds:
            # Check if user response has been provided
            if self.state and self.state.user_response:
                return  # Response provided, continue
            
            # Check if run was cancelled
            if self.cancel_flag.is_set():
                if self.state:
                    self.state.status = "cancelled"
                return
            
            # Sleep briefly before next check
            time.sleep(poll_interval)
            elapsed += poll_interval
        
        # Timeout reached - mark as error
        if self.state:
            self.state.status = "error"
            self.state.last_error = "User response timeout (1 hour)"



# Global orchestrator instance (created by server)
# CRITICAL: This MUST be updated by server.py when creating orchestrator
orchestrator: Optional[Orchestrator] = None

# Global state for idle/default
_DEFAULT_STATE = {
    "run_id": "",
    "goal": "",
    "plan": {},
    "current_step": 0,
    "total_steps": 0,
    "status": "idle",
    "user_response": None,
    "completed_steps": [],
    "errors": [],
    "notes": [],
    "logs": [],
    "last_error": "",
    "last_run_at": None,
    "updated_at": time.time(),
}


def get_state() -> Dict[str, Any]:
    """Get current orchestrator state. Always returns a complete state dict with defaults."""
    if orchestrator and orchestrator.state:
        return orchestrator.state.to_dict()
    # Return default idle state with current timestamp
    default = _DEFAULT_STATE.copy()
    default["updated_at"] = time.time()
    return default
