"""
EventBus: Ring buffer for real-time observability of orchestrator execution.
Stores the last 2000 events and broadcasts to WebSocket clients.
Also persists events to per-run JSONL logs in runs/<run_id>/events.jsonl.
"""

import json
import uuid
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import deque
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Event:
    """Immutable event record."""
    id: str  # UUID
    ts_iso: str  # ISO 8601 timestamp
    run_id: str  # Associated run ID
    step_id: Optional[str]  # Associated step ID (nullable)
    type: str  # Event type (run_started, step_started, worker_action, etc.)
    payload: Dict[str, Any]  # Event-specific data

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EventBus:
    """
    Central event bus with ring buffer (last 2000 events).
    Emit events and they're stored + broadcast to subscribers.
    """

    def __init__(self, max_events: int = 2000):
        self.max_events = max_events
        self.events: deque = deque(maxlen=max_events)
        self.subscribers: List[Callable] = []

    def _persist_event_to_file(self, event: Event) -> None:
        """
        Persist event to runs/<run_id>/events.jsonl file.
        Safe: creates directory if needed, appends to file.
        Failures are logged but don't crash the run.
        """
        try:
            # Create runs/<run_id> directory if it doesn't exist
            runs_dir = Path("runs") / event.run_id
            runs_dir.mkdir(parents=True, exist_ok=True)
            
            # Append event as JSON line to events.jsonl
            events_file = runs_dir / "events.jsonl"
            with open(events_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
                f.flush()  # Ensure write is synced
        except Exception as e:
            # Log persistence failure but don't crash
            print(f"[EventBus] Failed to persist event to runs/{event.run_id}/events.jsonl: {e}")
            # Emit a warning event about the persistence failure
            try:
                warning_event = Event(
                    id=str(uuid.uuid4()),
                    ts_iso=datetime.utcnow().isoformat() + "Z",
                    run_id=event.run_id,
                    step_id=None,
                    type="event_persistence_failed",
                    payload={"original_event_type": event.type, "error": str(e)},
                )
                # Append warning to buffer (but don't persist recursively)
                self.events.append(warning_event)
                # Notify subscribers of warning
                for callback in self.subscribers:
                    try:
                        callback(warning_event)
                    except Exception as callback_err:
                        print(f"[EventBus] Subscriber error on warning event: {callback_err}")
            except Exception as warning_err:
                print(f"[EventBus] Failed to emit warning for persistence failure: {warning_err}")

    def subscribe(self, callback: Callable[[Event], None]) -> None:
        """Subscribe to event emissions. Callback is called immediately when event is emitted."""
        self.subscribers.append(callback)

    def emit(
        self,
        event_type: str,
        payload: Dict[str, Any],
        run_id: str,
        step_id: Optional[str] = None,
    ) -> Event:
        """
        Emit an event. Returns the full Event object.
        
        Args:
            event_type: Type of event (run_started, step_started, worker_action, etc.)
            payload: Event-specific data dictionary
            run_id: ID of the associated orchestrator run
            step_id: Optional step ID within the run
            
        Returns:
            Event: The emitted event object
        """
        event = Event(
            id=str(uuid.uuid4()),
            ts_iso=datetime.utcnow().isoformat() + "Z",
            run_id=run_id,
            step_id=step_id,
            type=event_type,
            payload=payload,
        )
        self.events.append(event)
        
        # Persist event to run-specific JSONL file
        self._persist_event_to_file(event)
        
        # Notify subscribers
        for callback in self.subscribers:
            try:
                callback(event)
            except Exception as e:
                print(f"[EventBus] Subscriber error: {e}")
        
        return event

    def get_events(self, limit: Optional[int] = None, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get events from the buffer (most recent first).
        
        Args:
            limit: Max number of events to return (default: all)
            run_id: Filter by run_id (default: all runs)
            
        Returns:
            List of event dicts ordered newest to oldest
        """
        events = list(self.events)
        
        # Filter by run_id if specified
        if run_id:
            events = [e for e in events if e.run_id == run_id]
        
        # Reverse to newest first
        events.reverse()
        
        # Limit
        if limit:
            events = events[:limit]
        
        return [e.to_dict() for e in events]

    def clear(self) -> None:
        """Clear all events from the buffer."""
        self.events.clear()


# Global event bus instance
event_bus = EventBus()
