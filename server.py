"""
Server: FastAPI backend with WebSocket observability.
Serves the UI and real-time event streaming.
"""

import os
import asyncio
import json
import logging
import traceback
import uuid
from pathlib import Path
from typing import Set, Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import FileResponse
import uvicorn

from events import event_bus, Event
import orchestrator as orch_module
from orchestrator import Orchestrator, OrchestratorConfig, get_state, get_cancel_flag
import threading

# -------------------------
# Orchestrator Tracking
# -------------------------

# Global store for active orchestrator instances (per run_id)
_active_orchestrators_lock = threading.Lock()
_active_orchestrators: Dict[str, Orchestrator] = {}

def register_orchestrator(run_id: str, orch: Orchestrator) -> None:
    """Register an active orchestrator."""
    with _active_orchestrators_lock:
        _active_orchestrators[run_id] = orch

def unregister_orchestrator(run_id: str) -> None:
    """Unregister an orchestrator (cleanup after run)."""
    with _active_orchestrators_lock:
        if run_id in _active_orchestrators:
            del _active_orchestrators[run_id]

def get_active_orchestrator(run_id: str) -> Optional[Orchestrator]:
    """Retrieve an active orchestrator by run_id."""
    with _active_orchestrators_lock:
        return _active_orchestrators.get(run_id)


# -------------------------
# State Loading
# -------------------------

def get_run_state(run_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get state for a specific run.
    If run_id is None, returns current orchestrator state.
    If run_id is provided, loads from runs/<run_id>/state.json.
    """
    if run_id is None:
        # Return current orchestrator state
        return get_state()
    
    # Try to load from per-run directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_state_path = os.path.join(script_dir, "runs", run_id, "state.json")
    
    if os.path.exists(run_state_path):
        try:
            with open(run_state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load run state {run_id}: {e}")
            return {"status": "error", "error": f"Failed to load run state: {e}"}
    
    # If not found, return empty state
    return {"status": "unknown", "error": f"Run {run_id} not found"}

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("Server")


# -------------------------
# WebSocket Connection Manager
# -------------------------

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, event: Event):
        """Broadcast an event to all connected clients."""
        for connection in list(self.active_connections):
            try:
                await connection.send_json(event.to_dict())
            except Exception as e:
                logger.error(f"[WebSocket] Send error: {e}")
                self.disconnect(connection)


connection_manager = ConnectionManager()


# -------------------------
# Event Queue & Broadcasting
# -------------------------

event_queue: Optional[asyncio.Queue] = None
broadcast_task: Optional[asyncio.Task] = None
shutdown_event: Optional[asyncio.Event] = None


def on_event_emitted(event: Event):
    """Called when an event is emitted by the EventBus."""
    # Queue the event for async broadcast
    if event_queue:
        try:
            event_queue.put_nowait(event)
        except Exception as e:
            logger.exception(f"[EventBus] Queue error: {e}")


async def broadcast_queue_events():
    """Background task to broadcast queued events to WebSocket clients."""
    global event_queue, shutdown_event
    if event_queue is None:
        event_queue = asyncio.Queue()
    if shutdown_event is None:
        shutdown_event = asyncio.Event()
    
    logger.info("[Broadcast] Broadcast task started")
    while not shutdown_event.is_set():
        try:
            event = await asyncio.wait_for(event_queue.get(), timeout=1.0)
            await connection_manager.broadcast(event)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            logger.info("[Broadcast] Task cancelled, shutting down")
            break
        except Exception as e:
            logger.exception(f"[Broadcast] Error: {e}")
            await asyncio.sleep(0.5)
    logger.info("[Broadcast] Broadcast task stopped")


# Subscribe to event bus
event_bus.subscribe(on_event_emitted)


# -------------------------
# App Setup
# -------------------------

app = FastAPI(title="Orchestrator", description="Live orchestration with UI")


@app.on_event("startup")
async def startup_event():
    """Initialize event queue and start background broadcast task."""
    global event_queue, broadcast_task, shutdown_event
    if event_queue is None:
        event_queue = asyncio.Queue()
    if shutdown_event is None:
        shutdown_event = asyncio.Event()
    if broadcast_task is None:
        broadcast_task = asyncio.create_task(broadcast_queue_events())
        logger.info("[Server] Broadcast task started")


@app.on_event("shutdown")
async def shutdown_event_handler():
    """Gracefully shut down broadcast task."""
    global shutdown_event, broadcast_task
    if shutdown_event:
        shutdown_event.set()
    if broadcast_task:
        try:
            await asyncio.wait_for(broadcast_task, timeout=2.0)
        except asyncio.TimeoutError:
            broadcast_task.cancel()
        logger.info("[Server] Broadcast task stopped")


# -------------------------
# Routes
# -------------------------

@app.get("/")
async def get_ui():
    """Serve the main UI."""
    ui_path = Path(__file__).parent / "ui" / "index.html"
    if ui_path.exists():
        return FileResponse(ui_path, media_type="text/html")
    return {"error": "UI not found"}


@app.get("/api/state")
async def api_state(run_id: Optional[str] = None):
    """Get orchestrator state. Optionally specify run_id to load a past run."""
    return get_run_state(run_id)


@app.get("/api/events")
async def api_events(limit: int = 100, run_id: str = None):
    """Get recent events (newest first)."""
    return {
        "events": event_bus.get_events(limit=limit, run_id=run_id),
        "total": len(event_bus.events),
    }


@app.get("/api/run/{run_id}/events")
async def api_run_events(run_id: str):
    """Get all persisted events for a specific run from events.jsonl file."""
    import json
    from pathlib import Path
    
    events_file = Path("runs") / run_id / "events.jsonl"
    
    # Check if file exists
    if not events_file.exists():
        return {"ok": False, "error": f"No event log found for run {run_id[:8]}..."}
    
    # Read JSONL file
    events = []
    try:
        with open(events_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        event = json.loads(line)
                        events.append(event)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON in events.jsonl line: {e}")
    except Exception as e:
        return {"ok": False, "error": f"Failed to read event log: {str(e)}"}
    
    return {
        "ok": True,
        "run_id": run_id,
        "event_count": len(events),
        "events": events,
    }


@app.get("/api/run/{run_id}/events-download")
async def api_run_events_download(run_id: str):
    """Download JSONL event log file for a specific run."""
    from pathlib import Path
    from fastapi.responses import FileResponse
    
    events_file = Path("runs") / run_id / "events.jsonl"
    
    if not events_file.exists():
        return {"error": f"No event log found for run {run_id[:8]}..."}
    
    try:
        return FileResponse(
            events_file,
            media_type="application/jsonl",
            filename=f"events_{run_id[:8]}.jsonl"
        )
    except Exception as e:
        return {"error": f"Failed to download event log: {str(e)}"}


# -------------------------
# API Authentication
# -------------------------

API_TOKEN = os.getenv("API_TOKEN")  # Optional token from environment

def check_api_security(request_headers: Dict[str, str]) -> bool:
    """Check if request should be allowed (auth token validation)."""
    if not API_TOKEN:
        # No token configured - allow all
        return True
    # Token is configured - require it
    token = request_headers.get("x-api-token", "")
    return token == API_TOKEN


# -------------------------
# API Endpoints
# -------------------------

@app.post("/api/cancel")
async def api_cancel(request: Request, body: Dict[str, Any]):
    """Cancel a running orchestrator by run_id."""
    # Check authentication
    if not check_api_security(dict(request.headers)):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Token header")
    
    run_id = body.get("run_id", "").strip()
    
    if not run_id:
        return {"ok": False, "error": "Missing 'run_id' field"}
    
    try:
        cancel_flag = get_cancel_flag(run_id)
        cancel_flag.set()
        logger.info(f"[CANCEL] Run {run_id[:8]}... marked for cancellation")
        return {"ok": True, "message": f"Cancellation requested for run {run_id[:8]}..."}
    except Exception as e:
        logger.error(f"[CANCEL] Failed to cancel {run_id}: {e}")
        return {"ok": False, "error": str(e)}


@app.post("/api/respond")
async def api_respond(request: Request, body: Dict[str, Any]):
    """Provide user input to resume a waiting run."""
    # Check authentication
    if not check_api_security(dict(request.headers)):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Token header")
    
    run_id = body.get("run_id", "").strip()
    response = body.get("response", "")
    
    if not run_id:
        return {"ok": False, "error": "Missing 'run_id' field"}
    
    if response == "":
        return {"ok": False, "error": "Missing 'response' field"}
    
    try:
        # Get the active orchestrator
        orch = get_active_orchestrator(run_id)
        if not orch:
            return {"ok": False, "error": f"Run {run_id[:8]}... is not currently waiting for input"}
        
        # Check if run is waiting for user input
        if orch.state and orch.state.status != "waiting_user":
            return {"ok": False, "error": f"Run {run_id[:8]}... is not waiting for user input (status: {orch.state.status})"}
        
        # Provide the user input (this will update state and emit event)
        orch.provide_user_input(response)
        
        logger.info(f"[RESPOND] Run {run_id[:8]}... received user input")
        return {"ok": True, "run_id": run_id[:8], "status": "resumed"}
    except Exception as e:
        logger.error(f"[RESPOND] Failed to process response for {run_id}: {e}")
        return {"ok": False, "error": str(e)}


@app.post("/api/run")
async def api_run(request: Request, body: Dict[str, Any]):
    """
    Start a new orchestration run asynchronously.
    Returns: {"ok": true, "run_id": "...", "status": "started"}
    On error: {"ok": false, "error": "...", "hint": "...", "trace_id": "..."}
    """
    # Check authentication
    if not check_api_security(dict(request.headers)):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Token header")
    
    goal = body.get("goal", "").strip()
    if not goal:
        return {"ok": False, "error": "Missing 'goal' field", "hint": "Provide a 'goal' string", "trace_id": ""}
    
    trace_id = str(uuid.uuid4())[:8]
    
    try:
        # Get or create orchestrator
        cfg = OrchestratorConfig()
        orch = Orchestrator(cfg)
        
        # CRITICAL: Set the canonical global that get_state() reads
        orch_module.orchestrator = orch
        
        logger.info(f"[{trace_id}] Starting orchestrator run: {goal}")
        
        # Register orchestrator for user input / cancellation / etc
        run_id = orch.run_id
        register_orchestrator(run_id, orch)
        
        # Start run in background
        asyncio.create_task(run_background(orch, goal, trace_id))
        
        return {"ok": True, "run_id": run_id, "status": "started"}
    
    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        logger.error(f"[{trace_id}] Failed to start run: {error_msg}\n{tb}")
        
        return {
            "ok": False,
            "error": error_msg,
            "hint": "Check server logs for details. Ensure GEMINI_API_KEY is set.",
            "trace_id": trace_id,
        }


async def run_background(orch: Orchestrator, goal: str, trace_id: str = ""):
    """Run the orchestrator in a background task."""
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, orch.run, goal)
    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        logger.error(f"[{trace_id}] Background orchestrator failed: {error_msg}\n{tb}")
        
        # Update state to error
        if orch.state:
            orch.state.status = "error"
            orch.state.last_error = error_msg
            orch.state.logs.append(f"FATAL ERROR: {error_msg}")
            orch.state.logs.append(tb)
            orch._save_state()
        
        event_bus.emit("error", {"message": error_msg, "traceback": tb}, orch.run_id)
    finally:
        # Cleanup: unregister the orchestrator
        unregister_orchestrator(orch.run_id)


@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    """WebSocket endpoint for live event streaming."""
    try:
        await connection_manager.connect(websocket)
        logger.info(f"[WebSocket] Client connected. Total: {len(connection_manager.active_connections)}")
        
        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            # Can implement ping/pong or commands here
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
        logger.info(f"[WebSocket] Client disconnected. Total: {len(connection_manager.active_connections)}")
    except Exception as e:
        logger.exception(f"[WebSocket] Error: {e}")
        connection_manager.disconnect(websocket)


# -------------------------
# Note: Orchestrator instance is stored in orch_module.orchestrator (canonical global)
# Not maintained separately in this module to avoid synchronization issues
# -------------------------

# -------------------------
# Entry point
# -------------------------

if __name__ == "__main__":
    logger.info("[Server] Starting Orchestrator UI server...")
    logger.info("[Server] Open http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
