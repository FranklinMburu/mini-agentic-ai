# Mini Agentic AI - Multi-Worker LLM Orchestration System

> **⚠️ Copyright Notice**: This project is protected intellectual property. It is shared for recruitment, portfolio evaluation, and educational purposes only. Unauthorized copying, modification, or commercial use is strictly prohibited. See [LICENSE](LICENSE) for full terms.

A lightweight Python orchestration engine that coordinates specialized AI workers (Planner, Coder, Reviewer) to solve complex tasks. Built with FastAPI, WebSocket real-time event streaming, and Google Generative AI.

## Overview

This system implements a multi-agent architecture where specialized workers collaborate to break down goals into actionable steps, implement solutions, and verify results. The orchestrator manages workflow state, routes tasks to appropriate workers, executes tools deterministically, and streams all execution events in real-time to a web dashboard.

### Key Characteristics

- **Modular Worker Design**: Each worker (Planner, Coder, Reviewer) has specific responsibilities
- **Real-time Event Streaming**: WebSocket-based event bus for live execution monitoring
- **Strict Action Validation**: All LLM outputs are validated against expected JSON schemas before execution
- **State Persistence**: Automatic state snapshots after each operation for recovery
- **Rate Limited API Calls**: Token bucket rate limiter protects against quota exhaustion
- **Sandboxed Execution**: File operations confined to workspace boundary

## Architecture

```
User Goal
    ↓
┌─────────────────────────────────────┐
│ Orchestrator (Main Control Loop)    │
│  - Coordinate workers                │
│  - Execute tools                    │
│  - Emit events                      │
│  - Persist state                    │
└─────────────────────────────────────┘
    ↓    ↓    ↓
 Planner  Coder  Reviewer
    ↓    ↓    ↓
┌────────────────────────────────────┐
│ Google Generative AI (Gemini)      │
└────────────────────────────────────┘
    ↓
┌────────────────────────────────────┐
│ Tools (File I/O, Memory, State)    │
└────────────────────────────────────┘
    ↓
WebSocket Event Bus
    ↓
Live Dashboard UI
```

## Core Components

### Orchestrator (`orchestrator.py`)
Main control loop that:
- Parses user goals and generates execution plans
- Routes steps to appropriate workers
- Validates worker outputs
- Executes tools with proper error handling
- Saves state after each operation
- Broadcasts events to all connected clients

### Workers (`workers.py`)
Three specialized agents:

**Planner**
- Breaks down goals into step-by-step plans (≤8 steps)
- Generates reasoning for task approach
- Validates plan structure

**Coder**
- Implements solutions based on plan steps
- Generates file contents
- Creates tool call requests
- Handles implementation details

**Reviewer**
- Verifies work and outputs
- Checks for correctness and completeness
- Identifies edge cases or issues
- Approves step completion

### Tools (`tools.py`)
Executable operations:
- `write_file`: Create or modify files (sandboxed to workspace)
- `read_file`: Read file contents
- `remember`: Store facts in persistent memory
- `get_memory`: Retrieve stored facts
- Rate limiting for API calls

### Event System (`events.py`)
- Event bus with ring buffer (stores last 2000 events)
- Event types: run_started, step_started, llm_request, tool_executed, etc.
- Real-time broadcast via WebSocket to all connected clients

### Server (`server.py`)
FastAPI application with:
- REST endpoints for run management
- WebSocket endpoint for event streaming
- Static file serving for UI
- State and status query endpoints

## Installation

### Requirements
- Python 3.8+
- Google Generative AI API key (free tier available)

### Setup Steps

1. **Clone repository and navigate to project**
   ```bash
   cd mini_agent
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Google API key**
   ```bash
   export GEMINI_API_KEY="your-api-key"
   ```
   Get free key at: https://aistudio.google.com/app/apikey

## Running the System

### Start Server
```bash
source venv/bin/activate
python server.py
```

Output:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Access Dashboard
Open browser to: **http://localhost:8000**

## Usage

### Starting a Run

**Option 1: Web Dashboard**
1. Enter your goal in the text input
2. Click "Start Run"
3. Watch execution timeline in real-time

**Option 2: REST API**
```bash
curl -X POST http://localhost:8000/api/run \
  -H "Content-Type: application/json" \
  -d '{"goal": "Create a Python script that counts files in current directory"}'
```

### Monitoring Execution

The event timeline shows all operations in reverse chronological order:
- `run_started` - Execution begins
- `plan_created` - Steps generated
- `step_started` - Step processing begins
- `llm_request` - Worker called
- `llm_response` - Worker returns action
- `tool_started` / `tool_finished` - Tool execution
- `step_completed` - Step finishes
- `run_finished` - Execution complete

Click any event to view full JSON payload.

### Cancelling Execution

**Via Dashboard**: Click "Cancel Run" button (appears when running)

**Via API**:
```bash
curl -X POST http://localhost:8000/api/cancel \
  -H "Content-Type: application/json" \
  -d '{"run_id": "abc123ab"}'
```

## Configuration

### Environment Variables

```bash
GEMINI_API_KEY                  # Google API key (required)
LLM_MAX_REQUESTS_PER_MINUTE     # Rate limit (default: 5)
LLM_BURST                       # Burst capacity (default: 1)
```

### Configuration in Code

Edit `orchestrator.py` OrchestratorConfig:
- `model`: LLM model name (default: "gemini-2.5-flash")
- `max_steps`: Max execution steps (default: 50)
- `workspace_root`: Sandbox directory for file operations

## API Reference

### Endpoints

**Start Run**
```
POST /api/run
{
  "goal": "Your task description"
}
```

**Get Status**
```
GET /api/status/{run_id}
```

**Cancel Run**
```
POST /api/cancel
{
  "run_id": "run_identifier"
}
```

**Stream Events (WebSocket)**
```
WS /api/events/{run_id}
```

## File Structure

```
mini_agent/
├── orchestrator.py         # Main event loop & control logic
├── workers.py              # Planner, Coder, Reviewer implementations
├── tools.py                # Tool registry & execution
├── events.py               # Event bus system
├── server.py               # FastAPI server
├── agent.py                # Agent initialization
├── genai_wrapper.py        # Google Generative AI wrapper
├── requirements.txt        # Python dependencies
├── state.json              # Persisted orchestrator state
├── memory.json             # Persistent memory store
├── workspace/              # Sandbox directory for file operations
└── docs/                   # Architecture documentation
```

## Data Formats

### Worker Response (Validated JSON)

Workers must return strict JSON with action type:

```json
{
  "type": "plan",
  "steps": [
    {"description": "First step"},
    {"description": "Second step"}
  ]
}
```

```json
{
  "type": "tool_call",
  "name": "write_file",
  "args": {
    "path": "output.py",
    "content": "print('hello')"
  }
}
```

```json
{
  "type": "step_complete",
  "summary": "Step completed successfully",
  "next": "continue"
}
```

### State File (state.json)

Auto-saved after each operation:

```json
{
  "run_id": "abc123ab",
  "goal": "Task description",
  "status": "running",
  "current_step": 2,
  "plan": {...},
  "completed_steps": [...],
  "errors": []
}
```

## How It Works - Step by Step

### 1. Plan Generation
- User provides goal
- Planner generates plan with steps and reasoning
- Plan validated for structure and step count

### 2. Step Routing
- For each step, determine which worker should execute it
- Route keywords determine worker:
  - "code", "implement", "write", "create" → Coder
  - "verify", "check", "test", "review" → Reviewer
  - Other → Planner (for sub-tasks or analysis)

### 3. Worker Execution
- Selected worker receives step context and memories
- Worker generates JSON response
- Response validated against expected schema

### 4. Action Execution
- Validated action executed via tool registry
- Tool result captured and logged
- Event emitted with result

### 5. State Persistence
- State saved to `state.json`
- Memories updated in `memory.json`
- Event broadcast to all WebSocket clients

### 6. Completion
- Steps continue until plan complete or error
- Final state saved
- `run_finished` event emitted

## Error Handling

- **Invalid JSON from LLM**: Logged and rejected, run stops
- **Failed tool execution**: Logged in state, reviewer notified
- **API quota exceeded**: Rate limiter applies backoff
- **Path traversal attempts**: Blocked with validation error
- **Disconnected clients**: WebSocket closes gracefully

## Performance Characteristics

- **Max execution steps**: 50 (configurable)
- **Plan size**: Up to 8 steps per plan
- **Event buffer**: Last 2000 events kept in memory
- **State persistence**: ~1-10ms per save (disk I/O)
- **WebSocket broadcast**: <100ms to all connected clients
- **Typical task duration**: 30 seconds - 5 minutes depending on plan complexity

## Limitations

- Single orchestrator instance per process
- Plans limited to 8 steps (can be adjusted in code)
- Workspace file operations only (no shell execution)
- Free tier API: ~5 requests per minute
- Python 3.8+ compatibility

## Technologies

- **FastAPI** - Modern async web framework
- **WebSocket** - Real-time bi-directional communication
- **Google Generative AI (Gemini)** - LLM backbone
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation
- **asyncio** - Async orchestration

## Deployment Notes

- Runs on Linux, macOS, Windows (with venv setup)
- Single-threaded event loop (asyncio)
- No external database required (uses JSON files)
- Memory-resident event bus (2000 event limit)
- Suitable for single-user or small-team use

## Testing

Run unit tests:
```bash
pytest test_*.py -v
```

## Future Enhancements

- Database backend for execution history
- Multiple LLM provider support (OpenAI, Anthropic, etc.)
- Tool composition and conditional branching
- User approval workflows for sensitive operations
- Distributed worker execution
- Fine-tuned specialized models

## License

MIT

## Author

Franklin

---

**A transparent, observable, Python-native AI orchestration system built for understanding and control.**
