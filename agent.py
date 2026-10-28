import os
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from genai_wrapper import Client
from google import genai


# -------------------------
# Config
# -------------------------
# Default memory path is in the same directory as this script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class AgentConfig:
    model: str = "gemini-2.5-flash"
    memory_path: str = os.path.join(_SCRIPT_DIR, "memory.json")
    max_turns: int = 6


# -------------------------
# Memory (simple, explicit)
# -------------------------
class JsonMemory:
    def __init__(self, path: str):
        self.path = path
        self.state = self._load()

    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {"facts": [], "notes": []}
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    def add_fact(self, text: str) -> None:
        self.state.setdefault("facts", []).append(text)
        self.save()

    def add_note(self, text: str) -> None:
        self.state.setdefault("notes", []).append(text)
        self.save()

    def snapshot(self) -> str:
        # Keep it short so you can “scan” it fast.
        facts = self.state.get("facts", [])[-6:]
        notes = self.state.get("notes", [])[-6:]
        return (
            "MEMORY SNAPSHOT\n"
            f"- Facts: {facts}\n"
            f"- Notes: {notes}\n"
        )


# -------------------------
# Tools (1-step functions)
# -------------------------
ToolFn = Callable[[Dict[str, Any], JsonMemory], str]

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, ToolFn] = {}

    def register(self, name: str, fn: ToolFn) -> None:
        self.tools[name] = fn

    def run(self, name: str, args: Dict[str, Any], memory: JsonMemory) -> str:
        if name not in self.tools:
            return f"ERROR: unknown tool '{name}'"
        return self.tools[name](args, memory)


# Example tool: remember()
def tool_remember(args: Dict[str, Any], memory: JsonMemory) -> str:
    text = (args.get("text") or "").strip()
    kind = (args.get("kind") or "note").strip().lower()
    if not text:
        return "ERROR: 'text' is required"
    if kind == "fact":
        memory.add_fact(text)
        return "Saved as fact."
    memory.add_note(text)
    return "Saved as note."


# -------------------------
# Agent (explicit loop)
# -------------------------
class MiniAgent:
    def __init__(self, cfg: AgentConfig):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY env var.")
        self.cfg = cfg
        self.client = genai.Client(api_key=api_key)  # official client :contentReference[oaicite:2]{index=2}
        self.memory = JsonMemory(cfg.memory_path)
        self.tools = ToolRegistry()
        self.tools.register("remember", tool_remember)

    def _system(self) -> str:
        return (
            "You are a compact personal partner agent.\n"
            "Rules:\n"
            "1) Be brief, scannable.\n"
            "2) If you need to store something, output a TOOL_CALL.\n"
            "3) TOOL_CALL format:\n"
            "TOOL_CALL {\"name\":\"remember\",\"args\":{\"kind\":\"note|fact\",\"text\":\"...\"}}\n"
            "4) Otherwise, output normal text.\n"
        )

    def _prompt(self, user_text: str) -> str:
        return (
            f"{self._system()}\n\n"
            f"{self.memory.snapshot()}\n\n"
            f"USER:\n{user_text}\n"
        )

    def chat(self, user_text: str) -> None:
        text = user_text
        for _ in range(self.cfg.max_turns):
            resp = self.client.models.generate_content(
                model=self.cfg.model,
                contents=self._prompt(text),
            )
            out = (resp.text or "").strip()

            if out.startswith("TOOL_CALL"):
                # Parse the JSON after TOOL_CALL
                try:
                    payload = out[len("TOOL_CALL"):].strip()
                    call = json.loads(payload)
                    name = call.get("name")
                    args = call.get("args") or {}
                    tool_result = self.tools.run(name, args, self.memory)
                    text = f"Tool result: {tool_result}\nContinue."
                    continue
                except Exception as e:
                    print(f"[tool_parse_error] {e}\n{out}")
                    return

            print(out)
            return

        print("Reached max_turns without a final answer.")


if __name__ == "__main__":
    agent = MiniAgent(AgentConfig())
    while True:
        try:
            user = input("\nYou> ").strip()
            if user.lower() in {"exit", "quit"}:
                break
            agent.chat(user)
        except KeyboardInterrupt:
            break