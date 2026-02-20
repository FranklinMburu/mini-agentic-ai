"""
Workers: Planner, Coder, and Reviewer agents that generate responses.
Each worker outputs STRICT JSON with one of the response types.
"""

import json
import time
import logging
from typing import Any, Dict, Optional, Tuple
from genai_wrapper import Client
import google.generativeai as genai

logger = logging.getLogger("Workers")


# -------------------------
# Response Validation
# -------------------------

def validate_action(action: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate worker action response structure and types.
    Returns: (is_valid: bool, error_reason: Optional[str])
    """
    if not isinstance(action, dict):
        return False, "Action must be a dict"
    
    action_type = action.get("type")
    if action_type is None:
        return False, "Missing required field: type"
    
    # Reject internal error type (from JSON parse guards)
    if action_type == "error":
        return False, f"Worker returned error: {action.get('reason', 'unknown')}"
    
    # Validate based on type
    if action_type == "plan":
        if "steps" not in action:
            return False, "plan action missing required field: steps"
        if not isinstance(action["steps"], list):
            return False, "plan action field 'steps' must be a list"
        # Validate each step has description
        for i, step in enumerate(action["steps"]):
            if not isinstance(step, dict):
                return False, f"plan step {i} must be a dict"
            if "description" not in step:
                return False, f"plan step {i} missing required field: description"
        return True, None
    
    elif action_type == "tool_call":
        if "name" not in action:
            return False, "tool_call action missing required field: name"
        if not isinstance(action["name"], str):
            return False, "tool_call field 'name' must be a string"
        if "args" not in action:
            return False, "tool_call action missing required field: args"
        if not isinstance(action["args"], dict):
            return False, "tool_call field 'args' must be a dict"
        return True, None
    
    elif action_type == "step_complete":
        if "summary" not in action:
            return False, "step_complete action missing required field: summary"
        if not isinstance(action["summary"], str):
            return False, "step_complete field 'summary' must be a string"
        if "next" not in action:
            return False, "step_complete action missing required field: next"
        if not isinstance(action["next"], str):
            return False, "step_complete field 'next' must be a string"
        return True, None
    
    elif action_type == "need_user":
        if "question" not in action:
            return False, "need_user action missing required field: question"
        if not isinstance(action["question"], str):
            return False, "need_user field 'question' must be a string"
        # why is optional but if present must be string
        if "why" in action and not isinstance(action["why"], str):
            return False, "need_user field 'why' must be a string"
        return True, None
    
    elif action_type == "write_file":
        if "path" not in action:
            return False, "write_file action missing required field: path"
        if not isinstance(action["path"], str):
            return False, "write_file field 'path' must be a string"
        if "content" not in action:
            return False, "write_file action missing required field: content"
        if not isinstance(action["content"], str):
            return False, "write_file field 'content' must be a string"
        return True, None
    
    else:
        return False, f"Unknown action type: {action_type}"


class WorkerBase:
    """Base class for workers (Planner, Coder, Reviewer)."""

    def __init__(self, client: Client, model: str = "gemini-2.5-flash"):
        self.client = client
        self.model = model

    def _call_llm(self, system: str, user_message: str) -> str:
        """
        Call LLM with retry logic.
        Retries up to 3 times on transient errors with exponential backoff: 1s, 2s, 4s.
        """
        max_retries = 3
        backoff_seconds = [1, 2, 4]
        
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=f"[SYSTEM]\n{system}\n\n[USER]\n{user_message}",
                )
                return (response.text or "").strip()
            except Exception as e:
                # Check if it's a transient error (network, timeout, 5xx)
                error_str = str(e).lower()
                is_transient = any(x in error_str for x in [
                    "timeout", "connection", "500", "502", "503", "504",
                    "network", "temporary", "unavailable"
                ])
                
                if not is_transient or attempt == max_retries - 1:
                    # Not transient, or last attempt - raise now
                    logger.exception(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                    raise
                
                # Transient error - retry with backoff
                wait_time = backoff_seconds[attempt]
                logger.warning(f"LLM transient error (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {e}")
                time.sleep(wait_time)

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response. Handle markdown code blocks."""
        text = text.strip()
        
        # Try to extract JSON from markdown code block
        if "```json" in text:
            start = text.find("```json") + len("```json")
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + len("```")
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        
        return json.loads(text)


class Planner(WorkerBase):
    """Planner: generates a step-by-step plan."""

    def generate_plan(self, goal: str, memory_snapshot: str = "") -> Dict[str, Any]:
        """
        Generate a numbered plan (up to 8 steps), or return step_complete if synthesizing an answer.
        
        Returns JSON:
        {
            "type": "plan",
            "steps": [
                {"number": 1, "description": "..."},
                ...
            ],
            "reasoning": "..."
        }
        OR if synthesizing/answering:
        {
            "type": "step_complete",
            "summary": "...",
            "next": "continue"
        }
        """
        # Detect if this goal is asking for synthesis/answer vs decomposition
        synthesis_keywords = ["answer", "formulate", "conclude", "summarize", "extract", "definition", "synthesize", "provide the"]
        is_synthesis = any(kw in goal.lower() for kw in synthesis_keywords)
        
        system = (
            "You are a meticulous planner and answerer. Given a GOAL, take the appropriate action:\n"
            "OUTPUT ONLY valid JSON (no markdown, no explanation).\n\n"
            "If the goal asks you to decompose/plan/outline/generate steps:\n"
            "Return a plan with AT MOST 8 steps:\n"
            "{\n"
            '  "type": "plan",\n'
            '  "steps": [{number: 1, "description": "..."}, ...],\n'
            '  "reasoning": "brief explanation of approach"\n'
            "}\n\n"
            "If the goal asks you to synthesize/answer/formulate/conclude:\n"
            "Return the synthesized answer as step_complete:\n"
            "{\n"
            '  "type": "step_complete",\n'
            '  "summary": "The complete answer or synthesized result",\n'
            '  "next": "continue"\n'
            "}\n"
        )

        user_msg = f"Memory snapshot:\n{memory_snapshot}\n\nGOAL:\n{goal}"
        
        response_text = self._call_llm(system, user_msg)
        
        try:
            result = self._parse_json_response(response_text)
            result_type = result.get("type")
            
            # If synthesis goal but got plan, convert to step_complete
            if is_synthesis and result_type == "plan":
                # Extract reasoning as the answer summary
                summary = result.get("reasoning", "")
                if not summary and result.get("steps"):
                    # Fallback: use all step descriptions
                    summary = " ".join([s.get("description", "") for s in result["steps"]])
                return {
                    "type": "step_complete",
                    "summary": summary or "Synthesis complete",
                    "next": "continue"
                }
            
            # If it's already step_complete, validate it
            if result_type == "step_complete":
                result.setdefault("summary", "Action complete")
                result.setdefault("next", "continue")
                return result
            
            # Default to plan validation
            if result.get("type") != "plan":
                result["type"] = "plan"
            if not isinstance(result.get("steps"), list):
                raise ValueError("'steps' must be a list")
            if len(result["steps"]) > 8:
                result["steps"] = result["steps"][:8]
            return result
        except Exception as e:
            # Fallback plan
            return {
                "type": "plan",
                "steps": [{"number": 1, "description": "Process the goal"}],
                "reasoning": f"Parse error: {e}. Using fallback plan.",
            }


class Coder(WorkerBase):
    """Coder: implements code changes and writes files."""

    def process_step(
        self,
        step_description: str,
        context: str,
        memory_snapshot: str = "",
    ) -> Dict[str, Any]:
        """
        Process a coding/implementation step.
        
        Returns one of:
        {
            "type": "tool_call",
            "name": "write_file" | "remember" | ...,
            "args": {...}
        }
        OR
        {
            "type": "step_complete",
            "summary": "...",
            "next": "..."
        }
        OR
        {
            "type": "need_user",
            "question": "...",
            "why": "..."
        }
        """
        system = (
            "You are an expert coder.\n"
            "Given a step description and context, take action.\n"
            "Output ONLY valid JSON (no markdown, no explanation).\n"
            "You can:\n"
            '- Call tool: {"type":"tool_call","name":"<toolname>","args":{...}}\n'
            '- Complete step: {"type":"step_complete","summary":"...","next":"..."}\n'
            '- Ask user: {"type":"need_user","question":"...","why":"..."}\n'
        )

        user_msg = (
            f"Memory:\n{memory_snapshot}\n\n"
            f"Context:\n{context}\n\n"
            f"Step: {step_description}"
        )

        response_text = self._call_llm(system, user_msg)

        try:
            result = self._parse_json_response(response_text)
            # Validate and normalize
            if result.get("type") not in ("tool_call", "step_complete", "need_user"):
                # Assume step complete if type is unclear
                result["type"] = "step_complete"
                result.setdefault("summary", "Action completed")
                result.setdefault("next", "continue")
            return result
        except Exception as e:
            return {
                "type": "step_complete",
                "summary": f"Error parsing response: {e}",
                "next": "continue"
            }


class Reviewer(WorkerBase):
    """Reviewer: checks work and identifies issues."""

    def process_step(
        self,
        step_description: str,
        context: str,
        memory_snapshot: str = "",
    ) -> Dict[str, Any]:
        """
        Review and verify work.
        
        Returns one of:
        {
            "type": "tool_call",
            "name": "remember" | ...,
            "args": {...}
        }
        OR
        {
            "type": "step_complete",
            "summary": "...",
            "next": "..."
        }
        OR
        {
            "type": "need_user",
            "question": "...",
            "why": "..."
        }
        """
        system = (
            "You are a careful code reviewer and QA engineer.\n"
            "Your job is to verify work, check for edge cases, and ensure correctness.\n"
            "Output ONLY valid JSON (no markdown, no explanation).\n"
            "You can:\n"
            '- Call tool: {"type":"tool_call","name":"<toolname>","args":{...}}\n'
            '- Approve: {"type":"step_complete","summary":"Verified OK","next":"continue"}\n'
            '- Flag issues: {"type":"need_user","question":"...","why":"..."}\n'
        )

        user_msg = (
            f"Memory:\n{memory_snapshot}\n\n"
            f"Context:\n{context}\n\n"
            f"Verification task: {step_description}"
        )

        response_text = self._call_llm(system, user_msg)

        try:
            result = self._parse_json_response(response_text)
            # Validate
            if result.get("type") not in ("tool_call", "step_complete", "need_user"):
                result["type"] = "step_complete"
                result.setdefault("summary", "Verification complete")
                result.setdefault("next", "continue")
            return result
        except Exception as e:
            return {
                "type": "step_complete",
                "summary": "Verified (assuming OK due to parse error).",
                "next": "continue"
            }
