"""
Tools: Implementations for actions the agent can perform.
Each tool is a simple function that takes args and returns a string result.
"""

import os
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple


# -------------------------
# Rate Limiter (Token Bucket)
# -------------------------

class RateLimiter:
    """
    Token bucket rate limiter for LLM API calls.
    
    Configurable via environment variables:
    - LLM_MAX_REQUESTS_PER_MINUTE: max requests per minute (default: 5)
    - LLM_BURST: max burst tokens (default: 1)
    """
    
    def __init__(self, max_requests_per_minute: int = 5, burst: int = 1):
        """
        Initialize rate limiter with token bucket.
        
        Args:
            max_requests_per_minute: Sustainable request rate (e.g., 5 for free tier)
            burst: Max tokens to accumulate (allows bursts up to this count)
        """
        self.max_requests_per_minute = max_requests_per_minute
        self.burst = burst
        self.tokens = float(burst)  # Start with full burst capacity
        self.last_refill = time.time()
        self.refill_rate = max_requests_per_minute / 60.0  # tokens per second
    
    def acquire(self) -> Tuple[bool, float]:
        """
        Try to acquire a token for an API request.
        
        Returns:
            Tuple[bool, float]: (acquired: bool, wait_seconds: float)
            - (True, 0.0) if token acquired immediately
            - (False, wait_time) if rate limited; wait_time = seconds to wait before retry
        """
        now = time.time()
        time_passed = now - self.last_refill
        
        # Refill tokens based on time elapsed
        self.tokens = min(self.burst, self.tokens + time_passed * self.refill_rate)
        self.last_refill = now
        
        if self.tokens >= 1:
            self.tokens -= 1
            return True, 0.0
        else:
            # Calculate wait time to accumulate 1 token
            wait_seconds = (1 - self.tokens) / self.refill_rate
            return False, wait_seconds
    
    def reset(self) -> None:
        """Reset limiter to initial state."""
        self.tokens = float(self.burst)
        self.last_refill = time.time()




def validate_path(path: str, workspace_root: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that a path is safe for file operations.
    
    Rules:
    - No absolute paths
    - No path traversal (..)
    - Must resolve within workspace_root (defend against symlink escapes)
    
    Args:
        path: Path to validate
        workspace_root: Root directory for allowed operations
        
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_reason)
        - (True, None) if valid
        - (False, reason) if invalid
    """
    if not path or not path.strip():
        return False, "Empty path"
    
    path = path.strip()
    
    # Check for absolute path
    if os.path.isabs(path):
        return False, f"Absolute paths not allowed: {path}"
    
    # Check for path traversal (..)
    if ".." in path.split(os.sep):
        return False, f"Path traversal (..) not allowed: {path}"
    
    # Resolve paths to real paths (defend against symlink escapes)
    try:
        workspace_real = os.path.realpath(workspace_root)
        target_real = os.path.realpath(os.path.join(workspace_root, path))
        
        # Ensure target is under workspace
        if not target_real.startswith(workspace_real + os.sep) and target_real != workspace_real:
            return False, f"Path escapes workspace: {path} → {target_real}"
        
        return True, None
    except Exception as e:
        return False, f"Path resolution error: {e}"


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self, workspace_root: str = "./workspace"):
        self.tools: Dict[str, Callable] = {}
        self.workspace_root = workspace_root

    def set_workspace_root(self, workspace_root: str) -> None:
        """Update the workspace root for all tool operations."""
        self.workspace_root = workspace_root

    def register(self, name: str, fn: Callable) -> None:
        """Register a tool function."""
        self.tools[name] = fn

    def run(self, name: str, args: Dict[str, Any]) -> str:
        """Execute a registered tool. Returns result as string."""
        if name not in self.tools:
            return f"ERROR: unknown tool '{name}'. Available tools: {list(self.tools.keys())}"
        try:
            # Pass workspace_root to tool via args (tools check for it)
            args_with_workspace = {**args, "_workspace_root": self.workspace_root}
            result = self.tools[name](args_with_workspace)
            return str(result)
        except Exception as e:
            return f"ERROR executing tool '{name}': {e}"


# -------------------------
# Basic Tools
# -------------------------

def tool_remember(args: Dict[str, Any]) -> str:
    """Remember a fact or note (stored in memory.json by orchestrator)."""
    text = (args.get("text") or "").strip()
    kind = (args.get("kind") or "note").strip().lower()
    
    if not text:
        return "ERROR: 'text' field is required"
    
    if kind not in ("fact", "note"):
        return f"ERROR: 'kind' must be 'fact' or 'note', got '{kind}'"
    
    # Return success; orchestrator will handle storage
    return f"OK: memory recorded ({kind})"


def tool_inspect_directory(args: Dict[str, Any]) -> str:
    """Inspect a directory and list contents."""
    workspace_root = args.get("_workspace_root", "./workspace")
    path = (args.get("path") or ".").strip()
    
    # Validate path
    is_valid, error_reason = validate_path(path, workspace_root)
    if not is_valid:
        return f"ERROR: PATH_VIOLATION: {error_reason}"
    
    full_path = os.path.join(workspace_root, path)
    
    if not os.path.isdir(full_path):
        return f"ERROR: path '{path}' is not a directory"
    
    try:
        items = os.listdir(full_path)
        items.sort()
        
        result = []
        for item in items:
            item_path = os.path.join(full_path, item)
            if os.path.isdir(item_path):
                result.append(f"[DIR]  {item}")
            else:
                stat = os.stat(item_path)
                size = stat.st_size
                result.append(f"[FILE] {item} ({size} bytes)")
        
        return "\n".join(result) if result else "(empty directory)"
    except Exception as e:
        return f"ERROR inspecting '{path}': {e}"


def tool_read_file(args: Dict[str, Any]) -> str:
    """Read a file and return its contents."""
    workspace_root = args.get("_workspace_root", "./workspace")
    path = (args.get("path") or "").strip()
    
    if not path:
        return "ERROR: 'path' field is required"
    
    # Validate path
    is_valid, error_reason = validate_path(path, workspace_root)
    if not is_valid:
        return f"ERROR: PATH_VIOLATION: {error_reason}"
    
    full_path = os.path.join(workspace_root, path)
    
    if not os.path.isfile(full_path):
        return f"ERROR: file '{path}' not found"
    
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Truncate if too large
        max_len = 10000
        if len(content) > max_len:
            content = content[:max_len] + f"\n\n... ({len(content) - max_len} more bytes)"
        
        return content
    except Exception as e:
        return f"ERROR reading '{path}': {e}"


def tool_write_file(args: Dict[str, Any]) -> str:
    """Write content to a file. Note: orchestrator handles actual writing."""
    workspace_root = args.get("_workspace_root", "./workspace")
    path = (args.get("path") or "").strip()
    content = args.get("content") or ""
    
    if not path:
        return "ERROR: 'path' field is required"
    
    # Validate path
    is_valid, error_reason = validate_path(path, workspace_root)
    if not is_valid:
        return f"ERROR: PATH_VIOLATION: {error_reason}"
    
    # Return success; orchestrator will handle actual writing
    return f"OK: ready to write to '{path}' ({len(content)} bytes)"


def tool_list_files(args: Dict[str, Any]) -> str:
    """List Python files in the project with safety limits."""
    workspace_root = args.get("_workspace_root", "./workspace")
    root = (args.get("root") or ".").strip()
    
    # Validate root path
    is_valid, error_reason = validate_path(root, workspace_root)
    if not is_valid:
        return f"ERROR: PATH_VIOLATION: {error_reason}"
    
    full_root = os.path.join(workspace_root, root)
    max_files = 10000
    max_file_size_mb = 2
    
    if not os.path.isdir(full_root):
        return f"ERROR: root '{root}' is not a directory"
    
    # Directories to skip
    skip_dirs = {".git", "node_modules", ".venv", "venv", "dist", "build", "__pycache__", ".env", ".hg", ".svn"}
    
    try:
        py_files = []
        visited_inodes = set()  # Track visited inodes to detect symlink loops
        
        for dirpath, dirnames, filenames in os.walk(full_root, followlinks=False):
            # Get inode to detect loops
            try:
                stat = os.stat(dirpath)
                inode = (stat.st_dev, stat.st_ino)
                if inode in visited_inodes:
                    # Symlink loop detected - skip
                    dirnames[:] = []
                    continue
                visited_inodes.add(inode)
            except (OSError, PermissionError):
                # Can't stat, skip this dir
                continue
            
            # Skip directories
            dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]
            
            # Check file count limit
            if len(py_files) >= max_files:
                break
            
            for fn in filenames:
                if fn.endswith(".py"):
                    try:
                        full_path = os.path.join(dirpath, fn)
                        rel_path = os.path.relpath(full_path, full_root)
                        
                        # Check file size
                        stat = os.stat(full_path)
                        if stat.st_size > max_file_size_mb * 1024 * 1024:
                            # File too large, skip
                            continue
                        
                        py_files.append(rel_path)
                        
                        if len(py_files) >= max_files:
                            break
                    except (OSError, PermissionError) as e:
                        # Log but don't crash
                        print(f"[tool] Skipping file {full_path}: {e}")
                        continue
        
        py_files.sort()
        result = "\n".join(py_files)
        if len(py_files) >= max_files:
            result += f"\n\n(Truncated: reached limit of {max_files} files)"
        return result if result else "(no Python files found)"
    except Exception as e:
        return f"ERROR listing files in '{root}': {e}"


# -------------------------
# Tool Registry (singleton)
# -------------------------

tools = ToolRegistry()
tools.register("remember", tool_remember)
tools.register("inspect_directory", tool_inspect_directory)
tools.register("read_file", tool_read_file)
tools.register("write_file", tool_write_file)
tools.register("list_files", tool_list_files)
