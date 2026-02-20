"""
Minimal unit tests for filesystem sandbox path validation.

Tests cover:
1. Normal path (valid)
2. Path traversal (..) attempt (invalid)
3. Absolute path attempt (invalid)
4. Symlink escape attempt (invalid)
5. Nested valid path (valid)
"""

import os
import tempfile
import pytest
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import validate_path


class TestPathValidation:
    """Test suite for filesystem sandbox validation."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = os.path.join(tmpdir, "workspace")
            os.makedirs(workspace_root, exist_ok=True)
            
            # Create some test files
            os.makedirs(os.path.join(workspace_root, "nested", "dir"), exist_ok=True)
            Path(os.path.join(workspace_root, "test.txt")).touch()
            Path(os.path.join(workspace_root, "nested", "file.py")).touch()
            
            yield workspace_root

    def test_normal_path_is_valid(self, temp_workspace):
        """Test 1: Normal path within workspace is valid."""
        is_valid, error = validate_path("test.txt", temp_workspace)
        assert is_valid is True
        assert error is None

    def test_path_traversal_rejected(self, temp_workspace):
        """Test 2: Path traversal (..) is rejected."""
        is_valid, error = validate_path("../etc/passwd", temp_workspace)
        assert is_valid is False
        assert "Path traversal" in error or ".." in error

    def test_absolute_path_rejected(self, temp_workspace):
        """Test 3: Absolute paths are rejected."""
        is_valid, error = validate_path("/etc/passwd", temp_workspace)
        assert is_valid is False
        assert "Absolute" in error or "not allowed" in error

    def test_symlink_escape_rejected(self, temp_workspace):
        """Test 4: Symlink escape attempts are rejected."""
        # Create a symlink pointing outside workspace
        outside_dir = os.path.dirname(temp_workspace)
        symlink_path = os.path.join(temp_workspace, "escape_link")
        
        try:
            os.symlink(outside_dir, symlink_path)
        except (OSError, NotImplementedError):
            # Symlinks may not be supported on all systems; skip if not available
            pytest.skip("Symlinks not supported on this system")
        
        # Try to access parent directory via symlink
        is_valid, error = validate_path("escape_link/../../../etc/passwd", temp_workspace)
        assert is_valid is False
        assert "escapes workspace" in error or "escape" in error.lower()

    def test_nested_valid_path(self, temp_workspace):
        """Test 5: Nested valid path within workspace is allowed."""
        is_valid, error = validate_path("nested/dir/file.txt", temp_workspace)
        assert is_valid is True
        assert error is None

    def test_empty_path_rejected(self, temp_workspace):
        """Test empty path rejection."""
        is_valid, error = validate_path("", temp_workspace)
        assert is_valid is False
        assert "Empty" in error

    def test_current_dir_allowed(self, temp_workspace):
        """Test that '.' is allowed."""
        is_valid, error = validate_path(".", temp_workspace)
        assert is_valid is True
        assert error is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
