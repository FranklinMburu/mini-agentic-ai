#!/usr/bin/env python3
"""Tests for worker response validation."""
import pytest
from workers import validate_action


class TestValidateAction:
    """Test suite for validate_action function."""
    
    # Valid cases
    def test_valid_plan_action(self):
        """Valid plan action should pass validation."""
        action = {
            "type": "plan",
            "steps": [
                {"description": "Step 1"},
                {"description": "Step 2"}
            ]
        }
        ok, error = validate_action(action)
        assert ok, f"Valid plan failed: {error}"
        assert error is None
    
    def test_valid_tool_call(self):
        """Valid tool_call action should pass validation."""
        action = {
            "type": "tool_call",
            "name": "read_file",
            "args": {"path": "test.txt"}
        }
        ok, error = validate_action(action)
        assert ok, f"Valid tool_call failed: {error}"
        assert error is None
    
    def test_valid_step_complete(self):
        """Valid step_complete action should pass validation."""
        action = {
            "type": "step_complete",
            "summary": "Task completed",
            "next": "continue"
        }
        ok, error = validate_action(action)
        assert ok, f"Valid step_complete failed: {error}"
        assert error is None
    
    def test_valid_need_user(self):
        """Valid need_user action should pass validation."""
        action = {
            "type": "need_user",
            "question": "What should I do?",
            "why": "Need clarification"
        }
        ok, error = validate_action(action)
        assert ok, f"Valid need_user failed: {error}"
        assert error is None
    
    def test_valid_write_file(self):
        """Valid write_file action should pass validation."""
        action = {
            "type": "write_file",
            "path": "output.txt",
            "content": "Hello world"
        }
        ok, error = validate_action(action)
        assert ok, f"Valid write_file failed: {error}"
        assert error is None
    
    # Invalid cases - missing type
    def test_missing_type(self):
        """Missing type field should fail."""
        action = {"steps": []}
        ok, error = validate_action(action)
        assert not ok
        assert "type" in error.lower()
    
    # Invalid cases - not a dict
    def test_not_dict(self):
        """Non-dict action should fail."""
        action = "not a dict"
        ok, error = validate_action(action)
        assert not ok
        assert "dict" in error.lower()
    
    # Plan validation
    def test_plan_missing_steps(self):
        """Plan missing steps field should fail."""
        action = {"type": "plan"}
        ok, error = validate_action(action)
        assert not ok
        assert "steps" in error.lower()
    
    def test_plan_steps_not_list(self):
        """Plan with non-list steps should fail."""
        action = {
            "type": "plan",
            "steps": "not a list"
        }
        ok, error = validate_action(action)
        assert not ok
        assert "list" in error.lower()
    
    def test_plan_step_not_dict(self):
        """Plan step that's not a dict should fail."""
        action = {
            "type": "plan",
            "steps": ["should be dict"]
        }
        ok, error = validate_action(action)
        assert not ok
        assert "dict" in error.lower()
    
    def test_plan_step_missing_description(self):
        """Plan step missing description should fail."""
        action = {
            "type": "plan",
            "steps": [{"value": "no description"}]
        }
        ok, error = validate_action(action)
        assert not ok
        assert "description" in error.lower()
    
    # Tool_call validation
    def test_tool_call_missing_name(self):
        """Tool_call missing name should fail."""
        action = {
            "type": "tool_call",
            "args": {}
        }
        ok, error = validate_action(action)
        assert not ok
        assert "name" in error.lower()
    
    def test_tool_call_missing_args(self):
        """Tool_call missing args should fail."""
        action = {
            "type": "tool_call",
            "name": "test_tool"
        }
        ok, error = validate_action(action)
        assert not ok
        assert "args" in error.lower()
    
    def test_tool_call_args_not_dict(self):
        """Tool_call with non-dict args should fail."""
        action = {
            "type": "tool_call",
            "name": "test_tool",
            "args": "not dict"
        }
        ok, error = validate_action(action)
        assert not ok
        assert "dict" in error.lower()
    
    def test_tool_call_name_not_string(self):
        """Tool_call with non-string name should fail."""
        action = {
            "type": "tool_call",
            "name": 123,
            "args": {}
        }
        ok, error = validate_action(action)
        assert not ok
        assert "string" in error.lower()
    
    # Step_complete validation
    def test_step_complete_missing_summary(self):
        """step_complete missing summary should fail."""
        action = {
            "type": "step_complete",
            "next": "continue"
        }
        ok, error = validate_action(action)
        assert not ok
        assert "summary" in error.lower()
    
    def test_step_complete_missing_next(self):
        """step_complete missing next should fail."""
        action = {
            "type": "step_complete",
            "summary": "Done"
        }
        ok, error = validate_action(action)
        assert not ok
        assert "next" in error.lower()
    
    def test_step_complete_summary_not_string(self):
        """step_complete with non-string summary should fail."""
        action = {
            "type": "step_complete",
            "summary": 123,
            "next": "continue"
        }
        ok, error = validate_action(action)
        assert not ok
        assert "string" in error.lower()
    
    # Need_user validation
    def test_need_user_missing_question(self):
        """need_user missing question should fail."""
        action = {
            "type": "need_user",
            "why": "clarification"
        }
        ok, error = validate_action(action)
        assert not ok
        assert "question" in error.lower()
    
    def test_need_user_why_not_string(self):
        """need_user with non-string why should fail."""
        action = {
            "type": "need_user",
            "question": "What?",
            "why": 123
        }
        ok, error = validate_action(action)
        assert not ok
        assert "string" in error.lower()
    
    # Write_file validation
    def test_write_file_missing_path(self):
        """write_file missing path should fail."""
        action = {
            "type": "write_file",
            "content": "data"
        }
        ok, error = validate_action(action)
        assert not ok
        assert "path" in error.lower()
    
    def test_write_file_missing_content(self):
        """write_file missing content should fail."""
        action = {
            "type": "write_file",
            "path": "file.txt"
        }
        ok, error = validate_action(action)
        assert not ok
        assert "content" in error.lower()
    
    def test_write_file_path_not_string(self):
        """write_file with non-string path should fail."""
        action = {
            "type": "write_file",
            "path": 123,
            "content": "data"
        }
        ok, error = validate_action(action)
        assert not ok
        assert "string" in error.lower()
    
    def test_write_file_content_not_string(self):
        """write_file with non-string content should fail."""
        action = {
            "type": "write_file",
            "path": "file.txt",
            "content": 123
        }
        ok, error = validate_action(action)
        assert not ok
        assert "string" in error.lower()
    
    # Unknown type
    def test_unknown_action_type(self):
        """Unknown action type should fail."""
        action = {
            "type": "unknown_action"
        }
        ok, error = validate_action(action)
        assert not ok
        assert "unknown" in error.lower()
    
    # Error type (from JSON parse guards)
    def test_error_action_type(self):
        """Error action type (from parse failures) should fail."""
        action = {
            "type": "error",
            "reason": "JSON_PARSE_ERROR",
            "details": "Unexpected token"
        }
        ok, error = validate_action(action)
        assert not ok
        assert "error" in error.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
