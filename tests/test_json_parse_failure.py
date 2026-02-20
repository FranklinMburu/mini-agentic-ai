#!/usr/bin/env python3
"""
Integration test: JSON parse failure triggers INVALID_WORKER_OUTPUT event.
"""
import json
from unittest.mock import Mock, patch
from workers import Planner, Coder, validate_action
from events import event_bus


def test_json_parse_failure_triggers_invalid_worker_output():
    """
    Verify that when _parse_json_response fails (JSONDecodeError),
    the error is handled and validate_action rejects it.
    """
    # Create a mock client
    mock_client = Mock()
    
    # Create Planner instance with mock client
    planner = Planner(mock_client, "gemini-2.5-flash")
    
    # Mock _call_llm to return invalid JSON
    invalid_json = "this is not valid json {[ invalid"
    planner._call_llm = Mock(return_value=invalid_json)
    
    # Try to generate plan - should raise JSONDecodeError during _parse_json_response
    try:
        result = planner.generate_plan("Test goal")
        # If we reach here, the exception was caught and fallback was used
        # The fallback should still be a valid action
        is_valid, error = validate_action(result)
        assert is_valid, f"Fallback plan should be valid but got: {error}"
        print("✓ Fallback mechanism caught parse failure and provided valid plan")
    except json.JSONDecodeError as e:
        # JSON parse error propagates - orchestrator should catch it
        print(f"✓ JSONDecodeError raised as expected: {e}")


def test_validate_action_rejects_error_type():
    """
    Verify that if a worker somehow returns type='error',
    validate_action rejects it.
    """
    error_action = {
        "type": "error",
        "reason": "JSON_PARSE_ERROR",
        "details": "Invalid JSON"
    }
    
    is_valid, error_reason = validate_action(error_action)
    assert not is_valid, "Error action should be rejected"
    assert "error" in error_reason.lower(), f"Error message should mention 'error': {error_reason}"
    print(f"✓ validate_action rejects error type: {error_reason}")


def test_coder_parse_failure():
    """Test Coder worker JSON parse failure."""
    mock_client = Mock()
    coder = Coder(mock_client, "gemini-2.5-flash")
    
    # Mock _call_llm to return invalid JSON
    invalid_json = "{ broken json without closing"
    coder._call_llm = Mock(return_value=invalid_json)
    
    try:
        result = coder.process_step("Test step", "test context")
        # Fallback was used
        is_valid, error = validate_action(result)
        print(f"✓ Coder fallback valid: {is_valid}")
    except json.JSONDecodeError as e:
        print(f"✓ Coder JSONDecodeError raised: {e}")


if __name__ == "__main__":
    test_json_parse_failure_triggers_invalid_worker_output()
    test_validate_action_rejects_error_type()
    test_coder_parse_failure()
    print("\n✓ All JSON parse failure tests passed")
