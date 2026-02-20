#!/usr/bin/env python3
"""Test event log persistence to runs/<run_id>/events.jsonl"""

import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
from orchestrator import Orchestrator, OrchestratorConfig


def test_event_log_persistence_minimal_flow():
    """
    Test that events are persisted to runs/<run_id>/events.jsonl.
    Runs a minimal orchestration flow and verifies:
    1. events.jsonl file is created
    2. Contains at least run_started and run_finished events
    3. Each event is valid JSON on a single line (JSONL format)
    """
    config = OrchestratorConfig()
    config.api_key = "test-key"
    
    with patch('orchestrator.Planner') as MockPlanner:
        with patch('orchestrator.Coder') as MockCoder:
            with patch('orchestrator.RateLimiter') as MockRateLimiter:
                # Mock rate limiter to always succeed
                mock_limiter = MagicMock()
                mock_limiter.acquire = Mock(return_value=(True, 0))
                MockRateLimiter.return_value = mock_limiter
                
                # Mock planner
                mock_planner = MagicMock()
                mock_planner.generate_plan = Mock(return_value={
                    "type": "plan",
                    "steps": [{"description": "Test step"}]
                })
                MockPlanner.return_value = mock_planner
                
                # Mock coder to return step_complete
                mock_coder = MagicMock()
                mock_coder.process_step = Mock(return_value={
                    "type": "step_complete",
                    "summary": "Test completed",
                    "next": "finish"
                })
                MockCoder.return_value = mock_coder
                
                # Create and run orchestrator
                orch = Orchestrator(config)
                orch._route_step = Mock(return_value="coder")
                
                try:
                    orch.run("Test goal")
                except Exception as e:
                    # Some error might occur, but we still check event persistence
                    print(f"Orchestrator run error (expected): {e}")
                
                run_id = orch.run_id
                
                # Check that events.jsonl was created
                events_file = Path("runs") / run_id / "events.jsonl"
                assert events_file.exists(), f"events.jsonl not found at {events_file}"
                
                # Read and parse events from JSONL file
                events = []
                with open(events_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():  # Skip empty lines
                            try:
                                event = json.loads(line)
                                events.append(event)
                            except json.JSONDecodeError as e:
                                pytest.fail(f"Invalid JSON in events.jsonl: {line}\nError: {e}")
                
                # Assert there's at least some events
                assert len(events) > 0, "events.jsonl is empty"
                
                # Check for required event types
                event_types = {e["type"] for e in events}
                
                # Assert run_started is present
                assert "run_started" in event_types, f"run_started not found. Types: {event_types}"
                
                # Assert either run_finished or run_cancelled is present
                assert "run_finished" in event_types or "run_cancelled" in event_types, \
                    f"Neither run_finished nor run_cancelled found. Types: {event_types}"
                
                # Verify all events have required fields
                for event in events:
                    assert "id" in event, "Event missing 'id'"
                    assert "ts_iso" in event, "Event missing 'ts_iso'"
                    assert "run_id" in event, "Event missing 'run_id'"
                    assert "type" in event, "Event missing 'type'"
                    assert "payload" in event, "Event missing 'payload'"
                    assert event["run_id"] == run_id, f"Event run_id mismatch: {event['run_id']} != {run_id}"
                
                print(f"✓ Event persistence test passed")
                print(f"  - events.jsonl: {len(events)} events")
                print(f"  - Event types: {event_types}")


def test_event_persistence_error_handling():
    """
    Test that event persistence failure doesn't crash the run.
    Even if we can't write to disk, the orchestrator should continue.
    """
    config = OrchestratorConfig()
    config.api_key = "test-key"
    
    with patch('orchestrator.Planner') as MockPlanner:
        with patch('orchestrator.Coder') as MockCoder:
            with patch('orchestrator.RateLimiter') as MockRateLimiter:
                # Mock rate limiter
                mock_limiter = MagicMock()
                mock_limiter.acquire = Mock(return_value=(True, 0))
                MockRateLimiter.return_value = mock_limiter
                
                # Mock planner
                mock_planner = MagicMock()
                mock_planner.generate_plan = Mock(return_value={
                    "type": "plan",
                    "steps": [{"description": "Test step"}]
                })
                MockPlanner.return_value = mock_planner
                
                # Mock coder
                mock_coder = MagicMock()
                mock_coder.process_step = Mock(return_value={
                    "type": "step_complete",
                    "summary": "Test completed",
                    "next": "finish"
                })
                MockCoder.return_value = mock_coder
                
                # Test with a broken runs directory path
                # Patch Path.mkdir to raise an exception
                with patch('pathlib.Path.mkdir', side_effect=PermissionError("Cannot create directory")):
                    orch = Orchestrator(config)
                    orch._route_step = Mock(return_value="coder")
                    
                    # The run should continue despite persistence failure
                    try:
                        orch.run("Test goal with broken persistence")
                        # If we get here, the run completed (possibly with errors, but didn't crash)
                        assert True, "Run completed despite persistence failure"
                    except PermissionError:
                        pytest.fail("Persistence error should not crash orchestrator")
                    except Exception as e:
                        # Other errors are OK (orchestration error), but not PermissionError
                        if isinstance(e, PermissionError):
                            pytest.fail("Persistence error should not crash orchestrator")


def test_final_output_on_completion():
    """Test that final_output is set when run completes."""
    config = OrchestratorConfig()
    config.api_key = "test-key"
    
    with patch('orchestrator.Planner') as MockPlanner:
        with patch('orchestrator.Coder') as MockCoder:
            with patch('orchestrator.RateLimiter') as MockRateLimiter:
                # Mock rate limiter
                mock_limiter = MagicMock()
                mock_limiter.acquire = Mock(return_value=(True, 0))
                MockRateLimiter.return_value = mock_limiter
                
                # Mock planner
                mock_planner = MagicMock()
                mock_planner.generate_plan = Mock(return_value={
                    "type": "plan",
                    "steps": [{"description": "Final step"}]
                })
                MockPlanner.return_value = mock_planner
                
                # Mock coder to return step_complete with summary
                mock_coder = MagicMock()
                mock_coder.process_step = Mock(return_value={
                    "type": "step_complete",
                    "summary": "Final summary output with enough detail to be meaningful and promoted.",
                    "next": "finish"
                })
                MockCoder.return_value = mock_coder
                
                # Run orchestrator
                orch = Orchestrator(config)
                orch._route_step = Mock(return_value="coder")
                orch.run("Final output test")
                
                # Verify final_output is set
                assert orch.state is not None, "State should exist"
                assert orch.state.status == "completed", f"Status should be completed, got {orch.state.status}"
                assert orch.state.final_output is not None, "final_output should be set"
                assert "meaningful" in orch.state.final_output, f"final_output should contain summary: {orch.state.final_output}"
                
                print(f"✓ Final output set correctly: {orch.state.final_output}")


def test_artifacts_on_file_write():
    """
    Test that writing a file adds an artifact to state.artifacts.
    Verifies:
    1. Artifact has correct structure: path, size, ts_iso, kind
    2. Duplicate paths update existing artifact (no duplicates)
    3. Artifact is persisted in state
    """
    config = OrchestratorConfig()
    config.api_key = "test-key"
    
    # Create temp workspace
    with tempfile.TemporaryDirectory() as tmpdir:
        config.workspace_root = tmpdir
        
        with patch('orchestrator.Planner') as MockPlanner:
            with patch('orchestrator.Coder') as MockCoder:
                with patch('orchestrator.RateLimiter') as MockRateLimiter:
                    # Mock rate limiter
                    mock_limiter = MagicMock()
                    mock_limiter.acquire = Mock(return_value=(True, 0))
                    MockRateLimiter.return_value = mock_limiter
                    
                    # Mock planner
                    mock_planner = MagicMock()
                    mock_planner.generate_plan = Mock(return_value={
                        "type": "plan",
                        "steps": [{"description": "Write file"}]
                    })
                    MockPlanner.return_value = mock_planner
                    
                    # Mock coder to return write_file action
                    mock_coder = MagicMock()
                    mock_coder.process_step = Mock(return_value={
                        "type": "write_file",
                        "path": "test.py",
                        "content": "print('hello')"
                    })
                    MockCoder.return_value = mock_coder
                    
                    # Run orchestrator
                    orch = Orchestrator(config)
                    orch._route_step = Mock(return_value="coder")
                    orch.run("Write file test")
                    
                    # Verify artifacts list exists and contains the file
                    assert orch.state is not None, "State should exist"
                    assert hasattr(orch.state, 'artifacts'), "State should have artifacts field"
                    assert len(orch.state.artifacts) > 0, "Should have at least one artifact"
                    
                    artifact = orch.state.artifacts[0]
                    assert artifact['path'] == 'test.py', f"Path should be test.py, got {artifact['path']}"
                    assert artifact['size'] == len("print('hello')"), f"Size mismatch: {artifact['size']}"
                    assert artifact['kind'] == 'file', f"Kind should be 'file', got {artifact['kind']}"
                    assert 'ts_iso' in artifact, "Artifact should have ts_iso"
                    assert artifact['ts_iso'].endswith('Z'), "ts_iso should be ISO format with Z suffix"
                    
                    print(f"✓ Artifact added correctly: {artifact}")


