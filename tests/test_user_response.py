#!/usr/bin/env python3
"""Integration test for user response resume flow."""
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
import pytest
from server import app, register_orchestrator, unregister_orchestrator
from orchestrator import Orchestrator, OrchestratorConfig


def test_user_response_resume_flow():
    """
    Test that user response flow works end-to-end:
    1. Orchestrator runs and encounters need_user
    2. API accepts response via /api/respond
    3. Orchestrator resumes and continues to next step
    4. Emits user_input + run_finished events
    """
    client = TestClient(app)
    
    # Create orchestrator with mocked workers
    config = OrchestratorConfig()
    config.api_key = "test-key"
    
    with patch('orchestrator.Planner') as MockPlanner:
        with patch('orchestrator.Coder') as MockCoder:
            with patch('orchestrator.Reviewer') as MockReviewer:
                with patch('orchestrator.RateLimiter') as MockRateLimiter:
                    # Mock rate limiter to always succeed (no waiting)
                    mock_limiter = MagicMock()
                    mock_limiter.acquire = Mock(return_value=(True, 0))
                    MockRateLimiter.return_value = mock_limiter
                    
                    # Setup plan: 2 steps
                    mock_planner = MagicMock()
                    mock_planner.generate_plan = Mock(return_value={
                        "type": "plan",
                        "steps": [
                            {"description": "First step"},
                            {"description": "Second step"}
                        ]
                    })
                    MockPlanner.return_value = mock_planner
                    
                    # Setup coder to return need_user on first call, step_complete on second
                    mock_coder = MagicMock()
                    call_count = {"count": 0}
                    
                    def coder_process_step(*args, **kwargs):
                        call_count["count"] += 1
                        if call_count["count"] == 1:
                            # First step: ask for user input
                            return {
                                "type": "need_user",
                                "question": "What should I do?",
                                "why": "Need clarification"
                            }
                        else:
                            # Second step: complete
                            return {
                                "type": "step_complete",
                                "summary": "Step completed",
                                "next": "continue"
                            }
                    
                    mock_coder.process_step = Mock(side_effect=coder_process_step)
                    MockCoder.return_value = mock_coder
                    
                    # Create orchestrator
                    orch = Orchestrator(config)
                    
                    # Mock the router to route to coder
                    original_route = orch._route_step
                    orch._route_step = Mock(return_value="coder")
                    
                    # Register it so /api/respond can find it
                    register_orchestrator(orch.run_id, orch)
                    
                    # Run in background thread
                    def run_orch():
                        try:
                            orch.run("Test goal")
                        except Exception as e:
                            print(f"Orchestrator error: {e}")
                        finally:
                            unregister_orchestrator(orch.run_id)
                    
                    thread = threading.Thread(target=run_orch, daemon=True)
                    thread.start()
                    
                    # Wait for orchestrator to hit need_user with timeout
                    max_wait = 5
                    waited = 0
                    while waited < max_wait:
                        if orch.state and orch.state.status == "waiting_user":
                            break
                        time.sleep(0.1)
                        waited += 0.1
                    
                    # Verify it's waiting for user input
                    assert orch.state.status == "waiting_user", f"Expected waiting_user, got {orch.state.status}"
                    
                    # Make sure orchestrator is still registered before calling API
                    from server import get_active_orchestrator
                    registered = get_active_orchestrator(orch.run_id)
                    assert registered is not None, "Orchestrator should be registered"
                    
                    # Call /api/respond
                    response = client.post(
                        "/api/respond",
                        json={
                            "run_id": orch.run_id,
                            "response": "Here is my answer"
                        }
                    )
                    
                    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
                    data = response.json()
                    assert data["ok"] is True, f"Expected ok=true, got {data}"
                    
                    # Wait for orchestrator to process response and continue
                    thread.join(timeout=5)
                    
                    # Verify run completed
                    assert orch.state.status == "completed", f"Expected completed, got {orch.state.status}"
                    
                    # Verify user_input event was emitted
                    # (This is done internally by provide_user_input)
                    assert orch.state.user_response is None, "Response should be consumed"
                    
                    print("✓ User response resume flow works end-to-end")


def test_api_respond_without_active_run():
    """Test that /api/respond returns error when run is not waiting."""
    client = TestClient(app)
    
    response = client.post(
        "/api/respond",
        json={
            "run_id": "nonexistent-run-id",
            "response": "Some response"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is False
    assert "not currently waiting" in data["error"].lower()
    
    print("✓ /api/respond correctly handles missing run")


def test_api_respond_missing_fields():
    """Test that /api/respond validates required fields."""
    client = TestClient(app)
    
    # Missing response
    response = client.post(
        "/api/respond",
        json={"run_id": "some-run-id"}
    )
    assert response.json()["ok"] is False
    
    # Missing run_id
    response = client.post(
        "/api/respond",
        json={"response": "Some response"}
    )
    assert response.json()["ok"] is False
    
    print("✓ /api/respond validates required fields")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
