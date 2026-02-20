#!/usr/bin/env python3
"""Test final output promotion with meaningful summaries"""

from unittest.mock import Mock, patch, MagicMock
import pytest
from orchestrator import Orchestrator, OrchestratorConfig


def test_final_output_shows_meaningful_summary():
    """
    Test that final_output shows the actual summary when it's meaningful (>20 chars).
    When a step completes with a real answer, it should appear in final_output.
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
                
                # Mock planner to return plan
                mock_planner = MagicMock()
                mock_planner.generate_plan = Mock(return_value={
                    "type": "plan",
                    "steps": [{"description": "Implement the solution"}]
                })
                MockPlanner.return_value = mock_planner
                
                # Mock coder to return step_complete with meaningful summary
                mock_coder = MagicMock()
                mock_coder.process_step = Mock(return_value={
                    "type": "step_complete",
                    "summary": "Simplicity in applications means ease of use and minimal complexity, achieved through clear design and reduced feature bloat.",
                    "next": "continue"
                })
                MockCoder.return_value = mock_coder
                
                # Run orchestrator - route to coder so we get the summary
                orch = Orchestrator(config)
                orch._route_step = Mock(return_value="coder")  # Force coder route
                orch.run("what is simplicity in applications")
                
                # Verify final_output shows the summary (not "Run completed")
                assert orch.state is not None, "State should exist"
                assert orch.state.status == "completed", f"Status should be completed, got {orch.state.status}"
                assert orch.state.final_output is not None, "final_output should be set"
                
                # Should show the meaningful summary (>20 chars)
                assert "complexity" in orch.state.final_output.lower(), \
                    f"final_output should contain summary, got: {orch.state.final_output}"
                assert "Run completed" != orch.state.final_output, \
                    f"Should NOT be generic fallback, got: {orch.state.final_output}"
                
                print(f"✓ Final output shows answer: {orch.state.final_output[:60]}...")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
