#!/usr/bin/env python3
"""Test that Planner returns step_complete for synthesis goals"""

from unittest.mock import Mock, patch, MagicMock
import pytest
import json
from workers import Planner
from orchestrator import Orchestrator, OrchestratorConfig


class TestPlannerSynthesis:
    """Test Planner synthesis detection and step_complete conversion."""

    def test_planner_detects_synthesis_goal(self):
        """Test that planner converts synthesis goals to step_complete."""
        client = MagicMock()
        planner = Planner(client, "gemini-2.5-flash")
        
        # Mock LLM to return a plan-like response
        llm_response = json.dumps({
            "type": "plan",
            "steps": [
                {"number": 1, "description": "Define simplicity"},
                {"number": 2, "description": "Provide examples"},
            ],
            "reasoning": "Simplicity in applications means ease of use and minimal complexity, achieved through clear design and reduced feature bloat."
        })
        
        with patch.object(planner, '_call_llm', return_value=llm_response):
            # Goal with synthesis keywords should trigger conversion
            result = planner.generate_plan(
                "Formulate the answer: what is a simplicity app",
                "Previous context"
            )
        
        # Should have converted to step_complete
        assert result["type"] == "step_complete", \
            f"Expected step_complete for synthesis goal, got {result['type']}"
        assert "summary" in result, "step_complete should have 'summary' field"
        assert "next" in result, "step_complete should have 'next' field"
        assert result["next"] == "continue", "next field should be 'continue'"
        assert len(result["summary"]) > 20, "Summary should be meaningful (>20 chars)"
        assert "simplicity" in result["summary"].lower(), \
            f"Summary should contain answer content, got: {result['summary']}"


    def test_planner_keeps_plan_for_decomposition(self):
        """Test that planner keeps plan type for decomposition goals."""
        client = MagicMock()
        planner = Planner(client, "gemini-2.5-flash")
        
        # Mock LLM to return a plan response
        llm_response = json.dumps({
            "type": "plan",
            "steps": [
                {"number": 1, "description": "Research topic"},
                {"number": 2, "description": "Analyze findings"},
                {"number": 3, "description": "Write summary"},
            ],
            "reasoning": "Approaching this systematically"
        })
        
        with patch.object(planner, '_call_llm', return_value=llm_response):
            # Goal without synthesis keywords should keep plan
            result = planner.generate_plan(
                "Create a plan to implement user authentication",
                "Current codebase context"
            )
        
        # Should remain as plan
        assert result["type"] == "plan", \
            f"Expected plan for decomposition goal, got {result['type']}"
        assert "steps" in result, "plan should have 'steps' field"
        assert len(result["steps"]) > 0, "plan should have at least one step"


    def test_question_run_produces_step_complete(self):
        """
        Integration test: question-style run should produce at least one step_complete.
        Verifies final_output is not "Run completed".
        """
        config = OrchestratorConfig()
        config.api_key = "test-key"
        
        with patch('orchestrator.Planner') as MockPlanner:
            with patch('orchestrator.Coder') as MockCoder:
                with patch('orchestrator.Reviewer') as MockReviewer:
                    with patch('orchestrator.RateLimiter') as MockRateLimiter:
                        # Mock rate limiter
                        mock_limiter = MagicMock()
                        mock_limiter.acquire = Mock(return_value=(True, 0))
                        MockRateLimiter.return_value = mock_limiter
                        
                        # Mock planner to return step_complete on synthesis step
                        mock_planner = MagicMock()
                        mock_planner.generate_plan = Mock(side_effect=[
                            {  # Initial plan
                                "type": "plan",
                                "steps": [
                                    {"number": 1, "description": "Identify the core concept"},
                                ],
                                "reasoning": "Breaking down the question"
                            },
                            {  # Final synthesis step - should return step_complete
                                "type": "step_complete",
                                "summary": "Simplicity in applications means ease of use and minimal complexity achieved through clear design.",
                                "next": "continue"
                            }
                        ])
                        MockPlanner.return_value = mock_planner
                        
                        # Only planner is used, so stub others
                        MockCoder.return_value = MagicMock()
                        MockReviewer.return_value = MagicMock()
                        
                        # Run orchestrator
                        orch = Orchestrator(config)
                        orch.run("What is a simplicity app?")
                        
                        # Verify results
                        assert orch.state is not None, "State should exist"
                        assert orch.state.status == "completed", \
                            f"Status should be completed, got {orch.state.status}"
                        assert orch.state.last_summary, \
                            "last_summary should be set from step_complete"
                        assert orch.state.final_output is not None, \
                            "final_output should be set"
                        assert orch.state.final_output != "Run completed", \
                            f"final_output should NOT be generic fallback, got: {orch.state.final_output}"
                        assert "simplicity" in orch.state.final_output.lower(), \
                            f"final_output should show the answer, got: {orch.state.final_output}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
