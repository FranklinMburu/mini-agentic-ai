#!/usr/bin/env python3
"""Tests for API token authentication."""
import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from server import app, check_api_security


class TestAPITokenSecurity:
    """Test suite for API token authentication."""
    
    def test_check_api_security_no_token_configured(self):
        """
        When API_TOKEN is not set, all requests are allowed.
        """
        with patch('server.API_TOKEN', None):
            # No token configured - should allow
            result = check_api_security({})
            assert result is True, "Should allow when token not configured"
            
            # Even with wrong header should allow
            result = check_api_security({"x-api-token": "anything"})
            assert result is True, "Should allow any token when not configured"
            
            print("✓ No token configured: all requests allowed")
    
    def test_check_api_security_with_token_missing_header(self):
        """
        When API_TOKEN is set, missing header should be rejected.
        """
        with patch('server.API_TOKEN', 'secret-key'):
            # Missing header
            result = check_api_security({})
            assert result is False, "Should reject missing header"
            
            # Empty header
            result = check_api_security({"x-api-token": ""})
            assert result is False, "Should reject empty header"
            
            print("✓ Token configured, missing header: rejected")
    
    def test_check_api_security_with_token_wrong_header(self):
        """
        When API_TOKEN is set, wrong token should be rejected.
        """
        with patch('server.API_TOKEN', 'correct-secret'):
            # Wrong token
            result = check_api_security({"x-api-token": "wrong-secret"})
            assert result is False, "Should reject wrong token"
            
            # Different wrong token
            result = check_api_security({"x-api-token": "another-wrong"})
            assert result is False, "Should reject different wrong token"
            
            print("✓ Token configured, wrong header: rejected")
    
    def test_check_api_security_with_token_correct_header(self):
        """
        When API_TOKEN is set, correct token should be accepted.
        """
        with patch('server.API_TOKEN', 'my-secret-token'):
            # Correct token
            result = check_api_security({"x-api-token": "my-secret-token"})
            assert result is True, "Should accept correct token"
            
            print("✓ Token configured, correct header: accepted")
    
    def test_check_api_security_case_sensitive(self):
        """
        Token check should be case-sensitive.
        """
        with patch('server.API_TOKEN', 'MySecretToken'):
            # Wrong case
            result = check_api_security({"x-api-token": "mysecrettoken"})
            assert result is False, "Should be case-sensitive"
            
            # Correct case
            result = check_api_security({"x-api-token": "MySecretToken"})
            assert result is True, "Should accept with correct case"
            
            print("✓ Token check is case-sensitive")


class TestAPITokenEndpoints:
    """Test endpoints with token authentication."""
    
    def test_api_run_no_token_configured(self):
        """
        POST /api/run should work when API_TOKEN is not configured.
        """
        client = TestClient(app)
        
        with patch('server.API_TOKEN', None):
            # Request without token should succeed
            response = client.post(
                "/api/run",
                json={"goal": "Test goal"}
            )
            # Should not be 401 (might be other errors, but not auth)
            assert response.status_code != 401, "Should not require token when not configured"
            
            print("✓ /api/run accessible without token when not configured")
    
    def test_api_run_token_configured_missing_header(self):
        """
        POST /api/run should return 401 when token is configured but header is missing.
        """
        client = TestClient(app)
        
        with patch('server.API_TOKEN', 'test-secret'):
            # Request without token header should fail with 401
            response = client.post(
                "/api/run",
                json={"goal": "Test goal"}
            )
            assert response.status_code == 401, f"Expected 401, got {response.status_code}"
            
            print("✓ /api/run returns 401 when token configured, header missing")
    
    def test_api_run_token_configured_wrong_token(self):
        """
        POST /api/run should return 401 with wrong token.
        """
        client = TestClient(app)
        
        with patch('server.API_TOKEN', 'correct-token'):
            # Request with wrong token should fail with 401
            response = client.post(
                "/api/run",
                json={"goal": "Test goal"},
                headers={"X-API-Token": "wrong-token"}
            )
            assert response.status_code == 401, f"Expected 401, got {response.status_code}"
            
            print("✓ /api/run returns 401 with wrong token")
    
    def test_api_run_token_configured_correct_token(self):
        """
        POST /api/run should accept correct token (won't execute but won't auth fail).
        """
        client = TestClient(app)
        
        with patch('server.API_TOKEN', 'correct-token'):
            # Mock the API key to avoid needing real Gemini API
            with patch('server.OrchestratorConfig') as mock_config:
                mock_cfg = MagicMock()
                mock_cfg.api_key = "test-key"
                mock_config.return_value = mock_cfg
                
                # Request with correct token should pass auth (might fail on other validation)
                response = client.post(
                    "/api/run",
                    json={"goal": "Test goal"},
                    headers={"X-API-Token": "correct-token"}
                )
                # Should not be 401 (will be 500 or other error due to mocking, but not auth)
                assert response.status_code != 401, f"Got {response.status_code}, expected non-401"
                
                print("✓ /api/run accepts correct token (passes auth)")
    
    def test_api_cancel_no_token_configured(self):
        """
        POST /api/cancel should work when API_TOKEN is not configured.
        """
        client = TestClient(app)
        
        with patch('server.API_TOKEN', None):
            # Request without token should succeed (or fail on missing run_id, but not auth)
            response = client.post(
                "/api/cancel",
                json={"run_id": "abc123"}
            )
            # Should not be 401
            assert response.status_code != 401, "Should not require token when not configured"
            
            print("✓ /api/cancel accessible without token when not configured")
    
    def test_api_cancel_token_configured_missing_header(self):
        """
        POST /api/cancel should return 401 when token is configured but header is missing.
        """
        client = TestClient(app)
        
        with patch('server.API_TOKEN', 'test-secret'):
            # Request without token header should fail with 401
            response = client.post(
                "/api/cancel",
                json={"run_id": "abc123"}
            )
            assert response.status_code == 401, f"Expected 401, got {response.status_code}"
            
            print("✓ /api/cancel returns 401 when token configured, header missing")
    
    def test_api_cancel_token_configured_wrong_token(self):
        """
        POST /api/cancel should return 401 with wrong token.
        """
        client = TestClient(app)
        
        with patch('server.API_TOKEN', 'correct-token'):
            # Request with wrong token should fail with 401
            response = client.post(
                "/api/cancel",
                json={"run_id": "abc123"},
                headers={"X-API-Token": "wrong-token"}
            )
            assert response.status_code == 401, f"Expected 401, got {response.status_code}"
            
            print("✓ /api/cancel returns 401 with wrong token")
    
    def test_api_cancel_token_configured_correct_token(self):
        """
        POST /api/cancel should accept correct token.
        """
        client = TestClient(app)
        
        with patch('server.API_TOKEN', 'correct-token'):
            with patch('server.get_cancel_flag') as mock_flag:
                mock_flag_instance = MagicMock()
                mock_flag.return_value = mock_flag_instance
                
                # Request with correct token should pass auth
                response = client.post(
                    "/api/cancel",
                    json={"run_id": "abc123"},
                    headers={"X-API-Token": "correct-token"}
                )
                # Should not be 401
                assert response.status_code != 401, f"Got {response.status_code}, expected non-401"
                
                print("✓ /api/cancel accepts correct token (passes auth)")


class TestUnauthenticatedEndpoints:
    """Test that GET endpoints don't require authentication."""
    
    def test_get_root_no_token_required(self):
        """GET / should always work, even when API_TOKEN is set."""
        client = TestClient(app)
        
        with patch('server.API_TOKEN', 'some-token'):
            response = client.get("/")
            # Should not be 401
            assert response.status_code != 401, "GET / should not require token"
            
            print("✓ GET / doesn't require token")
    
    def test_get_api_state_no_token_required(self):
        """GET /api/state should always work, even when API_TOKEN is set."""
        client = TestClient(app)
        
        with patch('server.API_TOKEN', 'some-token'):
            response = client.get("/api/state")
            # Should not be 401
            assert response.status_code != 401, "GET /api/state should not require token"
            
            print("✓ GET /api/state doesn't require token")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
