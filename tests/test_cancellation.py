#!/usr/bin/env python3
"""Tests for run cancellation."""
import time
import threading
from unittest.mock import Mock, patch
import pytest
from orchestrator import get_cancel_flag, clear_cancel_flag


class TestCancellationMechanism:
    """Test suite for cancellation mechanism (not full orchestration)."""
    
    def test_cancel_flag_thread_safe(self):
        """
        Test that cancel flag is thread-safe.
        Multiple threads can set/check the flag correctly.
        """
        run_id = "test-run-123"
        
        # Get cancel flag
        flag = get_cancel_flag(run_id)
        assert not flag.is_set(), "Flag should start unset"
        
        # Set from another thread
        results = {"flag_set": False, "flag_checked": False}
        
        def setter():
            time.sleep(0.05)
            get_cancel_flag(run_id).set()
            results["flag_set"] = True
        
        def checker():
            time.sleep(0.1)
            results["flag_checked"] = get_cancel_flag(run_id).is_set()
        
        # Start threads
        t1 = threading.Thread(target=setter, daemon=True)
        t2 = threading.Thread(target=checker, daemon=True)
        
        t1.start()
        t2.start()
        
        t1.join(timeout=1)
        t2.join(timeout=1)
        
        # Verify
        assert results["flag_set"], "Flag should be set"
        assert results["flag_checked"], "Checker should see flag set"
        
        # Verify flag is actually set
        assert get_cancel_flag(run_id).is_set(), "Flag should remain set"
        
        # Cleanup
        clear_cancel_flag(run_id)
        print("✓ Cancel flag is thread-safe")
    
    def test_cancel_flag_per_run_isolation(self):
        """
        Test that each run_id gets its own cancel flag.
        Setting one flag doesn't affect others.
        """
        run_id_1 = "run-1"
        run_id_2 = "run-2"
        
        # Get flags for different runs
        flag1 = get_cancel_flag(run_id_1)
        flag2 = get_cancel_flag(run_id_2)
        
        # Set only flag1
        flag1.set()
        
        # Verify isolation
        assert flag1.is_set(), "Flag1 should be set"
        assert not flag2.is_set(), "Flag2 should not be affected"
        
        # Cleanup
        clear_cancel_flag(run_id_1)
        clear_cancel_flag(run_id_2)
        print("✓ Cancel flags are isolated per run_id")
    
    def test_cancel_flag_early_wakeup_from_sleep(self):
        """
        Test that checking cancel flag during sleep allows early exit.
        Simulates the 0.2s sleep loop in rate-limit wait.
        """
        run_id = "test-run-sleep"
        flag = get_cancel_flag(run_id)
        
        # Simulate rate-limit wait that checks cancel every 0.1s
        sleep_remaining = 0.5  # Total 500ms wait
        slices = []
        
        def cancel_after_delay():
            time.sleep(0.15)  # Cancel after 150ms
            flag.set()
        
        # Start cancel thread
        cancel_thread = threading.Thread(target=cancel_after_delay, daemon=True)
        cancel_thread.start()
        
        # Simulate the orchestrator's rate-limit wait loop
        start = time.time()
        slice_size = 0.1  # Check cancel every 100ms
        
        while sleep_remaining > 0:
            if flag.is_set():  # Check cancel
                break
            time.sleep(min(slice_size, sleep_remaining))
            elapsed = time.time() - start
            sleep_remaining -= slice_size
            slices.append(elapsed)
        
        total_elapsed = time.time() - start
        
        # Should exit after ~150ms (not full 500ms)
        assert total_elapsed < 0.3, f"Should exit early but took {total_elapsed:.2f}s"
        assert flag.is_set(), "Flag should be set"
        
        # Cleanup
        cancel_thread.join(timeout=1)
        clear_cancel_flag(run_id)
        print(f"✓ Early wakeup from sleep works ({total_elapsed:.3f}s < 0.3s)")
    
    def test_cancel_flag_cleared_properly(self):
        """
        Test that clear_cancel_flag actually removes the flag.
        """
        run_id = "test-run-clear"
        
        # Get and set flag
        flag1 = get_cancel_flag(run_id)
        flag1.set()
        assert flag1.is_set(), "Flag should be set"
        
        # Clear it
        clear_cancel_flag(run_id)
        
        # Get a new reference to verify it's truly cleared
        flag2 = get_cancel_flag(run_id)
        assert not flag2.is_set(), "New flag reference should not be set"
        
        # Cleanup
        clear_cancel_flag(run_id)
        print("✓ Cancel flag cleared properly")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
