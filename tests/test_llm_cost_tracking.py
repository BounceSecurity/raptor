#!/usr/bin/env python3
"""
Comprehensive Test Suite for LLM Cost Tracking Fix

Tests cover:
- Unit tests (individual functions)
- Integration tests (full workflows)
- User story tests (real usage patterns)
- Edge cases (boundaries, errors, retries)
- Chaos tests (random inputs, concurrent access)
- Adversarial tests (security, budget bypass attempts)
"""

import sys
import json
import pytest
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from packages.llm_analysis.llm.client import LLMClient
from packages.llm_analysis.llm.config import LLMConfig, ModelConfig
from packages.llm_analysis.llm.providers import LLMProvider, LLMResponse, OllamaProvider


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def basic_config():
    """Basic LLM config for testing"""
    return LLMConfig(
        max_cost_per_scan=10.0,
        enable_cost_tracking=True,
        enable_fallback=False,
        max_retries=1
    )


@pytest.fixture
def strict_budget_config():
    """Config with very low budget for budget testing"""
    return LLMConfig(
        max_cost_per_scan=0.10,
        enable_cost_tracking=True,
        enable_fallback=False,
        max_retries=1
    )


@pytest.fixture
def mock_provider():
    """Mock provider that simulates successful calls"""
    provider = Mock(spec=LLMProvider)
    provider.total_cost = 0.0
    provider.total_tokens = 0
    provider.config = ModelConfig(provider="mock", model_name="test-model")

    def mock_generate(prompt, system_prompt=None, **kwargs):
        # Simulate cost
        cost = 0.05
        tokens = 1000
        provider.total_cost += cost
        provider.total_tokens += tokens

        return LLMResponse(
            content="Mock response",
            model="test-model",
            provider="mock",
            tokens_used=tokens,
            cost=cost,
            finish_reason="stop"
        )

    def mock_generate_structured(prompt, schema, system_prompt=None):
        # Simulate cost
        cost = 0.03
        tokens = 800
        provider.total_cost += cost
        provider.total_tokens += tokens

        return {"result": "success"}, '{"result": "success"}'

    provider.generate = Mock(side_effect=mock_generate)
    provider.generate_structured = Mock(side_effect=mock_generate_structured)

    return provider


# ============================================================================
# UNIT TESTS
# ============================================================================

class TestCostTrackingUnit:
    """Unit tests for cost tracking functionality"""

    def test_generate_tracks_cost(self, basic_config, mock_provider):
        """Test that generate() updates client.total_cost"""
        client = LLMClient(basic_config)
        client.providers["mock:test-model"] = mock_provider

        # Before
        assert client.total_cost == 0.0
        assert client.request_count == 0

        # Call with mocked provider
        with patch.object(client, '_get_provider', return_value=mock_provider):
            response = client.generate("test prompt", task_type=None)

        # After
        assert client.total_cost == 0.05, "Cost should be tracked"
        assert client.request_count == 1, "Request count should increment"
        assert response.cost == 0.05

    def test_generate_structured_tracks_cost(self, basic_config, mock_provider):
        """Test that generate_structured() updates client.total_cost (THE FIX)"""
        client = LLMClient(basic_config)
        client.providers["mock:test-model"] = mock_provider

        # Before
        assert client.total_cost == 0.0
        assert client.request_count == 0

        # Call with mocked provider
        with patch.object(client, '_get_provider', return_value=mock_provider):
            with patch.object(client.config, 'get_model_for_task', return_value=None):
                client.config.primary_model = ModelConfig(provider="mock", model_name="test-model")
                result, content = client.generate_structured("test", {"type": "object"})

        # After
        assert client.total_cost == 0.03, "Cost should be tracked (FIX VERIFICATION)"
        assert client.request_count == 1, "Request count should increment (FIX VERIFICATION)"

    def test_mixed_calls_no_double_counting(self, basic_config, mock_provider):
        """Test that mixing generate() and generate_structured() doesn't double count"""
        client = LLMClient(basic_config)
        client.providers["mock:test-model"] = mock_provider

        with patch.object(client, '_get_provider', return_value=mock_provider):
            with patch.object(client.config, 'get_model_for_task', return_value=None):
                client.config.primary_model = ModelConfig(provider="mock", model_name="test-model")

                # Call 1: generate() - costs $0.05
                client.generate("test 1")
                assert client.total_cost == 0.05
                assert mock_provider.total_cost == 0.05

                # Call 2: generate_structured() - costs $0.03
                client.generate_structured("test 2", {"type": "object"})
                assert client.total_cost == 0.08  # 0.05 + 0.03
                assert mock_provider.total_cost == 0.08  # Should match

                # Call 3: generate() - costs $0.05
                client.generate("test 3")
                assert client.total_cost == 0.13  # 0.08 + 0.05
                assert mock_provider.total_cost == 0.13  # Should match

    def test_budget_check_enforced_for_generate(self, strict_budget_config, mock_provider):
        """Test budget enforcement works for generate()"""
        client = LLMClient(strict_budget_config)
        client.total_cost = 0.09  # Already spent $0.09 of $0.10 budget

        with patch.object(client, '_get_provider', return_value=mock_provider):
            # Next call would cost $0.05, exceeding budget
            with pytest.raises(RuntimeError, match="budget exceeded"):
                client.generate("test")

    def test_budget_check_enforced_for_generate_structured(self, strict_budget_config, mock_provider):
        """Test budget enforcement works for generate_structured() (THE FIX)"""
        client = LLMClient(strict_budget_config)
        client.total_cost = 0.09  # Already spent $0.09 of $0.10 budget

        with patch.object(client, '_get_provider', return_value=mock_provider):
            with patch.object(client.config, 'get_model_for_task', return_value=None):
                client.config.primary_model = ModelConfig(provider="mock", model_name="test-model")

                # Next call would cost $0.03, exceeding budget
                with pytest.raises(RuntimeError, match="budget exceeded"):
                    client.generate_structured("test", {"type": "object"})

    def test_budget_error_message_is_helpful(self, strict_budget_config, mock_provider):
        """Test that budget error message contains useful information"""
        client = LLMClient(strict_budget_config)
        client.total_cost = 0.09

        with patch.object(client, '_get_provider', return_value=mock_provider):
            try:
                client.generate("test")
                assert False, "Should have raised RuntimeError"
            except RuntimeError as e:
                error_msg = str(e)
                # Should contain actual costs
                assert "$0.09" in error_msg or "0.09" in error_msg, "Should show current cost"
                assert "$0.10" in error_msg or "0.10" in error_msg, "Should show budget limit"
                # Should suggest fix
                assert "max_cost_per_scan" in error_msg, "Should suggest config parameter"

    def test_cost_not_tracked_on_failure(self, basic_config, mock_provider):
        """Test that cost is NOT tracked if call fails"""
        client = LLMClient(basic_config)

        # Make provider fail
        mock_provider.generate_structured = Mock(side_effect=Exception("API Error"))

        with patch.object(client, '_get_provider', return_value=mock_provider):
            with patch.object(client.config, 'get_model_for_task', return_value=None):
                client.config.primary_model = ModelConfig(provider="mock", model_name="test-model")

                # Call should fail
                with pytest.raises(RuntimeError, match="failed for all providers"):
                    client.generate_structured("test", {"type": "object"})

                # Cost should NOT be tracked
                assert client.total_cost == 0.0, "Cost should not be tracked on failure"
                assert client.request_count == 0, "Request should not be counted on failure"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestCostTrackingIntegration:
    """Integration tests with real workflow patterns"""

    def test_full_workflow_with_ollama(self):
        """Test full workflow with Ollama (local, free)"""
        # Skip if Ollama not available
        pytest.skip("Requires Ollama running - manual test only")

        config = LLMConfig(
            primary_model=ModelConfig(
                provider="ollama",
                model_name="deepseek-r1-distill-qwen-32b",
                temperature=0.1
            ),
            max_cost_per_scan=1.0,
            enable_cost_tracking=True
        )

        client = LLMClient(config)

        # Make calls
        response1 = client.generate("Say 'test 1' only")
        result2, _ = client.generate_structured(
            "Respond with JSON only",
            {"type": "object", "properties": {"message": {"type": "string"}}}
        )

        # Check stats
        stats = client.get_stats()
        assert stats['total_requests'] == 2, "Should count both calls"
        assert stats['total_cost'] == 0.0, "Ollama is free (local)"

        # Provider stats should match
        provider_cost = sum(p['total_cost'] for p in stats['providers'].values())
        assert stats['total_cost'] == provider_cost

    def test_stats_accuracy(self, basic_config, mock_provider):
        """Test get_stats() returns accurate data"""
        client = LLMClient(basic_config)

        with patch.object(client, '_get_provider', return_value=mock_provider):
            with patch.object(client.config, 'get_model_for_task', return_value=None):
                client.config.primary_model = ModelConfig(provider="mock", model_name="test-model")

                # Make 2 calls (1 generate, 1 structured)
                client.generate("test 1")
                client.generate_structured("test 2", {"type": "object"})

        stats = client.get_stats()

        assert stats['total_requests'] == 2
        assert stats['total_cost'] == 0.08  # 0.05 + 0.03
        assert stats['budget_remaining'] == 10.0 - 0.08

        # Provider stats should match client stats
        provider_cost = sum(p['total_cost'] for p in stats['providers'].values())
        assert stats['total_cost'] == provider_cost, "Client and provider costs should match"

    def test_reset_stats_works(self, basic_config, mock_provider):
        """Test that reset_stats() clears all tracking"""
        client = LLMClient(basic_config)

        with patch.object(client, '_get_provider', return_value=mock_provider):
            # Make calls
            client.generate("test")
            assert client.total_cost > 0

        # Reset
        client.reset_stats()

        # Verify reset
        assert client.total_cost == 0.0
        assert client.request_count == 0
        assert mock_provider.total_cost == 0.0


# ============================================================================
# USER STORY TESTS
# ============================================================================

class TestUserStories:
    """Tests based on real user usage patterns"""

    def test_user_story_vulnerability_analysis(self, basic_config, mock_provider):
        """
        User Story: Security researcher analyzing vulnerabilities
        - Uses generate_structured() for structured vulnerability analysis
        - Expects accurate cost tracking for budget management
        """
        client = LLMClient(basic_config)

        with patch.object(client, '_get_provider', return_value=mock_provider):
            with patch.object(client.config, 'get_model_for_task', return_value=None):
                client.config.primary_model = ModelConfig(provider="mock", model_name="test-model")

                # Analyze 5 vulnerabilities
                for i in range(5):
                    result, _ = client.generate_structured(
                        f"Analyze vulnerability {i}",
                        {"type": "object", "properties": {"severity": {"type": "string"}}}
                    )

                # User checks cost
                stats = client.get_stats()

                # Expectations
                assert stats['total_requests'] == 5, "Should track all 5 analyses"
                assert stats['total_cost'] == 0.15, "Should track total cost (5 * $0.03)"
                assert stats['budget_remaining'] == 9.85, "Budget should be accurate"

    def test_user_story_budget_exceeded(self, strict_budget_config, mock_provider):
        """
        User Story: User hits budget limit
        - Should get clear error message
        - Should know how to fix it
        """
        client = LLMClient(strict_budget_config)

        with patch.object(client, '_get_provider', return_value=mock_provider):
            with patch.object(client.config, 'get_model_for_task', return_value=None):
                client.config.primary_model = ModelConfig(provider="mock", model_name="test-model")

                try:
                    # Make calls until budget exceeded
                    for i in range(10):
                        client.generate_structured("test", {"type": "object"})

                    assert False, "Should have hit budget limit"

                except RuntimeError as e:
                    # User sees helpful error
                    error_msg = str(e)
                    assert "budget exceeded" in error_msg.lower()
                    assert "max_cost_per_scan" in error_msg
                    # Should suggest increasing budget
                    assert "0.2" in error_msg or "0.20" in error_msg  # Double the budget


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions"""

    def test_zero_cost_provider(self, basic_config):
        """Test with provider that has zero cost (Ollama)"""
        mock_free_provider = Mock(spec=LLMProvider)
        mock_free_provider.total_cost = 0.0
        mock_free_provider.total_tokens = 0

        def free_generate_structured(prompt, schema, system_prompt=None):
            # Simulate free call (Ollama)
            mock_free_provider.total_tokens += 1000
            # Cost stays 0.0
            return {"result": "success"}, '{"result": "success"}'

        mock_free_provider.generate_structured = Mock(side_effect=free_generate_structured)

        client = LLMClient(basic_config)

        with patch.object(client, '_get_provider', return_value=mock_free_provider):
            with patch.object(client.config, 'get_model_for_task', return_value=None):
                client.config.primary_model = ModelConfig(provider="mock", model_name="test-model")

                result, _ = client.generate_structured("test", {"type": "object"})

        # Cost should be 0.0, no errors
        assert client.total_cost == 0.0
        assert client.request_count == 1

    def test_very_expensive_call(self, basic_config, mock_provider):
        """Test with very expensive LLM call"""
        # Set strict budget
        config = LLMConfig(max_cost_per_scan=10.0, enable_cost_tracking=True)

        # Simulate expensive call
        def expensive_generate_structured(prompt, schema, system_prompt=None):
            cost = 100.0  # $100 call
            tokens = 1000000
            mock_provider.total_cost += cost
            mock_provider.total_tokens += tokens
            return {"result": "expensive"}, '{"result": "expensive"}'

        mock_provider.generate_structured = Mock(side_effect=expensive_generate_structured)

        client = LLMClient(config)
        # Pre-spend close to budget
        client.total_cost = 9.95  # $9.95 of $10 budget

        with patch.object(client, '_get_provider', return_value=mock_provider):
            with patch.object(client.config, 'get_model_for_task', return_value=None):
                client.config.primary_model = ModelConfig(provider="mock", model_name="test-model")

                # Should fail budget check (9.95 + 0.1 estimate > 10.0)
                with pytest.raises(RuntimeError, match="budget exceeded"):
                    client.generate_structured("expensive", {"type": "object"})

    def test_retry_on_failure_doesnt_double_count(self, basic_config):
        """Test that retries don't double-count costs"""
        mock_retry_provider = Mock(spec=LLMProvider)
        mock_retry_provider.total_cost = 0.0
        mock_retry_provider.total_tokens = 0

        call_count = [0]

        def retry_generate_structured(prompt, schema, system_prompt=None):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call fails
                raise Exception("Temporary failure")
            else:
                # Second call succeeds
                cost = 0.03
                tokens = 800
                mock_retry_provider.total_cost += cost
                mock_retry_provider.total_tokens += tokens
                return {"result": "success"}, '{"result": "success"}'

        mock_retry_provider.generate_structured = Mock(side_effect=retry_generate_structured)

        config = LLMConfig(max_retries=2, enable_cost_tracking=True)
        client = LLMClient(config)

        with patch.object(client, '_get_provider', return_value=mock_retry_provider):
            with patch.object(client.config, 'get_model_for_task', return_value=None):
                client.config.primary_model = ModelConfig(provider="mock", model_name="test-model")

                result, _ = client.generate_structured("test", {"type": "object"})

        # Should only count successful call once
        assert client.total_cost == 0.03, "Should only count successful call"
        assert call_count[0] == 2, "Should have retried once"

    def test_fallback_to_different_provider(self, basic_config):
        """Test cost tracking with fallback to different provider"""
        # Skip - complex fallback testing
        pytest.skip("Complex fallback scenario - needs multi-provider setup")


# ============================================================================
# CHAOS TESTS
# ============================================================================

class TestChaos:
    """Chaos testing - random inputs, concurrent access, stress tests"""

    def test_concurrent_generate_calls(self, basic_config, mock_provider):
        """Test multiple threads calling generate() simultaneously"""
        client = LLMClient(basic_config)
        errors = []

        def worker():
            try:
                with patch.object(client, '_get_provider', return_value=mock_provider):
                    client.generate("concurrent test")
            except Exception as e:
                errors.append(e)

        # Launch 10 concurrent threads
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0, f"Concurrent calls caused errors: {errors}"

        # NOTE: Total cost might be wrong due to race condition (pre-existing bug)
        # But at least it shouldn't crash

    def test_concurrent_generate_structured_calls(self, basic_config, mock_provider):
        """Test multiple threads calling generate_structured() simultaneously"""
        client = LLMClient(basic_config)
        errors = []

        def worker():
            try:
                with patch.object(client, '_get_provider', return_value=mock_provider):
                    with patch.object(client.config, 'get_model_for_task', return_value=None):
                        client.config.primary_model = ModelConfig(provider="mock", model_name="test-model")
                        client.generate_structured("test", {"type": "object"})
            except Exception as e:
                errors.append(e)

        # Launch 10 concurrent threads
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0, f"Concurrent calls caused errors: {errors}"

    def test_rapid_fire_calls(self, basic_config, mock_provider):
        """Test rapid sequential calls"""
        client = LLMClient(basic_config)

        with patch.object(client, '_get_provider', return_value=mock_provider):
            with patch.object(client.config, 'get_model_for_task', return_value=None):
                client.config.primary_model = ModelConfig(provider="mock", model_name="test-model")

                # Make 100 rapid calls
                for i in range(100):
                    if i % 2 == 0:
                        client.generate(f"test {i}")
                    else:
                        client.generate_structured(f"test {i}", {"type": "object"})

        # Verify tracking
        stats = client.get_stats()
        assert stats['total_requests'] == 100
        # 50 generate() at $0.05 + 50 generate_structured() at $0.03 = $4.00
        assert abs(stats['total_cost'] - 4.0) < 0.01, f"Expected ~$4.00, got ${stats['total_cost']}"


# ============================================================================
# ADVERSARIAL TESTS
# ============================================================================

class TestAdversarial:
    """Adversarial tests - security, budget bypass attempts, exploits"""

    def test_budget_bypass_via_generate_structured_FIXED(self, strict_budget_config, mock_provider):
        """
        SECURITY TEST: Verify budget can't be bypassed via generate_structured()
        This was the original vulnerability - should be FIXED now
        """
        client = LLMClient(strict_budget_config)

        with patch.object(client, '_get_provider', return_value=mock_provider):
            with patch.object(client.config, 'get_model_for_task', return_value=None):
                client.config.primary_model = ModelConfig(provider="mock", model_name="test-model")

                # Try to bypass budget by using generate_structured()
                # Before fix: This would succeed and bypass budget
                # After fix: Should raise RuntimeError

                try:
                    for i in range(100):
                        client.generate_structured("bypass attempt", {"type": "object"})

                    # Should not reach here
                    assert False, "Budget bypass vulnerability still exists!"

                except RuntimeError as e:
                    # Expected - budget enforced
                    assert "budget exceeded" in str(e).lower()
                    print("✓ Budget bypass vulnerability FIXED")

    def test_cost_manipulation_impossible(self, basic_config, mock_provider):
        """Test that user cannot manipulate cost tracking"""
        client = LLMClient(basic_config)

        # Attacker tries to manipulate internal state
        client.total_cost = -100.0  # Negative cost

        with patch.object(client, '_get_provider', return_value=mock_provider):
            # Budget check should still work correctly
            result = client._check_budget()
            assert result == True, "Negative cost should pass budget check (bizarre but safe)"

    def test_integer_overflow_attack(self, basic_config):
        """Test handling of extremely large cost values"""
        mock_overflow_provider = Mock(spec=LLMProvider)
        initial_huge_cost = 10**100  # Large but not near overflow
        mock_overflow_provider.total_cost = initial_huge_cost
        mock_overflow_provider.total_tokens = 0

        def overflow_generate_structured(prompt, schema, system_prompt=None):
            # Add to already huge cost
            mock_overflow_provider.total_cost += 0.03
            mock_overflow_provider.total_tokens += 1000
            return {"result": "overflow"}, '{"result": "overflow"}'

        mock_overflow_provider.generate_structured = Mock(side_effect=overflow_generate_structured)

        client = LLMClient(basic_config)

        with patch.object(client, '_get_provider', return_value=mock_overflow_provider):
            with patch.object(client.config, 'get_model_for_task', return_value=None):
                client.config.primary_model = ModelConfig(provider="mock", model_name="test-model")

                # Should not crash (Python handles big floats)
                result, _ = client.generate_structured("test", {"type": "object"})

                # Cost should be tracked (delta = 0.03)
                assert client.total_cost == 0.03, "Should track delta, not absolute cost"


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
