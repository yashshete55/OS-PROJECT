"""
tests/test_agent.py — Comprehensive unit tests for LLM integration, tool calling,
error diagnostics, conversation history, and security.

All external API calls and Linux commands are mocked so that tests run
instantly on any platform without requiring live LLM credentials or Linux hardware.
"""

import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hardware_agent.agent as agent_module
from hardware_agent.agent import (
    OpenAIAgent,
    OfflineAgent,
    create_agent,
    execute_tool,
    handle_hardware_query,
    diagnose_openai_error,
    format_error_response,
    format_tool_result,
    _build_tool_definitions_openai,
)
from hardware_agent.config import (
    validate_config,
    is_placeholder_key,
    is_supported_openai_model,
)
from hardware_agent.system_utils import redact_secrets, run_command


# =====================================================================
# 1. ENVIRONMENT-VARIABLE LOADING & CONFIGURATION TESTS
# =====================================================================

class TestConfigLoading(unittest.TestCase):
    """Tests for configuration loading, validation, and model checks."""

    def test_placeholder_key_detection(self):
        """Test that placeholder API keys are detected."""
        self.assertTrue(is_placeholder_key("your_api_key_here"))
        self.assertTrue(is_placeholder_key("your-openai-api-key"))
        self.assertTrue(is_placeholder_key("sk-..."))
        self.assertTrue(is_placeholder_key(""))
        self.assertFalse(is_placeholder_key("sk-proj-validrealproductionkey1234567890"))

    def test_supported_model_detection(self):
        """Test that compatible and incompatible OpenAI models are identified."""
        self.assertTrue(is_supported_openai_model("gpt-4o"))
        self.assertTrue(is_supported_openai_model("gpt-4o-mini"))
        self.assertTrue(is_supported_openai_model("gpt-4-turbo"))
        self.assertFalse(is_supported_openai_model("text-davinci-003"))
        self.assertFalse(is_supported_openai_model("text-embedding-3-small"))
        self.assertFalse(is_supported_openai_model("davinci-002"))

    def test_validate_config_detects_missing_openai_key(self):
        """Test that validate_config flags empty OPENAI_API_KEY when provider is openai."""
        with patch("hardware_agent.config.LLM_PROVIDER", "openai"), \
             patch("hardware_agent.config.OPENAI_API_KEY", ""):
            warnings = validate_config()
            self.assertTrue(any("OPENAI_API_KEY is not set" in w for w in warnings))

    def test_validate_config_detects_placeholder_key(self):
        """Test that validate_config flags placeholder keys."""
        with patch("hardware_agent.config.LLM_PROVIDER", "openai"), \
             patch("hardware_agent.config.OPENAI_API_KEY", "your_api_key_here"):
            warnings = validate_config()
            self.assertTrue(any("placeholder" in w for w in warnings))

    def test_validate_config_detects_unsupported_provider(self):
        """Test that an invalid provider generates a warning."""
        with patch("hardware_agent.config.LLM_PROVIDER", "anthropic"):
            warnings = validate_config()
            self.assertTrue(any("Unsupported LLM provider" in w for w in warnings))


# =====================================================================
# 2. PROVIDER SELECTION & NO SILENT SWITCHING
# =====================================================================

class TestProviderSelection(unittest.TestCase):
    """Test that provider selection is explicit and never switches silently."""

    @patch("hardware_agent.agent.LLM_PROVIDER", "openai")
    @patch("hardware_agent.agent.OPENAI_API_KEY", "sk-proj-testkey1234567890abcdef")
    @patch("openai.OpenAI")
    def test_selects_openai_agent(self, mock_openai_cls):
        """When LLM_PROVIDER is openai, create_agent must return OpenAIAgent."""
        agent = create_agent()
        self.assertIsInstance(agent, OpenAIAgent)

    @patch("hardware_agent.agent.LLM_PROVIDER", "offline")
    def test_selects_offline_agent(self):
        """When LLM_PROVIDER is offline, create_agent returns OfflineAgent."""
        agent = create_agent()
        self.assertIsInstance(agent, OfflineAgent)

    @patch("hardware_agent.agent.LLM_PROVIDER", "invalid_provider")
    def test_invalid_provider_raises_error(self):
        """Unsupported provider must raise ValueError rather than silent fallback."""
        with self.assertRaises(ValueError):
            create_agent()

    @patch("hardware_agent.agent.LLM_PROVIDER", "openai")
    @patch("hardware_agent.agent.OPENAI_API_KEY", "")
    def test_openai_missing_key_raises_error_not_silent_fallback(self):
        """OpenAI provider without key must raise ValueError, NOT silently switch to offline."""
        with self.assertRaises(ValueError) as ctx:
            create_agent()
        self.assertIn("OPENAI_API_KEY is not set", str(ctx.exception))


# =====================================================================
# 3. API FAILURE DIAGNOSTICS & SECRET REDACTION
# =====================================================================

class TestErrorDiagnostics(unittest.TestCase):
    """Test classification of distinct OpenAI API errors and secret redaction."""

    def test_secret_redaction(self):
        """Test that OpenAI and Google keys are redacted from text."""
        raw = "Error with key sk-proj-1234567890abcdefABCDEF123456 on endpoint."
        redacted = redact_secrets(raw)
        self.assertNotIn("sk-proj-1234567890abcdef", redacted)
        self.assertIn("[REDACTED_OPENAI_KEY]", redacted)

    def test_diagnose_authentication_error(self):
        """Test HTTP 401 / AuthenticationError diagnosis."""
        class MockAuthError(Exception):
            pass
        MockAuthError.__name__ = "AuthenticationError"

        err = MockAuthError("Incorrect API key provided: sk-proj-1234567890abcdef.")
        diag = diagnose_openai_error(err)

        self.assertEqual(diag["category"], "Authentication Failure (Invalid API Key)")
        self.assertFalse(diag["retryable"])
        self.assertIn("platform.openai.com", diag["troubleshooting"])
        self.assertNotIn("sk-proj-1234567890abcdef", diag["description"])

    def test_diagnose_insufficient_quota(self):
        """Test RateLimitError with quota/billing issue diagnosis."""
        class MockRateLimitError(Exception):
            pass
        MockRateLimitError.__name__ = "RateLimitError"

        err = MockRateLimitError("You exceeded your current quota, please check your plan and billing details.")
        diag = diagnose_openai_error(err)

        self.assertEqual(diag["category"], "Insufficient Quota / Billing Restriction")
        self.assertFalse(diag["retryable"])
        self.assertIn("billing", diag["troubleshooting"])

    def test_diagnose_standard_rate_limit(self):
        """Test standard TPM/RPM rate limit diagnosis."""
        class MockRateLimitError(Exception):
            pass
        MockRateLimitError.__name__ = "RateLimitError"

        err = MockRateLimitError("Rate limit reached for requests per minute (RPM).")
        diag = diagnose_openai_error(err)

        self.assertEqual(diag["category"], "Rate Limit Exceeded")
        self.assertTrue(diag["retryable"])

    def test_diagnose_model_not_found(self):
        """Test NotFoundError / model not found diagnosis."""
        class MockNotFoundError(Exception):
            pass
        MockNotFoundError.__name__ = "NotFoundError"

        err = MockNotFoundError("The model 'gpt-5-turbo' does not exist.")
        diag = diagnose_openai_error(err)

        self.assertEqual(diag["category"], "Unsupported or Missing Model")
        self.assertFalse(diag["retryable"])

    def test_diagnose_connection_error(self):
        """Test APIConnectionError diagnosis."""
        class MockConnError(Exception):
            pass
        MockConnError.__name__ = "APIConnectionError"

        err = MockConnError("Connection error while connecting to api.openai.com.")
        diag = diagnose_openai_error(err)

        self.assertEqual(diag["category"], "Network / Connection Error")
        self.assertTrue(diag["retryable"])

    def test_diagnose_timeout_error(self):
        """Test APITimeoutError diagnosis."""
        class MockTimeoutError(Exception):
            pass
        MockTimeoutError.__name__ = "APITimeoutError"

        err = MockTimeoutError("Request timed out after 30 seconds.")
        diag = diagnose_openai_error(err)

        self.assertEqual(diag["category"], "Request Timeout")
        self.assertTrue(diag["retryable"])


# =====================================================================
# 4. NATURAL-LANGUAGE CONVERSATION & TOOL SELECTION (OPENAI AGENT)
# =====================================================================

class TestOpenAIAgentConversations(unittest.TestCase):
    """Test OpenAI tool calling, execution, conversation history, and follow-ups."""

    @patch("openai.OpenAI")
    def setUp(self, mock_openai_cls):
        self.mock_client = MagicMock()
        mock_openai_cls.return_value = self.mock_client

        with patch("hardware_agent.agent.OPENAI_API_KEY", "sk-proj-testkey1234567890abcdef"):
            self.agent = OpenAIAgent()

    def test_tool_definitions_generated_properly(self):
        """Test that OpenAI tool definitions match required schema."""
        tools = _build_tool_definitions_openai()
        self.assertTrue(len(tools) >= 10)
        tool_names = [t["function"]["name"] for t in tools]
        self.assertIn("get_memory_info", tool_names)
        self.assertIn("get_cpu_info", tool_names)
        self.assertIn("get_gpu_info", tool_names)
        self.assertIn("get_system_health", tool_names)

        # Check structure
        for t in tools:
            self.assertEqual(t["type"], "function")
            self.assertIn("name", t["function"])
            self.assertIn("description", t["function"])
            self.assertIn("parameters", t["function"])

    def test_ram_question_calls_get_memory_info(self):
        """Test tool execution flow for a RAM question."""
        # 1. Round 1: Model requests get_memory_info tool call
        mock_tc = MagicMock()
        mock_tc.id = "call_mem_1"
        mock_tc.function.name = "get_memory_info"
        mock_tc.function.arguments = "{}"

        round1_choice = MagicMock()
        round1_choice.message.tool_calls = [mock_tc]
        round1_choice.message.content = None
        round1_resp = MagicMock(choices=[round1_choice])

        # 2. Round 2: Model returns natural-language summary after tool result
        round2_choice = MagicMock()
        round2_choice.message.tool_calls = None
        round2_choice.message.content = "You have 16 GB of DDR4 RAM installed with 2 empty slots available."
        round2_resp = MagicMock(choices=[round2_choice])

        self.mock_client.chat.completions.create.side_effect = [round1_resp, round2_resp]

        with patch("hardware_agent.agent.execute_tool") as mock_exec:
            mock_exec.return_value = json.dumps({"total_capacity": "16 GB", "memory_type": "DDR4", "empty_slots": 2})

            answer = self.agent.query("How much RAM do I have?")

            self.assertIn("16 GB of DDR4 RAM", answer)
            mock_exec.assert_called_once_with("get_memory_info", {})

    def test_follow_up_question_preserves_conversation_history(self):
        """Test that follow-up questions retain context in messages list."""
        # Turn 1: Direct answer
        choice1 = MagicMock()
        choice1.message.tool_calls = None
        choice1.message.content = "Your system has 16 GB of DDR4 RAM."
        resp1 = MagicMock(choices=[choice1])

        # Turn 2: Follow-up answer
        choice2 = MagicMock()
        choice2.message.tool_calls = None
        choice2.message.content = "The RAM is running at 3200 MT/s."
        resp2 = MagicMock(choices=[choice2])

        self.mock_client.chat.completions.create.side_effect = [resp1, resp2]

        self.agent.query("Tell me about my RAM.")
        self.agent.query("What about its speed?")

        # Check messages list history
        user_msgs = [m["content"] for m in self.agent.messages if m["role"] == "user"]
        assistant_msgs = [m["content"] for m in self.agent.messages if m["role"] == "assistant"]

        self.assertEqual(user_msgs, ["Tell me about my RAM.", "What about its speed?"])
        self.assertEqual(assistant_msgs, ["Your system has 16 GB of DDR4 RAM.", "The RAM is running at 3200 MT/s."])

    def test_slowness_question_routes_to_system_health(self):
        """Test that questions about slowness invoke get_system_health."""
        mock_tc = MagicMock()
        mock_tc.id = "call_health_1"
        mock_tc.function.name = "get_system_health"
        mock_tc.function.arguments = "{}"

        round1_choice = MagicMock()
        round1_choice.message.tool_calls = [mock_tc]
        round1_choice.message.content = None

        round2_choice = MagicMock()
        round2_choice.message.tool_calls = None
        round2_choice.message.content = "Your CPU load is high (4.50) and available memory is low (512 MB)."

        self.mock_client.chat.completions.create.side_effect = [
            MagicMock(choices=[round1_choice]),
            MagicMock(choices=[round2_choice]),
        ]

        with patch("hardware_agent.agent.execute_tool") as mock_exec:
            mock_exec.return_value = json.dumps({"load_average_1m": "4.50", "available_memory": "512 MB"})
            answer = self.agent.query("Why is my computer running slowly?")
            self.assertIn("load is high", answer)
            mock_exec.assert_called_once_with("get_system_health", {})

    def test_api_error_returns_clean_diagnostic_without_corrupting_history(self):
        """Test that API errors return friendly messages and do not break message history."""
        class MockAuthError(Exception):
            pass
        MockAuthError.__name__ = "AuthenticationError"

        self.mock_client.chat.completions.create.side_effect = MockAuthError("Invalid API key sk-proj-badkey12345.")

        answer = self.agent.query("What CPU do I have?")

        self.assertIn("Authentication Failure", answer)
        self.assertIn("Next Step", answer)
        self.assertNotIn("sk-proj-badkey12345", answer)

        # The unanswered user message should be cleaned up from history
        self.assertEqual(len([m for m in self.agent.messages if m["role"] == "user"]), 0)


# =====================================================================
# 5. OFFLINE AGENT & FOLLOW-UP RESOLUTION
# =====================================================================

class TestOfflineAgent(unittest.TestCase):
    """Test OfflineAgent's deterministic routing and follow-up handling."""

    def test_offline_routes_ram_question(self):
        agent = OfflineAgent()
        with patch("hardware_agent.agent.search_hardware_info") as mock_search:
            mock_search.return_value = {
                "category": "memory",
                "data": {"total_capacity": "32 GB", "speed": "3600 MT/s", "errors": []},
            }
            answer = agent.query("How much RAM do I have?")

        self.assertIn("32 GB", answer)
        self.assertNotIn("{", answer)  # Must be plain English, not raw JSON

    def test_offline_follow_up_resolves_previous_topic(self):
        agent = OfflineAgent()
        with patch("hardware_agent.agent.search_hardware_info") as mock_search:
            mock_search.side_effect = [
                {"category": "memory", "data": {"total_capacity": "16 GB", "speed": "3200 MT/s", "errors": []}},
                {"category": "memory", "data": {"total_capacity": "16 GB", "speed": "3200 MT/s", "errors": []}},
            ]
            agent.query("Tell me about my RAM.")
            answer2 = agent.query("What about its speed?")

        self.assertIn("3200 MT/s", answer2)


# =====================================================================
# 6. TOOL DISPATCH & ARBITRARY SHELL EXECUTION PREVENTION
# =====================================================================

class TestToolDispatchSafety(unittest.TestCase):
    """Test that tool dispatcher prevents arbitrary command execution."""

    def test_reject_arbitrary_function_name(self):
        """Unknown tool names must be rejected."""
        result = json.loads(execute_tool("os_system", {"command": "rm -rf /"}))
        self.assertIn("error", result)
        self.assertIn("Unknown tool", result["error"])

    def test_zero_argument_tool_ignores_spurious_llm_kwargs(self):
        """Zero-argument tools should not crash when LLM passes extra kwargs."""
        with patch.dict(agent_module.TOOL_REGISTRY["get_cpu_info"], {"function": lambda: {"model_name": "Intel Core i9"}}):
            # LLM passes unexpected kwarg 'details': True
            result = json.loads(execute_tool("get_cpu_info", {"details": True}))
            self.assertNotIn("error", result)
            self.assertEqual(result["model_name"], "Intel Core i9")


# =====================================================================
# 7. OS LIMITATION HANDLING & MISSING UTILITIES
# =====================================================================

class TestOSLimitations(unittest.TestCase):
    """Test that operating system limitations are reported accurately."""

    def test_missing_utility_gives_package_guidance_on_linux(self):
        with patch("hardware_agent.system_utils.is_linux", return_value=True), \
             patch("hardware_agent.system_utils.shutil.which", return_value=None):
            result = run_command(["dmidecode", "-t", "memory"])
            self.assertFalse(result["success"])
            self.assertIn("sudo apt install dmidecode", result["error"])

    def test_windows_environment_limitation_reported_accurately(self):
        with patch("hardware_agent.system_utils.is_linux", return_value=False), \
             patch("hardware_agent.system_utils.shutil.which", return_value=None):
            result = run_command(["lscpu"])
            self.assertFalse(result["success"])
            self.assertIn("WSL2", result["error"])


# =====================================================================
# 8. MASTER SHELL INTERFACE
# =====================================================================

class TestMasterShellInterface(unittest.TestCase):
    """Test public handle_hardware_query interface."""

    def test_empty_query_handling(self):
        answer = handle_hardware_query("")
        self.assertIn("Please provide", answer)

    def test_none_query_handling(self):
        answer = handle_hardware_query(None)
        self.assertIn("Please provide", answer)

    @patch("hardware_agent.agent._agent_instance", None)
    @patch("hardware_agent.agent.LLM_PROVIDER", "offline")
    def test_successful_query_flow(self):
        answer = handle_hardware_query("How many CPU cores do I have?")
        self.assertIsInstance(answer, str)
        self.assertTrue(len(answer) > 0)


if __name__ == "__main__":
    unittest.main()
