"""
agent.py — LLM integration and tool-calling logic for the Hardware Introspection Agent.

This module connects the hardware inspection tools to an LLM supporting
function/tool calling (OpenAI and Google Gemini), maintaining conversational
context, executing tools securely, and translating OS data into natural language.
"""

import json
import re
import time
from typing import Optional, Dict, Any, List

from hardware_agent.config import (
    LLM_PROVIDER,
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
    GEMINI_MODEL,
    OPENAI_MODEL,
    SYSTEM_PROMPT,
    AGENT_NAME,
    AGENT_ID,
    AGENT_VERSION,
    validate_config,
    is_placeholder_key,
    is_supported_openai_model,
)
from hardware_agent.system_utils import redact_secrets
from hardware_agent.tools import TOOL_REGISTRY, search_hardware_info


# =====================================================================
# ERROR DIAGNOSTICS & SECRET SANITIZATION
# =====================================================================

def diagnose_openai_error(error: Exception) -> Dict[str, Any]:
    """
    Classify an OpenAI API error into a specific failure category
    with actionable troubleshooting steps, ensuring secrets are redacted.
    """
    err_str = redact_secrets(str(error))
    err_type = type(error).__name__

    # 1. Authentication / Invalid Key (401)
    if "AuthenticationError" in err_type or "401" in err_str or "invalid_api_key" in err_str.lower():
        return {
            "category": "Authentication Failure (Invalid API Key)",
            "description": "The configured OpenAI API key is invalid, revoked, or formatted incorrectly.",
            "troubleshooting": (
                "Verify OPENAI_API_KEY in your .env file. Ensure there are no extra quotes, spaces, "
                "or placeholders. You can generate a fresh API key at: https://platform.openai.com/api-keys"
            ),
            "retryable": False,
        }

    # 2. Insufficient Quota / Billing Restrictions (429 with quota)
    if (
        "RateLimitError" in err_type and any(kw in err_str.lower() for kw in ["quota", "billing", "insufficient_quota"])
    ) or "insufficient_quota" in err_str.lower():
        return {
            "category": "Insufficient Quota / Billing Restriction",
            "description": "Your OpenAI account has reached its spending limit or has insufficient credit balance.",
            "troubleshooting": (
                "Check your OpenAI account balance, active payment method, and usage tier at: "
                "https://platform.openai.com/account/billing"
            ),
            "retryable": False,
        }

    # 3. Standard Rate Limit (429)
    if "RateLimitError" in err_type or "429" in err_str:
        return {
            "category": "Rate Limit Exceeded",
            "description": "Too many requests or tokens sent to the OpenAI API in a short timeframe.",
            "troubleshooting": "Please wait a moment before sending another request, or reduce query frequency.",
            "retryable": True,
        }

    # 4. Model Not Found / Unsupported Model Name (404)
    if "NotFoundError" in err_type or "404" in err_str or "model_not_found" in err_str.lower():
        return {
            "category": "Unsupported or Missing Model",
            "description": f"The configured model '{OPENAI_MODEL}' was not found or your account lacks access to it.",
            "troubleshooting": (
                f"Update OPENAI_MODEL in .env to a supported model you have access to, "
                "such as 'gpt-4o-mini' or 'gpt-4o'."
            ),
            "retryable": False,
        }

    # 5. Bad Request / Tool Calling Unsupported (400)
    if "BadRequestError" in err_type or "400" in err_str:
        if any(kw in err_str.lower() for kw in ["tool", "function", "parameters"]):
            return {
                "category": "Tool Calling Unsupported / Invalid Tool Parameters",
                "description": f"The configured model '{OPENAI_MODEL}' does not support the required tool-calling schema.",
                "troubleshooting": (
                    "Switch to a standard tool-calling chat model in .env: set OPENAI_MODEL=gpt-4o-mini or OPENAI_MODEL=gpt-4o."
                ),
                "retryable": False,
            }
        return {
            "category": "Invalid Request Parameters",
            "description": f"OpenAI API rejected the request parameters: {err_str}",
            "troubleshooting": "Check your model parameters, prompt format, or input length.",
            "retryable": False,
        }

    # 6. Request Timeout
    if "APITimeoutError" in err_type or "timeout" in err_str.lower():
        return {
            "category": "Request Timeout",
            "description": "The request to the OpenAI API timed out before receiving a response.",
            "troubleshooting": "Check your internet connection stability and speed, then try again.",
            "retryable": True,
        }

    # 7. Network / Connection Errors
    if "APIConnectionError" in err_type or any(kw in err_str.lower() for kw in ["connection", "connect", "dns", "proxy"]):
        return {
            "category": "Network / Connection Error",
            "description": "Failed to establish a network connection to OpenAI API servers.",
            "troubleshooting": "Check your internet connectivity, local DNS settings, proxy, or firewall configuration.",
            "retryable": True,
        }

    # 8. Permission Denied (403)
    if "PermissionDeniedError" in err_type or "403" in err_str:
        return {
            "category": "Permission Denied",
            "description": "Access to the requested OpenAI API endpoint or organization was denied.",
            "troubleshooting": "Check project access permissions and country/region availability for your OpenAI account.",
            "retryable": False,
        }

    # 9. Missing Dependencies
    if isinstance(error, ImportError):
        return {
            "category": "Missing Python Package",
            "description": err_str,
            "troubleshooting": "Install required dependencies with: pip install -r requirements.txt",
            "retryable": False,
        }

    # 10. Configuration / Value Errors
    if isinstance(error, ValueError):
        return {
            "category": "Configuration Error",
            "description": err_str,
            "troubleshooting": "Review and update your configuration in .env.",
            "retryable": False,
        }

    # Fallback / Unexpected
    return {
        "category": "Unexpected API Error",
        "description": err_str or err_type,
        "troubleshooting": "Review error details or try again. Verify your network connection and API status at https://status.openai.com",
        "retryable": False,
    }


def format_error_response(diagnostic: Dict[str, Any]) -> str:
    """Format a diagnostic dictionary into a user-friendly terminal response."""
    return (
        f"[API Error - {diagnostic['category']}]:\n"
        f"   {diagnostic['description']}\n\n"
        f"Next Step: {diagnostic['troubleshooting']}"
    )


# =====================================================================
# NATURAL-LANGUAGE FORMATTING (for Offline and Fallback answers)
# =====================================================================

def _format_value(value: object) -> str:
    """Turn a tool value into human-readable text without raw JSON."""
    if isinstance(value, list):
        items = [_format_value(item) for item in value if item not in ("Unknown", None, "")]
        return ", ".join(items) if items else "none detected"
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            if key in {"errors", "serial_number"} or item in ("Unknown", "", None):
                continue
            label = key.replace("_", " ").capitalize()
            parts.append(f"{label}: {_format_value(item)}")
        return "; ".join(parts) or "not available"
    return str(value)


def format_tool_result(result: Dict[str, Any]) -> str:
    """Create a human-readable English summary from a validated hardware tool result."""
    errors = result.get("errors", [])
    details = []
    unavailable = []

    for key, value in result.items():
        if key in ("errors", "note", "category"):
            continue
        if value in ("Unknown", "", None, []):
            unavailable.append(key.replace("_", " "))
            continue

        label = key.replace("_", " ").capitalize()
        if key == "devices" and isinstance(value, list):
            if value:
                details.append(
                    f"{len(value)} device(s): "
                    + "; ".join(_format_value(item) for item in value)
                )
            else:
                details.append("No devices detected.")
        elif key == "gpus" and isinstance(value, list):
            details.append("Graphics: " + (_format_value(value) if value else "none detected"))
        elif key == "filesystems" and isinstance(value, list):
            fs_summary = "; ".join(
                f"{f.get('mountpoint', '?')}: {f.get('used', '?')}/{f.get('size', '?')} ({f.get('use_percent', '?')})"
                for f in value
            )
            details.append(f"Storage filesystems: {fs_summary}")
        else:
            details.append(f"{label}: {_format_value(value)}")

    answer = "Here is what I detected from your system: " + (
        ". ".join(details) if details else "no confirmed hardware details found"
    ) + "."

    if unavailable:
        answer += " The following information is not available: " + ", ".join(unavailable) + "."
    if errors:
        answer += " Notice: " + "; ".join(str(e) for e in errors) + "."

    return answer


# =====================================================================
# OFFLINE AGENT (Deterministic Conversational Fallback)
# =====================================================================

class OfflineAgent:
    """
    Deterministic conversational hardware agent used when running in offline mode.
    Maintains conversation context and follow-up question references.
    """

    def __init__(self):
        self.history: List[Dict[str, str]] = []
        self.last_category: Optional[str] = None
        self.last_tool_result: Optional[Dict[str, Any]] = None

    def query(self, user_question: str) -> str:
        self.history.append({"role": "user", "content": user_question})
        q_lower = user_question.lower()

        # Check for context-dependent follow-up indicators
        context_words = ["speed", "type", "size", "cores", "threads", "model", "frequency", "how many", "slots", "empty", "it", "its", "that"]
        is_followup = any(w in q_lower for w in context_words) and len(q_lower.split()) <= 6

        # Target query for routing
        target_query = user_question
        if is_followup and self.last_category:
            # Combine last category with follow-up
            target_query = f"{self.last_category} {user_question}"

        result = search_hardware_info(target_query)
        category = result.get("category", self.last_category)
        if category:
            self.last_category = category

        data = result.get("data", result)
        self.last_tool_result = data

        if "error" in result and not data:
            answer = result["error"]
        else:
            # Specific follow-up handling
            if "speed" in q_lower and category == "memory":
                speed = data.get("speed", "Unknown")
                answer = f"Your installed RAM is running at {speed}." if speed != "Unknown" else "Memory clock speed could not be determined without root privileges."
            elif any(w in q_lower for w in ["slot", "slots", "empty"]) and category == "memory":
                total = data.get("total_slots", "Unknown")
                empty = data.get("empty_slots", "Unknown")
                occ = data.get("occupied_slots", "Unknown")
                answer = f"You have {total} total memory slots: {occ} occupied and {empty} empty." if total != "Unknown" else "Memory slot occupancy is not exposed by the firmware without root privileges."
            elif any(w in q_lower for w in ["core", "cores", "thread", "threads"]) and category == "cpu":
                cores = data.get("cores", "Unknown")
                threads = data.get("threads", "Unknown")
                answer = f"Your CPU has {cores} physical core(s) and {threads} thread(s)."
            else:
                answer = format_tool_result(data)

        self.history.append({"role": "assistant", "content": answer})
        return answer


# =====================================================================
# TOOL SCHEMAS FOR LLM TOOL CALLING
# =====================================================================

def _build_tool_definitions_openai() -> List[Dict[str, Any]]:
    """Build OpenAI-compatible tool definitions from TOOL_REGISTRY."""
    tools = []
    for name, info in TOOL_REGISTRY.items():
        params = info.get("parameters", {})
        properties = {}
        required = []

        for pname, pinfo in params.items():
            properties[pname] = {
                "type": pinfo.get("type", "string"),
                "description": pinfo.get("description", ""),
            }
            required.append(pname)

        tool_def = {
            "type": "function",
            "function": {
                "name": name,
                "description": info["description"],
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
        tools.append(tool_def)
    return tools


def _build_tool_definitions_gemini() -> List[Dict[str, Any]]:
    """Build Gemini-compatible function declarations from TOOL_REGISTRY."""
    declarations = []
    for name, info in TOOL_REGISTRY.items():
        params = info.get("parameters", {})
        properties = {}
        required = []

        for pname, pinfo in params.items():
            properties[pname] = {
                "type": pinfo.get("type", "string"),
                "description": pinfo.get("description", ""),
            }
            required.append(pname)

        func_decl = {
            "name": name,
            "description": info["description"],
        }
        if properties:
            func_decl["parameters"] = {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        declarations.append(func_decl)
    return declarations


# =====================================================================
# TOOL EXECUTION — SAFE DISPATCHER
# =====================================================================

def execute_tool(tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
    """
    Execute a registered tool by name with validated arguments.
    Enforces strict allowlist from TOOL_REGISTRY to prevent arbitrary execution.
    """
    if tool_name not in TOOL_REGISTRY:
        return json.dumps({
            "error": f"Unknown tool: '{tool_name}'. Only predefined hardware inspection tools may be executed."
        })

    tool_info = TOOL_REGISTRY[tool_name]
    func = tool_info["function"]
    expected_params = tool_info.get("parameters", {})

    if arguments is None or not isinstance(arguments, dict):
        arguments = {}

    try:
        if not expected_params:
            # Function takes 0 parameters — ignore any spurious arguments generated by LLM
            result = func()
        else:
            # Filter arguments to expected parameters only
            valid_args = {k: v for k, v in arguments.items() if k in expected_params}
            result = func(**valid_args)

        return json.dumps(result, indent=2, default=str)
    except TypeError as e:
        return json.dumps({"error": f"Invalid arguments for '{tool_name}': {e}"})
    except Exception as e:
        return json.dumps({"error": f"Error executing '{tool_name}': {e}"})


# =====================================================================
# OPENAI AGENT
# =====================================================================

class OpenAIAgent:
    """Hardware Introspection Agent powered by the official OpenAI Python SDK."""

    def __init__(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "The 'openai' Python package is required. Install it using: pip install openai>=1.0.0"
            )

        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set or empty. "
                "Please configure OPENAI_API_KEY in your .env file or environment."
            )

        if is_placeholder_key(OPENAI_API_KEY):
            raise ValueError(
                "OPENAI_API_KEY appears to contain a placeholder. "
                "Please replace it with your actual OpenAI API key in .env."
            )

        if not is_supported_openai_model(OPENAI_MODEL):
            raise ValueError(
                f"Configured OPENAI_MODEL '{OPENAI_MODEL}' does not support chat/tool calling. "
                "Please set OPENAI_MODEL=gpt-4o-mini or OPENAI_MODEL=gpt-4o in .env."
            )

        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL
        self.tool_definitions = _build_tool_definitions_openai()
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def query(self, user_question: str) -> str:
        """
        Process a natural-language hardware question via OpenAI.
        Executes tool calls and maintains multi-turn conversation history.
        """
        self.messages.append({"role": "user", "content": user_question})

        max_rounds = 5
        for _ in range(max_rounds):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=self.tool_definitions,
                    tool_choice="auto",
                    temperature=0.1,
                )
            except Exception as e:
                # Remove unanswered user message so history isn't corrupted
                if self.messages and self.messages[-1].get("role") == "user":
                    self.messages.pop()
                diagnostic = diagnose_openai_error(e)
                return format_error_response(diagnostic)

            choice = response.choices[0]

            # If model requested tool execution
            if choice.message.tool_calls:
                # Add assistant message containing the tool calls
                assistant_msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": choice.message.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in choice.message.tool_calls
                    ],
                }
                self.messages.append(assistant_msg)

                for tool_call in choice.message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                    except json.JSONDecodeError:
                        args = {}

                    print(f"  [Tool Call] {tool_name}({args})")

                    tool_result = execute_tool(tool_name, args)

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    })
            else:
                # Final natural-language answer received
                final_answer = choice.message.content or "No response generated."
                self.messages.append({"role": "assistant", "content": final_answer})
                return final_answer

        return "I was unable to complete the hardware inspection within the allowed number of conversation turns."


# =====================================================================
# GEMINI AGENT
# =====================================================================

class GeminiAgent:
    """Hardware agent powered by Google Gemini with function calling."""

    def __init__(self):
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError(
                "google-genai package is required. Install with: pip install google-genai>=1.0.0"
            )

        if not GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY is not set. Add GOOGLE_API_KEY to your .env file or environment."
            )

        if is_placeholder_key(GOOGLE_API_KEY):
            raise ValueError(
                "GOOGLE_API_KEY contains a placeholder value. Replace it with your actual key in .env."
            )

        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model = GEMINI_MODEL
        self.genai = genai
        self.types = types
        self.contents: List[Any] = []
        self.tool_declarations = _build_tool_definitions_gemini()

    def query(self, user_question: str) -> str:
        types = self.types

        tools = types.Tool(
            function_declarations=[
                types.FunctionDeclaration(**decl)
                for decl in self.tool_declarations
            ]
        )

        self.contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_question)],
            )
        )

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[tools],
            temperature=0.1,
        )

        max_rounds = 5
        for _ in range(max_rounds):
            response = None
            for attempt in range(3):
                try:
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=self.contents,
                        config=config,
                    )
                    break
                except Exception as e:
                    error_text = str(e)
                    if attempt == 2 or not any(
                        marker in error_text
                        for marker in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED")
                    ):
                        return f"Google Gemini API error: {redact_secrets(error_text)}"
                    time.sleep(2**attempt)

            if response is None:
                return "Google Gemini API error: no response received after retries."

            candidate = response.candidates[0]
            has_function_call = False

            for part in candidate.content.parts:
                if part.function_call:
                    has_function_call = True
                    fc = part.function_call
                    tool_name = fc.name
                    arguments = dict(fc.args) if fc.args else {}

                    print(f"  [Tool Call] {tool_name}({arguments})")
                    tool_result = execute_tool(tool_name, arguments)

                    self.contents.append(candidate.content)
                    self.contents.append(
                        types.Content(
                            role="user",
                            parts=[
                                types.Part.from_function_response(
                                    name=tool_name,
                                    response=json.loads(tool_result),
                                )
                            ],
                        )
                    )
                    break

            if not has_function_call:
                final_text = candidate.content.parts[0].text
                return final_text

        return "I was unable to complete the analysis within the allowed number of steps."


# =====================================================================
# AGENT FACTORY
# =====================================================================

def create_agent() -> OpenAIAgent | GeminiAgent | OfflineAgent:
    """
    Create and return the agent instance corresponding to LLM_PROVIDER.
    Ensures that when OpenAI is selected, the application actually initializes
    OpenAIAgent rather than silently switching to offline mode.
    """
    warnings = validate_config()
    for w in warnings:
        print(f"[Notice] {w}")

    if LLM_PROVIDER == "openai":
        return OpenAIAgent()
    elif LLM_PROVIDER == "google":
        return GeminiAgent()
    elif LLM_PROVIDER == "offline":
        return OfflineAgent()
    else:
        raise ValueError(
            f"Unsupported LLM provider: '{LLM_PROVIDER}'. "
            "Supported providers: 'openai', 'google', 'offline'."
        )


# =====================================================================
# PUBLIC INTERFACE — FOR MASTER AGENTIC SHELL
# =====================================================================

_agent_instance: Optional[OpenAIAgent | GeminiAgent | OfflineAgent] = None


def handle_hardware_query(user_query: str) -> str:
    """
    Primary interface for the Master Agentic Shell.

    The master shell calls this function with a natural-language hardware
    question and receives a plain-English answer.

    Args:
        user_query: A hardware-related question in natural language.

    Returns:
        A string containing the agent's answer.
    """
    global _agent_instance

    if not user_query or not user_query.strip():
        return "Please provide a hardware-related question."

    try:
        if _agent_instance is None:
            _agent_instance = create_agent()

        return _agent_instance.query(user_query)
    except (ImportError, ValueError) as e:
        diagnostic = diagnose_openai_error(e)
        return format_error_response(diagnostic)
    except Exception as e:
        diagnostic = diagnose_openai_error(e)
        return format_error_response(diagnostic)


# =====================================================================
# AGENT METADATA — FOR DISCOVERY BY MASTER SHELL
# =====================================================================

AGENT_METADATA = {
    "id": AGENT_ID,
    "name": AGENT_NAME,
    "version": AGENT_VERSION,
    "description": (
        "A specialized Linux hardware inspection agent that answers natural-language "
        "questions about system hardware by inspecting the OS. Covers CPU, RAM, GPU, "
        "motherboard, BIOS, storage, USB, PCI, system health, and slowness analysis."
    ),
    "capabilities": [
        "memory_inspection",
        "cpu_inspection",
        "gpu_inspection",
        "motherboard_inspection",
        "usb_device_listing",
        "pci_device_listing",
        "storage_inspection",
        "system_summary",
        "system_health",
        "full_hardware_report",
        "hardware_search",
    ],
    "interface": "handle_hardware_query(user_query: str) -> str",
    "tools": list(TOOL_REGISTRY.keys()),
}
