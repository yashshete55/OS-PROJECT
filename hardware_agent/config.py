"""
config.py — Environment and configuration management for Hardware Introspection Agent.

Loads settings from environment variables or a .env file located in the project root.
Never hardcodes API keys or sensitive credentials.
"""

import os
from pathlib import Path
from typing import List

# ─── Project Paths & Environment Loading ─────────────────────────────
# Resolve project root regardless of current working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

def load_environment() -> None:
    """Load or reload .env file from the project directory."""
    try:
        from dotenv import load_dotenv
        if ENV_PATH.exists():
            load_dotenv(dotenv_path=ENV_PATH, override=True)
        else:
            load_dotenv(override=True)
    except ImportError:
        pass


# Initial load
load_environment()


# ─── Configuration Values ─────────────────────────────────────────────
# Provider: "openai", "google", or "offline"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google").strip().lower()

# API Keys — stripped of surrounding whitespace and quotes
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip().strip("'\"")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip().strip("'\"")

# Model Names
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

# Agent Metadata
AGENT_NAME = "Hardware Introspection Agent"
AGENT_VERSION = "1.0.0"
AGENT_ID = "hardware_introspection_agent"

# Command Execution Timeout in seconds
try:
    COMMAND_TIMEOUT = int(os.getenv("COMMAND_TIMEOUT", "10"))
except ValueError:
    COMMAND_TIMEOUT = 10

# Security: Mask serial numbers in hardware output
MASK_SERIALS = os.getenv("MASK_SERIALS", "false").strip().lower() in ("true", "1", "yes")

# Known placeholders that should not be treated as real keys
PLACEHOLDER_SUBSTRINGS = (
    "your_api_key_here",
    "your-openai-api-key",
    "your_openai_api_key",
    "your-google-api-key",
    "your_google_api_key_here",
    "placeholder",
    "sk-...",
)

# Known OpenAI model prefixes that DO NOT support chat/tool calling
UNSUPPORTED_OPENAI_PREFIXES = (
    "text-",
    "davinci",
    "babbage",
    "curie",
    "ada",
    "whisper",
    "tts",
    "dall-e",
    "text-embedding",
)

# ─── System Prompt for the LLM ───────────────────────────────────────
SYSTEM_PROMPT = """You are a Hardware Introspection Agent — a specialized Linux hardware
inspection assistant in a next-generation Agentic Shell. Your role is to answer questions about the user's computer
hardware by calling the appropriate inspection tools.

Rules:
1. ONLY report information returned by the tools. Never invent or guess hardware specs.
2. If a tool returns an error or "Unknown", tell the user honestly.
3. If elevated privileges (sudo) are needed for full details (such as dmidecode for RAM slots or BIOS), explain this clearly.
4. Keep answers clear, concise, and in plain English.
5. When presenting numeric data (RAM size, frequencies, disk capacity), use appropriate units.
6. For a "complete hardware report", call the get_hardware_report tool.
7. For questions about why the computer is running slowly, check system health (load averages, available memory) using get_system_health.
8. Maintain conversation context: if the user asks a follow-up question (e.g. "What about its speed?"), refer back to previously discussed hardware or call the relevant tool again.
9. Do NOT suggest running arbitrary shell commands. Only use the provided tools.
"""


def reload_config() -> None:
    """Reload environment variables from .env and refresh module-level constants."""
    global LLM_PROVIDER, OPENAI_API_KEY, GOOGLE_API_KEY, OPENAI_MODEL, GEMINI_MODEL, COMMAND_TIMEOUT, MASK_SERIALS
    load_environment()
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google").strip().lower()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip().strip("'\"")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip().strip("'\"")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
    try:
        COMMAND_TIMEOUT = int(os.getenv("COMMAND_TIMEOUT", "10"))
    except ValueError:
        COMMAND_TIMEOUT = 10
    MASK_SERIALS = os.getenv("MASK_SERIALS", "false").strip().lower() in ("true", "1", "yes")


def get_api_key() -> str:
    """Return the active API key for the configured LLM provider."""
    if LLM_PROVIDER == "openai":
        return OPENAI_API_KEY
    elif LLM_PROVIDER == "google":
        return GOOGLE_API_KEY
    return ""


def is_placeholder_key(key: str) -> bool:
    """Return True if the key is empty or matches known placeholder patterns."""
    if not key:
        return True
    key_lower = key.lower()
    return any(p in key_lower for p in PLACEHOLDER_SUBSTRINGS)


def is_supported_openai_model(model_name: str) -> bool:
    """Return True if the model name appears to support chat completions with tool calling."""
    if not model_name:
        return False
    lower = model_name.lower()
    if any(lower.startswith(prefix) for prefix in UNSUPPORTED_OPENAI_PREFIXES):
        return False
    return True


def validate_config() -> List[str]:
    """Check configuration and return a list of warning/error messages (empty = OK)."""
    warnings: List[str] = []

    if LLM_PROVIDER not in ("openai", "google", "offline"):
        warnings.append(
            f"Unsupported LLM provider: '{LLM_PROVIDER}'. "
            "Supported options are 'openai', 'google', or 'offline'."
        )
        return warnings

    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            warnings.append(
                "OPENAI_API_KEY is not set. Add OPENAI_API_KEY to your .env file or environment."
            )
        elif is_placeholder_key(OPENAI_API_KEY):
            warnings.append(
                "OPENAI_API_KEY contains a placeholder value. Replace it with your actual OpenAI API key in .env."
            )

        if not is_supported_openai_model(OPENAI_MODEL):
            warnings.append(
                f"Model '{OPENAI_MODEL}' may not support tool calling. "
                "Recommended models: 'gpt-4o-mini' or 'gpt-4o'."
            )

    elif LLM_PROVIDER == "google":
        if not GOOGLE_API_KEY:
            warnings.append(
                "GOOGLE_API_KEY is not set. Add GOOGLE_API_KEY to your .env file or environment."
            )
        elif is_placeholder_key(GOOGLE_API_KEY):
            warnings.append(
                "GOOGLE_API_KEY contains a placeholder value. Replace it with your actual Google API key in .env."
            )

    return warnings
