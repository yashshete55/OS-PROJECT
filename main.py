#!/usr/bin/env python3
"""
main.py — Interactive command-line interface for the Hardware Introspection Agent.

Run with:  python -m hardware_agent.main
    or:    python main.py

Enter hardware questions in natural language. Type 'exit' or 'quit' to stop.
"""

import sys
import os
import platform

# Ensure project root is on the path so imports work when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Ensure standard output doesn't crash on Windows with cp1252 encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from hardware_agent.config import (
    AGENT_NAME,
    AGENT_VERSION,
    LLM_PROVIDER,
    GEMINI_MODEL,
    OPENAI_MODEL,
    validate_config,
)
from hardware_agent.system_utils import is_linux
from hardware_agent.agent import handle_hardware_query


BANNER = rf"""
================================================================
          {AGENT_NAME} v{AGENT_VERSION}
   Ask any question about your hardware in plain English.
   Type 'exit' or 'quit' to leave.
================================================================
"""

EXAMPLES = """
Example questions you can ask:
  * "How much RAM do I have installed?"
  * "What type of RAM is installed?"
  * "How many memory slots are empty?"
  * "Tell me about my processor and speed."
  * "Which graphics card is installed?"
  * "Show my USB devices."
  * "What storage devices are connected?"
  * "Why is my computer running slowly?"
  * "Give me a complete hardware report."
"""


def main():
    """Run the interactive hardware agent CLI."""
    print(BANNER)

    # ── Operating System Check ───────────────────────────────────────
    if not is_linux():
        print(
            f"[Environment Notice] Running on {platform.system()}.\n"
            "   The Hardware Introspection Agent is built for Linux operating systems.\n"
            "   Hardware inspection utilities (dmidecode, lscpu, lspci, etc.) require Linux or WSL2.\n"
            "   For live hardware access, please run this agent inside Ubuntu/Debian or WSL2.\n"
        )

    # ── Configuration Check ──────────────────────────────────────────
    warnings = validate_config()
    if warnings:
        print("[Configuration Notices]")
        for w in warnings:
            print(f"   * {w}")
        print()
    else:
        model = GEMINI_MODEL if LLM_PROVIDER == "google" else OPENAI_MODEL
        print(f"[Configuration active] Provider='{LLM_PROVIDER}', Model='{model}'\n")

    print(EXAMPLES)

    # ── Main Interaction Loop ────────────────────────────────────────
    while True:
        try:
            user_input = input(f"{AGENT_NAME} > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("help", "?"):
            print(EXAMPLES)
            continue

        print(f"\n[*] Processing: \"{user_input}\"\n")

        answer = handle_hardware_query(user_input)

        print(f"\n[Answer]:\n{'=' * 60}")
        print(answer)
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
