"""
system_utils.py — Safe system command execution and parsing helpers.

All external command execution goes through this module so that:
- Commands use fixed argument lists (shell=False exclusively).
- Timeouts are strictly enforced.
- Execution is read-only.
- Operating system limitations (Linux vs Windows/WSL2) are reported accurately.
- Secrets and sensitive hardware identifiers are safely handled.
"""

import subprocess
import shutil
import os
import re
import platform
from typing import Optional, Dict, List, Any


# ─── OS & Privilege Helpers ──────────────────────────────────────────

def is_linux() -> bool:
    """Return True if running on Linux or WSL."""
    return platform.system().lower() == "linux"


def is_root() -> bool:
    """Return True if running with root / administrative privileges on POSIX."""
    if hasattr(os, "geteuid"):
        return os.geteuid() == 0
    return False


# ─── Safe Command Runner ─────────────────────────────────────────────

def run_command(
    cmd: List[str],
    timeout: int = 10,
    requires_root: bool = False,
) -> Dict[str, Any]:
    """
    Execute a Linux system command safely and return structured output.

    Args:
        cmd:            Command and arguments as a list (e.g. ["lscpu"]).
        timeout:        Maximum seconds to wait for the command.
        requires_root:  If True and not running as root, attempt prepend with 'sudo'.

    Returns:
        dict with keys:
            success (bool): Whether the command ran without error.
            output  (str):  Standard output text.
            error   (str):  Error message if something went wrong.
    """
    executable = cmd[0]

    # Check for Linux environment
    if not is_linux() and not shutil.which(executable):
        return {
            "success": False,
            "output": "",
            "error": (
                f"Command '{executable}' is not available because this system is running {platform.system()}. "
                "The Hardware Introspection Agent is designed for Linux. "
                "Please run this agent inside Linux or WSL2 (Windows Subsystem for Linux) "
                "to inspect real hardware."
            ),
        }

    # Check if executable is installed
    if not shutil.which(executable) and executable != "sudo":
        packages = {
            "dmidecode": "dmidecode",
            "lspci": "pciutils",
            "lsusb": "usbutils",
            "lsblk": "util-linux",
            "lscpu": "util-linux",
            "free": "procps",
            "df": "coreutils",
            "uname": "coreutils",
        }
        package = packages.get(executable)
        guidance = (
            f" Install it with 'sudo apt install {package}'." if package else ""
        )
        return {
            "success": False,
            "output": "",
            "error": f"Command '{executable}' is not installed on this system.{guidance}",
        }

    # Optionally prepend sudo if requested and not root
    if requires_root and not is_root():
        if shutil.which("sudo"):
            cmd = ["sudo"] + cmd
        else:
            return {
                "success": False,
                "output": "",
                "error": (
                    f"'{executable}' requires root privileges, but sudo is not available. "
                    "Please run this agent as root or install sudo."
                ),
            }

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,  # Security: NEVER use shell=True
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            # Common permission issue
            if "permission denied" in stderr.lower() or "operation not permitted" in stderr.lower():
                return {
                    "success": False,
                    "output": result.stdout.strip(),
                    "error": (
                        f"Permission denied when running '{' '.join(cmd)}'. "
                        "Try running the agent with sudo for full hardware details."
                    ),
                }
            return {
                "success": False,
                "output": result.stdout.strip(),
                "error": stderr or f"Command exited with code {result.returncode}.",
            }

        return {
            "success": True,
            "output": result.stdout.strip(),
            "error": "",
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": "",
            "error": f"Command '{' '.join(cmd)}' timed out after {timeout} seconds.",
        }
    except FileNotFoundError:
        return {
            "success": False,
            "output": "",
            "error": f"Command '{executable}' not found.",
        }
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": f"Unexpected error running '{' '.join(cmd)}': {e}",
        }


# ─── Parsing Helpers ─────────────────────────────────────────────────

def read_proc_file(path: str) -> Dict[str, Any]:
    """
    Read a virtual file from /proc or /sys and return its contents.
    Reports OS limitations accurately on non-Linux platforms.
    """
    if not is_linux() and not os.path.exists(path):
        return {
            "success": False,
            "output": "",
            "error": (
                f"Cannot read '{path}': Linux virtual filesystems (/proc, /sys) are only available "
                f"on Linux. Current operating system is {platform.system()}."
            ),
        }

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read().strip()
        return {"success": True, "output": content, "error": ""}
    except FileNotFoundError:
        return {"success": False, "output": "", "error": f"File '{path}' not found."}
    except PermissionError:
        return {"success": False, "output": "", "error": f"Permission denied reading '{path}'."}
    except Exception as e:
        return {"success": False, "output": "", "error": f"Error reading '{path}': {e}"}


def parse_key_value_output(text: str, separator: str = ":") -> Dict[str, str]:
    """Parse command output that uses 'Key: Value' format (like lscpu)."""
    result = {}
    for line in text.splitlines():
        if separator in line:
            key, _, value = line.partition(separator)
            key = key.strip()
            value = value.strip()
            if key:
                result[key] = value
    return result


def parse_dmidecode_sections(text: str) -> List[Dict[str, str]]:
    """Parse dmidecode output into a list of sections."""
    sections: List[Dict[str, str]] = []
    current: Dict[str, str] = {}

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("#") or stripped.startswith("SMBIOS") or stripped.startswith("Handle"):
            continue

        if line and not line[0].isspace() and ":" not in line:
            if current:
                sections.append(current)
            current = {"_section_title": stripped}
            continue

        if ":" in stripped:
            key, _, value = stripped.partition(":")
            current[key.strip()] = value.strip()

    if current:
        sections.append(current)

    return sections


def safe_int(value: Any, default: int = 0) -> int:
    """Convert a value or string to int safely, extracting leading digits."""
    if isinstance(value, int):
        return value
    if not value:
        return default
    try:
        match = re.match(r"(\d+)", str(value).strip())
        if match:
            return int(match.group(1))
        return default
    except (ValueError, TypeError):
        return default


def bytes_to_human(size_bytes: int) -> str:
    """Convert a byte count to a human-readable string (e.g. '16 GB')."""
    if size_bytes <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    idx = 0
    size = float(size_bytes)
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    if size == int(size):
        return f"{int(size)} {units[idx]}"
    return f"{size:.1f} {units[idx]}"


def mask_serial(serial: str) -> str:
    """Mask a serial number for privacy, showing only last 4 characters."""
    if not serial or serial.lower() in ("unknown", "n/a", "not specified", "none", ""):
        return "Not Available"
    if len(serial) <= 4:
        return "****"
    return "*" * (len(serial) - 4) + serial[-4:]


def redact_secrets(text: str) -> str:
    """Redact API keys or credentials from text strings, logs, and error messages."""
    if not text:
        return ""
    # Redact OpenAI keys: sk-...
    text = re.sub(r"sk-[a-zA-Z0-9_\-]{15,}", "[REDACTED_OPENAI_KEY]", text)
    # Redact Google keys: AIza...
    text = re.sub(r"AIza[a-zA-Z0-9_\-]{25,}", "[REDACTED_GOOGLE_KEY]", text)
    return text
