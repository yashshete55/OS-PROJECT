"""
tests/test_tools.py — Unit tests for the Hardware Introspection Agent.

All tests use mocked system command outputs so they run on any platform
without requiring physical hardware access, root privileges, or a live
LLM API.

Run with:  pytest tests/ -v
"""

import json
import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =====================================================================
# SAMPLE COMMAND OUTPUTS (realistic Linux outputs for mocking)
# =====================================================================

SAMPLE_LSCPU = """Architecture:                    x86_64
CPU op-mode(s):                  32-bit, 64-bit
Byte Order:                      Little Endian
CPU(s):                          8
On-line CPU(s) list:             0-7
Thread(s) per core:              2
Core(s) per socket:              4
Socket(s):                       1
Vendor ID:                       GenuineIntel
Model name:                      Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz
CPU MHz:                         3800.000
CPU max MHz:                     5100.0000
L1d cache:                       128 KiB
L1i cache:                       128 KiB
L2 cache:                        1 MiB
L3 cache:                        16 MiB"""

SAMPLE_DMIDECODE_MEMORY = """# dmidecode 3.3
Getting SMBIOS data from sysfs.
SMBIOS 3.2.0 present.

Handle 0x0036, DMI type 16, 23 bytes
Physical Memory Array
\tLocation: System Board Or Motherboard
\tUse: System Memory
\tError Correction Type: None
\tMaximum Capacity: 64 GB
\tNumber Of Devices: 4

Handle 0x0037, DMI type 17, 84 bytes
Memory Device
\tTotal Width: 64 bits
\tSize: 8192 MB
\tForm Factor: DIMM
\tLocator: DIMM_A1
\tType: DDR4
\tSpeed: 3200 MT/s
\tManufacturer: Kingston
\tSerial Number: AB1234CD
\tPart Number: KHX3200C16/8G

Handle 0x0038, DMI type 17, 84 bytes
Memory Device
\tTotal Width: 64 bits
\tSize: 8192 MB
\tForm Factor: DIMM
\tLocator: DIMM_A2
\tType: DDR4
\tSpeed: 3200 MT/s
\tManufacturer: Kingston
\tSerial Number: EF5678GH
\tPart Number: KHX3200C16/8G

Handle 0x0039, DMI type 17, 84 bytes
Memory Device
\tSize: No Module Installed
\tForm Factor: DIMM
\tLocator: DIMM_B1
\tType: Unknown

Handle 0x003A, DMI type 17, 84 bytes
Memory Device
\tSize: No Module Installed
\tForm Factor: DIMM
\tLocator: DIMM_B2
\tType: Unknown"""

SAMPLE_LSPCI = """00:00.0 Host bridge: Intel Corporation Device 9b53 (rev 03)
00:02.0 VGA compatible controller: Intel Corporation CometLake-S GT2 [UHD Graphics 630]
00:14.0 USB controller: Intel Corporation Device a3af
00:17.0 SATA controller: Intel Corporation Device a382
01:00.0 Network controller: Intel Corporation Wi-Fi 6 AX200
02:00.0 VGA compatible controller: NVIDIA Corporation GA106 [GeForce RTX 3060] (rev a1)"""

SAMPLE_LSUSB = """Bus 001 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub
Bus 001 Device 002: ID 046d:c52b Logitech, Inc. Unifying Receiver
Bus 002 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub
Bus 002 Device 003: ID 0951:16a4 Kingston Technology HyperX Alloy"""

SAMPLE_LSBLK = """sda       500G disk             Samsung_SSD_870
sda1      499G part /           Samsung_SSD_870   ext4
nvme0n1   1T   disk             Samsung_SSD_980_PRO
nvme0n1p1 512M part /boot/efi   Samsung_SSD_980_PRO vfat
nvme0n1p2 999.5G part /home     Samsung_SSD_980_PRO ext4"""

SAMPLE_DMIDECODE_BASEBOARD = """# dmidecode 3.3
Handle 0x0002, DMI type 2, 15 bytes
Base Board Information
\tManufacturer: ASUSTeK COMPUTER INC.
\tProduct Name: ROG STRIX Z490-E GAMING
\tVersion: Rev 1.xx
\tSerial Number: 200901234567890"""

SAMPLE_DMIDECODE_BIOS = """# dmidecode 3.3
Handle 0x0000, DMI type 0, 26 bytes
BIOS Information
\tVendor: American Megatrends Inc.
\tVersion: 2401
\tRelease Date: 03/15/2024
\tROM Size: 16 MB"""

SAMPLE_PROCMEMINFO = """MemTotal:       16384000 kB
MemFree:         8192000 kB
MemAvailable:   12288000 kB
Buffers:          262144 kB
Cached:          3145728 kB"""

SAMPLE_PROC_UPTIME = "123456.78 234567.89"


# =====================================================================
# HELPER: Mock run_command
# =====================================================================

def mock_run_command(cmd_outputs: dict):
    """
    Create a mock for system_utils.run_command that returns predefined
    outputs based on the command.

    Args:
        cmd_outputs: Dict mapping command tuple → output string.
                     Use None as value to simulate command not found.
    """
    def _mock(cmd, timeout=10, requires_root=False):
        key = tuple(cmd)
        # Also try just the executable name for simpler matching
        exe_key = (cmd[0],)

        for k in [key, exe_key]:
            if k in cmd_outputs:
                val = cmd_outputs[k]
                if val is None:
                    return {
                        "success": False,
                        "output": "",
                        "error": f"Command '{cmd[0]}' is not installed on this system.",
                    }
                return {"success": True, "output": val, "error": ""}

        # Match by first few args
        for k, v in cmd_outputs.items():
            if cmd[:len(k)] == list(k):
                if v is None:
                    return {
                        "success": False,
                        "output": "",
                        "error": f"Command '{cmd[0]}' is not installed.",
                    }
                return {"success": True, "output": v, "error": ""}

        return {
            "success": False,
            "output": "",
            "error": f"Command '{' '.join(cmd)}' not found in mock.",
        }

    return _mock


def mock_read_proc_file(file_outputs: dict):
    """Mock for system_utils.read_proc_file."""
    def _mock(path):
        if path in file_outputs:
            val = file_outputs[path]
            if val is None:
                return {"success": False, "output": "", "error": f"File '{path}' not found."}
            return {"success": True, "output": val, "error": ""}
        return {"success": False, "output": "", "error": f"File '{path}' not found."}
    return _mock


# =====================================================================
# TEST CLASS: Memory Information
# =====================================================================

class TestGetMemoryInfo(unittest.TestCase):
    """Tests for get_memory_info()."""

    @patch("hardware_agent.tools.read_proc_file")
    @patch("hardware_agent.tools.run_command")
    def test_correct_ram_extraction(self, mock_cmd, mock_proc):
        """Test that RAM capacity, type, and speed are correctly extracted."""
        mock_cmd.return_value = {
            "success": True,
            "output": SAMPLE_DMIDECODE_MEMORY,
            "error": "",
        }
        mock_proc.return_value = {
            "success": True,
            "output": SAMPLE_PROCMEMINFO,
            "error": "",
        }

        from hardware_agent.tools import get_memory_info
        result = get_memory_info()

        self.assertEqual(result["total_capacity"], "16 GB")
        self.assertEqual(result["memory_type"], "DDR4")
        self.assertEqual(result["speed"], "3200 MT/s")

    @patch("hardware_agent.tools.read_proc_file")
    @patch("hardware_agent.tools.run_command")
    def test_correct_slot_calculation(self, mock_cmd, mock_proc):
        """Test that empty slots are calculated correctly: 4 total - 2 occupied = 2 empty."""
        mock_cmd.return_value = {
            "success": True,
            "output": SAMPLE_DMIDECODE_MEMORY,
            "error": "",
        }
        mock_proc.return_value = {
            "success": True,
            "output": SAMPLE_PROCMEMINFO,
            "error": "",
        }

        from hardware_agent.tools import get_memory_info
        result = get_memory_info()

        self.assertEqual(result["total_slots"], 4)
        self.assertEqual(result["occupied_slots"], 2)
        self.assertEqual(result["empty_slots"], 2)

    @patch("hardware_agent.tools.read_proc_file")
    @patch("hardware_agent.tools.run_command")
    def test_populated_modules(self, mock_cmd, mock_proc):
        """Test that populated and empty modules are distinguished."""
        mock_cmd.return_value = {
            "success": True,
            "output": SAMPLE_DMIDECODE_MEMORY,
            "error": "",
        }
        mock_proc.return_value = {
            "success": True,
            "output": SAMPLE_PROCMEMINFO,
            "error": "",
        }

        from hardware_agent.tools import get_memory_info
        result = get_memory_info()

        populated = [m for m in result["modules"] if m["populated"]]
        empty = [m for m in result["modules"] if not m["populated"]]

        self.assertEqual(len(populated), 2)
        self.assertEqual(len(empty), 2)

    @patch("hardware_agent.tools.read_proc_file")
    @patch("hardware_agent.tools.run_command")
    def test_fallback_to_procmeminfo(self, mock_cmd, mock_proc):
        """Test fallback to /proc/meminfo when dmidecode fails."""
        mock_cmd.return_value = {
            "success": False,
            "output": "",
            "error": "Command 'dmidecode' is not installed on this system.",
        }
        mock_proc.return_value = {
            "success": True,
            "output": SAMPLE_PROCMEMINFO,
            "error": "",
        }

        from hardware_agent.tools import get_memory_info
        result = get_memory_info()

        # Should still get total capacity from /proc/meminfo
        self.assertNotEqual(result["total_capacity"], "Unknown")
        self.assertIn("error" in str(result["errors"]).lower() or len(result["errors"]) > 0, [True])


# =====================================================================
# TEST CLASS: CPU Information
# =====================================================================

class TestGetCpuInfo(unittest.TestCase):
    """Tests for get_cpu_info()."""

    @patch("hardware_agent.tools.read_proc_file")
    @patch("hardware_agent.tools.run_command")
    def test_cpu_model_extraction(self, mock_cmd, mock_proc):
        """Test correct CPU model name extraction."""
        mock_cmd.return_value = {
            "success": True,
            "output": SAMPLE_LSCPU,
            "error": "",
        }

        from hardware_agent.tools import get_cpu_info
        result = get_cpu_info()

        self.assertIn("i7-10700K", result["model_name"])
        self.assertEqual(result["architecture"], "x86_64")

    @patch("hardware_agent.tools.read_proc_file")
    @patch("hardware_agent.tools.run_command")
    def test_cpu_core_thread_count(self, mock_cmd, mock_proc):
        """Test correct core and thread count."""
        mock_cmd.return_value = {
            "success": True,
            "output": SAMPLE_LSCPU,
            "error": "",
        }

        from hardware_agent.tools import get_cpu_info
        result = get_cpu_info()

        self.assertEqual(result["cores"], 4)  # 4 cores per socket * 1 socket
        self.assertEqual(result["threads"], 8)

    @patch("hardware_agent.tools.read_proc_file")
    @patch("hardware_agent.tools.run_command")
    def test_cpu_lscpu_missing(self, mock_cmd, mock_proc):
        """Test fallback when lscpu is not available."""
        mock_cmd.return_value = {
            "success": False,
            "output": "",
            "error": "Command 'lscpu' is not installed.",
        }
        mock_proc.return_value = {
            "success": True,
            "output": "model name\t: Intel(R) Core(TM) i5-9400F\ncpu cores\t: 6\nsiblings\t: 6\ncpu MHz\t\t: 2900.000",
            "error": "",
        }

        from hardware_agent.tools import get_cpu_info
        result = get_cpu_info()

        self.assertIn("i5-9400F", result["model_name"])
        self.assertEqual(result["cores"], 6)


# =====================================================================
# TEST CLASS: GPU Information
# =====================================================================

class TestGetGpuInfo(unittest.TestCase):
    """Tests for get_gpu_info()."""

    @patch("hardware_agent.tools.run_command")
    def test_gpu_detection(self, mock_cmd):
        """Test that GPUs are correctly identified from lspci output."""
        mock_cmd.return_value = {
            "success": True,
            "output": SAMPLE_LSPCI,
            "error": "",
        }

        from hardware_agent.tools import get_gpu_info
        result = get_gpu_info()

        self.assertEqual(len(result["gpus"]), 2)  # Intel UHD + NVIDIA RTX 3060
        descriptions = [g["description"] for g in result["gpus"]]
        self.assertTrue(any("NVIDIA" in d for d in descriptions))
        self.assertTrue(any("Intel" in d for d in descriptions))

    @patch("hardware_agent.tools.run_command")
    def test_lspci_missing(self, mock_cmd):
        """Test handling when lspci is not available."""
        mock_cmd.return_value = {
            "success": False,
            "output": "",
            "error": "Command 'lspci' is not installed on this system.",
        }

        from hardware_agent.tools import get_gpu_info
        result = get_gpu_info()

        self.assertTrue(len(result["errors"]) > 0)


# =====================================================================
# TEST CLASS: Storage Information
# =====================================================================

class TestGetStorageInfo(unittest.TestCase):
    """Tests for get_storage_info()."""

    @patch("hardware_agent.tools.run_command")
    def test_storage_detection(self, mock_cmd):
        """Test that storage devices are correctly parsed."""
        mock_cmd.return_value = {
            "success": True,
            "output": SAMPLE_LSBLK,
            "error": "",
        }

        from hardware_agent.tools import get_storage_info
        result = get_storage_info()

        self.assertTrue(len(result["devices"]) > 0)
        names = [d["name"] for d in result["devices"]]
        self.assertIn("sda", names)


# =====================================================================
# TEST CLASS: USB Devices
# =====================================================================

class TestGetUsbDevices(unittest.TestCase):
    """Tests for get_usb_devices()."""

    @patch("hardware_agent.tools.run_command")
    def test_usb_listing(self, mock_cmd):
        """Test correct USB device parsing."""
        mock_cmd.return_value = {
            "success": True,
            "output": SAMPLE_LSUSB,
            "error": "",
        }

        from hardware_agent.tools import get_usb_devices
        result = get_usb_devices()

        self.assertEqual(result["count"], 4)
        descriptions = [d.get("description", "") for d in result["devices"]]
        self.assertTrue(any("Logitech" in d for d in descriptions))


# =====================================================================
# TEST CLASS: Motherboard & BIOS
# =====================================================================

class TestGetMotherboardInfo(unittest.TestCase):
    """Tests for get_motherboard_info()."""

    @patch("hardware_agent.tools.read_proc_file")
    @patch("hardware_agent.tools.run_command")
    def test_motherboard_extraction(self, mock_cmd, mock_proc):
        """Test motherboard info extraction from dmidecode."""
        def side_effect(cmd, timeout=10, requires_root=False):
            cmd_str = " ".join(cmd)
            if "baseboard" in cmd_str:
                return {"success": True, "output": SAMPLE_DMIDECODE_BASEBOARD, "error": ""}
            if "bios" in cmd_str:
                return {"success": True, "output": SAMPLE_DMIDECODE_BIOS, "error": ""}
            return {"success": False, "output": "", "error": "Unknown command"}

        mock_cmd.side_effect = side_effect
        mock_proc.return_value = {"success": False, "output": "", "error": "Not found"}

        from hardware_agent.tools import get_motherboard_info
        result = get_motherboard_info()

        self.assertIn("ASUSTeK", result["manufacturer"])
        self.assertIn("ROG STRIX", result["product_name"])
        self.assertIn("American Megatrends", result["bios_vendor"])
        self.assertEqual(result["bios_version"], "2401")


# =====================================================================
# TEST CLASS: System Summary
# =====================================================================

class TestGetSystemSummary(unittest.TestCase):
    """Tests for get_system_summary()."""

    @patch("hardware_agent.tools.read_proc_file")
    def test_uptime_parsing(self, mock_proc):
        """Test uptime is correctly parsed from /proc/uptime."""
        mock_proc.return_value = {
            "success": True,
            "output": SAMPLE_PROC_UPTIME,
            "error": "",
        }

        from hardware_agent.tools import get_system_summary
        result = get_system_summary()

        # 123456 seconds ≈ 1d 10h 17m
        self.assertIn("d", result["uptime"])
        self.assertIn("h", result["uptime"])


# =====================================================================
# TEST CLASS: Search / Query Router
# =====================================================================

class TestSearchHardwareInfo(unittest.TestCase):
    """Tests for search_hardware_info() query routing."""

    @patch("hardware_agent.tools.run_command")
    @patch("hardware_agent.tools.read_proc_file")
    def test_ram_query_routing(self, mock_proc, mock_cmd):
        """Test that RAM-related queries route to memory info."""
        mock_cmd.return_value = {"success": True, "output": SAMPLE_DMIDECODE_MEMORY, "error": ""}
        mock_proc.return_value = {"success": True, "output": SAMPLE_PROCMEMINFO, "error": ""}

        from hardware_agent.tools import search_hardware_info
        result = search_hardware_info("What RAM do I have?")
        self.assertEqual(result["category"], "memory")

    @patch("hardware_agent.tools.run_command")
    @patch("hardware_agent.tools.read_proc_file")
    def test_cpu_query_routing(self, mock_proc, mock_cmd):
        """Test that CPU-related queries route to CPU info."""
        mock_cmd.return_value = {"success": True, "output": SAMPLE_LSCPU, "error": ""}
        mock_proc.return_value = {"success": True, "output": "", "error": ""}

        from hardware_agent.tools import search_hardware_info
        result = search_hardware_info("What processor do I have?")
        self.assertEqual(result["category"], "cpu")

    @patch("hardware_agent.tools.run_command")
    def test_gpu_query_routing(self, mock_cmd):
        """Test that GPU-related queries route to GPU info."""
        mock_cmd.return_value = {"success": True, "output": SAMPLE_LSPCI, "error": ""}

        from hardware_agent.tools import search_hardware_info
        result = search_hardware_info("What graphics card is installed?")
        self.assertEqual(result["category"], "gpu")

    def test_empty_query_handling(self):
        """Test that empty queries return an error."""
        from hardware_agent.tools import search_hardware_info
        result = search_hardware_info("")
        self.assertIn("error", result)

    def test_none_query_handling(self):
        """Test that None queries are handled safely."""
        from hardware_agent.tools import search_hardware_info
        result = search_hardware_info(None)
        self.assertIn("error", result)


# =====================================================================
# TEST CLASS: Permission Denied Handling
# =====================================================================

class TestPermissionDenied(unittest.TestCase):
    """Test handling of permission-denied errors."""

    @patch("hardware_agent.tools.read_proc_file")
    @patch("hardware_agent.tools.run_command")
    def test_dmidecode_permission_denied(self, mock_cmd, mock_proc):
        """Test graceful handling when dmidecode requires root."""
        mock_cmd.return_value = {
            "success": False,
            "output": "",
            "error": "Permission denied when running 'dmidecode -t memory'. "
                     "Try running the agent with sudo for full hardware details.",
        }
        mock_proc.return_value = {
            "success": True,
            "output": SAMPLE_PROCMEMINFO,
            "error": "",
        }

        from hardware_agent.tools import get_memory_info
        result = get_memory_info()

        # Should have error messages but still return fallback data
        self.assertTrue(len(result["errors"]) > 0)
        self.assertNotEqual(result["total_capacity"], "Unknown")


# =====================================================================
# TEST CLASS: Missing Utilities
# =====================================================================

class TestMissingUtilities(unittest.TestCase):
    """Test handling when Linux utilities are not installed."""

    @patch("hardware_agent.tools.run_command")
    def test_lsusb_missing(self, mock_cmd):
        """Test handling when lsusb is not installed."""
        mock_cmd.return_value = {
            "success": False,
            "output": "",
            "error": "Command 'lsusb' is not installed on this system.",
        }

        from hardware_agent.tools import get_usb_devices
        result = get_usb_devices()

        self.assertEqual(result["count"], 0)
        self.assertTrue(len(result["errors"]) > 0)
        self.assertIn("not installed", result["errors"][0])

    @patch("hardware_agent.tools.run_command")
    def test_lspci_missing(self, mock_cmd):
        """Test handling when lspci is not installed."""
        mock_cmd.return_value = {
            "success": False,
            "output": "",
            "error": "Command 'lspci' is not installed on this system.",
        }

        from hardware_agent.tools import get_pci_devices
        result = get_pci_devices()

        self.assertEqual(result["count"], 0)
        self.assertTrue(len(result["errors"]) > 0)


# =====================================================================
# TEST CLASS: Malformed Output
# =====================================================================

class TestMalformedOutput(unittest.TestCase):
    """Test handling of malformed or unexpected command output."""

    @patch("hardware_agent.tools.read_proc_file")
    @patch("hardware_agent.tools.run_command")
    def test_malformed_lscpu(self, mock_cmd, mock_proc):
        """Test handling of malformed lscpu output."""
        mock_cmd.return_value = {
            "success": True,
            "output": "This is not valid lscpu output\nRandom garbage data",
            "error": "",
        }
        mock_proc.return_value = {"success": False, "output": "", "error": "Not found"}

        from hardware_agent.tools import get_cpu_info
        result = get_cpu_info()

        # Should not crash, should return Unknown for missing fields
        self.assertEqual(result["model_name"], "Unknown")

    @patch("hardware_agent.tools.run_command")
    def test_empty_lsusb_output(self, mock_cmd):
        """Test handling of empty lsusb output."""
        mock_cmd.return_value = {
            "success": True,
            "output": "",
            "error": "",
        }

        from hardware_agent.tools import get_usb_devices
        result = get_usb_devices()

        self.assertEqual(result["count"], 0)
        self.assertEqual(result["devices"], [])


# =====================================================================
# TEST CLASS: Tool Execution (Agent)
# =====================================================================

class TestToolExecution(unittest.TestCase):
    """Test the agent's tool execution dispatcher."""

    def test_execute_known_tool(self):
        """Test executing a known tool by name."""
        from hardware_agent.agent import execute_tool

        with patch("hardware_agent.tools.run_command") as mock_cmd:
            mock_cmd.return_value = {
                "success": True,
                "output": SAMPLE_LSCPU,
                "error": "",
            }

            result_json = execute_tool("get_cpu_info", {})
            result = json.loads(result_json)

            self.assertIn("model_name", result)

    def test_execute_unknown_tool(self):
        """Test that unknown tool names are rejected."""
        from hardware_agent.agent import execute_tool

        result_json = execute_tool("hack_the_mainframe", {})
        result = json.loads(result_json)

        self.assertIn("error", result)
        self.assertIn("Unknown tool", result["error"])

    def test_execute_tool_with_params(self):
        """Test executing search_hardware_info with a query parameter."""
        from hardware_agent.agent import execute_tool

        with patch("hardware_agent.tools.run_command") as mock_cmd, \
             patch("hardware_agent.tools.read_proc_file") as mock_proc:
            mock_cmd.return_value = {
                "success": True,
                "output": SAMPLE_LSCPU,
                "error": "",
            }
            mock_proc.return_value = {"success": True, "output": "", "error": ""}

            result_json = execute_tool(
                "search_hardware_info",
                {"query": "What CPU do I have?"},
            )
            result = json.loads(result_json)

            self.assertEqual(result["category"], "cpu")


# =====================================================================
# TEST CLASS: System Utils
# =====================================================================

class TestSystemUtils(unittest.TestCase):
    """Tests for system_utils helper functions."""

    def test_parse_key_value_output(self):
        """Test key-value parsing of lscpu-style output."""
        from hardware_agent.system_utils import parse_key_value_output

        text = "Architecture:  x86_64\nCPU(s):  8\nModel name:  Intel Core i7"
        result = parse_key_value_output(text)

        self.assertEqual(result["Architecture"], "x86_64")
        self.assertEqual(result["CPU(s)"], "8")

    def test_safe_int(self):
        """Test safe integer conversion."""
        from hardware_agent.system_utils import safe_int

        self.assertEqual(safe_int("8192 MB"), 8192)
        self.assertEqual(safe_int("not_a_number"), 0)
        self.assertEqual(safe_int("", 42), 42)

    def test_bytes_to_human(self):
        """Test human-readable byte conversion."""
        from hardware_agent.system_utils import bytes_to_human

        self.assertEqual(bytes_to_human(1024), "1 KB")
        self.assertEqual(bytes_to_human(1073741824), "1 GB")
        self.assertEqual(bytes_to_human(17179869184), "16 GB")
        self.assertEqual(bytes_to_human(0), "0 B")

    def test_mask_serial(self):
        """Test serial number masking."""
        from hardware_agent.system_utils import mask_serial

        self.assertEqual(mask_serial("AB1234CD5678"), "********5678")
        self.assertEqual(mask_serial("1234"), "****")
        self.assertEqual(mask_serial(""), "Not Available")
        self.assertEqual(mask_serial("Not Specified"), "Not Available")

    def test_parse_dmidecode_sections(self):
        """Test dmidecode section parsing."""
        from hardware_agent.system_utils import parse_dmidecode_sections

        sections = parse_dmidecode_sections(SAMPLE_DMIDECODE_BASEBOARD)

        # Should find the Base Board Information section
        board_sections = [
            s for s in sections
            if "Base Board" in s.get("_section_title", "")
        ]
        self.assertTrue(len(board_sections) > 0)
        self.assertIn("ASUSTeK", board_sections[0].get("Manufacturer", ""))


# =====================================================================
# TEST CLASS: Master Shell Integration
# =====================================================================

class TestMasterShellIntegration(unittest.TestCase):
    """Test the public interface for the master Agentic Shell."""

    def test_agent_metadata_structure(self):
        """Test that AGENT_METADATA has the required fields."""
        from hardware_agent.agent import AGENT_METADATA

        required_keys = ["id", "name", "version", "description", "capabilities",
                         "interface", "tools"]
        for key in required_keys:
            self.assertIn(key, AGENT_METADATA)

    def test_tool_registry_completeness(self):
        """Test that TOOL_REGISTRY contains all 10 required tools."""
        from hardware_agent.tools import TOOL_REGISTRY

        expected_tools = [
            "get_memory_info",
            "get_cpu_info",
            "get_gpu_info",
            "get_motherboard_info",
            "get_usb_devices",
            "get_pci_devices",
            "get_storage_info",
            "get_system_summary",
            "get_hardware_report",
            "search_hardware_info",
        ]
        for tool in expected_tools:
            self.assertIn(tool, TOOL_REGISTRY)
            self.assertIn("function", TOOL_REGISTRY[tool])
            self.assertIn("description", TOOL_REGISTRY[tool])
            self.assertTrue(callable(TOOL_REGISTRY[tool]["function"]))

    def test_handle_empty_query(self):
        """Test that empty queries are handled gracefully."""
        from hardware_agent.agent import handle_hardware_query

        result = handle_hardware_query("")
        self.assertIn("Please provide", result)

    def test_handle_none_query(self):
        """Test that None queries are handled gracefully."""
        from hardware_agent.agent import handle_hardware_query

        result = handle_hardware_query(None)
        self.assertIn("Please provide", result)


# =====================================================================
# TEST CLASS: Arbitrary Input Safety
# =====================================================================

class TestInputSafety(unittest.TestCase):
    """Test that arbitrary/malicious inputs are handled safely."""

    def test_injection_in_search(self):
        """Test that shell injection attempts in search are safe."""
        from hardware_agent.tools import search_hardware_info

        # These should NOT execute any shell commands
        with patch("hardware_agent.tools.run_command") as mock_cmd:
            mock_cmd.return_value = {"success": True, "output": "", "error": ""}
            result = search_hardware_info("; rm -rf /")
            # Should route to default (system summary) since no keywords match
            self.assertIn("category", result)

    def test_very_long_query(self):
        """Test that extremely long queries don't crash."""
        from hardware_agent.tools import search_hardware_info

        with patch("hardware_agent.tools.run_command") as mock_cmd, \
             patch("hardware_agent.tools.read_proc_file") as mock_proc:
            mock_cmd.return_value = {"success": True, "output": "", "error": ""}
            mock_proc.return_value = {"success": True, "output": "", "error": ""}

            long_query = "A" * 10000
            result = search_hardware_info(long_query)
            self.assertIn("category", result)


# =====================================================================

if __name__ == "__main__":
    unittest.main()
