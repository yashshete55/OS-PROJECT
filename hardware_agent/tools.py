"""
tools.py — Hardware inspection functions for the Hardware Introspection Agent.

Each function gathers real hardware data from the Linux operating system
using safe, predefined commands and standard Linux interfaces. Functions return structured dictionaries
that the LLM can interpret and present to the user.

All system commands are executed through system_utils.run_command(),
ensuring safety, timeouts, and proper error handling without shell=True.
"""

from hardware_agent.system_utils import (
    run_command,
    read_proc_file,
    parse_key_value_output,
    parse_dmidecode_sections,
    safe_int,
    bytes_to_human,
    mask_serial,
)
from hardware_agent.config import COMMAND_TIMEOUT, MASK_SERIALS
import platform
import os
import re
from datetime import timedelta
from typing import Dict, Any, List


# =====================================================================
# 1. MEMORY / RAM INFORMATION
# =====================================================================

def get_memory_info() -> Dict[str, Any]:
    """
    Return installed RAM capacity, memory type, speed, manufacturer,
    occupied slots, total slots, and available (empty) slots.

    Primary source : dmidecode -t memory   (requires root for slot details)
    Fallbacks      : free -b, /proc/meminfo (no root required)
    """
    result: Dict[str, Any] = {
        "total_capacity": "Unknown",
        "available_capacity": "Unknown",
        "memory_type": "Unknown",
        "speed": "Unknown",
        "modules": [],
        "total_slots": "Unknown",
        "occupied_slots": "Unknown",
        "empty_slots": "Unknown",
        "errors": [],
    }

    # ── 1. Try dmidecode for physical slot and DIMM module info ──────
    dmi = run_command(
        ["dmidecode", "-t", "memory"],
        timeout=COMMAND_TIMEOUT,
        requires_root=True,
    )

    if dmi["success"]:
        sections = parse_dmidecode_sections(dmi["output"])

        for sec in sections:
            title = sec.get("_section_title", "")
            if "Physical Memory Array" in title:
                num_devices = sec.get("Number Of Devices", "")
                if num_devices.isdigit():
                    result["total_slots"] = int(num_devices)

        modules: List[Dict[str, Any]] = []
        for sec in sections:
            title = sec.get("_section_title", "")
            if "Memory Device" in title:
                size = sec.get("Size", "No Module Installed")
                is_populated = (
                    "no module" not in size.lower()
                    and size.lower() != "unknown"
                    and size.strip() != ""
                )
                serial = sec.get("Serial Number", "Unknown")
                module = {
                    "size": size,
                    "type": sec.get("Type", "Unknown"),
                    "speed": sec.get("Speed", "Unknown"),
                    "manufacturer": sec.get("Manufacturer", "Unknown"),
                    "form_factor": sec.get("Form Factor", "Unknown"),
                    "locator": sec.get("Locator", "Unknown"),
                    "serial_number": mask_serial(serial) if MASK_SERIALS else serial,
                    "populated": is_populated,
                }
                modules.append(module)

        result["modules"] = modules

        populated = [m for m in modules if m["populated"]]
        result["occupied_slots"] = len(populated)

        if isinstance(result["total_slots"], int):
            result["empty_slots"] = max(0, result["total_slots"] - len(populated))

        # Sum total capacity from populated modules
        total_mb = 0
        for m in populated:
            size_str = m["size"]
            num = safe_int(size_str, 0)
            if "GB" in size_str.upper():
                total_mb += num * 1024
            elif "MB" in size_str.upper():
                total_mb += num
            elif num > 0:
                total_mb += num

        if total_mb > 0:
            result["total_capacity"] = bytes_to_human(total_mb * 1024 * 1024)

        if populated:
            result["memory_type"] = populated[0]["type"]
            result["speed"] = populated[0]["speed"]
    else:
        result["errors"].append(dmi["error"])

    # ── 2. Fallback / supplementary: free -b (Linux standard) ────────
    free_cmd = run_command(["free", "-b"], timeout=COMMAND_TIMEOUT)
    if free_cmd["success"]:
        lines = free_cmd["output"].splitlines()
        for line in lines:
            if line.startswith("Mem:"):
                parts = line.split()
                if len(parts) >= 2:
                    total_bytes = safe_int(parts[1], 0)
                    if total_bytes > 0 and result["total_capacity"] == "Unknown":
                        result["total_capacity"] = bytes_to_human(total_bytes)
                if len(parts) >= 7:
                    avail_bytes = safe_int(parts[6], 0)
                    if avail_bytes > 0:
                        result["available_capacity"] = bytes_to_human(avail_bytes)
                break

    # ── 3. Fallback: /proc/meminfo ────────────────────────────────────
    if result["total_capacity"] == "Unknown" or result["available_capacity"] == "Unknown":
        meminfo = read_proc_file("/proc/meminfo")
        if meminfo["success"]:
            for line in meminfo["output"].splitlines():
                if line.startswith("MemTotal:") and result["total_capacity"] == "Unknown":
                    parts = line.split()
                    if len(parts) >= 2:
                        kb = safe_int(parts[1], 0)
                        if kb > 0:
                            result["total_capacity"] = bytes_to_human(kb * 1024)
                elif line.startswith("MemAvailable:") and result["available_capacity"] == "Unknown":
                    parts = line.split()
                    if len(parts) >= 2:
                        kb = safe_int(parts[1], 0)
                        if kb > 0:
                            result["available_capacity"] = bytes_to_human(kb * 1024)
        else:
            result["errors"].append(meminfo["error"])

    return result


# =====================================================================
# 2. CPU INFORMATION
# =====================================================================

def get_cpu_info() -> Dict[str, Any]:
    """
    Return processor name, architecture, core count, thread count,
    and reported frequency.

    Primary source : lscpu
    Fallback       : /proc/cpuinfo
    """
    result: Dict[str, Any] = {
        "model_name": "Unknown",
        "architecture": "Unknown",
        "cores": "Unknown",
        "threads": "Unknown",
        "max_frequency_mhz": "Unknown",
        "current_frequency_mhz": "Unknown",
        "vendor": "Unknown",
        "cache": {},
        "errors": [],
    }

    # ── 1. lscpu ─────────────────────────────────────────────────────
    lscpu = run_command(["lscpu"], timeout=COMMAND_TIMEOUT)
    if lscpu["success"]:
        data = parse_key_value_output(lscpu["output"])
        result["model_name"] = data.get("Model name", "Unknown")
        result["architecture"] = data.get("Architecture", "Unknown")
        result["vendor"] = data.get("Vendor ID", "Unknown")

        cores = data.get("Core(s) per socket", "")
        sockets = data.get("Socket(s)", "1")
        if cores.isdigit() and sockets.isdigit():
            result["cores"] = int(cores) * int(sockets)

        threads = data.get("CPU(s)", "")
        if threads.isdigit():
            result["threads"] = int(threads)

        max_freq = data.get("CPU max MHz", data.get("CPU MHz", ""))
        if max_freq:
            result["max_frequency_mhz"] = max_freq

        cur_freq = data.get("CPU MHz", "")
        if cur_freq:
            result["current_frequency_mhz"] = cur_freq

        for key, value in data.items():
            if "cache" in key.lower():
                result["cache"][key] = value
    else:
        result["errors"].append(lscpu["error"])

    # ── 2. Fallback: /proc/cpuinfo ───────────────────────────────────
    if result["model_name"] == "Unknown":
        cpuinfo = read_proc_file("/proc/cpuinfo")
        if cpuinfo["success"]:
            for line in cpuinfo["output"].splitlines():
                if line.startswith("model name") and result["model_name"] == "Unknown":
                    result["model_name"] = line.split(":", 1)[1].strip()
                elif line.startswith("cpu cores") and result["cores"] == "Unknown":
                    result["cores"] = safe_int(line.split(":", 1)[1], 0)
                elif line.startswith("siblings") and result["threads"] == "Unknown":
                    result["threads"] = safe_int(line.split(":", 1)[1], 0)
                elif line.startswith("cpu MHz") and result["current_frequency_mhz"] == "Unknown":
                    result["current_frequency_mhz"] = line.split(":", 1)[1].strip()
        else:
            result["errors"].append(cpuinfo["error"])

    return result


# =====================================================================
# 3. GPU / GRAPHICS CARD INFORMATION
# =====================================================================

def get_gpu_info() -> Dict[str, Any]:
    """
    Identify graphics adapters and their available model information.

    Source: lspci -nn (filtering for VGA / 3D / Display controllers)
    """
    result: Dict[str, Any] = {
        "gpus": [],
        "errors": [],
    }

    # Use lspci -nn for device codes and clear GPU detection
    lspci = run_command(["lspci", "-nn"], timeout=COMMAND_TIMEOUT)

    # If lspci -nn failed, try lspci without flags as fallback
    if not lspci["success"]:
        lspci = run_command(["lspci"], timeout=COMMAND_TIMEOUT)

    if lspci["success"]:
        gpu_keywords = ["vga", "3d controller", "display controller"]
        for line in lspci["output"].splitlines():
            lower = line.lower()
            if any(kw in lower for kw in gpu_keywords):
                parts = line.split(": ", 1)
                pci_slot = parts[0].split()[0] if parts else "Unknown"
                description = parts[1] if len(parts) > 1 else line
                result["gpus"].append({
                    "pci_slot": pci_slot,
                    "description": description,
                })

        if not result["gpus"]:
            result["gpus"].append({
                "pci_slot": "N/A",
                "description": "No dedicated GPU detected (system may use integrated graphics).",
            })
    else:
        result["errors"].append(lspci["error"])

    return result


# =====================================================================
# 4. MOTHERBOARD INFORMATION
# =====================================================================

def get_motherboard_info() -> Dict[str, Any]:
    """
    Return motherboard manufacturer, product name, serial number,
    and BIOS details.

    Sources: dmidecode -t baseboard, dmidecode -t bios
    Fallback: /sys/class/dmi/id/
    """
    result: Dict[str, Any] = {
        "manufacturer": "Unknown",
        "product_name": "Unknown",
        "serial_number": "Unknown",
        "bios_vendor": "Unknown",
        "bios_version": "Unknown",
        "bios_release_date": "Unknown",
        "errors": [],
    }

    # ── 1. Baseboard ─────────────────────────────────────────────────
    board = run_command(
        ["dmidecode", "-t", "baseboard"],
        timeout=COMMAND_TIMEOUT,
        requires_root=True,
    )

    if board["success"]:
        sections = parse_dmidecode_sections(board["output"])
        for sec in sections:
            if "Base Board" in sec.get("_section_title", ""):
                result["manufacturer"] = sec.get("Manufacturer", "Unknown")
                result["product_name"] = sec.get("Product Name", "Unknown")
                serial = sec.get("Serial Number", "Unknown")
                result["serial_number"] = mask_serial(serial) if MASK_SERIALS else serial
                break
    else:
        result["errors"].append(board["error"])

    # Fallback to /sys/class/dmi/id/
    if result["manufacturer"] == "Unknown":
        for field, path in [
            ("manufacturer", "/sys/class/dmi/id/board_vendor"),
            ("product_name", "/sys/class/dmi/id/board_name"),
        ]:
            info = read_proc_file(path)
            if info["success"] and info["output"]:
                result[field] = info["output"]

    # ── 2. BIOS ──────────────────────────────────────────────────────
    bios = run_command(
        ["dmidecode", "-t", "bios"],
        timeout=COMMAND_TIMEOUT,
        requires_root=True,
    )

    if bios["success"]:
        sections = parse_dmidecode_sections(bios["output"])
        for sec in sections:
            if "BIOS Information" in sec.get("_section_title", ""):
                result["bios_vendor"] = sec.get("Vendor", "Unknown")
                result["bios_version"] = sec.get("Version", "Unknown")
                result["bios_release_date"] = sec.get("Release Date", "Unknown")
                break
    else:
        result["errors"].append(bios["error"])

    # Fallback to /sys/class/dmi/id/ for BIOS
    if result["bios_vendor"] == "Unknown":
        for field, path in [
            ("bios_vendor", "/sys/class/dmi/id/bios_vendor"),
            ("bios_version", "/sys/class/dmi/id/bios_version"),
            ("bios_release_date", "/sys/class/dmi/id/bios_date"),
        ]:
            info = read_proc_file(path)
            if info["success"] and info["output"]:
                result[field] = info["output"]

    return result


# =====================================================================
# 5. USB DEVICES
# =====================================================================

def get_usb_devices() -> Dict[str, Any]:
    """
    List connected USB devices.

    Source: lsusb
    """
    result: Dict[str, Any] = {
        "devices": [],
        "count": 0,
        "errors": [],
    }

    lsusb = run_command(["lsusb"], timeout=COMMAND_TIMEOUT)

    if lsusb["success"]:
        for line in lsusb["output"].splitlines():
            line = line.strip()
            if not line:
                continue
            # Bus 001 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub
            match = re.match(r"Bus\s+(\d+)\s+Device\s+(\d+):\s+ID\s+(\S+)\s+(.*)", line)
            if match:
                result["devices"].append({
                    "bus": match.group(1),
                    "device": match.group(2),
                    "id": match.group(3),
                    "description": match.group(4),
                })
            else:
                result["devices"].append({"raw": line})

        result["count"] = len(result["devices"])
    else:
        result["errors"].append(lsusb["error"])

    return result


# =====================================================================
# 6. PCI DEVICES
# =====================================================================

def get_pci_devices() -> Dict[str, Any]:
    """
    List PCI devices (network adapters, graphics controllers, bridges, etc.).

    Source: lspci
    """
    result: Dict[str, Any] = {
        "devices": [],
        "count": 0,
        "errors": [],
    }

    lspci = run_command(["lspci"], timeout=COMMAND_TIMEOUT)

    if lspci["success"]:
        for line in lspci["output"].splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            slot = parts[0] if parts else "Unknown"
            rest = parts[1] if len(parts) > 1 else line

            type_desc = rest.split(": ", 1)
            device_type = type_desc[0] if type_desc else "Unknown"
            description = type_desc[1] if len(type_desc) > 1 else ""

            result["devices"].append({
                "slot": slot,
                "type": device_type,
                "description": description,
            })

        result["count"] = len(result["devices"])
    else:
        result["errors"].append(lspci["error"])

    return result


# =====================================================================
# 7. STORAGE INFORMATION
# =====================================================================

def get_storage_info() -> Dict[str, Any]:
    """
    Report detected storage devices, capacities, device types, mount points,
    and filesystem space utilization.

    Sources: lsblk, df -h
    """
    result: Dict[str, Any] = {
        "devices": [],
        "filesystems": [],
        "errors": [],
    }

    # 1. Block devices via lsblk
    lsblk = run_command(
        ["lsblk", "-o", "NAME,SIZE,TYPE,MOUNTPOINT,MODEL,FSTYPE", "--noheadings"],
        timeout=COMMAND_TIMEOUT,
    )

    if lsblk["success"]:
        for line in lsblk["output"].splitlines():
            line = line.strip()
            if not line:
                continue
            clean = re.sub(r"[├└│─\s]+", " ", line).strip()
            parts = clean.split(None, 5)

            device = {
                "name": parts[0] if len(parts) > 0 else "Unknown",
                "size": parts[1] if len(parts) > 1 else "Unknown",
                "type": parts[2] if len(parts) > 2 else "Unknown",
                "mountpoint": parts[3] if len(parts) > 3 and parts[3] != "" else "Not mounted",
                "model": parts[4] if len(parts) > 4 else "Unknown",
                "filesystem": parts[5] if len(parts) > 5 else "Unknown",
            }
            result["devices"].append(device)
    else:
        result["errors"].append(lsblk["error"])

    # 2. Filesystem utilization via df -h
    df = run_command(
        ["df", "-h", "--output=target,size,used,avail,pcent"],
        timeout=COMMAND_TIMEOUT,
    )
    if not df["success"]:
        df = run_command(["df", "-h"], timeout=COMMAND_TIMEOUT)

    if df["success"]:
        lines = df["output"].splitlines()
        for line in lines[1:]:
            fields = line.split()
            if len(fields) >= 5 and fields[0] != "Filesystem":
                result["filesystems"].append({
                    "mountpoint": fields[0],
                    "size": fields[1],
                    "used": fields[2],
                    "available": fields[3],
                    "use_percent": fields[4],
                })
    else:
        result["errors"].append(df["error"])

    return result


# =====================================================================
# 8. SYSTEM SUMMARY
# =====================================================================

def get_system_summary() -> Dict[str, Any]:
    """
    Return operating system information, architecture, hostname, uptime,
    and kernel version.

    Sources: platform module, /etc/os-release, /proc/uptime, uname
    """
    result: Dict[str, Any] = {
        "hostname": "Unknown",
        "os": "Unknown",
        "distribution": "Unknown",
        "kernel": "Unknown",
        "architecture": "Unknown",
        "uptime": "Unknown",
        "python_version": platform.python_version(),
        "errors": [],
    }

    try:
        result["hostname"] = platform.node()
        result["os"] = platform.system()
        result["kernel"] = platform.release()
        result["architecture"] = platform.machine()
    except Exception as e:
        result["errors"].append(f"Platform info error: {e}")

    # Read /etc/os-release for Linux distribution name
    os_release = read_proc_file("/etc/os-release")
    if os_release["success"]:
        for line in os_release["output"].splitlines():
            if line.startswith("PRETTY_NAME="):
                result["distribution"] = line.split("=", 1)[1].strip('"')
                break
    elif hasattr(platform, "freedesktop_os_release"):
        try:
            rel = platform.freedesktop_os_release()
            result["distribution"] = rel.get("PRETTY_NAME", "Unknown")
        except Exception:
            pass

    # Read /proc/uptime
    uptime_info = read_proc_file("/proc/uptime")
    if uptime_info["success"]:
        try:
            seconds = float(uptime_info["output"].split()[0])
            td = timedelta(seconds=int(seconds))
            days = td.days
            hours, remainder = divmod(td.seconds, 3600)
            minutes, _ = divmod(remainder, 60)
            result["uptime"] = f"{days}d {hours}h {minutes}m"
        except (ValueError, IndexError):
            pass

    return result


# =====================================================================
# 9. SYSTEM HEALTH (Why is my computer running slowly?)
# =====================================================================

def get_system_health() -> Dict[str, Any]:
    """
    Return read-only system performance and health indicators to help explain
    why a computer is running slowly: load averages, memory availability,
    and disk space warnings.

    Sources: /proc/loadavg, /proc/meminfo, free, df
    """
    result: Dict[str, Any] = {
        "load_average_1m": "Unknown",
        "load_average_5m": "Unknown",
        "load_average_15m": "Unknown",
        "available_memory": "Unknown",
        "total_memory": "Unknown",
        "memory_used_percent": "Unknown",
        "high_disk_usage": [],
        "errors": [],
    }

    # 1. Load averages from /proc/loadavg
    load = read_proc_file("/proc/loadavg")
    if load["success"]:
        values = load["output"].split()
        if len(values) >= 3:
            result["load_average_1m"] = values[0]
            result["load_average_5m"] = values[1]
            result["load_average_15m"] = values[2]
    else:
        result["errors"].append(load["error"])

    # 2. Available and total memory
    meminfo = read_proc_file("/proc/meminfo")
    if meminfo["success"]:
        total_kb = 0
        avail_kb = 0
        for line in meminfo["output"].splitlines():
            if line.startswith("MemTotal:"):
                total_kb = safe_int(line.split()[1], 0)
                result["total_memory"] = bytes_to_human(total_kb * 1024)
            elif line.startswith("MemAvailable:"):
                avail_kb = safe_int(line.split()[1], 0)
                result["available_memory"] = bytes_to_human(avail_kb * 1024)
        if total_kb > 0 and avail_kb > 0:
            used_pct = round(((total_kb - avail_kb) / total_kb) * 100, 1)
            result["memory_used_percent"] = f"{used_pct}%"
    else:
        result["errors"].append(meminfo["error"])

    # 3. Check for filesystems with high utilization (>90%)
    df = run_command(["df", "-h", "--output=target,pcent,avail"], timeout=COMMAND_TIMEOUT)
    if df["success"]:
        for line in df["output"].splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 2:
                pct_str = parts[1].replace("%", "")
                if pct_str.isdigit() and int(pct_str) >= 90:
                    result["high_disk_usage"].append({
                        "mountpoint": parts[0],
                        "usage": parts[1],
                        "available": parts[2] if len(parts) > 2 else "low",
                    })

    return result


# =====================================================================
# 10. COMPLETE HARDWARE REPORT
# =====================================================================

def get_hardware_report() -> Dict[str, Any]:
    """
    Combine all hardware information into a single structured report.
    Calls every inspection tool and merges the results.
    """
    return {
        "system": get_system_summary(),
        "health": get_system_health(),
        "cpu": get_cpu_info(),
        "memory": get_memory_info(),
        "gpu": get_gpu_info(),
        "motherboard": get_motherboard_info(),
        "storage": get_storage_info(),
        "usb_devices": get_usb_devices(),
        "pci_devices": get_pci_devices(),
    }


# =====================================================================
# 11. SEARCH / QUERY ROUTER
# =====================================================================

def search_hardware_info(query: str) -> Dict[str, Any]:
    """
    Select and execute the appropriate inspection function based on
    a hardware-related natural-language query.
    """
    if not isinstance(query, str) or not query.strip():
        return {"error": "Please provide a valid hardware query."}

    q = query.lower()

    if any(kw in q for kw in ["ram", "memory", "dimm", "slot"]):
        return {"category": "memory", "data": get_memory_info()}

    if any(kw in q for kw in ["cpu", "processor", "core", "thread", "frequency", "ghz", "mhz"]):
        return {"category": "cpu", "data": get_cpu_info()}

    if any(kw in q for kw in ["gpu", "graphics", "video card", "display adapter", "nvidia", "amd", "geforce", "radeon"]):
        return {"category": "gpu", "data": get_gpu_info()}

    if any(kw in q for kw in ["motherboard", "mainboard", "baseboard", "bios", "firmware", "uefi"]):
        return {"category": "motherboard", "data": get_motherboard_info()}

    if any(kw in q for kw in ["usb"]):
        return {"category": "usb_devices", "data": get_usb_devices()}

    if any(kw in q for kw in ["pci", "pcie", "network adapter", "ethernet", "wifi", "nic"]):
        return {"category": "pci_devices", "data": get_pci_devices()}

    if any(kw in q for kw in ["storage", "disk", "drive", "ssd", "hdd", "nvme", "partition", "space"]):
        return {"category": "storage", "data": get_storage_info()}

    if any(kw in q for kw in ["slow", "sluggish", "freeze", "lag", "load", "performance", "health", "hanging"]):
        return {"category": "health", "data": get_system_health()}

    if any(kw in q for kw in ["system", "os", "hostname", "uptime", "kernel", "operating system"]):
        return {"category": "system", "data": get_system_summary()}

    if any(kw in q for kw in ["report", "all", "everything", "complete", "full", "summary"]):
        return {"category": "full_report", "data": get_hardware_report()}

    # Sensible fallback
    return {
        "category": "system",
        "note": "Could not determine specific hardware category; returning system summary.",
        "data": get_system_summary(),
    }


# =====================================================================
# TOOL REGISTRY — exported for LLM tool calling and Master Shell
# =====================================================================

TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "get_memory_info": {
        "function": get_memory_info,
        "description": (
            "Get detailed RAM and memory information including total capacity, "
            "available capacity, memory type (DDR4/DDR5), speed, manufacturer, "
            "occupied slots, total slots, and empty slots."
        ),
        "parameters": {},
    },
    "get_cpu_info": {
        "function": get_cpu_info,
        "description": (
            "Get CPU details including processor model name, architecture, "
            "core count, thread count, vendor, and clock frequency."
        ),
        "parameters": {},
    },
    "get_gpu_info": {
        "function": get_gpu_info,
        "description": (
            "Identify installed graphics adapters (GPU) including model names, "
            "vendors (NVIDIA, Intel, AMD), and PCI slots."
        ),
        "parameters": {},
    },
    "get_motherboard_info": {
        "function": get_motherboard_info,
        "description": (
            "Get motherboard manufacturer, product name, serial number, and "
            "BIOS vendor, version, and release date."
        ),
        "parameters": {},
    },
    "get_usb_devices": {
        "function": get_usb_devices,
        "description": "List all connected USB devices with bus, device ID, and description.",
        "parameters": {},
    },
    "get_pci_devices": {
        "function": get_pci_devices,
        "description": (
            "List all PCI expansion devices including network adapters, "
            "graphics controllers, sound cards, and bridges."
        ),
        "parameters": {},
    },
    "get_storage_info": {
        "function": get_storage_info,
        "description": (
            "Get storage drive information including disks, partitions, "
            "capacities, device types, mount points, and filesystem space utilization."
        ),
        "parameters": {},
    },
    "get_system_summary": {
        "function": get_system_summary,
        "description": (
            "Get operating system information, Linux distribution, kernel release, "
            "hostname, architecture, and system uptime."
        ),
        "parameters": {},
    },
    "get_system_health": {
        "function": get_system_health,
        "description": (
            "Check system performance indicators including CPU load averages (1m, 5m, 15m), "
            "available memory, and high disk usage to explain why a computer is running slowly."
        ),
        "parameters": {},
    },
    "get_hardware_report": {
        "function": get_hardware_report,
        "description": (
            "Generate a complete hardware report combining CPU, memory, GPU, "
            "motherboard, BIOS, storage, USB, PCI, system, and health information."
        ),
        "parameters": {},
    },
    "search_hardware_info": {
        "function": search_hardware_info,
        "description": (
            "Search for hardware information by natural-language query. "
            "Automatically selects and executes the appropriate inspection tool."
        ),
        "parameters": {
            "query": {
                "type": "string",
                "description": "A natural-language hardware question or keyword.",
            }
        },
    },
}
