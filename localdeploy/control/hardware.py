"""Cross-platform hardware detection and multi-GPU placement estimates.

The probe is deliberately best-effort. Vendor CLIs provide the best live VRAM
numbers; platform APIs and sysfs fill gaps without making server startup depend
on a particular driver stack. Every public function degrades to partial data.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from fastapi import APIRouter

router = APIRouter()

_NVIDIA_SMI_FIELDS = "name,memory.total,memory.free,memory.used,driver_version,utilization.gpu"
_VENDOR_IDS = {"0x10de": "NVIDIA", "0x1002": "AMD", "0x8086": "Intel"}


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _bytes_to_mb(value: Any) -> Optional[int]:
    number = _to_int(value)
    return int(number / (1024 * 1024)) if number is not None and number >= 0 else None


def _memory_to_mb(value: Any, key: str = "") -> Optional[int]:
    """Parse bytes, MiB/GiB strings, or numeric values whose key declares units."""
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*(gib|gb|mib|mb|bytes?|b)?", text, re.I)
    if not match:
        return None
    amount = float(match.group(1))
    unit = (match.group(2) or "").lower()
    key_lower = key.lower()
    if unit in {"gib", "gb"} or "gib" in key_lower or " gb" in key_lower:
        return int(amount * 1024)
    if unit in {"mib", "mb"} or "mib" in key_lower or " mb" in key_lower:
        return int(amount)
    if unit in {"byte", "bytes", "b"} or "(b)" in key_lower or "bytes" in key_lower:
        return int(amount / (1024 * 1024))
    # Vendor JSON APIs commonly emit raw bytes as large integers.
    return int(amount / (1024 * 1024)) if amount > 16_777_216 else int(amount)


def _run(command: List[str], timeout: int = 10) -> Optional[str]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def _windows_cpu_name() -> Optional[str]:
    # platform.processor() on Windows is the raw "AMD64 Family 25 Model 80 Stepping 0,
    # AuthenticAMD" identifier; the registry holds the marketing name.
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            value, _kind = winreg.QueryValueEx(key, "ProcessorNameString")
    except (ImportError, OSError):
        return None
    return " ".join(str(value).split()) or None


def _macos_cpu_name() -> Optional[str]:
    exe = shutil.which("sysctl")
    output = _run([exe, "-n", "machdep.cpu.brand_string"], timeout=5) if exe else None
    return (output or "").strip() or None


def _cpu_model() -> Optional[str]:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip() or None
    except OSError:
        pass
    system = platform.system()
    named = _windows_cpu_name() if system == "Windows" else _macos_cpu_name() if system == "Darwin" else None
    if named:
        return named
    proc = (platform.processor() or "").strip()
    return proc or (platform.machine() or "").strip() or None


def _cpu_and_memory() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "cpu_model": _cpu_model(),
        "logical_cores": os.cpu_count(),
        "physical_cores": None,
        "ram_total_mb": None,
        "ram_available_mb": None,
        "cpu_percent": None,
        "psutil_available": False,
        "ram_probe_message": "RAM details are unavailable because the system probe failed.",
    }
    try:
        import psutil

        info["psutil_available"] = True
        info["ram_probe_message"] = None
        info["physical_cores"] = psutil.cpu_count(logical=False)
        if info["logical_cores"] is None:
            info["logical_cores"] = psutil.cpu_count(logical=True)
        vm = psutil.virtual_memory()
        info["ram_total_mb"] = int(vm.total / (1024 * 1024))
        info["ram_available_mb"] = int(vm.available / (1024 * 1024))
        # Non-blocking (interval=None): compares against the last call rather than
        # sleeping, so this never adds latency to a hardware probe. The first call
        # in a process's lifetime returns 0.0 - meaningful readings need a second.
        info["cpu_percent"] = psutil.cpu_percent(interval=None)
    except Exception:
        pass
    return info


def _gpu(
    name: Optional[str],
    vendor: str,
    backend: str,
    total: Optional[int],
    free: Optional[int] = None,
    used: Optional[int] = None,
    driver: Optional[str] = None,
    *,
    unified: bool = False,
    source: str,
    estimated: bool = False,
    utilization_pct: Optional[float] = None,
) -> Dict[str, Any]:
    if free is None and total is not None and used is not None:
        free = max(0, total - used)
    if used is None and total is not None and free is not None:
        used = max(0, total - free)
    return {
        "name": name or f"{vendor} GPU",
        "vendor": vendor,
        "backend": backend,
        "vram_total_mb": total,
        "vram_free_mb": free,
        "vram_used_mb": used,
        "driver_version": driver,
        "unified_memory": unified,
        "memory_source": source,
        "vram_estimated": estimated,
        # Compute (SM) utilization, not memory bandwidth - only available from
        # vendor CLIs (nvidia-smi today). None on sources that can't report it.
        "utilization_pct": utilization_pct,
    }


def _query_nvidia_smi() -> List[Dict[str, Any]]:
    exe = shutil.which("nvidia-smi")
    output = _run([exe, f"--query-gpu={_NVIDIA_SMI_FIELDS}", "--format=csv,noheader,nounits"]) if exe else None
    if not output:
        return []
    gpus: List[Dict[str, Any]] = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        name, total, free, used, driver = parts[:5]
        utilization = _to_int(parts[5]) if len(parts) > 5 else None
        gpus.append(
            _gpu(
                name, "NVIDIA", "CUDA", _to_int(total), _to_int(free), _to_int(used), driver,
                source="nvidia-smi", utilization_pct=float(utilization) if utilization is not None else None,
            )
        )
    return gpus


def _find_value(item: Dict[str, Any], needles: Iterable[str]) -> Tuple[Optional[Any], str]:
    for key, value in item.items():
        normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
        if any(re.sub(r"[^a-z0-9]", "", needle.lower()) in normalized for needle in needles):
            return value, str(key)
    return None, ""


def _device_dicts(value: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(value, dict):
        keys = " ".join(str(k).lower() for k in value)
        if any(marker in keys for marker in ("vram", "memory total", "asic", "product name", "device name")):
            yield value
        for child in value.values():
            yield from _device_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _device_dicts(child)


def _parse_amd_json(payload: Any, source: str) -> List[Dict[str, Any]]:
    gpus: List[Dict[str, Any]] = []
    for item in _device_dicts(payload):
        total, total_key = _find_value(item, ("vram total memory", "total vram", "vram size", "memory total"))
        used, used_key = _find_value(item, ("vram total used", "used vram", "memory used"))
        name, _ = _find_value(item, ("card series", "product name", "device name", "asic name", "market name"))
        driver, _ = _find_value(item, ("driver version",))
        total_mb = _memory_to_mb(total, total_key)
        used_mb = _memory_to_mb(used, used_key)
        if total_mb is None and not name:
            continue
        gpus.append(
            _gpu(str(name) if name else None, "AMD", "ROCm", total_mb, used=used_mb, driver=str(driver) if driver else None, source=source)
        )
    return _dedupe_gpus(gpus)


def _query_amd_cli() -> List[Dict[str, Any]]:
    commands: List[Tuple[str, List[str]]] = []
    rocm = shutil.which("rocm-smi")
    amd = shutil.which("amd-smi")
    if rocm:
        commands.append(("rocm-smi", [rocm, "--showmeminfo", "vram", "--showproductname", "--showdriverversion", "--json"]))
    if amd:
        commands.append(("amd-smi", [amd, "static", "--asic", "--vram", "--driver", "--json"]))
    for source, command in commands:
        output = _run(command)
        if not output:
            continue
        try:
            parsed = _parse_amd_json(json.loads(output), source)
        except (TypeError, ValueError):
            parsed = []
        if parsed:
            return parsed
    return []


def _sysfs_gpu_name(card: Path, vendor: str) -> str:
    try:
        address = card.resolve().name
    except OSError:
        address = ""
    lspci = shutil.which("lspci")
    output = _run([lspci, "-s", address]) if lspci and address else None
    if output:
        name = output.strip().split(": ", 1)[-1].strip()
        if name:
            return name
    return f"{vendor} GPU ({card.name})"


def _read_int(path: Path) -> Optional[int]:
    try:
        return int(path.read_text(encoding="ascii").strip(), 0)
    except (OSError, ValueError):
        return None


def _query_linux_sysfs() -> List[Dict[str, Any]]:
    root = Path("/sys/class/drm")
    if platform.system() != "Linux" or not root.is_dir():
        return []
    gpus: List[Dict[str, Any]] = []
    for card in sorted(root.glob("card[0-9]*")):
        device = card / "device"
        vendor_id = (device / "vendor").read_text(encoding="ascii").strip().lower() if (device / "vendor").is_file() else ""
        vendor = _VENDOR_IDS.get(vendor_id)
        if vendor not in {"AMD", "Intel", "NVIDIA"}:
            continue
        total_bytes = None
        used_bytes = None
        for filename in ("mem_info_vram_total", "lmem_total_bytes"):
            total_bytes = _read_int(device / filename)
            if total_bytes is not None:
                break
        for filename in ("mem_info_vram_used", "lmem_used_bytes"):
            used_bytes = _read_int(device / filename)
            if used_bytes is not None:
                break
        backend = {"AMD": "Vulkan", "Intel": "SYCL", "NVIDIA": "CUDA"}[vendor]
        gpus.append(
            _gpu(
                _sysfs_gpu_name(device, vendor),
                vendor,
                backend,
                _bytes_to_mb(total_bytes),
                used=_bytes_to_mb(used_bytes),
                source="linux-sysfs",
            )
        )
    return gpus


def _vendor_from_name(name: str) -> Optional[str]:
    lowered = name.lower()
    if "nvidia" in lowered:
        return "NVIDIA"
    if "amd" in lowered or "radeon" in lowered or "ati " in lowered:
        return "AMD"
    if "intel" in lowered or re.search(r"\barc\s+[ab]\d", lowered):
        return "Intel"
    if "apple" in lowered:
        return "Apple"
    return None


def _looks_integrated(name: str, vendor: str) -> bool:
    lowered = name.lower()
    if vendor == "Intel":
        return not bool(re.search(r"\barc(?:\(tm\))?\s+(?:pro\s+)?[ab]\d", lowered))
    if vendor == "AMD":
        return "radeon graphics" in lowered and not bool(re.search(r"\brx\s*\d|\bpro\s+w\d|instinct", lowered))
    return False


def _query_windows_adapters(system: Dict[str, Any]) -> List[Dict[str, Any]]:
    if platform.system() != "Windows":
        return []
    shell = shutil.which("powershell") or shutil.which("pwsh")
    if not shell:
        return []
    command = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterRAM,DriverVersion,PNPDeviceID | ConvertTo-Json -Compress"
    )
    output = _run([shell, "-NoProfile", "-NonInteractive", "-Command", command], timeout=15)
    if not output:
        return []
    try:
        payload = json.loads(output)
    except ValueError:
        return []
    rows = payload if isinstance(payload, list) else [payload]
    gpus: List[Dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        name = str(item.get("Name") or "").strip()
        vendor = _vendor_from_name(name)
        if not vendor or "microsoft basic" in name.lower() or "remote display" in name.lower():
            continue
        unified = _looks_integrated(name, vendor)
        total = None if unified else _bytes_to_mb(item.get("AdapterRAM"))
        backend = {"NVIDIA": "CUDA", "AMD": "Vulkan", "Intel": "SYCL"}.get(vendor, "Vulkan")
        gpu = _gpu(
            name,
            vendor,
            backend,
            total,
            driver=str(item.get("DriverVersion") or "") or None,
            unified=unified,
            source="windows-cim",
            estimated=total is not None,
        )
        if unified:
            gpu["shared_memory_total_mb"] = system.get("ram_total_mb")
            gpu["shared_memory_available_mb"] = system.get("ram_available_mb")
        gpus.append(gpu)
    return gpus


def _query_macos_displays() -> List[Dict[str, Any]]:
    if platform.system() != "Darwin":
        return []
    profiler = shutil.which("system_profiler")
    output = _run([profiler, "SPDisplaysDataType", "-json"], timeout=20) if profiler else None
    if not output:
        return []
    try:
        rows = json.loads(output).get("SPDisplaysDataType", [])
    except (AttributeError, ValueError):
        return []
    gpus: List[Dict[str, Any]] = []
    for item in rows:
        name = str(item.get("sppci_model") or item.get("_name") or "").strip()
        vendor = _vendor_from_name(name)
        if not vendor or vendor == "Apple":
            continue
        memory = item.get("spdisplays_vram") or item.get("spdisplays_vram_shared")
        total = _memory_to_mb(memory, "vram")
        gpus.append(_gpu(name, vendor, "Metal", total, source="system-profiler"))
    return gpus


def _detect_apple_gpu(system: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None
    chip = None
    exe = shutil.which("sysctl")
    output = _run([exe, "-n", "machdep.cpu.brand_string"], timeout=5) if exe else None
    if output:
        chip = output.strip() or None
    gpu = _gpu(
        f"{chip} GPU (Metal)" if chip else "Apple Silicon GPU (Metal)",
        "Apple",
        "Metal",
        None,
        unified=True,
        source="unified-system-memory",
    )
    gpu["shared_memory_total_mb"] = system.get("ram_total_mb")
    gpu["shared_memory_available_mb"] = system.get("ram_available_mb")
    return gpu


def _name_key(gpu: Dict[str, Any]) -> str:
    name = re.sub(r"[^a-z0-9]", "", str(gpu.get("name") or "").lower())
    return f"{gpu.get('vendor')}:{name}"


def _dedupe_gpus(gpus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for candidate in gpus:
        # Two same-name records from the same probe are two physical cards.
        # A same-name record from another probe is duplicate telemetry for one
        # of those cards and should only fill missing fields.
        match = next(
            (
                item
                for item in out
                if _name_key(item) == _name_key(candidate)
                and item.get("memory_source") != candidate.get("memory_source")
            ),
            None,
        )
        if match is None:
            out.append(candidate)
            continue
        # Prefer live vendor telemetry, but fill any fields it omitted.
        for key, value in candidate.items():
            if match.get(key) is None and value is not None:
                match[key] = value
    return out


def summarize_gpus(gpus: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, Dict[str, Any]] = {}
    for index, gpu in enumerate(gpus):
        key = f"{gpu.get('vendor')}:{gpu.get('backend')}"
        group = groups.setdefault(
            key,
            {
                "key": key,
                "vendor": gpu.get("vendor"),
                "backend": gpu.get("backend"),
                "gpu_indexes": [],
                "gpu_count": 0,
                "vram_total_mb": 0,
                "vram_free_mb": 0,
                "memory_known": True,
            },
        )
        group["gpu_indexes"].append(index)
        group["gpu_count"] += 1
        total = gpu.get("vram_total_mb")
        free = gpu.get("vram_free_mb")
        if gpu.get("unified_memory"):
            total = gpu.get("shared_memory_total_mb")
            free = gpu.get("shared_memory_available_mb")
        if total is None:
            group["memory_known"] = False
        else:
            group["vram_total_mb"] += int(total)
        if free is None:
            if total is None:
                group["memory_known"] = False
            else:
                group["vram_free_mb"] += int(total)
        else:
            group["vram_free_mb"] += int(free)
    compatible = sorted(
        groups.values(),
        key=lambda item: (item["vram_free_mb"], item["vram_total_mb"]),
        reverse=True,
    )
    best = compatible[0] if compatible else None
    return {
        "gpu_count": len(gpus),
        "multi_gpu": len(gpus) > 1,
        "compatible_groups": compatible,
        "best_pool_key": best.get("key") if best else None,
        "best_pool_total_mb": best.get("vram_total_mb") if best and best.get("memory_known") else None,
        "best_pool_free_mb": best.get("vram_free_mb") if best and best.get("memory_known") else None,
        "single_gpu_max_total_mb": max((g.get("vram_total_mb") or 0 for g in gpus), default=0) or None,
        "single_gpu_max_free_mb": max((g.get("vram_free_mb") or g.get("vram_total_mb") or 0 for g in gpus), default=0) or None,
    }


def estimate_gpu_placement(required_gb: float, gpus: List[Dict[str, Any]]) -> Dict[str, Any]:
    required_mb = max(0, int(required_gb * 1024))
    candidates = [
        (
            index,
            gpu,
            gpu.get("vram_free_mb")
            or gpu.get("vram_total_mb")
            or gpu.get("shared_memory_available_mb")
            or gpu.get("shared_memory_total_mb"),
        )
        for index, gpu in enumerate(gpus)
        if (
            gpu.get("vram_free_mb")
            or gpu.get("vram_total_mb")
            or gpu.get("shared_memory_available_mb")
            or gpu.get("shared_memory_total_mb")
        )
        is not None
    ]
    candidates.sort(key=lambda item: int(item[2]), reverse=True)
    for index, gpu, available in candidates:
        if int(available) >= required_mb:
            return {
                "mode": "single_gpu",
                "supported": True,
                "required_mb": required_mb,
                "available_mb": int(available),
                "gpu_indexes": [index],
                "allocations": [{"gpu_index": index, "name": gpu.get("name"), "memory_mb": required_mb}],
                "utilization_pct": round(required_mb / max(1, int(available)) * 100, 1),
                "note": "The model fits on one GPU; no tensor split is required.",
            }

    summary = summarize_gpus(gpus)
    for group in summary["compatible_groups"]:
        if not group.get("memory_known") or group.get("gpu_count", 0) < 2:
            continue
        available = int(group.get("vram_free_mb") or 0)
        if available < required_mb:
            continue
        remaining = required_mb
        allocations = []
        for index in group["gpu_indexes"]:
            gpu = gpus[index]
            capacity = int(
                gpu.get("vram_free_mb")
                or gpu.get("vram_total_mb")
                or gpu.get("shared_memory_available_mb")
                or gpu.get("shared_memory_total_mb")
                or 0
            )
            assigned = min(capacity, remaining)
            if assigned:
                allocations.append({"gpu_index": index, "name": gpu.get("name"), "memory_mb": assigned})
                remaining -= assigned
            if remaining <= 0:
                break
        return {
            "mode": "multi_gpu_split",
            "supported": True,
            "required_mb": required_mb,
            "available_mb": available,
            "gpu_indexes": list(group["gpu_indexes"]),
            "allocations": allocations,
            "utilization_pct": round(required_mb / max(1, available) * 100, 1),
            "backend": group.get("backend"),
            "vendor": group.get("vendor"),
            "note": (
                "The estimate requires a split across compatible GPUs. Ollama may distribute layers "
                "automatically; verify the actual placement after loading because interconnect and model "
                "architecture affect whether the split is practical."
            ),
        }

    best_pool = summary.get("best_pool_free_mb")
    return {
        "mode": "cpu_offload",
        "supported": False,
        "required_mb": required_mb,
        "available_mb": best_pool,
        "gpu_indexes": [],
        "allocations": [],
        "utilization_pct": round(required_mb / best_pool * 100, 1) if best_pool else None,
        "note": "No compatible GPU pool has enough memory; partial or full CPU offload is required.",
    }


def detect_hardware() -> Dict[str, Any]:
    """Detect all GPUs plus CPU/RAM. Always returns a well-formed response."""
    system = _cpu_and_memory()
    gpus: List[Dict[str, Any]] = []
    gpus.extend(_query_nvidia_smi() or [])
    gpus.extend(_query_amd_cli() or [])
    gpus.extend(_query_linux_sysfs() or [])
    gpus.extend(_query_windows_adapters(system) or [])
    gpus.extend(_query_macos_displays() or [])
    apple = _detect_apple_gpu(system)
    if apple:
        gpus.append(apple)
    gpus = _dedupe_gpus(gpus)
    gpus.sort(key=lambda gpu: gpu.get("vram_free_mb") or gpu.get("vram_total_mb") or 0, reverse=True)
    summary = summarize_gpus(gpus)

    if not gpus:
        message = "No supported GPU telemetry was detected. CPU-only inference remains available."
    elif summary["multi_gpu"]:
        message = (
            f"Detected {summary['gpu_count']} GPUs. Fit checks use the largest compatible VRAM pool; "
            "mixed vendors/backends are not added together."
        )
    elif gpus[0].get("unified_memory"):
        message = "Unified memory GPU detected; fit checks use the shared available system-memory pool."
    else:
        message = None
    return {
        "success": True,
        "gpu_available": bool(gpus),
        "gpus": gpus,
        "gpu_summary": summary,
        "system": system,
        "message": message,
    }


@router.get("/system/hardware")
def system_hardware() -> Dict[str, Any]:
    return detect_hardware()
