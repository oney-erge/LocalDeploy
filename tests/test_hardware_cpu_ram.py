"""Phase 1 - hardware probe now reports CPU model, cores, and RAM."""
from __future__ import annotations

import sys
import types

from localdeploy.control import hardware as hardware_module
from localdeploy.control.hardware import _cpu_and_memory, detect_hardware


def _no_proc_cpuinfo(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise OSError("no /proc/cpuinfo on this platform")

    monkeypatch.setattr("builtins.open", refuse)


def test_windows_cpu_model_is_the_registry_name_not_the_raw_identifier(monkeypatch):
    class FakeKey:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    fake = types.SimpleNamespace(
        HKEY_LOCAL_MACHINE=object(),
        OpenKey=lambda *_a: FakeKey(),
        QueryValueEx=lambda _key, name: ("  AMD Ryzen 9 5900HS   with Radeon Graphics ", 1),
    )
    monkeypatch.setitem(sys.modules, "winreg", fake)
    monkeypatch.setattr(hardware_module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hardware_module.platform, "processor", lambda: "AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD")
    _no_proc_cpuinfo(monkeypatch)
    assert hardware_module._cpu_model() == "AMD Ryzen 9 5900HS with Radeon Graphics"


def test_windows_cpu_model_falls_back_when_the_registry_is_unreadable(monkeypatch):
    def broken(*_a):
        raise OSError("registry unavailable")

    monkeypatch.setitem(sys.modules, "winreg", types.SimpleNamespace(HKEY_LOCAL_MACHINE=object(), OpenKey=broken))
    monkeypatch.setattr(hardware_module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hardware_module.platform, "processor", lambda: "Intel64 Family 6 Model 140")
    _no_proc_cpuinfo(monkeypatch)
    assert hardware_module._cpu_model() == "Intel64 Family 6 Model 140"


def test_macos_cpu_model_uses_sysctl_brand_string(monkeypatch):
    monkeypatch.setattr(hardware_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hardware_module.platform, "processor", lambda: "arm")
    monkeypatch.setattr(hardware_module.shutil, "which", lambda name: "/usr/sbin/sysctl")
    monkeypatch.setattr(hardware_module, "_run", lambda command, timeout=10: "Apple M2 Pro\n")
    _no_proc_cpuinfo(monkeypatch)
    assert hardware_module._cpu_model() == "Apple M2 Pro"


def test_system_block_has_cpu_and_ram_keys():
    sys = detect_hardware()["system"]
    # The contract: these keys always exist (values may be None if psutil absent).
    for key in (
        "cpu_model",
        "logical_cores",
        "physical_cores",
        "ram_total_mb",
        "ram_available_mb",
        "psutil_available",
        "ram_probe_message",
    ):
        assert key in sys


def test_detect_hardware_never_raises_and_is_well_formed():
    hw = detect_hardware()
    assert hw["success"] is True
    assert isinstance(hw["gpus"], list)
    assert "system" in hw


def test_cpu_and_memory_logical_cores_populated():
    info = _cpu_and_memory()
    # logical_cores comes from os.cpu_count(); should be a positive int on any CI host.
    assert info["logical_cores"] is None or info["logical_cores"] >= 1


def test_psutil_path_populates_ram_when_available():
    info = _cpu_and_memory()
    try:
        import psutil  # noqa: F401
    except Exception:
        return  # psutil not installed - RAM legitimately None; nothing to assert
    assert info["ram_total_mb"] is not None and info["ram_total_mb"] > 0
    assert info["physical_cores"] is None or info["physical_cores"] >= 1
