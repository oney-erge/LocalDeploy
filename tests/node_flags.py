"""Command-line flags the frontend tests pass to Node, which differ between Node releases."""
from __future__ import annotations

import functools
import subprocess

MODULE_FLAG = "--experimental-default-type=module"


@functools.lru_cache(maxsize=1)
def node_module_flags() -> tuple[str, ...]:
    """Flags that make Node load the typeless ``.js`` files under ``localdeploy/web/js`` as ES modules.

    Node before 22.7 needs ``--experimental-default-type=module`` for that. Newer releases detect ES module
    syntax by themselves, and Node 25 removed the flag, so pass it only where this Node accepts it.
    """
    probe = subprocess.run(["node", MODULE_FLAG, "-e", "0"], capture_output=True, text=True)
    return (MODULE_FLAG,) if probe.returncode == 0 else ()
