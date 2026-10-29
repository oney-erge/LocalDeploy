"""Make a smaller, faster GGUF of a decision model with llama.cpp's own quantizer.

    python scripts/make_decision_gguf.py ggml-org/vllm-d3-lite-GGUF --quant Q4_K_M

The published d3 GGUF files are Q8_0 (and BF16). On a CPU the model is limited by memory bandwidth, so a
Q4_K_M file of the same model answers about 1.8 times faster with the same routing accuracy in LocalDeploy's
comparison (docs/MODELS.md). The script downloads the BF16 file, runs ``llama-quantize`` (it ships in every
llama.cpp release next to ``llama-server``) and prints the profile line to use. Needs ``huggingface_hub``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def find_quantizer(explicit: Optional[str]) -> str:
    """llama-quantize: the argument, else next to LLAMACPP_SERVER_PATH, else on PATH."""
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    server = os.getenv("LLAMACPP_SERVER_PATH")
    if server:
        candidates.append(Path(server).with_name("llama-quantize.exe" if os.name == "nt" else "llama-quantize"))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    found = shutil.which("llama-quantize")
    if not found:
        raise SystemExit(
            "llama-quantize not found. It is in every llama.cpp release (https://github.com/ggml-org/llama.cpp/releases): "
            "pass --llama-quantize PATH, or set LLAMACPP_SERVER_PATH to the llama-server in the same folder."
        )
    return found


def pick_source(files: List[str], wanted: Optional[str]) -> str:
    ggufs = [f for f in files if f.lower().endswith(".gguf") and not Path(f).name.lower().startswith("mmproj")]
    if wanted:
        if wanted not in ggufs:
            raise SystemExit(f"{wanted} is not a model file in that repository. Found: {', '.join(sorted(ggufs)) or 'none'}")
        return wanted
    for needle in ("bf16", "f16", "f32"):
        for name in ggufs:
            if needle in name.lower():
                return name
    raise SystemExit("No BF16/F16 file found to quantize from; name one with --source-file.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo", help="Hugging Face repository with the GGUF files, e.g. ggml-org/vllm-d3-lite-GGUF")
    parser.add_argument("--quant", default="Q4_K_M", help="target type, e.g. Q4_K_M, Q5_K_M, Q6_K (default Q4_K_M)")
    parser.add_argument("--source-file", help="file in the repository to quantize (default: its BF16 file)")
    parser.add_argument("--out", default="models/decision", help="folder for the result (default models/decision)")
    parser.add_argument("--llama-quantize", help="path to llama-quantize")
    parser.add_argument("--threads", type=int, default=0, help="threads for quantizing (default: all)")
    args = parser.parse_args(argv)

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        raise SystemExit("huggingface_hub is required: python -m pip install -r requirements.txt")

    quantizer = find_quantizer(args.llama_quantize)
    offline = os.getenv("OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}
    source_name = pick_source(HfApi().list_repo_files(args.repo) if not offline else [args.source_file or ""], args.source_file)
    print(f"downloading {args.repo}/{source_name} ...", flush=True)
    source = hf_hub_download(args.repo, source_name, local_files_only=offline)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(source_name).stem
    for tail in ("-BF16", "-bf16", "-F16", "-f16", "-F32", "-f32"):
        if stem.endswith(tail):
            stem = stem[: -len(tail)]
    target = out_dir / f"{stem}-{args.quant}.gguf"
    command = [quantizer, source, str(target), args.quant]
    if args.threads:
        command.append(str(args.threads))
    print("running:", " ".join(command), flush=True)
    result = subprocess.run(command, check=False)  # noqa: S603 - the quantizer path and arguments are the caller's own
    if result.returncode != 0 or not target.is_file():
        print(f"llama-quantize failed (exit {result.returncode}).", file=sys.stderr)
        return result.returncode or 1
    print(f"\nwrote {target} ({target.stat().st_size / 1e6:,.0f} MB). In config.json use:")
    print(f'  "adapter": "llamacpp", "model_id": "{target.resolve().as_posix()}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
