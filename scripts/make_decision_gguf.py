"""Make a smaller, faster GGUF of a decision model with llama.cpp's own quantizer.

    python scripts/make_decision_gguf.py ggml-org/vllm-d3-lite-GGUF --quant Q4_K_M

The published d3 GGUF files are Q8_0 (and BF16). On a CPU the model is limited by memory bandwidth, so a
Q4_K_M file of the same model answers about 1.8 times faster with the same routing accuracy in LocalDeploy's
comparison (docs/MODELS.md). A profile with ``"quantize": "Q4_K_M"`` makes this file on first use by itself;
this script makes it ahead of time, or into another folder. It downloads the BF16 file, runs
``llama-quantize`` (next to llama-server, on PATH, or from ``winget install ggml.llamacpp``) and prints the
profile line to use. Needs ``huggingface_hub`` (in requirements.txt).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from localdeploy.backends.decision import (  # noqa: E402
    LLAMACPP_GET,
    DecisionUnavailable,
    find_llama_quantize,
    find_llama_server,
    quantized_gguf,
)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo", help="Hugging Face repository with the GGUF files, e.g. ggml-org/vllm-d3-lite-GGUF")
    parser.add_argument("--quant", default="Q4_K_M", help="target type, e.g. Q4_K_M, Q5_K_M, Q6_K (default Q4_K_M)")
    parser.add_argument("--source-file", help="file in the repository to quantize (default: its BF16 file)")
    parser.add_argument("--out", default="models/decision", help="folder for the result (default models/decision)")
    parser.add_argument("--llama-quantize", help="path to llama-quantize")
    parser.add_argument("--threads", type=int, default=0, help="threads for quantizing (default: all)")
    args = parser.parse_args(argv)

    quantizer = args.llama_quantize or find_llama_quantize(find_llama_server(os.getenv("LLAMACPP_SERVER_PATH", "").strip()))
    if not quantizer:
        print(f"llama-quantize not found. It ships next to llama-server. {LLAMACPP_GET}", file=sys.stderr)
        return 1
    try:
        target = quantized_gguf(
            args.repo, args.quant, Path(args.out), quantizer=quantizer, source_file=args.source_file,
            threads=args.threads, log=lambda message: print(message, flush=True),
        )
    except DecisionUnavailable as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"\n{target} is ready ({target.stat().st_size / 1e6:,.0f} MB). In config.json use:")
    print(f'  "adapter": "llamacpp", "model_id": "{target.resolve().as_posix()}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
