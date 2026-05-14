"""Build TensorRT-LLM engine for OmniVoice Qwen3 backbone.

Usage:
    python -m omnivoice_triton.models.trtllm.build_engine \
        --model_dir /path/to/OmniVoice \
        --output_dir /path/to/trtllm_engine
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_tensorrt_llm_path() -> Path:
    """Find tensorrt_llm package path via pip."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", "tensorrt_llm"],
        capture_output=True,
        text=True,
        check=True,
    )
    location = [l.split(": ", 1)[1] for l in result.stdout.splitlines() if l.startswith("Location:")][0]
    return Path(location) / "tensorrt_llm"


def register_omnivoice_model(trtllm_path: Path, patch_dir: Path) -> None:
    """Copy patch/omnivoice into tensorrt_llm.models and register in MODEL_MAP."""
    target_dir = trtllm_path / "models" / "omnivoice"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(patch_dir / "omnivoice", target_dir)
    print(f"  Copied patch to {target_dir}")

    init_file = trtllm_path / "models" / "__init__.py"
    init_content = init_file.read_text()
    register_line = "\nfrom .omnivoice.model import OmniVoice\nMODEL_MAP['OmniVoice'] = OmniVoice\n"
    if "OmniVoice" not in init_content:
        with open(init_file, "a") as f:
            f.write(register_line)
        print("  Registered OmniVoice in MODEL_MAP")


def build_engine(args: argparse.Namespace) -> None:
    """Full pipeline: convert checkpoint -> register model -> build engine."""
    model_dir = Path(args.model_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    engine_dir = Path(args.output_dir)
    patch_dir = Path(__file__).parent / "omnivoice"

    # Step 1: Convert checkpoint
    print("Step 1: Converting HF checkpoint to TRT-LLM format")
    from omnivoice_triton.models.trtllm import convert_checkpoint

    convert_checkpoint.main_with_args(model_dir=str(model_dir), output_dir=str(ckpt_dir), dtype=args.dtype)

    # Step 2: Register OmniVoice model in tensorrt_llm
    print("Step 2: Registering OmniVoice model in tensorrt_llm")
    trtllm_path = find_tensorrt_llm_path()
    register_omnivoice_model(trtllm_path, patch_dir.parent)

    # Step 3: Build engine
    print("Step 3: Building TRT engine")
    cmd = [
        "trtllm-build",
        "--checkpoint_dir", str(ckpt_dir),
        "--max_batch_size", str(args.max_batch_size),
        "--output_dir", str(engine_dir),
        "--remove_input_padding", "disable",
    ]

    subprocess.run(cmd, check=True)
    print(f"Engine built at {engine_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Path to HF OmniVoice model")
    parser.add_argument("--output_dir", required=True, help="Path to save TRT engine")
    parser.add_argument("--checkpoint_dir", default="/tmp/trtllm_ckpt", help="Temp checkpoint dir")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max_batch_size", type=int, default=16)
    args = parser.parse_args()
    build_engine(args)


if __name__ == "__main__":
    main()
