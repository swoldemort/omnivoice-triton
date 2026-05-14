"""Baseline PyTorch runner for comparison."""

import sys
sys.path.insert(0, "/app/src")

from omnivoice_triton.models.base_runner import BaseRunner

runner = BaseRunner(device="cuda", model_id="k2-fsa/OmniVoice", dtype="fp16")
print("Loading model...")
runner.load_model()
print("Model loaded. Running generation...")
result = runner.generate("Hello, this is a test.", num_step=4)
print(f"Generated audio shape: {result['audio'].shape}")
print(f"Sample rate: {result['sample_rate']}")
print(f"Time: {result['time_s']:.2f}s")
print(f"VRAM: {result['peak_vram_gb']:.2f}GB")
