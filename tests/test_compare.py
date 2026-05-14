"""Compare PyTorch vs TRT-LLM for longer text and num_step=16."""

import sys
sys.path.insert(0, "/app/src")

from omnivoice_triton.models.base_runner import BaseRunner
from omnivoice_triton.models.trtllm_runner import TRTLLMRunner

TEXT = "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet at least once."
NUM_STEP = 16

def test_runner(name, runner):
    print(f"\n=== {name} ===")
    runner.load_model()
    result = runner.generate(TEXT, num_step=NUM_STEP)
    print(f"Time: {result['time_s']:.2f}s | VRAM: {result['peak_vram_gb']:.2f}GB")
    runner.unload_model()

test_runner("PyTorch Baseline", BaseRunner(device="cuda", model_id="k2-fsa/OmniVoice", dtype="fp16"))
test_runner("TRT-LLM", TRTLLMRunner(device="cuda", model_id="k2-fsa/OmniVoice", dtype="fp16", engine_dir="/tmp/trtllm_engine"))
