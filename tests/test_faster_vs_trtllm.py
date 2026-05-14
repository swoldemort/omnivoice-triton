"""Compare FasterRunner (CUDA Graph) vs TRTLLMRunner."""

import sys
sys.path.insert(0, "/app/src")

from omnivoice_triton.models.faster_runner import FasterRunner
from omnivoice_triton.models.trtllm_runner import TRTLLMRunner

TEXT = "The quick brown fox jumps over the lazy dog."
NUM_STEP = 16

def test_runner(name, runner):
    print(f"\n=== {name} ===")
    runner.load_model()
    result = runner.generate(TEXT, num_step=NUM_STEP)
    print(f"Time: {result['time_s']:.2f}s | VRAM: {result['peak_vram_gb']:.2f}GB")
    runner.unload_model()

test_runner("FasterRunner (CUDA Graph)", FasterRunner(device="cuda", model_id="k2-fsa/OmniVoice", dtype="fp16"))
test_runner("TRTLLMRunner", TRTLLMRunner(device="cuda", model_id="k2-fsa/OmniVoice", dtype="fp16", engine_dir="/tmp/trtllm_engine"))
