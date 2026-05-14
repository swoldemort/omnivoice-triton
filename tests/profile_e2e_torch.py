"""Profile e2e generation with PyTorch profiler."""

import sys

sys.path.insert(0, "/app/src")

import torch
from omnivoice_triton.models.trtllm_runner import TRTLLMRunner

TEXT = "The quick brown fox jumps over the lazy dog."
NUM_STEP = 16

def main():
    runner = TRTLLMRunner(
        device="cuda", model_id="k2-fsa/OmniVoice", dtype="fp16",
        engine_dir="/tmp/trtllm_engine"
    )
    print("Loading...")
    runner.load_model()
    
    # Warmup
    runner.generate(TEXT, num_step=NUM_STEP)
    
    print("Profiling...")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        runner.generate(TEXT, num_step=NUM_STEP)
    
    print("\n=== Top 20 CUDA ops by time ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    print("\n=== Top 20 CPU ops by time ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

if __name__ == "__main__":
    main()
