"""Final throughput comparison: FasterRunner vs TRTLLMRunner."""

import sys
import time
import statistics

sys.path.insert(0, "/app/src")

from omnivoice_triton.models.faster_runner import FasterRunner
from omnivoice_triton.models.trtllm_runner import TRTLLMRunner

TEXT = "The quick brown fox jumps over the lazy dog."
NUM_STEP = 16

def benchmark(runner, batch_size, n=15):
    texts = [TEXT] * batch_size
    for _ in range(2):
        runner.generate(texts, num_step=NUM_STEP)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        runner.generate(texts, num_step=NUM_STEP)
        times.append(time.perf_counter() - t0)
    return statistics.mean(times)

def main():
    print("Loading FasterRunner...")
    faster = FasterRunner(device="cuda", model_id="k2-fsa/OmniVoice", dtype="fp16")
    faster.load_model()
    
    print("Loading TRTLLMRunner...")
    trt = TRTLLMRunner(device="cuda", model_id="k2-fsa/OmniVoice", dtype="fp16", engine_dir="/tmp/trtllm_engine")
    trt.load_model()
    
    print("\n=== Throughput (req/s) ===")
    print(f"{'Batch':>6} | {'Faster':>10} | {'TRT-LLM':>10} | {'Speedup':>8}")
    print("-" * 45)
    for bs in [1, 4, 8, 16]:
        t_faster = benchmark(faster, bs)
        t_trt = benchmark(trt, bs)
        thr_faster = bs / t_faster
        thr_trt = bs / t_trt
        print(f"{bs:>6} | {thr_faster:>9.1f} | {thr_trt:>9.1f} | {thr_trt/thr_faster:>7.2f}x")
    
    faster.unload_model()
    trt.unload_model()

if __name__ == "__main__":
    main()
