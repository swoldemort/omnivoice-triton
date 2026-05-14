"""Benchmark TRT-LLM throughput with correct batching (no concurrent threads)."""

import sys
import time
import statistics

sys.path.insert(0, "/app/src")

from omnivoice_triton.models.trtllm_runner import TRTLLMRunner

TEXT = "The quick brown fox jumps over the lazy dog."
NUM_STEP = 16

def benchmark_batch(runner, batch_size=16, n=20):
    texts = [TEXT] * batch_size
    # Warmup
    for _ in range(2):
        runner.generate(texts, num_step=NUM_STEP)
    
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        runner.generate(texts, num_step=NUM_STEP)
        times.append(time.perf_counter() - t0)
    return times

def main():
    runner = TRTLLMRunner(
        device="cuda", model_id="k2-fsa/OmniVoice", dtype="fp16",
        engine_dir="/tmp/trtllm_engine"
    )
    print("Loading model...")
    runner.load_model()
    
    for bs in [1, 4, 8, 16]:
        times = benchmark_batch(runner, batch_size=bs, n=20)
        mean_t = statistics.mean(times)
        req_per_sec = bs / mean_t
        print(f"Batch={bs:2d}: latency={mean_t:.3f}s | throughput={req_per_sec:.1f} req/s | "
              f"ms/req={mean_t*1000/bs:.1f}ms")

if __name__ == "__main__":
    main()
