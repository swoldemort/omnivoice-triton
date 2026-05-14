"""Test DynamicBatcher with TRTLLMRunner."""

import sys
import time

sys.path.insert(0, "/app/src")

from omnivoice_triton.models.trtllm_runner import TRTLLMRunner
from omnivoice_triton.batching import DynamicBatcher, Request

TEXT = "The quick brown fox jumps over the lazy dog."
NUM_STEP = 16

def main():
    runner = TRTLLMRunner(
        device="cuda", model_id="k2-fsa/OmniVoice", dtype="fp16",
        engine_dir="/tmp/trtllm_engine"
    )
    print("Loading model...")
    runner.load_model()

    batcher = DynamicBatcher(runner, max_batch_size=8, max_wait_ms=50.0)
    batcher.start()

    # Test 1: single request
    print("\n=== Single request ===")
    req = batcher.submit(Request(text=TEXT, num_step=NUM_STEP))
    req.wait()
    print(f"Audio shape: {req.audio.shape if req.audio is not None else 'None'}")
    print(f"Error: {req.error}")

    # Test 2: 10 concurrent requests (should batch)
    print("\n=== 10 concurrent requests ===")
    requests = [batcher.submit(Request(text=TEXT, num_step=NUM_STEP)) for _ in range(10)]
    t0 = time.perf_counter()
    for r in requests:
        r.wait()
    elapsed = time.perf_counter() - t0
    print(f"Total time: {elapsed:.3f}s")
    print(f"Throughput: {10/elapsed:.1f} req/s")
    print(f"Latencies: p50={sorted([r.audio is not None for r in requests])[5]}")

    # Test 3: 20 requests fired as fast as possible
    print("\n=== 20 rapid-fire requests ===")
    requests = [batcher.submit(Request(text=TEXT, num_step=NUM_STEP)) for _ in range(20)]
    t0 = time.perf_counter()
    for r in requests:
        r.wait()
    elapsed = time.perf_counter() - t0
    print(f"Total time: {elapsed:.3f}s")
    print(f"Throughput: {20/elapsed:.1f} req/s")

    batcher.shutdown()
    print("\nDone.")

if __name__ == "__main__":
    main()
