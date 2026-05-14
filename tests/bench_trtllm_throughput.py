"""Benchmark TRT-LLM throughput with batching and Poisson arrivals."""

import sys
import time
import random
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/app/src")

from omnivoice_triton.models.trtllm_runner import TRTLLMRunner

TEXT = "The quick brown fox jumps over the lazy dog."
NUM_STEP = 16
WARMUP = 2

def benchmark_single(runner, n=10):
    times = []
    for _ in range(WARMUP):
        runner.generate(TEXT, num_step=NUM_STEP)
    for _ in range(n):
        t0 = time.perf_counter()
        runner.generate(TEXT, num_step=NUM_STEP)
        times.append(time.perf_counter() - t0)
    return times

def benchmark_batch(runner, batch_size=16, n=10):
    texts = [TEXT] * batch_size
    times = []
    for _ in range(WARMUP):
        runner.generate(texts, num_step=NUM_STEP)
    for _ in range(n):
        t0 = time.perf_counter()
        runner.generate(texts, num_step=NUM_STEP)
        times.append(time.perf_counter() - t0)
    return times

def benchmark_poisson(runner, lambda_rate=10, duration_sec=10):
    """Simulate Poisson arrivals and measure throughput."""
    results = []
    lock = False  # simple flag
    import threading
    result_lock = threading.Lock()
    
    def worker():
        t0 = time.perf_counter()
        runner.generate(TEXT, num_step=NUM_STEP)
        latency = time.perf_counter() - t0
        with result_lock:
            results.append(latency)
    
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=64) as pool:
        futures = []
        while time.perf_counter() - start < duration_sec:
            interval = random.expovariate(lambda_rate)
            time.sleep(interval)
            futures.append(pool.submit(worker))
        # Wait for all in-flight
        for f in as_completed(futures):
            pass
    
    elapsed = time.perf_counter() - start
    throughput = len(results) / elapsed
    return throughput, results

def main():
    runner = TRTLLMRunner(device="cuda", model_id="k2-fsa/OmniVoice", dtype="fp16", engine_dir="/tmp/trtllm_engine")
    print("Loading model...")
    runner.load_model()
    print("Warming up...")
    runner.generate(TEXT, num_step=NUM_STEP)
    
    print("\n=== Single request ===")
    times = benchmark_single(runner, n=10)
    print(f"Latency: {statistics.mean(times):.3f}s ± {statistics.stdev(times):.3f}s")
    print(f"Throughput: {1/statistics.mean(times):.1f} req/s")
    
    print("\n=== Batch = 16 ===")
    times = benchmark_batch(runner, batch_size=16, n=10)
    print(f"Batch latency: {statistics.mean(times):.3f}s ± {statistics.stdev(times):.3f}s")
    print(f"Throughput: {16/statistics.mean(times):.1f} req/s")
    
    for rate in [5, 10, 20]:
        print(f"\n=== Poisson λ={rate} req/s (10s) ===")
        throughput, latencies = benchmark_poisson(runner, lambda_rate=rate, duration_sec=10)
        print(f"Actual throughput: {throughput:.1f} req/s")
        if latencies:
            print(f"Latency: p50={statistics.quantiles(latencies, n=100)[49]:.3f}s p99={statistics.quantiles(latencies, n=100)[98]:.3f}s")

if __name__ == "__main__":
    main()
