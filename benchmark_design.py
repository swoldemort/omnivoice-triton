#!/usr/bin/env python3
"""Benchmark throughput for voice design with short texts."""

import concurrent.futures
import statistics
import time

import numpy as np
import tritonclient.http as httpclient

URL = "localhost:8000"
MODEL = "omnivoice"

INSTRUCTS = [
    "female, british accent",
    "male, american accent",
    "female, whisper",
    "male, elderly",
    "female, child",
    "male, low pitch",
    "female, high pitch",
    "male, british accent",
]

SHORT_TEXTS = [
    "Hello world.",
    "How are you?",
    "Good morning.",
    "See you soon.",
    "Nice to meet.",
    "Thank you much.",
    "Have a day.",
    "Call me now.",
]


def infer(text: str, instruct: str) -> float:
    client = httpclient.InferenceServerClient(URL)
    inputs = [
        httpclient.InferInput("text", [1, 1], "BYTES"),
        httpclient.InferInput("ref_audio", [1, 1], "BYTES"),
        httpclient.InferInput("ref_text", [1, 1], "BYTES"),
        httpclient.InferInput("instruct", [1, 1], "BYTES"),
        httpclient.InferInput("language", [1, 1], "BYTES"),
    ]
    inputs[0].set_data_from_numpy(np.array([[text.encode("utf-8")]], dtype=object))
    inputs[1].set_data_from_numpy(np.array([[b""]], dtype=object))
    inputs[2].set_data_from_numpy(np.array([[b""]], dtype=object))
    inputs[3].set_data_from_numpy(np.array([[instruct.encode("utf-8")]], dtype=object))
    inputs[4].set_data_from_numpy(np.array([[b"en"]], dtype=object))

    outputs = [
        httpclient.InferRequestedOutput("audio"),
        httpclient.InferRequestedOutput("sample_rate"),
    ]

    t0 = time.perf_counter()
    client.infer(model_name=MODEL, inputs=inputs, outputs=outputs)
    return time.perf_counter() - t0


def benchmark(n: int, concurrency: int):
    texts = [SHORT_TEXTS[i % len(SHORT_TEXTS)] for i in range(n)]
    instructs = [INSTRUCTS[i % len(INSTRUCTS)] for i in range(n)]
    print(f"\n=== Voice Design Benchmark: {n} requests, concurrency={concurrency} ===")

    overall_t0 = time.perf_counter()
    latencies = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(infer, t, ins) for t, ins in zip(texts, instructs)]
        for f in concurrent.futures.as_completed(futures):
            try:
                latencies.append(f.result())
            except Exception as e:
                print(f"Request failed: {e}")

    wall = time.perf_counter() - overall_t0
    throughput = n / wall

    print(f"Wall-clock time: {wall:.2f}s")
    print(f"Throughput: {throughput:.1f} req/s")
    print(f"Mean latency: {statistics.mean(latencies):.2f}s")
    print(f"Median latency: {statistics.median(latencies):.2f}s")
    print(f"P95 latency: {statistics.quantiles(latencies, n=20)[18]:.2f}s")
    print(f"P99 latency: {statistics.quantiles(latencies, n=100)[98]:.2f}s")
    print(f"Min latency: {min(latencies):.2f}s")
    print(f"Max latency: {max(latencies):.2f}s")
    return throughput


def main():
    # Warm-up
    print("Warming up...")
    infer("Hi there.", "female, British accent")
    time.sleep(1)

    for n, concurrency in [(16, 16), (32, 16), (64, 16)]:
        benchmark(n, concurrency)
        time.sleep(2)


if __name__ == "__main__":
    main()
