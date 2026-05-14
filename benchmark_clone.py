#!/usr/bin/env python3
"""Benchmark throughput for voice cloning with short texts."""

import concurrent.futures
import statistics
import time

import numpy as np
import tritonclient.http as httpclient

URL = "localhost:8000"
MODEL = "omnivoice"
REF_AUDIO_PATH = "input.wav"
REF_TEXT = "Hello world."

SHORT_TEXTS = [
    "Hello world.",
    "How are you?",
    "Good morning.",
    "See you soon.",
    "Nice to meet.",
    "Thank you much.",
    "Have a day.",
    "Call me now.",
    "Talk later.",
    "Sounds great.",
    "Let us go.",
    "Wait a bit.",
    "Come here now.",
    "Tell me more.",
    "Keep it up.",
    "Stay safe now.",
]


def load_ref_audio():
    with open(REF_AUDIO_PATH, "rb") as f:
        return f.read()


def infer(text: str, ref_audio_bytes: bytes) -> float:
    client = httpclient.InferenceServerClient(URL)
    inputs = [
        httpclient.InferInput("text", [1, 1], "BYTES"),
        httpclient.InferInput("ref_audio", [1, 1], "BYTES"),
        httpclient.InferInput("ref_text", [1, 1], "BYTES"),
        httpclient.InferInput("instruct", [1, 1], "BYTES"),
        httpclient.InferInput("language", [1, 1], "BYTES"),
    ]
    inputs[0].set_data_from_numpy(np.array([[text.encode("utf-8")]], dtype=object))
    inputs[1].set_data_from_numpy(np.array([[ref_audio_bytes]], dtype=object))
    inputs[2].set_data_from_numpy(np.array([[REF_TEXT.encode("utf-8")]], dtype=object))
    inputs[3].set_data_from_numpy(np.array([[b""]], dtype=object))
    inputs[4].set_data_from_numpy(np.array([[b"en"]], dtype=object))

    outputs = [
        httpclient.InferRequestedOutput("audio"),
        httpclient.InferRequestedOutput("sample_rate"),
    ]

    t0 = time.perf_counter()
    client.infer(model_name=MODEL, inputs=inputs, outputs=outputs)
    return time.perf_counter() - t0


def benchmark(n: int, concurrency: int, ref_audio_bytes: bytes):
    texts = [SHORT_TEXTS[i % len(SHORT_TEXTS)] for i in range(n)]
    print(f"\n=== Voice Clone Benchmark: {n} requests, concurrency={concurrency} ===")

    overall_t0 = time.perf_counter()
    latencies = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(infer, t, ref_audio_bytes) for t in texts]
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
    ref_audio_bytes = load_ref_audio()
    print(f"Loaded reference audio: {REF_AUDIO_PATH} ({len(ref_audio_bytes)} bytes)")

    # Warm-up
    print("Warming up...")
    infer("Hi there.", ref_audio_bytes)
    time.sleep(1)

    # Test different loads
    for n, concurrency in [(16, 16), (32, 16), (64, 16)]:
        benchmark(n, concurrency, ref_audio_bytes)
        time.sleep(2)


if __name__ == "__main__":
    main()
