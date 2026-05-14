#!/usr/bin/env python3
"""Test dynamic batching by sending N concurrent requests."""

import concurrent.futures
import time

import numpy as np
import soundfile as sf
import tritonclient.http as httpclient

URL = "localhost:8000"
MODEL = "omnivoice"

TEXTS = [
    "Hello, this is request one.",
    "The quick brown fox jumps over the lazy dog.",
    "Dynamic batching test number three.",
    "Four score and seven years ago.",
    "To be or not to be, that is the question.",
    "All human beings are born free and equal.",
    "Testing batch size sixteen configuration.",
    "The rain in Spain stays mainly in the plain.",
]


def infer(idx: int, text: str) -> float:
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
    inputs[3].set_data_from_numpy(np.array([[b""]], dtype=object))
    inputs[4].set_data_from_numpy(np.array([[b"en"]], dtype=object))

    outputs = [
        httpclient.InferRequestedOutput("audio"),
        httpclient.InferRequestedOutput("sample_rate"),
    ]

    t0 = time.perf_counter()
    response = client.infer(model_name=MODEL, inputs=inputs, outputs=outputs)
    elapsed = time.perf_counter() - t0

    audio = response.as_numpy("audio")
    sr = int(response.as_numpy("sample_rate").item())
    sf.write(f"batch_out_{idx:02d}.wav", audio, sr)
    return elapsed


def main():
    n = len(TEXTS)
    print(f"Sending {n} concurrent requests to Triton...")

    overall_t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
        futures = {executor.submit(infer, i, TEXTS[i]): i for i in range(n)}
        results = {}
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Request {idx} failed: {e}")
                results[idx] = None
    overall_elapsed = time.perf_counter() - overall_t0

    valid = [v for v in results.values() if v is not None]
    print(f"\nTotal wall-clock time for {n} concurrent requests: {overall_elapsed:.2f}s")
    print(f"Sum of individual latencies: {sum(valid):.2f}s")
    print(f"Average latency: {sum(valid)/len(valid):.2f}s")
    print(f"Min latency: {min(valid):.2f}s")
    print(f"Max latency: {max(valid):.2f}s")
    speedup = sum(valid) / overall_elapsed if overall_elapsed > 0 else 0
    print(f"Effective speedup from batching: {speedup:.2f}x")


if __name__ == "__main__":
    main()
