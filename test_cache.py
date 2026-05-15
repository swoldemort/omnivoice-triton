#!/usr/bin/env python3
"""Test cache: send same voice design request twice, measure time difference."""

import time
import numpy as np
import tritonclient.http as httpclient

URL = "localhost:8000"
MODEL = "omnivoice"

TEXT = "Hello world, this is a cache test."
INSTRUCT = "female, british accent"
LANGUAGE = "en"


def send_request():
    client = httpclient.InferenceServerClient(URL)
    inputs = [
        httpclient.InferInput("text", [1, 1], "BYTES"),
        httpclient.InferInput("ref_audio", [1, 1], "BYTES"),
        httpclient.InferInput("ref_text", [1, 1], "BYTES"),
        httpclient.InferInput("instruct", [1, 1], "BYTES"),
        httpclient.InferInput("language", [1, 1], "BYTES"),
    ]
    inputs[0].set_data_from_numpy(np.array([[TEXT.encode()]], dtype=object))
    inputs[1].set_data_from_numpy(np.array([[b""]], dtype=object))
    inputs[2].set_data_from_numpy(np.array([[b""]], dtype=object))
    inputs[3].set_data_from_numpy(np.array([[INSTRUCT.encode()]], dtype=object))
    inputs[4].set_data_from_numpy(np.array([[LANGUAGE.encode()]], dtype=object))

    outputs = [
        httpclient.InferRequestedOutput("audio"),
        httpclient.InferRequestedOutput("sample_rate"),
    ]

    t0 = time.perf_counter()
    result = client.infer(model_name=MODEL, inputs=inputs, outputs=outputs)
    latency = time.perf_counter() - t0
    audio = result.as_numpy("audio")
    sr = result.as_numpy("sample_rate").item()
    return latency, len(audio), sr


if __name__ == "__main__":
    print("=== Cache Test ===")
    print(f"Text: {TEXT}")
    print(f"Instruct: {INSTRUCT}")
    print()

    # First request — cache miss (generate)
    print("Request 1 (cache miss):")
    lat1, len1, sr1 = send_request()
    print(f"  Latency: {lat1:.3f}s | Audio samples: {len1} | SR: {sr1}")

    # Second request — cache hit (serve from disk)
    print("\nRequest 2 (cache hit):")
    lat2, len2, sr2 = send_request()
    print(f"  Latency: {lat2:.3f}s | Audio samples: {len2} | SR: {sr2}")

    print(f"\nSpeedup: {lat1/lat2:.1f}x faster")
    print(f"Time saved: {lat1 - lat2:.3f}s")

    # Check cache files
    import os
    cache_dir = "/mnt/hdd/omnivoice_cache/design"
    if os.path.exists(cache_dir):
        files = []
        for root, _, filenames in os.walk(cache_dir):
            for f in filenames:
                files.append(os.path.join(root, f))
        print(f"\nCache files in {cache_dir}: {len(files)}")
        for f in files:
            size = os.path.getsize(f)
            print(f"  {os.path.basename(f)}: {size/1024:.1f} KB")
