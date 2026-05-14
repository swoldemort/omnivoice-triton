#!/usr/bin/env python3
"""Generate Romanian TTS samples with voice design."""

import numpy as np
import soundfile as sf
import tritonclient.http as httpclient

URL = "localhost:8000"
MODEL = "omnivoice"

SAMPLES = [
    ("Bună ziua, mă bucur să vă cunosc.", "female, british accent"),
    ("Bună dimineața, ce mai faci?", "male, american accent"),
    ("Astăzi este o zi frumoasă.", "female, high pitch"),
    ("Mergem la plimbare în parc?", "male, low pitch"),
    ("Îmi place mult să ascult muzică.", "female, whisper"),
    ("Sper că ai o zi minunată!", "male, british accent"),
]


def infer(text: str, instruct: str, output: str):
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
    inputs[4].set_data_from_numpy(np.array([[b"ro"]], dtype=object))

    outputs = [
        httpclient.InferRequestedOutput("audio"),
        httpclient.InferRequestedOutput("sample_rate"),
    ]

    response = client.infer(model_name=MODEL, inputs=inputs, outputs=outputs)
    audio = response.as_numpy("audio")
    sr = int(response.as_numpy("sample_rate").item())
    sf.write(output, audio, sr)
    print(f"Saved: {output} ({len(audio)/sr:.2f}s @ {sr}Hz) — '{text}' [{instruct}]")


def main():
    for i, (text, instruct) in enumerate(SAMPLES):
        infer(text, instruct, f"romanian_{i:02d}.wav")


if __name__ == "__main__":
    main()
