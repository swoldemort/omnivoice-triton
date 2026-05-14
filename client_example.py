#!/usr/bin/env python3
"""Example Triton client for OmniVoice inference.

Demonstrates all three modes:
  1. Auto Voice (text only)
  2. Voice Design (text + instruct)
  3. Voice Cloning (text + ref_audio + optional ref_text)
"""

import argparse
import os

import numpy as np
import soundfile as sf
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException


def auto_voice_request(
    client: httpclient.InferenceServerClient,
    text: str,
    output_path: str,
    language: str = "en",
    duration: float = 0.0,
    speed: float = 0.0,
):
    """Generate speech in auto mode (random voice)."""
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
    inputs[4].set_data_from_numpy(np.array([[language.encode("utf-8")]], dtype=object))

    if duration and duration > 0:
        dur = httpclient.InferInput("duration", [1, 1], "FP32")
        dur.set_data_from_numpy(np.array([[float(duration)]], dtype=np.float32))
        inputs.append(dur)
    if speed and speed > 0:
        spd = httpclient.InferInput("speed", [1, 1], "FP32")
        spd.set_data_from_numpy(np.array([[float(speed)]], dtype=np.float32))
        inputs.append(spd)

    outputs = [
        httpclient.InferRequestedOutput("audio"),
        httpclient.InferRequestedOutput("sample_rate"),
    ]

    response = client.infer(model_name="omnivoice", inputs=inputs, outputs=outputs)
    audio = response.as_numpy("audio")
    sr = int(response.as_numpy("sample_rate").item())

    sf.write(output_path, audio, sr)
    print(f"[Auto] Saved to {output_path} ({len(audio)/sr:.2f}s @ {sr}Hz)")


def voice_design_request(
    client: httpclient.InferenceServerClient,
    text: str,
    instruct: str,
    output_path: str,
    language: str = "en",
    duration: float = 0.0,
    speed: float = 0.0,
):
    """Generate speech with voice design (instruct-based)."""
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
    inputs[4].set_data_from_numpy(np.array([[language.encode("utf-8")]], dtype=object))

    if duration and duration > 0:
        dur = httpclient.InferInput("duration", [1], "FP32")
        dur.set_data_from_numpy(np.array([float(duration)], dtype=np.float32))
        inputs.append(dur)
    if speed and speed > 0:
        spd = httpclient.InferInput("speed", [1], "FP32")
        spd.set_data_from_numpy(np.array([float(speed)], dtype=np.float32))
        inputs.append(spd)

    outputs = [
        httpclient.InferRequestedOutput("audio"),
        httpclient.InferRequestedOutput("sample_rate"),
    ]

    response = client.infer(model_name="omnivoice", inputs=inputs, outputs=outputs)
    audio = response.as_numpy("audio")
    sr = int(response.as_numpy("sample_rate").item())

    sf.write(output_path, audio, sr)
    print(f"[Design] Saved to {output_path} ({len(audio)/sr:.2f}s @ {sr}Hz)")


def voice_clone_request(
    client: httpclient.InferenceServerClient,
    text: str,
    ref_audio_path: str,
    ref_text: str,
    output_path: str,
    language: str = "en",
    duration: float = 0.0,
    speed: float = 0.0,
):
    """Generate speech by cloning a reference voice."""
    with open(ref_audio_path, "rb") as f:
        ref_audio_bytes = f.read()

    inputs = [
        httpclient.InferInput("text", [1, 1], "BYTES"),
        httpclient.InferInput("ref_audio", [1, 1], "BYTES"),
        httpclient.InferInput("ref_text", [1, 1], "BYTES"),
        httpclient.InferInput("instruct", [1, 1], "BYTES"),
        httpclient.InferInput("language", [1, 1], "BYTES"),
    ]

    inputs[0].set_data_from_numpy(np.array([[text.encode("utf-8")]], dtype=object))
    inputs[1].set_data_from_numpy(np.array([[ref_audio_bytes]], dtype=object))
    inputs[2].set_data_from_numpy(
        np.array([[ref_text.encode("utf-8") if ref_text else b""]], dtype=object)
    )
    inputs[3].set_data_from_numpy(np.array([[b""]], dtype=object))
    inputs[4].set_data_from_numpy(np.array([[language.encode("utf-8")]], dtype=object))

    if duration and duration > 0:
        dur = httpclient.InferInput("duration", [1], "FP32")
        dur.set_data_from_numpy(np.array([float(duration)], dtype=np.float32))
        inputs.append(dur)
    if speed and speed > 0:
        spd = httpclient.InferInput("speed", [1], "FP32")
        spd.set_data_from_numpy(np.array([float(speed)], dtype=np.float32))
        inputs.append(spd)

    outputs = [
        httpclient.InferRequestedOutput("audio"),
        httpclient.InferRequestedOutput("sample_rate"),
    ]

    response = client.infer(model_name="omnivoice", inputs=inputs, outputs=outputs)
    audio = response.as_numpy("audio")
    sr = int(response.as_numpy("sample_rate").item())

    sf.write(output_path, audio, sr)
    print(f"[Clone] Saved to {output_path} ({len(audio)/sr:.2f}s @ {sr}Hz)")


def main():
    parser = argparse.ArgumentParser(description="OmniVoice Triton client example")
    parser.add_argument(
        "--url",
        default="localhost:8000",
        help="Triton HTTP endpoint (default: localhost:8000)",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "design", "clone"],
        required=True,
        help="Generation mode",
    )
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument(
        "--instruct", default="", help="Voice design instruction (e.g. 'female, British accent')"
    )
    parser.add_argument("--ref-audio", default="", help="Path to reference audio WAV file")
    parser.add_argument("--ref-text", default="", help="Transcript of reference audio")
    parser.add_argument("--output", default="output.wav", help="Output WAV path")
    parser.add_argument("--language", default="en", help="Language code/name (default: en)")
    parser.add_argument("--duration", type=float, default=0.0, help="Fixed duration in seconds")
    parser.add_argument("--speed", type=float, default=0.0, help="Speaking speed factor")
    args = parser.parse_args()

    client = httpclient.InferenceServerClient(url=args.url)

    # Wait until model is ready
    if not client.is_model_ready("omnivoice"):
        raise RuntimeError("Model 'omnivoice' is not ready on the server.")

    kwargs = {
        "language": args.language,
        "duration": args.duration,
        "speed": args.speed,
    }

    if args.mode == "auto":
        auto_voice_request(client, args.text, args.output, **kwargs)
    elif args.mode == "design":
        if not args.instruct:
            raise ValueError("--instruct is required for voice design mode")
        voice_design_request(client, args.text, args.instruct, args.output, **kwargs)
    elif args.mode == "clone":
        if not args.ref_audio or not os.path.isfile(args.ref_audio):
            raise ValueError("--ref-audio must point to an existing WAV file")
        voice_clone_request(client, args.text, args.ref_audio, args.ref_text, args.output, **kwargs)


if __name__ == "__main__":
    main()
