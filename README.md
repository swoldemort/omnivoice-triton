# OmniVoice Triton Inference Server

Triton Inference Server deployment for [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) zero-shot TTS model.

## Features

- **Backend**: Triton Python Backend with PyTorch
- **Precision**: FP16 on GPU
- **Inference steps**: 16 (configurable)
- **Modes supported**:
  - Auto Voice (random voice)
  - Voice Design (text instruction, e.g. "female, British accent")
  - Voice Cloning (from reference audio WAV)
- **Batching**: Manual batching in Python backend per mode group
- **Endpoints**: Standard Triton HTTP (8000), gRPC (8001), Metrics (8002)

## Quick Start

### 1. Build and run

```bash
docker compose up --build
```

The first startup will download model weights from HuggingFace (~3.5 GB) into the `huggingface_cache` Docker volume.

### 2. Wait for readiness

```bash
curl http://localhost:8000/v2/health/ready
```

### 3. Run inference

Install client dependencies (on host):
```bash
pip install tritonclient[all] soundfile numpy
```

**Auto voice:**
```bash
python client_example.py --mode auto --text "Hello, this is a test." --output auto.wav
```

**Voice design:**
```bash
python client_example.py --mode design --text "Hello world." --instruct "female, British accent" --output design.wav
```

**Voice cloning:**
```bash
python client_example.py --mode clone --text "Hello world." --ref-audio ref.wav --ref-text "Reference transcript." --output clone.wav
```

### Optional parameters

- `--language` (default: `en`)
- `--duration` (fixed output length in seconds)
- `--speed` (speaking rate factor, >1 faster, <1 slower)

## Architecture

```
models/omnivoice/
├── config.pbtxt          # Triton model config
└── 1/
    ├── model.py          # Python backend (FP16, batching, 3 modes)
    └── requirements.txt  # Python dependencies
```

Requests are grouped by mode (clone/design/auto) inside `execute()` and sent to `OmniVoice.generate()` as batched lists. This maximizes throughput while keeping the implementation simple and robust.

## Hardware Requirements

- NVIDIA GPU with 8 GB+ VRAM (FP16)
- CUDA compute capability 7.5+
- ~10 GB disk space for container + model cache

## Notes

- `max_batch_size: 0` is used because STRING inputs have variable lengths; Triton's native dynamic batcher cannot group them. Batching is done manually in Python.
- Reference audio must be a valid WAV file sent as raw bytes in the `ref_audio` STRING tensor.
