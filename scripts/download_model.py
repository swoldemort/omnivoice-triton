#!/usr/bin/env python3
"""Pre-download OmniVoice model weights to avoid cold-start."""

from huggingface_hub import snapshot_download


def main():
    print("Downloading k2-fsa/OmniVoice ...")
    snapshot_download("k2-fsa/OmniVoice")
    print("Done.")

    # Also download the audio tokenizer if not bundled
    print("Downloading eustlb/higgs-audio-v2-tokenizer ...")
    snapshot_download("eustlb/higgs-audio-v2-tokenizer")
    print("Done.")

    # Pre-download ASR model for auto-transcription
    print("Downloading openai/whisper-large-v3-turbo ...")
    snapshot_download("openai/whisper-large-v3-turbo")
    print("Done.")


if __name__ == "__main__":
    main()
