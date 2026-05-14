"""TensorRT-LLM integration for OmniVoice Qwen3 backbone."""

from omnivoice_triton.models.trtllm.build_engine import build_engine, main
from omnivoice_triton.models.trtllm.convert_checkpoint import main_with_args
from omnivoice_triton.models.trtllm.omnivoice_trtllm import OmniVoiceTRTLLM

__all__ = ["OmniVoiceTRTLLM", "build_engine", "main", "main_with_args"]
