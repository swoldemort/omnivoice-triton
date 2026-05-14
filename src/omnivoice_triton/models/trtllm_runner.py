"""OmniVoice runner with TensorRT-LLM acceleration for the Qwen3 backbone.

Loads the full PyTorch model, then swaps llm.forward with a TRT-LLM engine
that runs the 28 transformer layers. All other components (embeddings,
audio heads, generation loop, codec) stay in PyTorch.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import torch

from omnivoice_triton.models.base_runner import BaseRunner
from omnivoice_triton.models.trtllm.omnivoice_trtllm import OmniVoiceTRTLLM

logger = logging.getLogger(__name__)


class TRTLLMRunner(BaseRunner):
    """BaseRunner with TRT-LLM accelerated transformer backbone.

    Args:
        device: Target device (default: "cuda").
        model_id: HuggingFace model ID or local path.
        dtype: Model dtype string.
        engine_dir: Directory containing the TRT-LLM engine (rank0.engine + config.json).
            If None, falls back to pure PyTorch mode.
        build_engine: If True and engine_dir is missing, automatically build the engine.
        max_batch_size: Passed to engine builder if build_engine=True.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_id: str = "k2-fsa/OmniVoice",
        dtype: str = "fp16",
        engine_dir: str | None = None,
        build_engine: bool = False,
        max_batch_size: int = 16,
    ) -> None:
        super().__init__(device=device, model_id=model_id, dtype=dtype)
        self._engine_dir = engine_dir
        self._build_engine = build_engine
        self._max_batch_size = max_batch_size
        self._trtllm: OmniVoiceTRTLLM | None = None

    def load_model(self) -> None:
        """Load model and optionally build/attach TRT-LLM engine."""
        super().load_model()

        if self._engine_dir is None:
            logger.info("TRTLLMRunner: no engine_dir provided, using PyTorch mode.")
            return

        engine_path = Path(self._engine_dir)
        if not engine_path.exists() and self._build_engine:
            logger.info("TRTLLMRunner: engine not found, building ...")
            self._build_trt_engine()

        if not engine_path.exists():
            logger.warning("TRTLLMRunner: engine_dir %s does not exist, using PyTorch mode.", self._engine_dir)
            return

        # Resolve local model path for HF model IDs
        if os.path.isdir(self.model_id):
            local_model_dir = self.model_id
        else:
            from huggingface_hub import snapshot_download
            local_model_dir = snapshot_download(self.model_id)

        # Load TRT-LLM wrapper
        config_file = engine_path / "config.json"
        with open(config_file) as f:
            trtllm_config = json.load(f)

        self._trtllm = OmniVoiceTRTLLM(
            trtllm_config=trtllm_config,
            tllm_model_dir=str(engine_path),
            model_dir=local_model_dir,
            device=torch.device(self.device),
            debug=False,
        )

        # Monkey-patch llm.forward
        trtllm = self._trtllm

        def _trt_llm_forward(inputs_embeds=None, attention_mask=None, **kwargs):
            B, S, _ = inputs_embeds.shape
            if attention_mask is not None and attention_mask.dim() == 4:
                # attention_mask: [B, 1, S, S] bool -> count True per batch
                input_lengths = attention_mask[:, 0, 0, :].sum(dim=-1).to(torch.int32)
            else:
                input_lengths = torch.full((B,), S, dtype=torch.int32, device=inputs_embeds.device)
            hidden_states = trtllm.forward_trt(inputs_embeds, input_lengths)
            from transformers.modeling_outputs import BaseModelOutputWithPast
            return BaseModelOutputWithPast(last_hidden_state=hidden_states)

        self._model.llm.forward = _trt_llm_forward
        logger.info("TRTLLMRunner: llm.forward patched with TRT engine.")

    def _build_trt_engine(self) -> None:
        """Run the convert + build pipeline."""
        from omnivoice_triton.models.trtllm.build_engine import build_engine
        import argparse

        args = argparse.Namespace(
            model_dir=self.model_id,
            output_dir=self._engine_dir,
            checkpoint_dir="/tmp/trtllm_ckpt",
            dtype="float16" if self.dtype == torch.float16 else "bfloat16" if self.dtype == torch.bfloat16 else "float32",
            max_batch_size=self._max_batch_size,
        )
        build_engine(args)

    def unload_model(self) -> None:
        """Release TRT session and unload model."""
        if self._trtllm is not None:
            # Session holds engine buffer; let GC handle it
            self._trtllm = None
        super().unload_model()
