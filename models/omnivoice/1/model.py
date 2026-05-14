#!/usr/bin/env python3
"""Triton Python Backend for OmniVoice TTS model.

Supports three modes:
  1. Voice Cloning (ref_audio provided)
  2. Voice Design (instruct provided)
  3. Auto Voice (neither provided)

Runs in FP16 with num_steps=16. Batching is done by processing all items
across all requests in execute() as batched lists sent to model.generate().
"""

import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import triton_python_backend_utils as pb_utils

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.audio import load_audio_bytes

logger = logging.getLogger(__name__)


class TritonPythonModel:
    """Triton Python model for OmniVoice inference."""

    def initialize(self, args: Dict[str, Any]) -> None:
        """Load the OmniVoice model and configure generation settings."""
        model_config = json.loads(args["model_config"])
        parameters = model_config.get("parameters", {})

        self.num_steps = int(self._get_param(parameters, "num_steps", "16"))
        self.model_name_or_path = self._get_param(
            parameters, "model_name_or_path", "k2-fsa/OmniVoice"
        )
        self.load_asr = self._get_param(parameters, "load_asr", "true").lower() == "true"

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(
            "Loading OmniVoice model from %s on %s with FP16 ...",
            self.model_name_or_path,
            device,
        )

        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.model = OmniVoice.from_pretrained(
            self.model_name_or_path,
            device_map=str(device),
            dtype=dtype,
            load_asr=self.load_asr,
        )
        self.model.eval()

        self.gen_config = OmniVoiceGenerationConfig(
            num_step=self.num_steps,
            guidance_scale=2.0,
            t_shift=0.1,
            layer_penalty_factor=5.0,
            position_temperature=5.0,
            class_temperature=0.0,
            denoise=True,
            preprocess_prompt=True,
            postprocess_output=True,
            audio_chunk_duration=15.0,
            audio_chunk_threshold=30.0,
        )

        self.sampling_rate = self.model.sampling_rate
        logger.info(
            "OmniVoice loaded. Sampling rate: %d Hz | num_steps: %d | dtype: %s",
            self.sampling_rate,
            self.num_steps,
            dtype,
        )

    def _get_param(self, parameters: Dict[str, Any], key: str, default: str) -> str:
        p = parameters.get(key)
        if p is None:
            return default
        if isinstance(p, dict):
            return p.get("string_value", default)
        return str(p) if p is not None else default

    def execute(self, requests: List[Any]) -> List[Any]:
        """Process batched Triton requests.

        Triton dynamic batcher may send requests with batch_size > 1.
        We flatten all items from all requests, group by mode, and call
        model.generate() with batched lists for maximum throughput.
        """
        if not requests:
            return []

        groups: Dict[str, List[Dict[str, Any]]] = {
            "clone": [],
            "design": [],
            "auto": [],
        }

        for request in requests:
            texts = self._get_string_list(request, "text")
            ref_audio_bytes_list = self._get_bytes_list(request, "ref_audio")
            ref_texts = self._get_string_list(request, "ref_text")
            instructs = self._get_string_list(request, "instruct")
            languages = self._get_string_list(request, "language")
            durations = self._get_float_list(request, "duration")
            speeds = self._get_float_list(request, "speed")

            batch_size = len(texts)
            for i in range(batch_size):
                text = texts[i]
                ref_audio_bytes = ref_audio_bytes_list[i] if i < len(ref_audio_bytes_list) else None
                ref_text = ref_texts[i] if i < len(ref_texts) else None
                instruct = instructs[i] if i < len(instructs) else None
                language = languages[i] if i < len(languages) else None
                duration = durations[i] if i < len(durations) else None
                speed = speeds[i] if i < len(speeds) else None

                # Treat 0.0 as unspecified
                if duration is not None and abs(duration) < 1e-6:
                    duration = None
                if speed is not None and abs(speed) < 1e-6:
                    speed = None
                if speed is None:
                    speed = 1.0

                if not text:
                    err_msg = "Missing required input 'text'"
                    logger.error(err_msg)
                    groups["auto"].append({
                        "request": request,
                        "batch_idx": i,
                        "error": err_msg,
                    })
                    continue

                item = {
                    "request": request,
                    "batch_idx": i,
                    "text": text,
                    "ref_audio_bytes": ref_audio_bytes,
                    "ref_text": ref_text,
                    "instruct": instruct,
                    "language": language,
                    "duration": duration,
                    "speed": speed,
                }

                if ref_audio_bytes and len(ref_audio_bytes) > 0:
                    groups["clone"].append(item)
                elif instruct and len(instruct) > 0:
                    groups["design"].append(item)
                else:
                    groups["auto"].append(item)

        # Pre-allocate responses per request
        responses: Dict[int, List[Any]] = {}
        for req_idx, request in enumerate(requests):
            # Determine batch size for this request from text tensor
            text_tensor = pb_utils.get_input_tensor_by_name(request, "text")
            if text_tensor is not None:
                batch_size = text_tensor.as_numpy().shape[0]
            else:
                batch_size = 1
            responses[id(request)] = [None] * batch_size

        for mode, items in groups.items():
            if not items:
                continue
            if mode == "clone":
                self._process_clone_group(items, responses)
            elif mode == "design":
                self._process_design_group(items, responses)
            else:
                self._process_auto_group(items, responses)

        # Build final response list
        final_responses = []
        for request in requests:
            res_list = responses.get(id(request), [])
            # Fill any missing slots with errors
            for i, res in enumerate(res_list):
                if res is None:
                    res_list[i] = pb_utils.InferenceResponse(
                        error=pb_utils.TritonError("Internal error: response was not generated")
                    )
            # Triton expects one response per request (the request already contains the batch)
            if len(res_list) == 1:
                final_responses.append(res_list[0])
            else:
                # If Triton sent a batched request, we need to return a single response
                # with batched outputs. However, OmniVoice produces variable-length audio.
                # Triton does not support ragged batched outputs well for variable-length audio.
                # As a workaround, we return the first response and log a warning.
                # In practice with TTS, clients usually send batch_size=1 requests.
                logger.warning(
                    "Batched request with %d items returned only first response due to variable audio lengths",
                    len(res_list),
                )
                final_responses.append(res_list[0])

        return final_responses

    def _process_clone_group(
        self, items: List[Dict[str, Any]], responses: Dict[int, List[Any]]
    ) -> None:
        texts = []
        ref_audios = []
        ref_texts = []
        languages = []
        durations = []
        speeds = []
        item_metadata = []

        for item in items:
            if "error" in item:
                req_id = id(item["request"])
                idx = item["batch_idx"]
                responses[req_id][idx] = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(item["error"])
                )
                continue
            texts.append(item["text"])
            wav_np = load_audio_bytes(item["ref_audio_bytes"], self.sampling_rate)
            wav_t = torch.from_numpy(wav_np)
            ref_audios.append((wav_t, self.sampling_rate))
            ref_texts.append(item["ref_text"] if item["ref_text"] else None)
            languages.append(item["language"] if item["language"] else None)
            durations.append(item["duration"])
            speeds.append(item["speed"])
            item_metadata.append(item)

        if not texts:
            return

        try:
            audios = self.model.generate(
                text=texts,
                ref_audio=ref_audios,
                ref_text=ref_texts,
                language=languages,
                duration=durations,
                speed=speeds,
                generation_config=self.gen_config,
            )
            self._fill_responses(item_metadata, audios, responses)
        except Exception as e:
            logger.exception("Voice clone batch failed")
            self._fill_errors(item_metadata, str(e), responses)

    def _process_design_group(
        self, items: List[Dict[str, Any]], responses: Dict[int, List[Any]]
    ) -> None:
        texts = []
        instructs = []
        languages = []
        durations = []
        speeds = []
        item_metadata = []

        for item in items:
            if "error" in item:
                req_id = id(item["request"])
                idx = item["batch_idx"]
                responses[req_id][idx] = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(item["error"])
                )
                continue
            texts.append(item["text"])
            instructs.append(item["instruct"])
            languages.append(item["language"] if item["language"] else None)
            durations.append(item["duration"])
            speeds.append(item["speed"])
            item_metadata.append(item)

        if not texts:
            return

        try:
            audios = self.model.generate(
                text=texts,
                instruct=instructs,
                language=languages,
                duration=durations,
                speed=speeds,
                generation_config=self.gen_config,
            )
            self._fill_responses(item_metadata, audios, responses)
        except Exception as e:
            logger.exception("Voice design batch failed")
            self._fill_errors(item_metadata, str(e), responses)

    def _process_auto_group(
        self, items: List[Dict[str, Any]], responses: Dict[int, List[Any]]
    ) -> None:
        texts = []
        languages = []
        durations = []
        speeds = []
        item_metadata = []

        for item in items:
            if "error" in item:
                req_id = id(item["request"])
                idx = item["batch_idx"]
                responses[req_id][idx] = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(item["error"])
                )
                continue
            texts.append(item["text"])
            languages.append(item["language"] if item["language"] else None)
            durations.append(item["duration"])
            speeds.append(item["speed"])
            item_metadata.append(item)

        if not texts:
            return

        try:
            audios = self.model.generate(
                text=texts,
                language=languages,
                duration=durations,
                speed=speeds,
                generation_config=self.gen_config,
            )
            self._fill_responses(item_metadata, audios, responses)
        except Exception as e:
            logger.exception("Auto voice batch failed")
            self._fill_errors(item_metadata, str(e), responses)

    def _fill_responses(
        self,
        items: List[Dict[str, Any]],
        audios: List[np.ndarray],
        responses: Dict[int, List[Any]],
    ) -> None:
        valid_items = [it for it in items if "error" not in it]
        if len(valid_items) != len(audios):
            logger.error(
                "Mismatch: %d valid items vs %d audio outputs",
                len(valid_items),
                len(audios),
            )
        for item, audio in zip(valid_items, audios):
            req_id = id(item["request"])
            batch_idx = item["batch_idx"]
            try:
                audio_tensor = pb_utils.Tensor("audio", audio.astype(np.float32))
                sr_tensor = pb_utils.Tensor(
                    "sample_rate",
                    np.array([self.sampling_rate], dtype=np.int32),
                )
                responses[req_id][batch_idx] = pb_utils.InferenceResponse(
                    output_tensors=[audio_tensor, sr_tensor]
                )
            except Exception as e:
                logger.exception("Failed to build response")
                responses[req_id][batch_idx] = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(str(e))
                )

    def _fill_errors(
        self,
        items: List[Dict[str, Any]],
        error_msg: str,
        responses: Dict[int, List[Any]],
    ) -> None:
        for item in items:
            if "error" not in item:
                req_id = id(item["request"])
                batch_idx = item["batch_idx"]
                responses[req_id][batch_idx] = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(error_msg)
                )

    def _get_string_list(self, request: Any, name: str) -> List[Optional[str]]:
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            return []
        arr = tensor.as_numpy()
        if arr.size == 0:
            return []
        result = []
        for val in arr.flat:
            if val is None or (isinstance(val, bytes) and len(val) == 0):
                result.append(None)
            elif isinstance(val, bytes):
                result.append(val.decode("utf-8", errors="replace"))
            elif isinstance(val, str):
                result.append(val)
            else:
                result.append(str(val))
        return result

    def _get_bytes_list(self, request: Any, name: str) -> List[Optional[bytes]]:
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            return []
        arr = tensor.as_numpy()
        if arr.size == 0:
            return []
        result = []
        for val in arr.flat:
            if val is None or (isinstance(val, bytes) and len(val) == 0):
                result.append(None)
            elif isinstance(val, bytes):
                result.append(val)
            elif isinstance(val, str):
                result.append(val.encode("utf-8"))
            else:
                result.append(bytes(val))
        return result

    def _get_float_list(self, request: Any, name: str) -> List[Optional[float]]:
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            return []
        arr = tensor.as_numpy()
        if arr.size == 0:
            return []
        result = []
        for val in arr.flat:
            if val is None or (isinstance(val, float) and val != val):
                result.append(None)
            else:
                result.append(float(val))
        return result
