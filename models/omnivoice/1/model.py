#!/usr/bin/env python3
"""Triton Python Backend for OmniVoice TTS model — Ultra Cache Edition.

Supports three modes:
  1. Voice Cloning (ref_audio provided)
  2. Voice Design (instruct provided)
  3. Auto Voice (neither provided)

Cache architecture (Ultra Cache):
  L1: In-memory LRU (OrderedDict) — audio + voice prompts
  L2: Redis (metadata, stats, TTL, voice vectors) at 192.168.1.190:6379
  L3: MinIO S3 (blobs: .npz audio, .pt voice prompts) at 192.168.1.190:9000

Advanced features:
  • Voice Prompt Caching — decouples voice identity extraction from text
  • ECAPA-TDNN Speaker Embeddings — semantic voice deduplication via
    cosine-similarity search over 192-D vectors stored in Redis
  • TeaCache — intra-inference activation caching for Qwen3DecoderLayer
  • TRITONCACHE native — C++ response cache enabled in config.pbtxt
"""

import hashlib
import io
import json
import logging
import os
import threading
import time
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import triton_python_backend_utils as pb_utils

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.models.omnivoice import VoiceClonePrompt
from omnivoice.utils.audio import load_audio_bytes

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
MEM_CACHE_MAX_AUDIO = int(os.environ.get("MEM_CACHE_MAX_AUDIO", "10000"))
MEM_CACHE_MAX_PROMPT = int(os.environ.get("MEM_CACHE_MAX_PROMPT", "5000"))

REDIS_HOST = os.environ.get("REDIS_HOST", "192.168.1.190")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "") or None
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
REDIS_SOCKET_TIMEOUT = float(os.environ.get("REDIS_SOCKET_TIMEOUT", "2.0"))

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://192.168.1.190:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "")
CACHE_BUCKET = os.environ.get("CACHE_BUCKET", "omnivoice-cache")

AUDIO_TTL_SECONDS = int(os.environ.get("CACHE_AUDIO_TTL", "604800"))      # 7 days
PROMPT_TTL_SECONDS = int(os.environ.get("CACHE_PROMPT_TTL", "2592000"))   # 30 days
VOICE_VECTOR_TTL = int(os.environ.get("CACHE_VECTOR_TTL", "2592000"))     # 30 days

# ECAPA-TDNN config
ECAPA_SIMILARITY_THRESHOLD = float(os.environ.get("ECAPA_THRESHOLD", "0.72"))
ECAPA_SR = 16000  # ECAPA-TDNN expects 16 kHz

# TeaCache config
TEACACHE_ENABLE = os.environ.get("TEACACHE_ENABLE", "false").lower() == "true"
TEACACHE_THRESHOLD = float(os.environ.get("TEACACHE_THRESHOLD", "0.015"))


def _hash_str(s: str) -> str:
    return hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest()


def _hash_bytes(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=16).hexdigest()


# ---------------------------------------------------------------------------
# TeaCache — Intra-Inference Activation Caching for Qwen3
# ---------------------------------------------------------------------------
class TeaCacheManager:
    """Caches decoder-layer outputs when layer inputs change minimally.

    Works by storing (input, output) pairs per layer instance.  On the next
    forward call the relative L1 distance between the new input and the cached
    input is computed.  If it falls below *threshold* the cached output is
    returned verbatim, skipping the expensive self-attention + MLP.
    """

    def __init__(self, threshold: float = 0.015):
        self.threshold = threshold
        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._enabled = False
        self._lock = threading.Lock()

    def enable(self) -> None:
        with self._lock:
            self._cache.clear()
            self._enabled = True
        logger.info("TeaCache ENABLED (threshold=%.4f)", self.threshold)

    def disable(self) -> None:
        with self._lock:
            self._enabled = False
            self._cache.clear()
        logger.info("TeaCache DISABLED")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def get(self, layer_id: int, current_input: torch.Tensor) -> Optional[torch.Tensor]:
        if not self._enabled:
            return None
        with self._lock:
            entry = self._cache.get(layer_id)
        if entry is None:
            return None
        prev_input, prev_output = entry
        # Relative mean L1 distance
        diff = torch.abs(current_input - prev_input).mean()
        base = torch.abs(current_input).mean() + 1e-8
        if (diff / base).item() < self.threshold:
            return prev_output
        return None

    def set(self, layer_id: int, current_input: torch.Tensor, output: torch.Tensor) -> None:
        if not self._enabled:
            return
        with self._lock:
            self._cache[layer_id] = (current_input.detach().clone(), output.detach().clone())


_tea_cache = TeaCacheManager(threshold=TEACACHE_THRESHOLD)


def _install_teacache_patch() -> None:
    """Monkey-patch Qwen3DecoderLayer.forward globally."""
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
    except Exception as exc:
        logger.warning("TeaCache: cannot import Qwen3DecoderLayer: %s", exc)
        return

    if getattr(Qwen3DecoderLayer, "_teacache_patched", False):
        return  # already patched

    _orig_forward = Qwen3DecoderLayer.forward

    def _patched_forward(self, hidden_states, *args, **kwargs):
        layer_id = id(self)
        cached = _tea_cache.get(layer_id, hidden_states)
        if cached is not None:
            return cached
        output = _orig_forward(self, hidden_states, *args, **kwargs)
        _tea_cache.set(layer_id, hidden_states, output)
        return output

    Qwen3DecoderLayer.forward = _patched_forward
    Qwen3DecoderLayer._teacache_patched = True  # type: ignore[attr-defined]
    logger.info("TeaCache patch installed on Qwen3DecoderLayer")


# ---------------------------------------------------------------------------
# Speaker Encoder — ECAPA-TDNN via SpeechBrain
# ---------------------------------------------------------------------------
class SpeakerEncoder:
    """Lightweight wrapper around SpeechBrain ECAPA-TDNN.

    Produces 192-D speaker embeddings invariant to format (WAV/MP3) and
    robust to light trimming / noise variations.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._classifier = None
        self._ready = False
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            try:
                from speechbrain.pretrained import EncoderClassifier
                self._classifier = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    run_opts={"device": str(self.device)},
                    savedir="/tmp/speechbrain_ecapa",
                )
                self._classifier.eval()
                self._ready = True
                logger.info("ECAPA-TDNN speaker encoder loaded on %s", self.device)
            except Exception as exc:
                logger.error("Failed to load ECAPA-TDNN: %s", exc)
                self._ready = False

    def encode(self, waveform: np.ndarray) -> Optional[np.ndarray]:
        """Encode a waveform to a 192-D unit vector.

        Args:
            waveform: numpy float32 array of shape (C, T) or (T,).

        Returns:
            L2-normalised 192-D embedding vector (np.ndarray) or None.
        """
        self._load()
        if not self._ready or self._classifier is None:
            return None

        try:
            if waveform.ndim == 1:
                waveform = waveform[np.newaxis, :]  # (1, T)
            if waveform.shape[0] > 1:
                waveform = np.mean(waveform, axis=0, keepdims=True)
            wav_t = torch.from_numpy(waveform.astype(np.float32)).to(self.device)
            # SpeechBrain expects (batch, samples)
            with torch.no_grad():
                emb = self._classifier.encode_batch(wav_t).squeeze(0)
                emb = torch.nn.functional.normalize(emb, dim=-1)
            return emb.cpu().numpy().astype(np.float32)
        except Exception as exc:
            logger.warning("ECAPA encoding failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# TritonPythonModel
# ---------------------------------------------------------------------------
class TritonPythonModel:
    """OmniVoice Triton backend with L1/L2/L3 multi-tier cache + ECAPA + TeaCache."""

    def initialize(self, args: Dict[str, Any]) -> None:
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
            "OmniVoice loaded. SR=%d Hz | num_steps=%d | dtype=%s | device=%s",
            self.sampling_rate,
            self.num_steps,
            dtype,
            device,
        )

        # ECAPA-TDNN speaker encoder
        self._speaker_encoder: Optional[SpeakerEncoder] = None
        try:
            self._speaker_encoder = SpeakerEncoder(device)
            logger.info("SpeakerEncoder initialised (lazy-load).")
        except Exception as exc:
            logger.warning("SpeakerEncoder init failed: %s", exc)

        # L3: MinIO S3 client
        self._s3 = None
        self._init_s3()

        # L2: Redis client
        self._redis = None
        self._init_redis()

        # L1: In-memory LRU caches
        self._audio_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._audio_cache_lock = threading.RLock()
        self._audio_cache_max = MEM_CACHE_MAX_AUDIO

        self._prompt_cache: OrderedDict[str, VoiceClonePrompt] = OrderedDict()
        self._prompt_cache_lock = threading.RLock()
        self._prompt_cache_max = MEM_CACHE_MAX_PROMPT

        # TeaCache
        if TEACACHE_ENABLE:
            _install_teacache_patch()
            _tea_cache.enable()
        else:
            logger.info("TeaCache is disabled (set TEACACHE_ENABLE=true to activate).")

        logger.info(
            "Cache tiers ready. L1(audio=%d,prompt=%d) | L2 Redis=%s:%d | "
            "L3 MinIO=%s | ECAPA threshold=%.2f",
            self._audio_cache_max,
            self._prompt_cache_max,
            REDIS_HOST,
            REDIS_PORT,
            MINIO_ENDPOINT,
            ECAPA_SIMILARITY_THRESHOLD,
        )

    # ------------------------------------------------------------------
    # External service initialisation
    # ------------------------------------------------------------------
    def _init_s3(self) -> None:
        try:
            import boto3
            from botocore.config import Config

            endpoint = MINIO_ENDPOINT.rstrip("/")
            self._s3 = boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=MINIO_ACCESS_KEY or None,
                aws_secret_access_key=MINIO_SECRET_KEY or None,
                config=Config(signature_version="s3v4"),
                region_name="us-east-1",
            )
            try:
                self._s3.head_bucket(Bucket=CACHE_BUCKET)
            except Exception:
                self._s3.create_bucket(Bucket=CACHE_BUCKET)
                logger.info("Created MinIO bucket: %s", CACHE_BUCKET)
            logger.info("MinIO S3 client connected: %s", endpoint)
        except Exception as e:
            logger.error("MinIO S3 init failed: %s", e)
            self._s3 = None

    def _init_redis(self) -> None:
        try:
            import redis as redis_lib

            self._redis = redis_lib.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD,
                db=REDIS_DB,
                decode_responses=True,
                socket_connect_timeout=REDIS_SOCKET_TIMEOUT,
                socket_timeout=REDIS_SOCKET_TIMEOUT,
                health_check_interval=30,
            )
            self._redis.ping()
            logger.info("Redis client connected: %s:%d", REDIS_HOST, REDIS_PORT)
        except Exception as e:
            logger.error("Redis init failed: %s", e)
            self._redis = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_param(self, parameters: Dict[str, Any], key: str, default: str) -> str:
        p = parameters.get(key)
        if p is None:
            return default
        if isinstance(p, dict):
            return p.get("string_value", default)
        return str(p) if p is not None else default

    def _normalize_instruct(self, instruct: str) -> str:
        if not instruct:
            return ""
        parts = [p.strip().lower() for p in instruct.split(",")]
        parts = sorted(parts)
        return ",".join(parts)

    # ------------------------------------------------------------------
    # ECAPA-TDNN semantic voice deduplication
    # ------------------------------------------------------------------
    def _get_voice_embedding(self, ref_audio_bytes: bytes) -> Optional[np.ndarray]:
        if self._speaker_encoder is None:
            return None
        try:
            wav = load_audio_bytes(ref_audio_bytes, ECAPA_SR)
            return self._speaker_encoder.encode(wav)
        except Exception as exc:
            logger.warning("Voice embedding extraction failed: %s", exc)
            return None

    def _find_similar_voice(self, embedding: np.ndarray) -> Optional[str]:
        """Brute-force cosine-similarity search over all vectors in Redis.

        Returns the voice_id of the best match if similarity exceeds the
        configured threshold, otherwise None.
        """
        if self._redis is None or embedding is None:
            return None
        try:
            embedding = embedding.flatten()
            keys = []
            vectors = []
            for key in self._redis.scan_iter(match="ov:voice:vector:*", count=500):
                vec_json = self._redis.get(key)
                if not vec_json:
                    continue
                vec = np.array(json.loads(vec_json), dtype=np.float32)
                keys.append(key)
                vectors.append(vec)
            if not vectors:
                return None
            mat = np.stack(vectors)  # (N, D)
            # Cosine similarity
            sim = (mat @ embedding) / (
                np.linalg.norm(mat, axis=1) * np.linalg.norm(embedding) + 1e-8
            )
            best_idx = int(np.argmax(sim))
            best_sim = float(sim[best_idx])
            logger.debug("Best ECAPA similarity: %.4f", best_sim)
            if best_sim > ECAPA_SIMILARITY_THRESHOLD:
                voice_id = keys[best_idx].rsplit(":", 1)[-1]
                logger.info("ECAPA semantic voice HIT: %s (sim=%.4f)", voice_id[:8], best_sim)
                return voice_id
        except Exception as exc:
            logger.warning("ECAPA vector search failed: %s", exc)
        return None

    def _save_voice_embedding(self, voice_id: str, embedding: np.ndarray, ref_text: Optional[str] = None) -> None:
        if self._redis is None:
            return
        try:
            vec_json = json.dumps(embedding.flatten().tolist())
            pipe = self._redis.pipeline()
            pipe.setex(f"ov:voice:vector:{voice_id}", VOICE_VECTOR_TTL, vec_json)
            pipe.hset(f"ov:voice:meta:{voice_id}", mapping={
                "ref_text": ref_text or "",
                "created": str(int(time.time())),
            })
            pipe.expire(f"ov:voice:meta:{voice_id}", VOICE_VECTOR_TTL)
            pipe.execute()
            logger.debug("Voice embedding saved: %s", voice_id[:8])
        except Exception as exc:
            logger.warning("Failed to save voice embedding: %s", exc)

    # ------------------------------------------------------------------
    # Key generators
    # ------------------------------------------------------------------
    def _voice_prompt_key(
        self,
        ref_audio_bytes: bytes,
        ref_text: Optional[str],
        waveform: Optional[np.ndarray] = None,
    ) -> Tuple[str, bool]:
        """Return (prompt_hash, is_semantic_hit).

        First attempts ECAPA-TDNN semantic deduplication.  If an existing
        voice vector is close enough, the existing voice_id is reused.
        Otherwise a fresh hash is computed from the normalised waveform.
        """
        # Try ECAPA semantic match
        emb = self._get_voice_embedding(ref_audio_bytes)
        if emb is not None:
            similar_voice = self._find_similar_voice(emb)
            if similar_voice is not None:
                return similar_voice, True
            # No similar voice — save this embedding for future matches
            # (the actual save happens later after prompt creation)

        # Fallback: format-agnostic hash from normalised waveform
        if waveform is not None:
            wav = waveform
        else:
            try:
                wav = load_audio_bytes(ref_audio_bytes, self.sampling_rate)
            except Exception:
                wav = None

        if wav is not None:
            if wav.ndim > 1:
                wav = np.mean(wav, axis=0)
            wav = wav.astype(np.float32)
            wav_bytes = wav.tobytes()
        else:
            wav_bytes = ref_audio_bytes

        rt = (ref_text or "").strip().lower()
        payload = wav_bytes + b"|" + rt.encode("utf-8") + b"|" + str(self.gen_config.preprocess_prompt).encode()
        return _hash_bytes(payload), False

    def _audio_cache_key(self, mode: str, item: Dict[str, Any], prompt_hash: Optional[str] = None) -> str:
        text = item["text"]
        language = item.get("language") or ""
        duration = item.get("duration")
        speed = item.get("speed")
        dur_str = str(duration) if duration is not None else ""
        spd_str = str(speed) if speed is not None else ""

        if mode == "clone":
            ph = prompt_hash or "unknown"
            key = f"c|{text}|{ph}|{language}|{dur_str}|{spd_str}"
        elif mode == "design":
            instruct = self._normalize_instruct(item.get("instruct") or "")
            key = f"d|{text}|{instruct}|{language}|{dur_str}|{spd_str}"
        else:
            key = f"a|{text}|{language}|{dur_str}|{spd_str}"
        return _hash_str(key)

    # ------------------------------------------------------------------
    # Redis helpers
    # ------------------------------------------------------------------
    def _redis_key_audio(self, mode: str, audio_hash: str) -> str:
        return f"ov:audio:{mode}:{audio_hash}"

    def _redis_key_prompt(self, prompt_hash: str) -> str:
        return f"ov:prompt:{prompt_hash}"

    def _redis_incr(self, counter: str) -> None:
        if self._redis is None:
            return
        try:
            self._redis.hincrby("ov:stats", counter, 1)
        except Exception:
            pass

    def _redis_set_meta(self, key: str, ttl: int) -> None:
        if self._redis is None:
            return
        try:
            self._redis.setex(key, ttl, str(int(time.time())))
        except Exception:
            pass

    def _redis_exists(self, key: str) -> bool:
        if self._redis is None:
            return False
        try:
            return bool(self._redis.exists(key))
        except Exception:
            return False

    # ------------------------------------------------------------------
    # MinIO helpers
    # ------------------------------------------------------------------
    def _s3_key_audio(self, mode: str, audio_hash: str) -> str:
        return f"audio/{mode}/{audio_hash}.npz"

    def _s3_key_prompt(self, prompt_hash: str) -> str:
        return f"prompt/{prompt_hash}.pt"

    # ------------------------------------------------------------------
    # L1 + L2 + L3 Voice Prompt Cache
    # ------------------------------------------------------------------
    def _try_load_prompt_cache(self, prompt_hash: str) -> Optional[VoiceClonePrompt]:
        # L1
        with self._prompt_cache_lock:
            if prompt_hash in self._prompt_cache:
                self._prompt_cache.move_to_end(prompt_hash)
                self._redis_incr("prompt_hits")
                return self._prompt_cache[prompt_hash]

        # L2: Redis metadata check
        if not self._redis_exists(self._redis_key_prompt(prompt_hash)):
            self._redis_incr("prompt_misses")
            return None

        # L3: MinIO blob fetch
        if self._s3 is None:
            return None
        try:
            resp = self._s3.get_object(Bucket=CACHE_BUCKET, Key=self._s3_key_prompt(prompt_hash))
            buf = io.BytesIO(resp["Body"].read())
            obj = torch.load(buf, map_location="cpu", weights_only=False)
            prompt = VoiceClonePrompt(
                ref_audio_tokens=obj["ref_audio_tokens"],
                ref_text=obj["ref_text"],
                ref_rms=obj["ref_rms"],
            )
            with self._prompt_cache_lock:
                if prompt_hash not in self._prompt_cache:
                    if len(self._prompt_cache) >= self._prompt_cache_max:
                        self._prompt_cache.popitem(last=False)
                    self._prompt_cache[prompt_hash] = prompt
            self._redis_incr("prompt_hits")
            logger.debug("Prompt cache L2/L3 HIT -> L1: %s", prompt_hash[:8])
            return prompt
        except Exception as e:
            logger.warning("Prompt MinIO read failed: %s", e)
            return None

    def _save_prompt_cache(self, prompt_hash: str, prompt: VoiceClonePrompt, embedding: Optional[np.ndarray] = None) -> None:
        if self._s3 is None:
            return
        try:
            buf = io.BytesIO()
            torch.save(
                {
                    "ref_audio_tokens": prompt.ref_audio_tokens,
                    "ref_text": prompt.ref_text,
                    "ref_rms": prompt.ref_rms,
                },
                buf,
            )
            buf.seek(0)
            self._s3.put_object(Bucket=CACHE_BUCKET, Key=self._s3_key_prompt(prompt_hash), Body=buf)
            self._redis_set_meta(self._redis_key_prompt(prompt_hash), PROMPT_TTL_SECONDS)
            with self._prompt_cache_lock:
                if prompt_hash not in self._prompt_cache:
                    if len(self._prompt_cache) >= self._prompt_cache_max:
                        self._prompt_cache.popitem(last=False)
                    self._prompt_cache[prompt_hash] = prompt
            # Save ECAPA embedding for semantic deduplication
            if embedding is not None:
                self._save_voice_embedding(prompt_hash, embedding, prompt.ref_text)
            logger.debug("Prompt cache SAVED: %s", prompt_hash[:8])
        except Exception as e:
            logger.warning("Failed to save prompt cache: %s", e)

    # ------------------------------------------------------------------
    # L1 + L2 + L3 Audio Cache
    # ------------------------------------------------------------------
    def _try_load_audio_cache(self, mode: str, item: Dict[str, Any], prompt_hash: Optional[str] = None) -> Optional[np.ndarray]:
        audio_hash = self._audio_cache_key(mode, item, prompt_hash)

        # L1
        with self._audio_cache_lock:
            if audio_hash in self._audio_cache:
                self._audio_cache.move_to_end(audio_hash)
                self._redis_incr("audio_hits")
                return self._audio_cache[audio_hash]

        # L2: Redis metadata check
        if not self._redis_exists(self._redis_key_audio(mode, audio_hash)):
            self._redis_incr("audio_misses")
            return None

        # L3: MinIO blob fetch
        if self._s3 is None:
            return None
        try:
            resp = self._s3.get_object(Bucket=CACHE_BUCKET, Key=self._s3_key_audio(mode, audio_hash))
            data = np.load(io.BytesIO(resp["Body"].read()))
            audio = data["audio"]
            with self._audio_cache_lock:
                if audio_hash not in self._audio_cache:
                    if len(self._audio_cache) >= self._audio_cache_max:
                        self._audio_cache.popitem(last=False)
                    self._audio_cache[audio_hash] = audio
            self._redis_incr("audio_hits")
            logger.debug("Audio cache L2/L3 HIT -> L1: %s/%s", mode, audio_hash[:8])
            return audio
        except Exception as e:
            logger.warning("Audio MinIO read failed: %s", e)
            return None

    def _save_audio_cache(self, mode: str, item: Dict[str, Any], audio: np.ndarray, prompt_hash: Optional[str] = None) -> None:
        if self._s3 is None:
            return
        audio_hash = self._audio_cache_key(mode, item, prompt_hash)
        try:
            buf = io.BytesIO()
            np.savez(buf, audio=audio, sample_rate=self.sampling_rate)
            buf.seek(0)
            self._s3.put_object(Bucket=CACHE_BUCKET, Key=self._s3_key_audio(mode, audio_hash), Body=buf)
            self._redis_set_meta(self._redis_key_audio(mode, audio_hash), AUDIO_TTL_SECONDS)
            with self._audio_cache_lock:
                if audio_hash not in self._audio_cache:
                    if len(self._audio_cache) >= self._audio_cache_max:
                        self._audio_cache.popitem(last=False)
                    self._audio_cache[audio_hash] = audio
            logger.debug("Audio cache SAVED: %s/%s", mode, audio_hash[:8])
        except Exception as e:
            logger.warning("Failed to save audio cache: %s", e)

    # ------------------------------------------------------------------
    # Response builder
    # ------------------------------------------------------------------
    def _build_response(self, audio: np.ndarray):
        audio_tensor = pb_utils.Tensor("audio", audio.astype(np.float32))
        sr_tensor = pb_utils.Tensor(
            "sample_rate",
            np.array([self.sampling_rate], dtype=np.int32),
        )
        return pb_utils.InferenceResponse(output_tensors=[audio_tensor, sr_tensor])

    # ------------------------------------------------------------------
    # execute
    # ------------------------------------------------------------------
    def execute(self, requests: List[Any]) -> List[Any]:
        if not requests:
            return []

        groups: Dict[str, List[Dict[str, Any]]] = {"clone": [], "design": [], "auto": []}
        responses: Dict[int, List[Any]] = {}

        for req_idx, request in enumerate(requests):
            text_tensor = pb_utils.get_input_tensor_by_name(request, "text")
            batch_size = text_tensor.as_numpy().shape[0] if text_tensor is not None else 1
            responses[id(request)] = [None] * batch_size

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

                if duration is not None and abs(duration) < 1e-6:
                    duration = None
                if speed is not None and abs(speed) < 1e-6:
                    speed = None
                if speed is None:
                    speed = 1.0

                if not text:
                    err_msg = "Missing required input 'text'"
                    logger.error(err_msg)
                    responses[id(request)][i] = pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(err_msg)
                    )
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
                    mode = "clone"
                elif instruct and len(instruct) > 0:
                    mode = "design"
                else:
                    mode = "auto"

                # Phase 1: Full Response Cache (skip for clone — needs prompt_hash)
                if mode != "clone":
                    cached_audio = self._try_load_audio_cache(mode, item)
                    if cached_audio is not None:
                        responses[id(request)][i] = self._build_response(cached_audio)
                        continue

                item["_mode"] = mode
                groups[mode].append(item)

        # Phase 2: Process missed items group by group
        for mode, items in groups.items():
            if not items:
                continue
            if mode == "clone":
                self._process_clone_group(items, responses)
            elif mode == "design":
                self._process_design_group(items, responses)
            else:
                self._process_auto_group(items, responses)

        # Build final responses
        final_responses = []
        for request in requests:
            res_list = responses.get(id(request), [])
            for i, res in enumerate(res_list):
                if res is None:
                    res_list[i] = pb_utils.InferenceResponse(
                        error=pb_utils.TritonError("Internal error: response was not generated")
                    )
            if len(res_list) == 1:
                final_responses.append(res_list[0])
            else:
                logger.warning(
                    "Batched request with %d items returned only first response due to variable audio lengths",
                    len(res_list),
                )
                final_responses.append(res_list[0])

        return final_responses

    # ------------------------------------------------------------------
    # Clone group with Voice Prompt Caching + ECAPA deduplication
    # ------------------------------------------------------------------
    def _process_clone_group(
        self, items: List[Dict[str, Any]], responses: Dict[int, List[Any]]
    ) -> None:
        texts = []
        prompts = []
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

            ref_audio_bytes = item["ref_audio_bytes"]
            ref_text = item["ref_text"] if item["ref_text"] else None

            # Decode audio once
            try:
                wav_np = load_audio_bytes(ref_audio_bytes, self.sampling_rate)
            except Exception as e:
                logger.exception("Audio decode failed")
                responses[id(item["request"])][item["batch_idx"]] = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(f"Audio decode error: {e}")
                )
                continue

            # Compute voice key (ECAPA semantic or format-agnostic hash)
            prompt_hash, is_semantic = self._voice_prompt_key(ref_audio_bytes, ref_text, waveform=wav_np)

            # Phase 1b: Full Response Cache
            cached_audio = self._try_load_audio_cache("clone", item, prompt_hash)
            if cached_audio is not None:
                responses[id(item["request"])][item["batch_idx"]] = self._build_response(cached_audio)
                continue

            # Phase 2: Voice Prompt Cache
            prompt = self._try_load_prompt_cache(prompt_hash)
            emb = None
            if prompt is None:
                try:
                    wav_t = torch.from_numpy(wav_np)
                    prompt = self.model.create_voice_clone_prompt(
                        ref_audio=(wav_t, self.sampling_rate),
                        ref_text=ref_text,
                        preprocess_prompt=self.gen_config.preprocess_prompt,
                    )
                    # Extract ECAPA embedding for future semantic matches
                    emb = self._get_voice_embedding(ref_audio_bytes)
                    self._save_prompt_cache(prompt_hash, prompt, embedding=emb)
                    if is_semantic:
                        logger.info("ECAPA semantic voice deduplication: reused prompt for %s", prompt_hash[:8])
                except Exception as e:
                    logger.exception("Voice prompt creation failed")
                    responses[id(item["request"])][item["batch_idx"]] = pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(f"Voice prompt error: {e}")
                    )
                    continue

            texts.append(item["text"])
            prompts.append(prompt)
            languages.append(item["language"] if item["language"] else None)
            durations.append(item["duration"])
            speeds.append(item["speed"])
            item["_prompt_hash"] = prompt_hash
            item_metadata.append(item)

        if not texts:
            return

        # Enable TeaCache for the generation phase
        if TEACACHE_ENABLE:
            _tea_cache.enable()

        try:
            audios = self.model.generate(
                text=texts,
                voice_clone_prompt=prompts,
                language=languages,
                duration=durations,
                speed=speeds,
                generation_config=self.gen_config,
            )
            self._fill_responses("clone", item_metadata, audios, responses)
        except Exception as e:
            logger.exception("Voice clone batch failed")
            self._fill_errors(item_metadata, str(e), responses)
        finally:
            if TEACACHE_ENABLE:
                _tea_cache.disable()

    # ------------------------------------------------------------------
    # Design / Auto groups
    # ------------------------------------------------------------------
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

        if TEACACHE_ENABLE:
            _tea_cache.enable()
        try:
            audios = self.model.generate(
                text=texts,
                instruct=instructs,
                language=languages,
                duration=durations,
                speed=speeds,
                generation_config=self.gen_config,
            )
            self._fill_responses("design", item_metadata, audios, responses)
        except Exception as e:
            logger.exception("Voice design batch failed")
            self._fill_errors(item_metadata, str(e), responses)
        finally:
            if TEACACHE_ENABLE:
                _tea_cache.disable()

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

        if TEACACHE_ENABLE:
            _tea_cache.enable()
        try:
            audios = self.model.generate(
                text=texts,
                language=languages,
                duration=durations,
                speed=speeds,
                generation_config=self.gen_config,
            )
            self._fill_responses("auto", item_metadata, audios, responses)
        except Exception as e:
            logger.exception("Auto voice batch failed")
            self._fill_errors(item_metadata, str(e), responses)
        finally:
            if TEACACHE_ENABLE:
                _tea_cache.disable()

    # ------------------------------------------------------------------
    # Fill responses and cache
    # ------------------------------------------------------------------
    def _fill_responses(
        self,
        mode: str,
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
                prompt_hash = item.get("_prompt_hash") if mode == "clone" else None
                self._save_audio_cache(mode, item, audio, prompt_hash)
                responses[req_id][batch_idx] = self._build_response(audio)
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

    # ------------------------------------------------------------------
    # Tensor extraction helpers
    # ------------------------------------------------------------------
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
