"""Dynamic batching for OmniVoice inference.

Collects incoming text requests into batches up to max_batch_size or
max_wait_ms, then runs a single batched generation through the runner.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Request:
    """Single inference request."""

    text: str
    language: str | None = None
    ref_audio: Any = None
    ref_text: str = ""
    instruct: str = ""
    num_step: int = 16
    guidance_scale: float = 2.0
    class_temperature: float = 0.0

    # Filled by batcher after generation
    audio: np.ndarray | None = field(default=None, repr=False)
    error: str | None = field(default=None)
    _event: threading.Event = field(default_factory=threading.Event, repr=False)

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the request is processed."""
        return self._event.wait(timeout)

    def _set_done(self) -> None:
        self._event.set()


class DynamicBatcher:
    """Queues requests and executes them in dynamically-sized batches.

    Args:
        runner: Any runner with a ``generate(texts: list[str], ...)`` method.
        max_batch_size: Maximum number of requests to batch together.
        max_wait_ms: Maximum time to wait (ms) before running a partial batch.
    """

    def __init__(
        self,
        runner: Any,
        max_batch_size: int = 16,
        max_wait_ms: float = 80.0,
    ) -> None:
        self.runner = runner
        self.max_batch_size = max_batch_size
        self.max_wait_s = max_wait_ms / 1000.0
        self._request_queue: queue.Queue[Request] = queue.Queue()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._shutdown = False

    def start(self) -> None:
        """Start the background batching worker."""
        self._worker_thread.start()
        logger.info("DynamicBatcher started (max_batch=%d, max_wait=%.0fms)", self.max_batch_size, self.max_wait_s * 1000)

    def submit(self, req: Request) -> Request:
        """Enqueue a request and return it (call ``req.wait()`` to block)."""
        self._request_queue.put(req)
        return req

    def shutdown(self, wait: bool = True) -> None:
        """Signal shutdown and optionally wait for the worker to finish."""
        self._shutdown = True
        self._request_queue.put(Request(text="__shutdown__"))  # wake worker
        if wait:
            self._worker_thread.join(timeout=5.0)

    def _worker_loop(self) -> None:
        """Main loop: gather requests → form batch → run inference → distribute results."""
        while not self._shutdown:
            batch = self._gather_batch()
            if not batch:
                continue
            if batch[0].text == "__shutdown__":
                break
            self._run_batch(batch)

    def _gather_batch(self) -> list[Request]:
        """Collect requests until batch is full or timeout expires."""
        batch: list[Request] = []
        deadline: float | None = None

        while len(batch) < self.max_batch_size:
            remaining = None if deadline is None else max(0.0, deadline - time.perf_counter())
            try:
                req = self._request_queue.get(timeout=remaining)
            except queue.Empty:
                break

            if req.text == "__shutdown__":
                batch.append(req)
                break

            batch.append(req)
            if deadline is None:
                deadline = time.perf_counter() + self.max_wait_s

        return batch

    def _run_batch(self, batch: list[Request]) -> None:
        """Execute a single batched generation and fill each request's result."""
        texts = [r.text for r in batch]
        num_step = batch[0].num_step
        guidance_scale = batch[0].guidance_scale

        logger.debug("Running batch: size=%d, num_step=%d", len(batch), num_step)
        t0 = time.perf_counter()

        try:
            result = self.runner.generate(
                text=texts,
                num_step=num_step,
                guidance_scale=guidance_scale,
            )
            audios = result.get("audios", [])
            if not isinstance(audios, list):
                audios = [audios]

            # Pad with None if runner returned fewer audios (shouldn't happen)
            while len(audios) < len(batch):
                audios.append(None)

            for req, audio in zip(batch, audios):
                req.audio = audio

        except Exception as exc:
            logger.exception("Batch inference failed")
            for req in batch:
                req.error = f"{type(exc).__name__}: {exc}"

        elapsed = time.perf_counter() - t0
        logger.info(
            "Batch done: size=%d | time=%.3fs | throughput=%.1f req/s",
            len(batch),
            elapsed,
            len(batch) / elapsed,
        )

        for req in batch:
            req._set_done()
