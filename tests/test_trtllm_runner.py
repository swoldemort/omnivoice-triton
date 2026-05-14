"""Smoke test for TRTLLMRunner."""

import pytest

from omnivoice_triton.models import TRTLLMRunner


@pytest.mark.skipif(
    not __import__("importlib.util").find_spec("tensorrt_llm"),
    reason="tensorrt-llm not installed",
)
def test_trtllm_runner_import():
    assert TRTLLMRunner is not None
