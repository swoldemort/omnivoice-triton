"""Profile end-to-end components to find the next bottleneck."""

import sys
import time

sys.path.insert(0, "/app/src")

import torch
from omnivoice_triton.models.trtllm_runner import TRTLLMRunner

TEXT = "The quick brown fox jumps over the lazy dog."
NUM_STEP = 16

def main():
    runner = TRTLLMRunner(
        device="cuda", model_id="k2-fsa/OmniVoice", dtype="fp16",
        engine_dir="/tmp/trtllm_engine"
    )
    print("Loading model...")
    runner.load_model()
    model = runner._model
    
    # Warmup
    runner.generate(TEXT, num_step=NUM_STEP)
    torch.cuda.synchronize()
    
    # E2E timing
    t0 = time.perf_counter()
    result = runner.generate(TEXT, num_step=NUM_STEP)
    torch.cuda.synchronize()
    e2e = time.perf_counter() - t0
    print(f"\nE2E latency: {e2e:.3f}s")
    
    # Profile internal steps manually
    from omnivoice import OmniVoiceGenerationConfig
    gen_config = OmniVoiceGenerationConfig(num_step=NUM_STEP, guidance_scale=2.0, class_temperature=0.0)
    
    # Tokenization
    t0 = time.perf_counter()
    text_ids = model.tokenize_text(TEXT)
    torch.cuda.synchronize()
    t_tokenize = time.perf_counter() - t0
    
    # Text embedding
    t0 = time.perf_counter()
    text_embeds = model.get_text_embeddings(text_ids)
    torch.cuda.synchronize()
    t_text_emb = time.perf_counter() - t0
    
    # Generation loop (just 1 step to measure per-step cost)
    # We'll hook into the model's forward to measure LLM vs heads vs codec
    
    # Measure one full generation step
    audio_tokens = model.prepare_audio_tokens(text_embeds)
    audio_mask = model.prepare_audio_mask(audio_tokens)
    
    # One LLM forward + audio heads
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.forward(audio_tokens, audio_mask)
    torch.cuda.synchronize()
    t_one_forward = time.perf_counter() - t0
    
    # Audio codec decode (simulate with dummy tokens)
    # Get actual generated audio and measure decode
    t0 = time.perf_counter()
    _ = model.audio_tokenizer.decode(audio_tokens)
    torch.cuda.synchronize()
    t_codec = time.perf_counter() - t0
    
    print(f"\n=== Component breakdown ===")
    print(f"Tokenization:       {t_tokenize*1000:.1f}ms")
    print(f"Text embedding:     {t_text_emb*1000:.1f}ms")
    print(f"One forward+heads:  {t_one_forward*1000:.1f}ms")
    print(f"Codec decode:       {t_codec*1000:.1f}ms")
    print(f"Estimated loop:     {t_one_forward*NUM_STEP*1000:.1f}ms ({NUM_STEP} steps)")
    print(f"\nSum estimate:       {(t_tokenize + t_text_emb + t_one_forward*NUM_STEP + t_codec)*1000:.1f}ms")
    print(f"Actual E2E:         {e2e*1000:.1f}ms")

if __name__ == "__main__":
    main()
