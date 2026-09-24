"""
Benchmark: LlamaGen decode — standard model.forward() vs Triton megakernel.

Both runs share an identical prefill. Outputs are compared for correctness and
memory bandwidth is reported. Pass --vq-ckpt to decode tokens to images.

Usage:
  python benchmark_megakernel.py --gpt-model GPT-XL \\
      --gpt-ckpt pretrained_models/c2i_XL_384.pt \\
      --vq-ckpt  pretrained_models/vq_ds16_c2i.pt \\
      --image-size 384 --class-label 207 --temperature 1.0 --top-k 2000
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time, argparse
import torch
from torchvision.utils import save_image

from autoregressive.models.gpt import GPT_models
from autoregressive.models.model_triton import setup_from_transformer, llamagen_decode, LlamaGenBuffers


# ── sampling ─────────────────────────────────────────────────────────────────

def sample_token(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    if logits.dim() == 2:
        logits = logits[0]
    if temperature == 0.0 or top_k == 1:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits.float() / max(temperature, 1e-5)
    if top_k > 0:
        threshold = torch.topk(logits, min(top_k, logits.size(-1))).values[-1]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)


# ── KV cache helpers ──────────────────────────────────────────────────────────

def snapshot_kv(model, num_layers: int, num_filled: int):
    """Clone the post-prefill KV cache for repeatable benchmark runs."""
    return [(model.layers[l].attention.kv_cache.k_cache[0, :, :num_filled].clone(),
             model.layers[l].attention.kv_cache.v_cache[0, :, :num_filled].clone())
            for l in range(num_layers)]


def restore_kv(model, snap, num_layers: int):
    for l in range(num_layers):
        kv = model.layers[l].attention.kv_cache
        n  = snap[l][0].shape[1]
        kv.k_cache[0, :, :n] = snap[l][0]
        kv.v_cache[0, :, :n] = snap[l][1]


def make_triton_buffers(snap, rope, params, max_context: int, dtype, device) -> LlamaGenBuffers:
    """
    Build LlamaGenBuffers from a KV snapshot.
    Model layout : (n_head, num_filled, head_dim)  [batch stripped]
    Triton layout: (layers, 2, max_ctx, n_kv_head, head_dim)
    """
    n = snap[0][0].shape[1]
    kv_cache = torch.zeros(len(snap), 2, max_context, params.num_kv_heads, params.head_dim,
                           dtype=dtype, device=device)
    for l, (k, v) in enumerate(snap):
        kv_cache[l, 0, :n] = k.permute(1, 0, 2).to(dtype)
        kv_cache[l, 1, :n] = v.permute(1, 0, 2).to(dtype)
    return LlamaGenBuffers(kv_cache=kv_cache, rope=rope, position=n)


# ── decode loops ──────────────────────────────────────────────────────────────

@torch.no_grad()
def decode_std(model, first_token, num_steps, cls_token_num, temperature, top_k):
    tokens = [first_token]
    cur    = first_token.view(1, 1)
    device = first_token.device
    for i in range(num_steps):
        pos = torch.tensor([cls_token_num + i], device=device, dtype=torch.int)
        logits, _ = model(cur, cond_idx=None, input_pos=pos)
        nxt = sample_token(logits[0, -1], temperature, top_k)
        tokens.append(nxt)
        cur = nxt.view(1, 1)
    return tokens


@torch.no_grad()
def decode_triton(first_token, params, buffers, num_steps, temperature, top_k):
    tokens = [first_token]
    cur    = first_token[:1]
    for _ in range(num_steps):
        logits = llamagen_decode(cur, params, buffers)
        nxt    = sample_token(logits[0], temperature, top_k)
        tokens.append(nxt)
        cur = nxt
    return tokens


# ── timing ────────────────────────────────────────────────────────────────────

def bench(fn, warmup, repeats):
    for _ in range(warmup):
        fn(); torch.cuda.synchronize()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        result = fn()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - t0) / repeats


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    precision = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.precision]

    latent_size      = args.image_size // args.downsample_size
    num_image_tokens = latent_size ** 2
    cls_token_num    = args.cls_token_num
    max_seq          = cls_token_num + num_image_tokens
    num_decode_steps = num_image_tokens - 1  # prefill produces token 0

    print(f"Model    : {args.gpt_model}  |  tokens: {num_image_tokens}  |  precision: {args.precision}")

    gpt = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size, block_size=latent_size ** 2,
        num_classes=args.num_classes, cls_token_num=cls_token_num, model_type="c2i",
    ).to(device=device, dtype=precision)

    if args.gpt_ckpt:
        ckpt = torch.load(args.gpt_ckpt, map_location="cpu")
        sd   = ckpt.get("model", ckpt.get("module", ckpt.get("state_dict", ckpt)))
        gpt.load_state_dict(sd, strict=False)
        print(f"Loaded   : {args.gpt_ckpt}")
    else:
        print("Checkpoint: none — random weights (throughput valid, outputs meaningless)")

    gpt.eval()
    with torch.device(device):
        gpt.setup_caches(max_batch_size=1, max_seq_length=max_seq, dtype=precision)

    # Shared prefill
    cond_idx = torch.tensor([args.class_label], device=device)
    print(f"\nPrefill  : class={args.class_label}  T={args.temperature}  top_k={args.top_k}")
    torch.manual_seed(args.seed)
    first_token = (lambda logits, _: sample_token(logits[0, -1], args.temperature, args.top_k))(
        *gpt(None, cond_idx, torch.arange(0, cls_token_num, device=device))
    )
    torch.cuda.synchronize()
    print(f"  first token = {first_token.item()}")

    kv_snap = snapshot_kv(gpt, gpt.n_layer, cls_token_num)
    params, init_bufs = setup_from_transformer(gpt, max_context=max_seq, dtype=precision)
    rope = init_bufs.rope

    # Standard decode
    def run_std():
        torch.manual_seed(args.seed + 1)
        restore_kv(gpt, kv_snap, gpt.n_layer)
        return decode_std(gpt, first_token, num_decode_steps, cls_token_num,
                          args.temperature, args.top_k)

    print("\nWarming up standard...")
    run_std()
    print("Timing standard...")
    std_tokens, t_std = bench(run_std, args.warmup, args.repeats)
    tps_std = num_decode_steps / t_std
    print(f"  {num_decode_steps} tokens in {t_std*1e3:.1f} ms")

    # Triton decode
    print("\nWarming up Triton (triggers JIT)...")
    decode_triton(first_token, params,
                  make_triton_buffers(kv_snap, rope, params, max_seq, precision, device),
                  min(5, num_decode_steps), args.temperature, args.top_k)
    torch.cuda.synchronize()
    print("  done.")

    def run_triton():
        torch.manual_seed(args.seed + 1)
        return decode_triton(first_token, params,
                             make_triton_buffers(kv_snap, rope, params, max_seq, precision, device),
                             num_decode_steps, args.temperature, args.top_k)

    print("Timing Triton...")
    triton_tokens, t_triton = bench(run_triton, args.warmup, args.repeats)
    tps_triton = num_decode_steps / t_triton
    print(f"  {num_decode_steps} tokens in {t_triton*1e3:.1f} ms")

    # Memory bandwidth
    # At bs=1 decode every weight is loaded once per token (memory-BW bound).
    # KV bytes = read all cached K/V at avg position + write current token.
    dtype_bytes  = 2 if precision != torch.float32 else 4
    avg_pos      = cls_token_num + num_decode_steps // 2
    weight_bytes = sum(p.numel() * dtype_bytes for p in [
        params.tok_embeddings, params.l_attn_norm, params.l_wqkv, params.l_wo,
        params.l_ffn_norm, params.l_w1, params.l_w3, params.l_w2, params.norm, params.output,
    ])
    kv_bytes = params.num_layers * (
        (2 * avg_pos + 2) * params.num_kv_heads * params.head_dim
    ) * dtype_bytes
    bps          = weight_bytes + kv_bytes
    gbps_std     = bps * tps_std    / 1e9
    gbps_triton  = bps * tps_triton / 1e9

    print(f"\nBytes/step: {bps/1e9:.2f} GB  (weights {weight_bytes/1e9:.2f} + KV {kv_bytes/1e9:.2f} @ avg pos {avg_pos})")
    print("\n" + "=" * 60)
    print(f"  {'':12s}  {'tok/s':>8s}  {'GB/s':>8s}")
    print(f"  {'Standard':12s}  {tps_std:>8.1f}  {gbps_std:>8.1f}")
    print(f"  {'Triton':12s}  {tps_triton:>8.1f}  {gbps_triton:>8.1f}")
    print(f"  {'Speedup':12s}  {tps_triton/tps_std:>8.2f}x  {gbps_triton/gbps_std:>8.2f}x")
    print("=" * 60)

    # Correctness (meaningful with --temperature 0; stochastic runs diverge naturally)
    std_seq    = torch.stack(std_tokens[1:]).squeeze(-1)
    triton_seq = torch.stack(triton_tokens[1:]).squeeze(-1)
    n_match = (std_seq == triton_seq).sum().item()
    pct     = 100.0 * n_match / num_decode_steps
    print(f"\nCorrectness: {n_match}/{num_decode_steps} tokens match ({pct:.1f}%)", end="")
    if n_match < num_decode_steps:
        i = (std_seq != triton_seq).nonzero(as_tuple=True)[0][0].item()
        print(f"  — first mismatch at step {i}  std={std_seq[i].item()}  triton={triton_seq[i].item()}")
    else:
        print("  — exact match")

    # Image save
    if args.vq_ckpt:
        from tokenizer.tokenizer_image.vq_model import VQ_models
        print("\nDecoding images...")
        vq = VQ_models[args.vq_model](codebook_size=args.codebook_size,
                                      codebook_embed_dim=args.codebook_embed_dim).to(device)
        vq.eval()
        ckpt = torch.load(args.vq_ckpt, map_location="cpu")
        vq.load_state_dict(ckpt["model"])
        del ckpt

        qzshape = [1, args.codebook_embed_dim, latent_size, latent_size]
        def to_idx(tl):
            return torch.stack(tl).squeeze(-1).unsqueeze(0)  # (1, num_tokens)

        with torch.no_grad():
            img_std    = vq.decode_code(to_idx(std_tokens),    qzshape)
            img_triton = vq.decode_code(to_idx(triton_tokens), qzshape)

        kw = dict(nrow=1, normalize=True, value_range=(-1, 1))
        save_image(torch.cat([img_std, img_triton]), f"sample_megakernel_class{args.class_label}.png", **kw)
        save_image(img_std,    f"sample_std_class{args.class_label}.png",    **kw)
        save_image(img_triton, f"sample_triton_class{args.class_label}.png", **kw)
        print(f"  sample_std_class{args.class_label}.png  (top=standard, bottom=triton in grid)")
        print(f"  sample_triton_class{args.class_label}.png")
    else:
        print(f"\nTokens (first 20):  std={std_seq[:20].tolist()}")
        print(f"                 triton={triton_seq[:20].tolist()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gpt-model",           default="GPT-L",   choices=list(GPT_models.keys()))
    p.add_argument("--gpt-ckpt",            default=None)
    p.add_argument("--precision",           default="bf16",    choices=["none", "fp16", "bf16"])
    p.add_argument("--cls-token-num",       default=1,  type=int)
    p.add_argument("--image-size",          default=256, type=int, choices=[256, 384, 512])
    p.add_argument("--downsample-size",     default=16,  type=int, choices=[8, 16])
    p.add_argument("--num-classes",         default=1000, type=int)
    p.add_argument("--codebook-size",       default=16384, type=int)
    p.add_argument("--vq-model",            default="VQ-16")
    p.add_argument("--vq-ckpt",             default=None)
    p.add_argument("--codebook-embed-dim",  default=8,  type=int)
    p.add_argument("--class-label",         default=207, type=int)
    p.add_argument("--seed",                default=42,  type=int)
    p.add_argument("--temperature",         default=1.0, type=float, help="0 = greedy")
    p.add_argument("--top-k",               default=2000, type=int)
    p.add_argument("--warmup",              default=1,   type=int)
    p.add_argument("--repeats",             default=3,   type=int)
    main(p.parse_args())
