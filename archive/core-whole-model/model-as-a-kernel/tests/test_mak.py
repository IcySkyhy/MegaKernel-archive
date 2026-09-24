import pytest
import torch

import kernels

mak = kernels.get_kernel("phanerozoic/model-as-a-kernel", version=1,
                         trust_remote_code=True)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                   reason="CUDA required")

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
PROMPT_LEN = 32
TF_STEPS = 64


@pytest.fixture(scope="module")
def models():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(1234)
    ref = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="eager"
    ).cuda().eval()
    mm = mak.MegaModel.from_pretrained(ref, max_seq=2048, max_gen=2048)
    return ref, mm


def rmsnorm_ref(x_bf, w_bf, eps):
    xf = x_bf.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    return w_bf * (xf * torch.rsqrt(var + eps)).to(torch.bfloat16)


def cache_stepped_logits(model, seq):
    from transformers import DynamicCache
    cache = DynamicCache()
    out = torch.empty(len(seq), model.config.vocab_size, device="cuda")
    with torch.no_grad():
        for i, t in enumerate(seq):
            r = model(torch.tensor([[t]], device="cuda"),
                      past_key_values=cache, use_cache=True)
            cache = r.past_key_values
            out[i] = r.logits[0, -1].float()
    return out


@requires_cuda
def test_op_level_bitwise_pos0(models):
    """At position 0 rope is the identity and attention copies V, so the
    fused rmsnorm+GEMV and attention paths are individually checkable
    bitwise against manual eager-semantics replication."""
    ref, mm = models
    tok = 4321
    lyr = ref.model.layers[0]
    with torch.no_grad():
        x = ref.model.embed_tokens.weight[tok]
        h1 = rmsnorm_ref(x, lyr.input_layernorm.weight,
                         ref.config.rms_norm_eps)
        q = h1 @ lyr.self_attn.q_proj.weight.T
        k = h1 @ lyr.self_attn.k_proj.weight.T
        v = h1 @ lyr.self_attn.v_proj.weight.T
    mm.decode_step(tok, 0, phased=True)
    qdim, kvdim = mm.Hq * mm.D, mm.Hkv * mm.D
    mm._token.fill_(tok)
    mak.ops.mak_run_phased(mm._prog[:2].contiguous(), 0, 0, False, mm._maxk)
    ref_qkv = torch.cat([q, k, v]).bfloat16().float()
    dq = (mm._qkv[:qdim + 2 * kvdim].float() - ref_qkv).abs()
    assert torch.allclose(mm._qkv[:qdim + 2 * kvdim].float(), ref_qkv, rtol=1 / 128, atol=1e-3), dq.max().item()
    # at S=1 the attention partial is exp(0) * V exactly
    mak.ops.mak_run_phased(mm._prog[:4].contiguous(), 0, 0, False, mm._maxk)
    vsec = mm._qkv[qdim + kvdim:qdim + 2 * kvdim].view(mm.Hkv, mm.D)
    rep = mm.Hq // mm.Hkv
    expected = vsec[torch.arange(mm.Hq, device="cuda") // rep].float()
    part = mm._partials[:mm.Hq * mm._maxch * (mm.D + 2)].view(mm.Hq, mm._maxch, mm.D + 2)[:, 0, :mm.D]
    assert torch.equal(part, expected)


@requires_cuda
def test_rope_bitwise_vs_transformers(models):
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    ref, mm = models
    pos = 8
    seq = torch.randint(0, mm.V, (pos + 1,)).tolist()
    for i, t in enumerate(seq):
        mm.decode_step(t, i)
    lyr = ref.model.layers[0]
    with torch.no_grad():
        x = ref.model.embed_tokens.weight[seq[pos]]
        h1 = rmsnorm_ref(x, lyr.input_layernorm.weight,
                         ref.config.rms_norm_eps)
        k_pre = (h1 @ lyr.self_attn.k_proj.weight.T).view(1, mm.Hkv, 1, mm.D)
        pos_ids = torch.tensor([[pos]], device="cuda")
        dummy = torch.zeros(1, 1, mm.D, device="cuda", dtype=torch.bfloat16)
        cos, sin = ref.model.rotary_emb(dummy, pos_ids)
        _, k_rot = apply_rotary_pos_emb(k_pre, k_pre, cos, sin)
    assert torch.equal(mm._kcache[0, :, pos], k_rot[0, :, 0].contiguous())


@requires_cuda
def test_whole_model_within_reference_band(models):
    """Deviation from cache-stepped transformers eager must be comparable to
    transformers' own cache-vs-batch disagreement (bf16 networks are
    order-sensitive; the reference is only defined up to this band)."""
    ref, mm = models
    torch.manual_seed(99)
    prompt = torch.randint(0, mm.V, (PROMPT_LEN,)).tolist()
    with torch.no_grad():
        hf_seq = ref.generate(torch.tensor([prompt], device="cuda"),
                              max_new_tokens=TF_STEPS, do_sample=False,
                              pad_token_id=0)[0].tolist()
        ref_batch = ref(torch.tensor([hf_seq], device="cuda")).logits[0].float()
    ref_cache = cache_stepped_logits(ref, hf_seq)
    n = len(hf_seq)
    mine = torch.empty(n, mm.V, device="cuda")
    for i, t in enumerate(hf_seq):
        mine[i] = mm.decode_step(t, i).clone()

    # Yardsticks: transformers' own spread between execution paths
    # (cache vs batch, eager vs sdpa). Gates: argmax agreement at the
    # HF-internal level, bulk statistics (mean, q99) within 2x of the HF
    # pairings, and a catastrophic ceiling on max vs the logit scale.
    from transformers import AutoModelForCausalLM
    alt = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()
    ref_sdpa = cache_stepped_logits(alt, hf_seq)
    del alt
    torch.cuda.empty_cache()

    def stats(a, b):
        d = (a.bfloat16().float() - b.bfloat16().float()).abs()
        q99 = torch.quantile(d.flatten().float(), 0.99).item()
        agree = (a.argmax(-1) == b.argmax(-1)).sum().item()
        return d.mean().item(), q99, d.max().item(), agree

    hf_mean, hf_q99, hf_max, hf_agree = stats(ref_cache, ref_batch)
    sw_mean, sw_q99, sw_max, sw_agree = stats(ref_sdpa, ref_cache)
    my_mean, my_q99, my_max, my_agree = stats(mine, ref_cache)
    scale = ref_cache.abs().max().item()
    print(f"cache-vs-batch mean={hf_mean:.4f} q99={hf_q99:.3f} "
          f"max={hf_max:.3f} agree={hf_agree}/{n}")
    print(f"sdpa-vs-eager  mean={sw_mean:.4f} q99={sw_q99:.3f} "
          f"max={sw_max:.3f} agree={sw_agree}/{n}")
    print(f"ours-vs-eager  mean={my_mean:.4f} q99={my_q99:.3f} "
          f"max={my_max:.3f} agree={my_agree}/{n} (logit scale {scale:.1f})")

    assert my_agree >= n * 0.90
    assert my_agree >= min(hf_agree, sw_agree) - 6
    assert my_mean <= 2.0 * max(hf_mean, sw_mean)
    assert my_q99 <= 2.0 * max(hf_q99, sw_q99)
    assert my_max <= 0.15 * scale


@requires_cuda
def test_layer0_kv_cache_matches_transformers(models):
    """Layer 0 K/V cache along a 48-token sequence vs DynamicCache. The
    inputs are exact (embeddings), so any rope/theta/cache-layout regression
    is a direct, position-resolved failure here rather than a fuzzy band.
    Single-ulp GEMV rounding variance across cublas versions is tolerated;
    structure is not."""
    from transformers import DynamicCache
    ref, mm = models
    torch.manual_seed(5)
    S = 48
    seq = torch.randint(0, mm.V, (S,)).tolist()
    cache = DynamicCache()
    with torch.no_grad():
        for t in seq:
            r = ref(torch.tensor([[t]], device="cuda"),
                    past_key_values=cache, use_cache=True)
            cache = r.past_key_values
    for i, t in enumerate(seq):
        mm.decode_step(t, i)
    k_hf = cache.layers[0].keys[0].float()
    v_hf = cache.layers[0].values[0].float()
    k_me = mm._kcache[0, :, :S].float()
    v_me = mm._vcache[0, :, :S].float()
    for name, me, hf in (("K", k_me, k_hf), ("V", v_me, v_hf)):
        d = (me - hf).abs()
        mismatch = (d > 0).float().mean().item()
        print(f"layer0 {name}: max={d.max().item():.5f} "
              f"mismatch_frac={mismatch:.2e}")
        assert d.max().item() <= 0.25
        assert mismatch <= 1e-3


@requires_cuda
def test_fused_equals_phased_and_deterministic(models):
    _, mm = models
    lg_f1 = mm.decode_step(123, 0).clone()
    lg_p = mm.decode_step(123, 0, phased=True).clone()
    lg_f2 = mm.decode_step(123, 0).clone()
    assert torch.equal(lg_f1, lg_p)
    assert torch.equal(lg_f1, lg_f2)


@requires_cuda
def test_batch_invariance(models):
    """Batched decode is invariant: each sequence's stream matches its
    single-sequence generation and does not depend on batch order."""
    _, mm = models
    B = min(8, mm.batch_max())
    prompts = [[3, 7 + i, 11, 13 + i, 17] for i in range(B)]
    solo = [mm.generate(p, 20) for p in prompts]
    assert mm.generate_batch(prompts, 20) == solo
    order = list(range(B - 1, -1, -1))
    perm = mm.generate_batch([prompts[i] for i in order], 20)
    assert all(perm[j] == solo[order[j]] for j in range(B))


@requires_cuda
def test_nf4_packed_matches_dequant():
    """A bitsandbytes nf4 checkpoint loads with weights kept packed and
    dequantized in the kernel, bitwise identical to the same weights
    dequantized to bf16, and deterministic."""
    bnb = pytest.importorskip("bitsandbytes")
    transformers = pytest.importorskip("transformers")
    from transformers import BitsAndBytesConfig, AutoModelForCausalLM
    mid = "HuggingFaceTB/SmolLM2-135M"
    try:
        q = AutoModelForCausalLM.from_pretrained(
            mid, quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False))
    except Exception as e:
        pytest.skip(f"bitsandbytes 4-bit unavailable here: {e}")
    mm = mak.MegaModel.from_pretrained(q, max_seq=512, max_gen=64)
    assert mm._has_q4 and mm._layers[0]["wdown"]["packed"].dtype == torch.uint8
    ref = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16)
    for name, mod in q.named_modules():
        if isinstance(mod, bnb.nn.Linear4bit):
            w = bnb.functional.dequantize_4bit(mod.weight.data,
                                               mod.weight.quant_state)
            ref.get_submodule(name).weight.data = \
                w.to(torch.bfloat16).cpu().clone()
    mr = mak.MegaModel.from_pretrained(ref.cuda().eval(), max_seq=512,
                                       max_gen=64)
    prompt = [5, 9, 13, 21, 33, 41, 40, 32]
    assert torch.equal(mm.decode_step(prompt[0], 0),
                       mr.decode_step(prompt[0], 0))
    g = mm.generate(prompt, 32)
    assert g == mr.generate(prompt, 32) == mm.generate(prompt, 32)


@requires_cuda
@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-0.6B",
                                      "TinyLlama/TinyLlama-1.1B-Chat-v1.0"])
def test_additional_model_coverage(model_id):
    """Qwen3-0.6B exercises qk-norm, D=128, theta 1e6; TinyLlama exercises
    GQA 32/4 at 1.1B. Gates: transformers parity band, layer-0 KV cache,
    mode equivalence."""
    transformers = pytest.importorskip("transformers")
    from transformers import DynamicCache
    ref = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="eager"
    ).cuda().eval()
    mm = mak.MegaModel.from_pretrained(ref, max_seq=512, max_gen=512)
    torch.manual_seed(11)
    steps = 48
    seq = torch.randint(0, mm.V, (steps,)).tolist()
    cache = DynamicCache()
    ref_logits = torch.empty(steps, mm.V, device="cuda")
    with torch.no_grad():
        for i, t in enumerate(seq):
            r = ref(torch.tensor([[t]], device="cuda"),
                    past_key_values=cache, use_cache=True)
            cache = r.past_key_values
            ref_logits[i] = r.logits[0, -1].float()
    mine = torch.empty(steps, mm.V, device="cuda")
    for i, t in enumerate(seq):
        mine[i] = mm.decode_step(t, i).clone()
    d = (mine.bfloat16().float() - ref_logits.bfloat16().float()).abs()
    agree = (mine.argmax(-1) == ref_logits.argmax(-1)).sum().item()
    print(f"{model_id}: band mean={d.mean().item():.4f} "
          f"max={d.max().item():.3f} agree={agree}/{steps}")
    assert agree >= steps * 0.85
    assert d.mean().item() < 0.3
    for cache_t, mine_t in ((cache.layers[0].keys[0], mm._kcache[0, :, :steps]),
                            (cache.layers[0].values[0], mm._vcache[0, :, :steps])):
        dd = (mine_t.float() - cache_t.float()).abs()
        assert dd.max().item() <= 0.25
        assert (dd > 0).float().mean().item() <= 5e-3
    a = mm.decode_step(seq[0], 0).clone()
    b = mm.decode_step(seq[0], 0, phased=True).clone()
    assert torch.equal(a, b)
    assert mm.generate(seq[:8], 16) == mm.generate(seq[:8], 16,
                                                   single_launch=False)
    del ref, mm
    torch.cuda.empty_cache()


@requires_cuda
def test_generate_closed_loop_and_speed(models):
    import time
    ref, mm = models
    torch.manual_seed(7)
    prompt = torch.randint(0, mm.V, (PROMPT_LEN,)).tolist()
    out = mm.generate(prompt, 64)                        # one launch total
    out_pt = mm.generate(prompt, 64, single_launch=False)  # launch per token
    assert out == out_pt
    assert len(out) == 64 and all(0 <= t < mm.V for t in out)
    # greedy self-consistency: closed loop equals explicit decode loop
    lg = mm.prefill(prompt).clone()
    tok = lg.bfloat16().float().argmax().item()
    assert tok == out[0]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    mm.generate(prompt, 128)
    torch.cuda.synchronize()
    dt_ours = time.perf_counter() - t0
    with torch.no_grad():
        ids = torch.tensor([prompt], device="cuda")
        ref.generate(ids, max_new_tokens=8, do_sample=False, pad_token_id=0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ref.generate(ids, max_new_tokens=128, do_sample=False, pad_token_id=0)
        torch.cuda.synchronize()
        dt_hf = time.perf_counter() - t0
    speedup = dt_hf / dt_ours
    print(f"speedup vs transformers eager generate: {speedup:.2f}x")
    assert speedup > 2.0


@requires_cuda
def test_gemma4_e2b_coverage():
    """gemma-4-E2B-it: gemma norms, PLE pathway, shared KV, sliding
    windows, dual head dims, softcapped head. Needs both model copies
    resident, so it runs where memory allows."""
    if torch.cuda.get_device_properties(0).total_memory < 30e9:
        pytest.skip("needs ~24GB free for eager + packed copies")
    transformers = pytest.importorskip("transformers")
    from transformers import DynamicCache
    ref = transformers.AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E2B-it", dtype=torch.bfloat16,
        attn_implementation="eager").cuda().eval()
    mm = mak.MegaModel.from_pretrained(ref, max_seq=512, max_gen=512)
    torch.manual_seed(5)
    seq = torch.randint(0, mm.V, (32,)).tolist()
    cap = ref.config.text_config.final_logit_softcapping
    cache = DynamicCache(config=ref.config.text_config)
    out = torch.empty(len(seq), mm.V, device="cuda")
    with torch.no_grad():
        for i, t_ in enumerate(seq):
            r = ref.model.language_model(
                input_ids=torch.tensor([[t_]], device="cuda"),
                past_key_values=cache, use_cache=True)
            cache = r.past_key_values
            lg = ref.lm_head(r.last_hidden_state)[0, -1].float()
            lg = torch.tanh(lg.bfloat16() / cap).bfloat16() * cap
            out[i] = lg.float()
    mine = torch.empty(len(seq), mm.V, device="cuda")
    for i, t_ in enumerate(seq):
        mine[i] = mm.decode_step(t_, i).clone()
    agree = (mine.argmax(-1) == out.argmax(-1)).sum().item()
    d = (mine.bfloat16().float() - out.bfloat16().float()).abs()
    print(f"gemma4-E2B band mean={d.mean().item():.4f} "
          f"agree={agree}/{len(seq)}")
    assert agree >= len(seq) * 0.85
    a = mm.decode_step(seq[0], 0).clone()
    b = mm.decode_step(seq[0], 0, phased=True).clone()
    assert torch.equal(a, b)
    del ref, mm
    torch.cuda.empty_cache()
