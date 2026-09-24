import hashlib

import pytest
import torch

import kernels

di = kernels.get_kernel("phanerozoic/det-infer", version=1,
                        trust_remote_code=True)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                   reason="CUDA required")

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
PROMPT_LEN, TF_STEPS, GEN = 32, 64, 64

# Golden digests, minted on sm89. Equality anywhere else is the
# certificate: logits and greedy token streams are bit-identical across
# CUDA architectures, torch releases, and CUDA toolchains. The digest is
# the SHA-256 of the fp32 logits bytes over a teacher-forced sequence
# (prompt seed 0x5EED, continuation seed 0xBEEF), and of the greedy token
# list's string form.
GOLDEN = {
    "HuggingFaceTB/SmolLM2-135M": dict(
        plen=32, tf=64, gen=64, max_seq=2048,
        logits="c99a6ec1b2d8f0576af17a94af356f6812f0495b1b9d0b746049be77475d1c70",
        tokens="bcc724774ecd0200cd9d300d3a6cc58e9152c087c3caa86bc916818b25d06865",
        head=[282, 260, 216, 33, 41, 40, 32, 99]),
    "Qwen/Qwen3-0.6B": dict(
        plen=16, tf=48, gen=32, max_seq=512,
        logits="5c5e3dd0e23dce518298d82adcc86cb1b4588186cfec3aaaa4dd20362723ebe1",
        tokens="cf327eed3fcd01b6534e1b57f1606dc17bb1cb403c3c3af46e50c8d674cd1ec3",
        head=[271, 32313, 11, 1077, 594, 1490, 13, 576]),
    "meta-llama/Llama-3.2-1B": dict(
        plen=32, tf=64, gen=64, max_seq=2048,
        logits="689becac61ddedecd49d07663576458aae7a665ba25566f0706443a3cc693245",
        tokens="d48309fa1c111e24f2e7b4945addd2065db3aa40ce6500adff715a9840ed329c",
        head=[105428, 15606, 3566, 2816, 11, 3566, 2816, 11]),
    "meta-llama/Llama-3.2-3B": dict(
        plen=32, tf=64, gen=64, max_seq=2048,
        logits="42befa81f5d9959c91047f6a9509212c41cad67017ad1da4649022126185dbc9",
        tokens="972c81d9b02af1726475b398f60c072a7f907a7d18cfe525fc69810c1fa7dce8",
        head=[372, 89, 81923, 74694, 128001, 128000, 755, 1925]),
    # Mistral architecture; loaded on CPU so packing fits a 24GB card
    "HuggingFaceH4/zephyr-7b-beta": dict(
        plen=32, tf=64, gen=64, max_seq=2048, cpu_load=True,
        logits="849df574bb6aea93dd73fe0d286a9d6bf3c84933aefed89669698ee086e1a86c",
        tokens="aaaf19ff81c85fb4430f0c23b03f6d7d87668081823f8e1506efe0d12ecaacd5",
        head=[28705, 13, 13, 28789, 28766, 1838, 28766, 28767]),
    # gemma-4 architecture (PLE pathway, shared KV, sliding windows,
    # gelu gating, softcapped head); loaded on CPU so packing fits 24GB
    "google/gemma-4-E2B-it": dict(
        plen=32, tf=64, gen=64, max_seq=1024, cpu_load=True,
        logits="894fd22febb35606640fa6019efbb6d04c456fe22270fca9e1eb11b708d34d5b",
        tokens="d582a31f8fde70f4f3f79939ccf31451a7b330e703a95acdf5c11723bbb8d739",
        head=[237293, 237293, 237293, 237293, 237293, 237293, 237293,
              237293]),
}


def lcg_tokens(n, vocab, seed=0x5EED):
    x = seed & 0xFFFFFFFFFFFFFFFF
    out = []
    for _ in range(n):
        x = (x * 6364136223846793005 + 1442695040888963407) % 2**64
        out.append((x >> 33) % vocab)
    return out


def check_golden(model_id):
    transformers = pytest.importorskip("transformers")
    cfg = GOLDEN[model_id]
    ref = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="eager").eval()
    if not cfg.get("cpu_load"):
        ref = ref.cuda()
    mm = di.MegaModel.from_pretrained(ref, max_seq=cfg["max_seq"],
                                      max_gen=cfg["max_seq"])
    del ref
    torch.cuda.empty_cache()
    prompt = lcg_tokens(cfg["plen"], mm.V)
    seq = prompt + lcg_tokens(cfg["tf"], mm.V, seed=0xBEEF)
    h = hashlib.sha256()
    for i, t in enumerate(seq):
        h.update(mm.decode_step(t, i).cpu().numpy().tobytes())
    assert h.hexdigest() == cfg["logits"], f"{model_id} logits digest"
    g = mm.generate(prompt, cfg["gen"])
    assert g[:8] == cfg["head"], f"{model_id} token head"
    ht = hashlib.sha256()
    ht.update(bytes(str(g), "ascii"))
    assert ht.hexdigest() == cfg["tokens"], f"{model_id} token digest"
    del mm
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def models():
    transformers = pytest.importorskip("transformers")
    ref = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="eager"
    ).cuda().eval()
    mm = di.MegaModel.from_pretrained(ref, max_seq=2048, max_gen=2048)
    return ref, mm


@requires_cuda
def test_golden_smollm2():
    check_golden("HuggingFaceTB/SmolLM2-135M")


@requires_cuda
def test_golden_qwen3():
    """qk-norm det path and D=128 rope."""
    check_golden("Qwen/Qwen3-0.6B")


@requires_cuda
def test_golden_llama_1b():
    """llama3 rope scaling through the decimal recipe."""
    check_golden("meta-llama/Llama-3.2-1B")


@requires_cuda
def test_golden_llama_3b():
    check_golden("meta-llama/Llama-3.2-3B")


@requires_cuda
def test_golden_mistral_7b():
    """Mistral architecture (zephyr-7b-beta)."""
    check_golden("HuggingFaceH4/zephyr-7b-beta")


@requires_cuda
def test_bitwise_modes_and_determinism(models):
    _, mm = models
    a = mm.decode_step(123, 0).clone()
    b = mm.decode_step(123, 0, phased=True).clone()
    c = mm.decode_step(123, 0).clone()
    assert torch.equal(a, b)
    assert torch.equal(a, c)


@requires_cuda
def test_single_launch_equals_per_token(models):
    _, mm = models
    prompt = lcg_tokens(PROMPT_LEN, mm.V, seed=0xCAFE)
    g1 = mm.generate(prompt, GEN)
    g2 = mm.generate(prompt, GEN, single_launch=False)
    assert g1 == g2


@requires_cuda
def test_batch_invariance(models):
    """A sequence's logits and greedy stream are bitwise identical whether
    it decodes alone or inside a batch, and independent of batch order."""
    _, mm = models
    B = min(8, mm.batch_max())
    prompts = [lcg_tokens(12 + i, mm.V, seed=0x100 + i) for i in range(B)]
    solo = [mm.generate(p, 24) for p in prompts]
    batched = mm.generate_batch(prompts, 24)
    assert batched == solo
    order = list(range(B - 1, -1, -1))
    perm = mm.generate_batch([prompts[i] for i in order], 24)
    assert all(perm[j] == solo[order[j]] for j in range(B))
    mm.generate_batch(prompts, 2)
    lg = mm.batch_logits().clone()
    for i, p in enumerate(prompts):
        mm.generate_batch([p], 2)
        assert torch.equal(mm.batch_logits()[0], lg[i])


@requires_cuda
def test_transformers_parity_band(models):
    from transformers import DynamicCache
    ref, mm = models
    seq = lcg_tokens(96, mm.V, seed=0xD00D)
    cache = DynamicCache()
    out = torch.empty(len(seq), mm.V, device="cuda")
    with torch.no_grad():
        for i, t in enumerate(seq):
            r = ref(torch.tensor([[t]], device="cuda"),
                    past_key_values=cache, use_cache=True)
            cache = r.past_key_values
            out[i] = r.logits[0, -1].float()
    mine = torch.empty(len(seq), mm.V, device="cuda")
    for i, t in enumerate(seq):
        mine[i] = mm.decode_step(t, i).clone()
    d = (mine.bfloat16().float() - out.bfloat16().float()).abs()
    agree = (mine.argmax(-1) == out.argmax(-1)).sum().item()
    print(f"band mean={d.mean().item():.4f} max={d.max().item():.3f} "
          f"agree={agree}/{len(seq)}")
    assert agree >= len(seq) * 0.90
    assert d.mean().item() < 0.3


@requires_cuda
def test_golden_gemma4_e2b():
    """gemma-4 architecture."""
    check_golden("google/gemma-4-E2B-it")


@requires_cuda
def test_nf4_packed_matches_dequant():
    """A bitsandbytes nf4 checkpoint loads with its weights kept packed and
    dequantized in the kernel; the result is bitwise identical to running
    the same weights dequantized to bf16, and is deterministic. The kernel
    dequant is integer unpack plus an IEEE multiply, so this holds on every
    card."""
    bnb = pytest.importorskip("bitsandbytes")
    transformers = pytest.importorskip("transformers")
    from transformers import BitsAndBytesConfig
    mid = "HuggingFaceTB/SmolLM2-135M"
    try:
        q = transformers.AutoModelForCausalLM.from_pretrained(
            mid, quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False))
    except Exception as e:  # bnb has no kernels for this GPU / toolchain
        pytest.skip(f"bitsandbytes 4-bit unavailable here: {e}")
    mm = di.MegaModel.from_pretrained(q, max_seq=512, max_gen=64)
    assert mm._has_q4
    assert mm._layers[0]["wdown"]["packed"].dtype == torch.uint8
    ref = transformers.AutoModelForCausalLM.from_pretrained(
        mid, dtype=torch.bfloat16)
    for name, mod in q.named_modules():
        if isinstance(mod, bnb.nn.Linear4bit):
            w = bnb.functional.dequantize_4bit(mod.weight.data,
                                               mod.weight.quant_state)
            ref.get_submodule(name).weight.data = \
                w.to(torch.bfloat16).cpu().clone()
    mr = di.MegaModel.from_pretrained(ref.cuda().eval(), max_seq=512,
                                      max_gen=64)
    prompt = lcg_tokens(24, mm.V, seed=0x4B1D)
    assert torch.equal(mm.decode_step(prompt[0], 0),
                       mr.decode_step(prompt[0], 0))
    g = mm.generate(prompt, 32)
    assert g == mr.generate(prompt, 32)
    assert g == mm.generate(prompt, 32)
