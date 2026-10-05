"""
mkrun -- the frozen harness for a compiled megakernel.

Two jobs, and they are deliberately separate:

  1. THE GATE.  Is the megakernel computing this model?  The reference is
     HuggingFace `transformers` running the same checkpoint -- third-party code
     this compiler did not write and cannot influence.  The comparison is
     teacher-forced: both are fed the SAME fixed token sequence and their logits
     are compared step by step.  Free-running greedy decode is not a gate; one
     near-tied argmax on step 3 makes every later token differ and reports
     catastrophic failure for a 1-ulp difference.

  2. THE METRIC.  decode ms/token, from CUDA events around N launches with no
     host round trip, and cross-checked against wall clock.

The gate is not modifiable by the thing under test: it lives here, it loads its
own reference, and `guard.py` separately verifies that the engine cannot have
cheated its way past it.
"""
import argparse, ctypes, json, os, sys, time
import numpy as np

def _preload_reference_stack():
    """Import torch/transformers BEFORE dlopen'ing libmk.so.

    Loading the generated shared object first breaks torch's lazy module
    machinery (its CUDA runtime and ours end up fighting over process-global
    state).  Importing the reference stack first costs nothing and removes the
    whole class of problem.
    """
    import torch  # noqa: F401
    import transformers  # noqa: F401
    from transformers import AutoModelForCausalLM  # noqa: F401


class Engine:
    """ctypes binding to a compiled megakernel's C ABI."""
    def __init__(self, outdir, model_dir, device=0, max_len=4096):
        _preload_reference_stack()
        self.dir = outdir
        self.cfg = json.load(open(os.path.join(outdir, "build.json")))
        self.lib = ctypes.CDLL(os.path.join(outdir, "libmk.so"))
        L = self.lib
        L.mk_create.restype = ctypes.c_void_p
        L.mk_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        for n, r, a in [
            ("mk_free", None, [ctypes.c_void_p]),
            ("mk_reset", ctypes.c_int, [ctypes.c_void_p]),
            ("mk_set_tokens", ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]),
            ("mk_set_pos", ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]),
            ("mk_set_prompt_len", ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]),
            ("mk_run", ctypes.c_int, [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]),
            ("mk_logits", ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]),
            ("mk_tokens", ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]),
            ("mk_pos", ctypes.c_int, [ctypes.c_void_p]),
            ("mk_last_ms", ctypes.c_double, [ctypes.c_void_p]),
            ("mk_vocab", ctypes.c_int, [ctypes.c_void_p]),
            ("mk_grid", ctypes.c_int, [ctypes.c_void_p]),
            ("mk_regs", ctypes.c_int, [ctypes.c_void_p]),
            ("mk_smem", ctypes.c_longlong, [ctypes.c_void_p]),
            ("mk_maxctx", ctypes.c_int, [ctypes.c_void_p]),
            ("mk_hidden", ctypes.c_int, [ctypes.c_void_p]),
            ("mk_hidden_state", ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]),
            ("mk_arena_bytes", ctypes.c_longlong, [ctypes.c_void_p]),
            ("mk_poke", ctypes.c_int, [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_void_p, ctypes.c_longlong]),
            ("mk_peek", ctypes.c_int, [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_void_p, ctypes.c_longlong]),
            ("mk_nweights", ctypes.c_int, [ctypes.c_void_p]),
            ("mk_weight_role", ctypes.c_char_p, [ctypes.c_void_p, ctypes.c_int]),
            ("mk_weight_off", ctypes.c_longlong, [ctypes.c_void_p, ctypes.c_int]),
            ("mk_weight_bytes", ctypes.c_longlong, [ctypes.c_void_p, ctypes.c_int]),
            ("mk_bench_stage", ctypes.c_int, [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]),
            ("mk_nstages", ctypes.c_int, [ctypes.c_void_p]),
            ("mk_stage_name", ctypes.c_char_p, [ctypes.c_void_p, ctypes.c_int]),
            ("mk_rope_zero", ctypes.c_int, [ctypes.c_void_p]),
            ("mk_rope_rebuild", ctypes.c_int, [ctypes.c_void_p]),
            ("mk_zero_weight", ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]),
            ("mk_reload_weight", ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]),
        ]:
            f = getattr(L, n); f.restype = r; f.argtypes = a
        t0 = time.time()
        self.h = L.mk_create(model_dir.encode(), 0, max_len, device)
        if not self.h:
            raise RuntimeError("mk_create failed")
        self.load_s = time.time() - t0
        self.vocab = L.mk_vocab(self.h)
        self.maxctx = L.mk_maxctx(self.h)
        self._log = np.zeros(self.vocab, dtype=np.float32)

    def reset(self):            self.lib.mk_reset(self.h)
    def set_tokens(self, ids):
        a = np.ascontiguousarray(np.asarray(ids, dtype=np.int32))
        self.lib.mk_set_tokens(self.h, a.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), len(a))
    def set_pos(self, p):       self.lib.mk_set_pos(self.h, p)
    def set_prompt_len(self, n): self.lib.mk_set_prompt_len(self.h, n)
    def run(self, n, greedy=0): self.lib.mk_run(self.h, n, 1 if greedy else 0)
    def logits(self):
        self.lib.mk_logits(self.h, self._log.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return self._log
    def tokens(self, n):
        a = np.zeros(n, dtype=np.int32)
        self.lib.mk_tokens(self.h, a.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), n)
        return a
    def pos(self):              return self.lib.mk_pos(self.h)
    def last_ms(self):          return self.lib.mk_last_ms(self.h)
    def grid(self):             return self.lib.mk_grid(self.h)
    def regs(self):             return self.lib.mk_regs(self.h)
    def smem(self):             return self.lib.mk_smem(self.h)
    def weights(self):
        return [(self.lib.mk_weight_role(self.h, i).decode(),
                 self.lib.mk_weight_off(self.h, i),
                 self.lib.mk_weight_bytes(self.h, i))
                for i in range(self.lib.mk_nweights(self.h))]
    def nstages(self):          return self.lib.mk_nstages(self.h)
    def stage_name(self, i):    return self.lib.mk_stage_name(self.h, i).decode()
    def bench_stage(self, i, reps=200):
        us = ctypes.c_double(0.0)
        self.lib.mk_bench_stage(self.h, i, reps, ctypes.byref(us))
        return us.value
    def rope_zero(self):        return self.lib.mk_rope_zero(self.h)
    def rope_rebuild(self):     return self.lib.mk_rope_rebuild(self.h)
    def zero_weight(self, i):   return self.lib.mk_zero_weight(self.h, i)
    def reload_weight(self, i): return self.lib.mk_reload_weight(self.h, i)
    def peek(self, off, n):
        b = (ctypes.c_ubyte * n)()
        self.lib.mk_peek(self.h, off, b, n)
        return bytes(b)
    def poke(self, off, data):
        b = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
        return self.lib.mk_poke(self.h, off, b, len(data))
    def free(self):
        if self.h: self.lib.mk_free(self.h); self.h = None


def dequantized_reference_logits(model_dir, ids, device="cuda"):
    """Reference logits from a weight-quantised checkpoint, dequantised to bf16.

    Why this exists.  A w8a8 checkpoint means "quantise the weights AND the
    activations"; `transformers` does both, and mkc keeps activations in bf16
    (weight-only, w8a16).  Comparing against their path therefore measures the
    activation quantisation error -- ~20-40 bf16 ULPs -- which drowns out any
    real bug.

    So the gate uses two references.  This one dequantises the SAME stored
    weights with the format's own formula and runs the ordinary bf16 model, so
    the only remaining difference is reduction order and the bound is as tight as
    for an unquantised model.  The other (`reference_logits`) is fully
    independent and catches a wrong scale convention, which this one could not.
    """
    import torch, json, glob, os
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_dir)
    if hasattr(cfg, "quantization_config"):
        del cfg.quantization_config
    for sub in ("text_config",):
        if hasattr(cfg, sub) and hasattr(getattr(cfg, sub), "quantization_config"):
            delattr(getattr(cfg, sub), "quantization_config")

    sd = {}
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        sd.update(load_file(f))
    # The reference must run in the checkpoint's OWN activation dtype.  An AWQ
    # checkpoint keeps its norms and embeddings in fp16; running the reference in
    # bf16 changes the rounding at every layer and shows up as ~50 ULPs of
    # "error" that is entirely the harness's fault.
    emb = next((v for k, v in sd.items() if k.endswith("embed_tokens.weight")), None)
    adt = torch.float16 if (emb is not None and emb.dtype == torch.float16) else torch.bfloat16
    # AWQ: eight int4 values per int32, packed along the OUTPUT axis in the
    # order {0,2,4,6,1,3,5,7}.  Dequantised here with the format's own
    # definition, on the GPU and in int32 -- the obvious int64-on-CPU version
    # builds gigabytes of intermediates and gets OOM-killed.
    AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]

    def unpack_awq(qw, dev):                  # [n, cols/8] int32 -> [n, cols]
        x = qw.to(dev)
        nib = torch.stack([(x >> (4 * t)) & 0xF for t in range(8)], dim=-1)
        out = torch.empty_like(nib)
        for t, o in enumerate(AWQ_ORDER):
            out[..., o] = nib[..., t]
        return out.reshape(x.shape[0], -1)

    out = {}
    for k, v in sd.items():
        if k.endswith(".qweight"):
            base = k[: -len(".qweight")]
            q = unpack_awq(sd[base + ".qweight"], device).float()      # [K, rows]
            z = unpack_awq(sd[base + ".qzeros"], device).float()       # [K/G, rows]
            sc = sd[base + ".scales"].to(device).float()               # [K/G, rows]
            g = q.shape[0] // sc.shape[0]
            w = (q - z.repeat_interleave(g, 0)) * sc.repeat_interleave(g, 0)
            out[base + ".weight"] = w.t().contiguous().to(adt).cpu()
            del q, z, sc, w
            torch.cuda.empty_cache()
            continue
        if k.endswith(("_scale_inv", "_scale", "_scales", "_blocks",
                       ".qzeros", ".qweight", ".scales", ".g_idx")):
            continue
        if v.dtype == torch.float8_e4m3fn:
            # fp8: per-tensor, per-output-channel, or (bn x bk) block scales
            base = k[: -len(".weight")]
            sc = sd.get(base + ".weight_scale_inv", sd.get(base + ".weight_scale"))
            w = v.to(torch.float32)
            sc = sc.to(torch.float32)
            if sc.numel() == 1:
                w = w * sc
            elif sc.shape in ((w.shape[0], 1), (w.shape[0],)):
                w = w * sc.reshape(-1, 1)
            else:
                bn = (w.shape[0] + sc.shape[0] - 1) // sc.shape[0]
                bk = (w.shape[1] + sc.shape[1] - 1) // sc.shape[1]
                w = w * sc.repeat_interleave(bn, 0)[: w.shape[0]] \
                        .repeat_interleave(bk, 1)[:, : w.shape[1]]
            out[k] = w.to(adt)
        else:
            out[k] = v

    m = AutoModelForCausalLM.from_config(cfg, dtype=adt)
    missing, unexpected = m.load_state_dict(out, strict=False)
    tied = getattr(cfg, "tie_word_embeddings", False)
    hard = [k for k in missing if not (tied and "lm_head" in k)]
    if hard:
        raise RuntimeError(f"dequantised reference is missing {len(hard)} tensors, e.g. {hard[:3]}")
    m = m.to(device).eval()
    with torch.no_grad():
        lg = m(input_ids=torch.tensor([ids], dtype=torch.long, device=device)).logits[0]
    lg = lg.float().cpu().numpy()
    del m
    torch.cuda.empty_cache()
    return lg


def reference_logits(model_dir, ids, device="cuda"):
    """HuggingFace transformers, one forward over the whole sequence.

    This is the definition of correct.  It is third-party code, loaded from the
    same checkpoint the compiler read, and nothing in the generated engine can
    influence it.
    """
    import torch
    from transformers import AutoModelForCausalLM
    torch.manual_seed(0)
    # transformers 5.16 crashes loading a DENSE fp8 checkpoint: its fp8 quantiser
    # looks up `config._experts_implementation` in a mixture-of-experts table and
    # dereferences the None it gets back.  Give the table a None entry so the
    # lookup yields an empty override set -- which is what a dense model wants.
    # Setting the implementation instead routes a dense model into a MoE kernel
    # and faults.  Third-party bug, worked around here, not in the engine.
    try:
        from transformers.integrations.finegrained_fp8 import FP8Experts
        FP8Experts._impl_tp_layer_overrides.setdefault(None, {})
    except Exception:
        pass
    # Its DeepGEMM fp8 path faults on this driver; transformers ships an env var
    # to force the Triton reference kernel, which is the intended escape hatch.
    os.environ.setdefault("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR", "1")
    kwargs_cfg = {}
    # An MXFP4 checkpoint makes transformers fetch a Triton kernel package from
    # the hub, which fails on a compute node with no network -- and would be the
    # wrong reference anyway, since it quantises the activations too.  Ask for
    # the dequantised path explicitly: same stored weights, ordinary bf16
    # arithmetic, which is what this engine's output should be compared against.
    try:
        import json as _json
        cfg = _json.load(open(os.path.join(model_dir, "config.json")))
        if (cfg.get("quantization_config") or {}).get("quant_method") == "mxfp4":
            from transformers import Mxfp4Config
            kwargs_cfg["quantization_config"] = Mxfp4Config(dequantize=True)
    except Exception:
        pass

    last = None
    for impl in ("sdpa", "eager"):
        try:
            m = AutoModelForCausalLM.from_pretrained(
                model_dir, dtype=torch.bfloat16, device_map=device,
                attn_implementation=impl, **kwargs_cfg)
            break
        except Exception as e:      # attention sinks have no sdpa path
            last = e
    else:
        raise last
    m.eval()
    with torch.no_grad():
        out = m(input_ids=torch.tensor([ids], dtype=torch.long, device=device))
    lg = out.logits[0].float().cpu().numpy()
    del m, out
    torch.cuda.empty_cache()
    return lg


def gate(ref, got, verbose=True):
    """Teacher-forced logit comparison.

    Top-1 must agree at every step, with genuine ties tolerated -- a
    disagreement counts only if the reference's own top-2 gap was larger than
    the numerical noise we measured.  Plus a loose absolute bound, in units of
    the reference's own output resolution, to catch a truly broken kernel.
    A tight tolerance is wrong here: an independent reimplementation differs by
    about 1 bf16 ulp per layer and that accumulates honestly.
    """
    n, V = ref.shape
    d = np.abs(ref - got)
    peak = np.abs(ref).max(axis=1)
    ulp = peak * 2.0 ** -8           # bf16 resolution at the peak logit
    rel = d.max(axis=1) / np.maximum(ulp, 1e-9)
    r1 = ref.argmax(axis=1); g1 = got.argmax(axis=1)
    srt = np.sort(ref, axis=1)
    gap = srt[:, -1] - srt[:, -2]
    noise = d.max(axis=1)
    real_mismatch = (r1 != g1) & (gap > noise)
    ok_top1 = int(real_mismatch.sum()) == 0
    # correlation of the whole logit vector, which a partial computation fails
    cc = np.array([np.corrcoef(ref[i], got[i])[0, 1] for i in range(min(n, 32))])
    res = dict(
        steps=n,
        top1_agree=float((r1 == g1).mean()),
        real_mismatches=int(real_mismatch.sum()),
        max_abs_err=float(d.max()),
        max_err_in_ulps=float(rel.max()),
        mean_err_in_ulps=float(rel.mean()),
        min_corr=float(cc.min()),
        # Top-1 agreement at every step over a large vocabulary is the strong
        # test; the other two only need to catch a kernel that is broken outright.
        passed=bool(ok_top1 and rel.max() < 64.0 and cc.min() > 0.99),
    )
    if verbose:
        print(f"  steps                {res['steps']}")
        print(f"  top-1 agreement      {100*res['top1_agree']:.2f}%  "
              f"({res['real_mismatches']} genuine mismatches)")
        print(f"  max |dlogit|         {res['max_abs_err']:.4e}")
        print(f"  max err / bf16 ulp   {res['max_err_in_ulps']:.2f}   (mean {res['mean_err_in_ulps']:.2f})")
        print(f"  min logit corr       {res['min_corr']:.6f}")
        print(f"  GATE                 {'PASS' if res['passed'] else 'FAIL'}")
    return res
