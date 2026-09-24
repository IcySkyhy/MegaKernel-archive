"""model-as-a-kernel: an entire llama-family decode step as one kernel launch.

A phase interpreter runs the whole forward pass (embedding, every layer's
norms, QKV, rope, attention over the KV cache, SwiGLU MLP, the LM head, and
the greedy argmax) inside a single persistent kernel, with a software grid
barrier between phases. The in-kernel step loop makes an entire greedy
generation, prompt consumption included, one kernel launch: each step's
argmax feeds the next step's embedding read without leaving the device.
"""
import os
import struct
from typing import List, Optional, Sequence

import torch

try:
    from ._ops import ops
except ImportError:  # local development build (torch.utils.cpp_extension)
    class _LocalOps:
        def __getattr__(self, name):
            return getattr(torch.ops.mak_ext, name)

    ops = _LocalOps()

__all__ = ["MegaModel"]

_OP_EMBED, _OP_GEMV, _OP_QKV_POST, _OP_ATTN = 0, 1, 2, 3
_OP_ARGMAX_PART, _OP_ARGMAX_FIN = 4, 5
_OP_KV_APPEND, _OP_NORMRES, _OP_PLEMIX = 6, 7, 8
_OP_GEMV_PLAIN, _OP_NORMB, _OP_GLUB, _OP_ATTNFINB = 9, 10, 11, 12
_OP_GEMV_Q4 = 13
_IT_NONE, _IT_RMSNORM, _IT_SWIGLU, _IT_ATTNFIN = 0, 1, 2, 3
_IT_RMSNORM_G, _IT_GELU_GLU = 4, 5
_EP_GELU_PLE, _EP_F32_AMAX_CAP = 5, 6
_ITF_REDUCE, _ITF_WRITEBACK = 8, 16
_EP_STORE, _EP_RESID, _EP_F32, _EP_F32_AMAX, _EP_PARTIAL = 0, 1, 2, 3, 4
_NSLICE = 4
# Widest [B][K] bf16 panel a batched projection stages in shared before
# spilling to the global-scratch path. 16384 (32 KB) keeps two blocks per
# SM; one block per SM measures slower than reading the input from L2.
_STAGE_BUDGET = 16384


def _fbits(f: float) -> int:
    return struct.unpack("<i", struct.pack("<f", float(f)))[0]


def _row(op, p1=0, p2=0, p3=0, p4=0, p5=0, p6=0, n=0, k=0, it=0, epi=0,
         hq=0, hkv=0, d=0, f0=0.0, i0=0):
    return [int(op), int(p1), int(p2), int(p3), int(p4), int(p5), int(p6),
            int(n), int(k), int(it), int(epi), int(hq), int(hkv), int(d),
            _fbits(f0), int(i0)]


class MegaModel:
    """A packed llama-family decoder whose decode step is one kernel launch.

    Supported architecture: RMSNorm decoder blocks with rotary attention
    (GQA, optional Qwen3-style per-head qk RMSNorm), SwiGLU MLP, no attention
    or MLP biases, plain (unscaled) rope, batch size 1. This covers SmolLM2,
    TinyLlama, Qwen3 dense, Llama-class checkpoints without rope scaling,
    and similar models.
    """

    def __init__(self, weights: dict, config: dict, device="cuda",
                 max_seq: int = 4096, max_gen: int = 4096):
        cfg = dict(config)
        self.L = int(cfg["num_hidden_layers"])
        self.hidden = int(cfg["hidden_size"])
        self.Hq = int(cfg["num_attention_heads"])
        self.Hkv = int(cfg.get("num_key_value_heads") or self.Hq)
        self.D = int(cfg.get("head_dim") or self.hidden // self.Hq)
        self.I = int(cfg["intermediate_size"])
        self.V = int(cfg["vocab_size"])
        self.theta = float(cfg.get("rope_theta", 10000.0))
        self.eps = float(cfg.get("rms_norm_eps", 1e-6))
        self.qk_norm = bool(cfg.get("qk_norm", False))
        self.max_seq = int(max_seq)
        self.max_gen = int(max_gen)
        self.device = torch.device(device)

        if self.D % 2 or self.D > 256:
            raise ValueError("head_dim must be even and <= 256")
        if self.hidden % 8 or self.I % 8 or (self.Hq * self.D) % 8:
            raise ValueError(
                "hidden, intermediate, and Hq*head_dim must be multiples of 8")
        if self.Hq % self.Hkv:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        dt = torch.bfloat16
        dev = self.device

        def pack(t):
            t = t.detach().to(device=dev, dtype=dt).contiguous()
            self._keep.append(t)
            return t

        # Row-tile-of-8 weight layout [N/8][K/8][8][8] (MAK_TILED=1):
        # measured slower on every card (H200 +15%, RTX PRO +9%, Ada
        # +14%), so the default is the plain layout; the flag rides in
        # the program and both layouts compute identical bits.
        env_t = os.environ.get("MAK_TILED", "").strip()
        self.tiled = env_t == "1"

        def pack_tiled(t):
            t = t.detach().to(device=dev, dtype=dt).contiguous()
            if not self.tiled:
                self._keep.append(t)
                return t
            n, k = t.shape
            assert k % 8 == 0
            n8 = (n + 7) // 8 * 8
            if n8 != n:
                t = torch.cat(
                    [t, torch.zeros(n8 - n, k, dtype=dt, device=dev)], 0)
            t = t.view(n8 // 8, 8, k // 8, 8).permute(0, 2, 1, 3).contiguous()
            self._keep.append(t)
            return t

        self._keep: List[torch.Tensor] = []
        embed = pack(weights["embed"])
        final_norm = pack(weights["norm"])
        lm_head = weights["lm_head"]  # packed below (dense-tiled or nf4)
        inv_freq = weights.get("inv_freq")
        if inv_freq is None:
            d_idx = torch.arange(0, self.D, 2, dtype=torch.float32)
            inv_freq = 1.0 / (self.theta ** (d_idx / self.D))
        inv_freq = inv_freq.detach().to(device=dev,
                                        dtype=torch.float32).contiguous()
        if inv_freq.numel() != self.D // 2:
            raise ValueError("inv_freq length must be head_dim / 2")
        self._keep.append(inv_freq)
        self._invf = inv_freq

        qdim, kvdim = self.Hq * self.D, self.Hkv * self.D

        def _is_q4(w):
            return isinstance(w, dict) and w.get("q4")

        def pack_w(w):
            # nf4-packed weights move to the device intact; dense weights take
            # the usual (optionally row-tiled) bf16 packing.
            if _is_q4(w):
                pk = w["packed"].to(dev).contiguous()
                am = w["absmax"].to(device=dev, dtype=torch.float32).contiguous()
                self._keep.append(pk)
                self._keep.append(am)
                return {"q4": True, "packed": pk, "absmax": am,
                        "N": int(w["N"]), "K": int(w["K"])}
            return pack_tiled(w)

        def _wshape(w, expected):
            got = (int(w["N"]), int(w["K"])) if _is_q4(w) else tuple(w.shape)
            assert got == expected, (got, expected)

        self._has_q4 = False
        self._layers = []
        for lw in weights["layers"]:
            _wshape(lw["wqkv"], (qdim + 2 * kvdim, self.hidden))
            _wshape(lw["wo"], (self.hidden, qdim))
            _wshape(lw["wgu"], (2 * self.I, self.hidden))
            _wshape(lw["wdown"], (self.hidden, self.I))
            layer = {
                "wqkv": pack_w(lw["wqkv"]), "wo": pack_w(lw["wo"]),
                "wgu": pack_w(lw["wgu"]), "wdown": pack_w(lw["wdown"]),
                "ln1": pack(lw["ln1"]), "ln2": pack(lw["ln2"]),
                "qn": pack(lw["qn"]) if self.qk_norm else None,
                "kn": pack(lw["kn"]) if self.qk_norm else None,
            }
            self._has_q4 |= any(_is_q4(layer[n]) for n in
                                ("wqkv", "wo", "wgu", "wdown"))
            self._layers.append(layer)
        assert len(self._layers) == self.L
        assert embed.shape == (self.V, self.hidden)
        lm_head = pack_w(lm_head)
        self._has_q4 |= _is_q4(lm_head)
        if not _is_q4(lm_head):
            assert lm_head.numel() >= self.V * self.hidden  # row-tiled, padded
        self._w_embed, self._w_fnorm, self._w_lmhead = (embed, final_norm,
                                                        lm_head)
        self._qdim, self._kvdim = qdim, kvdim

        # Working buffers. Pointers to these are baked into the program, so
        # they (like the packed weights) live for the model's lifetime.
        self._maxk = max(self.hidden, qdim, self.I)
        # prefill chunk size, bounded by the staging shared-memory budget
        # (16K bf16 elements; the cp.async weight ring rides alongside)
        self._chunk_m = max(1, min(8, 16384 // self._maxk))
        cm = 8  # buffers sized for the maximum chunk
        self._hidden = torch.empty(cm * self.hidden, dtype=dt, device=dev)
        self._qkv = torch.empty(cm * (qdim + 2 * kvdim), dtype=dt, device=dev)
        self._gu = torch.empty(cm * 2 * self.I, dtype=dt, device=dev)
        self._logits = torch.empty(self.V, dtype=torch.float32, device=dev)
        # attention chunk length (kernel maximum 128); shorter chunks give
        # the attention phase more grid parallelism at the cost of more
        # softmax partials to finalize
        env_ch = os.environ.get("MAK_CHUNK", "").strip()
        self._chunk = int(env_ch) if env_ch in ("32", "64", "128") else 128
        self._maxch = (self.max_seq + self._chunk - 1) // self._chunk
        self._partials = torch.empty(
            cm * self.Hq * self._maxch * (self.D + 2), dtype=torch.float32,
            device=dev)
        self._kcache = torch.zeros(self.L, self.Hkv, self.max_seq, self.D,
                                   dtype=dt, device=dev)
        self._vcache = torch.zeros(self.L, self.Hkv, self.max_seq, self.D,
                                   dtype=dt, device=dev)
        self._token = torch.zeros(1, dtype=torch.int32, device=dev)
        self._prompt_in = torch.zeros(self.max_seq, dtype=torch.int32,
                                      device=dev)
        self._tokens_out = torch.zeros(self.max_gen, dtype=torch.int32,
                                       device=dev)
        self._bar = torch.zeros(34, dtype=torch.int32, device=dev)

        probe = torch.zeros(1, 16, dtype=torch.int64, device=dev)
        self._nblocks = int(ops.mak_num_blocks(probe, self._maxk))
        self._parts = torch.zeros(self._nblocks, dtype=torch.int64, device=dev)
        try:
            self._bw_per_sm = float(ops.mak_bw_per_sm(probe))
        except (AttributeError, RuntimeError):
            self._bw_per_sm = 0.0

        self._batch = 0
        self._pos_b = None
        self._kv_bstride = 0
        self._gemma = False
        # transformed-input scratch: the nf4 and wide-batch paths write the
        # dense bf16 input here (one row per active token) for the following
        # plain/quant GEMV; sized for a prefill chunk (up to 8 rows)
        self._xg = torch.empty(cm * self._maxk, dtype=dt, device=dev)
        self._build_programs()

    # ------------------------------------------------------------------
    def _kvp(self, cache, li: int, b: int) -> int:
        t = cache[li]
        return (t[b] if t.dim() == 4 else t).data_ptr()

    def _build_rows(self, kv_slice: int = 0, batch: bool = False,
                    big: bool = False):
        """Program rows for the llama-family path. kv_slice selects the
        batch slice the KV pointers address (per-sequence prefill);
        batch=True emits the batched-decode tail (per-row fp32 logits and
        explicit argmax phases) instead of the fused LM-head amax. In batch
        mode each projection stays fused (input staged in shared) when its
        [B][K] panel fits the budget; the projections whose K is too wide
        transform into a global scratch and read it with a plain GEMV."""
        _tbit = (1 << 20) if self.tiled else 0  # GEMV weight layout flag
        qdim, kvdim = self._qdim, self._kvdim
        embed, lm_head = self._w_embed, self._w_lmhead
        final_norm = self._w_fnorm
        xg = self._xg.data_ptr()
        B = self._batch if batch else 1
        rows, names = [], []
        staged_elems = [0]  # widest [B][K] panel any fused projection stages

        def fits(K):
            return B * K <= _STAGE_BUDGET

        def note(K):
            staged_elems[0] = max(staged_elems[0], B * K)

        def _q4(w):
            return isinstance(w, dict) and w.get("q4")

        def _wgemv(out, w, N, K, epi, resid):
            # the GEMV after a transform-to-scratch phase: nf4-dequant when
            # the weight is packed, otherwise a plain bf16 GEMV
            if _q4(w):
                rows.append(_row(_OP_GEMV_Q4, p1=xg, p2=w["packed"].data_ptr(),
                                 p3=out, p5=resid, p6=w["absmax"].data_ptr(),
                                 n=N, k=K, epi=epi))
            else:
                rows.append(_row(_OP_GEMV_PLAIN, p1=xg, p2=w.data_ptr(),
                                 p3=out, p5=resid, n=N, k=K, it=_tbit,
                                 epi=epi))

        def norm_gemv(name, gamma, w, out, N, K, epi, inp, resid=0,
                      variant=_IT_RMSNORM):
            if _q4(w) or (big and not fits(K)):
                rows.append(_row(_OP_NORMB, p1=inp, p2=gamma, p3=xg, k=K,
                                 it=variant, f0=self.eps))
                names.append(name + ".norm")
                _wgemv(out, w, N, K, epi, resid)
                names.append(name)
            else:
                note(K)
                rows.append(_row(_OP_GEMV, p1=inp, p2=w.data_ptr(), p3=out,
                                 p4=gamma, p5=resid, n=N, k=K,
                                 it=variant | _tbit, epi=epi, f0=self.eps))
                names.append(name)

        def glu_gemv(name, gu, w, out, resid, variant=_IT_SWIGLU):
            if _q4(w) or (big and not fits(self.I)):
                rows.append(_row(_OP_GLUB, p1=gu, p3=xg, k=self.I, it=variant))
                names.append(name + ".glu")
                _wgemv(out, w, self.hidden, self.I, _EP_RESID, resid)
                names.append(name)
            else:
                note(self.I)
                rows.append(_row(_OP_GEMV, p1=gu, p2=w.data_ptr(), p3=out,
                                 p5=resid, n=self.hidden, k=self.I,
                                 it=variant | _tbit, epi=_EP_RESID))
                names.append(name)

        def attn_gemv(name, w, out, resid):
            if _q4(w) or (big and not fits(qdim)):
                rows.append(_row(_OP_ATTNFINB, p1=self._partials.data_ptr(),
                                 p3=xg, k=qdim, hq=self.Hq,
                                 hkv=self._chunk << 16, d=self.D,
                                 i0=self._maxch))
                names.append(name + ".fin")
                _wgemv(out, w, self.hidden, qdim, _EP_RESID, resid)
                names.append(name)
            else:
                note(qdim)
                rows.append(_row(_OP_GEMV, p1=self._partials.data_ptr(),
                                 p2=w.data_ptr(), p3=out, p5=resid,
                                 n=self.hidden, k=qdim,
                                 it=_IT_ATTNFIN | _tbit, epi=_EP_RESID,
                                 hq=self.Hq, hkv=self._chunk << 16, d=self.D,
                                 i0=self._maxch))
                names.append(name)

        rows.append(_row(_OP_EMBED, p2=embed.data_ptr(),
                         p3=self._hidden.data_ptr(),
                         p4=self._parts.data_ptr(),
                         p5=self._prompt_in.data_ptr(),
                         p6=self._token.data_ptr(),
                         n=self._tokens_out.data_ptr(), k=self.hidden))
        names.append("embed")
        scale = 1.0 / (self.D ** 0.5)
        hid = self._hidden.data_ptr()
        for li, lw in enumerate(self._layers):
            kc = self._kvp(self._kcache, li, kv_slice)
            vc = self._kvp(self._vcache, li, kv_slice)
            norm_gemv(f"L{li}.qkv", lw["ln1"].data_ptr(),
                      lw["wqkv"], self._qkv.data_ptr(),
                      qdim + 2 * kvdim, self.hidden, _EP_STORE, hid)
            rows.append(_row(6, p1=self._qkv.data_ptr(), p2=kc, p3=vc,
                             p6=lw["kn"].data_ptr() if self.qk_norm else 0,
                             n=self.max_seq, k=1 if self.qk_norm else 0,
                             it=_fbits(self.eps),
                             epi=self._invf.data_ptr(),
                             hq=self.Hq, hkv=self.Hkv, d=self.D))
            names.append(f"L{li}.kvappend")
            rows.append(_row(_OP_ATTN, p1=self._qkv.data_ptr(), p2=kc,
                             p3=vc, p4=self._partials.data_ptr(),
                             p5=lw["qn"].data_ptr() if self.qk_norm else 0,
                             p6=lw["kn"].data_ptr() if self.qk_norm else 0,
                             n=self.max_seq, k=1 if self.qk_norm else 0,
                             it=_fbits(self.eps),
                             epi=self._invf.data_ptr(),
                             hq=self.Hq,
                             hkv=self.Hkv | (self._chunk << 16),
                             d=self.D, f0=scale, i0=self._maxch))
            names.append(f"L{li}.attn")
            attn_gemv(f"L{li}.o", lw["wo"], hid, hid)
            norm_gemv(f"L{li}.gateup", lw["ln2"].data_ptr(),
                      lw["wgu"], self._gu.data_ptr(), 2 * self.I,
                      self.hidden, _EP_STORE, hid)
            glu_gemv(f"L{li}.down", self._gu.data_ptr(), lw["wdown"], hid, hid)
        if batch or _q4(lm_head):
            norm_gemv("lm_head", final_norm.data_ptr(), lm_head,
                      self._logits.data_ptr(), self.V, self.hidden, _EP_F32,
                      hid)
            rows.append(_row(_OP_ARGMAX_PART, p1=self._logits.data_ptr(),
                             p3=self._parts.data_ptr(), n=self.V))
            names.append("argmax.part")
            rows.append(_row(_OP_ARGMAX_FIN, p1=self._parts.data_ptr(),
                             p3=self._token.data_ptr(),
                             p4=self._tokens_out.data_ptr(),
                             k=self.max_gen))
            names.append("argmax.fin")
        else:
            rows.append(_row(_OP_GEMV, p1=hid, p2=lm_head.data_ptr(),
                             p3=self._logits.data_ptr(),
                             p4=final_norm.data_ptr(),
                             p6=self._parts.data_ptr(), n=self.V,
                             k=self.hidden, it=_IT_RMSNORM | _tbit,
                             epi=_EP_F32_AMAX, hkv=1, f0=self.eps))
            names.append("lm_head")
            rows.append(_row(_OP_ARGMAX_FIN, p1=self._parts.data_ptr(),
                             p3=self._token.data_ptr(),
                             p4=self._tokens_out.data_ptr()))
            names.append("argmax.fin")
        se = max(staged_elems[0], self._maxk)
        return rows, names, se

    def _build_programs(self):
        dev = self.device
        rows, names, _ = self._build_rows(kv_slice=0, batch=False)
        self._prog = torch.tensor(rows, dtype=torch.int64, device=dev)
        self.phase_names = names
        if self._batch:
            big = self._batch > min(8, 16384 // self._maxk)
            rows, _, se = self._build_rows(kv_slice=0, batch=True, big=big)
            self._prog_batch = torch.tensor(rows, dtype=torch.int64,
                                            device=dev)
            self._batch_stage_elems = int(se)
            self._progs_prefill = [self._prog]
            for b in range(1, self._batch):
                rows, _, _ = self._build_rows(kv_slice=b, batch=False)
                self._progs_prefill.append(
                    torch.tensor(rows, dtype=torch.int64, device=dev))

    def batch_max(self) -> int:
        """Largest batch size this model supports. Up to 8 sequences stage
        in shared; beyond that the transformed input moves to a global
        scratch, bounded by the kernel's register accumulators."""
        try:
            hard = int(ops.mak_batch_maxb())
        except (AttributeError, RuntimeError):
            hard = 8
        return 1 if self._gemma else hard

    # ------------------------------------------------------------------
    def enable_batch(self, B: int):
        """Allocate per-sequence KV caches, token buffers, and programs for
        batched decode. B up to 8 stages the input in shared; larger B (up
        to batch_max()) routes the transformed input through a global
        scratch so the batch width is not bounded by the projection size."""
        if self._gemma:
            raise ValueError("batched decode covers the llama-family path")
        maxb = self.batch_max()
        if not 1 <= B <= maxb:
            raise ValueError(f"B must be in [1, {maxb}] for this model")
        if self._batch == B:
            return
        dt, dev = torch.bfloat16, self.device
        self._kcache = torch.zeros(self.L, B, self.Hkv, self.max_seq,
                                   self.D, dtype=dt, device=dev)
        self._vcache = torch.zeros_like(self._kcache)
        self._token = torch.zeros(B, dtype=torch.int32, device=dev)
        self._tokens_out = torch.zeros(B * self.max_gen, dtype=torch.int32,
                                       device=dev)
        self._logits = torch.empty(B * self.V, dtype=torch.float32,
                                   device=dev)
        self._parts = torch.zeros(B * self._nblocks, dtype=torch.int64,
                                  device=dev)
        self._pos_b = torch.zeros(B, dtype=torch.int32, device=dev)
        # transformed-input scratch: one row per active token, and prefill
        # (during generate_batch) still uses up to chunk_m rows
        self._xg = torch.empty(max(8, B) * self._maxk, dtype=dt, device=dev)
        # working buffers hold one row per active sequence; prefill chunks
        # still use up to chunk_m rows, so keep at least the fused capacity
        nb = max(8, B)
        qkvd = self._qdim + 2 * self._kvdim
        self._hidden = torch.empty(nb * self.hidden, dtype=dt, device=dev)
        self._qkv = torch.empty(nb * qkvd, dtype=dt, device=dev)
        self._gu = torch.empty(nb * 2 * self.I, dtype=dt, device=dev)
        self._partials = torch.empty(
            nb * self.Hq * self._maxch * (self.D + 2), dtype=torch.float32,
            device=dev)
        self._kv_bstride = self.Hkv * self.max_seq * self.D
        self._batch = B
        self._build_programs()

    def decode_batch(self, tokens: Sequence[int], positions: Sequence[int],
                     steps: int = 1) -> torch.Tensor:
        """Batched greedy decode: sequence b consumes tokens[b] at
        positions[b]; `steps` in-kernel steps run with per-sequence argmax
        feedback. Returns the generated tokens [B, steps]; the final
        step's fp32 logits are live in batch_logits()."""
        B = self._batch
        if B < 1:
            raise ValueError("call enable_batch(B) first")
        if len(tokens) != B or len(positions) != B:
            raise ValueError("tokens and positions must have B entries")
        if steps < 1 or steps > self.max_gen:
            raise ValueError("steps out of range")
        if max(positions) + steps > self.max_seq:
            raise ValueError("sequence exceeds max_seq")
        self._token.copy_(torch.tensor(list(tokens), dtype=torch.int32))
        self._pos_b.copy_(torch.tensor(list(positions), dtype=torch.int32))
        ops.mak_run_batch(self._prog_batch, self._bar, self._pos_b, steps,
                          0, self._maxk, self._kv_bstride,
                          self._batch_stage_elems)
        return self._tokens_out.view(B, self.max_gen)[:, :steps]

    def batch_logits(self) -> torch.Tensor:
        """fp32 logits [B, V] of the most recent batched step."""
        return self._logits.view(max(self._batch, 1), self.V)

    def generate_batch(self, prompts: Sequence[Sequence[int]],
                       max_new: int) -> List[List[int]]:
        """Greedy generation for a batch of prompts: per-sequence prefill,
        then one batched decode launch. Each sequence's token stream is
        bit-identical to its own single-sequence generate()."""
        B = len(prompts)
        if max_new < 1 or max_new > self.max_gen:
            raise ValueError("max_new out of range")
        self.enable_batch(B)
        firsts: List[int] = []
        for b, ids in enumerate(prompts):
            P = len(ids)
            if P < 1 or P + max_new > self.max_seq:
                raise ValueError("sequence exceeds max_seq")
            self._prompt_in[:P].copy_(
                torch.tensor(list(ids), dtype=torch.int32))
            ops.mak_run_seq(self._progs_prefill[b], self._bar, 0, P, 1 - P,
                            P, self._maxk, self._chunk_m)
            firsts.append(int(self._token[0].item()))
        if max_new == 1:
            return [[t] for t in firsts]
        self._token.copy_(torch.tensor(firsts, dtype=torch.int32))
        self._pos_b.copy_(torch.tensor([len(p) for p in prompts],
                                       dtype=torch.int32))
        ops.mak_run_batch(self._prog_batch, self._bar, self._pos_b,
                          max_new - 1, 0, self._maxk, self._kv_bstride,
                          self._batch_stage_elems)
        rest = (self._tokens_out.view(B, self.max_gen)[:, :max_new - 1]
                .cpu().tolist())
        return [[firsts[b]] + rest[b] for b in range(B)]

    # ------------------------------------------------------------------
    @staticmethod
    def _dense_weight(mod):
        """The effective bf16 weight of a linear layer, dequantizing
        bitsandbytes 4-bit and 8-bit layers. Dequantization is exact and
        card-independent; weights materialize to bf16, so the quantized
        footprint is not preserved (that needs the in-kernel packed path)."""
        w = mod.weight
        qs = getattr(w, "quant_state", None)
        if qs is not None:  # bitsandbytes 4-bit (nf4 / fp4)
            import bitsandbytes as bnb
            return bnb.functional.dequantize_4bit(w.data, qs).to(torch.bfloat16)
        if getattr(w, "SCB", None) is not None or hasattr(mod, "SCB"):
            # bitsandbytes 8-bit (LLM.int8): row scales in SCB, int8 in CB
            scb = w.SCB if getattr(w, "SCB", None) is not None else mod.SCB
            cb = w.data if w.data.dtype == torch.int8 else mod.CB
            return (cb.to(torch.float32)
                    * (scb.to(torch.float32) / 127.0).unsqueeze(1)
                    ).to(torch.bfloat16)
        return w

    @staticmethod
    def _q4_parts(mod):
        """(packed [N, K/2] uint8, absmax [N*K/64] fp32) for a bitsandbytes
        nf4 layer whose weights stay packed, else None. Double-quantized
        absmax is materialized to fp32 so the kernel needs one scale array."""
        w = getattr(mod, "weight", None)
        qs = getattr(w, "quant_state", None)
        if qs is None or getattr(qs, "quant_type", None) != "nf4":
            return None
        import bitsandbytes as bnb
        n, k = int(qs.shape[0]), int(qs.shape[1])
        if k % 64:
            return None  # kernel assumes 64 | K for per-block absmax
        packed = w.data.reshape(n, k // 2).contiguous()
        if getattr(qs, "nested", False):
            absmax = bnb.functional.dequantize_blockwise(
                qs.absmax, qs.state2) + qs.offset
        else:
            absmax = qs.absmax
        return packed, absmax.float().reshape(-1).contiguous()

    @classmethod
    def _proj(cls, mods):
        """A projection, concatenated across `mods` (q/k/v or gate/up). Stays
        nf4-packed when every part is nf4, else a dense bf16 tensor."""
        parts = [cls._q4_parts(m) for m in mods]
        if all(p is not None for p in parts):
            packed = torch.cat([p[0] for p in parts], 0)
            absmax = torch.cat([p[1] for p in parts], 0)
            return {"q4": True, "packed": packed, "absmax": absmax,
                    "N": packed.shape[0], "K": packed.shape[1] * 2}
        return torch.cat([cls._dense_weight(m) for m in mods], 0)

    @classmethod
    def from_pretrained(cls, model, device="cuda", max_seq: int = 4096,
                        max_gen: int = 4096):
        """Build from a transformers model (object or repo id). A quantized
        checkpoint (bitsandbytes 4-bit/8-bit) is accepted directly; its
        weights are dequantized to bf16 at load."""
        if isinstance(model, str):
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model, torch_dtype=torch.bfloat16)
        if "gemma4" in getattr(model.config, "model_type", ""):
            return cls._from_gemma4(model, device, max_seq, max_gen)
        hf = model.config
        dec = model.model
        attn0 = dec.layers[0].self_attn
        qk_norm = hasattr(attn0, "q_norm") and attn0.q_norm is not None
        if getattr(attn0.q_proj, "bias", None) is not None:
            raise ValueError("attention biases are not supported")

        # rope config across transformers API generations: v5 keeps theta and
        # type in a rope_parameters dict; v4 used a rope_theta attribute and
        # an optional rope_scaling dict.
        rp = getattr(hf, "rope_parameters", None)
        if not isinstance(rp, dict):
            rs = getattr(hf, "rope_scaling", None)
            rp = rs if isinstance(rs, dict) else {}
        theta = rp.get("rope_theta")
        if theta is None:
            theta = getattr(hf, "rope_theta", 10000.0)
        rope_type = rp.get("rope_type", rp.get("type", "default"))
        if rope_type not in ("default", "llama3"):
            raise ValueError(f"rope type {rope_type!r} is not supported")
        # take the frequency table transformers computed (scaling included)
        rot = dec.rotary_emb
        att_scale = float(getattr(rot, "attention_scaling", 1.0))
        if att_scale != 1.0:
            raise ValueError("rope attention_scaling != 1 is not supported")
        inv_freq = rot.inv_freq.detach().float()

        pj = cls._proj
        layers = []
        for lyr in dec.layers:
            a, m = lyr.self_attn, lyr.mlp
            layers.append({
                "wqkv": pj([a.q_proj, a.k_proj, a.v_proj]),
                "wo": pj([a.o_proj]),
                "wgu": pj([m.gate_proj, m.up_proj]),
                "wdown": pj([m.down_proj]),
                "ln1": lyr.input_layernorm.weight,
                "ln2": lyr.post_attention_layernorm.weight,
                "qn": a.q_norm.weight if qk_norm else None,
                "kn": a.k_norm.weight if qk_norm else None,
            })
        weights = {"embed": dec.embed_tokens.weight, "norm": dec.norm.weight,
                   "lm_head": pj([model.lm_head]), "layers": layers,
                   "inv_freq": inv_freq}
        config = {
            "num_hidden_layers": hf.num_hidden_layers,
            "hidden_size": hf.hidden_size,
            "num_attention_heads": hf.num_attention_heads,
            "num_key_value_heads": getattr(hf, "num_key_value_heads", None),
            "head_dim": getattr(hf, "head_dim", None),
            "intermediate_size": hf.intermediate_size,
            "vocab_size": hf.vocab_size,
            "rope_theta": float(theta),
            "rms_norm_eps": hf.rms_norm_eps,
            "qk_norm": qk_norm,
        }
        return cls(weights, config, device=device, max_seq=max_seq,
                   max_gen=max_gen)

    # ------------------------------------------------------------------
    @classmethod
    def _from_gemma4(cls, model, device, max_seq, max_gen):
        """gemma-4-E2B-it program: gemma norms (fp32, raw weight),
        gelu-tanh gating, post-sublayer norm phases, per-layer-input
        pathway, sliding windows, shared KV, dual head dims, softcapped
        head. Scope is the E2B text model only."""
        self = cls.__new__(cls)
        cfg = model.config.text_config if hasattr(model.config, "text_config") \
            else model.config
        dec = model.model.language_model if hasattr(model.model, "language_model") \
            else model.model
        assert cfg.model_type == "gemma4_text", cfg.model_type
        assert not cfg.enable_moe_block

        self.L = int(cfg.num_hidden_layers)
        self.hidden = int(cfg.hidden_size)
        self.Hq = int(cfg.num_attention_heads)
        self.V = int(cfg.vocab_size)
        self.eps = float(cfg.rms_norm_eps)
        self.qk_norm = True
        self.max_seq = int(max_seq)
        self.max_gen = int(max_gen)
        self.device = torch.device(device)
        self.tiled = False
        win = int(cfg.sliding_window)
        cap = float(cfg.final_logit_softcapping)
        PL = int(cfg.hidden_size_per_layer_input)
        dt = torch.bfloat16
        dev = self.device
        self._keep = []

        def pack(t):
            t = t.detach().to(device=dev, dtype=dt).contiguous()
            self._keep.append(t)
            return t

        def packf(t):
            t = t.detach().to(device=dev, dtype=torch.float32).contiguous()
            self._keep.append(t)
            return t

        embed = pack(dec.embed_tokens.weight)
        ple_tab = pack(dec.embed_tokens_per_layer.weight)
        plm_w = pack(dec.per_layer_model_projection.weight)
        pln_w = pack(dec.per_layer_projection_norm.weight)
        fnorm = pack(dec.norm.weight)
        rot = dec.rotary_emb
        invf = {lt: packf(getattr(rot, f"{lt}_inv_freq").float())
                for lt in set(cfg.layer_types)}
        ascale = {lt: float(getattr(rot, f"{lt}_attention_scaling"))
                  for lt in set(cfg.layer_types)}

        layers = []
        for li, lyr in enumerate(dec.layers):
            a = lyr.self_attn
            D_l = int(a.head_dim)
            shared = bool(a.is_kv_shared_layer)
            ent = {
                "D": D_l, "shared": shared,
                "lscale": float(lyr.layer_scalar.float().item()),
                "type": cfg.layer_types[li],
                "I": int(lyr.mlp.intermediate_size),
                "qn": pack(a.q_norm.weight),
                "o": pack(a.o_proj.weight),
                "ln_in": pack(lyr.input_layernorm.weight),
                "ln_pa": pack(lyr.post_attention_layernorm.weight),
                "ln_pf": pack(lyr.pre_feedforward_layernorm.weight),
                "ln_ff": pack(lyr.post_feedforward_layernorm.weight),
                "ln_pl": pack(lyr.post_per_layer_input_norm.weight),
                "plig": pack(lyr.per_layer_input_gate.weight),
                "plpr": pack(lyr.per_layer_projection.weight),
                "wgu": pack(torch.cat([lyr.mlp.gate_proj.weight,
                                       lyr.mlp.up_proj.weight], 0)),
                "wdn": pack(lyr.mlp.down_proj.weight),
            }
            if shared:
                ent["wqkv"] = pack(a.q_proj.weight)
                ent["kn"] = None
            else:
                ent["wqkv"] = pack(torch.cat(
                    [a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], 0))
                ent["kn"] = pack(a.k_norm.weight)
                ent["kc"] = torch.zeros(self.max_seq, D_l, dtype=dt,
                                        device=dev)
                ent["vc"] = torch.zeros(self.max_seq, D_l, dtype=dt,
                                        device=dev)
                self._keep += [ent["kc"], ent["vc"]]
            layers.append(ent)
        # shared layers read the last non-shared layer of their type
        src = {}
        for li, ent in enumerate(layers):
            if not ent["shared"]:
                src[ent["type"]] = ent
        for ent in layers:
            if ent["shared"]:
                ent["kc"] = src[ent["type"]]["kc"]
                ent["vc"] = src[ent["type"]]["vc"]

        maxD = max(e["D"] for e in layers)
        maxI = max(e["I"] for e in layers)
        self._maxk = max(self.hidden, self.Hq * maxD, maxI)
        self._chunk_m = 1
        self._chunk = 128
        self._maxch = (self.max_seq + self._chunk - 1) // self._chunk
        self._hidden = torch.empty(self.hidden, dtype=dt, device=dev)
        self._tmp = torch.empty(self.hidden, dtype=dt, device=dev)
        self._qkv = torch.empty((self.Hq + 2) * maxD, dtype=dt, device=dev)
        self._gu = torch.empty(2 * maxI, dtype=dt, device=dev)
        self._t256 = torch.empty(PL, dtype=dt, device=dev)
        self._ctxraw = torch.empty(self.L * PL, dtype=dt, device=dev)
        self._pletok = torch.empty(self.L * PL, dtype=dt, device=dev)
        self._ple = torch.empty(self.L * PL, dtype=dt, device=dev)
        self._logits = torch.empty(self.V, dtype=torch.float32, device=dev)
        self._partials = torch.empty(
            self.Hq * self._maxch * (maxD + 2), dtype=torch.float32,
            device=dev)
        self._token = torch.zeros(1, dtype=torch.int32, device=dev)
        self._prompt_in = torch.zeros(self.max_seq, dtype=torch.int32,
                                      device=dev)
        self._tokens_out = torch.zeros(self.max_gen, dtype=torch.int32,
                                       device=dev)
        self._bar = torch.zeros(34, dtype=torch.int32, device=dev)
        probe = torch.zeros(1, 16, dtype=torch.int64, device=dev)
        self._nblocks = int(ops.mak_num_blocks(probe, self._maxk))
        self._parts = torch.zeros(self._nblocks, dtype=torch.int64,
                                  device=dev)
        try:
            self._bw_per_sm = float(ops.mak_bw_per_sm(probe))
        except (AttributeError, RuntimeError):
            self._bw_per_sm = 0.0
        self._invf = invf[layers[0]["type"]]
        self.Hkv, self.D, self.I = 1, maxD, maxI
        self.theta = 0.0

        emb_scale = float(torch.tensor(self.hidden ** 0.5,
                                       dtype=torch.float32).to(dt))
        ple_scale = float(torch.tensor(PL ** 0.5,
                                       dtype=torch.float32).to(dt))
        proj_scale = float(torch.tensor(self.hidden ** -0.5,
                                        dtype=torch.float32))
        hid = self._hidden.data_ptr()
        tmp = self._tmp.data_ptr()
        rows, names = [], []
        rows.append(_row(_OP_EMBED, p1=ple_tab.data_ptr(),
                         p2=embed.data_ptr(), p3=hid,
                         p4=self._parts.data_ptr(),
                         p5=self._prompt_in.data_ptr(),
                         p6=self._token.data_ptr(),
                         n=self._tokens_out.data_ptr(), k=self.hidden,
                         it=self._pletok.data_ptr(),
                         hkv=self.L * PL, f0=emb_scale,
                         i0=_fbits(ple_scale)))
        names.append("embed")
        rows.append(_row(_OP_GEMV, p1=hid, p2=plm_w.data_ptr(),
                         p3=self._ctxraw.data_ptr(), n=self.L * PL,
                         k=self.hidden, it=_IT_NONE, epi=_EP_STORE))
        names.append("plm")
        rows.append(_row(8, p1=self._ctxraw.data_ptr(),
                         p2=pln_w.data_ptr(), p3=self._ple.data_ptr(),
                         p4=self._pletok.data_ptr(), n=self.L, k=PL,
                         f0=self.eps, i0=_fbits(proj_scale)))
        names.append("plemix")

        for li, e in enumerate(layers):
            D_l, qdim = e["D"], self.Hq * e["D"]
            lt = e["type"]
            w_l = win if lt == "sliding_attention" else 0
            asb = _fbits(ascale[lt]) if ascale[lt] != 1.0 else 0
            flags = 1 | 2 | (0 if e["shared"] else 4) | \
                (8 if e["shared"] else 0)
            n_qkv = qdim if e["shared"] else qdim + 2 * D_l
            rows.append(_row(_OP_GEMV, p1=hid, p2=e["wqkv"].data_ptr(),
                             p3=self._qkv.data_ptr(),
                             p4=e["ln_in"].data_ptr(), n=n_qkv,
                             k=self.hidden, it=_IT_RMSNORM_G,
                             epi=_EP_STORE, f0=self.eps))
            names.append(f"L{li}.qkv")
            rows.append(_row(_OP_ATTN, p1=self._qkv.data_ptr(),
                             p2=e["kc"].data_ptr(), p3=e["vc"].data_ptr(),
                             p4=self._partials.data_ptr(),
                             p5=e["qn"].data_ptr(),
                             p6=e["kn"].data_ptr() if e["kn"] is not None
                             else 0,
                             n=self.max_seq,
                             k=flags | (w_l << 16), it=_fbits(self.eps),
                             epi=invf[lt].data_ptr(), hq=self.Hq,
                             hkv=1 | (self._chunk << 16), d=D_l, f0=1.0,
                             i0=self._maxch | (asb << 32)))
            names.append(f"L{li}.attn")
            rows.append(_row(_OP_GEMV, p1=self._partials.data_ptr(),
                             p2=e["o"].data_ptr(), p3=tmp,
                             n=self.hidden, k=qdim, it=_IT_ATTNFIN,
                             epi=_EP_STORE, hq=self.Hq,
                             hkv=self._chunk << 16, d=D_l,
                             i0=self._maxch | (w_l << 16)))
            names.append(f"L{li}.o")
            rows.append(_row(7, p1=tmp, p2=e["ln_pa"].data_ptr(), p3=hid,
                             n=self.hidden, f0=self.eps, i0=_fbits(1.0)))
            names.append(f"L{li}.nr_attn")
            rows.append(_row(_OP_GEMV, p1=hid, p2=e["wgu"].data_ptr(),
                             p3=self._gu.data_ptr(),
                             p4=e["ln_pf"].data_ptr(), n=2 * e["I"],
                             k=self.hidden, it=_IT_RMSNORM_G,
                             epi=_EP_STORE, f0=self.eps))
            names.append(f"L{li}.gateup")
            rows.append(_row(_OP_GEMV, p1=self._gu.data_ptr(),
                             p2=e["wdn"].data_ptr(), p3=tmp,
                             n=self.hidden, k=e["I"], it=_IT_GELU_GLU,
                             epi=_EP_STORE))
            names.append(f"L{li}.down")
            rows.append(_row(7, p1=tmp, p2=e["ln_ff"].data_ptr(), p3=hid,
                             n=self.hidden, f0=self.eps, i0=_fbits(1.0)))
            names.append(f"L{li}.nr_ffw")
            rows.append(_row(_OP_GEMV, p1=hid, p2=e["plig"].data_ptr(),
                             p3=self._t256.data_ptr(),
                             p6=self._ple.data_ptr() + li * PL * 2,
                             n=PL, k=self.hidden, it=_IT_NONE,
                             epi=_EP_GELU_PLE))
            names.append(f"L{li}.plig")
            rows.append(_row(_OP_GEMV, p1=self._t256.data_ptr(),
                             p2=e["plpr"].data_ptr(), p3=tmp,
                             n=self.hidden, k=PL, it=_IT_NONE,
                             epi=_EP_STORE))
            names.append(f"L{li}.plproj")
            rows.append(_row(7, p1=tmp, p2=e["ln_pl"].data_ptr(), p3=hid,
                             n=self.hidden, f0=self.eps,
                             i0=_fbits(e["lscale"])))
            names.append(f"L{li}.nr_ple")

        rows.append(_row(_OP_GEMV, p1=hid, p2=embed.data_ptr(),
                         p3=self._logits.data_ptr(),
                         p4=fnorm.data_ptr(),
                         p6=self._parts.data_ptr(), n=self.V,
                         k=self.hidden, it=_IT_RMSNORM_G,
                         epi=_EP_F32_AMAX_CAP, hkv=1, f0=self.eps,
                         i0=_fbits(cap)))
        names.append("lm_head")
        rows.append(_row(_OP_ARGMAX_FIN, p1=self._parts.data_ptr(),
                         p3=self._token.data_ptr(),
                         p4=self._tokens_out.data_ptr()))
        names.append("argmax.fin")
        self._prog = torch.tensor(rows, dtype=torch.int64, device=dev)
        self.phase_names = names
        self._gemma = True
        self._batch = 0
        return self

    # ------------------------------------------------------------------
    def decode_step(self, token: int, pos: int, phased: bool = False):
        """One decode step; returns the live fp32 logits buffer [V]."""
        if pos >= self.max_seq:
            raise ValueError("pos exceeds max_seq")
        self._token.fill_(int(token))
        if phased:
            ops.mak_run_phased(self._prog, pos, self.max_gen - 1, False,
                               self._maxk)
        else:
            ops.mak_run(self._prog, self._bar, pos, self.max_gen - 1,
                        self._maxk)
        return self._logits[:self.V]

    def decode_step_timed(self, token: int, pos: int):
        """One phase-per-launch step; returns (logits, per-phase ms)."""
        self._token.fill_(int(token))
        ms = ops.mak_run_phased(self._prog, pos, self.max_gen - 1, True,
                                self._maxk)
        return self._logits[:self.V], ms

    def prefill(self, ids: Sequence[int], phased: bool = False):
        """Consume a prompt; returns the last position's logits. Runs in
        chunks of up to chunk_m tokens with weight reads amortized across
        the chunk (bitwise identical to token-by-token consumption)."""
        P = len(ids)
        if phased:
            for i, t in enumerate(ids):
                self.decode_step(int(t), i, phased=True)
            return self._logits
        self._prompt_in[:P].copy_(torch.tensor(ids, dtype=torch.int32))
        ops.mak_run_seq(self._prog, self._bar, 0, P, 1 - P, P, self._maxk,
                        self._chunk_m)
        return self._logits[:self.V]

    def generate(self, prompt_ids: Sequence[int], max_new: int,
                 single_launch: bool = True) -> List[int]:
        """Greedy generation. With single_launch (default) the entire call,
        prompt consumption included, is one kernel launch: the in-kernel step
        loop reads prompt tokens from a staged device buffer, then feeds each
        step's argmax to the next step's embedding read. With
        single_launch=False the same computation runs as one launch per
        token; the two paths produce identical tokens.
        """
        P = len(prompt_ids)
        if max_new < 1 or max_new > self.max_gen:
            raise ValueError("max_new out of range")
        if P < 1 or P + max_new > self.max_seq:
            raise ValueError("sequence exceeds max_seq")
        if single_launch:
            self._prompt_in[:P].copy_(
                torch.tensor(prompt_ids, dtype=torch.int32))
            ops.mak_run_seq(self._prog, self._bar, 0, P + max_new - 1,
                            1 - P, P, self._maxk, self._chunk_m)
            return self._tokens_out[:max_new].cpu().tolist()
        pos = 0
        for t in prompt_ids[:-1]:
            self._token.fill_(int(t))
            ops.mak_run(self._prog, self._bar, pos, self.max_gen - 1,
                        self._maxk)
            pos += 1
        self._token.fill_(int(prompt_ids[-1]))
        ops.mak_run(self._prog, self._bar, pos, 0, self._maxk)
        pos += 1
        if max_new > 1:
            ops.mak_run_steps(self._prog, self._bar, pos, max_new - 1, 1,
                              self._maxk)
        return self._tokens_out[:max_new].cpu().tolist()



