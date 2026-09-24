#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Kernel-bundle scenario worker: the pack half and the consume half of each .pyptokb case.

Run as a subprocess by ``test_kernel_bundle.py``; never imported by the pytest process itself.
Two phases must be separate processes because the consume half plays the role of a downstream that only
has ``libtile_fwk_bundle.so`` + a ``.pyptokb`` -- no compiler front end, no recorded Program state.

    python3 bundle_cases.py pack    <case> <bundle_path>
    python3 bundle_cases.py consume <case> <bundle_path> [--api path|memory]

Cases:
  static_add       non-value-dependent, static workspace (0 B); packed through the emulation hook, so the
                   bundle carries a NON-EMPTY CtrlFlowCache snapshot.
  dyn_cellmatch    non-value-dependent but dynamically shaped: SymbolMeta carries real symbolic trees plus
                   the dynamic cell-match launch metas, so workspace is re-evaluated per launch shape.
  value_depend_pa  value-dependent control flow (loop bounds read from act_seqs tensor VALUES). Packed
                   through the value-depend hook, so the CtrlFlowCache segment is EMPTY and the on-device
                   interpreter resolves the trip counts at launch.
"""
import argparse
import ctypes
import os
import sys

# This worker runs as a standalone script (python3 bundle_cases.py ...), not under pytest, so the tests root
# has to reach sys.path before the st.* imports below -- hence the E402 waivers, as elsewhere in the tree.
_TESTS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TESTS_ROOT not in sys.path:
    sys.path.insert(0, _TESTS_ROOT)

import torch  # noqa: E402
import torch_npu  # noqa: E402, F401  -- registers the npu backend

import pypto  # noqa: E402
from st.bundle.bundle_abi import BundleClient, make_desc_array  # noqa: E402
from st.operator.test_page_attention import (  # noqa: E402
    TileConfig,
    op_page_attention,
    op_page_attention_golden,
)
from st.pypto_test import TestBuilder  # noqa: E402
from st.test_cellmatch_case import B_STATIC, D_STATIC, H_STATIC, k_tmp_to_d_emb  # noqa: E402


# ---------------------------------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------------------------------
def _device():
    dev_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    torch.npu.set_device(dev_id)
    return dev_id, f"npu:{dev_id}"


def _alloc_workspace(size: int, device: str):
    """Allocate the caller-owned workspace. Returns (keepalive_tensor, void_ptr); (None, None) when 0."""
    if size == 0:
        return None, None
    buf = torch.empty(size, dtype=torch.uint8, device=device)
    return buf, ctypes.c_void_p(buf.data_ptr())


def _max_diff(actual: torch.Tensor, golden: torch.Tensor) -> float:
    return (actual.detach().cpu().float() - golden.detach().cpu().float()).abs().max().item()


def _check(label: str, actual: torch.Tensor, golden: torch.Tensor, tol: float):
    diff = _max_diff(actual, golden)
    print(f"[ST] {label}: max diff = {diff:.6f} (tol {tol})")
    assert diff < tol, f"{label}: output mismatch, max_diff={diff} >= {tol}"


# ---------------------------------------------------------------------------------------------------
# case: static_add -- non-value-dependent, static, zero workspace
# ---------------------------------------------------------------------------------------------------
ADD_SHAPE = (1, 4, 1, 64)


def _make_add_kernel():
    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
    def add_kernel(
        x: pypto.Tensor([...], pypto.DT_FP32),
        y: pypto.Tensor([...], pypto.DT_FP32),
        out: pypto.Tensor([...], pypto.DT_FP32),
    ):
        pypto.set_vec_tile_shapes(1, 4, 1, 64)
        out[:] = x + y

    return add_kernel


def _add_inputs(device: str):
    torch.manual_seed(7)
    x = torch.rand(ADD_SHAPE, dtype=torch.float32, device=device)
    y = torch.rand(ADD_SHAPE, dtype=torch.float32, device=device)
    out = torch.zeros(ADD_SHAPE, dtype=torch.float32, device=device)
    return x, y, out


def pack_static_add():
    _, device = _device()
    x, y, out = _add_inputs(device)
    _make_add_kernel()(x, y, out)
    torch.npu.synchronize()
    _check("pack-run add", out, x + y, 3e-3)


def consume_static_add(bundle_path: str, api: str):
    _, device = _device()
    client = BundleClient(bundle_path, api)
    x, y, out = _add_inputs(device)

    # Unified operand list: outputs are folded into the input list, addressed by position.
    descs = make_desc_array([x, y, out])

    # 1) workspace query -- pure host, must not touch the device.
    ws_size = client.workspace(descs)
    print(f"[ST] workspace size = {ws_size}")
    assert ws_size == 0, f"static add should need no workspace, got {ws_size}"

    # 2) launch.
    ws_keep, ws_ptr = _alloc_workspace(ws_size, device)
    rc = client.launch(descs, ws_ptr, None, 1)
    assert rc == 0, f"PyptoLaunch rc={rc}"
    torch.npu.synchronize()
    del ws_keep

    _check(f"add ({api} API)", out, x + y, 3e-3)


# ---------------------------------------------------------------------------------------------------
# case: dyn_cellmatch -- non-value-dependent, dynamic shapes + dynamic cell-match workspace
# ---------------------------------------------------------------------------------------------------
# Reuses the kernel and the shape constants from the dynamic cell-match ST so the two cannot drift apart.
CM_PACK_L = 125
# Host-only workspace probes. The v1 CtrlFlowCache is a single static-shape snapshot, so only the packed
# shape may actually be launched -- see the note in test_kernel_bundle.py.
CM_PROBE_LENGTHS = [64, 125, 2000]


def _cm_inputs(seq_len: int, device: str):
    torch.manual_seed(44 + seq_len)
    dy = torch.randn(B_STATIC, seq_len, H_STATIC, D_STATIC, dtype=torch.float32, device=device)
    weight = torch.randn(H_STATIC, D_STATIC, D_STATIC, dtype=torch.float32, device=device)
    d_emb = torch.zeros(B_STATIC, seq_len, D_STATIC, dtype=torch.float32, device=device)
    return dy, weight, d_emb


def _cm_golden(dy: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (dy[:, :, 0, :].reshape(-1, dy.shape[-1]) @ weight[0].T).reshape(dy.shape[0], dy.shape[1], dy.shape[-1])


def pack_dyn_cellmatch():
    _, device = _device()
    dy, weight, d_emb = _cm_inputs(CM_PACK_L, device)
    k_tmp_to_d_emb(dy, weight, d_emb)
    torch.npu.synchronize()
    _check(f"pack-run cellmatch L={CM_PACK_L}", d_emb, _cm_golden(dy, weight), 1e-3)


def consume_dyn_cellmatch(bundle_path: str, api: str):
    _, device = _device()
    client = BundleClient(bundle_path, api)

    # 1) Dynamic workspace evaluation: the same bundle, queried at several shapes. This exercises the
    #    SymbolMeta symbolic trees (RUNTIME_GetInputShapeDim over ARG_dy) and the dynamic cell-match
    #    stride re-patching. A bundle with a baked constant would return the same number every time.
    sizes = []
    for seq_len in CM_PROBE_LENGTHS:
        dy, weight, d_emb = _cm_inputs(seq_len, device)
        ws = client.workspace(make_desc_array([dy, weight, d_emb]))
        print(f"[ST] workspace(L={seq_len}) = {ws}")
        sizes.append(ws)
        del dy, weight, d_emb
    assert all(s > 0 for s in sizes), f"dynamic workspace must be non-zero, got {sizes}"
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes), (
        f"workspace must grow strictly with the sequence length, got "
        f"{dict(zip(CM_PROBE_LENGTHS, sizes))} -- symbol evaluation likely fell back to a baked constant"
    )

    # 2) Launch at the packed shape and check the numerics end to end.
    dy, weight, d_emb = _cm_inputs(CM_PACK_L, device)
    descs = make_desc_array([dy, weight, d_emb])
    ws_size = client.workspace(descs)
    ws_keep, ws_ptr = _alloc_workspace(ws_size, device)
    assert ws_size > 0, "expected a non-zero workspace at the packed shape"
    rc = client.launch(descs, ws_ptr, None, 1)
    assert rc == 0, f"PyptoLaunch rc={rc}"
    torch.npu.synchronize()
    del ws_keep

    _check(f"cellmatch L={CM_PACK_L} ({api} API)", d_emb, _cm_golden(dy, weight), 1e-3)


# ---------------------------------------------------------------------------------------------------
# case: value_depend_pa -- value-dependent control flow (page attention)
# ---------------------------------------------------------------------------------------------------
# Loop trip counts come from act_seqs tensor VALUES (`cur_seq = act_seqs[b_idx]` -> GetTensorData ->
# bn_per_batch), which sets devProg->disableCtrlFlowCache. Every shape below is held constant across the
# two value sets -- the cache size is pinned by PA_MAX_BLOCKS, not by max(act_seqs) -- so the only thing
# that differs between the two consume launches is the tensor CONTENT.
PA_BLOCK_SIZE = 128
PA_MAX_BLOCKS = 4  # blocks reserved per batch; pins block_table / kv-cache shapes
PA_BATCH = 4
PA_N_TILE = 32

PA_ACT_SEQS_PACKED = [512, 512, 512, 512]  # bn_per_batch = [4, 4, 4, 4]
PA_ACT_SEQS_ALT = [128, 512, 256, 384]     # bn_per_batch = [1, 4, 2, 3] -- same shapes, different values

PA_TOL = 5e-4


def _pa_params():
    return {
        "block_size": PA_BLOCK_SIZE,
        "tile_config": TileConfig(
            head_num_q_tile=PA_N_TILE,
            c1_tile_shape=(32, 32, 64, 64, 128, 128),
            v1_tile_shape=(32, 64),
            c2_tile_shape=(32, 32, 64, 64, 128, 128),
            v2_tile_shape=(32, 64),
        ),
        "max_unroll_times": 1,
        "is_nz_format": False,
        "b": PA_BATCH,
        "n_q": 32,
        "block_num": PA_BATCH * PA_MAX_BLOCKS,
        "dtype": torch.float32,
        "s_q": 1,
        "n_kv": 1,
        "kv_lora_rank": 512,
        "qk_rope_dim": 64,
        "n_tile": PA_N_TILE,
    }


def _pa_make_inputs(params, act_seqs, device: str, seed: int = 11):
    """Build the 7-operand input list on CPU or device. Shapes depend only on params, never on act_seqs."""
    b = params["b"]
    n_q = params["n_q"]
    s_q = params["s_q"]
    n_kv = params["n_kv"]
    kv_lora_rank = params["kv_lora_rank"]
    qk_rope_dim = params["qk_rope_dim"]
    block_size = params["block_size"]
    block_num = params["block_num"]
    dtype = params["dtype"]
    d_q = kv_lora_rank + qk_rope_dim

    max_seq = max(act_seqs)
    assert max_seq <= PA_MAX_BLOCKS * block_size, (
        f"act_seqs {act_seqs} exceeds the {PA_MAX_BLOCKS * block_size}-token reservation")

    torch.manual_seed(seed)
    q_bnsd = (torch.rand([b * n_q * s_q, d_q], dtype=dtype) * 2 - 1).to(device)
    k_cache = (torch.rand([block_num, block_size, n_kv * d_q], dtype=dtype) * 2 - 1).to(device)
    v_cache = (torch.rand([block_num, block_size, n_kv * kv_lora_rank], dtype=dtype) * 2 - 1).to(device)

    # Identity block mapping: batch bi owns physical blocks [bi*PA_MAX_BLOCKS, (bi+1)*PA_MAX_BLOCKS).
    block_table = torch.arange(0, block_num, dtype=torch.int32).reshape(b, PA_MAX_BLOCKS).to(device)
    act = torch.tensor(act_seqs, dtype=torch.int32).to(device)

    nope_h = n_kv * kv_lora_rank
    q_nope = q_bnsd[:, :kv_lora_rank].contiguous()
    q_rope = q_bnsd[:, kv_lora_rank:].contiguous()
    k_cache_nope = k_cache[:, :, :nope_h].reshape(block_num * block_size, nope_h).contiguous()
    k_cache_rope = k_cache[:, :, nope_h:].reshape(block_num * block_size, n_kv * qk_rope_dim).contiguous()
    v_cache_2d = v_cache.reshape(block_num * block_size, nope_h).contiguous()
    return [q_nope, k_cache_nope, v_cache_2d, q_rope, k_cache_rope, block_table, act]


def _pa_golden(params, inputs):
    cpu_inputs = [t.cpu() for t in inputs]
    out, = op_page_attention_golden(params, *cpu_inputs, None)
    return out


def pack_value_depend_pa():
    class _PABundlePack(TestBuilder):
        def get_input_from_param(self):
            # TestBuilder feeds the host-side run-once path, which wants CPU tensors.
            inputs = _pa_make_inputs(self.params, self.params["act_seqs"], "cpu")
            self.setup_inputs(*inputs)
            self.set_tol(rtol=PA_TOL, atol=PA_TOL)
            # Mirrors PATest: the golden's trailing attention_out slot is unused, params rides along.
            return inputs + [self.params]

    params = _pa_params()
    params["act_seqs"] = PA_ACT_SEQS_PACKED
    _PABundlePack(params, op_page_attention, op_page_attention_golden, tiling=PA_N_TILE)()


def consume_value_depend_pa(bundle_path: str, api: str):
    _, device = _device()
    client = BundleClient(bundle_path, api)
    params = _pa_params()

    # Both launches reuse the one loaded bundle. Identical shapes, different act_seqs VALUES: if the device
    # were replaying a host-baked control-flow cache instead of resolving trip counts from the tensor, the
    # second case would read the wrong number of KV blocks and miss the golden.
    goldens = {}
    for label, act_seqs in (("packed", PA_ACT_SEQS_PACKED), ("alt", PA_ACT_SEQS_ALT)):
        inputs = _pa_make_inputs(params, act_seqs, device)
        out = torch.zeros(
            params["b"] * params["n_q"] * params["s_q"], params["kv_lora_rank"],
            dtype=params["dtype"], device=device)
        descs = make_desc_array(inputs + [out])

        ws_size = client.workspace(descs)
        print(f"[ST] {label} act_seqs={act_seqs}: workspace = {ws_size}")
        ws_keep, ws_ptr = _alloc_workspace(ws_size, device)
        rc = client.launch(descs, ws_ptr, None, 1)
        assert rc == 0, f"{label}: PyptoLaunch rc={rc}"
        torch.npu.synchronize()
        del ws_keep

        golden = _pa_golden(params, inputs)
        goldens[label] = golden
        _check(f"page-attention {label} ({api} API)", out.reshape(golden.shape), golden, PA_TOL)

    # Guard the guard: the two value sets share a seed, so they differ only through act_seqs. If they no
    # longer drive the output apart, the pair of launches above would pass even on a device that ignored
    # the tensor values entirely, and this case would stop testing value dependency at all.
    spread = _max_diff(goldens["packed"], goldens["alt"])
    print(f"[ST] golden spread between the two act_seqs value sets = {spread:.6f}")
    assert spread > PA_TOL * 10, (
        f"act_seqs {PA_ACT_SEQS_PACKED} vs {PA_ACT_SEQS_ALT} barely move the result (spread={spread}); "
        f"pick value sets whose per-batch block counts differ more"
    )


# ---------------------------------------------------------------------------------------------------
# registry + CLI
# ---------------------------------------------------------------------------------------------------
CASES = {
    "static_add": (pack_static_add, consume_static_add),
    "dyn_cellmatch": (pack_dyn_cellmatch, consume_dyn_cellmatch),
    "value_depend_pa": (pack_value_depend_pa, consume_value_depend_pa),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["pack", "consume"])
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("bundle_path")
    parser.add_argument("--api", choices=["path", "memory"], default="path")
    args = parser.parse_args()

    pack_fn, consume_fn = CASES[args.case]
    if args.phase == "pack":
        # Packing is gated by these two env vars; the driver sets them, assert rather than silently no-op.
        assert os.environ.get("PYPTO_ENABLE_KERNEL_BUNDLE") == "1", "PYPTO_ENABLE_KERNEL_BUNDLE must be 1"
        assert os.environ.get("PYPTO_KERNEL_BUNDLE_PATH") == args.bundle_path, \
            "PYPTO_KERNEL_BUNDLE_PATH must match the requested bundle path"
        pack_fn()
        assert os.path.exists(args.bundle_path), f"no bundle produced at {args.bundle_path}"
        print(f"[ST] packed {args.case} -> {args.bundle_path} ({os.path.getsize(args.bundle_path)} B)")
    else:
        consume_fn(args.bundle_path, args.api)
        print(f"[ST] consumed {args.case} via the {args.api} API")


if __name__ == "__main__":
    main()
    # The runtime singletons (DeviceRunner/DevicePerf) are torn down at interpreter exit, AFTER torch_npu has
    # already dropped the NPU context -- that races into a double free in ~DevicePerf. Everything under test
    # has been asserted by this point, so flush and exit hard rather than run the static destructors.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
