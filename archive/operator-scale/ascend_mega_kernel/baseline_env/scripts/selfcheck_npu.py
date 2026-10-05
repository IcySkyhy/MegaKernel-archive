#!/usr/bin/env python3
"""M30 device self-check: prove the torch/torch_npu stack really drives the Ascend 950PR.

`import torch_npu` alone proves nothing -- an import can succeed while every device call
fails. This script therefore does a full round trip: allocate on device, run aclnn-backed
ops (matmul, elementwise, reduction) on the NPU, copy the results back to host and compare
against the CPU reference computed by torch itself. Exit code is non-zero if any of the
numerical checks fail.

Usage:  /workspace/venvs/baseline/bin/python baseline_env/scripts/selfcheck_npu.py
"""

import sys
import time

import torch
import torch_npu  # noqa: F401  (registers the "npu" device backend)


def section(title):
    print(f"\n=== {title} ===")


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return ok


def main():
    failures = []

    section("versions / build ABI")
    print(f"torch            : {torch.__version__}")
    print(f"torch_npu        : {torch_npu.__version__}")
    print(f"torch CXX11 ABI  : {torch._C._GLIBCXX_USE_CXX11_ABI}  (ATB on this host: cxx_abi_1)")
    print(f"python           : {sys.version.split()[0]}")

    section("device discovery")
    n = torch.npu.device_count()
    print(f"torch.npu.device_count() = {n}")
    if not check("at least one NPU visible", n >= 1):
        return 1
    print(f"device name             : {torch.npu.get_device_name(0)}")
    props = torch.npu.get_device_properties(0)
    total = getattr(props, "total_memory", 0)
    print(f"device total memory     : {total} bytes ({total / 1024**3:.1f} GiB)")

    section("init (device context + stream)")
    t0 = time.time()
    torch.npu.set_device(0)
    a = torch.randn(2, 3, device="npu")
    torch.npu.synchronize()
    print(f"npu tensor created and stream synchronised in {time.time() - t0:.3f}s")
    if not check("tensor round trip to host", a.cpu().shape == (2, 3)):
        failures.append("tensor round trip")

    section("aclnn op round trip: matmul / elementwise / reduction")
    torch.manual_seed(0)
    x_cpu = torch.randn(512, 1024, dtype=torch.float16)
    w_cpu = torch.randn(1024, 512, dtype=torch.float16)

    x_npu = x_cpu.npu()
    w_npu = w_cpu.npu()
    torch.npu.synchronize()
    t0 = time.time()
    mm_npu = torch.matmul(x_npu, w_npu)          # aclnnMatmul
    s_npu = torch.softmax(mm_npu, dim=-1)        # aclnnSoftmax
    r_npu = torch.sum(mm_npu, dim=-1)            # aclnnReduceSum
    y_npu = (s_npu + 1.0) * 0.5                  # aclnn elementwise
    torch.npu.synchronize()
    dt = time.time() - t0

    mm_cpu = torch.matmul(x_cpu, w_cpu)
    s_cpu = torch.softmax(mm_cpu, dim=-1)
    r_cpu = torch.sum(mm_cpu, dim=-1)
    y_cpu = (s_cpu + 1.0) * 0.5

    for name, dev, ref in (
        ("matmul (aclnnMatmul)", mm_npu, mm_cpu),
        ("softmax (aclnnSoftmax)", s_npu, s_cpu),
        ("sum (aclnnReduceSum)", r_npu, r_cpu),
        ("elementwise mul/add", y_npu, y_cpu),
    ):
        dev_cpu = dev.cpu().float()
        ref_f = ref.float()
        max_abs = (dev_cpu - ref_f).abs().max().item()
        rel = max_abs / ref_f.abs().max().item()
        ok = rel < 1e-2  # fp16 accumulate on 1024-deep reduction
        print(f"  {name:26s} max_abs_diff={max_abs:.4e} rel={rel:.3e}")
        if not check(f"{name} numerics within fp16 tolerance", ok):
            failures.append(name)
    print(f"4 ops (fp16 512x1024x512 matmul + 3 epilogues) round trip in {dt * 1e3:.1f} ms")

    section("device memory accounting")
    alloc = torch.npu.memory_allocated() / 1024**2
    reserv = torch.npu.memory_reserved() / 1024**2
    print(f"memory_allocated = {alloc:.1f} MiB, memory_reserved = {reserv:.1f} MiB")
    if not check("allocator reports non-zero usage", alloc > 0):
        failures.append("allocator")

    section("host-side aclnn availability (no device)")
    try:
        import acl  # noqa: F401

        print("acl python bindings importable: yes")
    except Exception as exc:  # noqa: BLE001
        print(f"acl python bindings importable: no ({exc!r})")

    section("result")
    if failures:
        print(f"SELF-CHECK FAILED: {failures}")
        return 1
    print("SELF-CHECK PASSED: torch/torch_npu drives the Ascend 950PR with numerically correct ops")
    return 0


if __name__ == "__main__":
    sys.exit(main())
