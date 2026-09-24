# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Dump the generated MEGA_TRITON_KERNEL Triton-DSL source for Qwen3
# WITHOUT any GPU / NVSHMEM / real weights. Runs on pure CPU / NPU.
#
#   python -m triton_dist.mega_kernel_ascend.test.models.dump_qwen3_dsl --out /tmp/qwen3.py
#   python -m triton_dist.mega_kernel_ascend.test.models.dump_qwen3_dsl --out /tmp/qwen3.py --model Qwen/Qwen3-32B --num-layers 2
#   (or) python dump_qwen3_dsl.py --out /tmp/qwen3.py
#
# This file lives at:
#   python/triton_dist/mega_kernel_ascend/test/models/dump_qwen3_dsl.py
# and resolves the real repo codegen at:
#   python/triton_dist/mega_triton_kernel/
# via __file__ (three parents up), so it is path-portable: clone the repo, run
# from anywhere, no PYTHONPATH / env vars required.
#
# Strategy
#   This script runs the megakernel codegen on a GPU-less box (NPU/CPU) and
#   writes out the generated MEGA_TRITON_KERNEL Triton-DSL source. It is
#   environment-agnostic: `triton_dist` may be installed in site-packages (where
#   its `mega_triton_kernel` subpackage can be an empty shell) OR imported
#   straight from this repo. Either way the goal is the same — drive the real
#   repo `mega_triton_kernel` codegen while never touching CUDA/NVSHMEM/real
#   weights.
#
#   So we:
#     - a meta-path finder reroutes `triton_dist.mega_triton_kernel.*` to load
#       from the repo path (matters when site-packages ships an empty shell;
#       harmless when the repo is already on sys.path);
#     - a meta-path finder stubs `triton_dist.models` (+layers/kernels vendor
#       submodules that raise on CPU) so the eager backend never raises;
#     - a meta-path finder stubs `nvshmem` (model_builder does `import nvshmem`);
#     - patch torch.cuda.* (is_available / current_device /
#       get_device_properties / synchronize / Tensor.to('cuda') / .cuda()) so
#       CPU tensors survive the CUDA touch points. Note is_cuda()/is_ascend()
#       in triton_dist.utils are NOT affected — they check `which nvidia-smi` /
#       `which npu-smi`, not torch.cuda, so is_ascend() stays True here;
#     - replace DenseModel.init_parameters with CPU-random weights (REAL Qwen3
#       shapes from HF config) so build_fwd() registers all tasks; make_* only
#       does shape checks + task registration (string codegen), no kernel runs;
#     - replace ModelBuilder.compile to skip the CUDA-only steps
#       (enque_tasks/scoreboard/exec_module) and just run
#       CodeGenerator.generate_code, caching the DSL on the builder (see
#       _compile_dsl_only / main).
import argparse
import os
import sys
import types

# Resolve the repo `mega_triton_kernel` dir relative to THIS file so the script
# is portable across machines / clones. This file is at:
#   <repo>/python/triton_dist/mega_kernel_ascend/test/models/dump_qwen3_dsl.py
# so the real codegen package is three parents up, then into mega_triton_kernel.
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_MEGA = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "mega_triton_kernel"))
assert os.path.isdir(REPO_MEGA), (
    f"resolved REPO_MEGA={REPO_MEGA!r} does not exist; expected the repo "
    f"mega_triton_kernel package three dirs above this file ({__file__})"
)


# ----------------------------------------------------------------------------
# 1. nvshmem stub (any submodule) — model_builder does `import nvshmem`
# ----------------------------------------------------------------------------
class _Mock:
    def __getattr__(self, _): return _Mock()
    def __call__(self, *a, **k): return _Mock()

class _StubLoader:
    def __init__(self, mod): self._mod = mod
    def create_module(self, spec): return self._mod
    def exec_module(self, module): return None

class _AttrMockModule(types.ModuleType):
    def __getattr__(self, name): return _Mock()

class _NVMetaFinder:
    def find_spec(self, fullname, path, target=None):
        if fullname == "nvshmem" or fullname.startswith("nvshmem."):
            import importlib.util
            mod = _AttrMockModule(fullname)
            mod.__path__ = []
            if fullname == "nvshmem":
                class _Teams: TEAM_NODE = 0
                class _Core: Teams = _Teams
                class _Bindings:
                    @staticmethod
                    def mc_ptr(*a, **k): return 0
                mod.core = _Core()
                mod.bindings = _Bindings()
            spec = importlib.util.spec_from_loader(fullname, loader=None)
            spec.loader = _StubLoader(mod)
            return spec
        return None

# ----------------------------------------------------------------------------
# 2. reroute triton_dist.mega_triton_kernel.* to the repo, stub triton_dist.models
# ----------------------------------------------------------------------------
_MEGA = "triton_dist.mega_triton_kernel"

class _RepoMegaFinder:
    """Load triton_dist.mega_triton_kernel (and submodules) from the repo path."""
    def find_spec(self, fullname, path, target=None):
        if fullname == _MEGA or fullname.startswith(_MEGA + "."):
            import importlib.util
            # map fullname -> repo file path
            sub = fullname[len(_MEGA):]                 # "" or ".core.builder"
            parts = [p for p in sub.split(".") if p]
            pkg_dir = REPO_MEGA
            for p in parts:
                pkg_dir = os.path.join(pkg_dir, p)
            if os.path.isdir(pkg_dir):
                target_path = os.path.join(pkg_dir, "__init__.py")
            elif os.path.exists(pkg_dir + ".py"):
                target_path = pkg_dir + ".py"
            else:
                return None
            spec = importlib.util.spec_from_file_location(fullname, target_path)
            return spec
        return None

_STUB_PREFIXES = (
    "triton_dist.models",          # eager backend; raises on CPU
    "triton_dist.layers.nvidia",
    "triton_dist.layers.amd",
    "triton_dist.kernels.nvidia",
    "triton_dist.kernels.amd",
    "triton_dist.kernels.common_ops",
)

class _StubFinder:
    def find_spec(self, fullname, path, target=None):
        if any(fullname == p or fullname.startswith(p + ".") for p in _STUB_PREFIXES):
            import importlib.util
            mod = _AttrMockModule(fullname)
            mod.__path__ = []
            spec = importlib.util.spec_from_loader(fullname, loader=None)
            spec.loader = _StubLoader(mod)
            return spec
        return None

# order matters: reroute repo mega first, then stub eager, then nvshmem
sys.meta_path.insert(0, _NVMetaFinder())
sys.meta_path.insert(0, _StubFinder())
sys.meta_path.insert(0, _RepoMegaFinder())


# ----------------------------------------------------------------------------
# 3. patch torch.cuda so CPU-only torch survives the CUDA touch points
# ----------------------------------------------------------------------------
import torch

torch.version.cuda = "12.4"
torch.cuda.is_available = lambda: True

class _FakeDevProps:
    multi_processor_count = 8
    major = 9; minor = 0; name = "fake-cpu"
torch.cuda.get_device_properties = lambda device: _FakeDevProps()
torch.cuda.current_device = lambda: torch.device("cpu")
torch.cuda.synchronize = lambda *a, **k: None

_orig_to = torch.Tensor.to
_orig_cuda = torch.Tensor.cuda
def _to(self, *args, **kwargs):
    for a in (args + tuple(kwargs.values())):
        if isinstance(a, str) and "cuda" in a: return self.contiguous()
        if isinstance(a, torch.device) and a.type == "cuda": return self.contiguous()
        if isinstance(a, int): return self.contiguous()
    return _orig_to(self, *args, **kwargs)
def _cuda(self, *a, **k): return self.contiguous()
torch.Tensor.to = _to
torch.Tensor.cuda = _cuda

# flashinfer (imported by triton_dist.models.utils on nvidia branch, now stubbed away)
sys.modules.setdefault("flashinfer", _Mock())

# init a 1-process gloo group so torch.distributed.barrier() works
if not torch.distributed.is_initialized():
    try:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29512")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        torch.distributed.init_process_group(backend="gloo", world_size=1, rank=0)
    except Exception as e:
        print(f"[warn] init_process_group skipped: {e}", file=sys.stderr)


# ----------------------------------------------------------------------------
# 4. now import the real (repo) builders / codegen
# ----------------------------------------------------------------------------
from triton_dist.mega_triton_kernel import ModelBuilder  # noqa: E402
from triton_dist.mega_triton_kernel.models import DenseModel  # noqa: E402
from triton_dist.mega_triton_kernel.models.dense import DenseLayerBuilder  # noqa: E402
from triton_dist.mega_triton_kernel.models.layers.tp_attn import TPAttnBuilder  # noqa: E402
from triton_dist.mega_triton_kernel.models.layers.tp_mlp import TPMLPBuilder  # noqa: E402
from triton_dist.mega_triton_kernel.models.paged_kv_cache import PagedKVCache  # noqa: E402

# ----------------------------------------------------------------------------
# 5. Load the REAL HuggingFace config (same source as bench's
#    DenseModel.init_parameters -> AutoConfig.from_pretrained). We cannot reuse
#    init_model_cpu (it needs a GPU and triton_dist.models raises on CPU), so
#    weights stay CPU-random placeholders; but every SHAPE / hyperparam below now
#    comes from the actual HF config.json, so the dumped DSL matches the kernel
#    bench runs. Note: the previously-hardcoded QWEN3_32B dict had WRONG values
#    (intermediate_size 27648 vs real 25600, vocab_size 152064 vs real 151936,
#    eos 151643 vs real 151645) which silently changed the MLPFC*/Linear/lm_head
#    codegen literals.
# ----------------------------------------------------------------------------
def _load_hf_config(model_name: str, local_only: bool = True, num_layers_override=None) -> dict:
    """Return the real Qwen3 config dict loaded from HuggingFace.

    Uses transformers.AutoConfig (CPU-safe, no weights downloaded) for validated
    Qwen3-typed fields, then reads rope_theta from the raw config.json because
    AutoConfig.to_dict() returns None for rope_theta under the local transformers
    (it is consumed by the rope_scaling machinery). Falls back to raw json fully
    if AutoConfig is unavailable.
    """
    import json
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_name, local_files_only=local_only)

    # locate the raw config.json for fields AutoConfig mangles
    rope_theta = getattr(cfg, "rope_theta", None)
    if not rope_theta:
        try:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(repo_id=model_name, filename="config.json", local_files_only=local_only)
            with open(p) as f:
                rope_theta = json.load(f).get("rope_theta", 1000000.0)
        except Exception:
            rope_theta = 1000000.0  # Qwen3 default; matches all released Qwen3-32B revisions

    cfg_dict = dict(
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        num_hidden_layers=cfg.num_hidden_layers,
        head_dim=cfg.head_dim,
        vocab_size=cfg.vocab_size,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=rope_theta,
        max_position_embeddings=getattr(cfg, "max_position_embeddings", 40960),
        eos_token_id=getattr(cfg, "eos_token_id", 151645),
    )
    if num_layers_override is not None:
        cfg_dict["num_hidden_layers"] = num_layers_override
    return cfg_dict


# Populated in main() once we know the model name. Kept module-level so the
# patched _DenseModel_init / _fake_init_parameters can read it like the old
# hardcoded dict.
QWEN3_CONFIG: dict = {}


def _fake_init_parameters(self):
    """CPU-random weights with REAL Qwen3 shapes (mirrors real init_parameters
    but skips init_model_cpu / weight download / .cuda() — all need a GPU)."""
    cfg = self._hf_config
    H = cfg["hidden_size"]; nh = cfg["num_attention_heads"]; nkv = cfg["num_key_value_heads"]
    hd = cfg["head_dim"]; inter = cfg["intermediate_size"]; nl = cfg["num_hidden_layers"]
    vocab = cfg["vocab_size"]; eps = cfg["rms_norm_eps"]
    ws = self.world_size; rank = self.rank; dtype = torch.bfloat16

    def sh(t, dim):
        return t.split(t.shape[dim] // ws, dim=dim)[rank].contiguous()

    self.hidden_size = H; self.num_heads = nh; self.head_dim = hd
    self.num_key_value_heads = nkv; self.num_layers = nl
    self.max_position_embeddings = cfg["max_position_embeddings"]; self.rope_theta = cfg["rope_theta"]
    self.eos_token_id = cfg["eos_token_id"]

    self.embed_tokens = torch.empty((vocab, H), dtype=dtype)
    self.lm_head = torch.empty((vocab, H), dtype=dtype)
    self.norm_weight = torch.empty((H,), dtype=dtype)
    self.norm_variance_epsilon = eps

    inv_freq = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, hd, 2).float() / hd))
    t = torch.arange(self.max_length, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    self.cos_cache = emb.cos().to(torch.float32).unsqueeze(0)
    self.sin_cache = emb.sin().to(torch.float32).unsqueeze(0)

    self.layers: list = []
    for idx in range(nl):
        layer = DenseLayerBuilder(builder=self._builder, layer_idx=idx, head_dim=hd,
                                  rank=rank, world_size=ws)
        # real builders, but bypass _init_parameters (weights loaded manually)
        layer.mlp = TPMLPBuilder(builder=self._builder, rank=rank, world_size=ws)
        layer.mlp.gate_up_proj = torch.cat((sh(torch.empty((inter, H), dtype=dtype), 0),
                                            sh(torch.empty((inter, H), dtype=dtype), 0)), 0).contiguous()
        layer.mlp.down_proj = sh(torch.empty((H, inter), dtype=dtype), 1).contiguous()
        layer.mlp.ag_N_per_rank = layer.mlp.gate_up_proj.shape[0]
        layer.mlp.K = layer.mlp.gate_up_proj.shape[1]; layer.mlp.dtype = dtype

        layer.attn = TPAttnBuilder(builder=self._builder, layer_idx=idx, head_dim=hd,
                                   rank=rank, world_size=ws)
        q = torch.empty((nh * hd, H), dtype=dtype)
        k = torch.empty((nkv * hd, H), dtype=dtype)
        v = torch.empty((nkv * hd, H), dtype=dtype)
        o = torch.empty((H, nh * hd), dtype=dtype)   # o_proj.weight: (out=H, in=nh*hd)
        layer.attn.wqkv = torch.cat((sh(q, 0), sh(k, 0), sh(v, 0)), 0).contiguous()
        layer.attn.wo = sh(o, 1).contiguous()
        layer.attn.q_size = q.shape[0] // ws; layer.attn.kv_size = k.shape[0] // ws
        layer.attn.ag_N_per_rank = layer.attn.wqkv.shape[0]; layer.attn.K = layer.attn.wqkv.shape[1]
        layer.attn.dtype = dtype; layer.attn.sm_scale = hd ** -0.5; layer.attn.soft_cap = 0.0
        # Match the real bench path (layers/tp_attn.py): Qwen3 attention HAS q_norm/k_norm,
        # so skip_*_norm must be False. These bools are baked as codegen literals
        # (tasks/norm.py:105 -> kernels/norm.py:134), so True vs False yields different
        # generated source. The weights are zeros only as a placeholder (codegen reads
        # shapes, not values), but skip flags must match bench to dump the same kernel.
        layer.attn.skip_q_norm = False; layer.attn.skip_k_norm = False
        layer.attn.q_norm_w = torch.zeros((hd,), dtype=dtype); layer.attn.k_norm_w = torch.zeros((hd,), dtype=dtype)
        layer.attn.q_norm_eps = eps; layer.attn.k_norm_eps = eps

        layer.input_norm_eps = eps; layer.input_norm_w = torch.empty((H,), dtype=dtype)
        layer.post_norm_eps = eps; layer.post_norm_w = torch.empty((H,), dtype=dtype)
        self.layers.append(layer)


# ----------------------------------------------------------------------------
# 6. patch DenseModel.__init__ (no ModelConfig / AutoConfig needed)
# ----------------------------------------------------------------------------
def _DenseModel_init(self, batch_size, model_name="Qwen/Qwen3-32B", build_lm_head=True,
                     rank=0, world_size=1, builder=None, max_length=132, hf_config=None):
    self._builder = builder
    self.dtype = torch.bfloat16
    self.model_name = model_name; self.max_length = max_length; self.build_lm_head = build_lm_head
    self.rank = rank; self.world_size = world_size; self.batch_size = batch_size
    # real HF config (loaded by main via _load_hf_config); falls back to module-level
    # QWEN3_CONFIG if caller didn't pass one explicitly.
    self._hf_config = hf_config if hf_config is not None else QWEN3_CONFIG
    self.hidden_state_buffer = torch.empty((batch_size, 1, self._hf_config["hidden_size"]),
                                          dtype=torch.bfloat16)
    _fake_init_parameters(self)
    self.kv_cache = PagedKVCache(num_layers=self.num_layers, batch_size=self.batch_size,
                                 max_length=self.max_length,
                                 num_kv_heads=self.num_key_value_heads // self.world_size,
                                 head_dim=self.head_dim, dtype=self.dtype)
    self.mega_out = self.build_fwd(self.hidden_state_buffer, self.kv_cache)
    self._builder.compile()
    torch.cuda.synchronize()

DenseModel.__init__ = _DenseModel_init


# ----------------------------------------------------------------------------
# 7. patch ModelBuilder.compile to skip CUDA-only enque_tasks/scoreboard/exec.
#    The real compile() (model_builder.py) does 4 things: enque_tasks (needs
#    cuda tensors) -> torch.zeros(scoreboard, device=cuda) -> generate_code
#    -> exec_module. We only need generate_code; the rest is CUDA-bound and
#    irrelevant for DSL dumping. Cache the src on the builder so main() reads
#    it back instead of re-running generate_code a second time.
# ----------------------------------------------------------------------------
def _compile_dsl_only(self):
    cg = self._code_generator
    tasks = self.megakernel_tasks if not self._enable_dep_opt else self._graph.to_tasks()
    src, task_types_to_str = cg.generate_code(tasks, self._codegen_options)
    self.task_types_to_str = task_types_to_str
    self._gen_kernel = None
    self._dsl_src = src          # main() reads this; avoids a 2nd generate_code call
    return src

ModelBuilder.compile = _compile_dsl_only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-32B", type=str,
                    help="HuggingFace model name (must be downloadable / cached). "
                         "This is the same config source bench_qwen3 uses.")
    ap.add_argument("--out", default="/tmp/qwen3_mega_kernel.py")
    ap.add_argument("--num-layers", type=int, default=None,
                    help="override num_hidden_layers (default: from HF config; use 2 for a tiny smoke DSL)")
    ap.add_argument("--world-size", type=int, default=1)
    ap.add_argument("--online", default=False, action="store_true",
                    help="allow downloading the config from HF (default: local cache only, like a locked-down NPU box)")
    args = ap.parse_args()

    # Load the REAL config from HuggingFace — same AutoConfig.from_pretrained the
    # real DenseModel path uses. local_only=True by default so this runs offline.
    cfg = _load_hf_config(args.model, local_only=not args.online,
                          num_layers_override=args.num_layers)
    global QWEN3_CONFIG
    QWEN3_CONFIG = cfg
    print(f"loaded HF config for {args.model}: {cfg}")

    builder = ModelBuilder(rank=0, world_size=args.world_size,
                           local_world_size=args.world_size,
                           enable_runtime_scheduler=False)
    model = DenseModel(batch_size=1, model_name=args.model, build_lm_head=True,
                      rank=0, world_size=args.world_size, builder=builder, max_length=132,
                      hf_config=cfg)  # noqa: F841
    # DenseModel.__init__ already drove build_fwd + patched compile(), so the DSL
    # is cached on the builder. Read it back instead of re-running generate_code.
    src = builder._dsl_src
    task_types_to_str = builder.task_types_to_str
    tasks = builder.megakernel_tasks   # for the num_total_tasks log line only

    with open(args.out, "w") as f:
        f.write(src)
    print(f"DSL written to {args.out} ({len(src)} bytes, {src.count(chr(10))} lines)")
    print(f"num_total_tasks = {len(tasks)}, task_types = {task_types_to_str}")
    try:
        builder.finalize()
    except Exception:
        pass


if __name__ == "__main__":
    main()
