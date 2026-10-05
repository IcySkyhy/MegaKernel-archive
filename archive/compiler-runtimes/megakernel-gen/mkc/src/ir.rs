//! Model IR: the architecture-neutral description a megakernel is compiled from.
//!
//! Everything downstream (planner, cost model, codegen) sees only this.  An
//! architecture frontend's whole job is to turn a HuggingFace `config.json`
//! plus a safetensors weight index into one `Model`.

use std::collections::BTreeMap;
use std::fmt;

// ---------------------------------------------------------------- dtypes

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub enum DType {
    F32,
    F16,
    BF16,
    F8E4M3,
    F8E5M2,
    U8,
    I8,
    I32,
    I64,
    Bool,
}

impl DType {
    pub fn bits(self) -> usize {
        match self {
            DType::F32 | DType::I32 => 32,
            DType::F16 | DType::BF16 => 16,
            DType::F8E4M3 | DType::F8E5M2 | DType::U8 | DType::I8 => 8,
            DType::I64 => 64,
            DType::Bool => 8,
        }
    }
    pub fn bytes(self) -> usize {
        self.bits() / 8
    }
    /// C type used for a device pointer to this dtype.
    pub fn c_type(self) -> &'static str {
        match self {
            DType::F32 => "float",
            DType::F16 => "__half",
            DType::BF16 => "__nv_bfloat16",
            DType::F8E4M3 => "__nv_fp8_e4m3",
            DType::F8E5M2 => "__nv_fp8_e5m2",
            DType::U8 | DType::Bool => "uint8_t",
            DType::I8 => "int8_t",
            DType::I32 => "int",
            DType::I64 => "long long",
        }
    }
    pub fn from_st(s: &str) -> Option<DType> {
        Some(match s {
            "F32" => DType::F32,
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            "F8_E4M3" => DType::F8E4M3,
            "F8_E5M2" => DType::F8E5M2,
            "U8" => DType::U8,
            "I8" => DType::I8,
            "I32" => DType::I32,
            "I64" => DType::I64,
            "BOOL" => DType::Bool,
            _ => return None,
        })
    }
    pub fn st_name(self) -> &'static str {
        match self {
            DType::F32 => "F32",
            DType::F16 => "F16",
            DType::BF16 => "BF16",
            DType::F8E4M3 => "F8_E4M3",
            DType::F8E5M2 => "F8_E5M2",
            DType::U8 => "U8",
            DType::I8 => "I8",
            DType::I32 => "I32",
            DType::I64 => "I64",
            DType::Bool => "BOOL",
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.st_name().to_lowercase())
    }
}

// ---------------------------------------------------------------- quantisation

/// How the bytes of a weight matrix encode its values.  This is the single
/// switch that selects a gemv core in codegen, so every new format is one
/// variant here plus one core in the runtime library.
#[derive(Clone, PartialEq, Debug)]
pub enum Quant {
    /// Dense, values stored directly in `dtype`.
    Dense { dtype: DType },
    /// MXFP4: `block` e2m1 nibbles share one u8 power-of-two (e8m0) scale.
    /// Storage: blocks u8[rows][K/block][block/2], scales u8[rows][K/block].
    Mxfp4 { block: usize },
    /// FP8 e4m3 values with per-(bn x bk) block scales.  `bk == 0` means one
    /// scale per output row (the compressed-tensors per-channel form).
    Fp8Block { bn: usize, bk: usize },
    /// 4-bit integers, `group` values share one scale (+ zero point).
    /// `awq` distinguishes the two packings that share these tensor names:
    /// AWQ packs eight OUTPUT rows per word, GPTQ eight reduction steps.
    Int4Group {
        group: usize,
        sym: bool,
        awq: bool,
        scale_dtype: DType,
    },
}

impl Quant {
    /// Bytes needed to store `rows x k` values under this scheme.
    pub fn bytes(&self, rows: usize, k: usize) -> usize {
        match self {
            Quant::Dense { dtype } => rows * k * dtype.bytes(),
            Quant::Mxfp4 { block } => rows * (k / block) * (block / 2 + 1),
            Quant::Fp8Block { bn, bk } => {
                let bk = if *bk == 0 { k } else { *bk };
                rows * k + div_up(rows, *bn) * div_up(k, bk) * 4
            }
            Quant::Int4Group {
                group, sym, scale_dtype, ..
            } => {
                let g = k / group;
                rows * k / 2 + rows * g * scale_dtype.bytes() * if *sym { 1 } else { 2 }
            }
        }
    }
    /// The gemv core this format runs on -- and therefore the key the machine
    /// model is indexed by.  A rate is a property of the CORE, not of the
    /// format: fp8 with a 128x128 block scale and fp8 with a per-output-channel
    /// scale differ only in where one scalar comes from, and AWQ and GPTQ int4
    /// are the same nibbles after the loader has repacked them.  Calibrating
    /// each spelling separately would multiply the calibration for nothing --
    /// and, worse, leave any format the calibration happened not to spell
    /// exactly right falling back to a proxy.
    pub fn core(&self) -> &'static str {
        match self {
            Quant::Dense { dtype: DType::F16 } => "f16",
            Quant::Dense { .. } => "bf16",
            Quant::Mxfp4 { .. } => "mxfp4",
            Quant::Fp8Block { .. } => "fp8",
            Quant::Int4Group { .. } => "int4",
        }
    }

    /// The format's exact identity, for the loader, the report and build.json.
    pub fn tag(&self) -> String {
        match self {
            Quant::Dense { dtype } => format!("{}", dtype),
            Quant::Mxfp4 { block } => format!("mxfp4x{}", block),
            Quant::Fp8Block { bk: 0, .. } => "fp8ch".to_string(),
            Quant::Fp8Block { bn, bk } => format!("fp8b{}x{}", bn, bk),
            Quant::Int4Group { group, sym, awq, .. } => {
                format!("int4{}g{}{}", if *awq { "awq" } else { "gptq" }, group,
                        if *sym { "s" } else { "a" })
            }
        }
    }
}

pub fn div_up(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

// ---------------------------------------------------------------- weights

/// A slice of bytes inside one safetensors file.
#[derive(Clone, Debug)]
pub struct Slab {
    #[allow(dead_code)] // carried for diagnostics: which tensor a byte range came from
    pub name: String,
    pub file: usize,
    pub offset: u64,
    pub nbytes: u64,
    pub dtype: DType,
    pub shape: Vec<usize>,
}

/// A logical weight the kernel needs, and the recipe for building it on the
/// device from one or more safetensors slabs.  The recipe is resolved at
/// compile time and baked into the generated loader as a constant table --
/// there is no runtime name lookup and no runtime reshaping decision.
#[derive(Clone, Debug)]
pub struct Weight {
    #[allow(dead_code)] // the map key is authoritative; this is for diagnostics
    pub role: String,
    pub parts: Vec<Slab>,  // concatenated along dim 0 in this order
    pub out_dtype: DType,  // dtype after the (possibly casting) copy
    pub shape: Vec<usize>, // logical shape of the assembled tensor
    pub transform: Xform,
    /// destination rows produced per source row.  Orthogonal to `transform`:
    /// a block-scale tensor is both expanded (one row per weight row) and, for
    /// a gate/up matrix, interleaved -- the two compose.
    pub repeat: usize,
    /// element-level repack, for formats whose packing is not a row mapping.
    /// Also orthogonal: an AWQ gate/up matrix is repacked AND interleaved.
    pub repack: Repack,
}

/// Formats whose storage is not a permutation of rows, and so cannot be
/// expressed as a row mapping.  Each one is a kernel in the generated loader.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Repack {
    None,
    /// AWQ `qweight` [K, rows/8] int32 -> canonical nibbles [rows][K/2].  AWQ
    /// packs eight OUTPUT rows into one word, in the order {0,2,4,6,1,3,5,7},
    /// so unpacking it is a transpose and a permutation at once.
    AwqQ,
    /// AWQ `qzeros` [K/G, rows/8] int32 -> [rows][K/G] halves, same packing.
    AwqZ,
    /// `scales` [K/G, rows] -> [rows][K/G].
    ScalesT,
}

/// The only transformations a loader may perform.  Deliberately tiny: anything
/// that is not one of these is a modelling decision that belongs in the IR.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Xform {
    /// Byte-for-byte copy (after optional dtype cast), parts concatenated.
    Concat,
    /// Concatenate, then convert a [rows, k] fp32/f16 tensor to `out_dtype`.
    Cast,
    /// Interleave two parts row-wise in pairs: used to build a gate/up matrix
    /// whose even rows are gate and odd rows are up (halves smem traffic).
    InterleavePairs,
    /// Source rows are already interleaved (gpt-oss gate_up); keep as is.
    Identity,
    /// Source is a blocked [gate(0..I) ; up(I..2I)] matrix (Phi-3 style);
    /// emit it interleaved so one warp computes a gate/up pair together.
    BlockedToInterleaved,


}

// ---------------------------------------------------------------- norms / rope / acts

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum NormKind {
    /// x * rsqrt(mean(x^2) + eps) * w
    Rms,
    /// as Rms but scaled by (1 + w) -- Gemma
    RmsOnePlus,
    /// (x - mean) * rsqrt(var + eps) * w + b
    Layer,
}

#[derive(Clone, Copy, Debug)]
pub struct Norm {
    pub kind: NormKind,
    pub eps: f32,
    /// LayerNorm's beta.  Read by the LayerNorm lowering only.
    #[allow(dead_code)]
    pub bias: bool,
}

#[derive(Clone, Debug)]
pub enum RopeScaling {
    None,
    Linear { factor: f32 },
    /// Llama-3 style piecewise wavelength rescaling.
    Llama3 {
        factor: f32,
        low_freq_factor: f32,
        high_freq_factor: f32,
        original_max_position: usize,
    },
    /// YaRN (used by Qwen3 long-context, gpt-oss).
    Yarn {
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        original_max_position: usize,
        attn_factor: f32,
        truncate: bool,
    },
}

#[derive(Clone, Debug)]
pub struct Rope {
    pub theta: f32,
    pub scaling: RopeScaling,
    /// fraction of head_dim that is rotated (Phi-3 partial rope)
    pub rotary_dim: usize,
    /// true  => pairs are (i, i+rotary_dim/2)  [HF "neox" / default]
    /// false => pairs are (2i, 2i+1)           [GPT-J "interleaved"]
    pub halves: bool,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Act {
    Silu,
    Gelu,
    GeluTanh,
    /// gpt-oss: clamped swiglu with an alpha and a +1 on the linear branch
    SwigluClamped { alpha: f32, limit: f32 },
    Relu2,
}

// ---------------------------------------------------------------- attention

#[derive(Clone, Debug)]
pub struct Attention {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
    pub qkv_bias: bool,
    pub o_bias: bool,
    pub q_norm: Option<Norm>,
    pub k_norm: Option<Norm>,
    /// per-layer sliding window; None on a layer means full attention
    pub window: Vec<Option<usize>>,
    /// learned per-head attention sink logit (gpt-oss)
    pub sinks: bool,
    /// tanh soft-cap on attention logits (Gemma-2)
    pub logit_softcap: Option<f32>,
    pub kv_dtype: DType,
}

// ---------------------------------------------------------------- FFN

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum RouterScore {
    Softmax,
    Sigmoid,
}

#[derive(Clone, Debug)]
pub struct MoeCfg {
    pub n_experts: usize,
    pub top_k: usize,
    pub score: RouterScore,
    /// renormalise the top-k weights to sum to 1
    pub norm_topk: bool,
    /// softmax over the top-k logits only (gpt-oss) rather than all experts
    pub softmax_after_topk: bool,
    pub router_bias: bool,
    /// per-expert bias added to the router logits for selection only (DeepSeek)
    pub router_correction_bias: bool,
    /// layers that are dense MLPs rather than MoE.  The kernel emits one layer
    /// body for all layers, so a model that mixes them is refused rather than
    /// silently given experts everywhere.
    pub dense_layers: bool,
    pub expert_bias: bool,
    /// DeepSeek spells the shared-expert count separately from its size; the
    /// size is what the planner needs, so this is carried for diagnostics only.
    #[allow(dead_code)]
    pub n_shared: usize,
    pub routed_scale: f32,
}

#[derive(Clone, Debug)]
pub enum Ffn {
    Dense {
        intermediate: usize,
        act: Act,
        #[allow(dead_code)] // mlp_bias: bound when present, unused by the current lowering
        bias: bool,
    },
    Moe {
        intermediate: usize,
        act: Act,
        cfg: MoeCfg,
        shared_intermediate: usize,
    },
}

impl Ffn {
    pub fn intermediate(&self) -> usize {
        match self {
            Ffn::Dense { intermediate, .. } => *intermediate,
            Ffn::Moe { intermediate, .. } => *intermediate,
        }
    }
    pub fn act(&self) -> Act {
        match self {
            Ffn::Dense { act, .. } => *act,
            Ffn::Moe { act, .. } => *act,
        }
    }
}

// ---------------------------------------------------------------- model

#[derive(Clone, Debug)]
pub struct Model {
    pub name: String,
    pub arch: String,
    pub hidden: usize,
    pub n_layers: usize,
    pub vocab: usize,
    pub attn: Attention,
    pub ffn: Ffn,
    pub attn_norm: Norm,
    pub ffn_norm: Norm,
    pub final_norm: Norm,
    /// Gemma-3 style extra norms after attention / after the MLP
    pub post_attn_norm: bool,
    pub post_ffn_norm: bool,
    pub rope: Rope,
    pub tie_embeddings: bool,
    pub logit_softcap: Option<f32>,
    /// Gemma scales the embedding by sqrt(hidden)
    pub embed_scale: Option<f32>,
    /// Reported by `analyze`; the compiled kernel sizes its cache from `--ctx`.
    #[allow(dead_code)]
    pub max_position: usize,
    /// weight quantisation per matrix class
    pub q_attn: Quant,
    pub q_ffn: Quant,
    pub q_lm: Quant,
    /// activation dtype the reference materialises between ops
    pub act_dtype: DType,
    /// resolved weight table, role -> recipe
    pub weights: BTreeMap<String, Weight>,
    /// safetensors files, in index order
    pub files: Vec<String>,
    /// Carried through for a generation front end; the gate is teacher-forced.
    #[allow(dead_code)]
    pub eos_ids: Vec<u32>,
    #[allow(dead_code)]
    pub bos_id: Option<u32>,
}

impl Model {
    pub fn q_dim(&self) -> usize {
        self.attn.n_heads * self.attn.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.attn.n_kv_heads * self.attn.head_dim
    }
    pub fn qkv_dim(&self) -> usize {
        self.q_dim() + 2 * self.kv_dim()
    }
    pub fn is_moe(&self) -> bool {
        matches!(self.ffn, Ffn::Moe { .. })
    }
    pub fn moe(&self) -> Option<&MoeCfg> {
        match &self.ffn {
            Ffn::Moe { cfg, .. } => Some(cfg),
            _ => None,
        }
    }
}
