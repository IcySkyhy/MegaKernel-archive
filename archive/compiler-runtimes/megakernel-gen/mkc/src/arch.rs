//! Architecture frontend: HuggingFace config + tensor index  ->  `ir::Model`.
//!
//! This is deliberately ONE generic decoder-only frontend rather than a family
//! of per-model readers.  Every HF decoder differs only in (a) which config
//! keys are present, (b) which tensor names hold each role, (c) a handful of
//! numerical conventions.  All three are probed, so a model this compiler has
//! never seen still lands if it follows the house style -- and fails loudly,
//! naming the missing role, if it does not.

use crate::hf::HfModel;
use crate::ir::*;
use crate::quant;
use serde_json::Value;
use std::collections::BTreeMap;

pub struct Frontend<'a> {
    hf: &'a HfModel,
    w: BTreeMap<String, Weight>,
    missing: Vec<String>,
}

/// Build the layer-name prefix used by this checkpoint (`model.layers.` for
/// nearly everything, `transformer.h.` for GPT-2 descendants).
fn layer_prefix(hf: &HfModel) -> String {
    for p in [
        "model.layers.",
        "model.language_model.layers.",
        "language_model.model.layers.",
        "transformer.h.",
        "layers.",
    ] {
        if hf.tensors.keys().any(|k| k.starts_with(p)) {
            return p.to_string();
        }
    }
    "model.layers.".to_string()
}

fn c(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

impl<'a> Frontend<'a> {
    fn new(hf: &'a HfModel) -> Self {
        Frontend {
            hf,
            w: BTreeMap::new(),
            missing: Vec::new(),
        }
    }

    /// Register a weight from the first matching tensor name.  Returns whether
    /// it was found; `required` failures are collected and reported together.
    fn bind(&mut self, role: &str, cands: &[String], out_dtype: DType, xform: Xform,
            required: bool) -> bool {
        self.bind_rep(role, cands, out_dtype, xform, 1, Repack::None, None, required)
    }

    fn bind_rep(
        &mut self,
        role: &str,
        cands: &[String],
        out_dtype: DType,
        xform: Xform,
        repeat: usize,
        repack: Repack,
        dst: Option<(usize, usize)>,
        required: bool,
    ) -> bool {
        if let Some(s) = self.hf.any(cands) {
            let shape = match dst { Some((r, k)) => vec![r, k], None => s.shape.clone() };
            self.w.insert(
                role.to_string(),
                Weight {
                    role: role.to_string(),
                    parts: vec![s.clone()],
                    out_dtype,
                    shape,
                    transform: xform,
                    repeat,
                    repack,
                },
            );
            true
        } else {
            if required {
                self.missing.push(format!("{} (tried {:?})", role, cands));
            }
            false
        }
    }

    /// Register a weight assembled by concatenating several tensors along dim 0.
    fn bind_concat(&mut self, role: &str, groups: &[Vec<String>], out_dtype: DType,
                   xform: Xform, required: bool) -> bool {
        self.bind_concat_rep(role, groups, out_dtype, xform, 1, Repack::None, None, required)
    }

    fn bind_concat_rep(
        &mut self,
        role: &str,
        groups: &[Vec<String>],
        out_dtype: DType,
        xform: Xform,
        repeat: usize,
        repack: Repack,
        dst: Option<(usize, usize)>,
        required: bool,
    ) -> bool {
        let mut parts = Vec::new();
        for g in groups {
            match self.hf.any(g) {
                Some(s) => parts.push(s.clone()),
                None => {
                    if required {
                        self.missing.push(format!("{} part {:?}", role, g));
                    }
                    return false;
                }
            }
        }
        let mut shape = match dst { Some((r, k)) => vec![r, k], None => parts[0].shape.clone() };
        if dst.is_none() && !shape.is_empty() {
            shape[0] = parts.iter().map(|p| p.shape[0]).sum();
        }
        self.w.insert(
            role.to_string(),
            Weight {
                role: role.to_string(),
                parts,
                out_dtype,
                shape,
                transform: xform,
                repeat,
                repack,
            },
        );
        true
    }

    /// Bind every tensor one quantised matrix needs.
    ///
    /// `bases` are candidate tensor names with the trailing `.weight` REMOVED;
    /// the format's suffixes from `quant::hf_suffixes` are appended.  Each group
    /// is one source matrix; several groups are concatenated (q/k/v into qkv) or
    /// interleaved (gate/up into one channel-paired matrix) by `xform`.
    ///
    /// Block-scale tensors are expanded to one scale per row here, so the kernel
    /// sees a single canonical form whatever the checkpoint stored.
    fn bind_mat(&mut self, role: &str, bases: &[Vec<String>], q: &Quant, act: DType,
                xform: Xform, required: bool) -> bool {
        let sufs = quant::hf_suffixes(q);
        let parts = quant::parts(q, act);
        let mut ok = true;
        for (i, (cands, part)) in sufs.iter().zip(parts.iter()).enumerate() {
            // a scale tensor holds one row per block of `bn` weight rows
            let repeat = match q {
                Quant::Fp8Block { bn, .. } if i > 0 => *bn,
                _ => 1,
            };
            let repack = match q {
                Quant::Int4Group { .. } => match i {
                    0 => Repack::AwqQ,
                    1 => Repack::ScalesT,
                    _ => Repack::AwqZ,
                },
                _ => Repack::None,
            };
            let groups: Vec<Vec<String>> = bases
                .iter()
                .map(|g| g.iter().flat_map(|b| cands.iter().map(move |c| format!("{}{}", b, c)))
                          .collect())
                .collect();
            let r = format!("{}_{}", role, part.suffix);
            // canonical destination shape, for the formats that are repacked
            let dst = if repack != Repack::None {
                // derive (rows, K) from the first source: AWQ stores [K, rows/8]
                // canonical shapes: qweight -> [rows, K], scales/zeros -> [rows, K/G].
                // The stored tensors are transposed, so `rows` comes from dim 1.
                self.hf.any(&groups[0]).map(|sl| match repack {
                    Repack::AwqQ => (sl.shape[1] * 8 * groups.len(), sl.shape[0]),
                    Repack::AwqZ => (sl.shape[1] * 8 * groups.len(), sl.shape[0]),
                    _ => (sl.shape[1] * groups.len(), sl.shape[0]),
                })
            } else {
                None
            };
            ok &= if groups.len() == 1 {
                self.bind_rep(&r, &groups[0], part.dtype, xform, repeat, repack, dst, required)
            } else {
                self.bind_concat_rep(&r, &groups, part.dtype, xform, repeat, repack, dst, required)
            };
        }
        ok
    }

    fn has_any(&self, cands: &[String]) -> bool {
        self.hf.any(cands).is_some()
    }

    /// Does this base name exist in the given format (checked on its first part)?
    fn has_base(&self, bases: &[String], q: &Quant) -> bool {
        let sfx = &quant::hf_suffixes(q)[0];
        bases.iter().any(|b| sfx.iter().any(|s| self.hf.tensors.contains_key(&format!("{}{}", b, s))))
    }
}

fn act_from_str(s: &str) -> Act {
    match s {
        "silu" | "swish" => Act::Silu,
        "gelu" => Act::Gelu,
        "gelu_new" | "gelu_pytorch_tanh" | "gelu_tanh" | "gelu_fast" => Act::GeluTanh,
        "relu2" | "relu_squared" => Act::Relu2,
        _ => Act::Silu,
    }
}

fn rope_scaling(hf: &HfModel, max_pos: usize) -> RopeScaling {
    let Some(rs) = hf.get("rope_scaling") else {
        return RopeScaling::None;
    };
    if rs.is_null() {
        return RopeScaling::None;
    }
    let ty = rs
        .get("rope_type")
        .or_else(|| rs.get("type"))
        .and_then(|v| v.as_str())
        .unwrap_or("default");
    let f = |k: &str, d: f32| rs.get(k).and_then(|v| v.as_f64()).unwrap_or(d as f64) as f32;
    let orig = rs
        .get("original_max_position_embeddings")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .or_else(|| hf.usize_of("initial_context_length"))
        .unwrap_or(max_pos);
    match ty {
        "linear" => RopeScaling::Linear { factor: f("factor", 1.0) },
        "llama3" => RopeScaling::Llama3 {
            factor: f("factor", 8.0),
            low_freq_factor: f("low_freq_factor", 1.0),
            high_freq_factor: f("high_freq_factor", 4.0),
            original_max_position: orig,
        },
        "yarn" => RopeScaling::Yarn {
            factor: f("factor", 1.0),
            beta_fast: f("beta_fast", 32.0),
            beta_slow: f("beta_slow", 1.0),
            original_max_position: orig,
            attn_factor: f("attention_factor", 1.0),
            truncate: rs
                .get("truncate")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
        },
        _ => RopeScaling::None,
    }
}

/// Per-layer sliding window, from whichever of the four conventions this
/// checkpoint uses.
fn windows(hf: &HfModel, n_layers: usize) -> Vec<Option<usize>> {
    let sw = hf.usize_of("sliding_window").filter(|v| *v > 0);
    // 1. explicit layer_types (gpt-oss, gemma-3, qwen3-next)
    if let Some(Value::Array(a)) = hf.get("layer_types") {
        return a
            .iter()
            .map(|v| match v.as_str() {
                Some("sliding_attention") => sw,
                _ => None,
            })
            .collect();
    }
    let Some(sw) = sw else {
        return vec![None; n_layers];
    };
    // 2. gemma-2/3: every `sliding_window_pattern`-th layer is full attention.
    //    Gemma-2 does not write the key at all -- the alternation is a property
    //    of the architecture, and transformers hardcodes 2.
    let pat = hf.usize_of("sliding_window_pattern").or_else(|| {
        if hf.model_type() == "gemma2" { Some(2) } else { None }
    });
    if let Some(p) = pat.filter(|p| *p > 1) {
        return (0..n_layers)
            .map(|i| if (i + 1) % p == 0 { None } else { Some(sw) })
            .collect();
    }
    // 3. qwen2/3: sliding only above `max_window_layers`, and only if enabled
    if hf.bool_of("use_sliding_window") == Some(false) {
        return vec![None; n_layers];
    }
    if let Some(mw) = hf.usize_of("max_window_layers") {
        return (0..n_layers)
            .map(|i| if i >= mw { Some(sw) } else { None })
            .collect();
    }
    // 4. uniform sliding window (mistral)
    vec![Some(sw); n_layers]
}

/// Detect the weight quantisation of a matrix class from the checkpoint.
fn detect_quant(hf: &HfModel, dense_dtype: DType) -> Quant {
    let qc = hf.config.get("quantization_config");
    if let Some(qc) = qc {
        let m = qc
            .get("quant_method")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if m == "mxfp4" {
            // [..., G, 16] u8 blocks: 16 bytes of nibbles share one e8m0 scale
            return Quant::Mxfp4 { block: 32 };
        }
        if m == "compressed-tensors" {
            // per-output-channel fp8: one scale per row, no k blocking
            return Quant::Fp8Block { bn: 1, bk: 0 };
        }
        let group = qc
            .get("group_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(128) as usize;
        match m {
            "awq" => {
                return Quant::Int4Group {
                    group,
                    sym: false,
                    awq: true,
                    scale_dtype: DType::F16,
                }
            }
            "gptq" => {
                // Same tensor NAMES as AWQ, different packing: GPTQ puts eight
                // reduction steps in a word, AWQ eight output rows.  Detected so
                // that it is refused by name rather than repacked as if it were
                // AWQ, which would be silently wrong.
                return Quant::Int4Group {
                    group,
                    sym: qc.get("sym").and_then(|v| v.as_bool()).unwrap_or(true),
                    awq: false,
                    scale_dtype: DType::F16,
                }
            }
            "fp8" => {
                // `weight_block_size` absent means per-output-channel scales
                let bs = qc
                    .get("weight_block_size")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        (
                            a[0].as_u64().unwrap_or(128) as usize,
                            a[1].as_u64().unwrap_or(128) as usize,
                        )
                    })
                    .unwrap_or((1, 0));
                return Quant::Fp8Block { bn: bs.0, bk: bs.1 };
            }
            _ => {}
        }
    }
    Quant::Dense { dtype: dense_dtype }
}

/// The per-layer weight binding.
///
/// EXTENDING: adding a model family is almost always one tensor name added to
/// one candidate list in one of these three methods.  They are separate so that
/// "where do the attention weights come from" is a question with a 60-line
/// answer instead of a 240-line one.
struct LayerBind<'a> {
    lp: &'a str,
    act: DType,
    q_attn: &'a Quant,
    q_ffn: &'a Quant,
    ffn: &'a Ffn,
    attn_bias: bool,
    o_bias: bool,
    has_qk_norm: bool,
    has_sinks: bool,
    shared_inter: usize,
}

const F32: DType = DType::F32;

impl LayerBind<'_> {
    /// This checkpoint's prefix for layer `i`, e.g. `model.layers.7.`
    fn at(&self, i: usize) -> String {
        format!("{}{}.", self.lp, i)
    }

    fn bind(&self, fe: &mut Frontend, i: usize) {
        self.attention(fe, i);
        self.norms(fe, i);
        self.feedforward(fe, i);
    }

    fn attention(&self, fe: &mut Frontend, i: usize) {
        let p = self.at(i);
        let r = |s: &str| format!("l{}.{}", i, s);
        fe.bind(
            &r("attn_norm"),
            &c(&[
                &format!("{}input_layernorm.weight", p),
                &format!("{}ln_1.weight", p),
            ]),
            F32,
            Xform::Cast,
            true,
        );
        fe.bind(
            &r("attn_norm"),
            &c(&[
                &format!("{}input_layernorm.weight", p),
                &format!("{}ln_1.weight", p),
            ]),
            F32,
            Xform::Cast,
            true,
        );
        // q,k,v -> one qkv matrix (a fused source is accepted too)
        let fused_qkv = c(&[
            &format!("{}self_attn.qkv_proj", p),
            &format!("{}attn.c_attn", p),
        ]);
        if fe.has_base(&fused_qkv, self.q_attn) {
            fe.bind_mat(&r("qkv"), &[fused_qkv.clone()], self.q_attn, self.act, Xform::Concat, true);
            if self.attn_bias {
                fe.bind(
                    &r("qkv_b"),
                    &c(&[&format!("{}self_attn.qkv_proj.bias", p)]),
                    F32,
                    Xform::Cast,
                    false,
                );
            }
        } else {
            fe.bind_mat(
                &r("qkv"),
                &[
                    c(&[&format!("{}self_attn.q_proj", p)]),
                    c(&[&format!("{}self_attn.k_proj", p)]),
                    c(&[&format!("{}self_attn.v_proj", p)]),
                ],
                self.q_attn, self.act, Xform::Concat, true,
            );
            if self.attn_bias {
                fe.bind_concat(
                    &r("qkv_b"),
                    &[
                        c(&[&format!("{}self_attn.q_proj.bias", p)]),
                        c(&[&format!("{}self_attn.k_proj.bias", p)]),
                        c(&[&format!("{}self_attn.v_proj.bias", p)]),
                    ],
                    F32,
                    Xform::Cast,
                    true,
                );
            }
        }
        if self.has_qk_norm {
            fe.bind(
                &r("q_norm"),
                &c(&[&format!("{}self_attn.q_norm.weight", p)]),
                F32,
                Xform::Cast,
                true,
            );
            fe.bind(
                &r("k_norm"),
                &c(&[&format!("{}self_attn.k_norm.weight", p)]),
                F32,
                Xform::Cast,
                true,
            );
        }
        if self.has_sinks {
            fe.bind(
                &r("sinks"),
                &c(&[
                    &format!("{}self_attn.sinks", p),
                    &format!("{}self_attn.sinks.weight", p),
                ]),
                F32,
                Xform::Cast,
                true,
            );
        }
        fe.bind_mat(
            &r("o"),
            &[c(&[&format!("{}self_attn.o_proj", p), &format!("{}attn.c_proj", p)])],
            self.q_attn, self.act, Xform::Concat, true,
        );
        if self.o_bias {
            fe.bind(
                &r("o_b"),
                &c(&[&format!("{}self_attn.o_proj.bias", p)]),
                F32,
                Xform::Cast,
                false,
            );
        }
    }

    fn norms(&self, fe: &mut Frontend, i: usize) {
        let p = self.at(i);
        let r = |s: &str| format!("l{}.{}", i, s);
        // Gemma-2 and -3 have FOUR norms per layer and reuse the name
        // `post_attention_layernorm` for the one applied to the ATTENTION OUTPUT,
        // where every other model uses it for the pre-MLP norm.  A checkpoint
        // that also has `pre_feedforward_layernorm` is telling us which
        // convention it means; getting this wrong is silent and catastrophic.
        let pre_ffn = format!("{}pre_feedforward_layernorm.weight", p);
        let post_attn = format!("{}post_attention_layernorm.weight", p);
        let gemma_norms = fe.has_any(&[pre_ffn.clone()]);
        let ffn_norm_name = if gemma_norms { pre_ffn.clone() } else { post_attn.clone() };
        fe.bind(
            &r("ffn_norm"),
            &[ffn_norm_name, format!("{}ln_2.weight", p)],
            F32,
            Xform::Cast,
            true,
        );
        let mut post_attn_cands = vec![format!("{}post_self_attn_layernorm.weight", p)];
        if gemma_norms {
            post_attn_cands.push(post_attn);
        }
        fe.bind(&r("post_attn_norm"), &post_attn_cands, F32, Xform::Cast, false);
        fe.bind(
            &r("post_ffn_norm"),
            &c(&[&format!("{}post_feedforward_layernorm.weight", p)]),
            F32,
            Xform::Cast,
            false,
        );
    }

    fn feedforward(&self, fe: &mut Frontend, i: usize) {
        let p = self.at(i);
        let r = |s: &str| format!("l{}.{}", i, s);
        match self.ffn {
            Ffn::Dense { .. } => {
                let fused = c(&[&format!("{}mlp.gate_up_proj", p)]);
                if fe.has_base(&fused, self.q_ffn) {
                    fe.bind_mat(&r("gu"), &[fused], self.q_ffn, self.act,
                                Xform::BlockedToInterleaved, true);
                } else {
                    fe.bind_mat(
                        &r("gu"),
                        &[
                            c(&[&format!("{}mlp.gate_proj", p)]),
                            c(&[&format!("{}mlp.up_proj", p)]),
                        ],
                        self.q_ffn, self.act, Xform::InterleavePairs, true,
                    );
                }
                fe.bind_mat(&r("dn"), &[c(&[&format!("{}mlp.down_proj", p)])],
                            self.q_ffn, self.act, Xform::Concat, true);
            }
            Ffn::Moe { cfg, .. } => {
                fe.bind(
                    &r("router_w"),
                    &c(&[
                        &format!("{}mlp.router.weight", p),
                        &format!("{}mlp.gate.weight", p),
                    ]),
                    self.act,
                    Xform::Concat,
                    true,
                );
                if cfg.router_bias {
                    fe.bind(
                        &r("router_b"),
                        &c(&[
                            &format!("{}mlp.router.bias", p),
                            &format!("{}mlp.gate.bias", p),
                        ]),
                        F32,
                        Xform::Cast,
                        false,
                    );
                }
                if cfg.router_correction_bias {
                    fe.bind(
                        &r("router_cb"),
                        &c(&[&format!("{}mlp.gate.e_score_correction_bias", p)]),
                        F32,
                        Xform::Cast,
                        false,
                    );
                }
                // Experts come either as one fused [E, 2I, H] tensor (gpt-oss,
                // recent Qwen3-MoE) or as E separate matrices (Mixtral, older
                // Qwen3-MoE).  Both land in the same interleaved gate/up layout.
                let fused = c(&[&format!("{}mlp.experts.gate_up_proj", p)]);
                if fe.has_base(&fused, self.q_ffn) {
                    fe.bind_mat(&r("egu"), &[fused], self.q_ffn, self.act, Xform::Identity, true);
                    fe.bind_mat(&r("edn"), &[c(&[&format!("{}mlp.experts.down_proj", p)])],
                                self.q_ffn, self.act, Xform::Identity, true);
                } else {
                    let mut gu = Vec::new();
                    let mut dn = Vec::new();
                    for e in 0..cfg.n_experts {
                        gu.push(c(&[&format!("{}mlp.experts.{}.gate_proj", p, e),
                                    &format!("{}mlp.experts.{}.w1", p, e)]));
                        gu.push(c(&[&format!("{}mlp.experts.{}.up_proj", p, e),
                                    &format!("{}mlp.experts.{}.w3", p, e)]));
                        dn.push(c(&[&format!("{}mlp.experts.{}.down_proj", p, e),
                                    &format!("{}mlp.experts.{}.w2", p, e)]));
                    }
                    fe.bind_mat(&r("egu"), &gu, self.q_ffn, self.act, Xform::InterleavePairs, true);
                    fe.bind_mat(&r("edn"), &dn, self.q_ffn, self.act, Xform::Concat, true);
                }
                if cfg.expert_bias {
                    fe.bind(
                        &r("egu_b"),
                        &c(&[&format!("{}mlp.experts.gate_up_proj_bias", p)]),
                        F32,
                        Xform::Cast,
                        false,
                    );
                    fe.bind(
                        &r("edn_b"),
                        &c(&[&format!("{}mlp.experts.down_proj_bias", p)]),
                        F32,
                        Xform::Cast,
                        false,
                    );
                }
                if self.shared_inter > 0 {
                    fe.bind_mat(
                        &r("sgu"),
                        &[
                            c(&[&format!("{}mlp.shared_expert.gate_proj", p),
                                &format!("{}mlp.shared_experts.gate_proj", p)]),
                            c(&[&format!("{}mlp.shared_expert.up_proj", p),
                                &format!("{}mlp.shared_experts.up_proj", p)]),
                        ],
                        self.q_ffn, self.act, Xform::InterleavePairs, true,
                    );
                    fe.bind_mat(
                        &r("sdn"),
                        &[c(&[&format!("{}mlp.shared_expert.down_proj", p),
                              &format!("{}mlp.shared_experts.down_proj", p)])],
                        self.q_ffn, self.act, Xform::Concat, true,
                    );
                    // DeepSeek's shared experts are always on; Qwen2-MoE gates
                    // them with a sigmoid of a 1 x H projection
                    fe.bind(
                        &r("sgate_w"),
                        &c(&[&format!("{}mlp.shared_expert_gate.weight", p)]),
                        self.act,
                        Xform::Concat,
                        false,
                    );
                }
            }
        }
    }
}

pub fn build(hf: &HfModel, name: &str) -> Result<Model, String> {
    let lp = layer_prefix(hf);
    let mut fe = Frontend::new(hf);

    let hidden = hf
        .usize_of("hidden_size")
        .or_else(|| hf.usize_of("n_embd"))
        .ok_or("config: no hidden_size")?;
    let n_layers = hf
        .usize_of("num_hidden_layers")
        .or_else(|| hf.usize_of("n_layer"))
        .ok_or("config: no num_hidden_layers")?;
    let n_heads = hf
        .usize_of("num_attention_heads")
        .or_else(|| hf.usize_of("n_head"))
        .ok_or("config: no num_attention_heads")?;
    let n_kv_heads = hf.usize_of("num_key_value_heads").unwrap_or(n_heads);
    let head_dim = hf.usize_of("head_dim").unwrap_or(hidden / n_heads);
    let vocab = hf.usize_of("vocab_size").ok_or("config: no vocab_size")?;
    let max_position = hf.usize_of("max_position_embeddings").unwrap_or(8192);

    // ---- activation dtype: whatever the embedding is stored in ------------
    let embed_cands = c(&[
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
        "transformer.wte.weight",
        "embed_tokens.weight",
    ]);
    let act_dtype = hf
        .any(&embed_cands)
        .map(|s| match s.dtype {
            DType::F16 => DType::F16,
            _ => DType::BF16,
        })
        .unwrap_or(DType::BF16);

    // ---- norms -----------------------------------------------------------
    let mt = hf.model_type();
    let is_gemma = mt.starts_with("gemma");
    let norm_kind = if is_gemma { NormKind::RmsOnePlus } else { NormKind::Rms };
    let eps = hf
        .f32_of("rms_norm_eps")
        .or_else(|| hf.f32_of("layer_norm_epsilon"))
        .or_else(|| hf.f32_of("layer_norm_eps"))
        .unwrap_or(1e-5);
    let nrm = Norm { kind: norm_kind, eps, bias: false };

    // ---- rope ------------------------------------------------------------
    let partial = hf.f32_of("partial_rotary_factor").unwrap_or(1.0);
    let rope = Rope {
        theta: hf
            .f32_of("rope_theta")
            .or_else(|| hf.f32_of("rotary_emb_base"))
            .unwrap_or(10000.0),
        scaling: rope_scaling(hf, max_position),
        rotary_dim: ((head_dim as f32 * partial) as usize / 2) * 2,
        halves: true,
    };

    // ---- attention -------------------------------------------------------
    let scale = hf
        .f32_of("query_pre_attn_scalar")
        .map(|s| 1.0 / s.sqrt())
        .unwrap_or_else(|| 1.0 / (head_dim as f32).sqrt());
    let attn_bias = hf.bool_of("attention_bias").unwrap_or(false)
        || fe.has_any(&c(&[&format!("{}0.self_attn.q_proj.bias", lp)]));
    let o_bias = fe.has_any(&c(&[&format!("{}0.self_attn.o_proj.bias", lp)]));
    let has_qk_norm = fe.has_any(&c(&[&format!("{}0.self_attn.q_norm.weight", lp)]));
    let has_sinks = fe.has_any(&c(&[
        &format!("{}0.self_attn.sinks", lp),
        &format!("{}0.self_attn.sinks.weight", lp),
    ]));

    let attn = Attention {
        n_heads,
        n_kv_heads,
        head_dim,
        scale,
        qkv_bias: attn_bias,
        o_bias,
        q_norm: if has_qk_norm { Some(nrm) } else { None },
        k_norm: if has_qk_norm { Some(nrm) } else { None },
        window: windows(hf, n_layers),
        sinks: has_sinks,
        logit_softcap: hf.f32_of("attn_logit_softcapping"),
        kv_dtype: act_dtype,
    };

    // ---- ffn -------------------------------------------------------------
    let act = act_from_str(&hf.str_of("hidden_act").unwrap_or_else(|| "silu".into()));
    let n_experts = hf
        .usize_of("num_local_experts")
        .or_else(|| hf.usize_of("num_experts"))
        .or_else(|| hf.usize_of("n_routed_experts"))
        .unwrap_or(0);
    let top_k = hf
        .usize_of("num_experts_per_tok")
        .or_else(|| hf.usize_of("experts_per_token"))
        .or_else(|| hf.usize_of("moe_topk"))
        .unwrap_or(0);
    let dense_inter = hf.usize_of("intermediate_size").unwrap_or(4 * hidden);
    let moe_inter = hf
        .usize_of("moe_intermediate_size")
        .unwrap_or(dense_inter);
    let shared_inter = hf
        .usize_of("shared_expert_intermediate_size")
        .or_else(|| {
            hf.usize_of("n_shared_experts")
                .map(|n| n * hf.usize_of("moe_intermediate_size").unwrap_or(dense_inter))
        })
        .unwrap_or(0);

    let ffn = if n_experts > 0 && top_k > 0 {
        let act = if mt == "gpt_oss" {
            Act::SwigluClamped {
                alpha: 1.702,
                limit: hf.f32_of("swiglu_limit").unwrap_or(7.0),
            }
        } else {
            act
        };
        Ffn::Moe {
            intermediate: moe_inter,
            act,
            shared_intermediate: shared_inter,
            cfg: MoeCfg {
                n_experts,
                top_k,
                score: if hf
                    .str_of("scoring_func")
                    .map(|s| s == "sigmoid")
                    .unwrap_or(false)
                {
                    RouterScore::Sigmoid
                } else {
                    RouterScore::Softmax
                },
                norm_topk: hf
                    .bool_of("norm_topk_prob")
                    .or_else(|| hf.bool_of("norm_topk_prob"))
                    .unwrap_or(mt != "mixtral"),
                softmax_after_topk: mt == "gpt_oss",
                router_bias: fe.has_any(&c(&[
                    &format!("{}0.mlp.router.bias", lp),
                    &format!("{}0.mlp.gate.bias", lp),
                ])),
                router_correction_bias: fe.has_any(&c(&[&format!(
                    "{}0.mlp.gate.e_score_correction_bias",
                    lp
                )])),
                expert_bias: fe.has_any(&c(&[&format!(
                    "{}0.mlp.experts.gate_up_proj_bias",
                    lp
                )])),
                n_shared: hf.usize_of("n_shared_experts").unwrap_or(0),
                // `decoder_sparse_step > 1` or a non-empty `mlp_only_layers`
                // means some layers are dense
                dense_layers: hf.usize_of("decoder_sparse_step").map(|d| d > 1).unwrap_or(false)
                    || hf.get("mlp_only_layers")
                        .and_then(|v| v.as_array())
                        .map(|a| !a.is_empty())
                        .unwrap_or(false)
                    || hf.usize_of("first_k_dense_replace").map(|k| k > 0).unwrap_or(false),
                routed_scale: hf.f32_of("routed_scaling_factor").unwrap_or(1.0),
            },
        }
    } else {
        Ffn::Dense {
            intermediate: dense_inter,
            act,
            bias: hf.bool_of("mlp_bias").unwrap_or(false),
        }
    };

    // ---- quantisation ----------------------------------------------------
    // One detected format, then per matrix class: a checkpoint says which
    // classes it quantised by which companion tensors it actually shipped
    // (`modules_to_not_convert` and `ignore` lists are advisory and
    //  inconsistently spelled; the tensors are not).
    let detected = detect_quant(hf, act_dtype);
    let uses = |base: &str| -> Quant {
        let sufs = quant::hf_suffixes(&detected);
        if sufs.len() < 2 {
            return detected.clone();
        }
        let present = sufs[1].iter().any(|sfx| hf.tensors.contains_key(&format!("{}{}", base, sfx)));
        if present { detected.clone() } else { Quant::Dense { dtype: act_dtype } }
    };
    let q_attn = uses(&format!("{}0.self_attn.q_proj", lp));
    let q_ffn = if n_experts > 0 {
        let a = uses(&format!("{}0.mlp.experts.gate_up_proj", lp));
        if matches!(a, Quant::Dense { .. }) { uses(&format!("{}0.mlp.experts.0.gate_proj", lp)) } else { a }
    } else {
        uses(&format!("{}0.mlp.gate_proj", lp))
    };
    let q_lm = Quant::Dense { dtype: act_dtype };

    // ---- weights ---------------------------------------------------------
    let f32n = DType::F32;
    fe.bind("embed", &embed_cands, act_dtype, Xform::Concat, true);
    fe.bind(
        "final_norm",
        &c(&[
            "model.norm.weight",
            "model.language_model.norm.weight",
            "language_model.model.norm.weight",
            "transformer.ln_f.weight",
            "norm.weight",
        ]),
        f32n,
        Xform::Cast,
        true,
    );
    // A tied model sometimes still ships an `lm_head.weight` that is a copy of
    // the embedding.  Binding it loads hundreds of megabytes the kernel never
    // reads -- caught by the guard's weight-sensitivity probe, which is exactly
    // what that probe is for.
    let tie = hf.bool_of("tie_word_embeddings").unwrap_or(false);
    let lm_found = !tie
        && fe.bind(
            "lm_head",
            &c(&["lm_head.weight", "output.weight", "model.lm_head.weight"]),
            act_dtype,
            Xform::Concat,
            false,
        );
    let tie_embeddings = tie || !lm_found;

    let lb = LayerBind {
        lp: &lp,
        act: act_dtype,
        q_attn: &q_attn,
        q_ffn: &q_ffn,
        ffn: &ffn,
        attn_bias,
        o_bias,
        has_qk_norm,
        has_sinks,
        shared_inter,
    };
    for i in 0..n_layers {
        lb.bind(&mut fe, i);
    }

    if !fe.missing.is_empty() {
        // Adding an architecture is usually one name added to one candidate
        // list, so say what the checkpoint actually contains: the distinct
        // tensor-name shapes, with layer and expert indices collapsed.
        let mut pats: Vec<String> = hf
            .tensors
            .keys()
            .map(|k| {
                let mut out = String::new();
                let mut prev_dot = true;
                for part in k.split('.') {
                    if !prev_dot {
                        out.push('.');
                    }
                    prev_dot = false;
                    if part.chars().all(|c| c.is_ascii_digit()) && !part.is_empty() {
                        out.push('N');
                    } else {
                        out.push_str(part);
                    }
                }
                out
            })
            .collect();
        pats.sort();
        pats.dedup();
        let shown: Vec<String> = pats.into_iter().take(40).collect();
        let mut msg = format!("cannot resolve the weights of `{}` ({}).\n\n", name, hf.arch());
        msg.push_str("missing:\n  ");
        msg.push_str(&fe.missing.join("\n  "));
        msg.push_str("\n\nthe checkpoint contains these tensor names (indices shown as N):\n  ");
        msg.push_str(&shown.join("\n  "));
        msg.push_str("\n\nadding an architecture is usually one name added to one candidate list");
        msg.push_str("\nin mkc/src/arch.rs -- see docs/EXTENDING.md");
        return Err(msg);
    }

    Ok(Model {
        name: name.to_string(),
        arch: hf.arch(),
        hidden,
        n_layers,
        vocab,
        attn,
        ffn,
        attn_norm: nrm,
        ffn_norm: nrm,
        final_norm: nrm,
        post_attn_norm: fe.w.contains_key("l0.post_attn_norm"),
        post_ffn_norm: fe.w.contains_key("l0.post_ffn_norm"),
        rope,
        tie_embeddings,
        logit_softcap: hf.f32_of("final_logit_softcapping"),
        embed_scale: if is_gemma {
            Some((hidden as f32).sqrt())
        } else {
            None
        },
        max_position,
        q_attn,
        q_ffn,
        q_lm,
        act_dtype,
        weights: fe.w,
        files: hf.files.clone(),
        eos_ids: hf.eos_ids(),
        bos_id: hf.bos_id(),
    })
}
