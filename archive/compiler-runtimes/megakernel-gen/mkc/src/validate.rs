//! What the code generator can actually emit.
//!
//! The frontend is deliberately permissive: it reads any HuggingFace decoder it
//! recognises, including features the backend has no lowering for.  That is a
//! correctness hazard -- a model with FP8 weights or Gemma's post-attention norm
//! would otherwise compile into a kernel that computes something else and might
//! still look plausible.
//!
//! So the contract is explicit: mkc either produces a megakernel that passes the
//! gate, or it refuses and says exactly which feature it lacks.  "Any open
//! model" is only a useful claim if the failures are loud.

use crate::ir::*;

pub fn check(m: &Model) -> Result<(), String> {
    let mut bad: Vec<String> = Vec::new();

    // ---- combinations the backend has no lowering for -------------------
    // Each of these is a feature mkc supports and a feature mkc supports, that
    // together it does not.  They are easy to miss precisely because both
    // halves are tested: the check has to name the PAIR.
    if m.is_moe() {
        if !crate::quant::has_multi(&m.q_ffn) {
            bad.push(format!(
                "mixture-of-experts weights in {}: the top-k expert gemv interleaves \
                 several matrices in one loop and only the bf16/f16 and mxfp4 cores \
                 have that form", m.q_ffn.tag()));
        }
        if m.post_ffn_norm {
            bad.push(
                "a post-FFN norm on a mixture-of-experts model: the expert down-projection \
                 adds straight into the residual, so there is nowhere to normalise between"
                    .into());
        }
    }

    // ---- weight formats -------------------------------------------------
    for (what, q) in [("attention", &m.q_attn), ("ffn", &m.q_ffn), ("lm head", &m.q_lm)] {
        match q {
            Quant::Dense { dtype: DType::BF16 } | Quant::Dense { dtype: DType::F16 } => {}
            Quant::Mxfp4 { block: 32 } => {}
            Quant::Fp8Block { bn, bk } => {
                if *bk != 0 && *bk % 16 != 0 {
                    bad.push(format!("fp8 k-block {} is not a multiple of 16", bk));
                }
                if *bn > 1 && m.attn.head_dim % bn != 0 && what == "attention" {
                    bad.push(format!(
                        "fp8 row-block {} does not divide the attention projections", bn));
                }
            }
            Quant::Int4Group { group, awq, .. } => {
                if group % 32 != 0 {
                    bad.push(format!("int4 group {} is not a multiple of 32", group));
                }
                if !awq {
                    bad.push(
                        "GPTQ int4 packs along the reduction axis, not the output axis; \
                         only the AWQ packing has a repack kernel"
                            .into(),
                    );
                }
            }
            other => bad.push(format!(
                "{} weights are {}, and there is no gemv core for it \
                 (supported: bf16, f16, mxfp4 block 32, fp8 e4m3, int4 group)",
                what,
                other.tag()
            )),
        }
    }

    // ---- shapes the gemv and rotary cores require -----------------------
    let hd = m.attn.head_dim;
    if hd % 32 != 0 {
        bad.push(format!("head_dim {} is not a multiple of 32", hd));
    }
    if m.hidden % 32 != 0 {
        bad.push(format!("hidden {} is not a multiple of 32", m.hidden));
    }
    if m.ffn.intermediate() % 32 != 0 {
        bad.push(format!("intermediate {} is not a multiple of 32", m.ffn.intermediate()));
    }
    if m.rope.rotary_dim % 2 != 0 || m.rope.rotary_dim > hd {
        bad.push(format!("rotary_dim {} is not usable with head_dim {}", m.rope.rotary_dim, hd));
    }
    if !m.rope.halves && m.rope.rotary_dim % 32 != 0 {
        bad.push("interleaved rotary needs rotary_dim to be a multiple of 32".into());
    }
    if m.attn.n_heads % m.attn.n_kv_heads != 0 {
        bad.push(format!(
            "{} query heads do not divide into {} kv heads",
            m.attn.n_heads, m.attn.n_kv_heads
        ));
    }

    // ---- features the backend has no lowering for -----------------------
    if let Ffn::Moe { cfg, shared_intermediate, .. } = &m.ffn {
        if *shared_intermediate > 0 && *shared_intermediate % 64 != 0 {
            bad.push(format!("shared expert intermediate {} is not a multiple of 64",
                             shared_intermediate));
        }
        if cfg.dense_layers {
            bad.push(
                "this checkpoint mixes dense and MoE layers (decoder_sparse_step, \
                 mlp_only_layers or first_k_dense_replace); the kernel emits one \
                 layer body for all layers"
                    .into(),
            );
        }
        if cfg.top_k > 8 {
            // The expert down-projection holds `float acc[TOPK][R]` live across
            // its group loop; past eight the assembler spills whatever the
            // occupancy.  The arena scales with top_k and is not the limit.
            bad.push(format!(
                "top-k of {}: the fused expert down-projection keeps top_k accumulators \
                 per row in registers and spills past 8", cfg.top_k));
        }
    }
    if m.attn.q_norm.is_some() != m.attn.k_norm.is_some() {
        bad.push("q_norm and k_norm must both be present or both absent".into());
    }

    if bad.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "mkc cannot compile `{}` ({}):\n  - {}",
            m.name,
            m.arch,
            bad.join("\n  - ")
        ))
    }
}
