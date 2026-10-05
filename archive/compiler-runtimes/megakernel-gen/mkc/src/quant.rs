//! Weight formats, in one place.
//!
//! EXTENDING: adding a weight format is four small, local edits:
//!
//!   1. a `Quant` variant in `ir.rs`, with its `bytes()` formula
//!   2. `Mat::parts`  -- which tensors the frontend must bind, and as what dtype
//!   3. `Mat::fields` -- the device pointers the kernel needs
//!   4. `Mat::gemv`   -- the call a stage body emits, plus a core in
//!                       runtime/include/mk/gemv.cuh
//!
//! Nothing outside this file and `arch.rs`'s tensor-name probing branches on the
//! format: the planner sees only `Quant::bytes()`, and the stage bodies ask a
//! `Mat` for its call.

use crate::ir::{DType, Quant};

/// One tensor a format needs: a role suffix, the dtype it is stored in, and the
/// C type of the pointer the kernel reads it through.
pub struct Part {
    pub suffix: &'static str,
    /// the dtype the loader writes; used by the frontend when it binds the tensor
    #[allow(dead_code)]
    pub dtype: DType,
    pub ctype: &'static str,
}

/// A logical `[rows, k]` weight matrix, as the backend addresses it.
/// Whether `Mat::gemv_multi` exists for this format -- the top-k expert
/// down-projection interleaves several matrices in one group loop, and only the
/// cores whose inner loop is a single pointer walk have that form.  A MoE model
/// in any other format is refused (see validate.rs) rather than miscompiled.
pub fn has_multi(q: &Quant) -> bool {
    matches!(q, Quant::Dense { .. } | Quant::Mxfp4 { .. })
}

pub struct Mat<'a> {
    pub role: &'a str,
    pub q: &'a Quant,
    /// reduction length, and the symbolic name to emit for it (`M::H`) so the
    /// generated CUDA reads like the model rather than like a pile of integers
    pub k: usize,
    pub ksym: &'a str,
    /// number of output rows; formats with per-row-block scales need it
    #[allow(dead_code)]
    pub rows: usize,
    /// activation C type
    #[allow(dead_code)]
    pub at: &'a str,
}

/// The HuggingFace tensor-name suffixes this format stores, in binding order,
/// appended to a base name that has had its trailing `.weight` removed.
///
/// This is the other half of "one place per format": the checkpoint convention
/// lives here next to the kernel convention, so adding a format does not mean
/// hunting through the frontend for name patterns.
#[allow(dead_code)] // used once the frontend binds formats generically
pub fn hf_suffixes(q: &Quant) -> Vec<Vec<&'static str>> {
    match q {
        Quant::Dense { .. } => vec![vec![".weight"]],
        Quant::Mxfp4 { .. } => vec![vec!["_blocks"], vec!["_scales"]],
        // `weight_scale_inv` is the DeepSeek block form, `weight_scale` the
        // compressed-tensors per-channel form; the loader normalises both.
        Quant::Fp8Block { .. } => vec![vec![".weight"], vec![".weight_scale_inv", ".weight_scale"]],
        Quant::Int4Group { sym, .. } => {
            if *sym { vec![vec![".qweight"], vec![".scales"]] }
            else { vec![vec![".qweight"], vec![".scales"], vec![".qzeros"]] }
        }
    }
}

/// The tensors this format stores, in binding order.  `act` is the model's
/// activation dtype, used by formats that store values directly.
pub fn parts(q: &Quant, act: DType) -> Vec<Part> {
    let actc: &'static str = match act {
        DType::F16 => "__half",
        _ => "__nv_bfloat16",
    };
    match q {
        Quant::Dense { .. } => vec![Part { suffix: "w", dtype: act, ctype: actc }],
        Quant::Mxfp4 { .. } => vec![
            Part { suffix: "blk", dtype: DType::U8, ctype: "uint4" },
            Part { suffix: "scl", dtype: DType::U8, ctype: "uint8_t" },
        ],
        Quant::Fp8Block { .. } => vec![
            Part { suffix: "w", dtype: DType::F8E4M3, ctype: "__nv_fp8_e4m3" },
            // expanded to one scale per row by the loader (see `Mat::gemv`)
            Part { suffix: "s", dtype: DType::F32, ctype: "float" },
        ],
        Quant::Int4Group { sym, .. } => {
            let mut v = vec![
                Part { suffix: "q", dtype: DType::U8, ctype: "uint8_t" },
                Part { suffix: "s", dtype: DType::F16, ctype: "__half" },
            ];
            if !sym {
                v.push(Part { suffix: "z", dtype: DType::F16, ctype: "__half" });
            }
            v
        }
        // (a format's parts and its gemv call must stay in the same order)
    }
}

impl<'a> Mat<'a> {
    /// Symbolic group count for the block-scaled formats: `M::GK` when the
    /// reduction is the hidden size, `M::GI` when it is the intermediate.
    /// scale columns: one per k-block, or one for the whole row
    fn sk(&self, bk: usize) -> usize {
        if bk == 0 { 1 } else { crate::ir::div_up(self.k, bk) }
    }

    fn gsym(&self) -> String {
        match self.ksym {
            "M::H" => "M::GK".into(),
            "M::INTER" => "M::GI".into(),
            other => format!("({}/32)", other),
        }
    }

    /// `const T* role_suffix;` declarations for the layer's pointer struct.
    pub fn fields(&self, act: DType) -> String {
        parts(self.q, act)
            .iter()
            .map(|p| format!("    const {}* {}_{};\n", p.ctype, self.role, p.suffix))
            .collect()
    }

    /// The device pointer expressions, already offset to output row `row`.
    fn ptrs(&self, base: &str, row: &str) -> Vec<String> {
        let b = |s: &str| format!("{}{}_{}", base, self.role, s);
        match self.q {
            Quant::Dense { .. } => vec![format!("{} + (size_t)({})*{}", b("w"), row, self.ksym)],
            Quant::Mxfp4 { block } => {
                let g = self.k / block;
                let _ = g;
                vec![
                    format!("{} + (size_t)({})*{}", b("blk"), row, self.gsym()),
                    format!("{} + (size_t)({})*{}", b("scl"), row, self.gsym()),
                ]
            }
            Quant::Fp8Block { bk, .. } => vec![
                format!("{} + (size_t)({})*{}", b("w"), row, self.ksym),
                format!("{} + (size_t)({})*{}", b("s"), row, self.sk(*bk)),
            ],
            Quant::Int4Group { group, sym, .. } => {
                let ng = self.k / group;
                let mut v = vec![
                    format!("{} + (size_t)({})*{}", b("q"), row, self.k / 2),
                    format!("{} + (size_t)({})*{}", b("s"), row, ng),
                ];
                if !sym {
                    v.push(format!("{} + (size_t)({})*{}", b("z"), row, ng));
                }
                v
            }
        }
    }

    /// `mk::gemv_*(...)` computing `R` rows starting at `row` into `acc`.
    pub fn gemv(&self, base: &str, row: &str, r: &str, xs: &str, acc: &str) -> String {
        let p = self.ptrs(base, row);
        match self.q {
            Quant::Dense { .. } => format!(
                "mk::gemv_dense<M::AT, {k}, {r}>({w}, {k}, {xs}, lane, {acc});",
                k = self.ksym, r = r, w = p[0], xs = xs, acc = acc
            ),
            Quant::Mxfp4 { .. } => format!(
                "mk::gemv_mxfp4<{g}, {r}>({blk}, {scl}, {xs}, lane, {acc});",
                g = self.gsym(), r = r, blk = p[0], scl = p[1], xs = xs, acc = acc
            ),
            // BN is always 1 in the kernel: the loader expands a checkpoint's
            // per-block-row scales to per-row, so the interleaved gate/up layout
            // stays legal and one core serves both fp8 conventions.
            Quant::Fp8Block { bk, .. } => format!(
                "mk::gemv_fp8<{k}, {r}, 1, {bkv}>({w}, {s}, {sk}, 0, {xs}, lane, {acc});",
                k = self.ksym, r = r, bkv = if *bk == 0 { self.k } else { *bk },
                w = p[0], s = p[1], sk = self.sk(*bk), xs = xs, acc = acc
            ),
            Quant::Int4Group { group, sym, .. } => format!(
                "mk::gemv_int4<{k}, {r}, {g}, {sym}>({q}, {s}, {z}, {xs}, lane, {acc});",
                k = self.ksym, r = r, g = group, sym = sym, q = p[0], s = p[1],
                z = if *sym { "nullptr".to_string() } else { p[2].clone() },
                xs = xs, acc = acc
            ),
        }
    }

    /// Declarations + fill for the pointer arrays a `gemv_*_multi` call takes,
    /// and the call itself.  `count` matrices are interleaved.
    pub fn gemv_multi(&self, base: &str, row_of: &str, count: &str, r: &str,
                      xs_of: &str, acc: &str) -> String {
        let arrays: Vec<(&str, &str)> = match self.q {
            Quant::Mxfp4 { .. } => vec![("const uint4*", "bp"), ("const uint8_t*", "sp")],
            _ => vec![("const M::AT*", "wp")],
        };
        let mut s = String::new();
        for (ty, nm) in &arrays {
            s.push_str(&format!("            {} {}[{}];\n", ty, nm, count));
        }
        s.push_str(&format!("            const float* xf[{}];\n", count));
        s.push_str(&format!("            #pragma unroll\n            for (int t = 0; t < {}; ++t) {{\n", count));
        s.push_str(&format!("                const size_t row = {};\n", row_of));
        let p = self.ptrs(base, "row");
        match self.q {
            Quant::Mxfp4 { .. } => {
                s.push_str(&format!("                bp[t] = {}; sp[t] = {};\n", p[0], p[1]));
            }
            _ => {
                s.push_str(&format!("                wp[t] = {};\n", p[0]));
            }
        }
        s.push_str(&format!("                xf[t] = {};\n            }}\n", xs_of));
        s.push_str(&match self.q {
            Quant::Mxfp4 { .. } => format!(
                "            mk::gemv_mxfp4_multi<{}, {}, {}>(bp, sp, xf, lane, {});\n",
                self.gsym(), r, count, acc
            ),
            _ => format!(
                "            mk::gemv_dense_multi<M::AT, {}, {}, {}>(wp, xf, lane, {});\n",
                self.ksym, r, count, acc
            ),
        });
        s
    }
}
