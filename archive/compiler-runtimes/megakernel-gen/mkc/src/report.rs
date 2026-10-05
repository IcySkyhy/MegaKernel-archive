//! The bytes-per-token table and roofline.  This is the first thing the GUIDE
//! says to compute and the number every later decision is judged against, so
//! the compiler prints it whether or not you asked for code.

use crate::hw::Hw;
use crate::ir::*;

pub struct Line {
    pub what: String,
    pub bytes: u64,
    /// gemv core that reads these bytes, "" for the ones no gemv reads
    pub core: &'static str,
}

pub struct Roofline {
    pub lines: Vec<Line>,
    pub total: u64,
    pub ctx: usize,
}

fn q_bytes(q: &Quant, rows: usize, k: usize) -> u64 {
    q.bytes(rows, k) as u64
}

/// Bytes that MUST be read from DRAM to produce one token at `ctx` context.
pub fn roofline(m: &Model, ctx: usize) -> Roofline {
    let mut lines = Vec::new();
    let h = m.hidden;
    let nl = m.n_layers;
    let qd = m.q_dim();
    let kvd = m.kv_dim();

    // attention projections
    let qkv = q_bytes(&m.q_attn, qd + 2 * kvd, h) * nl as u64;
    let o = q_bytes(&m.q_attn, h, qd) * nl as u64;
    lines.push(Line {
        what: "attention projections".into(),
        bytes: qkv + o,
        core: m.q_attn.core(),
    });

    // kv cache: only the layers' own windows are read
    let mut kvb = 0u64;
    for l in 0..nl {
        let span = match m.attn.window[l] {
            Some(w) => ctx.min(w),
            None => ctx,
        };
        kvb += (span * 2 * kvd * m.attn.kv_dtype.bytes()) as u64;
    }
    lines.push(Line {
        what: "kv cache".into(),
        bytes: kvb,
        core: "",
    });

    match &m.ffn {
        Ffn::Dense { intermediate, .. } => {
            let gu = q_bytes(&m.q_ffn, 2 * intermediate, h) * nl as u64;
            let dn = q_bytes(&m.q_ffn, h, *intermediate) * nl as u64;
            lines.push(Line {
                what: "mlp".into(),
                bytes: gu + dn,
                core: m.q_ffn.core(),
            });
        }
        Ffn::Moe {
            intermediate,
            cfg,
            shared_intermediate,
            ..
        } => {
            let per_expert = q_bytes(&m.q_ffn, 2 * intermediate, h)
                + q_bytes(&m.q_ffn, h, *intermediate);
            lines.push(Line {
                what: format!(
                    "MoE experts, {} of {}, {}",
                    cfg.top_k,
                    cfg.n_experts,
                    m.q_ffn.tag()
                ),
                bytes: per_expert * cfg.top_k as u64 * nl as u64,
                core: m.q_ffn.core(),
            });
            lines.push(Line {
                what: "router".into(),
                bytes: q_bytes(&m.q_attn, cfg.n_experts, h) * nl as u64,
                core: m.q_attn.core(),
            });
            if *shared_intermediate > 0 {
                let sh = q_bytes(&m.q_ffn, 2 * shared_intermediate, h)
                    + q_bytes(&m.q_ffn, h, *shared_intermediate);
                lines.push(Line {
                    what: "shared expert".into(),
                    bytes: sh * nl as u64,
                    core: m.q_ffn.core(),
                });
            }
        }
    }

    // norms are tiny but real
    let norms = (nl * 2 + 1) * h * 4;
    lines.push(Line {
        what: "norms".into(),
        bytes: norms as u64,
        core: "",
    });

    lines.push(Line {
        what: "lm head".into(),
        bytes: q_bytes(&m.q_lm, m.vocab, h),
        core: m.q_lm.core(),
    });

    let total = lines.iter().map(|l| l.bytes).sum();
    Roofline { lines, total, ctx }
}

impl Roofline {
    /// `hw` is optional only so `mkc analyze` can print the table with no
    /// machine file at all.
    pub fn render(&self, peak_gbs: f64) -> String {
        let mb = |b: u64| b as f64 / 1e6;
        let mut s = String::new();
        s.push_str(&format!(
            "bytes/token at ctx={} :  {:.1} MB\n",
            self.ctx,
            mb(self.total)
        ));
        let mut ls: Vec<&Line> = self.lines.iter().collect();
        ls.sort_by(|a, b| b.bytes.cmp(&a.bytes));
        for l in ls {
            s.push_str(&format!(
                "  {:<38} {:>9.1} MB  {:>5.1}%\n",
                l.what,
                mb(l.bytes),
                100.0 * l.bytes as f64 / self.total as f64
            ));
        }
        s.push_str(&format!(
            "  {:<38} {:>9.3} ms  (at {:.0} GB/s)\n",
            "roofline",
            mb(self.total) / peak_gbs,
            peak_gbs
        ));
        s
    }
    pub fn ms_at(&self, gbs: f64) -> f64 {
        (self.total as f64 / 1e6) / gbs
    }

    /// The floor a KERNEL can reach, as opposed to the one a memcpy could.
    ///
    /// Each class of bytes is charged at the measured ceiling of the gemv core
    /// that reads it.  A 4-bit format cannot stream at the dense rate however
    /// the schedule is arranged -- the unpacking is inside the loop -- so
    /// charging it the dense rate quietly reports every quantised model as
    /// further from its limit than it is.
    pub fn floor_ms(&self, hw: &Hw) -> f64 {
        self.lines.iter().map(|l| {
            let gbs = match hw.gemv_best(l.core) {
                g if g > 0.0 => g,
                _ => hw.stream_peak_gbs,
            };
            (l.bytes as f64 / 1e6) / gbs
        }).sum()
    }
}
