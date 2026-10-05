//! The weight table: which bytes of which safetensors file become which
//! device pointer, resolved entirely at COMPILE time.
//!
//! Everything here is a pure function of the model IR.  The generated loader
//! performs no name lookup and makes no layout decision at run time -- it walks
//! a constant array.  That is what makes loading a checkpoint a memcpy plus, at
//! most, one elementwise convert kernel.

use crate::ir::*;
use std::fmt::Write as _;

pub fn dt_code(d: DType) -> i32 {
    match d {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 2,
        DType::U8 => 3,
        DType::I8 => 4,
        DType::F8E4M3 => 5,
        DType::F8E5M2 => 6,
        DType::I32 => 7,
        DType::I64 => 8,
        DType::Bool => 3,
    }
}

/// One source slab, plus where its rows land in the destination.
pub struct EPart {
    file: usize,
    off: u64,
    nbytes: u64,
    dtype: DType,
    nrows: u64,
    rowelems: u64,
    dst_row0: u64,
    dst_stride: u64,
    /// destination rows produced per source row (block-scale expansion)
    repeat: u64,
    /// 0 = row map + dtype cast; >0 selects a format repack kernel
    xf: u32,
    /// repack dimensions: destination rows, reduction length, group size
    rows: u64,
    kdim: u64,
    group: u64,
}

pub struct EWeight {
    pub role: String,
    pub parts: Vec<EPart>,
    pub out_dtype: DType,
    pub dst_off: u64,
    pub bytes: u64,
    pub rowelems: u64,
}

/// Resolve every weight into (source bytes -> destination row) form and lay the
/// whole model out in one device arena.  Both are compile-time facts.
/// Rows of the STORED tensor, and its row width: what the read loop iterates.
fn src_rows(sl: &Slab) -> (u64, u64) {
    if sl.shape.len() >= 3 {
        ((sl.shape[0] * sl.shape[1]) as u64, sl.shape[2..].iter().product::<usize>() as u64)
    } else if sl.shape.len() == 2 {
        (sl.shape[0] as u64, sl.shape[1] as u64)
    } else {
        (sl.shape.first().copied().unwrap_or(1) as u64, 1)
    }
}

/// DESTINATION rows one source slab contributes.  For a repacked format the
/// stored tensor is transposed, so this is not `shape[0]`.
fn part_rows(sl: &Slab, rk: Repack) -> u64 {
    match rk {
        Repack::AwqQ | Repack::AwqZ => sl.shape[1] as u64 * 8,
        Repack::ScalesT => sl.shape[1] as u64,
        Repack::None => {
            if sl.shape.len() >= 3 { (sl.shape[0] * sl.shape[1]) as u64 }
            else if sl.shape.is_empty() { 1 }
            else { sl.shape[0] as u64 }
        }
    }
}

pub fn layout(m: &Model) -> (Vec<EWeight>, u64) {
    let mut out = Vec::new();
    let mut cursor: u64 = 0;
    let align = |c: u64| (c + 255) & !255u64;

    for (role, w) in &m.weights {
        // total logical rows and row width
        let k: u64 = if w.shape.len() >= 2 {
            w.shape[1..].iter().product::<usize>() as u64
        } else {
            1
        };
        let rows: u64 = if w.shape.is_empty() { 1 } else { w.shape[0] as u64 };
        // for a 3-D expert tensor [E, rows, k] the row width is the last dim
        let (rows, k) = if w.shape.len() >= 3 {
            (
                (w.shape[0] * w.shape[1]) as u64,
                w.shape[2..].iter().product::<usize>() as u64,
            )
        } else {
            (rows, k)
        };
        // a block-scale tensor carries one row per block of weight rows; the
        // loader expands it so every format's scales are per-row in the kernel
        let expand = w.repeat.max(1) as u64;
        let rows = if w.repack == Repack::None { rows * expand } else { rows };
        let esz = w.out_dtype.bytes() as u64;
        // a repacked weight's shape is already the canonical destination shape
        let bytes = if w.repack == Repack::AwqQ { rows * k / 2 } else { rows * k * esz };
        let dst_off = align(cursor);
        cursor = dst_off + bytes;

        let mut parts = Vec::new();
        match w.transform {
            Xform::Concat | Xform::Cast | Xform::Identity => {
                let mut base = 0u64;
                for p in &w.parts {
                    let pr = part_rows(p, w.repack);
                    let (sr, pk) = src_rows(p);
                    parts.push(EPart {
                        file: p.file,
                        off: p.offset,
                        nbytes: p.nbytes,
                        dtype: p.dtype,
                        nrows: sr,
                        rowelems: pk,
                        // destination rows, so a block-scale tensor that expands
                        // by `expand` starts `expand` times further in
                        dst_row0: base * expand,
                        dst_stride: 1,
                        repeat: expand,
                        xf: 0, rows: 0, kdim: 0, group: 0,
                    });
                    base += pr;
                }
            }
            Xform::InterleavePairs => {
                // parts arrive as [gate_0, up_0, gate_1, up_1, ...]; expert e's
                // pair lands at rows e*2I + 2j + t so one warp computes both.
                for (i, p) in w.parts.iter().enumerate() {
                    let e = (i / 2) as u64;
                    let t = (i % 2) as u64;
                    let pr = part_rows(p, w.repack);
                    let (sr, pk) = src_rows(p);
                    parts.push(EPart {
                        file: p.file,
                        off: p.offset,
                        nbytes: p.nbytes,
                        dtype: p.dtype,
                        nrows: sr,
                        rowelems: pk,
                        dst_row0: e * 2 * pr * expand + t,
                        dst_stride: 2,
                        repeat: expand,
                        xf: 0, rows: 0, kdim: 0, group: 0,
                    });
                }
            }
            Xform::BlockedToInterleaved => {
                // one [2I, K] source: rows [0,I) are gate, [I,2I) are up.
                let p = &w.parts[0];
                let pr = p.shape[0] as u64;
                let pk = p.shape[1..].iter().product::<usize>() as u64;
                let half = pr / 2;
                let esrc = p.dtype.bytes() as u64;
                parts.push(EPart {
                    file: p.file,
                    off: p.offset,
                    nbytes: half * pk * esrc,
                    dtype: p.dtype,
                    nrows: half,
                    rowelems: pk,
                    dst_row0: 0,
                    dst_stride: 2,
                    repeat: expand,
                    xf: 0, rows: 0, kdim: 0, group: 0,
                });
                parts.push(EPart {
                    file: p.file,
                    off: p.offset + half * pk * esrc,
                    nbytes: half * pk * esrc,
                    dtype: p.dtype,
                    nrows: half,
                    rowelems: pk,
                    dst_row0: 1,
                    dst_stride: 2,
                    repeat: expand,
                    xf: 0, rows: 0, kdim: 0, group: 0,
                });
            }
        }
        // element-level repacks are not row mappings: tag every part with the
        // kernel that has to run, and the dimensions it needs
        let rk = match w.repack {
            Repack::None => 0u32,
            Repack::AwqQ => 1,
            Repack::AwqZ => 2,
            Repack::ScalesT => 3,
        };
        if rk != 0 {
            // `kdim` is the canonical last dimension: K for qweight, K/G for the
            // scales and zeros.  Both kernels want exactly the destination stride.
            for p in parts.iter_mut() {
                p.xf = rk;
                p.rows = rows;
                p.kdim = k;
                p.group = 1;
            }
        }
        out.push(EWeight {
            role: role.clone(),
            parts,
            out_dtype: w.out_dtype,
            dst_off,
            bytes,
            rowelems: k,
        });
    }
    (out, align(cursor))
}

pub fn weights_h(m: &Model, ws: &[EWeight], arena: u64) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "// generated by mkc {} -- the weight table, resolved at compile time.", crate::VERSION);
    let _ = writeln!(s, "// Every entry says: these bytes of this file become these device rows.");
    let _ = writeln!(s, "// Nothing here is discovered at run time.");
    let _ = writeln!(s, "#pragma once\n#include <cstdint>\n");
    let _ = writeln!(s, "#define MK_ARENA_BYTES {}ull", arena);
    let _ = writeln!(s, "#define MK_NFILES {}", m.files.len());
    let _ = writeln!(s, "static const char* MK_FILES[] = {{");
    for f in &m.files {
        let _ = writeln!(s, "    \"{}\",", f);
    }
    let _ = writeln!(s, "}};\n");
    let _ = writeln!(s, "typedef struct {{ int file, dtype, xf; unsigned long long off, nbytes, nrows, rowelems,
                  dst_row0, dst_stride, repeat, rows, kdim, group; }} MkPart;");
    let _ = writeln!(s, "typedef struct {{ const char* role; int first, n, out_dtype; unsigned long long dst_off, bytes, rowelems; }} MkWeight;\n");

    let _ = writeln!(s, "static const MkPart MK_PARTS[] = {{");
    for w in ws {
        for p in &w.parts {
            let _ = writeln!(
                s,
                "  {{{},{},{},{}ull,{}ull,{}ull,{}ull,{}ull,{}ull,{}ull,{}ull,{}ull,{}ull}},",
                p.file,
                dt_code(p.dtype),
                p.xf,
                p.off,
                p.nbytes,
                p.nrows,
                p.rowelems,
                p.dst_row0,
                p.dst_stride,
                p.repeat,
                p.rows,
                p.kdim,
                p.group
            );
        }
    }
    let _ = writeln!(s, "}};\n");

    let _ = writeln!(s, "static const MkWeight MK_WEIGHTS[] = {{");
    let mut first = 0usize;
    for w in ws {
        let _ = writeln!(
            s,
            "  {{\"{}\",{},{},{},{}ull,{}ull,{}ull}},",
            w.role,
            first,
            w.parts.len(),
            dt_code(w.out_dtype),
            w.dst_off,
            w.bytes,
            w.rowelems
        );
        first += w.parts.len();
    }
    let _ = writeln!(s, "}};");
    let _ = writeln!(s, "#define MK_NWEIGHTS {}", ws.len());

    // per-role offsets, so the runtime fills LayerW with no lookup at all
    let off_of = |role: &str| -> u64 { ws.iter().find(|w| w.role == role).map(|w| w.dst_off).unwrap_or(u64::MAX) };
    let _ = writeln!(s, "\n// role -> byte offset in the arena");
    for g in ["embed", "lm_head", "final_norm"] {
        let _ = writeln!(s, "#define OFF_{} {}ull", g, off_of(g));
    }
    let mut per_layer: Vec<String> = Vec::new();
    for w in ws {
        if let Some(rest) = w.role.strip_prefix("l0.") {
            per_layer.push(rest.to_string());
        }
    }
    for r in &per_layer {
        let _ = write!(s, "static const unsigned long long OFF_L_{}[] = {{", r);
        for l in 0..m.n_layers {
            let _ = write!(s, "{}{}ull", if l > 0 { "," } else { "" }, off_of(&format!("l{}.{}", l, r)));
        }
        let _ = writeln!(s, "}};");
    }
    s
}

