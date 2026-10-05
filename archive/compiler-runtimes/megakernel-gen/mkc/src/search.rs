//! `mkc build`: the search for a schedule the assembler will actually place.
//!
//! `plan.rs` decides what a schedule costs; this decides which of them exist.
//! It enumerates a small, explicit space -- each block size crossed with each
//! occupancy the register floor allows, plus the one that names no occupancy
//! and lets the assembler choose, and for a MoE model both ends of the
//! expert-interleave degree -- compiles them ALL to learn what the assembler
//! does with each, and keeps the cheapest that fits.  With `--verify` it then
//! times each survivor on the machine and keeps the measured winner.
//!
//! The compiles are independent (a different source tree each) and nvcc is
//! single-threaded per translation unit, so they run at once.  That is the
//! difference between a five-minute build and a twenty-five-second one, and it
//! is why the search can afford to enumerate every occupancy instead of
//! stopping at the first that fits.

use crate::{describe, die, emit, hw, ir, plan, report};
use std::path::{Path, PathBuf};

/// How many candidates to compile at once.  Each runs `make -j3`, so this is
/// about three times as many processes.
const JOBS: usize = 8;

/// A schedule's geometry: block size, the launch bound's minimum blocks (None =
/// the assembler's choice) and the expert-interleave degree.  Round two of the
/// search re-plans exactly the geometries round one could not place.
type Geom = (usize, Option<usize>, usize);

fn geom(p: &plan::Plan) -> Geom {
    (p.nt, p.launch_min_blocks, p.moe2_fuse)
}

/// One candidate schedule and the tree it is compiled in.
struct Cand {
    plan: plan::Plan,
    dir: PathBuf,
    /// what the assembler did: registers, spilled bytes, and why it was rejected
    regs: usize,
    spill: usize,
    err: Option<String>,
}

impl Cand {
    fn label(&self) -> String {
        format!("NT={:<5} bps={:<4} fuse={}", self.plan.nt,
                self.plan.launch_min_blocks.map(|b| b.to_string()).unwrap_or("free".into()),
                self.plan.moe2_fuse)
    }
}

fn run_in(dir: &Path, cmd: &str, a: &[&str]) -> Option<std::process::Output> {
    std::process::Command::new(cmd).args(a).current_dir(dir).output().ok()
}

/// Emit a tree and build one target in it.  No `make clean`: `emit_all` leaves
/// unchanged files alone and the Makefile lists the runtime headers as
/// prerequisites, so make rebuilds exactly what moved.
fn make_in(m: &ir::Model, h: &hw::Hw, ctx: usize, p: &plan::Plan, dir: &Path, target: &str) -> bool {
    if let Err(e) = emit::emit_all(m, p, h, dir, ctx) { die(e) }
    run_in(dir, "make", &["-j3", target]).map(|o| o.status.success()).unwrap_or(false)
}

/// Build one target in every tree, `JOBS` at a time.
fn build_all(m: &ir::Model, h: &hw::Hw, ctx: usize, items: &[(plan::Plan, PathBuf)],
             target: &str) -> Vec<bool> {
    let ok: Vec<std::sync::atomic::AtomicBool> =
        items.iter().map(|_| std::sync::atomic::AtomicBool::new(false)).collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|sc| {
        for _ in 0..JOBS.min(items.len().max(1)) {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some((p, dir)) = items.get(i) else { return };
                ok[i].store(make_in(m, h, ctx, p, dir, target),
                            std::sync::atomic::Ordering::Relaxed);
            });
        }
    });
    ok.iter().map(|b| b.load(std::sync::atomic::Ordering::Relaxed)).collect()
}

/// Compile every candidate and record what the assembler gave each.  Only
/// `megakernel.o`: the standalone per-stage kernels and the host runtime are
/// the same code compiled again, and the search does not read them.
fn compile_all(m: &ir::Model, h: &hw::Hw, ctx: usize, cands: &mut [Cand]) {
    let items: Vec<(plan::Plan, PathBuf)> =
        cands.iter().map(|c| (c.plan.clone(), c.dir.clone())).collect();
    let ok = build_all(m, h, ctx, &items, "megakernel.o");
    for (c, built) in cands.iter_mut().zip(ok) {
        let log = std::fs::read_to_string(c.dir.join("ptxas.log")).unwrap_or_default();
        if !built {
            c.err = Some(log.lines().find(|l| l.contains("rror"))
                         .unwrap_or("build failed").to_string());
            continue;
        }
        let (r, sp) = parse_ptxas(&log);
        c.regs = r;
        c.spill = sp;
        // an unparsable log is not evidence that it fits
        if r == 0 { c.err = Some("could not read the assembler's register count".into()) }
    }
}

/// Time one already-built tree.  Serial by construction: there is one GPU, and
/// a contended one lies about which schedule is faster.
fn time_in(dir: &Path, ctx: usize) -> Option<f64> {
    run_in(dir, "./mkbench", &[&ctx.to_string(), "64"])
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|t| t.trim().parse::<f64>().ok())
        .filter(|v| *v > 0.0)
}

/// Every schedule worth compiling: each block size crossed with each occupancy
/// the register floor allows, plus the one that names no occupancy at all and
/// lets the assembler choose -- and, for a MoE model, both ends of the
/// expert-interleave degree, whose register cost the calibration cannot see.
fn enumerate(m: &ir::Model, h: &hw::Hw, ctx: usize, out: &Path,
             fb: &plan::Feedback) -> Vec<Cand> {
    let fuses: Vec<usize> = match m.moe() {
        Some(c) if c.top_k > 1 => vec![1, c.top_k],
        _ => vec![1],
    };
    let mut v = Vec::new();
    for &nt in &[256usize, 512, 1024] {
        for &fuse in &fuses {
            for bps in (1..=plan::bps_ceiling(m, h, nt)).rev().map(Some).chain([None]) {
                let plan = match bps {
                    Some(b) => plan::plan_fixed(m, h, ctx, nt, b, fb, Some(fuse)),
                    None => plan::plan_free(m, h, ctx, nt, fb, Some(fuse)),
                };
                let tag = bps.map(|b| b.to_string()).unwrap_or("free".into());
                v.push(Cand {
                    plan,
                    dir: out.join(format!(".cand-{}-{}-{}", nt, tag, fuse)),
                    regs: 0, spill: 0, err: None,
                });
            }
        }
    }
    v
}

pub fn cmd_build(m: &ir::Model, h: &hw::Hw, args: &crate::Args, peak: f64) {
    let mut trace: Vec<String> = Vec::new();
    let mut compiles = 0usize;
    let mut keep: Vec<(plan::Plan, PathBuf)> = Vec::new();
    // every tree the search created, so the output directory is left holding
    // one megakernel and not fifteen half-built ones
    let mut scratch: Vec<PathBuf> = Vec::new();

    // Round 1 prices every candidate as planned.  Round 2 re-plans only the
    // ones that spilled, with registers withheld from the stages: the fused
    // kernel holds the attention state, the norm and the schedule itself on top
    // of what the calibration measured for a bare gemv, and a stage can buy
    // those back by blocking fewer rows.
    let mut fb = plan::Feedback::default();
    let mut retry: Option<Vec<Geom>> = None;
    for round in 0..2 {
        let mut cands = enumerate(m, h, args.ctx, &args.out, &fb);
        if let Some(g) = &retry {
            cands.retain(|c| g.contains(&geom(&c.plan)));
        }
        if cands.is_empty() { break }
        scratch.extend(cands.iter().map(|c| c.dir.clone()));
        compiles += cands.len();
        compile_all(m, h, args.ctx, &mut cands);
        fb.headroom += plan::Feedback::STEP;
        let mut failed: Vec<Geom> = Vec::new();
        for c in cands {
            if let Some(e) = &c.err {
                trace.push(format!("{}: {}", c.label(), e));
                continue;
            }
            // With no minBlocks the occupancy is an OUTPUT of the compile, so
            // the schedule is repriced at whatever the assembler's register
            // count actually allows.
            // Occupancy is bounded by three things, and pricing a `free`
            // candidate against only two of them made every single-wave stage
            // in a big-arena MoE look twice as fast as it is.
            let real_bps = (h.regs_per_sm / (c.regs * c.plan.nt))
                .min(h.max_threads_per_sm / c.plan.nt)
                .min(h.smem_per_sm / c.plan.smem_bytes.max(1))
                .max(1);
            let over = c.plan.launch_min_blocks.map(|b| real_bps < b).unwrap_or(false);
            if c.spill > SPILL_TOLERANCE || over {
                trace.push(format!("{} -> {} regs, {} B spill -- {}", c.label(), c.regs, c.spill,
                                   if round == 0 { "re-plan" } else { "no fit" }));
                failed.push(geom(&c.plan));
                continue;
            }
            let label = c.label();
            let mut p = c.plan;
            if p.launch_min_blocks.is_none() {
                p = p.repriced(h, m.n_layers, real_bps);
                p.predicted_regs = c.regs;
            }
            trace.push(format!("{} -> {:>3} regs, {} blk/SM, {:.3} ms predicted",
                               label, c.regs, real_bps, p.est_ms));
            // Two explicit occupancies can settle on the same kernel; time it
            // once.  A `free` candidate is never folded into an explicit one
            // even when they agree on registers and row blocking: naming a
            // minimum the assembler would have chosen anyway still changes how
            // it allocates, and measurably.
            if !keep.iter().any(|(k, _)| k.nt == p.nt && k.moe2_fuse == p.moe2_fuse
                                 && k.blocks_per_sm == p.blocks_per_sm
                                 && k.predicted_regs == p.predicted_regs
                                 && k.launch_min_blocks.is_none() == p.launch_min_blocks.is_none()
                                 && k.signature().4 == p.signature().4) {
                keep.push((p, c.dir));
            }
        }
        retry = Some(failed);
    }
    if keep.is_empty() { die("no feasible schedule".into()) }
    keep.sort_by(|x, y| x.0.est_ms.partial_cmp(&y.0.est_ms).unwrap());

    let mut p = keep[0].0.clone();
    if args.verify {
        // Every survivor is timed, not the few the cost model liked: across
        // (NT, blocks/SM) the predictions span 10% while the measurements span
        // 60%, so ranking by prediction here would be close to arbitrary.  The
        // BUILDS run at once; the RUNS do not, because there is one GPU and a
        // contended one lies about which schedule is faster.
        compiles += keep.len();
        let built = build_all(m, h, args.ctx, &keep, "mkbench");
        let mut bestm = f64::INFINITY;
        for ((c, dir), ok) in keep.iter().zip(built) {
            match if ok { time_in(dir, args.ctx) } else { None } {
                Some(ms) => {
                    trace.push(format!(
                        "verify NT={:<5} bps={:<4} fuse={} -> {:.3} ms measured ({:.3} predicted)",
                        c.nt, c.launch_min_blocks.map(|b| b.to_string()).unwrap_or("free".into()),
                        c.moe2_fuse, ms, c.est_ms));
                    if ms < bestm { bestm = ms; p = c.clone(); }
                }
                None => trace.push(format!("verify NT={} bps={:?}: run failed",
                                           c.nt, c.launch_min_blocks)),
            }
        }
        // With the geometry fixed, the key splits.  Attention is where the
        // byte-based cost model is least accurate: its per-item cost is
        // dominated by a query reload and a rotation that no byte count sees.
        if bestm.is_finite() && p.attn_splits > 1 {
            let base = p.attn_splits;
            let sw: Vec<(plan::Plan, PathBuf)> = [base / 2, base * 2, base * 4].iter()
                .filter(|s| **s >= 2 && **s <= 512)
                .map(|s| (plan::with_splits(&p, m, h, *s), args.out.join(format!(".split-{}", s))))
                .filter(|(c, _)| c.attn_splits != p.attn_splits)
                .collect();
            compiles += sw.len();
            scratch.extend(sw.iter().map(|(_, d)| d.clone()));
            let built = build_all(m, h, args.ctx, &sw, "mkbench");
            for ((c, dir), ok) in sw.iter().zip(built) {
                if let Some(ms) = if ok { time_in(dir, args.ctx) } else { None } {
                    trace.push(format!("verify splits={:<4} -> {:.3} ms measured", c.attn_splits, ms));
                    if ms < bestm { bestm = ms; p = c.clone(); }
                }
            }
        }
    }
    let ok = make_in(m, h, args.ctx, &p, &args.out, "all");
    for d in &scratch {
        let _ = std::fs::remove_dir_all(d);
    }
    if !ok { die("final build failed".into()) }
    println!("{}", describe(m));
    println!("{}", report::roofline(m, args.ctx).render(peak));
    println!("{}", p.render(h));
    println!("           register-allocation search ({} compiles):", compiles + 1);
    for t in &trace {
        println!("             {}", t);
    }
    println!("built {}", args.out.display());
}

/// A kernel that spills a few bytes in a path it rarely takes is not the same
/// thing as one that spills in the gemv loop, and ptxas does not say which.
/// Measured: 48 B of spill at two blocks per SM beat a clean kernel at one, and
/// 200 B never did.
const SPILL_TOLERANCE: usize = 16;

/// `Used N registers` / `N bytes spill stores` out of a -Xptxas -v log.
fn parse_ptxas(log: &str) -> (usize, usize) {
    let mut regs = 0usize;
    let mut spill = 0usize;
    for l in log.lines() {
        if let Some(i) = l.find("Used ") {
            if let Some(j) = l[i + 5..].find(" registers") {
                if let Ok(v) = l[i + 5..i + 5 + j].trim().parse::<usize>() {
                    regs = regs.max(v);
                }
            }
        }
        if let Some(i) = l.find(" bytes spill stores") {
            let pre: String = l[..i].chars().rev().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(v) = pre.chars().rev().collect::<String>().parse::<usize>() {
                spill += v;
            }
        }
    }
    (regs, spill)
}

