//! The planner: model IR + machine model -> a fully determined schedule.
//!
//! This is where the hill-climbing goes.  Every decision a human would reach by
//! sweeping compiled binaries -- block size, row blocking per stage, how many
//! ways to split the key axis, the grid, where the barriers fall, how big the
//! shared arena is -- is made here by evaluating a cost that is an
//! interpolation of *measured* primitive rates.  `plan()` is a pure function:
//! same model + same machine file => same schedule, every time.

use crate::hw::Hw;
use crate::ir::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StageKind {
    Embed,
    QkvRope,
    Attn,
    AttnReduce,
    OProj,
    PostAttnNorm,
    PostFfnNorm,
    Router,
    TopK,
    SharedMlp1,
    SharedMlp2,
    Moe1,
    Moe2,
    Mlp1,
    Mlp2,
    LmHead,
}

impl StageKind {
    pub fn short(self) -> &'static str {
        match self {
            StageKind::Embed => "embed",
            StageKind::QkvRope => "qkv",
            StageKind::Attn => "attn",
            StageKind::AttnReduce => "attn_red",
            StageKind::OProj => "oproj",
            StageKind::PostAttnNorm => "post_attn",
            StageKind::PostFfnNorm => "post_ffn",
            StageKind::Router => "router",
            StageKind::TopK => "topk",
            StageKind::SharedMlp1 => "shmlp1",
            StageKind::SharedMlp2 => "shmlp2",
            StageKind::Moe1 => "moe1",
            StageKind::Moe2 => "moe2",
            StageKind::Mlp1 => "mlp1",
            StageKind::Mlp2 => "mlp2",
            StageKind::LmHead => "lm_head",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Stage {
    pub kind: StageKind,
    pub items: usize,
    /// DRAM bytes the stage must read
    pub bytes: u64,
    /// gemv core this stage runs on, "" for the stages that are not a gemv
    pub core: String,
    pub r: usize,
    /// bytes one work item reads in one pass
    pub bpi: f64,
    /// sequential passes over the item set (top-k expert groups)
    pub groups: f64,
    /// shared memory this stage needs.  `arena_floats` bounds the arena from the
    /// model directly; this is here so a new stage declares its footprint next
    /// to its cost, in one place.
    pub smem: usize,
    pub barrier_after: bool,
    pub est_us: f64,
}

#[derive(Clone, Debug, Default)]
/// What the runtime needs to size that `model.h`'s constants do not already
/// say.  The partial-softmax buffers follow from NQ, SPLITS and HD; the FFN
/// activation buffer does not, because a shared expert can be wider than all
/// top_k routed ones together.
pub struct Scratch {
    pub act_elems: usize,
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub nt: usize,
    pub blocks: usize,
    pub blocks_per_sm: usize,
    /// The second `__launch_bounds__` argument, or None to omit it.
    ///
    /// Omitting it is a real, distinct schedule, not the absence of one: it
    /// tells the assembler to allocate registers as it sees fit for a block of
    /// this size, and the host then derives the grid from the occupancy that
    /// results.  On three of four measured models that beat every explicit
    /// occupancy -- pinning `minBlocksPerMultiprocessor` to the same value the
    /// assembler would have chosen anyway still changed how it allocated, and
    /// on gpt-oss-20b it turned 16 bytes of spill into 368.
    pub launch_min_blocks: Option<usize>,
    /// shared arena, in floats (the activation staging area)
    pub arena_floats: usize,
    pub smem_bytes: usize,
    pub layer: Vec<Stage>,
    pub tail: Vec<Stage>,
    pub attn_splits: usize,
    pub ctx: usize,
    pub est_ms: f64,
    pub barriers_per_token: usize,
    pub predicted_regs: usize,
    /// how many expert down-projections are interleaved in one group loop
    pub moe2_fuse: usize,
    pub scratch: Scratch,
}

const NT_CANDIDATES: [usize; 3] = [256, 512, 1024];
const R_CANDIDATES: [usize; 4] = [1, 2, 4, 8];

fn div_up(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

fn pad33(n: usize) -> usize {
    div_up(n, 32) * 33
}

/// Cost of one stage, from the measured stage table.  Falls back to the
/// streaming ceiling only if the machine file predates this quant.
fn cost(hw: &Hw, quant: &str, nt: usize, r: usize, items: usize, bpi: f64, warps: usize) -> f64 {
    if items == 0 {
        return 0.0;
    }
    if let Some(us) = hw.stage_us_at(quant, nt, r, items as u64, bpi, warps) {
        return us;
    }
    // proxy: the dense curve at the same geometry, scaled by the rate ratio
    let base = hw
        .stage_us_at("bf16", nt, r, items as u64, bpi, warps)
        .unwrap_or_else(|| (items as f64 * bpi) / (hw.stream_peak_gbs * 1e9) * 1e6);
    let a = hw.gemv_best(quant);
    let b = hw.gemv_best("bf16");
    if a > 0.0 && b > 0.0 {
        base * b / a
    } else {
        base
    }
}

/// A pure streaming stage that is not a gemv (kv reads, partial-sum reduction).
fn cost_stream(hw: &Hw, nt: usize, items: usize, bpi: f64, warps: usize) -> f64 {
    cost(hw, "bf16", nt, 1, items, bpi, warps)
}

struct GemvPick {
    r: usize,
    items: usize,
    bytes: u64,
    us: f64,
    regs: usize,
}

/// Choose the row-blocking factor for one gemv stage.
///
/// Larger R amortises the shared-memory reads of the activation vector and
/// raises achieved bandwidth, but divides the item count -- and once items fall
/// below the resident warp count most of the GPU idles.  Both halves are in the
/// measured table, so this is an evaluation, not a search over binaries.
fn pick_gemv(
    hw: &Hw,
    rows: usize,
    k: usize,
    q: &Quant,
    nt: usize,
    rmin: usize,
    rmax: usize,
    warps: usize,
    reg_budget: usize,
) -> GemvPick {
    let tag = q.core();
    let mut best = GemvPick { r: rmin.max(1), items: rows, bytes: 0, us: f64::INFINITY, regs: 32 };
    for &r in R_CANDIDATES.iter() {
        if r < rmin || r > rmax || rows % r != 0 {
            continue;
        }
        // Row blocking costs registers, and registers are the shared budget that
        // sets occupancy for EVERY stage.  A configuration that will not fit the
        // budget the plan has committed to is simply not a candidate.
        let regs = hw.gemv_peak(tag, nt, r).map(|g| g.regs).unwrap_or(48);
        if regs > reg_budget && r > rmin.max(1) {
            continue;
        }
        let items = rows / r;
        let bpi = q.bytes(r, k) as f64;
        let us = cost(hw, tag, nt, r, items, bpi, warps);
        if us < best.us {
            best = GemvPick { r, items, bytes: q.bytes(rows, k) as u64, us, regs };
        }
    }
    if best.bytes == 0 {
        best.bytes = q.bytes(rows, k) as u64;
        best.us = cost(hw, tag, nt, best.r, best.items, q.bytes(best.r, k) as f64, warps);
    }
    best
}

/// Enumerate (block size, blocks per SM) and keep the cheapest schedule.
///
/// Occupancy is a DECISION here, not an outcome.  Blocks per SM sets the
/// register budget every stage must fit in, and the register budget decides how
/// much row blocking each stage may use -- while occupancy simultaneously sets
/// how many warps are resident, which is what the achieved bandwidth of a
/// single-wave stage depends on.  The two pull in opposite directions, so the
/// only way to get it right is to price the whole schedule at each choice.
/// The emitted kernel then carries `__launch_bounds__(NT, BPS)` so the assembler
/// is held to the budget the plan was made under.
pub fn plan(m: &Model, hw: &Hw, ctx: usize) -> Plan {
    plan_with(m, hw, ctx, &Feedback::default())
}

/// What the previous compile of this model revealed about registers.
///
/// The calibration only ever measured a bare gemv, but the fused kernel also
/// holds the attention state, the norm, the top-k and the schedule itself.
/// `headroom` is how many registers per thread to withhold from the stages so
/// that the whole kernel fits the occupancy it was planned for.  It is zero on
/// the first pass and `mkc build` raises it when the assembler spills, which
/// makes the pair a fixpoint on a static resource rather than a search on a
/// measurement.
#[derive(Clone, Debug, Default)]
pub struct Feedback {
    pub headroom: usize,
}

impl Feedback {
    /// Doubling a stage's row blocking costs about this many registers, so it is
    /// the granularity at which withholding any changes a decision.
    pub const STEP: usize = 8;

}

/// Override the key-split count on an existing plan.  The attention stage is
/// where the cost model is least accurate (its per-item cost is dominated by a
/// query-head reload and a rotation that the byte-based model does not see), so
/// `--verify` sweeps a few values around the predicted one.
pub fn with_splits(p: &Plan, m: &Model, hw: &Hw, splits: usize) -> Plan {
    // Rebuild rather than patch.  The split count feeds the attention stage's
    // item count, the reduction's item count AND its bytes per item; an earlier
    // version of this patched two of those three and reported a reduce stage
    // thirty-two times too expensive.
    let mut q = build(m, hw, p.ctx, p.nt, p.blocks_per_sm, arena_floats_nt(m, p.nt), 0,
                      Some(splits));
    // These describe the compiled kernel, not the schedule's shape.  The rebuild
    // re-derives row blocking with no register headroom, so it can differ from
    // what the assembler accepted for `p`; that is safe because `mkc build`
    // compiles and TIMES this candidate and keeps it only if it is faster.
    q.launch_min_blocks = p.launch_min_blocks;
    q.predicted_regs = p.predicted_regs;
    apply_fuse(q, m, hw, p.moe2_fuse)
}

/// Re-price a plan at a different expert-interleave degree.  Same schedule: the
/// stage is F times the bytes per item over top_k/F sequential groups.
fn apply_fuse(mut p: Plan, m: &Model, hw: &Hw, f: usize) -> Plan {
    let Some(cfg) = m.moe() else { return p };
    if f == p.moe2_fuse || cfg.top_k % f != 0 {
        return p;
    }
    if let Some(st) = p.layer.iter_mut().find(|s| s.kind == StageKind::Moe2) {
        st.bpi = st.bpi / p.moe2_fuse as f64 * f as f64;
        st.groups = (cfg.top_k / f) as f64;
    }
    let bps = p.blocks_per_sm;
    p.moe2_fuse = f;
    p.repriced(hw, m.n_layers, bps)
}

/// Plan at a specific (block size, blocks per SM).  `mkc build` uses this to
/// enumerate the small space of feasible occupancies, compiling each once to
/// find out what the assembler will actually give it.
/// Plan with no `minBlocksPerMultiprocessor`: budget registers as if one block
/// per SM (which is what the assembler targets when left alone) but emit the
/// launch bounds without the second argument.  What occupancy this really gets
/// is not known until it is compiled, so `mkc build` reprices it afterwards.
pub fn plan_free(m: &Model, hw: &Hw, ctx: usize, nt: usize, fb: &Feedback,
                 force_fuse: Option<usize>) -> Plan {
    let mut p = plan_fixed(m, hw, ctx, nt, 1, fb, force_fuse);
    p.launch_min_blocks = None;
    p
}

pub fn plan_fixed(m: &Model, hw: &Hw, ctx: usize, nt: usize, bps: usize, fb: &Feedback,
                  force_fuse: Option<usize>) -> Plan {
    let p = build(m, hw, ctx, nt, bps, arena_floats_nt(m, nt), fb.headroom, None);
    match force_fuse {
        Some(f) => apply_fuse(p, m, hw, f),
        None => p,
    }
}

/// Fewer registers per thread than this has never produced a schedule that the
/// assembler could place without spilling, on any model in the zoo: the
/// attention state, the norm and the schedule itself are already most of it.
/// Occupancies that would demand less are not planned, which is worth roughly a
/// third of `mkc build`'s compiles.
const MIN_REGS_PER_THREAD: usize = 40;

/// Shared memory one block of the widest schedule needs.  `mkc` refuses a model
/// whose arena does not fit an SM's opt-in maximum, because the alternative is a
/// successful compile that dies at `cudaFuncSetAttribute`.
pub fn smem_needed(m: &Model, nt: usize) -> usize {
    smem_bytes(arena_floats_nt(m, nt), nt, m.moe().map(|c| c.top_k).unwrap_or(0))
}

/// Largest blocks/SM this block size could reach: threads, shared memory and
/// the register floor above.
pub fn bps_ceiling(m: &Model, hw: &Hw, nt: usize) -> usize {
    let smem = build(m, hw, 1024, nt, 1, arena_floats_nt(m, nt), 0, None).smem_bytes;
    (hw.max_threads_per_sm / nt)
        .min(if smem > 0 { hw.smem_per_sm / smem } else { 32 })
        .min(hw.regs_per_sm / (nt * MIN_REGS_PER_THREAD))
        .min(8)
        .max(1)
}

pub fn plan_with(m: &Model, hw: &Hw, ctx: usize, fb: &Feedback) -> Plan {
    let mut best: Option<Plan> = None;
    for &nt in NT_CANDIDATES.iter() {
        let arena = arena_floats_nt(m, nt);
        let max_bps = bps_ceiling(m, hw, nt);
        // A block count the assembler cannot actually reach is not a plan, it
        // is a spill.  Once a compile has told us what this model's stage set
        // really costs in registers, cap the occupancy at what that allows.
        let over = fb.headroom;
        for bps in 1..=max_bps {
            let p = build(m, hw, ctx, nt, bps, arena, over, None);
            if !p.fits(hw) {
                continue;
            }
            if best.as_ref().map(|b| p.est_ms < b.est_ms).unwrap_or(true) {
                best = Some(p);
            }
        }
    }
    // R=1 at one block per SM always fits, so this can only be None if the
    // occupancy ceiling itself was zero -- which bps_ceiling forbids.
    best.expect("bps_ceiling guarantees at least one candidate")
}

/// Where the shared expert stages its activation: after the normalised input,
/// which the ROUTED experts still need.  Sharing one region would have the
/// shared expert silently destroy the routed experts' input.
pub fn shared_arena_offset(m: &Model) -> usize {
    match &m.ffn {
        Ffn::Moe { shared_intermediate, .. } if *shared_intermediate > 0 => pad33(m.hidden),
        _ => 0,
    }
}

/// Dynamic shared memory one block needs: the activation arena, one float per
/// warp for the block reductions, and the selected-expert index and weight
/// arrays.
pub fn smem_bytes(arena_floats: usize, nt: usize, top_k: usize) -> usize {
    arena_floats * 4 + (nt / 32) * 4 + top_k.max(1) * 8 + 64
}

fn arena_floats(m: &Model) -> usize {
    // the attention stage borrows the arena for its per-warp rope scratch
    let mut a = pad33(m.hidden).max(pad33(m.q_dim()));
    match &m.ffn {
        Ffn::Dense { intermediate, .. } => a = a.max(pad33(*intermediate)),
        Ffn::Moe { intermediate, cfg, shared_intermediate, .. } => {
            a = a.max(cfg.top_k * pad33(*intermediate));
            if *shared_intermediate > 0 {
                a = a.max(pad33(m.hidden) + pad33(*shared_intermediate));
            }
        }
    }
    a
}

fn arena_floats_nt(m: &Model, nt: usize) -> usize {
    // the attention stage borrows the arena for its per-warp rope scratch, and
    // the cross-split reduction for its per-warp partial sums (32 floats each)
    // plus the two floats it publishes the softmax normaliser in
    arena_floats(m).max((nt / 32) * m.attn.head_dim.max(32) + 2)
}

/// The context every stage construction needs, so that adding a stage is one
/// method here plus one template in codegen.rs -- not an edit inside a
/// 400-line function.
struct Build<'a> {
    m: &'a Model,
    hw: &'a Hw,
    ctx: usize,
    nt: usize,
    bps: usize,
    blocks: usize,
    warps: usize,
    /// registers a stage may use, after the fused kernel's own overhead
    reg_budget: usize,
    bar: f64,
    /// None = emit `__launch_bounds__(NT)` and let the assembler choose
    launch_min: Option<usize>,
    /// Some = use this key-split count instead of searching for one
    force_splits: Option<usize>,
    regs_max: usize,
    moe2_fuse: usize,
    splits: usize,
    layer: Vec<Stage>,
    tail: Vec<Stage>,
    scratch: Scratch,
}

impl<'a> Build<'a> {
    fn new(m: &'a Model, hw: &'a Hw, ctx: usize, nt: usize, bps: usize,
           reg_overhead: usize) -> Self {
        let blocks = bps * hw.sms;
        Build {
            m, hw, ctx, nt, bps,
            blocks,
            warps: blocks * nt / 32,
            reg_budget: (hw.regs_per_sm / (nt * bps)).saturating_sub(reg_overhead),
            bar: hw.barrier_cost_us(blocks),
            launch_min: Some(bps),
            force_splits: None,
            regs_max: 32,
            moe2_fuse: 1,
            splits: 1,
            layer: Vec::new(),
            tail: Vec::new(),
            scratch: Scratch::default(),
        }
    }

    /// Row blocking for one gemv, and the stage that wraps it.
    fn gemv_stage(&mut self, kind: StageKind, rows: usize, k: usize, q: &Quant,
                  rmin: usize, rmax: usize, smem: usize) {
        let g = pick_gemv(self.hw, rows, k, q, self.nt, rmin, rmax, self.warps, self.reg_budget);
        self.regs_max = self.regs_max.max(g.regs);
        self.layer.push(Stage {
            kind,
            items: g.items,
            bytes: g.bytes,
            core: q.core().to_string(),
            r: g.r,
            bpi: q.bytes(g.r, k) as f64,
            groups: 1.0,
            smem,
            barrier_after: true,
            est_us: g.us,
        });
    }

    /// A stage that is not a gemv: attention, the partial-softmax reduction,
    /// the post-norms.  Priced from the streaming curve.
    fn plain_stage(&mut self, kind: StageKind, items: usize, bytes: u64, bpi: f64,
                   smem: usize, est_us: f64) {
        self.layer.push(Stage {
            kind, items, bytes, core: String::new(), r: 1, bpi, groups: 1.0,
            smem, barrier_after: true, est_us,
        });
    }

    /// Work items and bytes per item for the cross-split reduction.
    ///
    /// `mk::attn_reduce` gives one (head, 32-lane chunk) to a whole BLOCK and
    /// divides the split axis between that block's warps, so the item count the
    /// machine sees is the number of output chunks times the warps per block --
    /// not the number of output chunks, which on a model with many key splits
    /// would be a few dozen items for the whole GPU.
    fn reduce_shape(&self, splits: usize) -> (usize, f64) {
        let chunks = (self.m.attn.n_heads * self.m.attn.head_dim / 32).max(1);
        let nw = (self.nt / 32).min(splits);
        (chunks * nw, (32 * div_up(splits, nw) * 4) as f64)
    }

    /// Mean number of keys a layer attends over: sliding-window layers see less.
    fn mean_span(&self) -> usize {
        let m = self.m;
        let mut t = 0usize;
        for l in 0..m.n_layers {
            t += match m.attn.window[l] { Some(w) => self.ctx.min(w), None => self.ctx };
        }
        (t / m.n_layers.max(1)).max(1)
    }

    // ---------------------------------------------------------------- stages
    // One method per stage.  EXTENDING: a new stage is a method here (which
    // declares its item count, bytes, bytes-per-item and shared-memory
    // footprint) plus a template in codegen.rs.  The cost model needs nothing
    // new: it prices every stage from those four numbers.

    /// R consecutive rows of the fused qkv matrix.  R is bounded by head_dim so
    /// a warp's rows never straddle the q/k/v boundary or a head boundary in the
    /// v-cache write.
    fn qkv(&mut self) {
        let m = self.m;
        let rows = m.q_dim() + 2 * m.kv_dim();
        let smem = pad33(m.hidden) * 4;
        self.gemv_stage(StageKind::QkvRope, rows, m.hidden, &m.q_attn.clone(),
                        1, 8.min(m.attn.head_dim), smem);
    }

    /// Attention, and the reduction that recombines its key splits.
    ///
    /// How many ways to split the key axis is a genuine optimisation: more
    /// splits means more parallel items in a stage that is otherwise starved,
    /// but the reduction that recombines them reads n_heads x splits x head_dim
    /// floats and costs a barrier.  Both sides are priced here.
    fn attention(&mut self) {
        let m = self.m;
        let hd = m.attn.head_dim;
        let kvesz = m.attn.kv_dtype.bytes();
        let span = self.mean_span();

        let mut best = (1usize, f64::INFINITY, 0.0f64);
        let candidates: Vec<usize> = match self.force_splits {
            Some(s) => vec![s],
            None => vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        };
        for splits in candidates {
            if self.force_splits.is_none() && splits > 1 && span / splits < 8 {
                break;
            }
            let per = div_up(span, splits);
            let t_attn = cost_stream(self.hw, self.nt, m.attn.n_heads * splits,
                                     (2 * per * hd * kvesz) as f64, self.warps);
            let t_red = if splits > 1 {
                let (ritems, rbpi) = self.reduce_shape(splits);
                cost_stream(self.hw, self.nt, ritems, rbpi, self.warps) + self.bar
            } else {
                0.0
            };
            if t_attn + t_red < best.1 + best.2 {
                best = (splits, t_attn, t_red);
            }
        }
        let (splits, t_attn, t_red) = best;
        self.splits = splits;
        self.plain_stage(StageKind::Attn, m.attn.n_heads * splits,
                         (span * 2 * m.kv_dim() * kvesz) as u64,
                         (2 * div_up(span, splits) * hd * kvesz) as f64, 0, t_attn);
        if splits > 1 {
            let (ritems, rbpi) = self.reduce_shape(splits);
            self.plain_stage(StageKind::AttnReduce, ritems,
                             (m.attn.n_heads * splits * hd * 4) as u64,
                             rbpi, 0, (t_red - self.bar).max(0.05));
        }
    }

    fn oproj(&mut self) {
        let m = self.m;
        let smem = pad33(m.q_dim()) * 4;
        self.gemv_stage(StageKind::OProj, m.hidden, m.q_dim(), &m.q_attn.clone(),
                        1, 8, smem);
    }

    /// Gemma normalises a sublayer's OUTPUT before the residual add, which needs
    /// the whole vector and therefore its own stage and barrier.
    fn post_norm(&mut self, kind: StageKind) {
        let h = self.m.hidden;
        self.plain_stage(kind, div_up(h, 32), (h * 4) as u64, 128.0, 0, 0.4);
    }

    fn dense_ffn(&mut self) {
        let m = self.m;
        let inter = m.ffn.intermediate();
        let q = m.q_ffn.clone();
        let smem1 = pad33(m.hidden) * 4;
        self.gemv_stage(StageKind::Mlp1, 2 * inter, m.hidden, &q, 2, 8, smem1);
        self.scratch.act_elems = inter;
        let smem2 = pad33(inter) * 4;
        self.gemv_stage(StageKind::Mlp2, m.hidden, inter, &q, 1, 8, smem2);
    }

    fn moe_ffn(&mut self) {
        let m = self.m;
        let inter = m.ffn.intermediate();
        let q = m.q_ffn.clone();
        let (cfg, shared) = match &m.ffn {
            Ffn::Moe { cfg, shared_intermediate, .. } => (cfg.clone(), *shared_intermediate),
            _ => return,
        };

        // one expert per block: computing all gate logits on every block turns a
        // few hundred kB into tens of MB of L2 traffic
        let rb = m.q_attn.bytes(cfg.n_experts, m.hidden) as u64;
        let ritems = cfg.n_experts.min(self.blocks);
        let bpi = rb as f64 / ritems as f64;
        let us = cost(self.hw, m.q_attn.core(), self.nt, 1, ritems, bpi, self.warps).max(0.5);
        self.layer.push(Stage {
            kind: StageKind::Router, items: ritems, bytes: rb, core: m.q_attn.core().to_string(),
            r: 1, bpi, groups: 1.0, smem: pad33(m.hidden) * 4,
            barrier_after: true, est_us: us,
        });
        self.layer.push(Stage {
            kind: StageKind::TopK, items: 1, bytes: 0, core: String::new(),
            r: 1, bpi: 0.0, groups: 1.0, smem: 0, barrier_after: false, est_us: 0.0,
        });

        // Shared experts run BEFORE the routed ones: both consume the ffn_norm(x)
        // still sitting in the arena, and both add into the same residual, so the
        // order only has to keep that norm live.
        if shared > 0 {
            let smem1 = pad33(m.hidden) * 4;
            self.gemv_stage(StageKind::SharedMlp1, 2 * shared, m.hidden, &q, 2, 8, smem1);
            let smem2 = (pad33(m.hidden) + pad33(shared)) * 4;
            self.gemv_stage(StageKind::SharedMlp2, m.hidden, shared, &q, 1, 8, smem2);
            self.scratch.act_elems = self.scratch.act_elems.max(shared);
        }

        let smem1 = pad33(m.hidden) * 4;
        self.gemv_stage(StageKind::Moe1, cfg.top_k * 2 * inter, m.hidden, &q, 2, 8, smem1);
        self.scratch.act_elems = self.scratch.act_elems.max(cfg.top_k * inter);
        self.moe_down(&cfg, inter, &q);
    }

    /// The top-k expert down-projections read the same output rows from
    /// different matrices, so F of them can be interleaved into one group loop.
    /// That multiplies the loads in flight per warp by F -- which is what a
    /// single-wave stage is limited by -- at the cost of F times the live
    /// registers.  F is priced exactly like row blocking: a fused group of F is
    /// a stage with F times the bytes per item and the same item count.
    fn moe_down(&mut self, cfg: &MoeCfg, inter: usize, q: &Quant) {
        let h = self.m.hidden;
        let mut best: Option<(usize, usize, usize, f64, usize)> = None;
        for f in [1usize, 2, 4, 8] {
            if cfg.top_k % f != 0 {
                continue;
            }
            for &r in R_CANDIDATES.iter() {
                if h % r != 0 {
                    continue;
                }
                let regs = self.hw.gemv_peak(&q.core(), self.nt, (r * f).min(8))
                    .map(|g| g.regs).unwrap_or(64);
                if regs > self.reg_budget && !(r == 1 && f == 1) {
                    continue;
                }
                let bpi = (q.bytes(r, inter) * f) as f64;
                let us = cost(self.hw, &q.core(), self.nt, r, h / r, bpi, self.warps)
                    * (cfg.top_k / f) as f64;
                if best.as_ref().map(|b| us < b.3).unwrap_or(true) {
                    best = Some((r, f, h / r, us, regs));
                }
            }
        }
        let (r, f, items, us, regs) = best.expect("at least R=1, F=1 always fits");
        self.regs_max = self.regs_max.max(regs);
        self.moe2_fuse = f;
        self.layer.push(Stage {
            kind: StageKind::Moe2,
            items,
            bytes: (q.bytes(h, inter) * cfg.top_k) as u64,
            core: q.core().to_string(),
            r,
            bpi: (q.bytes(r, inter) * f) as f64,
            groups: (cfg.top_k / f) as f64,
            smem: cfg.top_k * pad33(inter) * 4,
            barrier_after: true,
            est_us: us,
        });
    }

    fn lm_head(&mut self) {
        let m = self.m;
        let rows = m.vocab - m.vocab % 8;
        let g = pick_gemv(self.hw, rows, m.hidden, &m.q_lm, self.nt, 1, 8,
                          self.warps, self.reg_budget);
        self.regs_max = self.regs_max.max(g.regs);
        self.tail.push(Stage {
            kind: StageKind::LmHead,
            items: g.items,
            bytes: m.q_lm.bytes(m.vocab, m.hidden) as u64,
            core: m.q_lm.core().to_string(),
            r: g.r,
            bpi: m.q_lm.bytes(g.r, m.hidden) as f64,
            groups: 1.0,
            smem: pad33(m.hidden) * 4,
            barrier_after: false,
            est_us: g.us,
        });
    }

    fn finish(self, arena: usize, reg_overhead: usize) -> Plan {
        // The arena is sized by the largest stage.  Deriving it from the stage
        // list rather than from a second formula means a new stage cannot
        // silently overflow it: it declares its footprint and the arena grows.
        let arena = arena.max(div_up(
            self.layer.iter().chain(self.tail.iter()).map(|s| s.smem).max().unwrap_or(0), 4));
        // The ONE shared-memory formula.  codegen emits SMEM_BYTES from
        // `Plan::smem_bytes`, so the number `bps_ceiling` divides the SM's
        // shared memory by is the number the kernel is launched with.  They
        // used to be two expressions in two files and disagreed by the
        // expert-index region.
        let smem = smem_bytes(arena, self.nt, self.m.moe().map(|c| c.top_k).unwrap_or(0));
        let nl = self.m.n_layers;
        let nbar = self.layer.iter().filter(|s| s.barrier_after).count();
        let est_ms = (self.layer.iter().map(|s| s.est_us).sum::<f64>() * nl as f64
            + self.tail.iter().map(|s| s.est_us).sum::<f64>()
            + (nbar * nl + 2) as f64 * self.bar)
            / 1000.0;
        Plan {
            nt: self.nt,
            blocks: self.blocks,
            blocks_per_sm: self.bps,
            launch_min_blocks: self.launch_min,
            arena_floats: arena,
            smem_bytes: smem,
            layer: self.layer,
            tail: self.tail,
            attn_splits: self.splits,
            ctx: self.ctx,
            est_ms,
            barriers_per_token: nbar * nl + 2,
            predicted_regs: self.regs_max + reg_overhead,
            moe2_fuse: self.moe2_fuse,
            scratch: self.scratch,
        }
    }
}

/// The whole schedule, in the order the megakernel runs it.
fn build(m: &Model, hw: &Hw, ctx: usize, nt: usize, bps: usize, arena: usize,
         reg_overhead: usize, force_splits: Option<usize>) -> Plan {
    let mut b = Build::new(m, hw, ctx, nt, bps, reg_overhead);
    b.force_splits = force_splits;
    b.qkv();
    b.attention();
    b.oproj();
    if m.post_attn_norm {
        b.post_norm(StageKind::PostAttnNorm);
    }
    match &m.ffn {
        Ffn::Dense { .. } => b.dense_ffn(),
        Ffn::Moe { .. } => b.moe_ffn(),
    }
    if m.post_ffn_norm {
        b.post_norm(StageKind::PostFfnNorm);
    }
    b.lm_head();
    b.finish(arena, reg_overhead)
}

impl Plan {
    /// Whether the register file can hold this schedule at this occupancy.
    /// A prediction, not a verdict: `mkc build` asks the assembler.  `est_ms`
    /// stays an honest time either way, so that a plan the assembler later
    /// accepts is ranked on what it costs, not on what was feared.
    pub fn fits(&self, hw: &Hw) -> bool {
        self.predicted_regs * self.nt * self.blocks_per_sm <= hw.regs_per_sm
    }

    /// Everything about a schedule that changes the generated source.  `mkc
    /// build` uses it to notice that a re-plan produced the same kernel, and so
    /// need not be compiled a second time.
    pub fn signature(&self) -> (usize, usize, usize, usize, Vec<usize>) {
        (self.nt, self.blocks_per_sm + 100 * self.launch_min_blocks.is_none() as usize,
         self.moe2_fuse, self.attn_splits,
         self.layer.iter().chain(self.tail.iter()).map(|s| s.r).collect())
    }

    /// Recompute every stage's cost at a different resident-warp count.
    ///
    /// `mkc build` compiles a candidate to find the occupancy the assembler
    /// actually gives it, then prices THAT schedule here.  The ranking stays
    /// predicted; only the occupancy, which is a static property of the
    /// compiled code, comes from the compiler.
    pub fn repriced(&self, hw: &Hw, m_layers: usize, bps: usize) -> Plan {
        let mut p = self.clone();
        p.blocks_per_sm = bps;
        p.blocks = bps * hw.sms;
        let warps = p.blocks * p.nt / 32;
        for st in p.layer.iter_mut().chain(p.tail.iter_mut()) {
            if st.bpi <= 0.0 || st.items == 0 {
                continue;
            }
            let q = if st.core.is_empty() { "bf16" } else { st.core.as_str() };
            st.est_us = cost(hw, q, p.nt, st.r, st.items, st.bpi, warps) * st.groups;
        }
        let bar = hw.barrier_cost_us(p.blocks);
        p.est_ms = (p.layer.iter().map(|s| s.est_us).sum::<f64>() * m_layers as f64
            + p.tail.iter().map(|s| s.est_us).sum::<f64>()
            + p.barriers_per_token as f64 * bar)
            / 1000.0;
        p
    }

    pub fn render(&self, hw: &Hw) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "plan       NT={} grid={} ({}/SM{}, {} regs) smem={} B  attn_splits={}\n",
            self.nt, self.blocks, self.blocks_per_sm,
            if self.launch_min_blocks.is_none() { ", assembler's choice" } else { "" },
            self.predicted_regs, self.smem_bytes, self.attn_splits
        ));
        s.push_str(&format!(
            "           {:<10} {:>9} {:>10} {:>4} {:>8} {:>9}\n",
            "stage", "items", "MB", "R", "us", "GB/s"
        ));
        for st in self.layer.iter().chain(self.tail.iter()) {
            let gbs = if st.est_us > 0.0 {
                (st.bytes as f64 / 1e9) / (st.est_us / 1e6)
            } else {
                0.0
            };
            s.push_str(&format!(
                "           {:<10} {:>9} {:>10.2} {:>4} {:>8.2} {:>9.0}{}\n",
                st.kind.short(),
                st.items,
                st.bytes as f64 / 1e6,
                st.r,
                st.est_us,
                gbs,
                if st.barrier_after { "  |" } else { "" }
            ));
        }
        s.push_str(&format!(
            "           barriers/token={} at {:.2} us = {:.3} ms\n",
            self.barriers_per_token,
            hw.barrier_cost_us(self.blocks),
            self.barriers_per_token as f64 * hw.barrier_cost_us(self.blocks) / 1000.0
        ));
        s.push_str(&format!("           predicted {:.3} ms/token\n", self.est_ms));
        s
    }
}
