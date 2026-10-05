//! Machine model.  Written once per GPU type by `mkc calibrate`, read by the
//! planner on every compile.  This file is the ONLY place measurement enters
//! the compiler: after it exists, compiling a model is pure computation.

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RampPoint {
    pub bytes: u64,
    pub gbs: f64,
}

/// Peak achieved bandwidth of one gemv core at full occupancy and a large
/// footprint, for one (quant, block size, row blocking) triple.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GemvPoint {
    pub quant: String,
    pub nt: usize,
    pub r: usize,
    pub gbs: f64,
    /// blocks per SM this configuration achieved (occupancy API)
    pub blocks_per_sm: usize,
    /// registers per thread, from cudaFuncGetAttributes -- the planner needs
    /// this to predict the fused kernel's occupancy without compiling it
    #[serde(default)]
    pub regs: usize,
}

/// One measured stage: `items` work items over `bytes` of cold weights, run
/// inside a persistent cooperative grid, barrier cost already subtracted.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StagePoint {
    pub quant: String,
    pub nt: usize,
    pub r: usize,
    pub k: usize,
    pub bytes: u64,
    pub items: u64,
    pub us: f64,
    pub gbs: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Hw {
    /// Machine-file schema.  Bumped when the meaning of a field changes; `mkc`
    /// refuses an older file rather than silently planning every stage from a
    /// proxy curve, which is what a missing table looks like from the inside.
    #[serde(default)]
    pub schema: u32,
    pub device: String,
    pub sm_arch: u32,
    pub sms: usize,
    pub clock_ghz: f64,
    pub l2_bytes: u64,
    pub smem_per_sm: usize,
    pub smem_per_block_max: usize,
    pub max_threads_per_sm: usize,
    pub regs_per_sm: usize,
    pub dram_peak_gbs: f64,
    /// measured pure-streaming read ceiling at a large footprint
    pub stream_peak_gbs: f64,
    /// achieved read bandwidth as a function of how many bytes are streamed in
    /// one go -- the reason a 17 MB stage never reaches the 1 GB number
    pub ramp: Vec<RampPoint>,
    pub gemv: Vec<GemvPoint>,
    #[serde(default)]
    pub stage: Vec<StagePoint>,
    /// cooperative grid barrier cost, microseconds, by grid size
    pub barrier_us: Vec<(usize, f64)>,
    /// one kernel launch, microseconds (for the non-mega comparison path)
    pub launch_us: f64,
    pub calibrated_at: String,
    pub mkc_version: String,
}

/// The schema `mkc` understands.  See `Quant::core`.
pub const SCHEMA: u32 = 2;

impl Hw {
    pub fn gemv_peak(&self, quant: &str, nt: usize, r: usize) -> Option<&GemvPoint> {
        self.gemv.iter().find(|g| g.quant == quant && g.nt == nt && g.r == r)
    }

    /// Best measured rate for a quant across all geometries -- the denominator
    /// for "how close is this stage to its own ceiling", and the physical floor
    /// the stage cost is clamped to.
    pub fn gemv_best(&self, quant: &str) -> f64 {
        self.gemv.iter().filter(|g| g.quant == quant).map(|g| g.gbs).fold(0.0, f64::max)
    }

    /// Resident warps the calibration had for this configuration.  The stage
    /// table is a function of concurrency, and the calibration ran at whatever
    /// occupancy the standalone core happened to get; a fused megakernel runs
    /// at a different one, so the lookup has to be corrected for it or every
    /// prediction is wrong by the occupancy ratio.
    pub fn calib_warps(&self, quant: &str, nt: usize, r: usize) -> usize {
        self.gemv_peak(quant, nt, r)
            .map(|g| g.blocks_per_sm * self.sms * nt / 32)
            .unwrap_or(self.sms * nt / 32)
            .max(1)
    }

    /// Predicted time of one megakernel stage at a given resident-warp count.
    ///
    /// t = waves * t_wave(concurrency), with t_wave read out of the measured
    /// table at the concurrency the plan will actually have.  Both factors
    /// matter: halving occupancy doubles the waves for a long stage, and caps
    /// the concurrency of a short one.
    pub fn stage_us_at(&self, quant: &str, nt: usize, r: usize, items: u64,
                       bytes_per_item: f64, warps_plan: usize) -> Option<f64> {
        let wc = self.calib_warps(quant, nt, r) as u64;
        let wp = warps_plan.max(1) as u64;
        let conc = items.min(wp);
        let waves = (items + wp - 1) / wp;
        let lookup = conc.min(wc);
        let t_wave = self.stage_us(quant, nt, r, lookup, bytes_per_item)?;
        // more concurrency than the calibration observed: the table point at
        // that item count already contains waves of its own, so remove them
        let t_wave = if conc > wc {
            let extra = ((conc + wc - 1) / wc) as f64;
            self.stage_us(quant, nt, r, conc, bytes_per_item)? / extra
        } else {
            t_wave
        };
        Some(waves as f64 * t_wave)
    }

    pub fn stage_us(&self, quant: &str, nt: usize, r: usize, items: u64, bytes_per_item: f64) -> Option<f64> {
        if items == 0 || bytes_per_item <= 0.0 {
            return Some(0.0);
        }
        // group the measured points by reduction length
        let mut ks: Vec<usize> = self
            .stage
            .iter()
            .filter(|s| s.quant == quant && s.nt == nt && s.r == r)
            .map(|s| s.k)
            .collect();
        ks.sort();
        ks.dedup();
        if ks.is_empty() {
            return None;
        }
        let mut per_k: Vec<(f64, f64)> = Vec::new(); // (ln bytes_per_item, ln us)
        for k in ks {
            let mut pts: Vec<(f64, f64)> = self
                .stage
                .iter()
                .filter(|s| s.quant == quant && s.nt == nt && s.r == r && s.k == k)
                .map(|s| ((s.items as f64).ln(), s.us.max(1e-4).ln()))
                .collect();
            if pts.is_empty() {
                continue;
            }
            pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let li = (items as f64).ln();
            let lu = interp_extrap_min1(&pts, li);
            let bpi = self
                .stage
                .iter()
                .find(|s| s.quant == quant && s.nt == nt && s.r == r && s.k == k)
                .map(|s| s.bytes as f64 / s.items as f64)
                .unwrap_or(1.0);
            per_k.push((bpi.ln(), lu));
        }
        if per_k.is_empty() {
            return None;
        }
        per_k.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let us = interp_extrap(&per_k, bytes_per_item.ln()).exp();
        // Physical floor.  Interpolating measured points can only be trusted
        // inside their range; beyond it the one thing that is certain is that
        // no stage moves bytes faster than this core's measured ceiling.
        let ceil = {
            let c = self.gemv_best(quant);
            if c > 0.0 { c } else { self.stream_peak_gbs }
        };
        let floor_us = (items as f64 * bytes_per_item) / (ceil * 1e9) * 1e6;
        Some(us.max(floor_us))
    }

    pub fn barrier_cost_us(&self, blocks: usize) -> f64 {
        if self.barrier_us.is_empty() {
            return 1.1;
        }
        let mut best = self.barrier_us[0];
        for &(b, c) in &self.barrier_us {
            if (b as i64 - blocks as i64).abs() < (best.0 as i64 - blocks as i64).abs() {
                best = (b, c);
            }
        }
        best.1
    }
}

/// As `interp_extrap`, but the slope used to extrapolate ABOVE the measured
/// range is at least 1.  Beyond saturation, doubling the work doubles the time;
/// a fitted sub-linear slope extrapolated far enough predicts a bandwidth the
/// machine does not have.
fn interp_extrap_min1(pts: &[(f64, f64)], x: f64) -> f64 {
    let n = pts.len();
    if n >= 2 && x > pts[n - 1].0 {
        let (a, b) = (pts[n - 2], pts[n - 1]);
        let slope = ((b.1 - a.1) / (b.0 - a.0)).max(1.0);
        return b.1 + (x - b.0) * slope;
    }
    interp_extrap(pts, x)
}

/// Piecewise-linear interpolation with linear extrapolation off both ends.
fn interp_extrap(pts: &[(f64, f64)], x: f64) -> f64 {
    if pts.len() == 1 {
        return pts[0].1;
    }
    if x <= pts[0].0 {
        let (a, b) = (pts[0], pts[1]);
        return a.1 + (x - a.0) * (b.1 - a.1) / (b.0 - a.0);
    }
    for w in pts.windows(2) {
        if x <= w[1].0 {
            let (a, b) = (w[0], w[1]);
            return a.1 + (x - a.0) * (b.1 - a.1) / (b.0 - a.0);
        }
    }
    let n = pts.len();
    let (a, b) = (pts[n - 2], pts[n - 1]);
    b.1 + (x - b.0) * (b.1 - a.1) / (b.0 - a.0)
}
