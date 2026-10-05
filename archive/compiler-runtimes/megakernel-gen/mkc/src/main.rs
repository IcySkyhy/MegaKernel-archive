//! mkc -- the megakernel compiler.
//!
//!   mkc analyze  <model_dir>              parse + roofline, no codegen
//!   mkc compile  <model_dir> -o <out>     emit the megakernel and its runtime
//!   mkc plan     <model_dir>              print the stage schedule it would emit
//!
//! Compilation is deterministic: the same model directory and the same machine
//! file always produce byte-identical output.  Nothing is searched or sampled.

mod arch;
mod codegen;
mod emit;
mod hf;
mod hw;
mod layout;
mod ir;
mod plan;
mod quant;
mod report;
mod search;
mod validate;

use std::path::{Path, PathBuf};
use std::process::exit;

pub const VERSION: &str = "0.1.0";

pub struct Args {
    cmd: String,
    model: PathBuf,
    out: PathBuf,
    hw: Option<PathBuf>,
    ctx: usize,
    name: Option<String>,
    layers: Option<usize>,
    verbose: bool,
    verify: bool,
}

fn usage() -> ! {
    eprintln!(
        "mkc {VERSION} -- megakernel compiler

USAGE
  mkc analyze <model_dir> [--ctx N]
  mkc plan    <model_dir> [--hw FILE] [--ctx N]
  mkc compile <model_dir> -o <out_dir> [--hw FILE] [--ctx N] [--name NAME] [--layers N]
  mkc build   <model_dir> -o <out_dir> [same]      compile, then re-plan against
                                                   the assembler's real register
                                                   count until they agree

OPTIONS
  --hw FILE     machine model from `mkc calibrate` (default: hw/<device>.json)
  --ctx N       context length the plan is optimised for (default 1024)
  --name NAME   symbol prefix for the generated code (default: dir basename)
  --layers N    compile only the first N layers (debugging aid)
  --verify      (build) time every schedule the assembler accepted, on
                synthetic weights, and keep the measured winner.  Bounded (the
                search compiles a handful) and deterministic; off by default,
                so a plain build touches no GPU.
  -v            print the resolved model IR
"
    );
    exit(2)
}

fn parse() -> Args {
    let mut a = std::env::args().skip(1);
    let cmd = a.next().unwrap_or_else(|| usage());
    if cmd == "--help" || cmd == "-h" {
        usage()
    }
    let model = PathBuf::from(a.next().unwrap_or_else(|| usage()));
    let mut out = PathBuf::from("out");
    let mut hw = None;
    let mut ctx = 1024;
    let mut name = None;
    let mut layers = None;
    let mut verbose = false;
    let mut verify = false;
    while let Some(f) = a.next() {
        match f.as_str() {
            "-o" | "--out" => out = PathBuf::from(a.next().unwrap_or_else(|| usage())),
            "--hw" => hw = Some(PathBuf::from(a.next().unwrap_or_else(|| usage()))),
            "--ctx" => ctx = a.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| usage()),
            "--name" => name = a.next(),
            "--layers" => layers = a.next().and_then(|v| v.parse().ok()),
            "-v" => verbose = true,
            "--verify" => verify = true,
            _ => usage(),
        }
    }
    Args { cmd, model, out, hw, ctx, name, layers, verbose, verify }
}

pub fn die(msg: String) -> ! {
    eprintln!("mkc: error: {}", msg);
    exit(1)
}

/// Resolve the machine model: `--hw`, else `$MKC_HW`, else
/// `hw/<device>.json` under `$MKGEN_ROOT`.  The env var is what lets a batch
/// job calibrate into a writable directory -- `/home` is read-only on many
/// compute nodes -- and still have every compile in that job use it.
/// The one check that needs the machine as well as the model: a shared arena
/// bigger than an SM's opt-in maximum compiles and then fails at launch.
fn check_fits(m: &ir::Model, h: &hw::Hw) {
    let smallest = [256usize, 512, 1024].iter().map(|&nt| plan::smem_needed(m, nt)).min().unwrap();
    if smallest > h.smem_per_block_max {
        die(format!(
            "the activation arena needs {} B of shared memory and {} allows {} B per block; \
             this model's intermediate or head dimension is too wide for the staged layout",
            smallest, h.device, h.smem_per_block_max));
    }
}

fn load_hw(explicit: &Option<PathBuf>) -> Option<hw::Hw> {
    let mut cands: Vec<PathBuf> = Vec::new();
    if let Some(p) = explicit {
        cands.push(p.clone());
    } else if let Ok(p) = std::env::var("MKC_HW") {
        cands.push(PathBuf::from(p));
    } else {
        let root = std::env::var("MKGEN_ROOT").unwrap_or_else(|_| ".".into());
        if let Ok(dev) = std::env::var("MKGEN_DEVICE_NAME") {
            let slug: String = dev
                .to_lowercase()
                .chars()
                .map(|c| if c.is_alphanumeric() { c } else { '_' })
                .collect();
            cands.push(Path::new(&root).join("hw").join(format!("{}.json", slug)));
        }
        cands.push(Path::new(&root).join("hw").join("default.json"));
    }
    for c in cands {
        if let Ok(t) = std::fs::read_to_string(&c) {
            match serde_json::from_str::<hw::Hw>(&t) {
                Ok(h) => {
                    if h.schema != hw::SCHEMA {
                        die(format!(
                            "{}: machine file is schema {}, this mkc needs {} -- \
                             re-run `mkc calibrate` (the cost tables are keyed \
                             differently and every stage would be priced from a proxy)",
                            c.display(), h.schema, hw::SCHEMA));
                    }
                    eprintln!("mkc: machine model {}", c.display());
                    return Some(h);
                }
                Err(e) => die(format!("{}: {}", c.display(), e)),
            }
        }
    }
    None
}

fn main() {
    let args = parse();
    let hfm = match hf::HfModel::load(&args.model) {
        Ok(h) => h,
        Err(e) => die(e),
    };
    let base = args
        .model
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "model".into());
    let name = args.name.clone().unwrap_or(base);
    let mut model = match arch::build(&hfm, &name) {
        Ok(m) => m,
        Err(e) => die(e),
    };
    if let Some(n) = args.layers {
        model.n_layers = model.n_layers.min(n);
        model.attn.window.truncate(model.n_layers);
    }

    // Refuse loudly rather than emit a kernel for a feature that is parsed but
    // not lowered.  `analyze` still runs, so the roofline of an unsupported
    // model is still available.
    if args.cmd != "analyze" {
        if let Err(e) = validate::check(&model) {
            die(e)
        }
    }

    let machine = load_hw(&args.hw);
    let peak = machine.as_ref().map(|h| h.stream_peak_gbs).unwrap_or(3176.0);

    match args.cmd.as_str() {
        "analyze" => {
            println!("{}", describe(&model));
            let rl = report::roofline(&model, args.ctx);
            println!("{}", rl.render(peak));
            if args.verbose {
                for (k, w) in &model.weights {
                    println!(
                        "  {:<24} {:>12} {:?} <- {} part(s) {:?}",
                        k,
                        format!("{:?}", w.shape),
                        w.out_dtype,
                        w.parts.len(),
                        w.transform
                    );
                }
            }
        }
        "plan" => {
            let Some(h) = machine else {
                die("plan needs a machine model; run `mkc calibrate` (see hw/README)".into())
            };
            check_fits(&model, &h);
            let p = plan::plan(&model, &h, args.ctx);
            println!("{}", describe(&model));
            println!("{}", report::roofline(&model, args.ctx).render(peak));
            println!("{}", p.render(&h));
        }
        "build" => {
            let Some(h) = machine else {
                die("build needs a machine model; run `mkc calibrate` first".into())
            };
            check_fits(&model, &h);
            search::cmd_build(&model, &h, &args, peak);
        }
        "compile" => {
            let Some(h) = machine else {
                die("compile needs a machine model; run `mkc calibrate` first".into())
            };
            check_fits(&model, &h);
            // MKC_NT / MKC_BPS force a geometry, for A/B experiments that
            // calibrate the planner's rules.  Unset in normal use.
            let p = match (std::env::var("MKC_NT").ok().and_then(|v| v.parse().ok()),
                           std::env::var("MKC_BPS").ok().and_then(|v| v.parse().ok())) {
                (Some(nt), Some(bps)) =>
                    plan::plan_fixed(&model, &h, args.ctx, nt, bps, &plan::Feedback::default(), None),
                _ => plan::plan(&model, &h, args.ctx),
            };
            if let Err(e) = emit::emit_all(&model, &p, &h, &args.out, args.ctx) {
                die(e)
            }
            println!("{}", describe(&model));
            println!("{}", report::roofline(&model, args.ctx).render(peak));
            println!("{}", p.render(&h));
            println!("wrote {}", args.out.display());
        }
        _ => usage(),
    }
}


pub fn describe(m: &ir::Model) -> String {
    let mut s = String::new();
    s.push_str(&format!("model      {}  ({})\n", m.name, m.arch));
    s.push_str(&format!(
        "shape      hidden={} layers={} vocab={} act={}\n",
        m.hidden, m.n_layers, m.vocab, m.act_dtype
    ));
    s.push_str(&format!(
        "attention  {}q/{}kv heads x {}  scale={:.5}{}{}{}\n",
        m.attn.n_heads,
        m.attn.n_kv_heads,
        m.attn.head_dim,
        m.attn.scale,
        if m.attn.q_norm.is_some() { "  qk-norm" } else { "" },
        if m.attn.sinks { "  sinks" } else { "" },
        match m.attn.window.iter().flatten().next() {
            Some(w) => format!(
                "  window={} on {}/{} layers",
                w,
                m.attn.window.iter().filter(|x| x.is_some()).count(),
                m.n_layers
            ),
            None => String::new(),
        }
    ));
    s.push_str(&format!("rope       theta={} rotary_dim={} {:?}\n", m.rope.theta, m.rope.rotary_dim, m.rope.scaling));
    match &m.ffn {
        ir::Ffn::Dense { intermediate, act, .. } => {
            s.push_str(&format!("ffn        dense inter={} act={:?} quant={}\n", intermediate, act, m.q_ffn.tag()));
        }
        ir::Ffn::Moe { intermediate, act, cfg, shared_intermediate } => {
            s.push_str(&format!(
                "ffn        moe {} experts, top{} inter={} act={:?} quant={} score={:?}{}\n",
                cfg.n_experts, cfg.top_k, intermediate, act, m.q_ffn.tag(), cfg.score,
                if *shared_intermediate > 0 { format!(" shared={}", shared_intermediate) } else { String::new() }
            ));
        }
    }
    s.push_str(&format!(
        "head       {}  weights={} roles\n",
        if m.tie_embeddings { "tied" } else { "untied" },
        m.weights.len()
    ));
    s
}
