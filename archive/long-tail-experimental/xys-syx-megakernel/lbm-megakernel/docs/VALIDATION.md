# Validation of the standalone extraction

`src/driver.cuh` runs the selected candidate against B0 in the same process.
B0 uses the original cell body; temporal kernels use the expression-matched
collision helper. `scripts/check_source.py` checks the expressions bytewise
after rebinding flag access. The source provenance manifest records the
original files and extracted snapshot hashes. The consolidated T30 V2 template
uses 63 registers/thread in this build, versus 56 in the original T30 binary;
its collision expressions, shared footprint and residency are unchanged.
Performance tables therefore use fresh standalone measurements throughout.

For each checkpoint (2, 4, 6 and the requested final step), the driver compares
all 41,040,000 physical FP32 populations and 2,160,000 flag bytes per buffer.
It also compares every byte of the initialized allocations, including unused
flag-slot bytes, margins and padding. It checks finite population bit patterns,
maximum absolute error, RMS, and relative error (denominator >= 1e-12).
The pass criterion is bitwise equality and zero nonfinite populations.

Both active and inactive allocations are checked against the appropriate
baseline states: current N; inactive N−1 for one-step kernels and N−2 for
fused kernels. This is an explicit interface difference, not a claim that
both raw buffers at N match the ordinary ping-pong pair.

`make check` tests every registered variant with nonuniform populations,
internal and file-based obstacles, nonzero allocation sentinels, and perturbed
never-produced population slots at six physical steps. It enumerates
324,236 invariant physical population slots and 178,204 physical slots whose
predecessors cross a geometric Y boundary by flat aliasing. Obstacles cross
cube, elongated and T30 interfaces, including partial end tiles. Velocity
files at 2/6 steps must match B0 bytewise. Zero, negative, odd and nonnumeric
step counts are rejected.

`python3 scripts/validate.py --full --sanitizers` adds, for all 20 variants:

- Default LDC at 1000 and 1002 steps, testing both fused output parities.
- Nonuniform populations and internal obstacles at 1000 steps.
- External interface obstacles at 102 steps.
- Six-step patterned/sentinel/external-obstacle runs under memcheck,
  racecheck, synccheck and global-memory initcheck.

Initcheck here checks global memory; racecheck and numerical gates cover
shared accesses. This extracted production code has no debug ownership
atomics or launch-assertion hooks. Original experiments also used instrumented
writer-count gates; those are not claimed as newly run in this repository.
The saved `validation.json` states exactly which suite was run and records
binary/source hashes. Benchmarking requires those hashes to match.

An optional `--reference /path/to/upstream/lbm_clang_cuda` compares B0 output
at 2/6/100 steps against an independently built original executable. This
cross-check was used during extraction, but is not a build or test dependency
for a standalone checkout. Output hashes are retained after large files are
removed. The original tests and the new checks provide evidence for these
inputs and horizons, not arbitrary mutable boundaries or untested layouts.

A source-only copy was also built outside the original workspace; its binary
was byte-identical to the validated build (`results/rtx5090/portability.json`).
SASS inspection found zero barriers in each spatial kernel and one barrier
in each temporal kernel, with no atomic instructions (`sass.json`). Those
files record extraction-time checks; the default validation script does not
repeat the isolated copy/build or SASS inspection.
