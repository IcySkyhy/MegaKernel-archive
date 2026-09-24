"""Benchmark decode-kernel experiment variants on the current GPU."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

RUNNER = REPO_ROOT / "experimental" / "variant_runner.py"
RESULTS_DIR = REPO_ROOT / "experimental" / "results"


def _tail_to_text(value: object, max_chars: int = 1000) -> str:
    """Convert subprocess output tails to JSON-safe text."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")[-max_chars:]
    return str(value)[-max_chars:]


def _run_variant(
    variant: str,
    tokens: int,
    warmup: int,
    runs: int,
    prompt: str,
    timeout_s: int,
) -> dict:
    cmd = [
        sys.executable,
        str(RUNNER),
        "--variant",
        variant,
        "--tokens",
        str(tokens),
        "--warmup",
        str(warmup),
        "--runs",
        str(runs),
        "--prompt",
        prompt,
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "variant": variant,
            "tokens": tokens,
            "warmup": warmup,
            "runs": runs,
            "status": "timeout",
            "error": f"timeout after {timeout_s}s",
            "stdout_tail": _tail_to_text(exc.stdout),
            "stderr_tail": _tail_to_text(exc.stderr),
        }

    if proc.returncode != 0:
        return {
            "variant": variant,
            "tokens": tokens,
            "warmup": warmup,
            "runs": runs,
            "status": "failed",
            "error": f"exit code {proc.returncode}",
            "stdout_tail": _tail_to_text(proc.stdout),
            "stderr_tail": _tail_to_text(proc.stderr),
        }

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"variant '{variant}' produced no output")

    # Result JSON is the last line from variant_runner.
    try:
        result = json.loads(lines[-1])
    except json.JSONDecodeError:
        return {
            "variant": variant,
            "tokens": tokens,
            "warmup": warmup,
            "runs": runs,
            "status": "failed",
            "error": "missing result json",
            "stdout_tail": _tail_to_text(proc.stdout),
            "stderr_tail": _tail_to_text(proc.stderr),
        }

    result["status"] = "ok"

    return result


def _print_summary(results: list[dict]) -> None:
    baseline = None
    for row in results:
        if row["variant"] == "baseline" and row.get("status") == "ok":
            baseline = row
            break

    print()
    print("Decode Variant Results")
    print("=" * 84)
    print(f"{'Variant':<24} {'tok/s':>10} {'ms/tok':>10} {'mean ms':>10} {'speedup':>10}")
    print("-" * 84)

    for row in results:
        if row.get("status") != "ok":
            print(f"{row['variant']:<24} {'ERR':>10} {'ERR':>10} {'ERR':>10} {'ERR':>10}")
            continue

        speedup = 1.0
        if baseline is not None:
            speedup = row["tok_s"] / baseline["tok_s"]
        print(
            f"{row['variant']:<24} "
            f"{row['tok_s']:>10.1f} "
            f"{row['ms_tok']:>10.3f} "
            f"{row['mean_s'] * 1000:>10.2f} "
            f"{speedup:>10.3f}x"
        )


def main() -> int:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from experimental.variant_sources import list_variants

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="*", default=list(list_variants()))
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--timeout-s", type=int, default=600)
    args = parser.parse_args()

    results = []
    for variant in args.variants:
        print(f"[run] {variant}")
        result = _run_variant(
            variant=variant,
            tokens=args.tokens,
            warmup=args.warmup,
            runs=args.runs,
            prompt=args.prompt,
            timeout_s=args.timeout_s,
        )
        results.append(result)

    _print_summary(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / "decode_variant_results.json"
    output_file.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print()
    print(f"Saved: {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
