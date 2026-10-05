"""The published benchmark: contexts x batch sizes x engines, every measurement in a fresh process.

Engines: ``hoid`` (vLLM + megakernel, this project's environment), ``stock`` (vLLM), ``sglang`` and
``trtllm`` (TensorRT-LLM). Each peer engine gets its own environment under ``--venv-dir``, built
from ``envs/*.lock`` on first use. Per (context, batch) the engine order runs forward then
reversed, repeated (``--repeats 4``), so slow drift on the box lands on every engine alike. SGLang
and TensorRT-LLM run with the tuned options ``peer_configs.json`` records for each batch size.
The FP32 reference is produced first when missing. Re-running the same ``--out`` resumes: finished
reports are kept, missing ones are run. Finally ``summarize.py`` writes ``summary.json`` and
``RESULTS.md`` into ``--out``.
"""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from workload import BATCHES, PROMPT_TOKENS

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent
BUNDLE = ROOT / 'src' / 'vllm_qwen3_megakernel' / 'cubins'
PLUGIN_ENV = {'VLLM_PLUGINS': 'qwen3_megakernel', 'VLLM_USE_V2_MODEL_RUNNER': '1'}
ENGINES = ('hoid', 'stock', 'sglang', 'trtllm')
LOCKS = {'stock': 'vllm.lock', 'sglang': 'sglang.lock', 'trtllm': 'trtllm.lock'}
# sglang 0.5.21 pins a pre-release (cuda-tile); TensorRT-LLM comes from NVIDIA's index.
INSTALL_FLAGS = {'stock': [], 'sglang': ['--prerelease=allow'],
                 'trtllm': ['--extra-index-url', 'https://pypi.nvidia.com', '--index-strategy', 'unsafe-best-match']}
IMPORT_CHECK = {'stock': 'import vllm', 'sglang': 'import sglang', 'trtllm': 'import tensorrt_llm'}


def order(engines, repeats):
    if repeats < 2 or repeats % 2:
        sys.exit('--repeats must be even and >= 2')
    return [arm for _ in range(repeats // 2) for arm in (*engines, *reversed(engines))]


def environment(engine, venv_dir):
    """Python of ``engine``'s environment, built from its lock file unless already current."""
    if engine == 'hoid':
        return Path(sys.executable)
    lock = BENCH / 'envs' / LOCKS[engine]
    venv = venv_dir / engine
    python = venv / 'bin' / 'python'
    stamp = venv / '.lock.sha256'
    digest = hashlib.sha256(lock.read_bytes()).hexdigest()
    if python.exists() and stamp.exists() and stamp.read_text() == digest:
        return python
    uv = shutil.which('uv') or sys.exit('uv is required: https://docs.astral.sh/uv/')
    print(f'building the {engine} environment in {venv} ...', flush=True)
    env = dict(os.environ, UV_LINK_MODE='copy')
    # --no-config: this project's own uv settings pin the vLLM stack and must not leak into peer envs.
    subprocess.run([uv, 'venv', '--no-config', '--allow-existing', '--python', '3.12', str(venv)], check=True, env=env)
    subprocess.run([uv, 'pip', 'install', '--no-config', '--python', str(python), *INSTALL_FLAGS[engine],
                    '-r', str(lock)], check=True, env=env)
    subprocess.run([str(python), '-c', IMPORT_CHECK[engine]], check=True, env=child_env(engine))
    stamp.write_text(digest)
    return python


def child_env(engine):
    env = {k: v for k, v in os.environ.items() if k not in (*PLUGIN_ENV, 'VLLM_ATTENTION_BACKEND')}
    if engine == 'hoid':
        env.update(PLUGIN_ENV)
    return env


def command(engine, python, batch, context, args, report):
    common = ['--batch', str(batch), '--context', context, '--weights', str(args.weights), '--report', str(report)]
    if engine in ('hoid', 'stock'):
        return [str(python), 'run_one.py', '--engine', engine, '--bundle', str(BUNDLE), *common]
    tuned = json.loads((BENCH / 'peer_configs.json').read_text())[engine][str(batch)]
    sets = [f'--set={key}={json.dumps(value)}' for key, value in tuned.items()]
    argv = [str(python), 'run_peer.py', '--engine', engine, *common, *sets]
    if engine == 'trtllm':
        # TensorRT-LLM spawns its executor with MPI_Comm_Spawn, which Open MPI 5 only
        # supports under its launcher.
        argv = [str(python.parent / 'mpirun'), '-n', '1', '--allow-run-as-root', *argv]
    return argv


def run(command, log, env, locks):
    """One child process in its own session (its watchdog kills only its own process group),
    holding every ``--lock`` file exclusively for its whole life."""
    with ExitStack() as stack:
        for path in locks:
            handle = stack.enter_context(open(path, 'a'))
            fcntl.flock(handle, fcntl.LOCK_EX)
        with open(log, 'w') as output:
            child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=env, cwd=BENCH,
                                     start_new_session=True)
            try:
                return child.wait()
            except BaseException:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--weights', type=Path, help='checkpoint directory (default: the pinned download)')
    parser.add_argument('--out', type=Path, default=ROOT / 'results' / 'latest')
    parser.add_argument('--venv-dir', type=Path, default=Path(os.environ.get('VENV_DIR', ROOT / '.venvs')))
    parser.add_argument('--contexts', default='8k')
    parser.add_argument('--batches', default=','.join(map(str, BATCHES)))
    parser.add_argument('--engines', default=','.join(ENGINES))
    parser.add_argument('--repeats', type=int, default=4, help='fresh processes per engine (even)')
    parser.add_argument('--lock', action='append', default=[], type=Path,
                        help='file to flock around every GPU process (repeat); for shared boxes')
    args = parser.parse_args()
    engines = args.engines.split(',')
    if not set(engines) <= set(ENGINES):
        sys.exit(f'--engines must be drawn from {ENGINES}')
    if not set(args.contexts.split(',')) <= set(PROMPT_TOKENS):
        sys.exit(f'--contexts must be drawn from {tuple(PROMPT_TOKENS)}')
    # Every engine, the peers included, is measured only on the GPU the published numbers came from.
    from vllm_qwen3_megakernel.gpu import require_h200
    require_h200()
    if args.weights is None:
        from vllm_qwen3_megakernel.weights import checkpoint
        args.weights = checkpoint()
    arms = order(engines, args.repeats)
    python = {engine: environment(engine, args.venv_dir.resolve()) for engine in engines}
    out = args.out.resolve()
    (out / 'runs').mkdir(parents=True, exist_ok=True)
    subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'],
                   stdout=open(out / 'gpu.csv', 'w'), check=True)

    reference = out / 'reference.json'
    if not reference.exists():
        print('FP32 reference ...', flush=True)
        code = run([sys.executable, 'reference.py', '--weights', str(args.weights), '--out', str(reference)],
                   out / 'reference.log', child_env('stock'), args.lock)
        if code:
            sys.exit(f'reference failed; see {out / "reference.log"}')

    failures = []
    for context in args.contexts.split(','):
        for batch in map(int, args.batches.split(',')):
            if 'stock' in engines:
                # vLLM's PDL kernels only pay off when loaded from its compile cache; the process
                # that compiles them decodes ~1% slower. One untimed process fills the cache so
                # every timed vLLM process starts warm.
                primer = out / 'primers' / f'{context}-b{batch}-stock.json'
                if not primer.exists():
                    primer.parent.mkdir(exist_ok=True)
                    print(f'[{context} b{batch}] stock compile-cache primer', flush=True)
                    if run(command('stock', python['stock'], batch, context, args, primer),
                           primer.with_suffix('.log'), child_env('stock'), args.lock):
                        failures.append(primer.name)
                        print(f'  FAILED, see {primer.with_suffix(".log")}', flush=True)
            for index, engine in enumerate(arms):
                report = out / 'runs' / f'{context}-b{batch}-{index:02d}-{engine}.json'
                if report.exists():
                    continue
                print(f'[{context} b{batch}] {index:02d} {engine}', flush=True)
                argv = command(engine, python[engine], batch, context, args, report)
                if run(argv, report.with_suffix('.log'), child_env(engine), args.lock):
                    failures.append(report.name)
                    print(f'  FAILED, see {report.with_suffix(".log")}', flush=True)
    subprocess.run([sys.executable, str(BENCH / 'summarize.py'), str(out)], check=True)
    if failures:
        sys.exit(f'{len(failures)} runs failed: {failures}')


if __name__ == '__main__':
    main()
