"""Standard-library-only experiment helpers; all GPU work is serial."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / 'build/lbm'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_hashes():
    return {str(p.relative_to(ROOT)): digest(p)
            for folder in ('src', 'kernels', 'include')
            for p in sorted((ROOT / folder).iterdir()) if p.is_file()}

def variants():
    return subprocess.check_output([str(BIN), '--list'], text=True).splitlines()

def run(args, path):
    result = subprocess.run([str(a) for a in args], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, cwd=ROOT)
    path.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f'Exit {result.returncode}: {args}; see {path}')
    return result.stdout

def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

def environment():
    result = {'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES']}
    for name, command in {
        'gpu': ['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,memory.used,utilization.gpu', '--format=csv'],
        'build': ['cat', str(ROOT / 'build/config.txt')],
        'nvcc': ['/usr/local/cuda-12.9/bin/nvcc', '--version'],
    }.items():
        try:
            result[name] = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
        except (OSError, subprocess.CalledProcessError) as exc:
            result[name] = str(exc)
    return result
