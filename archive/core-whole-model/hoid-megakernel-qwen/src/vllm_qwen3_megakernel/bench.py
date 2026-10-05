"""``hoid-bench``: the published benchmark (``bench/run_all.py`` in this repository's checkout)."""
from pathlib import Path
import sys


def main():
    bench = Path(__file__).resolve().parents[2] / 'bench'
    if not (bench / 'run_all.py').exists():
        sys.exit('hoid-bench runs from a checkout of this repository: uv sync, then uv run hoid-bench')
    sys.path.insert(0, str(bench))
    import run_all
    run_all.main()
