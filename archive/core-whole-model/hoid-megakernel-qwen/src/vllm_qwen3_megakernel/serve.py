"""``hoid-serve``: an OpenAI-compatible vLLM server for Qwen3-4B-Instruct-2507 with every decode
step on the Hoid megakernel. Extra arguments are passed on to ``vllm serve`` and override these."""
import json
import os
from pathlib import Path
import sys

from .gpu import require_h200
from .weights import REPOSITORY, checkpoint

BUNDLE = Path(__file__).resolve().parent / 'cubins'


def main():
    require_h200()
    weights = checkpoint()
    megakernel = {'bundle': str(BUNDLE), 'raw_config': json.loads((weights / 'config.json').read_text())}
    os.environ.update(VLLM_PLUGINS='qwen3_megakernel', VLLM_USE_V2_MODEL_RUNNER='1')
    argv = [
        sys.executable, '-m', 'vllm.entrypoints.cli.main', 'serve', str(weights),
        '--served-model-name', REPOSITORY,
        '--dtype', 'bfloat16', '--kv-cache-dtype', 'bfloat16', '--block-size', '32',
        '--max-model-len', '33792', '--max-num-batched-tokens', '33792', '--max-num-seqs', '8',
        '--no-enable-prefix-caching', '--no-enable-chunked-prefill',
        '--attention-config', json.dumps({'backend': 'FLASH_ATTN', 'flash_attn_version': 3}),
        '--compilation-config', json.dumps({'cudagraph_mode': 'FULL_DECODE_ONLY',
                                            'cudagraph_capture_sizes': [1, 2, 4, 8]}),
        '--worker-cls', 'vllm_qwen3_megakernel.worker.StandaloneWorker',
        '--scheduler-cls', 'vllm_qwen3_megakernel.scheduler.MegakernelScheduler',
        '--hf-overrides', json.dumps({'architectures': ['StandaloneQwen3ForCausalLM']}),
        '--middleware', 'vllm_qwen3_megakernel.adapter.request_envelope',
        '--additional-config', json.dumps({'megakernel': megakernel}),
        *sys.argv[1:],
    ]
    os.execv(sys.executable, argv)
