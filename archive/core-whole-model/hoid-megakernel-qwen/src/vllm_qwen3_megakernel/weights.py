"""The one checkpoint the megakernel is built for, fetched at its pinned revision."""
import hashlib
from pathlib import Path

from .config import SUPPORTED_MODEL

REPOSITORY = 'Qwen/Qwen3-4B-Instruct-2507'


def checkpoint():
    """Local path of the pinned checkpoint, downloading it into the Hugging Face cache once."""
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(REPOSITORY, revision=SUPPORTED_MODEL.revision))
    if hashlib.sha256((path / 'config.json').read_bytes()).hexdigest() != SUPPORTED_MODEL.config_sha256:
        raise RuntimeError(f'{path}/config.json does not match revision {SUPPORTED_MODEL.revision}')
    return path
