import os
import datetime
import logging
from typing import Any

import wandb

from .constants import EXPERIMENT_DIR


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(name="fused_transformer")
info = _logger.info
warning = _logger.warning
error = _logger.error


class WandbLogger:
    def __init__(
        self,
        name: str,
        group: str = None,
        job_type: str = None,
        config: dict[str, Any] | None = None,
        project: str = "fused_transformer",
        **kwargs,
    ):
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self._name = f"{name}-{now}"
        self.run = wandb.init(
            project=project,
            name=self._name,
            group=group,
            job_type=job_type,
            config=config,
            dir=os.path.join(EXPERIMENT_DIR, "wandb"),
            **kwargs,
        )
        self.artifact_dir = os.path.join(EXPERIMENT_DIR, self._name)
        if not os.path.exists(self.artifact_dir):
            os.makedirs(self.artifact_dir, exist_ok=True)
        self._step = 0

    def log(self, data: dict[str, Any]):
        wandb.log(data, step=self._step, commit=False)

    def commit(self):
        wandb.log({}, step=self._step, commit=True)
        self._step += 1

    def close(self):
        self.run.finish()
