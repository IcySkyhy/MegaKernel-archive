import logging
import os

# "Build" logs: schedule construction, workspace allocation, per-operator
# (RMSNorm/Matmul/Attention) kernel construction -- everything that happens
# once per compiled kernel variant.
_BUILD_LOGGERS = (
    "transformer_megakernel.megakernel",
    "transformer_megakernel.scheduler",
    "transformer_megakernel.model",
    "transformer_megakernel.operators.rmsnorm",
    "transformer_megakernel.operators.matmul",
    "transformer_megakernel.operators.attention",
)

# "Autotune" logs: the per-candidate search in autotuner.py.
_AUTOTUNE_LOGGERS = (
    "transformer_megakernel.autotuner",
)


def _level_from_env(var_name: str, default: str) -> int:
    name = os.environ.get(var_name, default).upper()
    return getattr(logging, name, logging.WARNING)


def configure_logging(root_level: int = logging.INFO) -> None:
    """Configures transformer_megakernel logging, gated by env vars:

      TM_LOG_AUTOTUNE - level for autotuner search logs   (default: WARNING, i.e. off)
      TM_LOG_BUILD    - level for build/schedule logs     (default: WARNING, i.e. off)
      TM_LOG_DEBUG    - if set, forces every transformer_megakernel logger to DEBUG,
                        overriding TM_LOG_AUTOTUNE/TM_LOG_BUILD

    Each accepts a standard logging level name (DEBUG/INFO/WARNING/...).
    """
    logging.basicConfig(level=root_level, format="%(levelname)s - %(message)s")

    autotune_level = _level_from_env("TM_LOG_AUTOTUNE", "WARNING")
    build_level = _level_from_env("TM_LOG_BUILD", "WARNING")

    for name in _AUTOTUNE_LOGGERS:
        logging.getLogger(name).setLevel(autotune_level)
    for name in _BUILD_LOGGERS:
        logging.getLogger(name).setLevel(build_level)

    if os.environ.get("TM_LOG_DEBUG"):
        for name in (*_AUTOTUNE_LOGGERS, *_BUILD_LOGGERS):
            logging.getLogger(name).setLevel(logging.DEBUG)
