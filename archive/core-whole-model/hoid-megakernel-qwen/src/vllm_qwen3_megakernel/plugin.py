"""Installed vLLM general-plugin registration without CUDA initialization."""


def register():
    from vllm import ModelRegistry
    from .adapter import ARCHITECTURE, install_request_guard
    from .model import StandaloneQwen3ForCausalLM
    # The model module is CPU-safe. Register its class directly so vLLM can
    # inspect capabilities without starting a second interpreter.
    ModelRegistry.register_model(ARCHITECTURE, StandaloneQwen3ForCausalLM)
    install_request_guard()
