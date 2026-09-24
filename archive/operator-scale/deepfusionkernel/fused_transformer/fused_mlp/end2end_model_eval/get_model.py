from os import PathLike
from transformers import PreTrainedModel, AutoConfig, AutoModelForCausalLM

from .. import NAME
from .triton_model import MyTritonLlamaForCausalLM, MyTritonLlamaMLP
from .triton_config import MyTritonLlamaConfig
from .llama_model import (
    LlamaForCausalLM as ProfiledLlamaForCausalLM,
    LlamaConfig,
    LlamaMLP as ProfiledLlamaMLP,
)


VERSIONS = NAME


def get_model(
    version: str,
    name: str = "meta-llama/Llama-3.2-1B",
    task: str = "clm",
    attn_impl="flash_attention_2",
    pretrained: bool = True,
    checkpoint: str | PathLike = None,
    **kwargs,
) -> PreTrainedModel:
    """
    Args:
        version (`str`):
            Either `"torch"` or one of `"triton-[tiling_strategy]"`
        name (`str`):
            Model path on Huggingface
        task (`str`):
            Task name. Currently only support causal language modeling
        attn_impl (`str`):
            Attention implementation. Can be either:
                - `"eager"`: manual implementation of the attention
                - `"sdpa"`: using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), may invoke PyTorch's C++ impl, FlashAttention-2, or Memory-Efficient Attention (xformer)
                - `"flash_attention_2"`: using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention))
        pretrained (`bool`):
            Whether to load the pre-trained model weights from Huggingface
        checkpoint (`str` or `os.PathLike`):
            Path to the checkpoint of weights to be loaded
    """
    if version == "torch":
        model = _get_torch_model(name, task, attn_impl, pretrained, checkpoint)
    elif version == "torch-profiled":
        model = _get_torch_profiled_model(name, task, attn_impl, pretrained, checkpoint)
    elif version.startswith("triton-"):
        tiling_strategy = version.removeprefix("triton-")
        assert (
            tiling_strategy in VERSIONS
        ), f"Invalid triton kernel tiling strategy {tiling_strategy}"
        model = _get_triton_model(
            tiling_strategy, name, task, attn_impl, pretrained, checkpoint
        )
    elif version.startswith("single_layer-"):
        version = version.removeprefix("single_layer-")
        model = _get_single_layer(version, name, **kwargs)
    else:
        raise ValueError(f"Invalid model version {version}")

    return model


def _get_single_layer(
    version: str,
    name: str = "meta-llama/Llama-3.2-1B",
    hidden_size: int = None,
    intermediate_size: int = None,
) -> PreTrainedModel:
    """
    Assumed not pretrained / no checkpoint
    """
    if version == "torch":
        config = LlamaConfig.from_pretrained(name)
        if hidden_size is not None:
            config.hidden_size = hidden_size
        if intermediate_size is not None:
            config.intermediate_size = intermediate_size
        model = ProfiledLlamaMLP(config)
    elif version.startswith("triton-"):
        tiling_strategy = version.removeprefix("triton-")
        assert (
            tiling_strategy in VERSIONS
        ), f"Invalid triton kernel tiling strategy {tiling_strategy}"
        config = MyTritonLlamaConfig.from_pretrained(
            name,
            triton_tiling_strategy=tiling_strategy,
        )
        if hidden_size is not None:
            config.hidden_size = hidden_size
        if intermediate_size is not None:
            config.intermediate_size = intermediate_size
        model = MyTritonLlamaMLP(config)
    else:
        raise ValueError(f"Invalid single layer version {version}")

    return model


def _get_torch_model(
    name: str = "meta-llama/Llama-3.2-1B",
    task: str = "clm",
    attn_impl="flash_attention_2",
    pretrained: bool = True,
    checkpoint: str | PathLike = None,
) -> PreTrainedModel:
    match task:
        case "clm" | "causal_language_modeling":
            if pretrained:
                model = AutoModelForCausalLM.from_pretrained(
                    name if checkpoint is None else checkpoint,
                    attn_implementation=attn_impl,
                )
            else:
                config = AutoConfig.from_pretrained(
                    name if checkpoint is None else checkpoint
                )
                model = AutoModelForCausalLM.from_config(
                    config, attn_implementation=attn_impl
                )
        case _:
            raise ValueError(f"Task {task} is not supported for {name}")

    return model


def _get_torch_profiled_model(
    name: str = "meta-llama/Llama-3.2-1B",
    task: str = "clm",
    attn_impl="flash_attention_2",
    pretrained: bool = True,
    checkpoint: str | PathLike = None,
) -> PreTrainedModel:
    match task:
        case "clm" | "causal_language_modeling":
            config = LlamaConfig.from_pretrained(
                name if checkpoint is None else checkpoint,
            )
        case _:
            raise ValueError(f"Task {task} is not supported for {name}")
    if pretrained:
        model = ProfiledLlamaForCausalLM.from_pretrained(
            name if checkpoint is None else checkpoint,
            config=config,
            attn_implementation=attn_impl,
        )
    else:
        model = ProfiledLlamaForCausalLM(config, attn_implementation=attn_impl)

    return model


def _get_triton_model(
    tiling_strategy: str,
    name: str = "meta-llama/Llama-3.2-1B",
    task: str = "clm",
    attn_impl="flash_attention_2",
    pretrained: bool = True,
    checkpoint: str | PathLike = None,
) -> PreTrainedModel:
    match task:
        case "clm" | "causal_language_modeling":
            config = MyTritonLlamaConfig.from_pretrained(
                name if checkpoint is None else checkpoint,
                triton_tiling_strategy=tiling_strategy,
            )
        case _:
            raise ValueError(f"Task {task} is not supported for {name}")
    if pretrained:
        model = MyTritonLlamaForCausalLM.from_pretrained(
            name if checkpoint is None else checkpoint,
            config=config,
            attn_implementation=attn_impl,
        )
    else:
        model = MyTritonLlamaForCausalLM(config, attn_implementation=attn_impl)

    return model
