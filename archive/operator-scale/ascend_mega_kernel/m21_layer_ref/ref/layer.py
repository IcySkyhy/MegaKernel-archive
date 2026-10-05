"""One Qwen4Exp decoder layer — the delayed-combine boundary.

Direct transcription of `Qwen4ExpDecoderLayer.forward`
(vllm/models/qwen4_exp/nvidia/model.py:276-331), minus the PLE branch:

    :287-288   assert prev_injection is None if prev_block_output is None
    :289       attn_hc = self.attn_hyper_connection
    :290-306   PLE branch -- NOT IMPLEMENTED (mission boundary; see README)
    :308-314   fuse the pending combine into attn_hc's mix when there is one
    :316-322   attn_out = linear_attn(block_input) | self_attn(block_input, positions)
    :326-329   (hidden, block_input, injection) = mlp_hc.combine_and_mix(hidden, attn_out, injection)
    :330       mlp_out = self.mlp(block_input)
    :331       return hidden, mlp_out, injection

Three cross-layer tensors, not one (docs/14 §4.2):

    hidden_states   [T, 10240] bf16   the 4-stream residual (HC outer, HS inner)
    block_output    [T, 2560]  bf16   the pending sub-layer output (delayed combine)
    injection       [T, 4]     bf16   the pending per-stream gate logits

`layer_type` is taken from the checkpoint's `layer_types`. Note docs/14 §11 item 5:
12 layers are labelled `full_attention` but are really QSA, because the QSA
dispatch test is `layer_type == QSA_LAYER_TYPE or indexer_n_heads is not None`
(nvidia/model.py:216-220). Picking the wrong branch here silently yields dense GQA.
"""

from __future__ import annotations

import torch

from .gdn import GDN
from .hc import GatedResidual
from .moe import MoE
from .qsa import QSA

QSA_LAYER_TYPE = "qwen_sparse_attention"
ATTENTION_LAYER_TYPES = ("full_attention", QSA_LAYER_TYPE)


class DecoderLayer:
    def __init__(self, layer_idx: int, weights, text_config: dict, precision: str = "fp32"):
        self.layer_idx = layer_idx
        self.cfg = text_config
        self.layer_type = text_config["layer_types"][layer_idx]
        self.hidden_size = text_config["hidden_size"]
        self.hc_count = text_config["hc_count"]

        self.attn_hc = GatedResidual(*self._four(weights.hyper_connection("attn_hyper_connection")),
                                     hc_count=self.hc_count)
        self.mlp_hc = GatedResidual(*self._four(weights.hyper_connection("mlp_hyper_connection")),
                                    hc_count=self.hc_count)

        self.attn = None
        self.gdn = None
        if self.layer_type == "linear_attention":
            self.gdn = GDN(weights.gdn(), text_config)
        elif self.layer_type in ATTENTION_LAYER_TYPES:
            # nvidia/model.py:216-220 -- indexer fields decide, not layer_type
            is_qsa = (
                self.layer_type == QSA_LAYER_TYPE
                or text_config.get("indexer_n_heads") is not None
            )
            if not is_qsa:
                raise NotImplementedError(
                    "dense GQA attention is not implemented in this harness; "
                    "this checkpoint routes all full_attention layers to QSA"
                )
            self.attn = QSA(weights.qsa(), text_config)
        else:
            raise ValueError(f"invalid layer_type {self.layer_type!r}")

        self.moe = MoE(weights.moe(), text_config, precision=precision)

    @staticmethod
    def _four(hc):
        return (hc.hc_norm_weight, hc.down_weight, hc.up_weight, hc.inject_weight)

    # ------------------------------------------------------------------ state
    def init_state(self):
        if self.gdn is not None:
            return self.gdn.init_states()
        return None

    @property
    def uses_state(self) -> bool:
        return self.gdn is not None

    # -------------------------------------------------------------------- fwd
    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor | None,
        prev_injection: torch.Tensor | None,
        positions: torch.Tensor,
        state=None,
        taps: dict | None = None,
    ):
        if prev_block_output is None:
            assert prev_injection is None  # :287-288
        taps = {} if taps is None else taps

        # :308-314
        if prev_block_output is not None:
            hidden_states, block_input, injection = self.attn_hc.combine_and_mix(
                hidden_states, prev_block_output, prev_injection
            )
        else:
            hidden_states, block_input, injection = self.attn_hc.mix(hidden_states)
        taps["attn_hc.hidden"] = hidden_states
        taps["attn_hc.block_input"] = block_input
        taps["attn_hc.injection"] = injection

        # :316-322
        if self.gdn is not None:
            conv_state, ssm_state = state
            attn_out, new_conv_state, _, ataps = self.gdn.forward(
                block_input, conv_state, ssm_state
            )
            state = (new_conv_state, ssm_state)
            for k, v in ataps.items():
                taps[k] = v
        else:
            attn_out, ataps = self.attn.forward(block_input, positions)
            for k, v in ataps.items():
                taps[k] = v
        taps["attn.out"] = attn_out

        # :326-329
        hidden_states, block_input2, injection = self.mlp_hc.combine_and_mix(
            hidden_states, attn_out, injection
        )
        taps["mlp_hc.hidden"] = hidden_states
        taps["mlp_hc.block_input"] = block_input2
        taps["mlp_hc.injection"] = injection

        # :330
        mlp_out, mtaps = self.moe.forward(block_input2)
        for k, v in mtaps.items():
            taps[k] = v
        taps["moe.block_input"] = block_input2

        # :331
        return hidden_states, mlp_out, injection, state, taps


def final_mixer(hidden_states, block_output, injection, mixer_weights, hc_count):
    """The global `hyper_connection_mixer` after the last layer.

    nvidia/model.py:437-441 builds it with `use_combine=False`; :577-590 runs
    `combine_and_mix` and returns `multi_hidden` [T,10240] plus `sample_hidden`
    [T,2560] (the gated stream mean, which feeds lm_head directly).
    """
    mixer = GatedResidual(
        mixer_weights.hc_norm_weight,
        mixer_weights.down_weight,
        mixer_weights.up_weight,
        None,
        hc_count=hc_count,
        use_combine=False,
    )
    multi_hidden, sample_hidden, _ = mixer.combine_and_mix(
        hidden_states, block_output, injection
    )
    return mixer, multi_hidden, sample_hidden
