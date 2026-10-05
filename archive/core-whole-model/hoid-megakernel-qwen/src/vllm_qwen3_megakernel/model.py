"""Stock checkpoint/prefill/head with an independent full-layer decode boundary."""
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM

from .adapter import AdapterConfig, Phase, validate_resolved_model


class StandaloneQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, *, vllm_config, prefix=''):
        self.megakernel_config = AdapterConfig.from_vllm(vllm_config)
        validate_resolved_model(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.megakernel_adapter = None
        self.megakernel_final_norms = {}

    def prepare_final_norm(self, example):
        """``example`` is one bucket's [rows, hidden] final residual."""
        from .stock_norm import prepare_residual_final_norm
        rows = example.shape[0]
        if rows in self.megakernel_final_norms:
            raise RuntimeError(f'final norm already prepared for bucket {rows}')
        self.megakernel_final_norms[rows] = prepare_residual_final_norm(self.model.norm, example)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        adapter = self.megakernel_adapter
        if adapter is None:
            raise RuntimeError('StandaloneWorker must prepare the independent decoder')
        adapter.state.check()
        if adapter.state.phase in (Phase.PREFILL, Phase.PROFILE):
            return super().forward(input_ids, positions, intermediate_tensors, inputs_embeds)
        if adapter.state.phase not in (Phase.DECODE, Phase.CAPTURE):
            adapter.state.fail('model execution has no semantic runner phase')
        if intermediate_tensors is not None or inputs_embeds is not None:
            adapter.state.fail('decode accepts stock token embeddings only')
        output = adapter.decode(self.embed_input_ids(input_ids))
        return self.megakernel_final_norms[output.shape[0]](output)
