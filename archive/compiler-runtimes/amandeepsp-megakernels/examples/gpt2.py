import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from megakernels import MegakernelBackend

MODEL = "openai-community/gpt2"
model = AutoModelForCausalLM.from_pretrained(MODEL).to(device="cuda")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
assert tokenizer is not None

model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to(
    model.device
)

megakernel_backend = MegakernelBackend()

compiled_model = torch.compile(
    model, backend=megakernel_backend, fullgraph=True, dynamic=False
)

with torch.inference_mode():
    for _ in range(3):
        compiled_model(**model_inputs, use_cache=False)
        torch.cuda.synchronize()

with (
    torch.inference_mode(),
    torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof,
):
    outputs = compiled_model(**model_inputs, use_cache=False)
    torch.cuda.synchronize()

# prof.export_chrome_trace("./trace.json")

# print(
#     prof.key_averages().table(
#         sort_by="self_cuda_time_total",
#         row_limit=30,
#     )
# )
