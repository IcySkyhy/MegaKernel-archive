# Qwen3.8 chat

This local interface loads the published Qwen3.8-27B checkpoint once, formats text with its
checked-in tokenizer, and streams every greedy token from the live Lean CUDA megakernel into the
browser. The bridge binds to all interfaces by default and has no external frontend dependencies.
The browser enables the checkpoint's published thinking mode and keeps its generated
`<think>...</think>` trace in an expandable section separate from the final answer. Reasoning is
not preserved in prior assistant turns. The command-line client prints the same two channels
separately.

From the repository root:

```bash
export QWEN_MODEL_DIR=/path/to/Qwen3.8-27B
./examples/qwen38_chat/run.sh
```

Then open <http://127.0.0.1:8080> locally, or use the configured host and port. Set
`QWEN_MODEL_DIR` to the published checkpoint. Override the listener with `QWEN_CHAT_HOST` or
`QWEN_CHAT_PORT`. Set `QWEN_CHAT_REBUILD=1` to force a worker rebuild.

With the server running, make a one-shot reasoning request from the repository root:

```bash
./examples/qwen38_chat/cli.py --reasoning-effort xhigh \
  "Which is larger, 9.11 or 9.9? Explain."
```

Run `./examples/qwen38_chat/cli.py` without a prompt for an interactive session. Supported
reasoning efforts are `xhigh`, `medium`, and `low`; `--no-thinking` restores answer-only
generation, and `--raw` prints the streaming NDJSON events. The CLI generates until
`<|im_end|>` or the remaining model context is exhausted. Set `QWEN_CHAT_URL` or pass `--url`
when the server is not on `http://127.0.0.1:8080`.

For a real-checkpoint, correctness-gated decode measurement, build the worker and run the
sequential harness:

```bash
worker=$(./examples/qwen38_chat/build_worker.sh)
./examples/qwen38_chat/benchmark_workers.py \
  --model-dir "$QWEN_MODEL_DIR" \
  --llama-bench /tmp/llama-b8892/build/bin/llama-bench \
  --gguf /tmp/Qwen3.8-27B-BF16.gguf \
  --worker "current=$worker" --warmup 1 --repeats 5 --tokens 128 \
  --require-beats-llama \
  --output /tmp/qwen38-worker-benchmark.json
```

Repeat `--worker LABEL=PATH` to compare prebuilt candidates. Each worker exits before the next one
loads. For every Lean repeat, the worker records one whole-run decode sample: elapsed time from the
first token callback through the final callback, divided by the remaining token count. The harness
first generates an untimed autoregressive reference through the published BF16 head, requires every
timed MXFP8 token sequence to match it exactly, and rejects nondeterministic repeats. It takes the
median of the Lean per-run throughputs and the median of llama.cpp's raw `samples_ts`, then writes
the exact commands and per-candidate speedups into one JSON artifact. The Lean measurement includes
greedy sampling and callback publication;
`llama-bench` excludes sampling, so `--require-beats-llama` is a deliberately strict gate.

The production sampler treats the full-vocabulary MXFP8 projection as a nomination pass rather
than the final authority. Each of the 288 resident CTAs nominates its best row, then recomputes that
candidate against the original BF16 head before the exact indexed argmax. This streams only 288
BF16 rows per token while preserving exact BF16 autoregressive token identity.

Inference uses the checkpoint's 262,144-token context and greedy decoding. The serving checkpoint
loads weights and allocates one full-context arena once, then recycles its activations and one
logits/output row across serialized requests. Exact chat-history extensions retain full-attention
KV directly and restore snapshotted DeltaNet recurrent and convolution state; a divergent history
falls back to a clean prefill in the same arena. Prefill runs inside the persistent kernel in
two-row, layer-major tiles: dense Q/K/V, output, and MLP projections apply each weight transaction
to two prompt rows, while causal attention and DeltaNet state updates remain position ordered.
The separate training kernel still retains position-major logits, gradients, losses, and hidden
states with a 256-token training bound. Full attention uses online softmax, so shared memory no
longer grows with context. The server removes the oldest complete turns when necessary, and both
chat clients generate until `<|im_end|>` or the context boundary.

Run the bridge unit tests with:

```bash
cd examples/qwen38_chat
python3 -m unittest -v test_server.py test_cli.py test_benchmark_workers.py
```
