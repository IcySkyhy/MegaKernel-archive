"""The benchmark workload, shared by both engines and the FP32 reference.

Every request is a deterministic prompt of ordinary-vocabulary token ids (row ``r`` always gets
the same prompt at a given length), decoded greedily for ``GENERATED`` tokens with EOS ignored.
The first generated token comes out of prefill; the remaining ``GENERATED - 1`` come from decode
steps, and only those are timed.
"""
import math
import random

BATCHES = (1, 2, 4, 8)
PROMPT_TOKENS = {'8k': 8191, '32k': 32767}
GENERATED = 1000
GATED = 10  # leading generated tokens that must equal the FP32 reference
MAX_MODEL_LEN = 33792  # the megakernel's KV envelope; both engines run with it
# Qwen3 special tokens start at 151643; prompts draw ordinary vocabulary only.
PROMPT_VOCABULARY = 151643


def prompt(row, length, seed=0):
    """``random.Random(int)`` is stable across Python versions."""
    rng = random.Random(1_000_003 * seed + row)
    return [rng.randrange(PROMPT_VOCABULARY) for _ in range(length)]


# Correctness-gate prompts: a chat turn whose long body pads the context to the benchmark length,
# followed by an ordinary question, so the ten gated tokens are real language the decode computes.
QUESTIONS = (
    'Explain why the sky looks blue in two short sentences.',
    'Name the three primary colors of light and explain what makes them primary.',
    'What is the capital of Australia, and why was it chosen?',
    'Describe how a bicycle gear system works.',
    'Give three tips for learning a new language.',
    'Why do cats purr? Answer briefly.',
    'Summarize the plot of Romeo and Juliet in one paragraph.',
    'How does a refrigerator keep food cold?',
)
LEXICON = ('river', 'garden', 'window', 'music', 'morning', 'silver', 'harbor', 'forest', 'letter',
           'engine', 'candle', 'market', 'winter', 'bridge', 'meadow', 'lantern', 'orchard', 'village',
           'thunder', 'library', 'compass', 'island', 'violet', 'station', 'kitchen', 'mountain', 'paper',
           'journey', 'shadow', 'feather', 'copper', 'valley')


def gate_prompt(tokenizer, row, length):
    """Token ids of exactly ``length``: chat header, deterministic notes, then question ``row``."""
    state, words = 7919 * (row + 1), []
    for _ in range(40000):
        state = (state * 1103515245 + 12345) & 0x7fffffff
        words.append(LEXICON[state % len(LEXICON)])
    head = tokenizer.encode('<|im_start|>user\nHere are some notes:\n')
    tail = tokenizer.encode(f'\n\nIgnore the notes above. {QUESTIONS[row]}<|im_end|>\n<|im_start|>assistant\n')
    body = tokenizer.encode(' '.join(words))[:length - len(head) - len(tail)]
    ids = head + body + tail
    if len(ids) != length:
        raise ValueError(f'gate prompt {row} is {len(ids)} tokens, expected {length}')
    return ids


def percentile(values, percent):
    values = sorted(values)
    x = (len(values) - 1) * percent / 100
    lo = int(x)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (x - lo)


def decode_window(events, rows, steps):
    """Throughput from engine-core output timestamps.

    ``events`` is ``[{'timestamp': float, 'tokens': {row: count}}]`` in delivery order, one per
    engine step. The window opens at the step that delivers the last row's first (prefill) token
    and closes at the step delivering the final token. Every step inside it must carry exactly one
    token for every row, so the window holds ``steps`` full-batch decode steps and nothing else.
    """
    counts = {row: 0 for row in rows}
    opened = None
    for index, event in enumerate(events):
        for row, count in event['tokens'].items():
            counts[row] += count
        if opened is None:
            if any(count > 1 for count in counts.values()):
                raise ValueError('a row decoded before every prefill finished')
            if all(count == 1 for count in counts.values()):
                opened = index
    if opened is None:
        raise ValueError('some row never received its prefill token')
    window = [event for event in events[opened + 1:] if any(event['tokens'].values())]
    if len(window) != steps:
        raise ValueError(f'expected {steps} decode steps, saw {len(window)}')
    if any(event['tokens'] != {row: 1 for row in rows} for event in window):
        raise ValueError('a decode step did not deliver exactly one token per row')
    stamps = [events[opened]['timestamp']] + [event['timestamp'] for event in window]
    return window_stats(stamps, len(rows), steps)


def stream_window(arrivals, steps):
    """Throughput from per-token arrival stamps, for engines without vLLM's step stamps.

    ``arrivals`` maps each row to the arrival time of every generated token, as the engine
    streamed them to this process. The window opens when the last row's first (prefill) token
    arrives and closes when the final token arrives, as in ``decode_window``. No row may receive
    a decode token before every prefill token arrived, and the per-step times are the
    batch-wide arrivals of token k (the latest row's), so they hold ``steps`` full-batch steps.
    """
    rows = list(arrivals)
    if any(len(stamps) != steps + 1 for stamps in arrivals.values()):
        raise ValueError(f'every row must stream {steps + 1} tokens')
    opened = max(stamps[0] for stamps in arrivals.values())
    if any(stamps[1] < opened for stamps in arrivals.values()):
        raise ValueError('a row decoded before every prefill finished')
    stamps = [max(arrivals[row][k] for row in rows) for k in range(steps + 1)]
    return window_stats(stamps, len(rows), steps)


def window_stats(stamps, rows, steps):
    seconds = stamps[-1] - stamps[0]
    if not (seconds > 0 and math.isfinite(seconds)):
        raise ValueError('empty decode window')
    intervals = [(b - a) * 1000 for a, b in zip(stamps, stamps[1:])]
    return dict(decode_steps=steps, decode_tokens=rows * steps, window_seconds=seconds,
                tokens_per_second=rows * steps / seconds,
                tpot_ms_p50=percentile(intervals, 50), tpot_ms_p95=percentile(intervals, 95),
                tpot_ms_mean=seconds * 1000 / steps)
