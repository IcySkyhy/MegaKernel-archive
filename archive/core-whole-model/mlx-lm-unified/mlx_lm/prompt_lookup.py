"""Proposal backends for draft-free (prompt-lookup) speculative decoding.

Two interchangeable proposers feed the shared verify/accept/cache core in
``generate.prompt_lookup_generate_step``:

- ``NgramProposer`` — tail n-gram lookup against the running sequence. Simple,
  stateless, zero dependencies. Good default.
- ``SuffixAutomatonProposer`` — an online suffix automaton that returns the
  longest repeated suffix's continuation. Strictly stronger retrieval (finds
  longer/earlier matches than a fixed n-gram) at ~microseconds/token.

Both expose the same interface:
    observe(token: int)                      # feed each committed token
    propose(seq, max_span, prompt_len) -> list[int]

The ``SuffixAutomaton`` and ``HybridStats`` classes here come from the
hybrid-speculative work in this project (SuffixAutomaton retrieval + per-source
accounting); they are reused verbatim so the two efforts converge on one core.
"""
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Sequence, Tuple


class SuffixAutomaton:
    """Online suffix automaton over token ids.

    Built incrementally (``extend`` per committed token), it answers, in
    O(suffix-link chain) per call: what is the longest suffix of the current
    sequence that also occurs ending at an earlier position, and where does that
    earlier occurrence end? ``first_end`` is the end position of the FIRST
    occurrence of a state's substrings, fixed at creation (clones inherit it).
    """

    __slots__ = ("seq", "_len", "_link", "_next", "_first_end", "_last")

    def __init__(self, tokens: Sequence[int] = ()):
        self.seq: List[int] = []
        self._len = [0]
        self._link = [-1]
        self._next: List[dict] = [{}]
        self._first_end = [-1]
        self._last = 0
        for t in tokens:
            self.extend(t)

    def __len__(self) -> int:
        return len(self.seq)

    def extend(self, token: int) -> None:
        token = int(token)
        pos = len(self.seq)
        self.seq.append(token)
        lens, link, nxt, first_end = self._len, self._link, self._next, self._first_end
        cur = len(lens)
        lens.append(lens[self._last] + 1)
        link.append(-1)
        nxt.append({})
        first_end.append(pos)
        p = self._last
        while p != -1 and token not in nxt[p]:
            nxt[p][token] = cur
            p = link[p]
        if p == -1:
            link[cur] = 0
        else:
            q = nxt[p][token]
            if lens[p] + 1 == lens[q]:
                link[cur] = q
            else:
                clone = len(lens)
                lens.append(lens[p] + 1)
                link.append(link[q])
                nxt.append(dict(nxt[q]))
                first_end.append(first_end[q])
                while p != -1 and nxt[p].get(token) == q:
                    nxt[p][token] = clone
                    p = link[p]
                link[q] = clone
                link[cur] = clone
        self._last = cur

    def longest_suffix_match(self, max_len: int = 16) -> Tuple[int, int]:
        """Return (match_len, next_pos): the longest suffix (<= max_len) that
        also occurs ending strictly before the current end, and the index right
        after that earlier occurrence (``seq[next_pos:]`` is the continuation).
        Returns (0, -1) when no suffix repeats."""
        n = len(self.seq)
        if n < 2:
            return 0, -1
        v = self._last
        while v != 0 and self._first_end[v] >= n - 1:
            v = self._link[v]
        if v == 0:
            return 0, -1
        return min(self._len[v], max_len), self._first_end[v] + 1


PLD_CORPUS_MODES = ("target", "uncompacted", "hybrid", "recent_hot")


def _validate_pressure(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def normalize_lookback_ladder(values: Sequence[int]) -> tuple[int, ...]:
    """Return a strictly increasing, positive prompt-lookup ladder."""
    ladder = tuple(int(value) for value in values)
    if not ladder or any(value <= 0 for value in ladder):
        raise ValueError("lookback ladder must contain positive integers")
    if any(right <= left for left, right in zip(ladder, ladder[1:])):
        raise ValueError("lookback ladder must be strictly increasing")
    return ladder


@dataclass(frozen=True)
class PromptLookupSchedulerSignals:
    """Request-boundary signals supplied by the serving scheduler.

    Context is an opportunity signal: a rung is useful only when enough history
    exists to search it. Batch, queue, memory, and latency are pressure signals
    that can reduce the granted rung or decline this single-stream engine.
    """

    context_tokens: int
    batch_size: int = 1
    batch_pressure: float = 0.0
    memory_pressure: float = 0.0
    latency_pressure: float = 0.0

    def __post_init__(self) -> None:
        if self.context_tokens < 0:
            raise ValueError("context_tokens must be non-negative")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        _validate_pressure("batch_pressure", self.batch_pressure)
        _validate_pressure("memory_pressure", self.memory_pressure)
        _validate_pressure("latency_pressure", self.latency_pressure)


@dataclass(frozen=True)
class PromptLookupSchedule:
    enabled: bool
    lookback_cap: int
    reason: str


@dataclass(frozen=True)
class PromptLookupAffordability:
    """Scheduler-owned permission and proposal-width ceiling."""

    enabled: bool
    proposal_cap: int
    reason: str


def schedule_prompt_lookup_affordability(
    proposal_ladder: Sequence[int],
    signals: PromptLookupSchedulerSignals,
    *,
    pressure_threshold: float = 0.70,
    critical_threshold: float = 0.90,
) -> PromptLookupAffordability:
    """Keep system affordability independent from copyability evidence.

    The scheduler may decline PLD or cap its verify width. It does not decide
    whether the current text is copyable; admission and latch controllers own
    that decision inside the granted envelope.
    """
    ladder = normalize_lookback_ladder(proposal_ladder)
    pressure_threshold = _validate_pressure(
        "pressure_threshold", pressure_threshold
    )
    critical_threshold = _validate_pressure(
        "critical_threshold", critical_threshold
    )
    if critical_threshold < pressure_threshold:
        raise ValueError("critical threshold must be >= pressure threshold")
    if signals.batch_size > 1:
        return PromptLookupAffordability(
            False, ladder[0], "batched_engine_unsupported"
        )
    pressure = max(
        signals.batch_pressure,
        signals.memory_pressure,
        signals.latency_pressure,
    )
    if pressure >= critical_threshold:
        return PromptLookupAffordability(True, ladder[0], "critical_pressure")
    if pressure >= pressure_threshold:
        return PromptLookupAffordability(
            True, ladder[min(1, len(ladder) - 1)], "elevated_pressure"
        )
    return PromptLookupAffordability(True, ladder[-1], "available")


class AdaptiveProposalController:
    """Evidence-driven proposal width inside a scheduler-owned cap."""

    def __init__(
        self,
        ladder: Sequence[int],
        *,
        cap: int | None = None,
        accepts_to_widen: int = 3,
        rejects_to_narrow: int = 1,
    ) -> None:
        self.ladder = normalize_lookback_ladder(ladder)
        if accepts_to_widen < 1 or rejects_to_narrow < 1:
            raise ValueError("proposal transition thresholds must be positive")
        self.accepts_to_widen = int(accepts_to_widen)
        self.rejects_to_narrow = int(rejects_to_narrow)
        self._index = 0
        self._accepts = 0
        self._rejects = 0
        self._cap = self.ladder[-1]
        self.widen_events = 0
        self.narrow_events = 0
        self.pressure_clamps = 0
        self.peak = self.ladder[0]
        self.cycles: dict[int, int] = {}
        self.set_cap(self.ladder[-1] if cap is None else cap)

    @property
    def current(self) -> int:
        return self.ladder[self._index]

    @property
    def cap(self) -> int:
        return self._cap

    def set_cap(self, cap: int) -> None:
        allowed = [value for value in self.ladder if value <= int(cap)]
        new_cap = allowed[-1] if allowed else self.ladder[0]
        self._cap = new_cap
        max_index = self.ladder.index(new_cap)
        if self._index > max_index:
            self._index = max_index
            self.pressure_clamps += 1
            self._accepts = 0
            self._rejects = 0

    def begin_cycle(self, cap: int | None = None) -> int:
        if cap is not None:
            self.set_cap(cap)
        width = min(self.current, self._cap)
        self.cycles[width] = self.cycles.get(width, 0) + 1
        self.peak = max(self.peak, width)
        return width

    def observe(self, proposed: int, accepted: int) -> None:
        if proposed <= 0:
            return
        if accepted == proposed:
            self._rejects = 0
            self._accepts += 1
            if self._accepts >= self.accepts_to_widen:
                max_index = self.ladder.index(self._cap)
                if self._index < max_index:
                    self._index += 1
                    self.widen_events += 1
                self._accepts = 0
            return
        self._accepts = 0
        self._rejects += 1
        if self._rejects >= self.rejects_to_narrow:
            if self._index > 0:
                self._index -= 1
                self.narrow_events += 1
            self._rejects = 0


def should_replay_pld_boundary(
    mode: str,
    *,
    margin: float,
    verify_rows: int,
    accepted: int,
    proposed: int,
    margin_threshold: float = -1.0,
    verify_rows_threshold: int = 0,
    replay_on_reject: bool = False,
) -> bool:
    """Return whether a verified boundary needs the strict B=1 oracle."""
    if mode not in ("never", "always", "risk"):
        raise ValueError(f"unknown boundary replay mode {mode!r}")
    if mode == "always":
        return True
    if mode == "never":
        return False
    return bool(
        (margin_threshold >= 0 and margin <= margin_threshold)
        or (verify_rows_threshold > 0 and verify_rows >= verify_rows_threshold)
        or (replay_on_reject and accepted < proposed)
    )


def schedule_prompt_lookup_lookback(
    ladder: Sequence[int],
    signals: PromptLookupSchedulerSignals,
    *,
    pressure_threshold: float = 0.70,
    critical_threshold: float = 0.90,
) -> PromptLookupSchedule:
    """Grant a bounded lookback budget at a scheduler-safe boundary.

    Prompt lookup remains a single-stream engine. A real batch therefore gets
    declined instead of silently migrating an active stream. Otherwise context
    unlocks useful rungs (a window must fit in the existing history), and system
    pressure can remove the wider rungs. The adaptive controller below chooses
    within this ceiling; it never overrides it.
    """
    ladder = normalize_lookback_ladder(ladder)
    pressure_threshold = _validate_pressure(
        "pressure_threshold", pressure_threshold
    )
    critical_threshold = _validate_pressure(
        "critical_threshold", critical_threshold
    )
    if critical_threshold < pressure_threshold:
        raise ValueError("critical threshold must be >= pressure threshold")
    if signals.batch_size > 1:
        return PromptLookupSchedule(False, ladder[0], "batched_engine_unsupported")

    # Do not grant a search window larger than the history it could search.
    context_cap = ladder[0]
    for rung in ladder:
        if rung <= max(signals.context_tokens, ladder[0]):
            context_cap = rung

    pressure = max(
        signals.batch_pressure,
        signals.memory_pressure,
        signals.latency_pressure,
    )
    if pressure >= critical_threshold:
        return PromptLookupSchedule(True, ladder[0], "critical_pressure")
    if pressure >= pressure_threshold:
        return PromptLookupSchedule(
            True,
            min(context_cap, ladder[min(1, len(ladder) - 1)]),
            "elevated_pressure",
        )
    return PromptLookupSchedule(True, context_cap, "context_budget")


class AdaptiveLookbackController:
    """Miss-driven widening with rejection backoff inside a scheduler cap."""

    def __init__(
        self,
        ladder: Sequence[int],
        *,
        cap: int | None = None,
        misses_to_widen: int = 4,
        rejects_to_narrow: int = 2,
        reset_cooldown: int = 16,
    ) -> None:
        self.ladder = normalize_lookback_ladder(ladder)
        if misses_to_widen < 1 or rejects_to_narrow < 1 or reset_cooldown < 0:
            raise ValueError(
                "lookback transition thresholds must be positive and cooldown "
                "must be non-negative"
            )
        self.misses_to_widen = int(misses_to_widen)
        self.rejects_to_narrow = int(rejects_to_narrow)
        self.reset_cooldown = int(reset_cooldown)
        self._index = 0
        self._misses = 0
        self._rejects = 0
        self.widen_events = 0
        self.narrow_events = 0
        self.pressure_clamps = 0
        self.reset_events = 0
        self.peak_lookback = self.ladder[0]
        self.searches: dict[int, int] = {}
        self._cap = self.ladder[-1]
        self._cooldown = 0
        self.set_cap(self.ladder[-1] if cap is None else cap)

    @property
    def current(self) -> int:
        return self.ladder[self._index]

    @property
    def cap(self) -> int:
        return self._cap

    def _max_index(self, cap: int) -> int:
        allowed = [i for i, rung in enumerate(self.ladder) if rung <= cap]
        if not allowed:
            raise ValueError("lookback cap must allow the first ladder rung")
        return allowed[-1]

    def set_cap(self, cap: int) -> None:
        cap = int(cap)
        maximum = self._max_index(cap)
        self._cap = cap
        if self._index > maximum:
            self._index = maximum
            self._misses = 0
            self._rejects = 0
            self.pressure_clamps += 1

    def begin_search(self, cap: int | None = None) -> int:
        if cap is not None:
            self.set_cap(cap)
        lookback = self.current
        self.searches[lookback] = self.searches.get(lookback, 0) + 1
        self.peak_lookback = max(self.peak_lookback, lookback)
        return lookback

    def observe(self, proposed: int, accepted: int) -> None:
        if proposed < 0 or accepted < 0 or accepted > proposed:
            raise ValueError("invalid proposal outcome")
        if proposed == 0:
            self._rejects = 0
            if self._cooldown:
                self._cooldown -= 1
                return
            self._misses += 1
            maximum = self._max_index(self._cap)
            if self._misses >= self.misses_to_widen:
                self._misses = 0
                if self._index < maximum:
                    self._index += 1
                    self.widen_events += 1
                elif self._index > 0:
                    self._index = 0
                    self._cooldown = self.reset_cooldown
                    self.reset_events += 1
            return
        self._misses = 0
        if accepted:
            self._rejects = 0
            return
        self._rejects += 1
        if self._rejects >= self.rejects_to_narrow:
            if self._index > 0:
                self._index -= 1
                self.narrow_events += 1
            self._rejects = 0


class NgramProposer:
    """Tail n-gram lookup. Stateless: proposes the continuation of the rightmost
    earlier occurrence of the tail n-gram (largest n first)."""

    def __init__(
        self,
        ngram_max: int = 3,
        ngram_min: int = 1,
        prompt_only: bool = False,
        max_lookback: int = 4096,
        retrieval_corpus: Sequence[int] | None = None,
        retrieval_segments: Sequence[Sequence[int]] | None = None,
        retrieval_max_lookback: int | None = None,
        corpus_mode: str = "target",
    ):
        if corpus_mode not in PLD_CORPUS_MODES:
            raise ValueError(
                f"unknown prompt-lookup corpus mode {corpus_mode!r}; "
                f"expected one of {PLD_CORPUS_MODES}"
            )
        if retrieval_corpus is not None and retrieval_segments is not None:
            raise ValueError(
                "pass retrieval_corpus or retrieval_segments, not both"
            )
        if corpus_mode != "target" and (
            retrieval_corpus is None and retrieval_segments is None
        ):
            raise ValueError(
                f"prompt-lookup corpus mode {corpus_mode!r} requires an "
                "uncompacted retrieval corpus"
            )
        if max_lookback < 0:
            raise ValueError("max_lookback must be non-negative")
        if retrieval_max_lookback is not None and retrieval_max_lookback < 0:
            raise ValueError("retrieval_max_lookback must be non-negative")
        self.ngram_max = ngram_max
        self.ngram_min = ngram_min
        self.prompt_only = prompt_only
        # Bound the backward scan: on novel text the miss case otherwise
        # walks the FULL sequence per verify cycle for every g — quadratic
        # over a long generation. Proposals are verified anyway, so a bounded
        # window is lossless (worst case: fewer proposals). 0 = unbounded.
        self.max_lookback = max_lookback
        self.retrieval_corpus = (
            [int(token) for token in retrieval_corpus]
            if retrieval_corpus is not None
            else None
        )
        self.retrieval_segments = (
            [[int(token) for token in segment] for segment in retrieval_segments]
            if retrieval_segments is not None
            else None
        )
        if self.retrieval_segments is not None and any(
            not segment for segment in self.retrieval_segments
        ):
            raise ValueError("retrieval segments must be non-empty")
        self.retrieval_max_lookback = retrieval_max_lookback
        self.corpus_mode = corpus_mode

    def observe(self, token: int) -> None:  # stateless
        pass

    def propose(self, seq: List[int], max_span: int, prompt_len: int) -> List[int]:
        n = len(seq)
        retrieval = self.retrieval_segments
        if retrieval is None and self.retrieval_corpus is not None:
            retrieval = [self.retrieval_corpus]
        retrieval = retrieval or []
        target = [(seq, prompt_len if self.prompt_only else None, True)]
        external = [(segment, None, False) for segment in retrieval]
        if self.corpus_mode == "target":
            corpora = target
        elif self.corpus_mode == "uncompacted":
            corpora = external
        elif self.corpus_mode == "recent_hot":
            corpora = target + external
        else:
            corpora = external + target
        for g in range(self.ngram_max, self.ngram_min - 1, -1):
            # An external corpus can supply the earlier occurrence even when
            # the compacted target history contains only the live key itself.
            if n < g:
                continue
            key = seq[-g:]
            for corpus, search_len, is_target in corpora:
                limit = (
                    len(corpus)
                    if search_len is None
                    else min(search_len, len(corpus))
                )
                last = limit - g
                if corpus is seq:
                    last = min(last, n - g - 1)
                lookback = (
                    self.max_lookback
                    if is_target or self.retrieval_max_lookback is None
                    else self.retrieval_max_lookback
                )
                floor = (
                    -1
                    if not lookback
                    else max(-1, last - lookback)
                )
                for i in range(last, floor, -1):
                    if corpus[i : i + g] == key:
                        cont = corpus[i + g : i + g + max_span]
                        if cont:
                            return cont
        return []


class IndexedNgramProposer:
    """Indexed exact n-gram retrieval with context-aware source selection.

    Unlike :class:`NgramProposer`, this backend does not linearly scan the
    lookback window.  It keeps occurrence lists for the live target and static
    proposal-only segments, then disambiguates identical keys using the tokens
    *before* the key.  This matters for common code fragments where a 3-token
    key can have many unrelated continuations.

    ``strategy`` controls the acceptance experiment:

    - ``recent`` reproduces rightmost-match selection with an index.
    - ``context`` prefers the occurrence with the longest exact backward
      context, then recency.
    - ``consensus`` first groups occurrences by proposed continuation, then
      prefers the continuation with the most supporting occurrences and the
      strongest backward context.

    Rejected source choices can be suppressed for ``reject_ttl`` proposal
    calls.  This is proposal policy only: every returned token is still target
    verified by the generator.
    """

    def __init__(
        self,
        ngram_max: int = 6,
        ngram_min: int = 3,
        *,
        max_lookback: int = 0,
        retrieval_segments: Sequence[Sequence[int]] | None = None,
        retrieval_max_lookback: int | None = None,
        strategy: str = "context",
        context_match: int = 64,
        reject_ttl: int = 0,
        prefer_external: bool = False,
    ) -> None:
        if ngram_min < 1 or ngram_max < ngram_min:
            raise ValueError("invalid indexed n-gram bounds")
        if max_lookback < 0 or (
            retrieval_max_lookback is not None and retrieval_max_lookback < 0
        ):
            raise ValueError("lookback bounds must be non-negative")
        if strategy not in ("recent", "context", "consensus"):
            raise ValueError(f"unknown indexed n-gram strategy {strategy!r}")
        if context_match < 0 or reject_ttl < 0:
            raise ValueError("context_match and reject_ttl must be non-negative")
        self.ngram_max = int(ngram_max)
        self.ngram_min = int(ngram_min)
        self.max_lookback = int(max_lookback)
        self.retrieval_max_lookback = retrieval_max_lookback
        self.strategy = strategy
        self.context_match = int(context_match)
        self.reject_ttl = int(reject_ttl)
        self.prefer_external = bool(prefer_external)
        self.seq: list[int] = []
        self._target_index: dict[int, dict[tuple[int, ...], list[int]]] = {
            g: defaultdict(list) for g in range(self.ngram_min, self.ngram_max + 1)
        }
        self._segments = [list(map(int, segment)) for segment in (retrieval_segments or ())]
        if any(not segment for segment in self._segments):
            raise ValueError("retrieval segments must be non-empty")
        self._external_index: dict[int, dict[tuple[int, ...], list[tuple[int, int]]]] = {
            g: defaultdict(list) for g in range(self.ngram_min, self.ngram_max + 1)
        }
        for segment_id, segment in enumerate(self._segments):
            for g in range(self.ngram_min, self.ngram_max + 1):
                for start in range(0, len(segment) - g + 1):
                    self._external_index[g][tuple(segment[start : start + g])].append(
                        (segment_id, start)
                    )
        self._proposal_clock = 0
        self._rejected_until: dict[tuple, int] = {}
        self._last_source: tuple | None = None

    def observe(self, token: int) -> None:
        self.seq.append(int(token))
        end = len(self.seq)
        for g in range(self.ngram_min, self.ngram_max + 1):
            start = end - g
            if start >= 0:
                self._target_index[g][tuple(self.seq[start:end])].append(start)

    @staticmethod
    def _backward_match(
        corpus: Sequence[int], start: int, live: Sequence[int], live_start: int, cap: int
    ) -> int:
        matched = 0
        while matched < cap and start > matched and live_start > matched:
            if corpus[start - matched - 1] != live[live_start - matched - 1]:
                break
            matched += 1
        return matched

    def feedback(self, proposed: int, accepted: int) -> None:
        """Remember a rejected source choice without affecting correctness."""
        if self._last_source is None or proposed <= 0:
            return
        if accepted <= 0 and self.reject_ttl:
            self._rejected_until[self._last_source] = (
                self._proposal_clock + self.reject_ttl
            )
        elif accepted > 0:
            self._rejected_until.pop(self._last_source, None)

    def propose(self, seq: List[int], max_span: int, prompt_len: int) -> List[int]:
        del prompt_len
        if max_span <= 0:
            return []
        # The generator seeds and observes exactly once. Fail loud if a caller
        # violates that contract instead of silently returning misindexed data.
        if len(seq) != len(self.seq) or list(seq[-8:]) != self.seq[-8:]:
            raise ValueError("indexed proposer is out of sync with live sequence")
        self._proposal_clock += 1
        self._last_source = None
        n = len(self.seq)
        candidates = []
        for g in range(min(self.ngram_max, n), self.ngram_min - 1, -1):
            live_start = n - g
            key = tuple(self.seq[live_start:n])
            for start in self._target_index[g].get(key, ()):
                if start >= live_start:
                    continue
                if self.max_lookback and live_start - start > self.max_lookback:
                    continue
                continuation = tuple(self.seq[start + g : start + g + max_span])
                if not continuation:
                    continue
                source = ("target", g, start, continuation)
                if self._rejected_until.get(source, -1) >= self._proposal_clock:
                    continue
                back = self._backward_match(
                    self.seq, start, self.seq, live_start, self.context_match
                )
                candidates.append((g, back, 0, start, continuation, source))
            for segment_id, start in self._external_index[g].get(key, ()):
                segment = self._segments[segment_id]
                if (
                    self.retrieval_max_lookback
                    and len(segment) - start > self.retrieval_max_lookback
                ):
                    continue
                continuation = tuple(segment[start + g : start + g + max_span])
                if not continuation:
                    continue
                source = ("external", segment_id, g, start, continuation)
                if self._rejected_until.get(source, -1) >= self._proposal_clock:
                    continue
                back = self._backward_match(
                    segment, start, self.seq, live_start, self.context_match
                )
                candidates.append((g, back, 1, start, continuation, source))
            # A longer exact key strictly dominates shorter-key ambiguity.
            if candidates:
                break
        if not candidates:
            return []

        external_sign = 1 if self.prefer_external else -1
        if self.strategy == "recent":
            chosen = max(
                candidates,
                key=lambda item: (item[0], external_sign * item[2], item[3]),
            )
        elif self.strategy == "context":
            chosen = max(
                candidates,
                key=lambda item: (
                    item[1], item[0], external_sign * item[2], item[3]
                ),
            )
        else:
            support: dict[tuple[int, ...], int] = defaultdict(int)
            for item in candidates:
                support[item[4]] += 1
            chosen = max(
                candidates,
                key=lambda item: (
                    support[item[4]], item[1], item[0],
                    external_sign * item[2], item[3]
                ),
            )
        self._last_source = chosen[5]
        return list(chosen[4])


class SuffixAutomatonProposer:
    """Suffix-automaton retrieval: continuation of the longest repeated suffix."""

    def __init__(self, min_match: int = 3, max_lookback: int = 32,
                 initial_tokens: Sequence[int] = ()):
        self.min_match = min_match
        self.max_lookback = max_lookback
        self.sam = SuffixAutomaton(initial_tokens)

    def observe(self, token: int) -> None:
        self.sam.extend(token)

    def propose(self, seq: List[int], max_span: int, prompt_len: int) -> List[int]:
        mlen, nxt = self.sam.longest_suffix_match(self.max_lookback)
        if mlen >= self.min_match and 0 <= nxt < len(seq):
            return seq[nxt : nxt + max_span]
        return []


def make_proposer(spec):
    """Build an EMPTY proposer from a `backend` string, or pass through a proposer
    object. The caller seeds it (feeds the prompt via observe()) — do not seed here
    too, or a stateful backend's coordinates desync from the caller's sequence.
    spec: "ngram" | "suffix_automaton" | a proposer instance."""
    if hasattr(spec, "propose"):
        return spec
    if spec in (None, "ngram"):
        return NgramProposer()
    if spec == "suffix_automaton":
        return SuffixAutomatonProposer()
    raise ValueError(f"unknown prompt-lookup backend {spec!r}")


def snap_proposal_around_verify_cliff(
    proposal: Sequence[int], pending_rows: int = 1
) -> List[int]:
    """Avoid the measured M5 verify-batch cliff without inventing tokens.

    Target verification forwards ``pending_rows + len(proposal)`` rows.  Local
    measurements show that rows 9..15 pay the same attention plateau as a much
    longer batch, so a proposal that would land in that band is shortened to
    keep the verify batch at eight rows.  Proposals already large enough to
    reach 16 rows are preserved.  This is deliberately opt-in at the generator
    boundary because the crossover is hardware/model dependent.
    """
    if pending_rows < 1:
        raise ValueError("pending_rows must be >= 1")
    verify_rows = pending_rows + len(proposal)
    if 9 <= verify_rows <= 15:
        return list(proposal[: max(8 - pending_rows, 0)])
    return list(proposal)


def plan_proposal_around_verify_cliff(
    nominal_span: int, available_span: int, pending_rows: int = 1
) -> int:
    """Choose a safe proposal span, preferring the far side of the cliff.

    ``nominal_span`` is the configured span, while ``available_span`` includes
    the continuation and output-budget limits.  If the nominal verify shape
    lands at L=9..15, extend to L=16 when the continuation exists; otherwise
    shrink to L=8.  The opt-in caller owns the decision to exceed its nominal
    span in order to escape the measured plateau.
    """
    if nominal_span < 0 or available_span < 0:
        raise ValueError("proposal spans must be >= 0")
    if pending_rows < 1:
        raise ValueError("pending_rows must be >= 1")
    span = min(nominal_span, available_span)
    verify_rows = pending_rows + span
    if 9 <= verify_rows <= 15:
        long_span = 16 - pending_rows
        if available_span >= long_span:
            return long_span
        return min(span, max(8 - pending_rows, 0))
    return span


@dataclass
class HybridStats:
    """Per-source accounting for one prompt-lookup generation run."""

    cycles: int = 0
    retrieval_cycles: int = 0
    plain_cycles: int = 0
    retrieval_proposed: int = 0
    retrieval_accepted: int = 0
    bonus_tokens: int = 0
    plain_tokens: int = 0
    span_snap_cycles: int = 0
    span_snap_tokens: int = 0
    span_extend_cycles: int = 0
    span_extend_tokens: int = 0
    exact_replay_cycles: int = 0
    exact_replay_tokens: int = 0
    boundary_cycles: int = 0
    boundary_initial_replays: int = 0
    boundary_risk_replays: int = 0
    boundary_audit_replays: int = 0
    boundary_mismatches: int = 0
    boundary_min_margin: float | None = None
    boundary_margin_sum: float = 0.0
    boundary_margin_samples: int = 0
    verify_span_hist: dict[int, int] = field(default_factory=dict)
    latched: bool = False
    # measured-rate gate (rate_gate=True)
    rate_gate_probed: bool = False
    rate_gate_delatched: bool = False
    rate_gate_spec_ms_per_tok: float = 0.0
    rate_gate_plain_ms_per_tok: float = 0.0
    latched_at_token: int | None = None
    latch_reason: str | None = None
    retrieval_fraction_at_latch: float = 0.0
    # Deferred-admission gate: collect copyability evidence under ordinary
    # one-token decode before allowing any speculative verify batch.
    admission_probed: bool = False
    admission_probe_tokens: int = 0
    admission_matches: int = 0
    admission_fraction: float = 0.0
    admission_activated: bool = False
    admission_windows: int = 0
    admission_best_fraction: float = 0.0
    admission_activated_at: int | None = None
    admission_qualifying_windows: int = 0
    admission_consecutive_windows: int = 0
    admission_trial_tokens: int = 0
    admission_trial_passed: bool = False
    adaptive_acceptance_ewma: float = 0.0
    adaptive_acceptance_samples: int = 0
    proposal_ladder: tuple[int, ...] = ()
    proposal_cap: int | None = None
    proposal_current: int | None = None
    proposal_peak: int | None = None
    proposal_widen_events: int = 0
    proposal_narrow_events: int = 0
    proposal_pressure_clamps: int = 0
    proposal_cycles: dict[int, int] = field(default_factory=dict)
    retrieval_corpus_mode: str = "target"
    retrieval_corpus_tokens: int = 0
    # Adaptive recent-history lookup. Counters are intentionally allocation-
    # free in the hot path except for the small bounded per-rung histogram.
    lookback_ladder: tuple[int, ...] = ()
    lookback_cap: int | None = None
    lookback_current: int | None = None
    lookback_peak: int | None = None
    lookback_widen_events: int = 0
    lookback_narrow_events: int = 0
    lookback_pressure_clamps: int = 0
    lookback_reset_events: int = 0
    lookback_searches: dict[int, int] = field(default_factory=dict)

    @property
    def total_emitted(self) -> int:
        return self.retrieval_accepted + self.bonus_tokens + self.plain_tokens

    def summary(self) -> str:
        tot = max(self.total_emitted, 1)
        acc = self.retrieval_accepted / max(self.retrieval_proposed, 1)
        return (
            f"cycles {self.cycles} (retrieval {self.retrieval_cycles}, plain {self.plain_cycles}) | "
            f"tokens {self.total_emitted}: retrieval {self.retrieval_accepted} "
            f"({self.retrieval_accepted / tot:.0%}) + bonus {self.bonus_tokens} + plain {self.plain_tokens} | "
            f"retrieval acceptance {acc:.0%} | latched={self.latched}"
        )


# Canonical/upstream name for the per-run prompt-lookup accounting struct.
# Our tree renamed it to ``HybridStats``; keep the original name as an alias so
# upstream code and tests (e.g. the rate-gate suite) resolve against it.
PromptLookupStats = HybridStats
