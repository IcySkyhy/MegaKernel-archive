"""Tests for draft-free prompt-lookup (PLD) speculative decoding.

Covers the proposer backends, the cache-lifecycle helpers, and end-to-end
correctness — including the two cache-reconciliation regressions found in review:
cache REUSE (a non-empty incoming prompt_cache must not be trimmed) and CacheList
models (the reconcile must not assume a flat ``.offset``).
"""
import time
import unittest

import mlx.core as mx
import mlx.nn as nn

# Generation owns a module-level execution stream, so select CPU before
# importing it for this CPU-only correctness suite.
mx.set_default_device(mx.cpu)

from mlx_lm import load
from mlx_lm.generate import (
    _pld_offset,
    _pld_rewind,
    _pld_snapshot,
    _pld_start_speculation,
    generate_step,
    prompt_lookup_generate_step,
)
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache, make_prompt_cache
from mlx_lm.prompt_lookup import (
    AdaptiveLookbackController,
    AdaptiveProposalController,
    IndexedNgramProposer,
    NgramProposer,
    PromptLookupSchedulerSignals,
    SuffixAutomaton,
    SuffixAutomatonProposer,
    make_proposer,
    plan_proposal_around_verify_cliff,
    schedule_prompt_lookup_affordability,
    schedule_prompt_lookup_lookback,
    should_replay_pld_boundary,
    snap_proposal_around_verify_cliff,
)
from mlx_lm.sample_utils import make_sampler

GREEDY = make_sampler(temp=0.0)


class TestProposers(unittest.TestCase):
    def test_adaptive_proposal_width_widens_narrows_and_obeys_cap(self):
        controller = AdaptiveProposalController(
            (2, 4, 8), accepts_to_widen=2, rejects_to_narrow=1
        )
        self.assertEqual(controller.begin_cycle(), 2)
        controller.observe(2, 2)
        controller.observe(2, 2)
        self.assertEqual(controller.begin_cycle(), 4)
        controller.observe(4, 0)
        self.assertEqual(controller.current, 2)
        controller.observe(2, 2)
        controller.observe(2, 2)
        self.assertEqual(controller.begin_cycle(cap=2), 2)
        self.assertEqual(controller.widen_events, 2)
        self.assertEqual(controller.narrow_events, 1)

    def test_scheduler_affordability_is_separate_from_copyability(self):
        ladder = (2, 4, 8)
        busy = schedule_prompt_lookup_affordability(
            ladder,
            PromptLookupSchedulerSignals(
                context_tokens=65536, memory_pressure=0.75
            ),
        )
        self.assertTrue(busy.enabled)
        self.assertEqual(busy.proposal_cap, 4)
        self.assertEqual(busy.reason, "elevated_pressure")
        batched = schedule_prompt_lookup_affordability(
            ladder,
            PromptLookupSchedulerSignals(context_tokens=65536, batch_size=2),
        )
        self.assertFalse(batched.enabled)
        self.assertEqual(batched.reason, "batched_engine_unsupported")

    def test_selective_boundary_replay_classifier(self):
        self.assertTrue(
            should_replay_pld_boundary(
                "risk", margin=0.01, verify_rows=4, accepted=3, proposed=3,
                margin_threshold=0.02,
            )
        )
        self.assertTrue(
            should_replay_pld_boundary(
                "risk", margin=1.0, verify_rows=9, accepted=0, proposed=3,
                verify_rows_threshold=8,
            )
        )
        self.assertFalse(
            should_replay_pld_boundary(
                "risk", margin=1.0, verify_rows=4, accepted=3, proposed=3,
                margin_threshold=0.02, verify_rows_threshold=8,
                replay_on_reject=True,
            )
        )

    def test_indexed_ngram_context_disambiguates_common_key(self):
        # The rightmost ``1, 2`` occurrence has the wrong continuation, while
        # the older occurrence shares a longer backward context with the live
        # suffix and therefore proposes the accepted continuation.
        seq = [8, 9, 1, 2, 7, 6, 0, 1, 2, 4, 3, 8, 9, 1, 2]
        recent = IndexedNgramProposer(2, 2, strategy="recent")
        context = IndexedNgramProposer(2, 2, strategy="context")
        for token in seq:
            recent.observe(token)
            context.observe(token)
        self.assertEqual(recent.propose(seq, 2, 0), [4, 3])
        self.assertEqual(context.propose(seq, 2, 0), [7, 6])

    def test_indexed_ngram_rejection_cache_tries_another_source(self):
        seq = [1, 2, 7, 8, 0, 1, 2, 4, 3, 1, 2]
        proposer = IndexedNgramProposer(
            2, 2, strategy="recent", reject_ttl=4
        )
        for token in seq:
            proposer.observe(token)
        self.assertEqual(proposer.propose(seq, 2, 0), [4, 3])
        proposer.feedback(2, 0)
        self.assertEqual(proposer.propose(seq, 2, 0), [7, 8])

    def test_indexed_ngram_consensus_prefers_supported_continuation(self):
        seq = [1, 2, 7, 8, 0, 1, 2, 7, 8, 0, 1, 2, 4, 3, 1, 2]
        proposer = IndexedNgramProposer(2, 2, strategy="consensus")
        for token in seq:
            proposer.observe(token)
        self.assertEqual(proposer.propose(seq, 2, 0), [7, 8])

    def test_adaptive_lookback_widens_on_misses_and_backs_off_rejections(self):
        controller = AdaptiveLookbackController(
            (256, 1024, 4096), misses_to_widen=2, rejects_to_narrow=2
        )
        self.assertEqual(controller.begin_search(), 256)
        controller.observe(0, 0)
        controller.observe(0, 0)
        self.assertEqual(controller.begin_search(), 1024)
        controller.observe(3, 0)
        controller.observe(3, 0)
        self.assertEqual(controller.current, 256)
        self.assertEqual(controller.widen_events, 1)
        self.assertEqual(controller.narrow_events, 1)

    def test_adaptive_lookback_obeys_a_shrinking_scheduler_cap(self):
        controller = AdaptiveLookbackController(
            (256, 1024, 4096), misses_to_widen=1, reset_cooldown=0
        )
        controller.observe(0, 0)
        controller.observe(0, 0)
        self.assertEqual(controller.current, 4096)
        self.assertEqual(controller.begin_search(cap=1024), 1024)
        self.assertEqual(controller.pressure_clamps, 1)
        for _ in range(4):
            controller.observe(0, 0)
        self.assertEqual(controller.current, 1024)

    def test_adaptive_lookback_periodically_resets_after_ceiling_misses(self):
        controller = AdaptiveLookbackController(
            (256, 1024, 4096), misses_to_widen=2, reset_cooldown=3
        )
        for _ in range(6):
            controller.begin_search()
            controller.observe(0, 0)
        self.assertEqual(controller.current, 256)
        self.assertEqual(controller.reset_events, 1)
        for _ in range(3):
            controller.begin_search()
            controller.observe(0, 0)
        self.assertEqual(controller.current, 256)

    def test_scheduler_context_unlocks_rungs_and_pressure_caps_them(self):
        ladder = (256, 1024, 4096)
        small = schedule_prompt_lookup_lookback(
            ladder, PromptLookupSchedulerSignals(context_tokens=700)
        )
        self.assertEqual((small.enabled, small.lookback_cap), (True, 256))
        roomy = schedule_prompt_lookup_lookback(
            ladder, PromptLookupSchedulerSignals(context_tokens=16_000)
        )
        self.assertEqual(roomy.lookback_cap, 4096)
        pressured = schedule_prompt_lookup_lookback(
            ladder,
            PromptLookupSchedulerSignals(
                context_tokens=16_000, memory_pressure=0.75
            ),
        )
        self.assertEqual(pressured.lookback_cap, 1024)
        critical = schedule_prompt_lookup_lookback(
            ladder,
            PromptLookupSchedulerSignals(
                context_tokens=16_000, latency_pressure=0.95
            ),
        )
        self.assertEqual(critical.lookback_cap, 256)
        batched = schedule_prompt_lookup_lookback(
            ladder,
            PromptLookupSchedulerSignals(context_tokens=16_000, batch_size=2),
        )
        self.assertFalse(batched.enabled)

    def test_ngram_lookback_bound(self):
        # A match beyond max_lookback is not scanned (bounds the per-cycle
        # cost on novel text)...
        seq = [1, 2, 7, 8] + [3] * 64 + [1, 2]
        bounded = NgramProposer(ngram_max=2, ngram_min=2, max_lookback=16)
        self.assertEqual(bounded.propose(seq, 2, 0), [])
        # ...while the default window still finds it, and the same proposer
        # still finds an in-window match.
        full = NgramProposer(ngram_max=2, ngram_min=2)
        self.assertEqual(full.propose(seq, 2, 0), [7, 8])
        near = [9, 9, 1, 2, 7, 8, 1, 2]
        self.assertEqual(bounded.propose(near, 2, 0), [7, 8])

    def test_ngram_finds_earlier_continuation(self):
        # "a b c ... a b" -> proposes the continuation after the earlier "a b".
        seq = [1, 2, 3, 9, 9, 1, 2]
        p = NgramProposer(ngram_max=2, ngram_min=1)
        self.assertEqual(p.propose(seq, max_span=3, prompt_len=len(seq)), [3, 9, 9])

    def test_ngram_no_match(self):
        p = NgramProposer(ngram_max=3, ngram_min=2)
        self.assertEqual(p.propose([1, 2, 3, 4], max_span=4, prompt_len=4), [])

    def test_ngram_can_read_uncompacted_corpus_without_target_visibility(self):
        # The compacted target prompt no longer contains the earlier copy, but
        # its live suffix still keys a continuation in the immutable transcript.
        compacted = [90, 91, 7, 8]
        uncompacted = [1, 2, 7, 8, 30, 31, 32]
        p = NgramProposer(
            ngram_max=2,
            ngram_min=2,
            retrieval_corpus=uncompacted,
            corpus_mode="uncompacted",
        )
        self.assertEqual(p.propose(compacted, 3, len(compacted)), [30, 31, 32])

    def test_ngram_corpus_mode_is_selectable_and_fail_closed(self):
        target = [4, 5, 40, 41, 4, 5]
        unrelated = [7, 8, 9]
        hybrid = NgramProposer(
            ngram_max=2,
            ngram_min=2,
            retrieval_corpus=unrelated,
            corpus_mode="hybrid",
        )
        self.assertEqual(hybrid.propose(target, 2, len(target)), [40, 41])
        with self.assertRaises(ValueError):
            NgramProposer(corpus_mode="uncompacted")
        with self.assertRaises(ValueError):
            NgramProposer(retrieval_corpus=unrelated, corpus_mode="mystery")

    def test_recent_hot_searches_recent_target_before_ranked_segments(self):
        target = [7, 8, 70, 71, 7, 8]
        hot = [[7, 8, 90, 91]]
        proposer = NgramProposer(
            ngram_max=2,
            ngram_min=2,
            max_lookback=16,
            retrieval_segments=hot,
            retrieval_max_lookback=0,
            corpus_mode="recent_hot",
        )
        self.assertEqual(proposer.propose(target, 2, len(target)), [70, 71])

    def test_segmented_retrieval_does_not_match_across_segment_boundary(self):
        target = [5, 6]
        segments = [[1, 5], [6, 9]]
        proposer = NgramProposer(
            ngram_max=2,
            ngram_min=2,
            retrieval_segments=segments,
            retrieval_max_lookback=0,
            corpus_mode="uncompacted",
        )
        self.assertEqual(proposer.propose(target, 2, len(target)), [])

    def test_retrieval_lookback_is_independent_from_target_lookback(self):
        target = [1, 2]
        segment = [1, 2, 7, 8] + [3] * 64
        proposer = NgramProposer(
            ngram_max=2,
            ngram_min=2,
            max_lookback=4,
            retrieval_segments=[segment],
            retrieval_max_lookback=0,
            corpus_mode="uncompacted",
        )
        self.assertEqual(proposer.propose(target, 2, len(target)), [7, 8])

    def test_segmented_retrieval_validation(self):
        with self.assertRaises(ValueError):
            NgramProposer(
                retrieval_corpus=[1],
                retrieval_segments=[[1]],
                corpus_mode="uncompacted",
            )
        with self.assertRaises(ValueError):
            NgramProposer(
                retrieval_segments=[[]],
                corpus_mode="uncompacted",
            )

    def test_suffix_automaton_longest_repeat(self):
        sam = SuffixAutomaton([5, 6, 7, 8, 5, 6])
        mlen, nxt = sam.longest_suffix_match(max_len=16)
        self.assertEqual(mlen, 2)          # "5 6" repeats
        self.assertEqual([5, 6, 7, 8, 5, 6][nxt], 7)  # continuation is "7"

    def test_make_proposer_returns_empty(self):
        # The caller is the single seeding authority; make_proposer must not seed.
        p = make_proposer("suffix_automaton")
        self.assertIsInstance(p, SuffixAutomatonProposer)
        self.assertEqual(len(p.sam), 0)
        with self.assertRaises(ValueError):
            make_proposer("nope")

    def test_verify_cliff_span_snapping(self):
        proposal = list(range(32))
        self.assertEqual(
            snap_proposal_around_verify_cliff(proposal[:7]), proposal[:7]
        )
        self.assertEqual(
            snap_proposal_around_verify_cliff(proposal[:8]), proposal[:7]
        )
        self.assertEqual(
            snap_proposal_around_verify_cliff(proposal[:14]), proposal[:7]
        )
        self.assertEqual(
            snap_proposal_around_verify_cliff(proposal[:15]), proposal[:15]
        )
        # Two pending rows need a six-token proposal to stay at eight total.
        self.assertEqual(
            snap_proposal_around_verify_cliff(proposal[:7], pending_rows=2),
            proposal[:6],
        )
        with self.assertRaises(ValueError):
            snap_proposal_around_verify_cliff(proposal, pending_rows=0)
        self.assertEqual(plan_proposal_around_verify_cliff(8, 20), 15)
        self.assertEqual(plan_proposal_around_verify_cliff(10, 12), 7)
        self.assertEqual(plan_proposal_around_verify_cliff(16, 20), 16)


class TestCacheHelpers(unittest.TestCase):
    def test_pld_offset_flat(self):
        c = KVCache()
        c.offset = 11
        self.assertEqual(_pld_offset(c), 11)

    def test_pld_offset_cachelist(self):
        # CacheList has no .offset of its own; the helper must descend.
        cl = CacheList(KVCache(), KVCache())
        cl.caches[0].offset = 7
        cl.caches[1].offset = 7
        self.assertFalse(hasattr(cl, "offset"))
        self.assertEqual(_pld_offset(cl), 7)

    def test_snapshot_rejects_unsupported_cache(self):
        # ArraysCache cannot be snapshotted until speculation recording is on.
        with self.assertRaises(NotImplementedError):
            _pld_snapshot([ArraysCache(size=2)])

    def test_arrays_cache_snapshot_rewinds_recorded_delta(self):
        c = ArraysCache(size=1)
        c.cache = [mx.array([0])]
        c.start_speculation()
        c.record_rollback(2, lambda m: [mx.array([m])], list(c.cache))
        snap = _pld_snapshot([c])
        c.record_rollback(3, lambda m: [mx.array([2 + m])], [mx.array([2])])
        _pld_rewind([c], snap)
        self.assertEqual(sum(r[0] for r in c._rollbacks), 2)
        self.assertEqual(c.cache[0].tolist(), [2])

    def test_arrays_cache_snapshot_survives_bounded_history_eviction(self):
        c = ArraysCache(size=1)
        c.cache = [mx.array([0])]
        c.start_speculation(rollback_window=4)
        for position in range(1, 9):
            before = position - 1
            c.record_rollback(
                1,
                lambda m, before=before: [mx.array([before + m])],
                [mx.array([before])],
            )
            c.cache = [mx.array([position])]
        self.assertLess(sum(record.span for record in c._rollbacks), 8)

        snap = _pld_snapshot([c])
        c.record_rollback(
            3,
            lambda m: [mx.array([8 + m])],
            [mx.array([8])],
        )
        c.cache = [mx.array([11])]
        _pld_rewind([c], snap)
        self.assertEqual(c.cache[0].tolist(), [8])
        self.assertEqual(c.rollback_marker(), snap[0][1])

    def test_arrays_cache_marker_refuses_future_stale_and_over_rewind(self):
        c = ArraysCache(size=1)
        c.cache = [mx.array([0])]
        c.start_speculation(rollback_window=4)
        start = c.rollback_marker()
        for position in range(1, 7):
            before = position - 1
            c.record_rollback(
                1,
                lambda m, before=before: [mx.array([before + m])],
                [mx.array([before])],
            )
            c.cache = [mx.array([position])]
        with self.assertRaisesRegex(RuntimeError, "ahead of live state"):
            c.rewind_to_rollback_marker((start[0], 7))
        with self.assertRaisesRegex(ValueError, "negative position"):
            c.rewind_to_rollback_marker((start[0], -1))
        with self.assertRaisesRegex(RuntimeError, "only 4 tokens"):
            c.rewind_to_rollback_marker(start)
        c.stop_speculation()
        with self.assertRaisesRegex(RuntimeError, "stale epoch"):
            c.rewind_to_rollback_marker(start)

    def test_arrays_cache_positions_adopt_batch_after_empty_first_record(self):
        c = ArraysCache(size=1)
        c.start_speculation(rollback_window=4)

        # Recurrent layers stage the first rollback before publishing their
        # newly initialized state, so batch_size is still the empty-cache
        # default (one) here even though the forward has four rows.
        c.record_rollback(
            2,
            lambda m: [mx.full((4, 1), m)],
            [None],
        )
        c.cache = [mx.full((4, 1), 2)]

        # The next record sees the materialized batch and carries the first
        # uniform advance into every row instead of treating B=4 as a change.
        before = list(c.cache)
        c.record_rollback(
            1,
            lambda m: [mx.full((4, 1), 2 + m)],
            before,
        )
        c.cache = [mx.full((4, 1), 3)]
        self.assertEqual(c._rollback_positions, [3, 3, 3, 3])

        c.trim_ragged([1, 0, 1, 0])
        self.assertEqual(c._rollback_positions, [2, 3, 2, 3])
        self.assertEqual(c.cache[0].reshape(-1).tolist(), [2, 3, 2, 3])

    def test_arrays_cache_positions_reject_real_unannounced_batch_change(self):
        c = ArraysCache(size=1)
        c.cache = [mx.zeros((2, 1))]
        c.start_speculation(rollback_window=4)
        c.record_rollback(1, lambda m: [mx.full((2, 1), m)], list(c.cache))
        self.assertEqual(c._rollback_positions, [1, 1])

        # Once a live batch has recorded history, changing its membership
        # without filter()/extend() must not silently reinterpret old records.
        c.cache = [mx.zeros((3, 1))]
        with self.assertRaisesRegex(RuntimeError, "do not match the live batch"):
            c.record_rollback(1, lambda m: [mx.full((3, 1), m)], list(c.cache))

    def test_arrays_cache_full_rewind_allows_batched_reinitialization(self):
        c = ArraysCache(size=1)
        c.start_speculation(rollback_window=4)
        c.record_rollback(1, lambda m: [mx.full((2, 1), m)], [None])
        c.cache = [mx.ones((2, 1))]
        c.record_rollback(
            1,
            lambda m: [mx.full((2, 1), 1 + m)],
            list(c.cache),
        )
        c.cache = [mx.full((2, 1), 2)]
        self.assertEqual(c._rollback_positions, [2, 2])

        c.trim(2)
        self.assertIsNone(c.cache[0])
        self.assertIsNone(c._rollback_positions)
        self.assertEqual(c._rollback_position, 0)

        # The same logical batch may now initialize again from empty state.
        c.record_rollback(1, lambda m: [mx.full((2, 1), m)], [None])
        c.cache = [mx.ones((2, 1))]
        c.record_rollback(
            1,
            lambda m: [mx.full((2, 1), 1 + m)],
            list(c.cache),
        )
        self.assertEqual(c._rollback_positions, [2, 2])

    def test_arrays_cache_membership_api_invalidates_empty_history_marker(self):
        c = ArraysCache(size=1)
        c.cache = [mx.zeros((1, 1))]
        c.start_speculation(rollback_window=4)
        marker = c.rollback_marker()
        c.filter([0])
        with self.assertRaisesRegex(RuntimeError, "stale epoch"):
            c.rewind_to_rollback_marker(marker)

    def test_cachelist_snapshot_uses_arrays_epoch_marker(self):
        arrays = ArraysCache(size=1)
        arrays.cache = [mx.array([0])]
        kv = KVCache()
        kv.offset = 0
        cache = CacheList(arrays, kv)
        cache.start_speculation(rollback_window=4)
        snap = _pld_snapshot([cache])
        arrays.record_rollback(2, lambda m: [mx.array([m])], [mx.array([0])])
        arrays.cache = [mx.array([2])]
        kv.offset = 2
        _pld_rewind([cache], snap)
        self.assertEqual(arrays.cache[0].tolist(), [0])
        self.assertEqual(kv.offset, 0)


class _LifecycleCache:
    def __init__(self, fail_start=False):
        self.offset = 0
        self.speculating = False
        self.fail_start = fail_start
        self.start_offsets = []
        self.stop_calls = 0

    @property
    def state(self):
        return []

    def start_speculation(self, rollback_window=None):
        self.start_offsets.append(self.offset)
        self.speculating = True
        if self.fail_start:
            raise RuntimeError("start failed")

    def stop_speculation(self):
        self.stop_calls += 1
        self.speculating = False

    def is_trimmable(self):
        return True

    def trim(self, n):
        self.offset -= n
        return n


class _LifecycleModel:
    def __init__(self, fail_calls=()):
        self.calls = []
        self.input_lengths = []
        self.fail_calls = set(fail_calls)

    def __call__(self, x, cache=None):
        call_number = len(self.calls) + 1
        self.calls.append(cache[0].speculating)
        self.input_lengths.append(x.shape[-1])
        if call_number in self.fail_calls:
            raise RuntimeError(f"model call {call_number} failed")
        cache[0].offset += x.shape[-1]
        return mx.zeros((x.shape[0], x.shape[1], 8))


class TestPromptLookupLifecycle(unittest.TestCase):
    def test_deferred_admission_rejects_before_speculative_forward(self):
        class WrongProposer:
            def observe(self, token):
                pass

            def propose(self, seq, max_span, prompt_len):
                return [7] * max_span

        from mlx_lm.prompt_lookup import HybridStats

        cache = _LifecycleCache()
        model = _LifecycleModel()
        stats = HybridStats()
        out = list(
            prompt_lookup_generate_step(
                mx.array([1]), model, prompt_cache=[cache], max_tokens=12,
                num_draft=3, backend=WrongProposer(),
                deferred_admission=True, admission_warmup=4,
                admission_gate=0.5, stats=stats,
            )
        )

        self.assertEqual([int(token) for token, _, _ in out], [0] * 12)
        self.assertTrue(stats.admission_probed)
        self.assertEqual(stats.admission_probe_tokens, 4)
        self.assertEqual(stats.admission_matches, 0)
        self.assertEqual(stats.admission_fraction, 0.0)
        self.assertFalse(stats.admission_activated)
        self.assertTrue(stats.latched)
        self.assertEqual(stats.latch_reason, "admission_fraction")
        self.assertEqual(stats.latched_at_token, 4)
        self.assertTrue(all(length == 1 for length in model.input_lengths))
        self.assertEqual(cache.start_offsets, [])
        self.assertEqual(cache.stop_calls, 0)
        self.assertEqual(cache.offset, 13)

    def test_deferred_admission_activates_after_copyable_probe(self):
        class CorrectProposer:
            def observe(self, token):
                pass

            def propose(self, seq, max_span, prompt_len):
                return [0] * max_span

        from mlx_lm.prompt_lookup import HybridStats

        cache = _LifecycleCache()
        model = _LifecycleModel()
        stats = HybridStats()
        out = list(
            prompt_lookup_generate_step(
                mx.array([1]), model, prompt_cache=[cache], max_tokens=12,
                num_draft=3, backend=CorrectProposer(),
                deferred_admission=True, admission_warmup=4,
                admission_gate=0.5, stats=stats,
            )
        )

        self.assertEqual([int(token) for token, _, _ in out], [0] * 12)
        self.assertEqual(stats.admission_probe_tokens, 4)
        self.assertEqual(stats.admission_matches, 4)
        self.assertEqual(stats.admission_fraction, 1.0)
        self.assertTrue(stats.admission_activated)
        self.assertFalse(stats.latched)
        self.assertGreater(stats.retrieval_accepted, 0)
        self.assertTrue(any(length > 1 for length in model.input_lengths))
        self.assertEqual(cache.offset, 13)

    def test_logits_processors_receive_exact_history_across_adaptive_tail(self):
        class WrongProposer:
            def observe(self, token):
                pass

            def propose(self, seq, max_span, prompt_len):
                return [7] * max_span

        seen = []

        def history_processor(tokens, logits):
            history = tokens.tolist()
            seen.append(history)
            forced = len(history) % logits.shape[-1]
            bias = mx.zeros_like(logits)
            bias[:, forced] = 100.0
            return logits + bias

        cache = _LifecycleCache()
        prompt = [1, 2, 3]
        out = [
            int(token)
            for token, _, _ in prompt_lookup_generate_step(
                mx.array(prompt),
                _LifecycleModel(),
                prompt_cache=[cache],
                max_tokens=8,
                num_draft=3,
                backend=WrongProposer(),
                logits_processors=[history_processor],
                adaptive=True,
                warmup=2,
                gate=1.0,
            )
        ]

        self.assertEqual(out, [3, 4, 5, 6, 7, 0, 1, 2])
        self.assertEqual(seen[0], prompt)
        # Speculative verification may also process tentative future rows, but
        # every committed decision must see the exact prompt+output prefix.
        for i in range(len(out)):
            self.assertIn(prompt + out[:i], seen)
        self.assertEqual(cache.offset, len(prompt) + len(out))

    def test_early_close_counts_only_yielded_tokens(self):
        class AcceptAllProposer:
            def observe(self, token):
                pass

            def propose(self, seq, max_span, prompt_len):
                return [0] * max_span

        from mlx_lm.prompt_lookup import HybridStats

        cache = _LifecycleCache()
        stats = HybridStats()
        generator = prompt_lookup_generate_step(
            mx.array([1]),
            _LifecycleModel(),
            prompt_cache=[cache],
            max_tokens=16,
            num_draft=8,
            backend=AcceptAllProposer(),
            stats=stats,
        )
        token, _logprobs, from_draft = next(generator)
        self.assertEqual(token, 0)
        self.assertTrue(from_draft)
        generator.close()

        self.assertEqual(stats.retrieval_proposed, 8)
        self.assertEqual(stats.retrieval_accepted, 1)
        self.assertEqual(stats.bonus_tokens, 0)
        self.assertEqual(stats.total_emitted, 1)
        self.assertEqual(cache.offset, 2)  # one prompt + one delivered token

    def test_exact_bonus_replay_uses_single_token_boundary_forwards(self):
        class AcceptAllProposer:
            def observe(self, token):
                pass

            def propose(self, seq, max_span, prompt_len):
                return [0] * max_span

        from mlx_lm.prompt_lookup import HybridStats

        cache = _LifecycleCache()
        model = _LifecycleModel()
        stats = HybridStats()
        out = list(
            prompt_lookup_generate_step(
                mx.array([1]),
                model,
                prompt_cache=[cache],
                max_tokens=4,
                num_draft=2,
                backend=AcceptAllProposer(),
                exact_bonus_replay=True,
                stats=stats,
            )
        )
        self.assertEqual([token for token, _, _ in out], [0, 0, 0, 0])
        self.assertGreater(stats.exact_replay_cycles, 0)
        self.assertGreater(stats.exact_replay_tokens, 0)
        self.assertIn(1, model.input_lengths)
        self.assertEqual(cache.offset, 5)

    def test_risk_selected_boundary_replay_records_diagnostics(self):
        class AcceptAllProposer:
            def observe(self, token):
                pass

            def propose(self, seq, max_span, prompt_len):
                return [0] * max_span

        from mlx_lm.prompt_lookup import HybridStats

        stats = HybridStats()
        list(
            prompt_lookup_generate_step(
                mx.array([1]),
                _LifecycleModel(),
                prompt_cache=[_LifecycleCache()],
                max_tokens=6,
                num_draft=2,
                backend=AcceptAllProposer(),
                exact_replay_mode="risk",
                exact_replay_margin_threshold=0.0,
                exact_replay_audit_interval=2,
                stats=stats,
            )
        )
        self.assertGreater(stats.boundary_cycles, 0)
        self.assertEqual(stats.boundary_margin_samples, stats.boundary_cycles)
        self.assertEqual(stats.boundary_min_margin, 0.0)
        self.assertEqual(stats.boundary_risk_replays, stats.boundary_cycles)
        self.assertEqual(stats.boundary_audit_replays, 0)
        self.assertEqual(stats.exact_replay_cycles, stats.boundary_cycles)

    def test_audit_mismatch_escalates_remaining_boundaries_to_strict_replay(self):
        class AcceptAllProposer:
            def observe(self, token):
                pass

            def propose(self, seq, max_span, prompt_len):
                return [0] * max_span

        class ShapeSensitiveModel(_LifecycleModel):
            def __call__(self, x, cache=None):
                self.calls.append(cache[0].speculating)
                self.input_lengths.append(x.shape[-1])
                cache[0].offset += x.shape[-1]
                logits = mx.zeros((x.shape[0], x.shape[1], 8))
                if x.shape[-1] > 1:
                    logits[:, -1, 1] = 10.0
                return logits

        from mlx_lm.prompt_lookup import HybridStats

        stats = HybridStats()
        out = list(
            prompt_lookup_generate_step(
                mx.array([1]), ShapeSensitiveModel(),
                prompt_cache=[_LifecycleCache()], max_tokens=8,
                num_draft=2, backend=AcceptAllProposer(),
                exact_replay_mode="never", exact_replay_audit_interval=1,
                exact_replay_escalate_on_mismatch=True, stats=stats,
            )
        )
        self.assertEqual([token for token, _, _ in out], [0] * 8)
        self.assertEqual(stats.boundary_audit_replays, 1)
        self.assertGreater(stats.boundary_risk_replays, 0)
        self.assertEqual(stats.boundary_mismatches, stats.boundary_cycles)
        self.assertEqual(stats.exact_replay_cycles, stats.boundary_cycles)

    def test_initial_boundary_replay_protects_short_speculative_trial(self):
        class AcceptAllProposer:
            def observe(self, token):
                pass

            def propose(self, seq, max_span, prompt_len):
                return [0] * max_span

        class ShapeSensitiveModel(_LifecycleModel):
            def __call__(self, x, cache=None):
                self.calls.append(cache[0].speculating)
                self.input_lengths.append(x.shape[-1])
                cache[0].offset += x.shape[-1]
                logits = mx.zeros((x.shape[0], x.shape[1], 8))
                if x.shape[-1] > 1:
                    logits[:, -1, 1] = 10.0
                return logits

        from mlx_lm.prompt_lookup import HybridStats

        stats = HybridStats()
        out = list(
            prompt_lookup_generate_step(
                mx.array([1]), ShapeSensitiveModel(),
                prompt_cache=[_LifecycleCache()], max_tokens=6,
                num_draft=2, backend=AcceptAllProposer(),
                exact_replay_mode="risk", exact_replay_initial_cycles=2,
                stats=stats,
            )
        )
        self.assertEqual([token for token, _, _ in out], [0] * 6)
        self.assertEqual(stats.boundary_cycles, 2)
        self.assertEqual(stats.boundary_initial_replays, 2)
        self.assertEqual(stats.boundary_risk_replays, 0)

    def test_proposal_width_ladder_widens_after_full_acceptance(self):
        class AcceptAllProposer:
            def observe(self, token):
                pass

            def propose(self, seq, max_span, prompt_len):
                return [0] * max_span

        from mlx_lm.prompt_lookup import HybridStats

        model = _LifecycleModel()
        stats = HybridStats()
        list(
            prompt_lookup_generate_step(
                mx.array([1]), model, prompt_cache=[_LifecycleCache()],
                max_tokens=10, backend=AcceptAllProposer(),
                proposal_ladder=(2, 4), proposal_accepts_to_widen=1,
                stats=stats,
            )
        )
        self.assertEqual(stats.proposal_ladder, (2, 4))
        self.assertEqual(stats.proposal_peak, 4)
        self.assertEqual(stats.proposal_widen_events, 1)
        self.assertIn(3, model.input_lengths)
        self.assertIn(5, model.input_lengths)

    def test_admission_trial_latches_false_positive(self):
        class ProbeOnlyProposer:
            def observe(self, token):
                pass

            def propose(self, seq, max_span, prompt_len):
                return [0] if max_span == 1 else [7] * max_span

        from mlx_lm.prompt_lookup import HybridStats

        stats = HybridStats()
        list(
            prompt_lookup_generate_step(
                mx.array([1]), _LifecycleModel(),
                prompt_cache=[_LifecycleCache()], max_tokens=10,
                num_draft=2, backend=ProbeOnlyProposer(),
                deferred_admission=True, admission_warmup=2,
                admission_gate=1.0, admission_trial_tokens=3,
                admission_trial_gate=0.5, stats=stats,
            )
        )
        self.assertTrue(stats.admission_activated)
        self.assertFalse(stats.admission_trial_passed)
        self.assertTrue(stats.latched)
        self.assertEqual(stats.latch_reason, "admission_trial")

    def test_acceptance_ewma_can_deactivate_after_warmup(self):
        class WrongProposer:
            def observe(self, token):
                pass

            def propose(self, seq, max_span, prompt_len):
                return [7] * max_span

        from mlx_lm.prompt_lookup import HybridStats

        stats = HybridStats()
        list(
            prompt_lookup_generate_step(
                mx.array([1]), _LifecycleModel(),
                prompt_cache=[_LifecycleCache()], max_tokens=8,
                num_draft=2, backend=WrongProposer(), adaptive=True,
                warmup=2, adaptive_ewma_alpha=1.0,
                adaptive_deactivate_gate=0.5, stats=stats,
            )
        )
        self.assertTrue(stats.latched)
        self.assertEqual(stats.latch_reason, "acceptance_ewma")
        self.assertEqual(stats.adaptive_acceptance_ewma, 0.0)

    def test_rolling_admission_can_activate_after_an_early_miss(self):
        class LateHitProposer:
            def observe(self, token):
                pass

            def propose(self, seq, max_span, prompt_len):
                token = 1 if len(seq) < 4 else 0
                return [token] * max_span

        from mlx_lm.prompt_lookup import HybridStats

        stats = HybridStats()
        out = list(
            prompt_lookup_generate_step(
                mx.array([1]),
                _LifecycleModel(),
                prompt_cache=[_LifecycleCache()],
                max_tokens=10,
                num_draft=2,
                backend=LateHitProposer(),
                deferred_admission=True,
                admission_warmup=2,
                admission_gate=0.5,
                admission_reprobe_interval=1,
                exact_bonus_replay=True,
                stats=stats,
            )
        )
        self.assertEqual([token for token, _, _ in out], [0] * 10)
        self.assertEqual(stats.admission_windows, 2)
        self.assertTrue(stats.admission_activated)
        self.assertEqual(stats.admission_activated_at, 5)

    def test_rolling_admission_can_require_two_consecutive_windows(self):
        class LateHitProposer:
            def observe(self, token):
                pass

            def propose(self, seq, max_span, prompt_len):
                token = 1 if len(seq) < 4 else 0
                return [token] * max_span

        from mlx_lm.prompt_lookup import HybridStats

        stats = HybridStats()
        list(
            prompt_lookup_generate_step(
                mx.array([1]), _LifecycleModel(),
                prompt_cache=[_LifecycleCache()], max_tokens=12,
                num_draft=2, backend=LateHitProposer(),
                deferred_admission=True, admission_warmup=2,
                admission_gate=0.5, admission_reprobe_interval=1,
                admission_confirm_windows=2, stats=stats,
            )
        )
        self.assertEqual(stats.admission_windows, 3)
        self.assertEqual(stats.admission_qualifying_windows, 2)
        self.assertEqual(stats.admission_consecutive_windows, 2)
        self.assertTrue(stats.admission_activated)
        self.assertEqual(stats.admission_activated_at, 8)

    def test_uncompacted_corpus_is_proposal_only_and_accounted(self):
        from mlx_lm.prompt_lookup import HybridStats

        cache = _LifecycleCache()
        stats = HybridStats()
        out = list(
            prompt_lookup_generate_step(
                mx.array([1]),
                _LifecycleModel(),
                prompt_cache=[cache],
                max_tokens=2,
                num_draft=1,
                ngram_max=1,
                ngram_min=1,
                retrieval_corpus=mx.array([1, 0, 0]),
                retrieval_corpus_mode="uncompacted",
                stats=stats,
            )
        )
        self.assertEqual([token for token, _, _ in out], [0, 0])
        self.assertEqual(cache.offset, 3)
        self.assertEqual(stats.retrieval_corpus_mode, "uncompacted")
        self.assertEqual(stats.retrieval_corpus_tokens, 3)
        self.assertGreater(stats.retrieval_proposed, 0)

    def test_cliff_aware_span_is_opt_in_and_tracks_trimming(self):
        class FixedProposer:
            def observe(self, token):
                pass

            def propose(self, seq, max_span, prompt_len):
                return [0] * max_span

        from mlx_lm.prompt_lookup import HybridStats

        default_cache = _LifecycleCache()
        default_model = _LifecycleModel()
        default_gen = prompt_lookup_generate_step(
            mx.array([1]), default_model, prompt_cache=[default_cache],
            max_tokens=16, num_draft=10, backend=FixedProposer(),
        )
        next(default_gen)
        default_gen.close()
        self.assertEqual(default_model.input_lengths[0], 11)

        stats = HybridStats()
        snapped_cache = _LifecycleCache()
        snapped_model = _LifecycleModel()
        snapped_gen = prompt_lookup_generate_step(
            mx.array([1]), snapped_model, prompt_cache=[snapped_cache],
            max_tokens=16, num_draft=10, backend=FixedProposer(),
            cliff_aware_span=True, stats=stats,
        )
        next(snapped_gen)
        snapped_gen.close()
        self.assertEqual(snapped_model.input_lengths[0], 16)
        self.assertEqual(stats.span_extend_cycles, 1)
        self.assertEqual(stats.span_extend_tokens, 5)
        self.assertEqual(stats.verify_span_hist, {16: 1})

        class ShortProposer(FixedProposer):
            def propose(self, seq, max_span, prompt_len):
                return [0] * min(max_span, 10)

        short_stats = HybridStats()
        short_model = _LifecycleModel()
        short_gen = prompt_lookup_generate_step(
            mx.array([1]), short_model, prompt_cache=[_LifecycleCache()],
            max_tokens=16, num_draft=10, backend=ShortProposer(),
            cliff_aware_span=True, stats=short_stats,
        )
        next(short_gen)
        short_gen.close()
        self.assertEqual(short_model.input_lengths[0], 8)
        self.assertEqual(short_stats.span_snap_cycles, 1)
        self.assertEqual(short_stats.span_snap_tokens, 3)
        self.assertEqual(short_stats.verify_span_hist, {8: 1})

    def test_speculation_starts_after_prompt_prefill(self):
        cache = _LifecycleCache()
        model = _LifecycleModel()
        gen = prompt_lookup_generate_step(
            mx.array([1, 2, 3]), model, prompt_cache=[cache], max_tokens=2
        )
        next(gen)
        gen.close()
        self.assertEqual(model.calls[0], False)
        self.assertTrue(all(model.calls[i] for i in range(1, len(model.calls))))
        self.assertEqual(cache.start_offsets, [2])
        self.assertFalse(cache.speculating)

    def test_validation_and_proposer_errors_do_not_start_speculation(self):
        for prompt, backend in ((mx.array([]), "ngram"), (mx.array([1]), "bad")):
            cache = _LifecycleCache()
            with self.assertRaises(ValueError):
                list(
                    prompt_lookup_generate_step(
                        prompt, _LifecycleModel(), prompt_cache=[cache],
                        backend=backend,
                    )
                )
            self.assertEqual(cache.start_offsets, [])
            self.assertFalse(cache.speculating)

    def test_prefill_error_does_not_start_speculation(self):
        cache = _LifecycleCache()
        with self.assertRaises(RuntimeError):
            list(
                prompt_lookup_generate_step(
                    mx.array([1, 2]), _LifecycleModel(fail_calls={1}),
                    prompt_cache=[cache],
                )
            )
        self.assertEqual(cache.start_offsets, [])
        self.assertFalse(cache.speculating)

    def test_loop_and_reconciliation_errors_stop_speculation(self):
        loop_cache = _LifecycleCache()
        with self.assertRaises(RuntimeError):
            list(
                prompt_lookup_generate_step(
                    mx.array([1]), _LifecycleModel(fail_calls={1, 2}),
                    prompt_cache=[loop_cache],
                )
            )
        self.assertFalse(loop_cache.speculating)
        self.assertGreater(loop_cache.stop_calls, 0)

        reconcile_cache = _LifecycleCache()
        gen = prompt_lookup_generate_step(
            mx.array([1]), _LifecycleModel(fail_calls={2}),
            prompt_cache=[reconcile_cache], max_tokens=2,
        )
        next(gen)
        with self.assertRaises(RuntimeError):
            gen.close()
        self.assertFalse(reconcile_cache.speculating)
        self.assertGreater(reconcile_cache.stop_calls, 0)

    def test_partial_start_failure_stops_all_caches(self):
        caches = [_LifecycleCache(), _LifecycleCache(fail_start=True)]
        with self.assertRaises(RuntimeError):
            _pld_start_speculation(caches, 8)
        self.assertTrue(all(not c.speculating for c in caches))
        self.assertTrue(all(c.stop_calls > 0 for c in caches))


class _TinyCacheListModel(nn.Module):
    """Minimal model whose per-layer cache is ``CacheList(KVCache, KVCache)`` —
    mirroring deepseek_v32 / longcat_flash, which put two KV caches per layer.
    Lets the CacheList path (snapshot/rewind + the finally reconcile) be tested
    without downloading a large real model."""

    def __init__(self, vocab=48, dim=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, vocab, bias=False)

    def make_cache(self):
        return [CacheList(KVCache(), KVCache())]

    def __call__(self, x, cache=None):
        h = self.embed(x)
        if cache is not None:
            B, S, D = h.shape
            kv = self.q(h).reshape(B, 1, S, D)  # [B, heads=1, S, D]
            for sub in cache[0].caches:
                sub.update_and_fetch(kv, kv)     # advance both sub-caches
        return self.out(h)


class _SlowFirstCycleCopyModel(nn.Module):
    """Greedy argmax is ``(token + 1) % period``, so a periodic prompt is
    retrieved with 100 % acceptance; the first forward after prefill sleeps to
    stand in for a prompt-cache restore (or kernel warm-up) — one-time cost
    that must not count as speculation."""

    def __init__(self, period=8, vocab=16, dim=4, delay_s=0.5):
        super().__init__()
        self.period, self.vocab, self.dim, self.delay_s = period, vocab, dim, delay_s
        self.calls = 0
        self.embed = nn.Embedding(vocab, dim)

    def make_cache(self):
        return [KVCache()]

    def __call__(self, x, cache=None):
        self.calls += 1
        if self.calls == 2:  # call 1 is the prefill; call 2 the first cycle
            time.sleep(self.delay_s)
        B, S = x.shape
        if cache is not None:
            kv = mx.zeros((B, 1, S, self.dim))
            cache[0].update_and_fetch(kv, kv)
        nxt = (x + 1) % self.period
        return mx.where(
            mx.arange(self.vocab)[None, None, :] == nxt[..., None], 1.0, 0.0
        )


class TestRateGateWindow(unittest.TestCase):
    def test_first_cycle_cost_does_not_delatch_the_rate_gate(self):
        # A 0.5 s first cycle over ~9 tokens read as ~55 ms/token under the
        # old window (measured 25 ms vs 10 ms plain on the 35B on a cache hit)
        # and de-latched copy-heavy work to the plain tail. The window is armed
        # after the first cycle now.
        from mlx_lm.prompt_lookup import HybridStats

        model = _SlowFirstCycleCopyModel()
        mx.eval(model.parameters())
        period = model.period
        prompt = mx.array([i % period for i in range(40)])
        cache = make_prompt_cache(model)
        stats = HybridStats()
        out = [
            int(t)
            for t, _, _ in prompt_lookup_generate_step(
                prompt, model, max_tokens=64, sampler=GREEDY,
                prompt_cache=cache, backend="ngram", num_draft=8, ngram_max=3,
                adaptive=True, warmup=8, gate=0.12,
                rate_gate=True, rate_gate_probe=4, stats=stats,
            )
        ]
        self.assertEqual(out, [(40 + i) % period for i in range(64)])
        self.assertTrue(stats.rate_gate_probed)
        self.assertFalse(stats.rate_gate_delatched)
        self.assertFalse(stats.latched)
        self.assertIsNone(stats.latched_at_token)
        self.assertIsNone(stats.latch_reason)
        self.assertLess(stats.rate_gate_spec_ms_per_tok, 20.0)
        self.assertGreater(stats.retrieval_accepted, 40)


class TestCacheListModel(unittest.TestCase):
    def test_pld_runs_on_cachelist_model(self):
        # Regression for the finally-reconcile CacheList bug: prompt_cache[0] is a
        # CacheList (no flat .offset). PLD must run and leave the cache exact.
        mx.random.seed(0)
        model = _TinyCacheListModel()
        mx.eval(model.parameters())
        prompt = mx.array([3, 7, 1, 3, 7])  # repeated suffix -> retrieval fires
        cache = make_prompt_cache(model)
        self.assertIsInstance(cache[0], CacheList)
        out = [
            int(t)
            for t, _, _ in prompt_lookup_generate_step(
                prompt, model, max_tokens=24, sampler=GREEDY,
                prompt_cache=cache, backend="suffix_automaton",
            )
        ]
        self.assertEqual(len(out), 24)
        self.assertEqual(_pld_offset(cache[0]), prompt.size + len(out))


class TestPromptLookupGenerate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.tokenizer = load("mlx-community/Qwen1.5-0.5B-Chat-4bit")

    def _prompt(self, text):
        return mx.array(self.tokenizer.encode(text))

    def _run(self, prompt, cache, backend, max_tokens):
        return [
            int(t)
            for t, _, _ in prompt_lookup_generate_step(
                prompt, self.model, max_tokens=max_tokens,
                sampler=GREEDY, prompt_cache=cache, backend=backend,
            )
        ]

    def test_backends_run_and_cache_exact(self):
        prompt = self._prompt("def add(a, b):\n    return a + b\n# repeat: def add")
        for backend in ("ngram", "suffix_automaton"):
            cache = make_prompt_cache(self.model)
            out = self._run(prompt, cache, backend, max_tokens=48)
            self.assertGreater(len(out), 0)
            # Cache is left EXACTLY at prompt + emitted.
            self.assertEqual(_pld_offset(cache[0]), prompt.size + len(out))

    def test_ngram_lookup_composes_with_cached_prefix_history(self):
        full = self._prompt(
            "alpha beta gamma delta alpha beta gamma delta "
            "alpha beta gamma delta continue the sequence"
        )
        split = max(1, full.size // 2)
        prefix, tail = full[:split], full[split:]

        cold_cache = make_prompt_cache(self.model)
        cold = [
            int(token)
            for token, _, _ in prompt_lookup_generate_step(
                full,
                self.model,
                max_tokens=32,
                sampler=GREEDY,
                prompt_cache=cold_cache,
                backend="ngram",
                ngram_max=3,
            )
        ]

        cached = make_prompt_cache(self.model)
        self.model(prefix[None], cache=cached)
        mx.eval([entry.state for entry in cached])
        warm = [
            int(token)
            for token, _, _ in prompt_lookup_generate_step(
                tail,
                self.model,
                max_tokens=32,
                sampler=GREEDY,
                prompt_cache=cached,
                backend="ngram",
                ngram_max=3,
                history_prompt=full,
            )
        ]

        self.assertEqual(warm, cold)
        self.assertEqual(_pld_offset(cached[0]), full.size + len(warm))

    def test_matches_target_batched_greedy(self):
        # PLD output must equal the target's own greedy over prompt+output, i.e.
        # every emitted token is the batched argmax given its prefix. (This is the
        # correct losslessness bar; bit-identity with sequential generate_step is
        # NOT expected for any speculative decoder — batched verify != sequential.)
        prompt = self._prompt("List: apple, banana, apple, banana, apple,")
        cache = make_prompt_cache(self.model)
        out = self._run(prompt, cache, "suffix_automaton", max_tokens=40)
        full = mx.array(prompt.tolist() + out)[None]
        logits = self.model(full)
        L = prompt.size
        for i, tok in enumerate(out):
            self.assertEqual(int(mx.argmax(logits[0, L - 1 + i]).item()), tok)

    def test_max_tokens_boundary_cache_exact(self):
        prompt = self._prompt("Count: 1 2 3 1 2 3 1 2 3")
        for mt in (1, 7, 20):
            cache = make_prompt_cache(self.model)
            out = self._run(prompt, cache, "ngram", max_tokens=mt)
            self.assertEqual(len(out), mt)
            self.assertEqual(_pld_offset(cache[0]), prompt.size + mt)

    def test_cache_reuse_preserves_base(self):
        # Regression: a reused (non-empty) cache's existing prefix must survive
        # the end-of-run reconciliation.
        cache = make_prompt_cache(self.model)
        p1 = self._prompt("Write a haiku about the sea.")
        g1 = self._run(p1, cache, "suffix_automaton", max_tokens=32)
        base = _pld_offset(cache[0])
        self.assertEqual(base, p1.size + len(g1))
        # Second turn reuses the same cache.
        p2 = self._prompt("\nNow one about the mountains.\n")
        g2 = self._run(p2, cache, "suffix_automaton", max_tokens=32)
        self.assertEqual(_pld_offset(cache[0]), base + p2.size + len(g2))

    def test_adaptive_latch_runs(self):
        prompt = self._prompt("Explain why the sky is blue in one paragraph.")
        cache = make_prompt_cache(self.model)
        from mlx_lm.prompt_lookup import HybridStats
        stats = HybridStats()
        out = [
            int(t)
            for t, _, _ in prompt_lookup_generate_step(
                prompt, self.model, max_tokens=80, sampler=GREEDY,
                prompt_cache=cache, backend="suffix_automaton",
                adaptive=True, warmup=16, gate=0.5, stats=stats,
            )
        ]
        self.assertEqual(len(out), 80)
        self.assertEqual(_pld_offset(cache[0]), prompt.size + len(out))
        self.assertTrue(stats.latched)
        self.assertGreaterEqual(stats.latched_at_token, 16)
        self.assertEqual(stats.latch_reason, "retrieval_fraction")
        self.assertLess(stats.retrieval_fraction_at_latch, 0.5)


if __name__ == "__main__":
    unittest.main()
