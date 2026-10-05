"""Backend-neutral contracts for speculative decoding prototypes.

This module deliberately imports neither MLX nor a model implementation.  It
defines the control-plane pieces needed to qualify two mechanisms before they
are allowed into a serving path:

* amendment of an MTP-cycle QSA selection when compressed groups become
  complete after the selection was captured; and
* phase-separated compiled-variant arenas, so output-producing MTP draft calls
  cannot replace no-output catch-up (or target-verify) variants with identical
  tensor signatures.

Keeping these contracts pure Python makes the CPU qualification meaningful and
gives ``mlx2.ExecutionAdapter`` a small, model-neutral seam to adopt later.
It does *not* qualify an MLX graph, a model, or a serving route by itself.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Generic, Hashable, Mapping, TypeVar

__all__ = [
    "MTPPhase",
    "PhaseSeparatedVariantRegistry",
    "PhaseVariantKey",
    "QSAGroupAmendment",
    "VariantResolution",
    "plan_qsa_group_amendment",
]


@dataclass(frozen=True)
class QSAGroupAmendment:
    """Host-side plan for groups completed since a QSA selection capture."""

    captured_blocks: int
    current_blocks: int
    append_block_ids: tuple[int, ...]

    @property
    def changed(self) -> bool:
        return bool(self.append_block_ids)


def _nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def plan_qsa_group_amendment(
    captured_blocks: int, current_blocks: int
) -> QSAGroupAmendment:
    """Name every compressed group closed since capture, in grid order.

    A smaller current grid means the cycle-local selection survived a rewind.
    That is a lifecycle violation, not an empty amendment, so it fails closed.
    The caller remains responsible for representing per-row/ragged selections;
    this planner intentionally makes no tensor-layout assumptions.
    """

    captured = _nonnegative_int("captured_blocks", captured_blocks)
    current = _nonnegative_int("current_blocks", current_blocks)
    if current < captured:
        raise RuntimeError("QSA shared selection outlived a block-grid rewind")
    return QSAGroupAmendment(captured, current, tuple(range(captured, current)))


class MTPPhase(str, Enum):
    """Semantically distinct graph families in a speculative transaction."""

    DRAFT_OUTPUT = "draft_output"
    CATCHUP_NO_OUTPUT = "catchup_no_output"
    TARGET_VERIFY = "target_verify"


@dataclass(frozen=True)
class PhaseVariantKey:
    """Stable identity for one compiled speculative graph variant.

    ``phase`` is mandatory even when every tensor shape matches.  Runtime and
    state fingerprints prevent a compiled graph from surviving a layout or
    transaction-schema change.  ``config_epoch`` gives soft-reload callers an
    explicit invalidation dimension without inspecting backend objects.
    """

    phase: MTPPhase
    model_identity: str
    input_shape: tuple[int, ...]
    dtype: str
    cache_fingerprint: Hashable
    state_fingerprint: Hashable
    config_epoch: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.phase, MTPPhase):
            raise TypeError("phase must be an MTPPhase")
        if not self.model_identity or not self.dtype:
            raise ValueError("model_identity and dtype are required")
        if not self.input_shape or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in self.input_shape
        ):
            raise ValueError("input_shape must contain positive integers")
        if (
            isinstance(self.config_epoch, bool)
            or not isinstance(self.config_epoch, int)
            or self.config_epoch < 0
        ):
            raise ValueError("config_epoch must be a non-negative integer")
        hash(self.cache_fingerprint)
        hash(self.state_fingerprint)


Variant = TypeVar("Variant")


@dataclass(frozen=True)
class VariantResolution(Generic[Variant]):
    value: Variant
    built: bool


class PhaseSeparatedVariantRegistry(Generic[Variant]):
    """Bounded LRU arenas isolated by speculative phase.

    This is intentionally a registry rather than an ``mx.compile`` wrapper.
    A backend supplies ``builder(key)`` and owns execution/state threading.
    Alternating phases with identical tensor signatures therefore retain one
    live variant each instead of replacing a single shared graph-result slot.

    Counters are always-on Python integers: no timers, array reads, or device
    synchronization occur here.  ``snapshot`` is the mechanism receipt a
    later qualification harness can require before accepting throughput data.
    """

    SCHEMA = "mlx-lm.speculative-phase-variants/v1"

    def __init__(self, max_variants_per_phase: int = 8) -> None:
        if (
            isinstance(max_variants_per_phase, bool)
            or not isinstance(max_variants_per_phase, int)
            or max_variants_per_phase < 1
        ):
            raise ValueError("max_variants_per_phase must be a positive integer")
        self.max_variants_per_phase = max_variants_per_phase
        self._arenas: dict[MTPPhase, OrderedDict[PhaseVariantKey, Variant]] = {
            phase: OrderedDict() for phase in MTPPhase
        }
        self._builds: Counter[MTPPhase] = Counter()
        self._replays: Counter[MTPPhase] = Counter()
        self._evictions: Counter[MTPPhase] = Counter()
        self._build_failures: Counter[MTPPhase] = Counter()
        self._fallbacks: Counter[str] = Counter()

    def resolve(
        self,
        key: PhaseVariantKey,
        builder: Callable[[PhaseVariantKey], Variant],
    ) -> VariantResolution[Variant]:
        if not isinstance(key, PhaseVariantKey):
            raise TypeError("key must be a PhaseVariantKey")
        arena = self._arenas[key.phase]
        if key in arena:
            value = arena.pop(key)
            arena[key] = value
            self._replays[key.phase] += 1
            return VariantResolution(value=value, built=False)
        try:
            value = builder(key)
        except Exception:
            self._build_failures[key.phase] += 1
            raise
        if len(arena) >= self.max_variants_per_phase:
            arena.popitem(last=False)
            self._evictions[key.phase] += 1
        arena[key] = value
        self._builds[key.phase] += 1
        return VariantResolution(value=value, built=True)

    def note_fallback(self, phase: MTPPhase, reason: str) -> None:
        if not isinstance(phase, MTPPhase):
            raise TypeError("phase must be an MTPPhase")
        reason = str(reason).strip()
        if not reason:
            raise ValueError("fallback reason is required")
        self._fallbacks[f"{phase.value}:{reason}"] += 1

    def clear(self) -> None:
        for arena in self._arenas.values():
            arena.clear()

    def snapshot(self) -> Mapping[str, object]:
        phases = {}
        for phase in MTPPhase:
            phases[phase.value] = {
                "live_variants": len(self._arenas[phase]),
                "builds": self._builds[phase],
                "replays": self._replays[phase],
                "evictions": self._evictions[phase],
                "build_failures": self._build_failures[phase],
            }
        return {
            "schema": self.SCHEMA,
            "max_variants_per_phase": self.max_variants_per_phase,
            "phases": phases,
            "fallbacks": dict(sorted(self._fallbacks.items())),
        }
