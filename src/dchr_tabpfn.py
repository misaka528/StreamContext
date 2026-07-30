"""DCHR detection and support-state management for a frozen TabPFN stream.

The module is deliberately independent of the GNN experiment.  It implements
the two execution branches described by StreamContext:

* passive recovery: contract the active context after progressive degradation;
* active arbitration: freeze an unreliable support state, collect post-change
  evidence, and use frozen-TabPFN matching loss to reuse a dormant state or
  create a new one.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class DetectorConfig:
    short_labels: int = 8
    reference_labels: int = 32
    abrupt_z: float = 3.0
    gradual_z: float = 2.0
    gradual_patience: int = 3
    min_loss_delta: float = 0.10
    cooldown_samples: int = 512
    warmup_samples: int = 1024


class DualTimescaleChangeDetector:
    """Causal loss detector with a label-free TabPFN warning channel."""

    def __init__(self, config: DetectorConfig):
        self.config = config
        self.losses = deque(maxlen=config.short_labels + config.reference_labels)
        self.proxy_fast: float | None = None
        self.proxy_slow: float | None = None
        self.proxy_var = 1e-4
        self.proxy_updates = 0
        self.loss_gradual_count = 0
        self.proxy_gradual_count = 0
        self.last_event_position = -(10**12)

    def reset_after_event(self, position: int, recent_losses: np.ndarray | None) -> None:
        self.losses.clear()
        if recent_losses is not None and len(recent_losses):
            seeded = np.resize(recent_losses, self.config.reference_labels)
            self.losses.extend(float(value) for value in seeded)
        self.proxy_fast = None
        self.proxy_slow = None
        self.proxy_var = 1e-4
        self.proxy_updates = 0
        self.loss_gradual_count = 0
        self.proxy_gradual_count = 0
        self.last_event_position = int(position)

    def update(
        self,
        position: int,
        label_losses: np.ndarray,
        proxy_value: float,
    ) -> tuple[str | None, dict[str, float]]:
        cfg = self.config
        proxy_value = float(proxy_value)
        self.proxy_updates += 1
        if self.proxy_fast is None:
            self.proxy_fast = proxy_value
            self.proxy_slow = proxy_value
        else:
            residual = proxy_value - float(self.proxy_slow)
            self.proxy_var = 0.97 * self.proxy_var + 0.03 * residual * residual
            self.proxy_fast = 0.70 * self.proxy_fast + 0.30 * proxy_value
            self.proxy_slow = 0.97 * self.proxy_slow + 0.03 * proxy_value
        proxy_z = (float(self.proxy_fast) - float(self.proxy_slow)) / np.sqrt(
            self.proxy_var + 1e-6
        )

        in_cooldown = int(position) - self.last_event_position < cfg.cooldown_samples
        in_warmup = int(position) < cfg.warmup_samples
        loss_z = 0.0
        loss_delta = 0.0
        event_recent: np.ndarray | None = None
        abrupt_from_loss = False

        # Labels are processed sequentially so a fully labeled mini-batch cannot
        # overwrite both timescale windows before the causal comparison is made.
        for value in np.asarray(label_losses).reshape(-1):
            if not np.isfinite(value):
                continue
            self.losses.append(float(np.clip(value, 0.0, 10.0)))
            if len(self.losses) < cfg.reference_labels + cfg.short_labels:
                continue
            values = np.asarray(self.losses, dtype=np.float64)
            reference = values[: cfg.reference_labels]
            recent = values[-cfg.short_labels :]
            current_delta = float(recent.mean() - reference.mean())
            stderr = np.sqrt(
                reference.var(ddof=1) / max(1, len(reference))
                + recent.var(ddof=1) / max(1, len(recent))
                + 1e-4
            )
            current_z = current_delta / max(float(stderr), 0.05)
            if current_z > loss_z:
                loss_z = float(current_z)
                loss_delta = current_delta
                event_recent = recent.copy()
            if (
                not in_cooldown
                and not in_warmup
                and current_z >= cfg.abrupt_z
                and current_delta >= 2.0 * cfg.min_loss_delta
            ):
                abrupt_from_loss = True
                loss_z = float(current_z)
                loss_delta = current_delta
                event_recent = recent.copy()
                break

        diagnostics = {
            "score": float(max(loss_z, proxy_z)),
            "loss_z": float(loss_z),
            "loss_delta": float(loss_delta),
            "proxy_z": float(proxy_z),
        }
        if in_cooldown or in_warmup:
            return None, diagnostics

        proxy_ready = self.proxy_updates >= 20
        if abrupt_from_loss or (proxy_ready and proxy_z >= cfg.abrupt_z + 1.0):
            self.reset_after_event(position, event_recent)
            return "abrupt", diagnostics

        loss_gradual = loss_z >= cfg.gradual_z and loss_delta >= cfg.min_loss_delta
        proxy_gradual = proxy_ready and proxy_z >= cfg.gradual_z
        self.loss_gradual_count = self.loss_gradual_count + 1 if loss_gradual else 0
        self.proxy_gradual_count = self.proxy_gradual_count + 1 if proxy_gradual else 0
        if max(self.loss_gradual_count, self.proxy_gradual_count) >= cfg.gradual_patience:
            self.reset_after_event(position, event_recent)
            return "gradual", diagnostics
        return None, diagnostics


class SupportState:
    """A bounded reusable set of labeled support indices."""

    def __init__(self, state_id: int, capacity: int, indices: np.ndarray | None = None):
        self.state_id = int(state_id)
        self.capacity = int(capacity)
        self.status = "dormant"
        self.indices: deque[int] = deque(maxlen=self.capacity)
        if indices is not None:
            self.extend(indices)

    def extend(self, indices: np.ndarray | list[int]) -> None:
        existing = set(self.indices)
        for index in indices:
            index = int(index)
            if index not in existing:
                self.indices.append(index)
                existing.add(index)

    def discard_from(self, floor: int) -> None:
        retained = [index for index in self.indices if index < int(floor)]
        self.indices = deque(retained, maxlen=self.capacity)


@dataclass(frozen=True)
class StateManagerConfig:
    context_size: int = 1024
    min_context_size: int = 256
    context_recovery_step: int = 64
    confirmation_samples: int = 64
    rollback_samples: int = 128
    reuse_margin: float = 0.05
    reuse_loss_factor: float = 1.25


class TabPFNSupportStateManager:
    """StreamContext-style lifecycle manager for a single frozen TabPFN."""

    def __init__(self, config: StateManagerConfig):
        self.config = config
        initial = SupportState(1, config.context_size)
        initial.status = "active"
        self.states: dict[int, SupportState] = {1: initial}
        self.active_id: int | None = 1
        self.next_id = 2
        self.effective_context_size = config.context_size
        self.pending: dict[str, Any] | None = None

    @property
    def arbitration_pending(self) -> bool:
        return self.pending is not None

    def recover_context_budget(self) -> None:
        if self.pending is not None:
            return
        self.effective_context_size = min(
            self.config.context_size,
            self.effective_context_size + self.config.context_recovery_step,
        )

    def context_indices(self, stream_position: int) -> np.ndarray:
        if self.pending is not None:
            indices = list(self.pending["evidence_indices"])
        elif self.active_id is not None:
            indices = list(self.states[self.active_id].indices)
        else:
            indices = []
        indices = [index for index in indices if index < int(stream_position)]
        return np.asarray(indices[-self.effective_context_size :], dtype=np.int64)

    def handle_detection(self, mode: str, position: int) -> dict[str, Any]:
        if mode == "gradual":
            self.effective_context_size = max(
                self.config.min_context_size,
                self.config.context_size // 4,
            )
            return {
                "action": "passive_context_contraction",
                "from_state": self.active_id,
                "to_state": self.active_id,
                "decision_position": int(position),
            }

        if self.pending is not None:
            raise RuntimeError("Cannot start a second arbitration while one is pending")
        previous_id = int(self.active_id) if self.active_id is not None else None
        if previous_id is not None:
            previous = self.states[previous_id]
            previous.status = "dormant"
            previous.discard_from(int(position) - self.config.rollback_samples)
        self.active_id = None
        self.effective_context_size = self.config.min_context_size
        self.pending = {
            "detection_position": int(position),
            "from_state": previous_id,
            "evidence_indices": deque(maxlen=self.config.confirmation_samples),
        }
        return {
            "action": "collect_post_change_evidence",
            "from_state": previous_id,
            "to_state": None,
            "decision_position": None,
        }

    def commit_labeled(
        self,
        indices: np.ndarray,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
    ) -> None:
        if self.pending is not None:
            self.pending["evidence_indices"].extend(int(index) for index in indices)
        elif self.active_id is not None:
            self.states[self.active_id].extend(indices)

    def observe_prequential(self, label_losses: np.ndarray, event_mode: str | None) -> None:
        del label_losses, event_mode

    def ready_to_arbitrate(self) -> bool:
        return (
            self.pending is not None
            and len(self.pending["evidence_indices"]) >= self.config.confirmation_samples
        )

    @staticmethod
    def _matching_loss(
        extractor: Any,
        x: np.ndarray,
        y: np.ndarray,
        context_indices: np.ndarray,
        validation_indices: np.ndarray,
        n_classes: int,
    ) -> float:
        if len(context_indices) == 0 or len(validation_indices) == 0:
            return float("inf")
        _, _, probabilities = extractor.extract(
            x[context_indices],
            y[context_indices],
            x[validation_indices],
            n_classes,
        )
        probabilities = probabilities.detach().float().cpu().clamp_min(1e-8)
        labels = torch.from_numpy(y[validation_indices]).long()
        return float(-probabilities[torch.arange(len(labels)), labels].log().mean().item())

    def arbitrate(
        self,
        extractor: Any,
        x: np.ndarray,
        y: np.ndarray,
        n_classes: int,
        decision_position: int,
    ) -> dict[str, Any]:
        if self.pending is None:
            raise RuntimeError("No pending support-state arbitration")
        evidence = np.asarray(self.pending["evidence_indices"], dtype=np.int64)
        if len(evidence) < 16:
            raise RuntimeError("At least 16 post-change labels are required")

        split = max(8, len(evidence) // 2)
        split = min(split, len(evidence) - 8)
        recent_support = evidence[:split]
        validation = evidence[split:]
        recent_loss = self._matching_loss(
            extractor,
            x,
            y,
            recent_support,
            validation,
            n_classes,
        )

        candidate_losses: dict[int, float] = {}
        for state_id, state in self.states.items():
            if state.status != "dormant" or not state.indices:
                continue
            context = np.asarray(list(state.indices)[-self.config.context_size :], dtype=np.int64)
            candidate_losses[state_id] = self._matching_loss(
                extractor,
                x,
                y,
                context,
                validation,
                n_classes,
            )
        if candidate_losses:
            best_state = min(candidate_losses, key=candidate_losses.get)
            best_loss = float(candidate_losses[best_state])
        else:
            best_state = None
            best_loss = float("inf")

        absolute_threshold = np.log(max(2, n_classes)) * self.config.reuse_loss_factor
        should_reuse = (
            best_state is not None
            and best_loss <= absolute_threshold
            and best_loss <= recent_loss + self.config.reuse_margin
        )
        if should_reuse:
            chosen_id = int(best_state)
            action = "reuse_state"
        else:
            chosen_id = self.next_id
            self.next_id += 1
            self.states[chosen_id] = SupportState(
                chosen_id,
                self.config.context_size,
                evidence,
            )
            action = "new_state"

        for state in self.states.values():
            state.status = "dormant"
        chosen = self.states[chosen_id]
        chosen.status = "active"
        chosen.extend(evidence)
        previous_id = self.pending["from_state"]
        detection_position = self.pending["detection_position"]
        self.active_id = chosen_id
        self.pending = None
        self.effective_context_size = self.config.context_size
        return {
            "action": action,
            "from_state": previous_id,
            "to_state": chosen_id,
            "detection_position": int(detection_position),
            "decision_position": int(decision_position),
            "recent_loss": float(recent_loss),
            "match_loss": None if best_state is None else best_loss,
            "matched_state": best_state,
            "candidate_losses": {str(key): value for key, value in candidate_losses.items()},
        }

    def state_summary(self) -> list[dict[str, Any]]:
        return [
            {
                "state_id": state_id,
                "status": state.status,
                "support_size": len(state.indices),
            }
            for state_id, state in sorted(self.states.items())
        ]


class SlidingWindowContextManager:
    """Frozen-TabPFN baseline using only the most recent labeled samples."""

    def __init__(self, context_size: int):
        self.context_size = int(context_size)
        if self.context_size <= 0:
            raise ValueError("context_size must be positive")
        self.indices: deque[int] = deque(maxlen=self.context_size)
        self.effective_context_size = self.context_size
        self.active_id = 1
        self.recent_ratio = 1.0
        self.degradation_z = 0.0

    @property
    def arbitration_pending(self) -> bool:
        return False

    def recover_context_budget(self) -> None:
        pass

    def context_indices(self, stream_position: int) -> np.ndarray:
        return np.asarray(
            [index for index in self.indices if index < int(stream_position)],
            dtype=np.int64,
        )

    def observe_prequential(self, label_losses: np.ndarray, event_mode: str | None) -> None:
        del label_losses, event_mode

    def handle_detection(self, mode: str, position: int) -> dict[str, Any]:
        del mode
        return {
            "action": "observe_only_sliding_window",
            "from_state": 1,
            "to_state": 1,
            "decision_position": int(position),
            "router_mode": "top1",
        }

    def commit_labeled(
        self,
        indices: np.ndarray,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
    ) -> None:
        del x, y
        self.indices.extend(int(index) for index in indices)

    def ready_to_arbitrate(self) -> bool:
        return False

    def state_summary(self) -> list[dict[str, Any]]:
        return [
            {
                "state_id": 1,
                "status": "active",
                "support_size": len(self.indices),
                "effective_context_size": self.context_size,
                "recent_ratio": 1.0,
                "window_type": "strict_recent",
            }
        ]


@dataclass(frozen=True)
class PassiveMemoryConfig:
    """Configuration for the complete passive StreamContext branch."""

    context_size: int = 1024
    min_context_size: int = 256
    recent_window_size: int = 256
    stable_recent_ratio: float = 0.25
    max_recent_ratio: float = 0.90
    context_recovery_step: int = 64
    ratio_recovery_step: float = 0.02
    redundancy_radius: float = 0.08
    boundary_replacement_margin: float = 0.02
    max_sample_age: int = 0


@dataclass
class ArchiveItem:
    index: int
    label: int
    vector: np.ndarray
    boundary_distance: float
    inserted_at: int


class OnlineFeatureStandardizer:
    """Causal Welford standardizer used only by support-memory compression."""

    def __init__(self, dimension: int):
        self.count = 0
        self.mean = np.zeros(dimension, dtype=np.float64)
        self.m2 = np.zeros(dimension, dtype=np.float64)

    def update(self, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=np.float64)
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    def transform(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if self.count < 2:
            return value.astype(np.float32)
        variance = self.m2 / max(1, self.count - 1)
        return ((value - self.mean) / np.sqrt(variance + 1e-6)).astype(np.float32)


class PassiveContextManager:
    """Passive StreamContext with bounded recent/archive support evidence.

    The recent window and archive are disjoint.  Samples leaving the recent
    window are promoted into a label-aware compressed archive.  Reliability
    controls both the exposed context budget and the recent/archive allocation.
    No support-state transition is allowed in this class.
    """

    def __init__(self, config: PassiveMemoryConfig, feature_dimension: int):
        self.config = config
        if config.recent_window_size >= config.context_size:
            raise ValueError("recent_window_size must be smaller than context_size")
        self.archive_capacity = config.context_size - config.recent_window_size
        self.recent_indices: deque[int] = deque()
        self.archive: list[ArchiveItem] = []
        self.scaler = OnlineFeatureStandardizer(feature_dimension)
        self.effective_context_size = config.context_size
        self.recent_ratio = config.stable_recent_ratio
        self.target_context_size = config.context_size
        self.target_recent_ratio = config.stable_recent_ratio
        self.active_id = 1
        self.loss_fast: float | None = None
        self.loss_slow: float | None = None
        self.loss_var = 1e-4
        self.degradation_z = 0.0
        self.degradation_streak = 0
        self.insert_counter = 0
        self.compression_stats = {
            "promoted": 0,
            "redundant_discarded": 0,
            "boundary_replaced": 0,
            "capacity_evicted": 0,
            "stale_evicted_archive": 0,
            "stale_evicted_recent": 0,
        }

    @property
    def arbitration_pending(self) -> bool:
        return False

    def _move_controls(self) -> None:
        if self.effective_context_size > self.target_context_size:
            self.effective_context_size = max(
                self.target_context_size,
                self.effective_context_size - max(128, self.config.context_size // 8),
            )
        elif self.effective_context_size < self.target_context_size:
            self.effective_context_size = min(
                self.target_context_size,
                self.effective_context_size + self.config.context_recovery_step,
            )
        if self.recent_ratio < self.target_recent_ratio:
            self.recent_ratio = min(self.target_recent_ratio, self.recent_ratio + 0.10)
        elif self.recent_ratio > self.target_recent_ratio:
            self.recent_ratio = max(
                self.target_recent_ratio,
                self.recent_ratio - self.config.ratio_recovery_step,
            )

    def recover_context_budget(self) -> None:
        self._move_controls()

    @staticmethod
    def _distances(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        if len(matrix) == 0:
            return np.empty(0, dtype=np.float32)
        return np.linalg.norm(matrix - vector[None, :], axis=1) / np.sqrt(matrix.shape[1])

    def _boundary_distance(self, vector: np.ndarray, label: int) -> float:
        different = [item.vector for item in self.archive if item.label != label]
        if not different:
            return float("inf")
        return float(self._distances(vector, np.stack(different)).min())

    def _prune_stale(self, stream_position: int) -> None:
        """Remove support examples older than the configured global stream age."""
        max_age = int(self.config.max_sample_age)
        if max_age <= 0:
            return
        cutoff = int(stream_position) - max_age
        if cutoff <= 0:
            return

        archive_before = len(self.archive)
        self.archive = [item for item in self.archive if item.index >= cutoff]
        self.compression_stats["stale_evicted_archive"] += (
            archive_before - len(self.archive)
        )

        recent_before = len(self.recent_indices)
        self.recent_indices = deque(
            index for index in self.recent_indices if int(index) >= cutoff
        )
        self.compression_stats["stale_evicted_recent"] += (
            recent_before - len(self.recent_indices)
        )

    def _insert_archive(self, index: int, label: int, vector: np.ndarray) -> None:
        self.insert_counter += 1
        same_positions = [i for i, item in enumerate(self.archive) if item.label == label]
        different_vectors = [item.vector for item in self.archive if item.label != label]
        boundary_distance = (
            float(self._distances(vector, np.stack(different_vectors)).min())
            if different_vectors
            else float("inf")
        )
        if same_positions:
            same_matrix = np.stack([self.archive[i].vector for i in same_positions])
            same_distances = self._distances(vector, same_matrix)
            nearest_offset = int(same_distances.argmin())
            nearest_position = same_positions[nearest_offset]
            if float(same_distances[nearest_offset]) <= self.config.redundancy_radius:
                old = self.archive[nearest_position]
                old_boundary = self._boundary_distance(old.vector, old.label)
                if boundary_distance + self.config.boundary_replacement_margin < old_boundary:
                    self.archive[nearest_position] = ArchiveItem(
                        int(index), int(label), vector, boundary_distance, self.insert_counter
                    )
                    self.compression_stats["boundary_replaced"] += 1
                else:
                    self.compression_stats["redundant_discarded"] += 1
                return

        new_item = ArchiveItem(
            int(index), int(label), vector, boundary_distance, self.insert_counter
        )
        if len(self.archive) < self.archive_capacity:
            self.archive.append(new_item)
            self.compression_stats["promoted"] += 1
            return

        counts: dict[int, int] = {}
        for item in self.archive:
            counts[item.label] = counts.get(item.label, 0) + 1
        largest_label = max(counts, key=counts.get)
        candidate_positions = [
            i for i, item in enumerate(self.archive) if item.label == largest_label
        ]


        # Prefer evicting old interior evidence. Infinite distances occur before
        # both classes are observed and are therefore treated as least useful.
        evict_position = max(
            candidate_positions,
            key=lambda i: (
                self.archive[i].boundary_distance,
                -self.archive[i].inserted_at,
            ),
        )
        self.archive[evict_position] = new_item
        self.compression_stats["capacity_evicted"] += 1

    def commit_labeled(
        self,
        indices: np.ndarray,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
    ) -> None:
        if x is None or y is None:
            raise ValueError("PassiveContextManager requires feature and label arrays")
        for index in indices:
            index = int(index)
            self._prune_stale(index)
            if len(self.recent_indices) >= self.config.recent_window_size:
                promoted = self.recent_indices.popleft()
                vector = self.scaler.transform(x[promoted])
                self._insert_archive(promoted, int(y[promoted]), vector)
            self.recent_indices.append(index)
            self.scaler.update(x[index])

    def _select_archive(self, count: int, excluded: set[int]) -> list[int]:
        if count <= 0:
            return []
        by_label: dict[int, list[ArchiveItem]] = {}
        for item in self.archive:
            if item.index in excluded:
                continue
            by_label.setdefault(item.label, []).append(item)
        for items in by_label.values():
            items.sort(key=lambda item: (item.boundary_distance, -item.inserted_at))
        selected: list[int] = []
        labels = sorted(by_label)
        offset = 0
        while len(selected) < count and labels:
            remaining = []
            for label in labels:
                items = by_label[label]
                if offset < len(items):
                    selected.append(items[offset].index)
                    if len(selected) >= count:
                        break
                if offset + 1 < len(items):
                    remaining.append(label)
            labels = remaining
            offset += 1
        return selected

    def context_indices_with_budget(
        self,
        stream_position: int,
        budget_override: int | None = None,
    ) -> np.ndarray:
        self._prune_stale(stream_position)
        budget = min(self.config.context_size, int(self.effective_context_size))
        if budget_override is not None:
            budget = min(budget, max(0, int(budget_override)))
        recent_target = min(
            len(self.recent_indices),
            int(round(budget * self.recent_ratio)),
        )
        recent = list(self.recent_indices)[-recent_target:] if recent_target else []
        archive = self._select_archive(budget - len(recent), set(recent))
        if len(recent) + len(archive) < budget:
            extra_recent = [
                index for index in self.recent_indices if index not in set(recent)
            ]
            recent = extra_recent[-(budget - len(recent) - len(archive)) :] + recent
        indices = sorted(index for index in archive + recent if index < int(stream_position))
        return np.asarray(indices, dtype=np.int64)

    def context_indices(self, stream_position: int) -> np.ndarray:
        return self.context_indices_with_budget(stream_position)

    def observe_prequential(self, label_losses: np.ndarray, event_mode: str | None) -> None:
        values = np.asarray(label_losses, dtype=np.float64)
        values = values[np.isfinite(values)]
        if not len(values):
            return
        current = float(np.clip(values.mean(), 0.0, 10.0))
        if self.loss_fast is None:
            self.loss_fast = current
            self.loss_slow = current
        else:
            residual = current - float(self.loss_slow)
            self.loss_var = 0.97 * self.loss_var + 0.03 * residual * residual
            self.loss_fast = 0.80 * self.loss_fast + 0.20 * current
            self.loss_slow = 0.98 * self.loss_slow + 0.02 * current
        self.degradation_z = max(
            0.0,
            (float(self.loss_fast) - float(self.loss_slow)) / np.sqrt(self.loss_var + 1e-6),
        )
        severity = float(np.clip((self.degradation_z - 0.5) / 2.5, 0.0, 1.0))
        if event_mode == "gradual":
            severity = max(severity, 0.65)
        elif event_mode == "abrupt":
            severity = 1.0
        self.target_context_size = int(
            round(
                self.config.context_size
                - severity * (self.config.context_size - self.config.min_context_size)
            )
        )
        self.target_recent_ratio = float(
            self.config.stable_recent_ratio
            + severity * (self.config.max_recent_ratio - self.config.stable_recent_ratio)
        )
        if severity >= 0.65:
            self.degradation_streak += 1
        elif severity <= 0.20:
            self.degradation_streak = 0

    def handle_detection(self, mode: str, position: int) -> dict[str, Any]:
        severity = 0.70 if mode == "gradual" else 1.0
        self.target_context_size = int(
            round(
                self.config.context_size
                - severity * (self.config.context_size - self.config.min_context_size)
            )
        )
        self.target_recent_ratio = float(
            self.config.stable_recent_ratio
            + severity * (self.config.max_recent_ratio - self.config.stable_recent_ratio)
        )
        self.effective_context_size = min(
            self.effective_context_size,
            self.target_context_size,
        )
        self.recent_ratio = max(self.recent_ratio, self.target_recent_ratio)
        return {
            "action": "passive_context_adjustment",
            "from_state": 1,
            "to_state": 1,
            "decision_position": int(position),
        }

    def ready_to_arbitrate(self) -> bool:
        return False

    def state_summary(self) -> list[dict[str, Any]]:
        return [
            {
                "state_id": 1,
                "status": "active",
                "recent_size": len(self.recent_indices),
                "archive_size": len(self.archive),
                "support_size": len(self.recent_indices) + len(self.archive),
                "effective_context_size": self.effective_context_size,
                "recent_ratio": self.recent_ratio,
                "degradation_z": self.degradation_z,
                "compression": dict(self.compression_stats),
            }
        ]


@dataclass(frozen=True)
class PriorityPassiveMemoryConfig:
    """Time-aware priority memory reproduced from the Electricity B=1 run.

    Capacity eviction scores every retained and newly labelled sample using
    recency, within-class representativeness, class balance, and a small
    admission bonus.  The full StreamContext controller may temporarily expose
    fewer than ``context_size`` samples, but the underlying priority memory is
    kept intact so that its context budget can recover smoothly.
    """

    context_size: int = 1500
    min_context_size: int = 256
    context_recovery_step: int = 64
    time_decay_tau: float = 4096.0
    priority_w_time: float = 0.55
    priority_w_repr: float = 0.35
    priority_w_class: float = 0.10
    new_sample_bonus: float = 0.02
    repr_k: int = 5
    max_score_items: int = 4096
    num_classes: int = 2
    max_sample_age: int = 0
    rebase_time_on_reactivation: bool = True
    restore_context_on_reactivation: bool = True


@dataclass
class PriorityContextItem:
    index: int
    label: int
    vector: np.ndarray
    item_id: int
    priority_time: int
    is_new: int = 0


def _minmax01(values: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return values
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if (
        not np.isfinite(minimum)
        or not np.isfinite(maximum)
        or maximum - minimum < epsilon
    ):
        return np.zeros_like(values, dtype=np.float64)
    return (values - minimum) / (maximum - minimum + epsilon)


class PriorityPassiveContextManager:
    """Passive memory used by the 94.63% Electricity batch-size-one run.

    This class preserves the old script's all-candidate top-priority eviction
    rule while implementing the same interface as the other StreamContext
    memories.  It is causal: a sample becomes eligible only in
    :meth:`commit_labeled`, after its prequential prediction has been scored.
    """

    def __init__(
        self,
        config: PriorityPassiveMemoryConfig,
        feature_dimension: int,
    ):
        self.config = config
        if config.context_size <= 0:
            raise ValueError("context_size must be positive")
        if config.min_context_size <= 0:
            raise ValueError("min_context_size must be positive")
        if config.min_context_size > config.context_size:
            raise ValueError("min_context_size cannot exceed context_size")
        if config.max_score_items < config.context_size + 1:
            raise ValueError("max_score_items must exceed context_size")
        self.feature_dimension = int(feature_dimension)
        self.items: list[PriorityContextItem] = []
        self.next_item_id = 0
        self.effective_context_size = int(config.context_size)
        self.target_context_size = int(config.context_size)
        # Kept for the common logging/controller interface. Selection itself is
        # determined by the priority score rather than a hard recent quota.
        self.recent_ratio = 0.0
        self.active_id = 1
        self.loss_fast: float | None = None
        self.loss_slow: float | None = None
        self.loss_var = 1e-4
        self.degradation_z = 0.0
        self.stats = {
            "added": 0,
            "capacity_evicted": 0,
            "stale_evicted": 0,
            "trim_calls": 0,
            "reactivations": 0,
            "time_rebases": 0,
        }

    @property
    def arbitration_pending(self) -> bool:
        return False

    def recover_context_budget(self) -> None:
        if self.effective_context_size < self.target_context_size:
            self.effective_context_size = min(
                self.target_context_size,
                self.effective_context_size + self.config.context_recovery_step,
            )
        elif self.effective_context_size > self.target_context_size:
            self.effective_context_size = max(
                self.target_context_size,
                self.effective_context_size
                - max(128, self.config.context_size // 8),
            )

    def _prune_stale(self, stream_position: int) -> None:
        max_age = int(self.config.max_sample_age)
        if max_age <= 0:
            return
        cutoff = int(stream_position) - max_age
        before = len(self.items)
        self.items = [item for item in self.items if item.index >= cutoff]
        self.stats["stale_evicted"] += before - len(self.items)

    @staticmethod
    def _standardize_for_distance(matrix: np.ndarray) -> np.ndarray:
        mean = matrix.mean(axis=0, keepdims=True)
        std = matrix.std(axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        return np.clip((matrix - mean) / std, -6.0, 6.0).astype(np.float32)

    def _representativeness_score(
        self,
        matrix: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        standardized = self._standardize_for_distance(matrix)
        output = np.zeros(len(labels), dtype=np.float64)
        for label in np.unique(labels):
            positions = np.where(labels == label)[0]
            count = len(positions)
            if count <= 2:
                output[positions] = 1.0
                continue
            class_matrix = standardized[positions]
            center = class_matrix.mean(axis=0, keepdims=True)
            center_distance = np.sqrt(
                np.sum((class_matrix - center) ** 2, axis=1)
            )
            gram = class_matrix @ class_matrix.T
            squared_norm = np.sum(
                class_matrix * class_matrix, axis=1, keepdims=True
            )
            pairwise_squared = np.maximum(
                squared_norm + squared_norm.T - 2.0 * gram,
                0.0,
            )
            np.fill_diagonal(pairwise_squared, np.inf)
            neighbor_count = max(1, min(self.config.repr_k, count - 1))
            nearest = np.partition(
                pairwise_squared,
                kth=neighbor_count - 1,
                axis=1,
            )[:, :neighbor_count]
            nearest_distance = np.sqrt(np.mean(nearest, axis=1))
            uniqueness = _minmax01(nearest_distance)
            core = 1.0 - _minmax01(center_distance)
            score = 0.65 * uniqueness + 0.35 * core
            if count >= 20:
                percentile_95 = float(np.quantile(center_distance, 0.95))
                score = np.where(
                    center_distance > percentile_95,
                    0.75 * score,
                    score,
                )
            output[positions] = np.clip(score, 0.0, 1.0)
        return output

    def _class_balance_score(self, labels: np.ndarray) -> np.ndarray:
        class_count = max(
            int(self.config.num_classes),
            int(labels.max()) + 1 if len(labels) else 1,
        )
        counts = np.bincount(labels, minlength=class_count).astype(np.float64)
        ideal = max(float(len(labels)), 1.0) / max(float(class_count), 1.0)
        raw = np.clip(ideal / np.maximum(counts, 1.0), 0.0, 3.0)
        return _minmax01(raw)[labels]

    def _compute_priorities(self, current_t: int) -> np.ndarray:
        if not self.items:
            return np.empty(0, dtype=np.float64)
        if len(self.items) > self.config.max_score_items:
            raise ValueError(
                "Too many priority-memory items to score exactly: "
                f"n={len(self.items)}, max_score_items={self.config.max_score_items}"
            )
        matrix = np.stack([item.vector for item in self.items]).astype(np.float32)
        labels = np.asarray([item.label for item in self.items], dtype=np.int64)
        timestamps = np.asarray(
            [item.priority_time for item in self.items],
            dtype=np.float64,
        )
        new_flags = np.asarray([item.is_new for item in self.items], dtype=np.float64)
        ages = np.maximum(0.0, float(current_t) - timestamps)
        time_score = np.exp(
            -ages / max(float(self.config.time_decay_tau), 1e-6)
        )
        return np.asarray(
            self.config.priority_w_time * time_score
            + self.config.priority_w_repr
            * self._representativeness_score(matrix, labels)
            + self.config.priority_w_class * self._class_balance_score(labels)
            + self.config.new_sample_bonus * new_flags,
            dtype=np.float64,
        )

    def _top_priority_positions(self, count: int, current_t: int) -> np.ndarray:
        count = min(max(0, int(count)), len(self.items))
        if count == 0:
            return np.empty(0, dtype=np.int64)
        if count == len(self.items):
            return np.arange(len(self.items), dtype=np.int64)
        scores = self._compute_priorities(current_t)
        # Same deterministic tie-break as the source experiment: score, then
        # newer stream position, then item id.
        order = np.lexsort(
            (
                np.asarray([item.item_id for item in self.items], dtype=np.int64),
                np.asarray([item.index for item in self.items], dtype=np.int64),
                scores,
            )
        )
        return order[-count:].astype(np.int64, copy=False)

    def _trim_to_capacity(self, current_t: int) -> None:
        if len(self.items) <= self.config.context_size:
            return
        keep_positions = set(
            self._top_priority_positions(self.config.context_size, current_t).tolist()
        )
        before = len(self.items)
        self.items = [
            item for position, item in enumerate(self.items) if position in keep_positions
        ]
        self.stats["capacity_evicted"] += before - len(self.items)
        self.stats["trim_calls"] += 1

    def commit_labeled(
        self,
        indices: np.ndarray,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
    ) -> None:
        if x is None or y is None:
            raise ValueError(
                "PriorityPassiveContextManager requires feature and label arrays"
            )
        indices = np.asarray(indices, dtype=np.int64)
        if not len(indices):
            return
        for index in indices:
            index = int(index)
            vector = np.asarray(x[index], dtype=np.float32)
            if vector.ndim != 1 or len(vector) != self.feature_dimension:
                raise ValueError("Unexpected feature shape in priority memory")
            self.items.append(
                PriorityContextItem(
                    index=index,
                    label=int(y[index]),
                    vector=vector,
                    item_id=self.next_item_id,
                    priority_time=index,
                    is_new=1,
                )
            )
            self.next_item_id += 1
        self.stats["added"] += len(indices)
        current_t = int(indices[-1]) + 1
        self._prune_stale(current_t)
        self._trim_to_capacity(current_t)
        for item in self.items:
            item.is_new = 0

    def reactivate(self, stream_position: int) -> None:
        """Resume a dormant concept without charging it for inactive time.

        Absolute ``index`` values remain unchanged for causal data access and
        optional hard age expiry.  Only the clock used by the soft priority
        recency term is shifted, preserving all within-state age differences.
        """
        stream_position = int(stream_position)
        self.stats["reactivations"] += 1
        if self.config.rebase_time_on_reactivation and self.items:
            newest_priority_time = max(item.priority_time for item in self.items)
            shift = stream_position - newest_priority_time
            if shift > 0:
                for item in self.items:
                    item.priority_time += shift
                self.stats["time_rebases"] += 1
        if self.config.restore_context_on_reactivation:
            self.effective_context_size = int(self.config.context_size)
            self.target_context_size = int(self.config.context_size)
        self.loss_fast = None
        self.loss_slow = None
        self.loss_var = 1e-4
        self.degradation_z = 0.0

    def context_indices_with_budget(
        self,
        stream_position: int,
        budget_override: int | None = None,
    ) -> np.ndarray:
        self._prune_stale(stream_position)
        budget = min(self.config.context_size, int(self.effective_context_size))
        if budget_override is not None:
            budget = min(budget, max(0, int(budget_override)))
        positions = self._top_priority_positions(budget, int(stream_position))
        indices = sorted(
            self.items[position].index
            for position in positions
            if self.items[position].index < int(stream_position)
        )
        if indices:
            recent_cutoff = int(stream_position) - min(256, self.config.context_size)
            self.recent_ratio = float(
                np.mean(np.asarray(indices, dtype=np.int64) >= recent_cutoff)
            )
        else:
            self.recent_ratio = 0.0
        return np.asarray(indices, dtype=np.int64)

    def context_indices(self, stream_position: int) -> np.ndarray:
        return self.context_indices_with_budget(stream_position)

    def observe_prequential(
        self,
        label_losses: np.ndarray,
        event_mode: str | None,
    ) -> None:
        values = np.asarray(label_losses, dtype=np.float64)
        values = values[np.isfinite(values)]
        if not len(values):
            return
        current = float(np.clip(values.mean(), 0.0, 10.0))
        if self.loss_fast is None:
            self.loss_fast = current
            self.loss_slow = current
        else:
            residual = current - float(self.loss_slow)
            self.loss_var = 0.97 * self.loss_var + 0.03 * residual * residual
            self.loss_fast = 0.80 * self.loss_fast + 0.20 * current
            self.loss_slow = 0.98 * self.loss_slow + 0.02 * current
        self.degradation_z = max(
            0.0,
            (float(self.loss_fast) - float(self.loss_slow))
            / np.sqrt(self.loss_var + 1e-6),
        )
        severity = float(np.clip((self.degradation_z - 0.5) / 2.5, 0.0, 1.0))
        if event_mode == "gradual":
            severity = max(severity, 0.65)
        elif event_mode == "abrupt":
            severity = 1.0
        self.target_context_size = int(
            round(
                self.config.context_size
                - severity
                * (self.config.context_size - self.config.min_context_size)
            )
        )

    def handle_detection(self, mode: str, position: int) -> dict[str, Any]:
        severity = 0.70 if mode == "gradual" else 1.0
        self.target_context_size = int(
            round(
                self.config.context_size
                - severity
                * (self.config.context_size - self.config.min_context_size)
            )
        )
        self.effective_context_size = min(
            self.effective_context_size,
            self.target_context_size,
        )
        return {
            "action": "priority_passive_context_adjustment",
            "from_state": 1,
            "to_state": 1,
            "decision_position": int(position),
        }

    def ready_to_arbitrate(self) -> bool:
        return False

    def state_summary(self) -> list[dict[str, Any]]:
        labels = np.asarray([item.label for item in self.items], dtype=np.int64)
        class_count = max(int(self.config.num_classes), 1)
        counts = np.bincount(labels, minlength=class_count) if len(labels) else np.zeros(class_count, dtype=np.int64)
        return [
            {
                "state_id": self.active_id,
                "status": "active",
                "support_size": len(self.items),
                "effective_context_size": self.effective_context_size,
                "recent_ratio": self.recent_ratio,
                "degradation_z": self.degradation_z,
                "memory_type": "time_aware_priority",
                "class_counts": {
                    str(index): int(value) for index, value in enumerate(counts)
                },
                "stats": dict(self.stats),
            }
        ]


@dataclass(frozen=True)
class TabPFNGuidedHybridConfig:
    """Recent-first memory with a small TabPFN-ranked historical tail."""

    context_size: int = 2048
    recent_window_size: int = 1792
    history_capacity: int = 8192
    stable_recent_ratio: float = 0.75
    max_recent_ratio: float = 0.875
    ratio_recovery_step: float = 0.01
    ratio_alert_step: float = 0.05
    utility_ema: float = 0.90
    history_recency_tau: float = 20_000.0


@dataclass
class HistoryUtilityItem:
    index: int
    utility: float
    inserted_at: int


class TabPFNGuidedHybridContextManager:
    """Fixed-budget hybrid context selected by causal TabPFN signatures.

    Most context positions are always filled by a strict recent window.  Items
    that leave that window enter a bounded historical pool.  Their utility is
    learned causally from the query-to-context signatures returned by TabPFN
    on earlier batches; no raw-space distance, class balancing, or boundary
    heuristic is used.
    """

    def __init__(self, config: TabPFNGuidedHybridConfig, feature_dimension: int):
        del feature_dimension
        self.config = config
        if not 0.0 < config.stable_recent_ratio <= config.max_recent_ratio <= 1.0:
            raise ValueError("recent ratios must satisfy 0 < stable <= max <= 1")
        required_recent = int(round(config.context_size * config.max_recent_ratio))
        if config.recent_window_size < required_recent:
            raise ValueError("recent_window_size cannot satisfy max_recent_ratio")
        self.recent_indices: deque[int] = deque()
        self.history: dict[int, HistoryUtilityItem] = {}
        self.utility: dict[int, float] = {}
        self.effective_context_size = config.context_size
        self.recent_ratio = config.stable_recent_ratio
        self.target_recent_ratio = config.stable_recent_ratio
        self.active_id = 1
        self.loss_fast: float | None = None
        self.loss_slow: float | None = None
        self.loss_var = 1e-4
        self.degradation_z = 0.0
        self.insert_counter = 0
        self.stats = {
            "promoted_to_history": 0,
            "history_capacity_evicted": 0,
            "signature_updates": 0,
            "signature_columns_observed": 0,
        }

    @property
    def arbitration_pending(self) -> bool:
        return False

    def recover_context_budget(self) -> None:
        if self.recent_ratio < self.target_recent_ratio:
            self.recent_ratio = min(
                self.target_recent_ratio,
                self.recent_ratio + self.config.ratio_alert_step,
            )
        elif self.recent_ratio > self.target_recent_ratio:
            self.recent_ratio = max(
                self.target_recent_ratio,
                self.recent_ratio - self.config.ratio_recovery_step,
            )

    def _effective_history_utility(self, item: HistoryUtilityItem, position: int) -> float:
        age = max(0, int(position) - item.index)
        decay = np.exp(-age / max(self.config.history_recency_tau, 1.0))
        return float(item.utility * decay)

    def _promote(self, index: int) -> None:
        self.insert_counter += 1
        item = HistoryUtilityItem(
            index=int(index),
            utility=float(self.utility.pop(int(index), 0.0)),
            inserted_at=self.insert_counter,
        )
        self.history[item.index] = item
        self.stats["promoted_to_history"] += 1
        if len(self.history) <= self.config.history_capacity:
            return
        # Capacity eviction is based on task-aware utility with mild temporal
        # decay.  Old evidence can survive when TabPFN repeatedly attends to it.
        evict_index = min(
            self.history,
            key=lambda key: (
                self._effective_history_utility(self.history[key], item.index),
                self.history[key].inserted_at,
            ),
        )
        del self.history[evict_index]
        self.stats["history_capacity_evicted"] += 1

    def commit_labeled(
        self,
        indices: np.ndarray,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
    ) -> None:
        del x, y
        for index in indices:
            index = int(index)
            if len(self.recent_indices) >= self.config.recent_window_size:
                self._promote(self.recent_indices.popleft())
            self.recent_indices.append(index)
            self.utility.setdefault(index, 0.0)

    def observe_tabpfn_context(
        self,
        context_indices: np.ndarray,
        signatures: torch.Tensor | np.ndarray | None,
    ) -> None:
        if signatures is None or len(context_indices) == 0:
            return
        values = torch.as_tensor(signatures).detach().float().cpu()
        if values.ndim != 2 or values.shape[1] != len(context_indices):
            return
        # Rescaling makes a uniform signature have utility near one regardless
        # of context length; only the ranking matters downstream.
        relevance = values.clamp_min(0).mean(dim=0).numpy() * len(context_indices)
        ema = float(self.config.utility_ema)
        observed = 0
        for index, value in zip(context_indices, relevance):
            index = int(index)
            value = float(value)
            if not np.isfinite(value):
                continue
            if index in self.history:
                old = self.history[index]
                old.utility = ema * old.utility + (1.0 - ema) * value
            elif index in self.utility:
                self.utility[index] = ema * self.utility[index] + (1.0 - ema) * value
            else:
                continue
            observed += 1
        if observed:
            self.stats["signature_updates"] += 1
            self.stats["signature_columns_observed"] += observed

    def context_indices_with_budget(
        self,
        stream_position: int,
        budget_override: int | None = None,
    ) -> np.ndarray:
        budget = self.config.context_size
        if budget_override is not None:
            budget = min(budget, max(0, int(budget_override)))
        recent_target = min(
            len(self.recent_indices),
            int(round(budget * self.recent_ratio)),
        )
        recent = list(self.recent_indices)[-recent_target:] if recent_target else []
        history_target = max(0, budget - len(recent))
        eligible = [
            item for item in self.history.values() if item.index < int(stream_position)
        ]
        eligible.sort(
            key=lambda item: (
                self._effective_history_utility(item, stream_position),
                item.index,
            ),
            reverse=True,
        )
        selected_history = [item.index for item in eligible[:history_target]]
        if len(recent) + len(selected_history) < budget:
            selected = set(recent)
            older_recent = [index for index in self.recent_indices if index not in selected]
            missing = budget - len(recent) - len(selected_history)
            recent = older_recent[-missing:] + recent
        return np.asarray(
            sorted(
                index
                for index in selected_history + recent
                if index < int(stream_position)
            ),
            dtype=np.int64,
        )

    def context_indices(self, stream_position: int) -> np.ndarray:
        return self.context_indices_with_budget(stream_position)

    def observe_prequential(self, label_losses: np.ndarray, event_mode: str | None) -> None:
        values = np.asarray(label_losses, dtype=np.float64)
        values = values[np.isfinite(values)]
        if not len(values):
            return
        current = float(np.clip(values.mean(), 0.0, 10.0))
        if self.loss_fast is None:
            self.loss_fast = current
            self.loss_slow = current
        else:
            residual = current - float(self.loss_slow)
            self.loss_var = 0.97 * self.loss_var + 0.03 * residual * residual
            self.loss_fast = 0.80 * self.loss_fast + 0.20 * current
            self.loss_slow = 0.98 * self.loss_slow + 0.02 * current
        self.degradation_z = max(
            0.0,
            (float(self.loss_fast) - float(self.loss_slow)) / np.sqrt(self.loss_var + 1e-6),
        )
        severity = float(np.clip((self.degradation_z - 0.5) / 2.5, 0.0, 1.0))
        if event_mode == "gradual":
            severity = max(severity, 0.65)
        elif event_mode == "abrupt":
            severity = 1.0
        self.target_recent_ratio = float(
            self.config.stable_recent_ratio
            + severity * (self.config.max_recent_ratio - self.config.stable_recent_ratio)
        )

    def handle_detection(self, mode: str, position: int) -> dict[str, Any]:
        severity = 0.70 if mode == "gradual" else 1.0
        self.target_recent_ratio = float(
            self.config.stable_recent_ratio
            + severity * (self.config.max_recent_ratio - self.config.stable_recent_ratio)
        )
        self.recent_ratio = max(self.recent_ratio, self.target_recent_ratio)
        return {
            "action": "hybrid_recent_reweight",
            "from_state": self.active_id,
            "to_state": self.active_id,
            "decision_position": int(position),
        }

    def ready_to_arbitrate(self) -> bool:
        return False

    def state_summary(self) -> list[dict[str, Any]]:
        return [
            {
                "state_id": self.active_id,
                "status": "active",
                "recent_size": len(self.recent_indices),
                "history_size": len(self.history),
                "support_size": len(self.recent_indices) + len(self.history),
                "effective_context_size": self.config.context_size,
                "recent_ratio": self.recent_ratio,
                "degradation_z": self.degradation_z,
                "memory_type": "tabpfn_guided_recent_first",
                "stats": dict(self.stats),
            }
        ]

@dataclass(frozen=True)
class FullStreamContextConfig:
    """Active branch layered on top of the complete passive memory branch."""

    passive: (
        PassiveMemoryConfig
        | PriorityPassiveMemoryConfig
        | TabPFNGuidedHybridConfig
    )
    confirmation_samples: int = 64
    recovery_loss_margin: float = 0.08
    reuse_margin: float = 0.15
    reuse_loss_factor: float = 1.25
    mixture_loss_gap: float = 0.03
    mixture_js_threshold: float = 0.02
    mixture_max_experts: int = 2
    recurrence_policy: str = "reuse"
    routing_anchor_size: int = 0
    quarantine_abrupt_evidence: bool = True


class FullStreamContextManager:
    """Passive-first state manager with sparse recurrent-concept routing.

    A detector alarm first changes only the current state's memory controls.
    Structural arbitration is delayed until a labeled confirmation window is
    available.  If predictive loss recovers, the current state is retained.
    Otherwise archived states are matched against causal validation evidence;
    the router uses one state by default and mixes two only when both their
    losses and their predictive distributions are compatible.
    """

    def __init__(
        self,
        config: FullStreamContextConfig,
        feature_dimension: int,
        memory_type: str = "compressed_archive",
    ):
        self.config = config
        self.feature_dimension = int(feature_dimension)
        self.memory_type = str(memory_type)
        initial = self._new_memory()
        self.context_size = int(initial.config.context_size)
        self.states: dict[int, Any] = {1: initial}
        self.status: dict[int, str] = {1: "active"}
        self.active_id = 1
        self.next_id = 2
        self.pending: dict[str, Any] | None = None
        self.selected_state_weights: dict[int, float] = {1: 1.0}
        self.routing_anchor_size = int(config.routing_anchor_size)
        if self.routing_anchor_size < 0:
            raise ValueError("routing_anchor_size cannot be negative")
        if config.recurrence_policy not in {"reuse", "fork"}:
            raise ValueError("recurrence_policy must be 'reuse' or 'fork'")
        self.routing_anchors: dict[int, deque[int]] = {
            1: deque(maxlen=max(1, self.routing_anchor_size))
        }

    def _new_memory(self) -> Any:
        if self.memory_type == "tabpfn_guided_hybrid":
            return TabPFNGuidedHybridContextManager(
                self.config.passive,
                self.feature_dimension,
            )
        if self.memory_type == "time_aware_priority":
            return PriorityPassiveContextManager(
                self.config.passive,
                self.feature_dimension,
            )
        if self.memory_type != "compressed_archive":
            raise ValueError(f"Unsupported memory_type: {self.memory_type}")
        return PassiveContextManager(self.config.passive, self.feature_dimension)

    @property
    def arbitration_pending(self) -> bool:
        return self.pending is not None

    @property
    def active(self) -> Any:
        return self.states[self.active_id]

    @property
    def effective_context_size(self) -> int:
        return self.active.effective_context_size

    @property
    def recent_ratio(self) -> float:
        return self.active.recent_ratio

    @property
    def degradation_z(self) -> float:
        return self.active.degradation_z

    def recover_context_budget(self) -> None:
        if self.pending is not None and self.config.quarantine_abrupt_evidence:
            return
        self.active.recover_context_budget()

    def context_indices(self, stream_position: int) -> np.ndarray:
        budget = min(
            self.context_size,
            int(self.active.effective_context_size),
        )
        if self.pending is not None and self.config.quarantine_abrupt_evidence:
            evidence = [
                int(index)
                for index in self.pending["evidence_indices"]
                if int(index) < int(stream_position)
            ]
            recent_budget = min(
                self.context_size,
                int(self.config.confirmation_samples),
            )
            return np.asarray(evidence[-recent_budget:], dtype=np.int64)
        weighted = sorted(
            self.selected_state_weights.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if len(weighted) == 1:
            return self.states[weighted[0][0]].context_indices_with_budget(
                stream_position, budget
            )
        allocations: list[int] = []
        remaining = budget
        for offset, (_, weight) in enumerate(weighted):
            if offset == len(weighted) - 1:
                allocation = remaining
            else:
                allocation = min(remaining, max(1, int(round(budget * weight))))
            allocations.append(allocation)
            remaining -= allocation
        merged: list[int] = []
        seen: set[int] = set()
        for (state_id, _), allocation in zip(weighted, allocations):
            indices = self.states[state_id].context_indices_with_budget(
                stream_position, allocation
            )
            for index in indices:
                if int(index) not in seen:
                    merged.append(int(index))
                    seen.add(int(index))
        return np.asarray(sorted(merged[-budget:]), dtype=np.int64)

    def observe_prequential(self, label_losses: np.ndarray, event_mode: str | None) -> None:
        if self.pending is not None and self.config.quarantine_abrupt_evidence:
            return
        self.active.observe_prequential(label_losses, event_mode)

    def observe_tabpfn_context(
        self,
        context_indices: np.ndarray,
        signatures: torch.Tensor | np.ndarray | None,
    ) -> None:
        for manager in self.states.values():
            callback = getattr(manager, "observe_tabpfn_context", None)
            if callback is not None:
                callback(context_indices, signatures)

    def handle_detection(self, mode: str, position: int) -> dict[str, Any]:
        passive_action = self.active.handle_detection(mode, position)
        if mode == "gradual":
            return {
                **passive_action,
                "action": "passive_gradual_adjustment",
                "router_mode": "top1",
            }
        if self.pending is not None:
            raise RuntimeError("Cannot start a second recovery gate while one is pending")
        self.pending = {
            "detection_position": int(position),
            "from_state": int(self.active_id),
            "baseline_loss": (
                None if self.active.loss_slow is None else float(self.active.loss_slow)
            ),
            "evidence_indices": deque(maxlen=self.config.confirmation_samples),
        }
        return {
            "action": "passive_recovery_gate",
            "from_state": self.active_id,
            "to_state": self.active_id,
            "decision_position": None,
            "router_mode": "top1",
        }

    def commit_labeled(
        self,
        indices: np.ndarray,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
    ) -> None:
        if self.pending is not None and self.config.quarantine_abrupt_evidence:
            self.pending["evidence_indices"].extend(
                int(index) for index in indices
            )
            return
        self.active.commit_labeled(indices, x, y)
        if self.routing_anchor_size > 0:
            self.routing_anchors[self.active_id].extend(
                int(index) for index in indices
            )
        if self.pending is not None:
            self.pending["evidence_indices"].extend(int(index) for index in indices)

    def _commit_to_state(
        self,
        state_id: int,
        indices: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
    ) -> None:
        self.states[state_id].commit_labeled(indices, x, y)
        if self.routing_anchor_size > 0:
            self.routing_anchors[state_id].extend(
                int(index) for index in indices
            )

    def ready_to_arbitrate(self) -> bool:
        return (
            self.pending is not None
            and len(self.pending["evidence_indices"]) >= self.config.confirmation_samples
        )

    @staticmethod
    def _extract_probabilities(
        extractor: Any,
        x: np.ndarray,
        y: np.ndarray,
        context_indices: np.ndarray,
        validation_indices: np.ndarray,
        n_classes: int,
    ) -> torch.Tensor | None:
        if len(context_indices) == 0 or len(validation_indices) == 0:
            return None
        _, _, probabilities = extractor.extract(
            x[context_indices],
            y[context_indices],
            x[validation_indices],
            n_classes,
        )
        return probabilities.detach().float().cpu().clamp_min(1e-8)

    @classmethod
    def _matching_result(
        cls,
        extractor: Any,
        x: np.ndarray,
        y: np.ndarray,
        context_indices: np.ndarray,
        validation_indices: np.ndarray,
        n_classes: int,
    ) -> tuple[float, torch.Tensor | None]:
        probabilities = cls._extract_probabilities(
            extractor,
            x,
            y,
            context_indices,
            validation_indices,
            n_classes,
        )
        if probabilities is None:
            return float("inf"), None
        labels = torch.from_numpy(y[validation_indices]).long()
        loss = -probabilities[torch.arange(len(labels)), labels].log().mean()
        return float(loss.item()), probabilities

    @staticmethod
    def _js_divergence(first: torch.Tensor, second: torch.Tensor) -> float:
        mixture = 0.5 * (first + second)
        value = 0.5 * (
            (first * (first.log() - mixture.log())).sum(dim=1)
            + (second * (second.log() - mixture.log())).sum(dim=1)
        ).mean()
        return float(value.item())

    def _make_state_from_evidence(
        self,
        state_id: int,
        evidence: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Any:
        manager = self._new_memory()
        manager.commit_labeled(evidence, x, y)
        self.states[state_id] = manager
        self.status[state_id] = "dormant"
        self.routing_anchors[state_id] = deque(
            (int(index) for index in evidence[-self.routing_anchor_size :]),
            maxlen=max(1, self.routing_anchor_size),
        )
        return manager

    def _routing_context(
        self,
        state_id: int,
        stream_position: int,
    ) -> np.ndarray:
        if self.routing_anchor_size <= 0:
            return self.states[state_id].context_indices(stream_position)
        return np.asarray(
            [
                index
                for index in self.routing_anchors[state_id]
                if index < int(stream_position)
            ],
            dtype=np.int64,
        )

    def _fork_state_from_parent(
        self,
        state_id: int,
        parent_id: int,
        evidence: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Any:
        manager = self._new_memory()
        parent = self._routing_context(parent_id, int(evidence.min()))
        inherited_count = max(0, self.context_size - len(evidence))
        inherited = parent[-inherited_count:] if inherited_count else parent[:0]
        support = np.asarray(
            list(dict.fromkeys([*(int(i) for i in inherited), *(int(i) for i in evidence)])),
            dtype=np.int64,
        )
        manager.commit_labeled(support, x, y)
        self.states[state_id] = manager
        self.status[state_id] = "dormant"
        self.routing_anchors[state_id] = deque(
            (int(index) for index in support[-self.routing_anchor_size :]),
            maxlen=max(1, self.routing_anchor_size),
        )
        return manager

    def arbitrate(
        self,
        extractor: Any,
        x: np.ndarray,
        y: np.ndarray,
        n_classes: int,
        decision_position: int,
    ) -> dict[str, Any]:
        if self.pending is None:
            raise RuntimeError("No pending recovery gate")
        evidence = np.asarray(self.pending["evidence_indices"], dtype=np.int64)
        if len(evidence) < 16:
            raise RuntimeError("At least 16 post-alarm labels are required")
        split = min(max(8, len(evidence) // 2), len(evidence) - 8)
        recent_support, validation = evidence[:split], evidence[split:]
        validation_floor = int(validation.min())
        recent_loss, _ = self._matching_result(
            extractor, x, y, recent_support, validation, n_classes
        )

        current_id = int(self.active_id)
        current_context = self._routing_context(current_id, validation_floor)
        current_loss, _ = self._matching_result(
            extractor, x, y, current_context, validation, n_classes
        )
        baseline = self.pending["baseline_loss"]
        fast_loss = self.active.loss_fast
        fast_recovered = (
            baseline is not None
            and fast_loss is not None
            and float(fast_loss) <= float(baseline) + self.config.recovery_loss_margin
        )
        # The alarm is also cancelled when the adaptively refreshed current
        # context is already as predictive as a post-alarm specialist.  This
        # second gate is important on noisy real streams: it prevents a single
        # loss spike from manufacturing a new expert even if the EWMA has not
        # numerically returned to its old baseline yet.
        context_recovered = (
            current_loss <= recent_loss + self.config.reuse_margin
            and current_loss
            <= np.log(max(2, n_classes)) * self.config.reuse_loss_factor
        )
        recovered = fast_recovered or context_recovered

        candidate_losses: dict[int, float] = {}
        candidate_probabilities: dict[int, torch.Tensor] = {}
        if not recovered:
            for state_id, state in self.states.items():
                if state_id == current_id or self.status.get(state_id) != "dormant":
                    continue
                context = self._routing_context(state_id, validation_floor)
                loss, probabilities = self._matching_result(
                    extractor, x, y, context, validation, n_classes
                )
                candidate_losses[state_id] = loss
                if probabilities is not None:
                    candidate_probabilities[state_id] = probabilities

        absolute_threshold = np.log(max(2, n_classes)) * self.config.reuse_loss_factor
        ranked = sorted(candidate_losses, key=candidate_losses.get)
        best_id = ranked[0] if ranked else None
        best_loss = candidate_losses[best_id] if best_id is not None else float("inf")
        reusable = (
            best_id is not None
            and best_loss <= absolute_threshold
            and best_loss <= recent_loss + self.config.reuse_margin
        )

        router_mode = "top1"
        mixture_js: float | None = None
        selected_weights: dict[int, float]
        if recovered:
            chosen_id = current_id
            action = "keep_state_after_passive_recovery"
            selected_weights = {chosen_id: 1.0}
        elif reusable:
            if self.config.recurrence_policy == "fork":
                chosen_id = self.next_id
                self.next_id += 1
                self._fork_state_from_parent(
                    chosen_id,
                    int(best_id),
                    evidence,
                    x,
                    y,
                )
                action = "fork_recurrent_state"
            else:
                chosen_id = int(best_id)
                action = "reuse_state"
            selected_weights = {chosen_id: 1.0}
            if (
                action == "reuse_state"
                and self.config.mixture_max_experts >= 2
                and len(ranked) >= 2
            ):
                second_id = int(ranked[1])
                second_loss = float(candidate_losses[second_id])
                if (
                    second_loss <= absolute_threshold
                    and second_loss <= recent_loss + self.config.reuse_margin
                    and second_loss - best_loss <= self.config.mixture_loss_gap
                ):
                    mixture_js = self._js_divergence(
                        candidate_probabilities[chosen_id],
                        candidate_probabilities[second_id],
                    )
                    if mixture_js <= self.config.mixture_js_threshold:
                        inverse = np.asarray(
                            [1.0 / max(best_loss, 1e-6), 1.0 / max(second_loss, 1e-6)]
                        )
                        inverse /= inverse.sum()
                        selected_weights = {
                            chosen_id: float(inverse[0]),
                            second_id: float(inverse[1]),
                        }
                        router_mode = "compatible_top2_mixture"
        else:
            chosen_id = self.next_id
            self.next_id += 1
            self._make_state_from_evidence(chosen_id, evidence, x, y)
            action = "new_state"
            selected_weights = {chosen_id: 1.0}

        if chosen_id != current_id:
            self.status[current_id] = "dormant"
            self.status[chosen_id] = "active"
            self.active_id = int(chosen_id)
            if action == "reuse_state":
                reactivation = getattr(self.states[chosen_id], "reactivate", None)
                if reactivation is not None:
                    reactivation(decision_position)
                self._commit_to_state(chosen_id, evidence, x, y)
        else:
            self.status[chosen_id] = "active"
            if recovered and self.config.quarantine_abrupt_evidence:
                reactivation = getattr(self.states[chosen_id], "reactivate", None)
                if reactivation is not None:
                    reactivation(decision_position)
                self._commit_to_state(chosen_id, evidence, x, y)
        self.selected_state_weights = selected_weights
        previous_id = int(self.pending["from_state"])
        detection_position = int(self.pending["detection_position"])
        self.pending = None
        return {
            "action": action,
            "from_state": previous_id,
            "to_state": int(chosen_id),
            "detection_position": detection_position,
            "decision_position": int(decision_position),
            "recent_loss": float(recent_loss),
            "current_loss": float(current_loss),
            "baseline_loss": baseline,
            "fast_loss": fast_loss,
            "fast_recovered": bool(fast_recovered),
            "context_recovered": bool(context_recovered),
            "match_loss": None if best_id is None else float(best_loss),
            "matched_state": best_id,
            "candidate_losses": {
                str(key): float(value) for key, value in candidate_losses.items()
            },
            "router_mode": router_mode,
            "router_weights": {
                str(key): float(value) for key, value in selected_weights.items()
            },
            "mixture_js": mixture_js,
        }

    def state_summary(self) -> list[dict[str, Any]]:
        summary: list[dict[str, Any]] = []
        for state_id, manager in sorted(self.states.items()):
            record = manager.state_summary()[0]
            record.update(
                {
                    "state_id": state_id,
                    "status": self.status[state_id],
                    "router_weight": self.selected_state_weights.get(state_id, 0.0),
                }
            )
            summary.append(record)
        return summary
