#!/usr/bin/env python3
"""Prequential SINE/Electricity experiment with frozen-TabPFN support states."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dchr_tabpfn import (
    DetectorConfig,
    DualTimescaleChangeDetector,
    FullStreamContextConfig,
    FullStreamContextManager,
    PassiveContextManager,
    PassiveMemoryConfig,
    PriorityPassiveContextManager,
    PriorityPassiveMemoryConfig,
    SlidingWindowContextManager,
    StateManagerConfig,
    TabPFNGuidedHybridConfig,
    TabPFNSupportStateManager,
)


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from tabpfn_extractor import (  # noqa: E402
    DEFAULT_CKPT,
    DEFAULT_TABPFN_SRC,
    TabPFNAttentionExtractor,
)

REPO_ROOT = THIS_DIR.parent


TRUE_CHANGES = (5000, 10000, 15000, 20000, 25000)
CONCEPT_SEQUENCE = (1, 2, 1, 2, 1, 2)


def load_sine(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = np.load(path)
    x = payload["x_train"].astype(np.float32)
    y_raw = payload["y_train"]
    y = y_raw.argmax(axis=1).astype(np.int64) if y_raw.ndim == 2 else y_raw.astype(np.int64)
    if len(x) != 30_000:
        raise ValueError(f"Expected 30000 SINE instances, found {len(x)}")
    concept = np.empty(len(x), dtype=np.int64)
    for segment, concept_id in enumerate(CONCEPT_SEQUENCE):
        concept[segment * 5000 : (segment + 1) * 5000] = concept_id
    return x, y, concept


def load_stream(
    path: Path,
    dataset: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...], str]:
    inferred = path.stem.lower()
    dataset = inferred if dataset == "auto" else dataset.lower()
    if dataset == "sine":
        x, y, concept = load_sine(path)
        return x, y, concept, TRUE_CHANGES, "Sine"
    if dataset != "electricity":
        raise ValueError(f"Unsupported dataset: {dataset}")
    payload = np.load(path)
    x = payload["x_train"].astype(np.float32)
    y_raw = payload["y_train"]
    y = y_raw.argmax(axis=1).astype(np.int64) if y_raw.ndim == 2 else y_raw.astype(np.int64)
    labels = sorted(np.unique(y).tolist())
    mapping = {label: index for index, label in enumerate(labels)}
    y = np.asarray([mapping[int(label)] for label in y], dtype=np.int64)
    concept = np.ones(len(x), dtype=np.int64)
    # Electricity contains natural changes but has no point-wise drift annotation.
    return x, y, concept, (), "Electricity"


def prequential_loss(probabilities: torch.Tensor, labels: np.ndarray) -> np.ndarray:
    p = probabilities.detach().float().cpu().clamp_min(1e-8)
    target = torch.from_numpy(labels).long()
    return -p[torch.arange(len(target)), target].log().numpy()


def normalized_entropy(probabilities: torch.Tensor) -> float:
    p = probabilities.detach().float().cpu().clamp_min(1e-8)
    entropy = -(p * p.log()).sum(dim=1).mean()
    return float(entropy / max(np.log(p.shape[1]), 1e-8))


def evaluate_events(
    events: list[dict],
    true_changes: tuple[int, ...],
    tolerance: int = 2048,
) -> dict:
    if not true_changes:
        return {
            "ground_truth_available": False,
            "recall": None,
            "false_positives": None,
            "mean_delay": None,
            "matches": [],
            "unverified_events": events,
        }
    unused = set(range(len(events)))
    matches: list[dict] = []
    for change in true_changes:
        candidates = [
            index
            for index in unused
            if change <= int(events[index]["position"]) <= change + tolerance
        ]
        if not candidates:
            continue
        index = min(candidates, key=lambda item: int(events[item]["position"]))
        unused.remove(index)
        matches.append(
            {
                "true_position": change,
                "detected_position": int(events[index]["position"]),
                "delay": int(events[index]["position"]) - change,
                "mode": events[index]["mode"],
                "action": events[index]["action"],
                "to_state": events[index].get("to_state"),
            }
        )
    return {
        "ground_truth_available": True,
        "recall": len(matches) / len(true_changes),
        "false_positives": len(unused),
        "mean_delay": float(np.mean([item["delay"] for item in matches])) if matches else None,
        "matches": matches,
        "unmatched_events": [events[index] for index in sorted(unused)],
    }


def run_seed(args: argparse.Namespace, seed: int) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    x, y, concept, true_changes, dataset_name = load_stream(args.data, args.dataset)
    if args.max_samples > 0:
        limit = min(int(args.max_samples), len(x))
        x, y, concept = x[:limit], y[:limit], concept[:limit]
    n_classes = int(y.max()) + 1
    extractor = TabPFNAttentionExtractor(
        args.tabpfn_src,
        args.tabpfn_ckpt,
        args.tabpfn_device,
        seed,
    )
    detector = DualTimescaleChangeDetector(
        DetectorConfig(
            short_labels=args.short_labels,
            reference_labels=args.reference_labels,
            abrupt_z=args.abrupt_z,
            gradual_z=args.gradual_z,
            gradual_patience=args.gradual_patience,
            min_loss_delta=args.min_loss_delta,
            cooldown_samples=args.cooldown_samples,
            warmup_samples=args.warmup_samples,
        )
    )
    passive_config = PassiveMemoryConfig(
        context_size=args.context_size,
        min_context_size=args.min_context_size,
        recent_window_size=args.recent_window_size,
        stable_recent_ratio=args.stable_recent_ratio,
        max_recent_ratio=args.max_recent_ratio,
        context_recovery_step=args.context_recovery_step,
        ratio_recovery_step=args.ratio_recovery_step,
        redundancy_radius=args.redundancy_radius,
        boundary_replacement_margin=args.boundary_replacement_margin,
        max_sample_age=args.max_sample_age,
    )
    priority_config = PriorityPassiveMemoryConfig(
        context_size=args.context_size,
        min_context_size=args.min_context_size,
        context_recovery_step=args.context_recovery_step,
        time_decay_tau=args.priority_time_decay_tau,
        priority_w_time=args.priority_w_time,
        priority_w_repr=args.priority_w_repr,
        priority_w_class=args.priority_w_class,
        new_sample_bonus=args.priority_new_sample_bonus,
        repr_k=args.priority_repr_k,
        max_score_items=args.priority_max_score_items,
        num_classes=n_classes,
        max_sample_age=args.max_sample_age,
        rebase_time_on_reactivation=bool(args.priority_rebase_on_reactivation),
        restore_context_on_reactivation=bool(
            args.priority_restore_context_on_reactivation
        ),
    )
    hybrid_config = TabPFNGuidedHybridConfig(
        context_size=args.context_size,
        recent_window_size=args.hybrid_recent_window_size,
        history_capacity=args.hybrid_history_capacity,
        stable_recent_ratio=args.hybrid_stable_recent_ratio,
        max_recent_ratio=args.hybrid_max_recent_ratio,
        ratio_recovery_step=args.hybrid_ratio_recovery_step,
        ratio_alert_step=args.hybrid_ratio_alert_step,
        utility_ema=args.hybrid_utility_ema,
        history_recency_tau=args.hybrid_history_recency_tau,
    )
    if args.streamcontext_mode == "sliding_window":
        manager = SlidingWindowContextManager(args.context_size)
    elif args.streamcontext_mode == "passive_only":
        if args.passive_memory == "priority":
            manager = PriorityPassiveContextManager(
                priority_config,
                feature_dimension=x.shape[1],
            )
        else:
            manager = PassiveContextManager(
                passive_config,
                feature_dimension=x.shape[1],
            )
    elif args.streamcontext_mode == "full":
        selected_passive_config = (
            priority_config if args.passive_memory == "priority" else passive_config
        )
        manager = FullStreamContextManager(
            FullStreamContextConfig(
                passive=selected_passive_config,
                confirmation_samples=args.confirmation_samples,
                recovery_loss_margin=args.recovery_loss_margin,
                reuse_margin=args.full_reuse_margin,
                reuse_loss_factor=args.reuse_loss_factor,
                mixture_loss_gap=args.mixture_loss_gap,
                mixture_js_threshold=args.mixture_js_threshold,
                mixture_max_experts=args.mixture_max_experts,
                recurrence_policy=args.full_recurrence_policy,
                routing_anchor_size=args.full_routing_anchor_size,
                quarantine_abrupt_evidence=bool(
                    args.full_quarantine_abrupt_evidence
                ),
            ),
            feature_dimension=x.shape[1],
            memory_type=(
                "time_aware_priority"
                if args.passive_memory == "priority"
                else "compressed_archive"
            ),
        )
    elif args.streamcontext_mode == "hybrid_full":
        manager = FullStreamContextManager(
            FullStreamContextConfig(
                passive=hybrid_config,
                confirmation_samples=args.hybrid_confirmation_samples,
                recovery_loss_margin=args.recovery_loss_margin,
                reuse_margin=args.full_reuse_margin,
                reuse_loss_factor=args.reuse_loss_factor,
                mixture_loss_gap=args.mixture_loss_gap,
                mixture_js_threshold=args.mixture_js_threshold,
                mixture_max_experts=args.mixture_max_experts,
                recurrence_policy=args.full_recurrence_policy,
                routing_anchor_size=args.full_routing_anchor_size,
                quarantine_abrupt_evidence=bool(
                    args.full_quarantine_abrupt_evidence
                ),
            ),
            feature_dimension=x.shape[1],
            memory_type="tabpfn_guided_hybrid",
        )
    else:
        manager = TabPFNSupportStateManager(
            StateManagerConfig(
                context_size=args.context_size,
                min_context_size=args.min_context_size,
                context_recovery_step=args.context_recovery_step,
                confirmation_samples=args.confirmation_samples,
                rollback_samples=args.rollback_samples,
                reuse_margin=args.reuse_margin,
                reuse_loss_factor=args.reuse_loss_factor,
            )
        )

    started = time.perf_counter()
    total_correct = 0
    segment_count = 6 if dataset_name == "Sine" else 1
    segment_correct = np.zeros(segment_count, dtype=np.int64)
    segment_total = np.zeros(segment_count, dtype=np.int64)
    post_drift_correct = np.zeros(len(true_changes), dtype=np.int64)
    post_drift_total = np.zeros(len(true_changes), dtype=np.int64)
    previous_embedding: torch.Tensor | None = None
    events: list[dict] = []
    pending_event_index: int | None = None
    records: list[dict] = []

    for batch_index, start in enumerate(range(0, len(x), args.batch_size)):
        end = min(start + args.batch_size, len(x))
        manager.recover_context_budget()
        context_indices = manager.context_indices(start)
        fusion = extractor.extract_fusion(
            x[context_indices] if len(context_indices) else x[:0],
            y[context_indices] if len(context_indices) else y[:0],
            x[start:end],
            n_classes,
        )
        signature_observer = getattr(manager, "observe_tabpfn_context", None)
        if signature_observer is not None:
            signature_observer(context_indices, fusion.get("signatures"))
        probabilities = fusion["probs"].detach().float().cpu()
        predictions = probabilities.argmax(dim=1).numpy()
        correct = int((predictions == y[start:end]).sum())
        total_correct += correct
        segment = start // 5000 if dataset_name == "Sine" else 0
        segment_correct[segment] += correct
        segment_total[segment] += end - start
        for change_index, change in enumerate(true_changes):
            overlap_start = max(start, change)
            overlap_end = min(end, change + 10 * args.batch_size)
            if overlap_start < overlap_end:
                local_start = overlap_start - start
                local_end = overlap_end - start
                post_drift_correct[change_index] += int(
                    (predictions[local_start:local_end] == y[overlap_start:overlap_end]).sum()
                )
                post_drift_total[change_index] += overlap_end - overlap_start

        losses = prequential_loss(probabilities, y[start:end])
        entropy = normalized_entropy(probabilities)
        query_embeddings = fusion["query_embeddings"].detach().float().cpu()
        current_embedding = query_embeddings.mean(dim=0)
        embedding_shift = 0.0
        if previous_embedding is not None and float(current_embedding.abs().sum()) > 0:
            embedding_shift = float(
                1.0
                - F.cosine_similarity(
                    current_embedding.unsqueeze(0),
                    previous_embedding.unsqueeze(0),
                ).item()
            )
        previous_embedding = current_embedding
        proxy_value = entropy + max(0.0, embedding_shift)
        event_mode, diagnostics = detector.update(end, losses, proxy_value)
        manager.observe_prequential(losses, event_mode)

        if event_mode is not None and not manager.arbitration_pending:
            action = manager.handle_detection(event_mode, end)
            event_record = {
                "position": int(end),
                "mode": event_mode,
                **action,
                **diagnostics,
                "entropy": entropy,
                "embedding_shift": embedding_shift,
            }
            events.append(event_record)
            if event_mode == "abrupt":
                pending_event_index = len(events) - 1
            print(
                f"[event] seed={seed} position={end} mode={event_mode} "
                f"action={action['action']} score={diagnostics['score']:.3f}",
                flush=True,
            )

        current_indices = np.arange(start, end, dtype=np.int64)
        manager.commit_labeled(current_indices, x, y)
        if manager.ready_to_arbitrate():
            decision = manager.arbitrate(extractor, x, y, n_classes, end)
            if pending_event_index is None:
                raise RuntimeError("Arbitration completed without a pending event record")
            events[pending_event_index].update(decision)
            print(
                f"[arbitration] seed={seed} detected={decision['detection_position']} "
                f"decided={end} action={decision['action']} "
                f"state={decision['from_state']}->{decision['to_state']} "
                f"match_loss={decision['match_loss']} recent_loss={decision['recent_loss']:.4f}",
                flush=True,
            )
            pending_event_index = None

        records.append(
            {
                "seed": seed,
                "batch": batch_index,
                "start": start,
                "end": end,
                "concept": int(concept[start]),
                "accuracy": correct / (end - start),
                "cumulative_accuracy": total_correct / end,
                "context_size": len(context_indices),
                "effective_context_budget": manager.effective_context_size,
                "recent_ratio": getattr(manager, "recent_ratio", None),
                "degradation_z": getattr(manager, "degradation_z", None),
                "active_state": manager.active_id,
                "router_mode": (
                    "mixture"
                    if len(getattr(manager, "selected_state_weights", {1: 1.0})) > 1
                    else "top1"
                ),
                "arbitration_pending": manager.arbitration_pending,
                "entropy": entropy,
                "embedding_shift": embedding_shift,
                "detector_score": diagnostics["score"],
                "detector_mode": event_mode or "stable",
            }
        )

    elapsed = time.perf_counter() - started
    serialized_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    result = {
        "seed": seed,
        "dataset": dataset_name,
        "accuracy": total_correct / len(x),
        "elapsed_seconds": elapsed,
        "segment_accuracy": {
            str(index + 1): (
                float(segment_correct[index] / segment_total[index])
                if segment_total[index]
                else None
            )
            for index in range(segment_count)
        },
        "first_10_batches_after_drift_accuracy": {
            str(index + 2): (
                float(post_drift_correct[index] / post_drift_total[index])
                if post_drift_total[index]
                else None
            )
            for index in range(len(true_changes))
        },
        "events": events,
        "event_evaluation": evaluate_events(events, true_changes),
        "states": manager.state_summary(),
        "config": serialized_config,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"result_seed{seed}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    with (args.out_dir / f"batches_seed{seed}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    plot_records(
        records,
        events,
        true_changes,
        args.out_dir / f"accuracy_seed{seed}.png",
    )
    print(
        f"[done] seed={seed} accuracy={result['accuracy']:.6f} "
        f"events={len(events)} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return result


def plot_records(
    records: list[dict],
    events: list[dict],
    true_changes: tuple[int, ...],
    destination: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = np.asarray([row["end"] for row in records])
    accuracy = np.asarray([row["accuracy"] for row in records])
    window = min(50, len(accuracy))
    kernel = np.ones(window) / window
    rolling = np.convolve(accuracy, kernel, mode="valid")
    rolling_positions = positions[window - 1 :]
    figure, axis = plt.subplots(figsize=(11, 4.5))
    axis.plot(
        rolling_positions,
        rolling,
        label="TabPFN + StreamContext/DCHR",
        linewidth=1.4,
    )
    for index, change in enumerate(true_changes):
        axis.axvline(change, color="black", linestyle="--", alpha=0.35, label="true drift" if index == 0 else None)
    for index, event in enumerate(events):
        color = "tab:red" if event["mode"] == "abrupt" else "tab:orange"
        axis.axvline(event["position"], color=color, alpha=0.55, label="detected" if index == 0 else None)
    axis.set(xlabel="stream position", ylabel="rolling batch accuracy", ylim=(0, 1.02))
    axis.legend(loc="lower right")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def aggregate(out_dir: Path) -> dict:
    results = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(out_dir.glob("result_seed*.json"))]
    if not results:
        raise FileNotFoundError(f"No result_seed*.json files under {out_dir}")
    accuracy = np.asarray([item["accuracy"] for item in results])
    elapsed = np.asarray([item["elapsed_seconds"] for item in results])
    recall_values = [
        item["event_evaluation"]["recall"]
        for item in results
        if item["event_evaluation"]["recall"] is not None
    ]
    recalls = np.asarray(recall_values, dtype=np.float64)
    delays = [
        match["delay"]
        for item in results
        for match in item["event_evaluation"]["matches"]
    ]
    structural_actions = {
        "new_state",
        "reuse_state",
        "fork_recurrent_state",
    }
    structural_counts = [
        sum(event.get("action") in structural_actions for event in item["events"])
        for item in results
    ]
    absorbed_counts = [
        sum(
            event.get("action") == "keep_state_after_passive_recovery"
            for event in item["events"]
        )
        for item in results
    ]
    mixture_counts = [
        sum(event.get("router_mode") == "compatible_top2_mixture" for event in item["events"])
        for item in results
    ]
    summary = {
        "seeds": [item["seed"] for item in results],
        "accuracy": {"mean": float(accuracy.mean()), "std": float(accuracy.std())},
        "elapsed_seconds": {"mean": float(elapsed.mean()), "std": float(elapsed.std())},
        "dataset": results[0].get("dataset"),
        "detection_recall": (
            {"mean": float(recalls.mean()), "by_seed": recalls.tolist()}
            if len(recalls)
            else None
        ),
        "mean_detection_delay": float(np.mean(delays)) if delays else None,
        "false_positives": (
            int(sum(item["event_evaluation"]["false_positives"] for item in results))
            if len(recalls)
            else None
        ),
        "event_count": {
            "mean": float(np.mean([len(item["events"]) for item in results])),
            "by_seed": [len(item["events"]) for item in results],
        },
        "structural_transition_count": {
            "mean": float(np.mean(structural_counts)),
            "by_seed": structural_counts,
        },
        "passive_recovery_absorbed_count": {
            "mean": float(np.mean(absorbed_counts)),
            "by_seed": absorbed_counts,
        },
        "top2_mixture_count": {
            "mean": float(np.mean(mixture_counts)),
            "by_seed": mixture_counts,
        },
        "events_by_seed": [{"seed": item["seed"], "events": item["events"]} for item in results],
        "states_by_seed": [{"seed": item["seed"], "states": item["states"]} for item in results],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prequential frozen-TabPFN experiments with adaptive stream contexts."
    )
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "Sine.npz")
    parser.add_argument("--dataset", choices=("auto", "sine", "electricity"), default="auto")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--streamcontext-mode",
        choices=(
            "hard_switch",
            "sliding_window",
            "passive_only",
            "full",
            "hybrid_full",
        ),
        default="hard_switch",
    )
    parser.add_argument("--context-size", type=int, default=1024)
    parser.add_argument(
        "--passive-memory",
        choices=("priority", "compressed_archive"),
        default="priority",
        help=(
            "Passive memory used by passive_only/full. 'priority' reproduces "
            "the time/representativeness/class-balanced Electricity method; "
            "'compressed_archive' keeps the previous recent/archive branch."
        ),
    )
    parser.add_argument("--min-context-size", type=int, default=256)
    parser.add_argument("--recent-window-size", type=int, default=256)
    parser.add_argument("--stable-recent-ratio", type=float, default=0.25)
    parser.add_argument("--max-recent-ratio", type=float, default=0.90)
    parser.add_argument("--ratio-recovery-step", type=float, default=0.02)
    parser.add_argument("--redundancy-radius", type=float, default=0.08)
    parser.add_argument("--boundary-replacement-margin", type=float, default=0.02)
    parser.add_argument("--priority-time-decay-tau", type=float, default=4096.0)
    parser.add_argument("--priority-w-time", type=float, default=0.55)
    parser.add_argument("--priority-w-repr", type=float, default=0.35)
    parser.add_argument("--priority-w-class", type=float, default=0.10)
    parser.add_argument("--priority-new-sample-bonus", type=float, default=0.02)
    parser.add_argument("--priority-repr-k", type=int, default=5)
    parser.add_argument("--priority-max-score-items", type=int, default=4096)
    parser.add_argument(
        "--priority-rebase-on-reactivation",
        type=int,
        choices=(0, 1),
        default=1,
        help="Pause the priority recency clock while a concept state is dormant.",
    )
    parser.add_argument(
        "--priority-restore-context-on-reactivation",
        type=int,
        choices=(0, 1),
        default=1,
        help="Restore the full context budget when a matched state is reactivated.",
    )
    parser.add_argument(
        "--max-sample-age",
        type=int,
        default=0,
        help="Discard support samples older than this many stream positions; 0 disables age expiry.",
    )
    parser.add_argument("--hybrid-recent-window-size", type=int, default=1792)
    parser.add_argument("--hybrid-history-capacity", type=int, default=8192)
    parser.add_argument("--hybrid-stable-recent-ratio", type=float, default=0.75)
    parser.add_argument("--hybrid-max-recent-ratio", type=float, default=0.875)
    parser.add_argument("--hybrid-ratio-recovery-step", type=float, default=0.01)
    parser.add_argument("--hybrid-ratio-alert-step", type=float, default=0.05)
    parser.add_argument("--hybrid-utility-ema", type=float, default=0.90)
    parser.add_argument("--hybrid-history-recency-tau", type=float, default=20000.0)
    parser.add_argument("--context-recovery-step", type=int, default=64)
    parser.add_argument("--confirmation-samples", type=int, default=64)
    parser.add_argument("--hybrid-confirmation-samples", type=int, default=256)
    parser.add_argument("--rollback-samples", type=int, default=128)
    parser.add_argument("--reuse-margin", type=float, default=0.05)
    parser.add_argument("--full-reuse-margin", type=float, default=0.15)
    parser.add_argument("--reuse-loss-factor", type=float, default=1.25)
    parser.add_argument("--recovery-loss-margin", type=float, default=0.08)
    parser.add_argument("--mixture-loss-gap", type=float, default=0.03)
    parser.add_argument("--mixture-js-threshold", type=float, default=0.02)
    parser.add_argument("--mixture-max-experts", type=int, default=2)
    parser.add_argument(
        "--full-recurrence-policy",
        choices=("reuse", "fork"),
        default="reuse",
        help=(
            "Reuse a matched dormant state directly, or create a new child "
            "state initialized from the matched concept."
        ),
    )
    parser.add_argument(
        "--full-routing-anchor-size",
        type=int,
        default=1024,
        help=(
            "Independent FIFO concept anchor used only for state matching; "
            "0 matches with the adaptive prediction memory."
        ),
    )
    parser.add_argument(
        "--full-quarantine-abrupt-evidence",
        type=int,
        choices=(0, 1),
        default=1,
        help=(
            "Keep the abrupt-change confirmation window out of the outgoing "
            "state and commit it once to the selected state after matching."
        ),
    )
    parser.add_argument("--short-labels", type=int, default=8)
    parser.add_argument("--reference-labels", type=int, default=32)
    parser.add_argument("--abrupt-z", type=float, default=3.0)
    parser.add_argument("--gradual-z", type=float, default=2.0)
    parser.add_argument("--gradual-patience", type=int, default=3)
    parser.add_argument("--min-loss-delta", type=float, default=0.10)
    parser.add_argument("--cooldown-samples", type=int, default=512)
    parser.add_argument("--warmup-samples", type=int, default=1024)
    parser.add_argument("--tabpfn-src", type=Path, default=DEFAULT_TABPFN_SRC)
    parser.add_argument("--tabpfn-ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--tabpfn-device", default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.aggregate_only:
        aggregate(args.out_dir)
        return
    for seed in args.seeds:
        run_seed(args, seed)
    aggregate(args.out_dir)


if __name__ == "__main__":
    main()
