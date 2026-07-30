"""Minimal TabPFN v2.6 inference and internal-signal extractor.

This module is intentionally independent of the earlier GCN experiments.  It
keeps TabPFN frozen, obtains prequential class probabilities, and exposes the
query embeddings and query-to-context attention used by the stream manager.
"""

from __future__ import annotations

import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TABPFN_SRC = Path(
    os.environ.get("TABPFN_SRC", REPO_ROOT / "external" / "TabPFN" / "src")
)
DEFAULT_CKPT = Path(
    os.environ.get(
        "TABPFN_CKPT",
        REPO_ROOT / "checkpoints" / "tabpfn-v2.6-classifier-v2.6_default.ckpt",
    )
)


def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    """Symmetrically normalize a non-negative 2-D or batched adjacency."""
    adj = torch.nan_to_num(adj.float(), nan=0.0, posinf=0.0, neginf=0.0)
    adj = torch.clamp(adj, min=0.0)
    adj = torch.maximum(adj, adj.transpose(-1, -2))
    if adj.ndim == 2:
        adj = adj.clone()
        adj.fill_diagonal_(1.0)
        degree = adj.sum(dim=1).clamp_min(1e-8)
        inv_sqrt = degree.pow(-0.5)
        return inv_sqrt[:, None] * adj * inv_sqrt[None, :]

    eye = torch.eye(adj.shape[-1], device=adj.device, dtype=adj.dtype)
    adj = adj.clone() + eye.unsqueeze(0)
    degree = adj.sum(dim=-1).clamp_min(1e-8)
    inv_sqrt = degree.pow(-0.5)
    return inv_sqrt[:, :, None] * adj * inv_sqrt[:, None, :]


class TabPFNAttentionExtractor:
    """Run frozen TabPFN and capture contextual embeddings and attention."""

    def __init__(self, tabpfn_src: Path, ckpt: Path, device: str, seed: int):
        tabpfn_src = Path(tabpfn_src).expanduser().resolve()
        ckpt = Path(ckpt).expanduser().resolve()
        if not tabpfn_src.is_dir():
            raise FileNotFoundError(
                f"TabPFN source directory not found: {tabpfn_src}. "
                "Pass --tabpfn-src or set TABPFN_SRC."
            )
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"TabPFN checkpoint not found: {ckpt}. "
                "Pass --tabpfn-ckpt or set TABPFN_CKPT."
            )
        if str(tabpfn_src) not in sys.path:
            sys.path.insert(0, str(tabpfn_src))

        from tabpfn import TabPFNClassifier
        import tabpfn.architectures.tabpfn_v2_6 as arch

        self.arch = arch
        classifier = TabPFNClassifier(
            n_estimators=1,
            model_path=str(ckpt),
            device=device,
            ignore_pretraining_limits=True,
            fit_mode="fit_preprocessors",
            inference_precision="auto",
            memory_saving_mode="auto",
            random_state=seed,
            n_preprocessing_jobs=1,
        )
        # A small fit instantiates the model and loads the supplied checkpoint.
        probe_x = np.random.default_rng(seed).normal(size=(8, 8)).astype(np.float32)
        probe_y = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
        classifier.fit(probe_x, probe_y)
        self.clf = classifier
        self.model = classifier.models_[0].eval()
        self.device = next(self.model.parameters()).device
        self.features_per_group = int(self.model.features_per_group)
        self.num_thinking_rows = int(self.model.add_thinking_rows.num_thinking_rows)
        self.embedding_size = int(self.model.input_size)
        self._last_block_output: torch.Tensor | None = None
        self._last_block_hook = self.model.blocks[-1].register_forward_hook(
            self._capture_last_block_output
        )
        self._patch_attention_modules()
        self.reset(0, 0)

    def _capture_last_block_output(self, _module, _inputs, output) -> None:
        self._last_block_output = output.detach()

    def reset(self, context_len: int, query_len: int) -> None:
        self.context_len = int(context_len)
        self.query_len = int(query_len)
        self.query_start_after_thinking = self.num_thinking_rows + self.context_len
        self.feature_sum: torch.Tensor | None = None
        self.feature_count = 0
        self.signature_sum: torch.Tensor | None = None
        self.signature_count = 0
        self._last_block_output = None

    def _patch_attention_modules(self) -> None:
        collector = self
        original_attention = self.arch._batched_scaled_dot_product_attention

        def make_row_forward():
            def row_forward(module, x_BrSE: torch.Tensor) -> torch.Tensor:
                batch_rows, columns, _ = x_BrSE.shape
                q = module.q_projection(x_BrSE).view(
                    batch_rows, columns, -1, module.head_dim
                )
                k = module.k_projection(x_BrSE).view(
                    batch_rows, columns, -1, module.head_dim
                )
                v = module.v_projection(x_BrSE).view(
                    batch_rows, columns, -1, module.head_dim
                )
                qh = q.permute(0, 2, 1, 3)
                kh = k.permute(0, 2, 1, 3)
                vh = v.permute(0, 2, 1, 3)
                attention = torch.softmax(
                    torch.matmul(qh, kh.transpose(-2, -1))
                    / math.sqrt(module.head_dim),
                    dim=-1,
                )
                out = torch.matmul(attention, vh).permute(0, 2, 1, 3)

                if collector.query_len > 0:
                    start = collector.query_start_after_thinking
                    end = start + collector.query_len
                    if end <= batch_rows and columns > 1:
                        feature_groups = columns - 1
                        feature_attention = attention[
                            start:end, :, :feature_groups, :feature_groups
                        ].mean(dim=1).detach().float()
                        if collector.feature_sum is None:
                            collector.feature_sum = torch.zeros_like(feature_attention)
                        if collector.feature_sum.shape == feature_attention.shape:
                            collector.feature_sum += feature_attention
                            collector.feature_count += 1

                output = out.reshape(
                    batch_rows, columns, module.head_dim * module.num_heads
                )
                return module.out_projection(output)

            return row_forward

        def make_column_forward():
            def column_forward(
                module,
                x_BcRE: torch.Tensor,
                single_eval_pos: int | None = None,
            ) -> torch.Tensor:
                batch_columns, rows, _ = x_BcRE.shape
                context_end = rows if single_eval_pos is None else single_eval_pos
                q = module.q_projection(x_BcRE).view(
                    batch_columns, rows, -1, module.head_dim
                )
                k = module.k_projection(x_BcRE[:, :context_end]).view(
                    batch_columns, context_end, -1, module.head_dim
                )
                v = module.v_projection(x_BcRE[:, :context_end]).view(
                    batch_columns, context_end, -1, module.head_dim
                )

                if single_eval_pos == rows or single_eval_pos is None:
                    out = original_attention(q, k, v)
                else:
                    out_train = original_attention(q[:, :context_end], k, v)
                    q_test = q[:, context_end:]
                    # TabPFN v2.6 uses the first key/value head for test queries.
                    k_one = k[:, :, :1]
                    v_one = v[:, :, :1]
                    qh = q_test.permute(0, 2, 1, 3)
                    kh = k_one.permute(0, 2, 1, 3).expand(
                        -1, qh.shape[1], -1, -1
                    )
                    vh = v_one.permute(0, 2, 1, 3).expand(
                        -1, qh.shape[1], -1, -1
                    )
                    attention = torch.softmax(
                        torch.matmul(qh, kh.transpose(-2, -1))
                        / math.sqrt(module.head_dim),
                        dim=-1,
                    )
                    out_test = torch.matmul(attention, vh).permute(0, 2, 1, 3)

                    if (
                        collector.query_len > 0
                        and collector.context_len > 0
                        and batch_columns > 1
                    ):
                        feature_groups = batch_columns - 1
                        key_start = collector.num_thinking_rows
                        key_end = key_start + collector.context_len
                        signature = attention[
                            :feature_groups,
                            :,
                            : collector.query_len,
                            key_start:key_end,
                        ].mean(dim=(0, 1)).detach().float()
                        if collector.signature_sum is None:
                            collector.signature_sum = torch.zeros_like(signature)
                        if collector.signature_sum.shape == signature.shape:
                            collector.signature_sum += signature
                            collector.signature_count += 1

                    out = torch.cat([out_train, out_test], dim=1)

                output = out.reshape(
                    batch_columns, rows, module.head_dim * module.num_heads
                )
                return module.out_projection(output)

            return column_forward

        for block in self.model.blocks:
            row_module = block.per_sample_attention_between_features
            column_module = block.per_column_attention_between_cells
            row_module.forward = types.MethodType(make_row_forward(), row_module)
            column_module.forward = types.MethodType(
                make_column_forward(), column_module
            )

    def probabilities_from_logits(
        self, logits: torch.Tensor, n_classes: int
    ) -> torch.Tensor:
        if logits.ndim == 3:
            logits = logits[:, 0] if logits.shape[1] == 1 else logits.mean(dim=1)
        elif logits.ndim == 1:
            logits = logits[:, None]
        logits = logits.float()
        temperature = float(
            getattr(
                self.clf,
                "softmax_temperature_",
                getattr(self.clf, "softmax_temperature", 1.0),
            )
        )
        probabilities = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
        if probabilities.shape[1] < n_classes:
            probabilities = F.pad(
                probabilities, (0, n_classes - probabilities.shape[1])
            )
        elif probabilities.shape[1] > n_classes:
            probabilities = probabilities[:, :n_classes]
        return probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def _embedding_views(self, output) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(output, dict) and {
            "train_embeddings",
            "test_embeddings",
        }.issubset(output):
            train = output["train_embeddings"].detach().float()
            test = output["test_embeddings"].detach().float()
            if train.ndim == 3:
                train = train[:, 0]
            if test.ndim == 3:
                test = test[:, 0]
            return train, test

        if self._last_block_output is None:
            raise RuntimeError(
                "TabPFN did not return embeddings and its final block hook produced no output."
            )
        states = self._last_block_output[0, :, -1].detach().float()
        train_start = self.num_thinking_rows
        train_end = train_start + self.context_len
        return states[train_start:train_end], states[train_end:]

    @torch.inference_mode()
    def extract_fusion(
        self,
        context_x: np.ndarray,
        context_y: np.ndarray,
        query_x: np.ndarray,
        n_classes: int = 2,
        stream_prompt: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        context_x = context_x.astype(np.float32, copy=False)
        query_x = query_x.astype(np.float32, copy=False)
        if len(context_x) == 0:
            query_count = len(query_x)
            feature_groups = math.ceil(
                query_x.shape[1] / self.features_per_group
            )
            return {
                "feature_adj": torch.eye(feature_groups).expand(
                    query_count, feature_groups, feature_groups
                ).clone(),
                "signatures": torch.empty(query_count, 0),
                "probs": torch.full(
                    (query_count, n_classes), 1.0 / float(n_classes)
                ),
                "context_embeddings": torch.empty(0, self.embedding_size),
                "query_embeddings": torch.zeros(
                    query_count, self.embedding_size
                ),
                "thinking_embeddings": torch.zeros(
                    self.num_thinking_rows, self.embedding_size
                ),
            }

        self.reset(len(context_x), len(query_x))
        all_x = np.concatenate([context_x, query_x], axis=0).astype(
            np.float32, copy=False
        )
        x_tensor = torch.from_numpy(all_x).to(self.device).view(len(all_x), 1, -1)
        y_tensor = torch.from_numpy(
            context_y.astype(np.float32, copy=False)
        ).to(self.device).view(len(context_y), 1)
        forward_kwargs = {
            "only_return_standard_out": False,
            "save_peak_memory_factor": None,
        }
        # The stream method itself does not require prompts.  Passing this keyword
        # only when requested preserves compatibility with unmodified TabPFN builds.
        if stream_prompt is not None:
            forward_kwargs["stream_prompt"] = stream_prompt
        output = self.model(x_tensor, y_tensor, **forward_kwargs)
        logits = output["standard"] if isinstance(output, dict) else output
        probabilities = self.probabilities_from_logits(logits, n_classes)

        if self.feature_sum is None or self.feature_count == 0:
            feature_groups = math.ceil(
                query_x.shape[1] / self.features_per_group
            )
            feature_attention = torch.eye(
                feature_groups, device=self.device
            ).expand(len(query_x), feature_groups, feature_groups).clone()
        else:
            feature_attention = self.feature_sum / float(self.feature_count)

        if self.signature_sum is None or self.signature_count == 0:
            signatures = torch.empty(len(query_x), 0, device=self.device)
        else:
            signatures = self.signature_sum / float(self.signature_count)

        train_embeddings, test_embeddings = self._embedding_views(output)
        if self._last_block_output is None:
            thinking_embeddings = torch.zeros(
                self.num_thinking_rows,
                self.embedding_size,
                device=self.device,
            )
        else:
            thinking_embeddings = self._last_block_output[
                0, : self.num_thinking_rows, -1
            ].detach().float()

        return {
            "feature_adj": normalize_adj(feature_attention).detach().clone(),
            "signatures": signatures.detach().clone(),
            "probs": probabilities.detach().clone(),
            "context_embeddings": train_embeddings.detach().clone(),
            "query_embeddings": test_embeddings.detach().clone(),
            "thinking_embeddings": thinking_embeddings.detach().clone(),
        }
