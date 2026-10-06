from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
import urllib.request
from collections import defaultdict, deque
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Sampler
from torchvision import datasets, transforms


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

HCL_COMMIT = "799349bd41f7e37b23aa03338d4f115a180e256c"
HIERARCHY_URL = (
    "https://raw.githubusercontent.com/JNNNNYao/HCL-FF/"
    f"{HCL_COMMIT}/data/hierarchy_cifar100.json"
)

CIFAR100_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR100_STD = (0.2023, 0.1994, 0.2010)

NO_SHORTCUT = 0
ADD_SHORTCUT = 1
CONCAT_SHORTCUT = 2


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic kernels can be much slower. We keep deterministic RNG
    # but allow cuDNN to pick fast kernels.
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------
# Hierarchy utilities
# ---------------------------------------------------------------------

def ensure_hierarchy_file(path: Path) -> Path:
    """Download the fixed official CIFAR-100 hierarchy once if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    print(f"[setup] Downloading fixed HCL CIFAR-100 hierarchy -> {path}")
    try:
        urllib.request.urlretrieve(HIERARCHY_URL, path)
    except Exception as e:
        raise RuntimeError(
            "Could not download hierarchy_cifar100.json.\n"
            f"URL: {HIERARCHY_URL}\n"
            "Download it manually and place it at the path shown above."
        ) from e
    return path


def json_to_hierarchy(graph_json: dict, class_order: Sequence[str]) -> Dict[int, List[List[int]]]:
    """
    Convert the HCL graph JSON into a depth-indexed partition of CIFAR-100 classes.
    Each level covers every fine class exactly once. If a branch terminates early,
    its leaves remain as singleton groups at deeper levels.
    """
    nodes = graph_json.get("nodes", [])
    links = graph_json.get("links", [])

    id_to_label = {n["id"]: n["label"] for n in nodes if "label" in n}
    label_to_id = {v: k for k, v in id_to_label.items()}

    children: Dict[str, List[str]] = defaultdict(list)
    parents: Dict[str, List[str]] = defaultdict(list)
    all_nodes = set()
    for edge in links:
        s, t = edge["source"], edge["target"]
        children[s].append(t)
        parents[t].append(s)
        all_nodes.add(s)
        all_nodes.add(t)

    class_to_idx = {name: i for i, name in enumerate(class_order)}
    leaf_ids = {label_to_id[name] for name in class_order if name in label_to_id}
    if len(leaf_ids) != len(class_order):
        missing = [name for name in class_order if name not in label_to_id]
        raise ValueError(f"Hierarchy is missing CIFAR-100 labels: {missing[:10]}")

    roots = [n for n in all_nodes if not parents[n]]
    if not roots:
        raise ValueError("Hierarchy graph has no root.")

    def descendant_leaf_count(start: str) -> int:
        q, seen = deque([start]), {start}
        count = 0
        while q:
            u = q.popleft()
            if u in leaf_ids:
                count += 1
                continue
            for v in children.get(u, []):
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        return count

    root = max(roots, key=descendant_leaf_count)

    def descendant_class_indices(start: str) -> List[int]:
        q, seen = deque([start]), {start}
        found = []
        while q:
            u = q.popleft()
            if u in leaf_ids:
                found.append(class_to_idx[id_to_label[u]])
                continue
            for v in children.get(u, []):
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        return sorted(set(found))

    nodes_by_depth: Dict[int, List[str]] = defaultdict(list)
    q, seen = deque([(root, 0)]), {root}
    max_depth = 0
    while q:
        u, depth = q.popleft()
        nodes_by_depth[depth].append(u)
        max_depth = max(max_depth, depth)
        for v in children.get(u, []):
            if v not in seen:
                seen.add(v)
                q.append((v, depth + 1))

    hierarchy: Dict[int, List[List[int]]] = {}
    all_idx = set(range(len(class_order)))
    for depth in range(1, max_depth + 1):
        groups: List[List[int]] = []
        covered = set()
        for node_id in nodes_by_depth.get(depth, []):
            group = descendant_class_indices(node_id)
            if group:
                groups.append(group)
                covered.update(group)
        for idx in sorted(all_idx - covered):
            groups.append([idx])
        if groups:
            hierarchy[depth] = groups

    if not hierarchy:
        hierarchy[1] = [[i] for i in range(len(class_order))]

    # Safety: every level must be an exact partition.
    for depth, groups in hierarchy.items():
        flat = [i for g in groups for i in g]
        if sorted(flat) != list(range(len(class_order))) or len(flat) != len(set(flat)):
            raise ValueError(f"Hierarchy level {depth} is not a valid partition.")

    return hierarchy


def balanced_layer_mapping(depth: int, n_layers: int = 17) -> List[int]:
    """
    Balanced HCL mapping: spread tree levels across all 17 layers.
    This mirrors the released HCL 'linear/balanced' intent while ensuring
    the first layer is level 1 and the last layer reaches the deepest level.
    """
    if depth <= 1:
        return [1] * n_layers
    return [
        min(depth, 1 + math.floor(i * depth / n_layers))
        for i in range(n_layers)
    ]


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

class BalancedClassBatchSampler(Sampler[List[int]]):
    """
    Small-GPU SupCon sampler.

    Each batch contains classes_per_batch classes and samples_per_class examples
    from each selected class. This guarantees positive pairs for supervised
    contrastive learning even when the physical batch is much smaller than the
    512 used in the HCL paper.
    """

    def __init__(
        self,
        labels: Sequence[int],
        classes_per_batch: int = 32,
        samples_per_class: int = 4,
        seed: int = 2222,
    ) -> None:
        self.labels = [int(x) for x in labels]
        self.classes_per_batch = int(classes_per_batch)
        self.samples_per_class = int(samples_per_class)
        self.seed = int(seed)
        self.epoch = 0

        self.by_class: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.by_class[label].append(idx)

        self.classes = sorted(self.by_class)
        if self.classes_per_batch > len(self.classes):
            raise ValueError("classes_per_batch exceeds number of classes.")
        if self.samples_per_class < 2:
            raise ValueError("samples_per_class must be >=2 for SupCon positives.")

        self.batch_size = self.classes_per_batch * self.samples_per_class
        self.num_batches = max(1, len(self.labels) // self.batch_size)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        queues: Dict[int, List[int]] = {}
        pointers: Dict[int, int] = {}

        def refill(c: int):
            q = list(self.by_class[c])
            rng.shuffle(q)
            queues[c] = q
            pointers[c] = 0

        for c in self.classes:
            refill(c)

        for _ in range(self.num_batches):
            chosen = rng.sample(self.classes, self.classes_per_batch)
            batch = []
            for c in chosen:
                need = self.samples_per_class
                while need > 0:
                    p = pointers[c]
                    q = queues[c]
                    available = len(q) - p
                    if available == 0:
                        refill(c)
                        continue
                    take = min(need, available)
                    batch.extend(q[p:p + take])
                    pointers[c] += take
                    need -= take
            rng.shuffle(batch)
            yield batch


def build_dataloaders(
    root: Path,
    seed: int,
    classes_per_batch: int,
    samples_per_class: int,
    workers: int,
    eval_batch_size: int,
):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    base_for_split = datasets.CIFAR100(root=root, train=True, download=True)
    train_full = datasets.CIFAR100(root=root, train=True, transform=train_tf, download=False)
    valid_full = datasets.CIFAR100(root=root, train=True, transform=eval_tf, download=False)
    test_set = datasets.CIFAR100(root=root, train=False, transform=eval_tf, download=True)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(base_for_split), generator=g).tolist()
    train_idx, valid_idx = perm[:45000], perm[45000:]

    train_set = Subset(train_full, train_idx)
    valid_set = Subset(valid_full, valid_idx)

    train_labels = [base_for_split.targets[i] for i in train_idx]
    batch_sampler = BalancedClassBatchSampler(
        train_labels,
        classes_per_batch=classes_per_batch,
        samples_per_class=samples_per_class,
        seed=seed,
    )

    loader_kwargs = dict(
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
    )

    train_loader = DataLoader(
        train_set,
        batch_sampler=batch_sampler,
        **loader_kwargs,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, valid_loader, test_loader, train_full.classes


# ---------------------------------------------------------------------
# Losses / heads
# ---------------------------------------------------------------------

class SupConLoss(nn.Module):
    """Stable single-view supervised contrastive loss."""

    def forward(self, features: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
        # Keep similarity math in FP32 even when convolutions use BF16.
        z = F.normalize(features.float(), dim=1)
        labels = labels.view(-1)
        n = z.size(0)

        sim = (z @ z.T) / float(temperature)
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        eye = torch.eye(n, device=z.device, dtype=torch.bool)
        same = labels[:, None].eq(labels[None, :])
        pos_mask = same & ~eye
        denom_mask = ~eye

        exp_sim = torch.exp(sim) * denom_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-12))

        pos_count = pos_mask.sum(dim=1)
        valid = pos_count > 0
        if not valid.any():
            # Should not happen with BalancedClassBatchSampler.
            return z.sum() * 0.0

        mean_log_prob_pos = (
            (log_prob * pos_mask).sum(dim=1) /
            pos_count.clamp_min(1)
        )
        return -mean_log_prob_pos[valid].mean()


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 128):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.bn(self.linear(x)), dim=1)


class DetachedFusionHead(nn.Module):
    """
    DTG-inspired detached multi-layer readout:
    concat GAP features -> BN -> dropout -> linear.
    Gradients stop at the pooled features, so the FF backbone remains layer-local.
    """
    def __init__(self, in_dim: int, num_classes: int = 100, dropout: float = 0.2):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(self.bn(x)))


# ---------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------

class CWConvLayer(nn.Module):
    """
    Channel-wise FF convolution with three paths:
      1) raw activation -> goodness
      2) class-group normalization -> propagation + SupCon
      3) raw GAP -> detached fusion readout
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_classes: int = 100,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if out_channels % num_classes != 0:
            raise ValueError("out_channels must be divisible by num_classes")

        self.num_classes = num_classes
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        self.dropout = nn.Dropout(dropout)

        if out_channels <= 100:
            self.contrast_pool = 4
        elif out_channels <= 400:
            self.contrast_pool = 2
        else:
            self.contrast_pool = 1

        self.out_channels = out_channels

    def forward(self, x: torch.Tensor):
        x = x.detach()

        raw = self.dropout(F.relu(self.conv(x), inplace=False))

        # Local class goodness: raw magnitude information is preserved here.
        goodness = raw.reshape(raw.size(0), self.num_classes, -1).mean(dim=2)

        # Strict class-subset goodness decoupling for inter-layer propagation.
        prop = F.group_norm(raw, num_groups=self.num_classes)

        contrast_feat = F.adaptive_avg_pool2d(
            prop, (self.contrast_pool, self.contrast_pool)
        ).flatten(1)

        # Separate readout path; detached later before the fusion classifier.
        fusion_feat = F.adaptive_avg_pool2d(raw, 1).flatten(1)

        return prop, goodness, contrast_feat, fusion_feat


class HCLFusionFF(nn.Module):
    def __init__(
        self,
        hierarchy: Dict[int, List[List[int]]],
        epochs: int,
        lr: float = 0.04,
        lr_min: float = 2e-4,
        weight_decay: float = 1e-4,
        projection_dim: int = 128,
        backbone_dropout: float = 0.0,
        fusion_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_classes = 100
        self.epochs = epochs
        self.current_epoch = 0

        widths = [100, 200, 400, 800]
        specs = [
            # index, in, out, stride
            (3,   100, 1),   # 0
            (100, 100, 1),   # 1
            (100, 100, 1),   # 2
            (100, 100, 1),   # 3
            (100, 100, 1),   # 4 -> concat => 200
            (200, 200, 2),   # 5
            (200, 200, 1),   # 6
            (200, 200, 1),   # 7
            (200, 200, 1),   # 8 -> concat => 400
            (400, 400, 2),   # 9
            (400, 400, 1),   # 10
            (400, 400, 1),   # 11
            (400, 400, 1),   # 12 -> concat => 800
            (800, 800, 1),   # 13
            (800, 800, 1),   # 14
            (800, 800, 1),   # 15
            (800, 800, 1),   # 16
        ]
        self.layers = nn.ModuleList([
            CWConvLayer(inp, out, self.num_classes, stride=stride, dropout=backbone_dropout)
            for inp, out, stride in specs
        ])

        # Residual behavior follows the 17-layer HCL/DeeperForward layout.
        self.capture_shortcut = [
            True, False, True, False, True,
            False, True, False, True,
            False, True, False, True,
            False, True, False, True,
        ]
        self.shortcut_mode = [
            NO_SHORTCUT, NO_SHORTCUT, ADD_SHORTCUT, NO_SHORTCUT, CONCAT_SHORTCUT,
            NO_SHORTCUT, ADD_SHORTCUT, NO_SHORTCUT, CONCAT_SHORTCUT,
            NO_SHORTCUT, ADD_SHORTCUT, NO_SHORTCUT, CONCAT_SHORTCUT,
            NO_SHORTCUT, ADD_SHORTCUT, NO_SHORTCUT, ADD_SHORTCUT,
        ]
        self.shortcut_downsample = [
            False, False, False, False, True,
            False, False, False, True,
            False, False, False, False,
            False, False, False, False,
        ]

        self.proj_heads = nn.ModuleList()
        for layer in self.layers:
            dim = layer.out_channels * layer.contrast_pool * layer.contrast_pool
            self.proj_heads.append(ProjectionHead(dim, projection_dim))

        # HCL final detached classifier.
        self.final_head = nn.Linear(800, self.num_classes)

        # DTG-inspired multi-layer detached fusion: exclude only layer 0.
        fusion_dim = sum(layer.out_channels for layer in self.layers[1:])
        self.fusion_head = DetachedFusionHead(
            fusion_dim, self.num_classes, dropout=fusion_dropout
        )

        # Hierarchy maps.
        self.hierarchy = hierarchy
        depth = max(hierarchy)
        self.layer_to_level = balanced_layer_mapping(depth, len(self.layers))
        print(f"[model] hierarchy depth={depth}")
        print(f"[model] balanced layer mapping={self.layer_to_level}")

        self._prepare_hierarchy_buffers()

        # Per-layer local optimizers.
        self.layer_optimizers = [
            torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=weight_decay)
            for m in self.layers
        ]
        self.proj_optimizers = [
            torch.optim.AdamW(m.parameters(), lr=lr * 2.0, weight_decay=weight_decay)
            for m in self.proj_heads
        ]
        self.final_optimizer = torch.optim.AdamW(
            self.final_head.parameters(), lr=lr, weight_decay=weight_decay
        )
        # DTG paper head recipe.
        self.fusion_optimizer = torch.optim.AdamW(
            self.fusion_head.parameters(), lr=2e-4, weight_decay=1e-3
        )

        self.layer_schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=epochs, eta_min=lr_min)
            for o in self.layer_optimizers
        ]
        self.proj_schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=epochs, eta_min=lr_min * 2.0)
            for o in self.proj_optimizers
        ]
        self.final_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.final_optimizer, T_max=epochs, eta_min=lr_min
        )
        self.fusion_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.fusion_optimizer, T_max=epochs, eta_min=1e-5
        )

        self.supcon = SupConLoss()

    def _prepare_hierarchy_buffers(self) -> None:
        for level, groups in self.hierarchy.items():
            lut = torch.empty(self.num_classes, dtype=torch.long)
            sizes = torch.empty(len(groups), dtype=torch.float32)
            for gid, group in enumerate(groups):
                sizes[gid] = len(group)
                for class_idx in group:
                    lut[class_idx] = gid
            self.register_buffer(f"hier_lut_{level}", lut, persistent=False)
            self.register_buffer(f"hier_sizes_{level}", sizes, persistent=False)

    def hierarchy_group_logits(
        self, fine_logits: torch.Tensor, labels: torch.Tensor, level: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lut = getattr(self, f"hier_lut_{level}")
        sizes = getattr(self, f"hier_sizes_{level}")
        grouped = fine_logits.new_zeros(fine_logits.size(0), sizes.numel())
        grouped.scatter_add_(
            1,
            lut.view(1, -1).expand(fine_logits.size(0), -1),
            fine_logits,
        )
        grouped = grouped / sizes.view(1, -1)
        mapped_labels = lut[labels]
        return grouped, mapped_labels

    @staticmethod
    def _apply_shortcut(
        x: torch.Tensor,
        shortcut: Optional[torch.Tensor],
        mode: int,
        num_classes: int,
    ) -> torch.Tensor:
        if mode == NO_SHORTCUT:
            return x
        if shortcut is None:
            raise RuntimeError("Shortcut requested before one was captured.")
        if mode == ADD_SHORTCUT:
            return x + shortcut
        if mode == CONCAT_SHORTCUT:
            b, _, h, w = x.shape
            xg = x.view(b, num_classes, -1, h, w)
            sg = shortcut.view(b, num_classes, -1, h, w)
            return torch.cat([xg, sg], dim=2).flatten(1, 2)
        raise ValueError(mode)

    def _capture(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if self.shortcut_downsample[layer_idx]:
            x = F.avg_pool2d(x, 2, 2)
        return x.detach()

    def supcon_temperature(self, epoch: int) -> float:
        # Paper appendix: 0.8 -> 0.2 for first 100 epochs, then cosine -> 0.08.
        warm = min(100, max(1, self.epochs))
        if epoch < warm:
            frac = epoch / max(1, warm - 1)
            return 0.8 + (0.2 - 0.8) * frac

        remain = max(1, self.epochs - warm)
        frac = min(1.0, (epoch - warm) / remain)
        return 0.08 + 0.5 * (0.2 - 0.08) * (1.0 + math.cos(math.pi * frac))

    def amp_context(self, device: torch.device, use_bf16: bool):
        if device.type == "cuda" and use_bf16:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def train_epoch(
        self,
        loader: DataLoader,
        device: torch.device,
        epoch: int,
        use_bf16: bool,
        grad_clip: float = 1.0,
        max_steps: int = 0,
    ) -> Dict[str, float]:
        self.train()
        tau = self.supcon_temperature(epoch)

        local_ce_total = 0.0
        supcon_total = 0.0
        final_total = 0.0
        fusion_total = 0.0
        n_batches = 0

        for batch_idx, (images, labels) in enumerate(loader):
            if max_steps and batch_idx >= max_steps:
                break

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            x = F.layer_norm(images, images.shape[1:])
            shortcut: Optional[torch.Tensor] = None
            fusion_features: List[torch.Tensor] = []

            for i, (layer, proj) in enumerate(zip(self.layers, self.proj_heads)):
                self.layer_optimizers[i].zero_grad(set_to_none=True)
                self.proj_optimizers[i].zero_grad(set_to_none=True)

                with self.amp_context(device, use_bf16):
                    prop, fine_goodness, contrast_feat, fusion_feat = layer(x)
                    embedding = proj(contrast_feat)

                level = self.layer_to_level[i]
                grouped, mapped = self.hierarchy_group_logits(
                    fine_goodness.float(), labels, level
                )
                ce = F.cross_entropy(grouped, mapped)
                con = self.supcon(embedding, labels, tau)
                local_loss = ce + con

                local_loss.backward()
                torch.nn.utils.clip_grad_norm_(layer.parameters(), grad_clip)
                torch.nn.utils.clip_grad_norm_(proj.parameters(), grad_clip)
                self.layer_optimizers[i].step()
                self.proj_optimizers[i].step()

                local_ce_total += float(ce.detach())
                supcon_total += float(con.detach())

                # Raw GAP readout is deliberately detached.
                fusion_features.append(fusion_feat.detach())

                # Propagation path is detached at every boundary.
                x = self._apply_shortcut(
                    prop.detach(), shortcut, self.shortcut_mode[i], self.num_classes
                )
                if self.capture_shortcut[i]:
                    shortcut = self._capture(x, i)

                del prop, fine_goodness, contrast_feat, embedding, grouped, mapped
                del ce, con, local_loss

            # Final-layer head (HCL-style detached head).
            self.final_optimizer.zero_grad(set_to_none=True)
            final_feat = F.adaptive_avg_pool2d(x, 1).flatten(1).detach()
            final_logits = self.final_head(final_feat)
            final_loss = F.cross_entropy(final_logits.float(), labels)
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.final_head.parameters(), grad_clip)
            self.final_optimizer.step()
            final_total += float(final_loss.detach())

            # Multi-layer detached fusion head.
            self.fusion_optimizer.zero_grad(set_to_none=True)
            fused = torch.cat(fusion_features[1:], dim=1).detach().float()
            fusion_logits = self.fusion_head(fused)
            fusion_loss = F.cross_entropy(
                fusion_logits.float(), labels, label_smoothing=0.1
            )
            fusion_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.fusion_head.parameters(), grad_clip)
            self.fusion_optimizer.step()
            fusion_total += float(fusion_loss.detach())

            n_batches += 1

        for s in self.layer_schedulers:
            s.step()
        for s in self.proj_schedulers:
            s.step()
        self.final_scheduler.step()
        self.fusion_scheduler.step()

        denom = max(1, n_batches)
        layer_denom = denom * len(self.layers)
        return {
            "tau": tau,
            "local_ce": local_ce_total / layer_denom,
            "supcon": supcon_total / layer_denom,
            "final_ce": final_total / denom,
            "fusion_ce": fusion_total / denom,
            "lr": self.layer_optimizers[0].param_groups[0]["lr"],
        }

    @torch.no_grad()
    def forward_components(
        self,
        images: torch.Tensor,
        use_bf16: bool,
    ):
        device = images.device
        self.eval()
        x = F.layer_norm(images, images.shape[1:])
        shortcut: Optional[torch.Tensor] = None
        goodness_by_layer: List[torch.Tensor] = []
        fusion_features: List[torch.Tensor] = []

        with self.amp_context(device, use_bf16):
            for i, layer in enumerate(self.layers):
                prop, goodness, _, fusion_feat = layer(x)
                goodness_by_layer.append(goodness.float())
                fusion_features.append(fusion_feat.float())

                x = self._apply_shortcut(
                    prop, shortcut, self.shortcut_mode[i], self.num_classes
                )
                if self.capture_shortcut[i]:
                    shortcut = self._capture(x, i)

            final_feat = F.adaptive_avg_pool2d(x, 1).flatten(1)
            final_logits = self.final_head(final_feat).float()

            fused = torch.cat(fusion_features[1:], dim=1)
            fusion_logits = self.fusion_head(fused).float()

        return goodness_by_layer, final_logits, fusion_logits

    def optimizer_state(self) -> dict:
        return {
            "layer_optimizers": [o.state_dict() for o in self.layer_optimizers],
            "proj_optimizers": [o.state_dict() for o in self.proj_optimizers],
            "final_optimizer": self.final_optimizer.state_dict(),
            "fusion_optimizer": self.fusion_optimizer.state_dict(),
            "layer_schedulers": [s.state_dict() for s in self.layer_schedulers],
            "proj_schedulers": [s.state_dict() for s in self.proj_schedulers],
            "final_scheduler": self.final_scheduler.state_dict(),
            "fusion_scheduler": self.fusion_scheduler.state_dict(),
        }

    def load_optimizer_state(self, state: dict) -> None:
        for o, s in zip(self.layer_optimizers, state["layer_optimizers"]):
            o.load_state_dict(s)
        for o, s in zip(self.proj_optimizers, state["proj_optimizers"]):
            o.load_state_dict(s)
        self.final_optimizer.load_state_dict(state["final_optimizer"])
        self.fusion_optimizer.load_state_dict(state["fusion_optimizer"])
        for sch, s in zip(self.layer_schedulers, state["layer_schedulers"]):
            sch.load_state_dict(s)
        for sch, s in zip(self.proj_schedulers, state["proj_schedulers"]):
            sch.load_state_dict(s)
        self.final_scheduler.load_state_dict(state["final_scheduler"])
        self.fusion_scheduler.load_state_dict(state["fusion_scheduler"])


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(
    model: HCLFusionFF,
    loader: DataLoader,
    device: torch.device,
    use_bf16: bool,
):
    all_labels = []
    all_goodness = [[] for _ in range(len(model.layers))]
    all_final = []
    all_fusion = []

    model.eval()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        g_layers, final_logits, fusion_logits = model.forward_components(
            images, use_bf16
        )

        all_labels.append(labels.cpu())
        for i, g in enumerate(g_layers):
            all_goodness[i].append(g.cpu())
        all_final.append(final_logits.cpu())
        all_fusion.append(fusion_logits.cpu())

    labels = torch.cat(all_labels)
    goodness = [torch.cat(parts) for parts in all_goodness]
    final_logits = torch.cat(all_final)
    fusion_logits = torch.cat(all_fusion)
    return labels, goodness, final_logits, fusion_logits


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return 100.0 * (logits.argmax(1) == labels).float().mean().item()


def choose_goodness_interval(
    goodness: List[torch.Tensor],
    labels: torch.Tensor,
) -> Tuple[Tuple[int, int], torch.Tensor, float]:
    L = len(goodness)
    best_acc = -1.0
    best_interval = (1, L - 1)
    best_logits = None

    # Layer 0 is excluded by default, following the HCL/DeeperForward pruning tendency.
    for start in range(1, L):
        running = torch.zeros_like(goodness[0])
        for end in range(start, L):
            running = running + goodness[end]
            acc = accuracy(running, labels)
            if acc > best_acc:
                best_acc = acc
                best_interval = (start, end)
                best_logits = running.clone()

    assert best_logits is not None
    return best_interval, best_logits, best_acc


def goodness_logits_for_interval(
    goodness: List[torch.Tensor],
    interval: Tuple[int, int],
) -> torch.Tensor:
    s, e = interval
    out = torch.zeros_like(goodness[0])
    for i in range(s, e + 1):
        out = out + goodness[i]
    return out


def calibrate_ensemble(
    goodness_logits: torch.Tensor,
    final_logits: torch.Tensor,
    fusion_logits: torch.Tensor,
    labels: torch.Tensor,
):
    pg = F.softmax(goodness_logits, dim=1)
    ph = F.softmax(final_logits, dim=1)
    pf = F.softmax(fusion_logits, dim=1)

    # Small validation-only grid. No test labels are used to choose weights.
    g_weights = [0.5, 1.0, 1.5, 2.0]
    f_weights = [0.5, 1.0, 1.5, 2.0]
    h_weights = [0.0, 0.5, 1.0]

    best = (-1.0, (1.0, 0.0, 1.0))
    for wg in g_weights:
        for wh in h_weights:
            for wf in f_weights:
                probs = wg * pg + wh * ph + wf * pf
                acc = accuracy(probs, labels)
                if acc > best[0]:
                    best = (acc, (wg, wh, wf))
    return best


def evaluate_with_selection(
    model: HCLFusionFF,
    valid_loader: DataLoader,
    device: torch.device,
    use_bf16: bool,
):
    labels, goodness, final_logits, fusion_logits = collect_predictions(
        model, valid_loader, device, use_bf16
    )
    interval, g_logits, g_acc = choose_goodness_interval(goodness, labels)
    final_acc = accuracy(final_logits, labels)
    fusion_acc = accuracy(fusion_logits, labels)
    ensemble_acc, weights = calibrate_ensemble(
        g_logits, final_logits, fusion_logits, labels
    )

    return {
        "interval": interval,
        "weights": weights,
        "goodness_acc": g_acc,
        "final_acc": final_acc,
        "fusion_acc": fusion_acc,
        "ensemble_acc": ensemble_acc,
    }


def evaluate_fixed(
    model: HCLFusionFF,
    loader: DataLoader,
    device: torch.device,
    use_bf16: bool,
    interval: Tuple[int, int],
    weights: Tuple[float, float, float],
):
    labels, goodness, final_logits, fusion_logits = collect_predictions(
        model, loader, device, use_bf16
    )
    g_logits = goodness_logits_for_interval(goodness, interval)

    wg, wh, wf = weights
    ensemble = (
        wg * F.softmax(g_logits, dim=1)
        + wh * F.softmax(final_logits, dim=1)
        + wf * F.softmax(fusion_logits, dim=1)
    )
    return {
        "goodness_acc": accuracy(g_logits, labels),
        "final_acc": accuracy(final_logits, labels),
        "fusion_acc": accuracy(fusion_logits, labels),
        "ensemble_acc": accuracy(ensemble, labels),
    }


# ---------------------------------------------------------------------
# Checkpointing / CLI
# ---------------------------------------------------------------------

@dataclass
class RunConfig:
    epochs: int
    lr: float
    lr_min: float
    weight_decay: float
    seed: int
    classes_per_batch: int
    samples_per_class: int
    workers: int
    eval_batch_size: int
    eval_every: int
    grad_clip: float
    bf16: bool
    max_steps: int


def save_checkpoint(
    path: Path,
    model: HCLFusionFF,
    epoch: int,
    best_val: float,
    best_interval: Tuple[int, int],
    best_weights: Tuple[float, float, float],
    config: RunConfig,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": model.optimizer_state(),
        "best_val": best_val,
        "best_interval": best_interval,
        "best_weights": best_weights,
        "config": asdict(config),
    }, path)


def load_checkpoint(path: Path, model: HCLFusionFF, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.load_optimizer_state(ckpt["optimizer"])
    return ckpt


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args():
    p = argparse.ArgumentParser(
        description="HCL-style Forward-Forward + detached multi-layer fusion for CIFAR-100"
    )
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.04)
    p.add_argument("--lr-min", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=2222)

    # 32 x 4 = physical batch 128: intended starting point for RTX 4050 6 GB.
    p.add_argument("--classes-per-batch", type=int, default=32)
    p.add_argument("--samples-per-class", type=int, default=4)

    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--data-dir", type=Path, default=Path("./datasets"))
    p.add_argument("--hierarchy-json", type=Path, default=Path("./assets/hierarchy_cifar100.json"))
    p.add_argument("--run-dir", type=Path, default=Path("./runs/hcl_fusion_4050"))

    p.add_argument(
        "--no-bf16",
        action="store_true",
        help="Disable BF16 autocast. RTX 4050 should normally support BF16.",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Limit train batches per epoch for smoke tests only. 0 = full epoch.",
    )
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This run is intended for your RTX 4050.")

    device = torch.device("cuda:0")
    print(f"[device] {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"[device] VRAM: {props.total_memory / 1024**3:.2f} GB")

    bf16_supported = bool(torch.cuda.is_bf16_supported())
    use_bf16 = (not args.no_bf16) and bf16_supported
    print(f"[device] BF16 autocast: {use_bf16}")

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    hierarchy_path = ensure_hierarchy_file(args.hierarchy_json)

    train_loader, valid_loader, test_loader, class_order = build_dataloaders(
        root=args.data_dir,
        seed=args.seed,
        classes_per_batch=args.classes_per_batch,
        samples_per_class=args.samples_per_class,
        workers=args.workers,
        eval_batch_size=args.eval_batch_size,
    )

    with hierarchy_path.open("r", encoding="utf-8") as f:
        graph = json.load(f)
    hierarchy = json_to_hierarchy(graph, class_order)

    config = RunConfig(
        epochs=args.epochs,
        lr=args.lr,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        seed=args.seed,
        classes_per_batch=args.classes_per_batch,
        samples_per_class=args.samples_per_class,
        workers=args.workers,
        eval_batch_size=args.eval_batch_size,
        eval_every=args.eval_every,
        grad_clip=args.grad_clip,
        bf16=use_bf16,
        max_steps=args.max_steps,
    )

    model = HCLFusionFF(
        hierarchy=hierarchy,
        epochs=args.epochs,
        lr=args.lr,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
    ).to(device)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )

    last_path = args.run_dir / "last.pt"
    best_path = args.run_dir / "best.pt"
    log_path = args.run_dir / "metrics.csv"

    start_epoch = 0
    best_val = -1.0
    best_interval = (1, 16)
    best_weights = (1.0, 0.0, 1.0)

    if last_path.exists() and not args.no_resume:
        ckpt = load_checkpoint(last_path, model, device)
        start_epoch = int(ckpt["epoch"]) + 1
        best_val = float(ckpt.get("best_val", -1.0))
        best_interval = tuple(ckpt.get("best_interval", best_interval))
        best_weights = tuple(ckpt.get("best_weights", best_weights))
        print(f"[resume] epoch {start_epoch}/{args.epochs}")

    print(
        f"[train] physical batch = "
        f"{args.classes_per_batch} classes x {args.samples_per_class} samples "
        f"= {args.classes_per_batch * args.samples_per_class}"
    )

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_metrics = model.train_epoch(
            train_loader,
            device,
            epoch,
            use_bf16,
            grad_clip=args.grad_clip,
            max_steps=args.max_steps,
        )

        do_eval = (
            epoch == 0
            or (epoch + 1) % args.eval_every == 0
            or epoch + 1 == args.epochs
        )

        row = {
            "epoch": epoch + 1,
            "tau": train_metrics["tau"],
            "lr": train_metrics["lr"],
            "local_ce": train_metrics["local_ce"],
            "supcon": train_metrics["supcon"],
            "final_ce": train_metrics["final_ce"],
            "fusion_ce": train_metrics["fusion_ce"],
            "val_goodness": "",
            "val_final": "",
            "val_fusion": "",
            "val_ensemble": "",
            "interval": "",
            "weights": "",
            "minutes": (time.time() - t0) / 60.0,
        }

        if do_eval:
            val = evaluate_with_selection(model, valid_loader, device, use_bf16)
            row.update({
                "val_goodness": f"{val['goodness_acc']:.4f}",
                "val_final": f"{val['final_acc']:.4f}",
                "val_fusion": f"{val['fusion_acc']:.4f}",
                "val_ensemble": f"{val['ensemble_acc']:.4f}",
                "interval": str(val["interval"]),
                "weights": str(val["weights"]),
            })

            print(
                f"[epoch {epoch+1:04d}/{args.epochs}] "
                f"val goodness={val['goodness_acc']:.2f}% | "
                f"fusion={val['fusion_acc']:.2f}% | "
                f"ensemble={val['ensemble_acc']:.2f}% | "
                f"interval={val['interval']} weights={val['weights']} | "
                f"{row['minutes']:.1f} min"
            )

            if val["ensemble_acc"] > best_val:
                best_val = val["ensemble_acc"]
                best_interval = tuple(val["interval"])
                best_weights = tuple(val["weights"])
                save_checkpoint(
                    best_path,
                    model,
                    epoch,
                    best_val,
                    best_interval,
                    best_weights,
                    config,
                )
                print(f"[best] new validation best = {best_val:.2f}%")
        else:
            print(
                f"[epoch {epoch+1:04d}/{args.epochs}] "
                f"localCE={train_metrics['local_ce']:.4f} "
                f"SupCon={train_metrics['supcon']:.4f} "
                f"fusionCE={train_metrics['fusion_ce']:.4f} | "
                f"{row['minutes']:.1f} min"
            )

        append_csv(log_path, row)

        save_checkpoint(
            last_path,
            model,
            epoch,
            best_val,
            best_interval,
            best_weights,
            config,
        )

    # Final test is performed ONCE using validation-selected checkpoint/interval/weights.
    if not best_path.exists():
        raise RuntimeError("No best checkpoint was created.")

    best_ckpt = load_checkpoint(best_path, model, device)
    interval = tuple(best_ckpt["best_interval"])
    weights = tuple(best_ckpt["best_weights"])

    test = evaluate_fixed(
        model,
        test_loader,
        device,
        use_bf16,
        interval=interval,
        weights=weights,
    )

    print("\n================ FINAL HELD-OUT TEST ================")
    print(f"Best validation ensemble: {best_ckpt['best_val']:.2f}%")
    print(f"Goodness interval: {interval}")
    print(f"Ensemble weights (goodness, final, fusion): {weights}")
    print(f"Test goodness: {test['goodness_acc']:.2f}%")
    print(f"Test final-head: {test['final_acc']:.2f}%")
    print(f"Test fusion-head: {test['fusion_acc']:.2f}%")
    print(f"Test ensemble: {test['ensemble_acc']:.2f}%")
    print("=====================================================")

    (args.run_dir / "final_test.json").write_text(
        json.dumps({
            "best_val": best_ckpt["best_val"],
            "interval": interval,
            "weights": weights,
            **test,
        }, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
