#!/usr/bin/env python3
"""
HyTra Latency Data-Collection Pipeline
=======================================

Pipeline hoàn chỉnh cho Khóa luận tốt nghiệp — Hardware-aware NAS kết hợp
không gian tìm kiếm HyTra (BossNAS) và bộ dự đoán độ trễ MHLP.

Bốn tác vụ (Tasks):
  1. generate_random_hytra()      — Sinh mẫu kiến trúc HyTra hợp lệ
  2. measure_latency()            — Đo độ trễ chuẩn mực trên nhiều thiết bị
  3. pdwrs_sample()               — Lọc mẫu đại diện bằng PDWRS (KDE-based)
  4. main()                       — Pipeline ghép nối, xuất CSV

Dữ liệu đầu ra `hytra_latency_dataset.csv` được dùng trực tiếp cho MHLP
DataLoader (xem hytra_encoding.py & mhlp_predictor.py).

Usage:
    python hytra_latency_pipeline.py
    python hytra_latency_pipeline.py --num_generate 2000 --num_select 500
    python hytra_latency_pipeline.py --device cpu --warmup 10 --runs 30

"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# ── Đường dẫn dự án ──────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "searching"))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "OpenSelfSup"))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CONSTANTS — Không gian tìm kiếm HyTra                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# -- 6 toán tử: op_index → (block_type, scale_ratio) ----------------------
OP_INDEX_TO_TUPLE: Dict[int, Tuple[str, float]] = {
    0: ("ResAtt",  1 / 32),   # depth 0 — 7×7    — 2048 ch
    1: ("ResConv", 1 / 32),   # depth 0 — 7×7    — 2048 ch
    2: ("ResAtt",  1 / 16),   # depth 1 — 14×14  — 1024 ch
    3: ("ResConv", 1 / 16),   # depth 1 — 14×14  — 1024 ch
    4: ("ResConv", 1 / 8),    # depth 2 — 28×28  — 512  ch
    5: ("ResConv", 1 / 4),    # depth 3 — 56×56  — 256  ch
}

TUPLE_TO_OP_INDEX: Dict[Tuple[str, float], int] = {
    v: k for k, v in OP_INDEX_TO_TUPLE.items()
}

# -- Ràng buộc ResAtt: chỉ ở scale 1/16 và 1/32 --------------------------
RESATT_ALLOWED_SCALES: set = {1 / 16, 1 / 32}

# -- Scales hợp lệ cho choice-block (không tính scale 1 = input) ----------
VALID_BLOCK_SCALES: List[float] = [1 / 4, 1 / 8, 1 / 16, 1 / 32]

# -- Cấu trúc Stage -------------------------------------------------------
NUM_STAGES: int = 4
LAYERS_PER_STAGE: int = 4            # 4 layers mỗi stage → 16 layers tổng
NUM_CHOICE_BLOCKS: int = NUM_STAGES * LAYERS_PER_STAGE   # 16
STAGE_DEPTHS: List[int] = [4, 3, 2, 2]

# s_rank[depth] → list op indices tại depth đó
S_RANK: Dict[int, List[int]] = {0: [0, 1], 1: [2, 3], 2: [4], 3: [5]}

# -- Restricted Paths theo từng depth (đã xác minh từ hytra_paths.py) -----
RESTRICTED_PATHS: Dict[int, List[List[int]]] = {
    4: [
        [4, 2, 0, 0], [4, 2, 0, 1], [4, 2, 1, 0], [4, 2, 1, 1],
        [4, 3, 0, 0], [4, 3, 0, 1], [4, 3, 1, 0], [4, 3, 1, 1],
        [4, 2, 2, 0], [4, 2, 2, 1], [4, 2, 3, 0], [4, 2, 3, 1],
        [4, 3, 2, 0], [4, 3, 2, 1], [4, 3, 3, 0], [4, 3, 3, 1],
        [4, 2, 2, 2], [4, 2, 2, 3], [4, 2, 3, 2], [4, 2, 3, 3],
        [4, 3, 2, 2], [4, 3, 2, 3], [4, 3, 3, 2], [4, 3, 3, 3],
        [4, 4, 2, 0], [4, 4, 2, 1], [4, 4, 3, 0], [4, 4, 3, 1],
        [4, 4, 2, 2], [4, 4, 2, 3], [4, 4, 3, 2], [4, 4, 3, 3],
        [4, 4, 4, 2], [4, 4, 4, 3], [4, 4, 4, 4],
        [5, 4, 2, 0], [5, 4, 2, 1], [5, 4, 3, 0], [5, 4, 3, 1],
        [5, 4, 2, 2], [5, 4, 2, 3], [5, 4, 3, 2], [5, 4, 3, 3],
        [5, 4, 4, 2], [5, 4, 4, 3], [5, 4, 4, 4],
        [5, 5, 4, 2], [5, 5, 4, 3], [5, 5, 4, 4],
        [5, 5, 5, 4], [5, 5, 5, 5],
    ],
    3: [
        [2, 0, 0, 0], [2, 0, 0, 1], [2, 0, 1, 0], [2, 0, 1, 1],
        [2, 1, 0, 0], [2, 1, 0, 1], [2, 1, 1, 0], [2, 1, 1, 1],
        [3, 0, 0, 0], [3, 0, 0, 1], [3, 0, 1, 0], [3, 0, 1, 1],
        [3, 1, 0, 0], [3, 1, 0, 1], [3, 1, 1, 0], [3, 1, 1, 1],
        [2, 2, 0, 0], [2, 2, 0, 1], [2, 2, 1, 0], [2, 2, 1, 1],
        [2, 3, 0, 0], [2, 3, 0, 1], [2, 3, 1, 0], [2, 3, 1, 1],
        [3, 2, 0, 0], [3, 2, 0, 1], [3, 2, 1, 0], [3, 2, 1, 1],
        [3, 3, 0, 0], [3, 3, 0, 1], [3, 3, 1, 0], [3, 3, 1, 1],
        [2, 2, 2, 0], [2, 2, 2, 1], [2, 2, 3, 0], [2, 2, 3, 1],
        [2, 3, 2, 0], [2, 3, 2, 1], [2, 3, 3, 0], [2, 3, 3, 1],
        [3, 2, 2, 0], [3, 2, 2, 1], [3, 2, 3, 0], [3, 2, 3, 1],
        [3, 3, 2, 0], [3, 3, 2, 1], [3, 3, 3, 0], [3, 3, 3, 1],
        [2, 2, 2, 2], [2, 2, 2, 3], [2, 2, 3, 2], [2, 2, 3, 3],
        [2, 3, 2, 2], [2, 3, 2, 3], [2, 3, 3, 2], [2, 3, 3, 3],
        [3, 2, 2, 2], [3, 2, 2, 3], [3, 2, 3, 2], [3, 2, 3, 3],
        [3, 3, 2, 2], [3, 3, 2, 3], [3, 3, 3, 2], [3, 3, 3, 3],
        [4, 2, 0, 0], [4, 2, 0, 1], [4, 2, 1, 0], [4, 2, 1, 1],
        [4, 3, 0, 0], [4, 3, 0, 1], [4, 3, 1, 0], [4, 3, 1, 1],
        [4, 2, 2, 0], [4, 2, 2, 1], [4, 2, 3, 0], [4, 2, 3, 1],
        [4, 3, 2, 0], [4, 3, 2, 1], [4, 3, 3, 0], [4, 3, 3, 1],
        [4, 2, 2, 2], [4, 2, 2, 3], [4, 2, 3, 2], [4, 2, 3, 3],
        [4, 3, 2, 2], [4, 3, 2, 3], [4, 3, 3, 2], [4, 3, 3, 3],
        [4, 4, 2, 0], [4, 4, 2, 1], [4, 4, 3, 0], [4, 4, 3, 1],
        [4, 4, 2, 2], [4, 4, 2, 3], [4, 4, 3, 2], [4, 4, 3, 3],
        [4, 4, 4, 2], [4, 4, 4, 3], [4, 4, 4, 4],
    ],
    2: [
        [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
        [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
        [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
        [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1],
        [2, 0, 0, 0], [2, 0, 0, 1], [2, 0, 1, 0], [2, 0, 1, 1],
        [2, 1, 0, 0], [2, 1, 0, 1], [2, 1, 1, 0], [2, 1, 1, 1],
        [3, 0, 0, 0], [3, 0, 0, 1], [3, 0, 1, 0], [3, 0, 1, 1],
        [3, 1, 0, 0], [3, 1, 0, 1], [3, 1, 1, 0], [3, 1, 1, 1],
        [2, 2, 0, 0], [2, 2, 0, 1], [2, 2, 1, 0], [2, 2, 1, 1],
        [2, 3, 0, 0], [2, 3, 0, 1], [2, 3, 1, 0], [2, 3, 1, 1],
        [3, 2, 0, 0], [3, 2, 0, 1], [3, 2, 1, 0], [3, 2, 1, 1],
        [3, 3, 0, 0], [3, 3, 0, 1], [3, 3, 1, 0], [3, 3, 1, 1],
        [2, 2, 2, 0], [2, 2, 2, 1], [2, 2, 3, 0], [2, 2, 3, 1],
        [2, 3, 2, 0], [2, 3, 2, 1], [2, 3, 3, 0], [2, 3, 3, 1],
        [3, 2, 2, 0], [3, 2, 2, 1], [3, 2, 3, 0], [3, 2, 3, 1],
        [3, 3, 2, 0], [3, 3, 2, 1], [3, 3, 3, 0], [3, 3, 3, 1],
        [2, 2, 2, 2], [2, 2, 2, 3], [2, 2, 3, 2], [2, 2, 3, 3],
        [2, 3, 2, 2], [2, 3, 2, 3], [2, 3, 3, 2], [2, 3, 3, 3],
        [3, 2, 2, 2], [3, 2, 2, 3], [3, 2, 3, 2], [3, 2, 3, 3],
        [3, 3, 2, 2], [3, 3, 2, 3], [3, 3, 3, 2], [3, 3, 3, 3],
    ],
    1: [
        [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
        [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
        [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
        [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1],
    ],
}

# -- Hệ số giả lập độ trễ cho thiết bị khác (so với RTX 4080) -------------
#    Các hệ số này được ước lượng từ dữ liệu thực (xem mhlp_data_preprocessing.py)
DEVICE_LATENCY_SCALE: Dict[str, float] = {
    "RTX4080":    1.0,    # Baseline
    "Macbook_M1": 1.8,    # ~1.8× chậm hơn RTX 4080
    "ip15":       3.2,    # ~3.2× chậm hơn RTX 4080
}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TÁC VỤ 1: TRÌNH SINH KIẾN TRÚC HYTRA                                    ║
# ║  (HyTra Architecture Generator)                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _op_indices_to_tuples(op_indices: List[List[int]]) -> List[Tuple[str, float]]:
    """
    Chuyển từ BossNAS op-index format sang tuple format.

    Args:
        op_indices: [[op0,op1,op2,op3], ...] — 4 stages × 4 layers.

    Returns:
        List 16 tuples ``(block_type, scale_ratio)``.

    Raises:
        KeyError: nếu op index không tồn tại.

    Example:
        >>> _op_indices_to_tuples([[4,2,0,1],[3,3,2,0],[1,0,0,1],[3,3,0,0]])
        [('ResConv', 0.125), ('ResAtt', 0.0625), ...]
    """
    tuples: List[Tuple[str, float]] = []
    for stage_ops in op_indices:
        for op in stage_ops:
            tuples.append(OP_INDEX_TO_TUPLE[op])
    return tuples


def _tuples_to_arch_string(arch: List[Tuple[str, float]]) -> str:
    """
    Chuyển list tuples thành chuỗi deterministric dùng làm key / lưu CSV.

    Format: ``"ResConv@0.125|ResAtt@0.0625|..."`` — 16 phần tử, nối bằng ``|``.
    """
    return "|".join(f"{bt}@{sr}" for bt, sr in arch)


def _arch_string_to_tuples(s: str) -> List[Tuple[str, float]]:
    """Nghịch đảo của ``_tuples_to_arch_string``."""
    parts = s.split("|")
    tuples: List[Tuple[str, float]] = []
    for p in parts:
        bt, sr = p.split("@")
        tuples.append((bt, float(sr)))
    return tuples


def generate_random_hytra(seed: Optional[int] = None) -> List[Tuple[str, float]]:
    """
    Sinh **một** kiến trúc HyTra ngẫu nhiên hợp lệ.

    Không gian HyTra gồm 16 lớp (4 stages × 4 layers).
    Khối ứng viên: ``ResConv`` và ``ResAtt``.
    Các tỉ lệ scale: 1/4, 1/8, 1/16, 1/32.

    **Ràng buộc cốt lõi:**
      - ``ResAtt`` chỉ xuất hiện ở scale 1/16 và 1/32
        (do self-attention O(n²) không khả thi với feature-map lớn).
      - Within mỗi stage, scale tuân theo *monotone non-increasing*
        (scale chỉ giảm hoặc giữ nguyên; ``RESTRICTED_PATHS`` mã hoá ràng buộc
        này dưới dạng danh sách đường hợp lệ cho mỗi stage-depth).

    Thuật toán:
      1. Với mỗi stage ``k`` (k = 0..3), lấy depth = ``STAGE_DEPTHS[k]``.
      2. Chọn ngẫu nhiên một *restricted path* hợp lệ từ ``RESTRICTED_PATHS[depth]``.
      3. Chuyển 4 op-indices thành 4 tuples ``(block_type, scale_ratio)``.
      4. Ghép 4 stages → 16 tuples.

    Args:
        seed: Seed tuỳ chọn cho reproducibility (``None`` = không set).

    Returns:
        List[Tuple[str, float]] — 16 tuples ``(block_type, scale_ratio)``.

    Example:
        >>> arch = generate_random_hytra(seed=42)
        >>> len(arch)
        16
        >>> all(bt in ('ResConv', 'ResAtt') for bt, _ in arch)
        True
        >>> all(sr in (1/4, 1/8, 1/16, 1/32) for _, sr in arch)
        True
    """
    if seed is not None:
        random.seed(seed)

    op_indices: List[List[int]] = []
    for stage_idx in range(NUM_STAGES):
        depth = STAGE_DEPTHS[stage_idx]
        path = random.choice(RESTRICTED_PATHS[depth])
        op_indices.append(list(path))           # shallow copy

    return _op_indices_to_tuples(op_indices)


def generate_batch_hytra(
    n: int,
    *,
    seed: int = 42,
    deduplicate: bool = True,
    max_attempts_factor: int = 5,
) -> List[Dict[str, Any]]:
    """
    Sinh ``n`` kiến trúc HyTra **duy nhất** (không trùng lặp).

    Args:
        n:                    Số kiến trúc cần sinh.
        seed:                 Random seed.
        deduplicate:          Loại bỏ kiến trúc trùng (mặc định ``True``).
        max_attempts_factor:  Hệ số giới hạn thử (``n * factor``).

    Returns:
        List[Dict]:
            Mỗi phần tử chứa::

                {
                    "arch_tuples":  List[Tuple[str, float]],  # 16 tuples
                    "arch_string":  str,                       # serializable key
                    "op_indices":   List[List[int]],           # BossNAS format
                    "complexity":   float,                     # proxy complexity
                }
    """
    random.seed(seed)
    np.random.seed(seed)

    results: List[Dict[str, Any]] = []
    seen: set = set()
    attempts = 0
    max_attempts = n * max_attempts_factor

    while len(results) < n and attempts < max_attempts:
        attempts += 1
        arch_tuples = generate_random_hytra()   # seed đã set ở trên
        arch_str = _tuples_to_arch_string(arch_tuples)

        if deduplicate and arch_str in seen:
            continue
        seen.add(arch_str)

        # Convert back → BossNAS op_indices cho tiện lưu JSON
        op_indices: List[List[int]] = []
        for stage in range(NUM_STAGES):
            stage_ops: List[int] = []
            for pos in range(LAYERS_PER_STAGE):
                idx = stage * LAYERS_PER_STAGE + pos
                stage_ops.append(TUPLE_TO_OP_INDEX[arch_tuples[idx]])
            op_indices.append(stage_ops)

        # Complexity estimate (proxy)
        complexity = sum(op * 1.5 + 1 for stage in op_indices for op in stage)

        results.append(
            {
                "arch_tuples": arch_tuples,
                "arch_string": arch_str,
                "op_indices": op_indices,
                "complexity": complexity,
            }
        )

    if len(results) < n:
        warnings.warn(
            f"[generate_batch_hytra] Chỉ sinh được {len(results)}/{n} kiến trúc "
            f"duy nhất sau {attempts} lần thử."
        )

    print(f"[Generator] Sinh {len(results)} kiến trúc HyTra "
          f"(unique, {attempts} attempts)")
    return results


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TÁC VỤ 2: ĐO ĐỘ TRỄ                                                     ║
# ║  (Standardized Latency Profiler)                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# --------------------------------------------------------------------------
#  Xây dựng proxy model (lightweight) để đo latency
# --------------------------------------------------------------------------

class _SelfAttentionBlock(nn.Module):
    """Simplified multi-head self-attention cho profiling."""

    def __init__(self, in_c: int, out_c: int, num_heads: int = 8):
        super().__init__()
        self.proj_in = (
            nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        )
        self.qkv = nn.Conv2d(out_c, out_c * 3, 1)
        self.proj_out = nn.Conv2d(out_c, out_c, 1)
        self.norm = nn.BatchNorm2d(out_c)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _C, H, W = x.shape
        x = self.proj_in(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        head_dim = q.shape[1] // self.num_heads
        n = H * W
        q = q.view(B, self.num_heads, head_dim, n)
        k = k.view(B, self.num_heads, head_dim, n)
        v = v.view(B, self.num_heads, head_dim, n)
        attn = torch.softmax(
            q.transpose(-2, -1) @ k / (head_dim ** 0.5), dim=-1
        )
        out = (v @ attn).view(B, -1, H, W)
        return self.norm(self.proj_out(out))


class HyTraProxyModel(nn.Module):
    """
    Proxy model chuyên dùng cho latency profiling.

    Kiến trúc: Stem → 4 Stages → AvgPool → FC(1000).
    Mỗi stage gồm 4 blocks tuỳ theo op-index (ResConv / ResAtt).
    Channels: [64→256, 256→512, 512→1024, 1024→2048].

    **Không** dùng để huấn luyện chính xác; chỉ để đo chi phí suy diễn
    (number of ops, memory access pattern, kernel launch).
    """

    STAGE_CHANNELS: List[Tuple[int, int]] = [
        (64, 256), (256, 512), (512, 1024), (1024, 2048),
    ]
    def __init__(
        self,
        arch_tuples: List[Tuple[str, float]],
        input_size: int = 224,
    ):
        super().__init__()
        self.arch = arch_tuples
        self.input_size = input_size

        # ── Stem: điều chỉnh theo input_size ──────────────────────────
        if input_size <= 32:
            # CIFAR 32x32: không giảm spatial size ở stem
            # 32x32 → 32x32
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
        else:
            # ImageNet 224x224: stem chuẩn ResNet
            # 224x224 → 56x56
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1),
            )

        # ── 4 Stages ──────────────────────────────────────────────────
        # ImageNet: 56→28→14→7→7
        # CIFAR:    32→32→16→8→8
        self.stages = nn.ModuleList()
        for stage_idx in range(NUM_STAGES):
            in_c, out_c = self.STAGE_CHANNELS[stage_idx]
            layers = []

            if input_size <= 32:
                # CIFAR: downsample ở stage 1 và 2 thôi
                should_downsample = stage_idx in (1, 2)
            else:
                # ImageNet: downsample ở stage 1, 2, 3
                should_downsample = stage_idx > 0

            if should_downsample:
                layers.append(nn.Sequential(
                    nn.Conv2d(in_c, out_c, 1, stride=2, bias=False),
                    nn.BatchNorm2d(out_c),
                ))
                current_c = out_c
            else:
                current_c = in_c

            for pos in range(LAYERS_PER_STAGE):
                block_type, _scale = arch_tuples[
                    stage_idx * LAYERS_PER_STAGE + pos
                ]
                layers.append(self._make_block(block_type, current_c, out_c))
                current_c = out_c

            self.stages.append(nn.Sequential(*layers))

        # ── Head ──────────────────────────────────────────────────────
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, 1000)


    # ----- factory helpers ------------------------------------------------

    @staticmethod
    def _make_block(block_type: str, in_c: int, out_c: int) -> nn.Module:
        """Tạo block tuỳ ``block_type``."""
        if block_type == "ResConv":
            mid_c = max(out_c // 4, 1)
            return nn.Sequential(
                nn.Conv2d(in_c, mid_c, 1, bias=False),
                nn.BatchNorm2d(mid_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_c, mid_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(mid_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_c, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )
        elif block_type == "ResAtt":
            return _SelfAttentionBlock(in_c, out_c)
        else:
            raise ValueError(f"Unknown block_type: {block_type}")

    # ----- forward --------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# --------------------------------------------------------------------------
#  Hàm đo latency
# --------------------------------------------------------------------------

def _synchronize(device_name: str) -> None:
    """Đồng bộ GPU/NPU trước & sau khi bấm giờ."""
    if device_name == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device_name == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    # cpu: không cần synchronize


def measure_latency(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device_name: str,
    *,
    warmup_iters: int = 50,
    measure_iters: int = 100,
) -> float:
    """
    Đo **median latency** (ms) của ``model`` theo quy trình chuẩn mực.

    Quy trình chống nhiễu OS:
      1. **Warm-up** (``warmup_iters`` lần) — làm nóng GPU/NPU, JIT compilation,
         cache allocation.  Không bấm giờ.
      2. **Đo đạc thực tế** (``measure_iters`` lần):
         - ``synchronize()`` ngay TRƯỚC ``perf_counter``
         - Forward pass
         - ``synchronize()`` ngay SAU forward
         - Ghi nhận số đo
      3. Trả về **trung vị** (robust với outliers do GC, OS scheduler, …).

    Thiết bị hỗ trợ:
      - ``cuda``  — NVIDIA RTX 4080 (``torch.cuda.synchronize()``)
      - ``mps``   — Macbook M1 GPU (``torch.mps.synchronize()``)
      - ``cpu``   — Bất kỳ (không cần sync; hoặc giả lập CoreML iPhone 15)

    Args:
        model:          PyTorch model (đã ``.to(device)`` và ``.eval()``).
        input_tensor:   Tensor đầu vào đã nằm trên ``device``.
        device_name:    ``"cuda"`` | ``"mps"`` | ``"cpu"``.
        warmup_iters:   Số iterations warm-up (default 50).
        measure_iters:  Số iterations đo thực (default 100).

    Returns:
        float — Median latency tính bằng mili-giây (ms).
    """
    model.eval()

    # ── Warm-up ──────────────────────────────────────────────────────────
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(input_tensor)
            _synchronize(device_name)

    # ── Đo đạc thực tế ──────────────────────────────────────────────────
    latencies: List[float] = []
    with torch.no_grad():
        for _ in range(measure_iters):
            _synchronize(device_name)               # sync TRƯỚC bấm giờ
            t_start = time.perf_counter()
            _ = model(input_tensor)
            _synchronize(device_name)               # sync SAU forward
            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000.0)   # → ms

    median_ms = float(np.median(latencies))
    return median_ms


def measure_latency_full(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device_name: str,
    *,
    warmup_iters: int = 50,
    measure_iters: int = 100,
) -> Dict[str, float]:
    """
    Giống ``measure_latency`` nhưng trả về dict đầy đủ thống kê.

    Returns:
        Dict với keys: ``mean``, ``std``, ``min``, ``max``,
        ``median``, ``p95``, ``p99``.  Đơn vị: ms.
    """
    model.eval()

    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(input_tensor)
            _synchronize(device_name)

    latencies: List[float] = []
    with torch.no_grad():
        for _ in range(measure_iters):
            _synchronize(device_name)
            t0 = time.perf_counter()
            _ = model(input_tensor)
            _synchronize(device_name)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    arr = np.array(latencies)
    return {
        "mean":   float(np.mean(arr)),
        "std":    float(np.std(arr)),
        "min":    float(np.min(arr)),
        "max":    float(np.max(arr)),
        "median": float(np.median(arr)),
        "p95":    float(np.percentile(arr, 95)),
        "p99":    float(np.percentile(arr, 99)),
    }


def profile_architecture(
    arch_tuples: List[Tuple[str, float]],
    device_name: str = "cuda",
    batch_size: int = 1,
    input_size: int = 224,
    warmup_iters: int = 50,
    measure_iters: int = 100,
) -> Dict[str, Any]:
    """
    Convenience: build → profile → cleanup cho **một** kiến trúc.

    Returns:
        Dict chứa ``latency_ms`` (median), ``latency_stats``, ``params``.
    """
    # Chọn device thực tế
    if device_name == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA không khả dụng, fallback → cpu.")
        device_name = "cpu"
    if device_name == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        warnings.warn("MPS không khả dụng, fallback → cpu.")
        device_name = "cpu"

    device = torch.device(device_name)

    # Build model
    model = HyTraProxyModel(arch_tuples).to(device).eval()
    input_tensor = torch.randn(batch_size, 3, input_size, input_size, device=device)

    # Measure
    stats = measure_latency_full(
        model, input_tensor, device_name,
        warmup_iters=warmup_iters,
        measure_iters=measure_iters,
    )
    params = sum(p.numel() for p in model.parameters())

    # Cleanup
    del model, input_tensor
    if device_name == "cuda":
        torch.cuda.empty_cache()

    return {
        "latency_ms": stats["median"],
        "latency_stats": stats,
        "params": params,
    }


def profile_batch(
    architectures: List[Dict[str, Any]],
    device_name: str = "cuda",
    batch_size: int = 1,
    input_size: int = 224,
    warmup_iters: int = 50,
    measure_iters: int = 100,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Đo latency cho **tất cả** kiến trúc trong danh sách.

    Args:
        architectures:  Output từ ``generate_batch_hytra()``.
        device_name:    ``"cuda"`` | ``"mps"`` | ``"cpu"``.
        Các tham số còn lại: xem ``profile_architecture``.

    Returns:
        (successes, failures) — hai list dicts.
    """
    try:
        from tqdm import tqdm as _tqdm
        iterator = _tqdm(enumerate(architectures), total=len(architectures),
                         desc=f"Profiling ({device_name})")
    except ImportError:
        iterator = enumerate(architectures)

    successes: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for i, arch_info in iterator:
        try:
            result = profile_architecture(
                arch_info["arch_tuples"],
                device_name=device_name,
                batch_size=batch_size,
                input_size=input_size,
                warmup_iters=warmup_iters,
                measure_iters=measure_iters,
            )
            successes.append({**arch_info, **result})
        except Exception as exc:
            failures.append({**arch_info, "error": str(exc)})
            if verbose:
                print(f"\n[FAIL] arch #{i}: {exc}")

    if verbose:
        print(f"[Profiler] Thành công: {len(successes)}, "
              f"Thất bại: {len(failures)}")
    return successes, failures


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TÁC VỤ 3: PDWRS — Probability Density-Weighted Random Sampling       ║
# ║  Lọc mẫu đại diện dựa trên Gaussian KDE                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def pdwrs_sample(
    architecture_list: List[Dict[str, Any]],
    latency_list: List[float],
    n_samples: int = 500,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    PDWRS: Lấy mẫu đại diện dựa trên mật độ xác suất của phân phối latency.

    Thuật toán:
      1. Ước lượng hàm mật độ xác suất (PDF) của ``latency_list`` bằng
         ``scipy.stats.gaussian_kde`` (Gaussian Kernel Density Estimation).
      2. Tính giá trị mật độ ``d_i = kde(latency_i)`` cho mỗi mẫu.
      3. Tạo trọng số lấy mẫu ``w_i = d_i / Σ d_j`` — mẫu nằm ở vùng
         mật độ cao hơn có xác suất được chọn lớn hơn ⇒ đảm bảo coverage
         tốt ở mọi dải latency.
      4. Rút trích ``n_samples`` mẫu **không hoàn lại** (without replacement)
         theo phân phối ``w``.

    Ý nghĩa: Các kiến trúc có latency nằm ở vùng *đặc trưng* (peak) của
    phân phối sẽ được ưu tiên, giúp dữ liệu huấn luyện MHLP phủ đều không
    gian latency mà không bị *bias* về phía các giá trị cực đoan.

    Args:
        architecture_list:  List dict (output ``profile_batch``), mỗi phần tử
                            chứa ``arch_tuples``, ``arch_string``, v.v.
        latency_list:       List float — latency median (ms) tương ứng, đo
                            trên **RTX 4080** (thiết bị nguồn).
        n_samples:          Số mẫu cần lọc (mặc định 500).
        seed:               Random seed.

    Returns:
        List[Dict] — ``n_samples`` phần tử con được chọn từ ``architecture_list``.

    Raises:
        ValueError: Nếu ``n_samples > len(architecture_list)``.
    """
    from scipy.stats import gaussian_kde

    if len(architecture_list) != len(latency_list):
        raise ValueError(
            f"architecture_list ({len(architecture_list)}) và "
            f"latency_list ({len(latency_list)}) phải có cùng độ dài."
        )
    if n_samples > len(architecture_list):
        raise ValueError(
            f"n_samples ({n_samples}) không được lớn hơn số kiến trúc "
            f"hiện có ({len(architecture_list)})."
        )

    rng = np.random.RandomState(seed)

    # ── Bước 1: KDE trên tập latency ────────────────────────────────────
    latency_arr = np.array(latency_list, dtype=np.float64)
    kde = gaussian_kde(latency_arr, bw_method="scott")

    # ── Bước 2: Tính mật độ tại mỗi điểm ───────────────────────────────
    densities = kde.evaluate(latency_arr)          # shape (N,)

    # ── Bước 3: Chuẩn hoá → trọng số xác suất ──────────────────────────
    weights = densities / densities.sum()

    # ── Bước 4: Lấy mẫu không hoàn lại ─────────────────────────────────
    chosen_indices = rng.choice(
        len(architecture_list),
        size=n_samples,
        replace=False,
        p=weights,
    )
    chosen_indices.sort()

    selected = [architecture_list[i] for i in chosen_indices]

    # ── Thống kê ─────────────────────────────────────────────────────────
    sel_lats = latency_arr[chosen_indices]
    print(f"[PDWRS] Chọn {n_samples}/{len(architecture_list)} mẫu")
    print(f"  Latency range đầu vào : {latency_arr.min():.2f} — "
          f"{latency_arr.max():.2f} ms")
    print(f"  Latency range được chọn: {sel_lats.min():.2f} — "
          f"{sel_lats.max():.2f} ms")
    print(f"  Mean ± Std (chọn)      : {sel_lats.mean():.2f} ± "
          f"{sel_lats.std():.2f} ms")

    return selected


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TÁC VỤ 4: PIPELINE THU THẬP & LƯU TRỮ DỮ LIỆU                      ║
# ║  (Data Formatting & Export)                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _simulate_device_latency(
    base_latency_ms: float,
    target_device: str,
    noise_std_frac: float = 0.05,
    rng: Optional[np.random.RandomState] = None,
) -> float:
    """
    Giả lập latency trên thiết bị khác bằng scale factor + nhiễu Gaussian.

    Args:
        base_latency_ms:   Latency đo thực trên RTX 4080 (ms).
        target_device:     ``"Macbook_M1"`` hoặc ``"ip15"``.
        noise_std_frac:    Phần trăm nhiễu (default 5%).
        rng:               RandomState cho reproducibility.

    Returns:
        float — latency giả lập (ms).
    """
    if rng is None:
        rng = np.random.RandomState()

    scale = DEVICE_LATENCY_SCALE.get(target_device, 1.0)
    base_scaled = base_latency_ms * scale
    noise = rng.normal(0, noise_std_frac * base_scaled)
    return max(base_scaled + noise, 0.1)   # clamp tránh âm


def export_csv(
    rows: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """
    Xuất dữ liệu ra CSV với cấu trúc cột chuẩn cho MHLP DataLoader:

        ``architecture_string, hardware_type, latency_ms``

    - ``architecture_string``: chuỗi ``"ResConv@0.125|ResAtt@0.0625|..."``
    - ``hardware_type``:       ``"RTX4080"`` | ``"Macbook_M1"`` | ``"ip15"``
    - ``latency_ms``:          median latency (ms)

    Args:
        rows:         List dict, mỗi dict chứa ít nhất các key trên.
        output_path:  Đường dẫn tuyệt đối file CSV đầu ra.
    """
    fieldnames = ["architecture_string", "hardware_type", "latency_ms"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "architecture_string": row["architecture_string"],
                    "hardware_type": row["hardware_type"],
                    "latency_ms": f'{row["latency_ms"]:.6f}',
                }
            )
    print(f"[Export] Đã lưu {len(rows)} dòng → {output_path}")


# --------------------------------------------------------------------------
#  main() — Pipeline hoàn chỉnh
# --------------------------------------------------------------------------

def main(
    num_generate: int = 2000,
    num_select: int = 500,
    device_name: str = "cuda",
    batch_size: int = 1,
    input_size: int = 224,
    warmup_iters: int = 50,
    measure_iters: int = 100,
    seed: int = 42,
    output_csv: Optional[str] = None,
    output_json: Optional[str] = None,
) -> str:
    """
    Pipeline ghép nối 4 tác vụ — chạy từ đầu đến cuối.

    Bước 1: Sinh ``num_generate`` kiến trúc HyTra ngẫu nhiên.
    Bước 2: Đo latency trên ``device_name`` (đại diện RTX 4080). BS=1.
    Bước 3: Dùng PDWRS lọc ``num_select`` mẫu đại diện.
    Bước 4: Giả lập đo tiếp trên Macbook M1 và iPhone 15.
    Bước 5: Xuất CSV ``hytra_latency_dataset.csv``.

    Returns:
        Đường dẫn file CSV đã lưu.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 72)
    print("  HyTra Latency Data-Collection Pipeline")
    print(f"  Timestamp : {timestamp}")
    print(f"  Device    : {device_name}")
    print(f"  Generate  : {num_generate}")
    print(f"  Select    : {num_select}")
    print("=" * 72)

    # ── BƯỚC 1: Sinh kiến trúc ──────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("BƯỚC 1 / 5 — Sinh kiến trúc HyTra")
    print(f"{'─'*72}")
    architectures = generate_batch_hytra(num_generate, seed=seed)

    # ── BƯỚC 2: Đo latency trên thiết bị nguồn ─────────────────────────
    print(f"\n{'─'*72}")
    print(f"BƯỚC 2 / 5 — Đo latency trên {device_name} (batch_size={batch_size})")
    print(f"{'─'*72}")
    successes, failures = profile_batch(
        architectures,
        device_name=device_name,
        batch_size=batch_size,
        input_size=input_size,
        warmup_iters=warmup_iters,
        measure_iters=measure_iters,
        verbose=True,
    )

    if not successes:
        raise RuntimeError("Không đo được latency cho bất kỳ kiến trúc nào.")

    # ── BƯỚC 3: PDWRS — lọc mẫu đại diện ──────────────────────────────
    print(f"\n{'─'*72}")
    print(f"BƯỚC 3 / 5 — PDWRS: lọc {num_select} mẫu từ {len(successes)}")
    print(f"{'─'*72}")
    latency_list = [s["latency_ms"] for s in successes]
    selected = pdwrs_sample(
        successes, latency_list, n_samples=num_select, seed=seed,
    )

    # ── BƯỚC 4: Giả lập đo trên Macbook M1 & iPhone 15 ────────────────
    print(f"\n{'─'*72}")
    print("BƯỚC 4 / 5 — Giả lập latency trên Macbook_M1 và ip15")
    print(f"{'─'*72}")
    rng = np.random.RandomState(seed)
    csv_rows: List[Dict[str, Any]] = []

    for arch_info in selected:
        base_lat = arch_info["latency_ms"]
        arch_str = arch_info["arch_string"]

        # RTX 4080 (đo thực)
        csv_rows.append(
            {
                "architecture_string": arch_str,
                "hardware_type": "RTX4080",
                "latency_ms": base_lat,
            }
        )
        # Macbook M1 (giả lập)
        csv_rows.append(
            {
                "architecture_string": arch_str,
                "hardware_type": "Macbook_M1",
                "latency_ms": _simulate_device_latency(
                    base_lat, "Macbook_M1", rng=rng
                ),
            }
        )
        # iPhone 15 (giả lập)
        csv_rows.append(
            {
                "architecture_string": arch_str,
                "hardware_type": "ip15",
                "latency_ms": _simulate_device_latency(
                    base_lat, "ip15", rng=rng
                ),
            }
        )

    print(f"  Tổng dòng CSV: {len(csv_rows)} "
          f"({num_select} archs × 3 devices)")

    # ── BƯỚC 5: Xuất CSV + JSON (tuỳ chọn) ─────────────────────────────
    print(f"\n{'─'*72}")
    print("BƯỚC 5 / 5 — Lưu dữ liệu")
    print(f"{'─'*72}")

    if output_csv is None:
        output_csv = os.path.join(_SCRIPT_DIR, "hytra_latency_dataset.csv")
    export_csv(csv_rows, output_csv)

    # Lưu JSON bổ sung (metadata + 500 kiến trúc đã chọn) cho debug/analysis
    if output_json is None:
        output_json = os.path.join(_SCRIPT_DIR, "hytra_latency_dataset.json")
    json_data = {
        "timestamp": timestamp,
        "pipeline_config": {
            "num_generate": num_generate,
            "num_select": num_select,
            "device_name": device_name,
            "batch_size": batch_size,
            "input_size": input_size,
            "warmup_iters": warmup_iters,
            "measure_iters": measure_iters,
            "seed": seed,
        },
        "num_measured": len(successes),
        "num_failed": len(failures),
        "num_selected": len(selected),
        "selected_architectures": [
            {
                "arch_string": s["arch_string"],
                "op_indices": s["op_indices"],
                "complexity": s["complexity"],
                "latency_ms_rtx4080": s["latency_ms"],
                "latency_stats": s.get("latency_stats", {}),
                "params": s.get("params", 0),
            }
            for s in selected
        ],
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"[Export] Metadata JSON → {output_json}")

    # ── Tóm tắt ─────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  PIPELINE HOÀN TẤT")
    print(f"  CSV  : {output_csv}")
    print(f"  JSON : {output_json}")
    print(f"  Rows : {len(csv_rows)} ({num_select} archs × 3 devices)")
    print(f"{'='*72}")

    return output_csv


# --------------------------------------------------------------------------
#  CLI entry-point
# --------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HyTra Latency Data-Collection Pipeline cho MHLP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num_generate", type=int, default=2000,
                    help="Số kiến trúc HyTra sinh ngẫu nhiên")
    p.add_argument("--num_select", type=int, default=500,
                    help="Số mẫu giữ lại sau PDWRS")
    p.add_argument("--device", type=str, default="cuda",
                    choices=["cuda", "mps", "cpu"],
                    help="Thiết bị đo latency")
    p.add_argument("--batch_size", type=int, default=1,
                    help="Batch size cho profiling")
    p.add_argument("--input_size", type=int, default=224,
                    help="Kích thước ảnh đầu vào")
    p.add_argument("--warmup", type=int, default=50,
                    help="Số iterations warm-up")
    p.add_argument("--runs", type=int, default=100,
                    help="Số iterations đo thực")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed")
    p.add_argument("--output_csv", type=str, default=None,
                    help="Đường dẫn file CSV đầu ra")
    p.add_argument("--output_json", type=str, default=None,
                    help="Đường dẫn file JSON metadata")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        num_generate=args.num_generate,
        num_select=args.num_select,
        device_name=args.device,
        batch_size=args.batch_size,
        input_size=args.input_size,
        warmup_iters=args.warmup,
        measure_iters=args.runs,
        seed=args.seed,
        output_csv=args.output_csv,
        output_json=args.output_json,
    )
