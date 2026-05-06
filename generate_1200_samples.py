# generate_1200_samples.py
# Chạy PHẦN A trên server GPU trước
# Sau đó copy file arch_1200_fixed.json sang các thiết bị khác

"""
HƯỚNG DẪN CHẠY:
  Server RTX 4060: python generate_1200_samples.py --phase generate --device cuda --device-name rtx4060
  Server RTX 3050: python generate_1200_samples.py --phase measure  --device cuda --device-name rtx3050
  Server RTX 5060: python generate_1200_samples.py --phase measure  --device cuda --device-name rtx5060
  MacBook M1:      python generate_1200_samples.py --phase measure  --device mps  --device-name macbook_m1
  Windows CPU:     python generate_1200_samples.py --phase measure  --device cpu  --device-name windows_cpu
  Raspberry Pi:    python generate_1200_samples.py --phase measure  --device cpu  --device-name raspi4 --n-samples 600
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# ── Path setup ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "searching"))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "OpenSelfSup"))


def _resolve_path(path: str) -> str:
    """Resolve CLI-provided paths robustly.

    - Absolute paths are kept as-is.
    - Relative paths are resolved relative to this script's directory.

    This makes the script runnable from any working directory.
    """
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(_SCRIPT_DIR, path))

from hytra_latency_pipeline import (
    HyTraProxyModel,
    OP_INDEX_TO_TUPLE,
    RESTRICTED_PATHS,
    STAGE_DEPTHS,
    TUPLE_TO_OP_INDEX,
    _arch_string_to_tuples,
    _synchronize,
    _tuples_to_arch_string,
    measure_latency_full,
    NUM_STAGES,
    LAYERS_PER_STAGE,
)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PHẦN A: SINH 1200 MẪU STRATIFIED                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def count_resatt(arch_tuples: List[Tuple[str, float]]) -> int:
    """Đếm số ResAtt blocks trong kiến trúc."""
    return sum(1 for bt, _ in arch_tuples if bt == "ResAtt")


def classify_arch(arch_tuples: List[Tuple[str, float]]) -> str:
    """
    Phân loại kiến trúc vào 1 trong 4 nhóm:
      pure_conv   : 0 ResAtt blocks
      heavy_attn  : ≥6 ResAtt blocks
      hybrid      : 3–5 ResAtt blocks
      random      : còn lại (1–2 ResAtt blocks)
    """
    n = count_resatt(arch_tuples)
    if n == 0:
        return "pure_conv"
    elif n >= 6:
        return "heavy_attn"
    elif 3 <= n <= 5:
        return "hybrid"
    else:
        return "light_attn"  # 1–2 ResAtt — nhóm random sẽ cover


def generate_stratified_1200(seed: int = 42) -> List[Dict[str, Any]]:
    """
    Sinh 1200 mẫu stratified đảm bảo diversity.

    Phân bổ:
      pure_conv  : 250 mẫu — CNN baseline
      heavy_attn : 250 mẫu — Transformer-heavy
      hybrid     : 450 mẫu — Vùng BossNAS thường tìm ra
      random     : 250 mẫu — Coverage tổng quát

    Returns:
        List 1200 dicts, mỗi dict có:
          arch_id, arch_string, arch_tuples, op_indices,
          group, n_resatt, complexity
    """
    rng = random.Random(seed)

    TARGETS = {
        "pure_conv":  250,
        "heavy_attn": 250,
        "hybrid":     450,
        "random":     250,
    }

    groups: Dict[str, List[Dict[str, Any]]] = {k: [] for k in TARGETS}
    seen_strings: set[str] = set()

    conv_ops = {1, 3, 4, 5}
    attn_ops = {0, 2}

    def _count_resatt_from_op_indices(op_indices: List[List[int]]) -> int:
        return sum(1 for stage in op_indices for op in stage if op in attn_ops)

    def _build_arch_info(op_indices: List[List[int]], group: str) -> Dict[str, Any]:
        arch_tuples: List[Tuple[str, float]] = []
        for stage in op_indices:
            for op in stage:
                arch_tuples.append(OP_INDEX_TO_TUPLE[op])
        arch_str = _tuples_to_arch_string(arch_tuples)
        n_resatt = count_resatt(arch_tuples)
        complexity = sum(op * 1.5 + 1 for stage in op_indices for op in stage)
        return {
            "arch_tuples": arch_tuples,
            "arch_string": arch_str,
            "op_indices": op_indices,
            "complexity": complexity,
            "group": group,
            "n_resatt": n_resatt,
        }

    # Precompute stage-path buckets by attention-count for each stage.
    stage_paths_by_attn: List[Dict[int, List[List[int]]]] = []
    stage_possible_counts: List[List[int]] = []
    for stage_idx in range(NUM_STAGES):
        depth = STAGE_DEPTHS[stage_idx]
        paths = RESTRICTED_PATHS[depth]
        buckets: Dict[int, List[List[int]]] = {}
        for path in paths:
            attn_count = sum(1 for op in path if op in attn_ops)
            buckets.setdefault(attn_count, []).append(list(path))
        stage_paths_by_attn.append(buckets)
        stage_possible_counts.append(sorted(buckets.keys()))

    stage_min = [min(v) for v in stage_possible_counts]
    stage_max = [max(v) for v in stage_possible_counts]

    def _sample_counts_for_total(total_attn: int) -> Optional[List[int]]:
        """Randomized backtracking to find per-stage attention counts summing to total_attn."""
        chosen: List[int] = []

        def rec(stage_idx: int, remaining: int) -> bool:
            if stage_idx == NUM_STAGES:
                return remaining == 0

            # Prune by min/max possible of remaining stages (including this one).
            min_possible = sum(stage_min[stage_idx:])
            max_possible = sum(stage_max[stage_idx:])
            if remaining < min_possible or remaining > max_possible:
                return False

            options = stage_possible_counts[stage_idx][:]
            rng.shuffle(options)
            for c in options:
                if c > remaining:
                    continue

                # Prune for next stages.
                next_min = sum(stage_min[stage_idx + 1 :]) if stage_idx + 1 < NUM_STAGES else 0
                next_max = sum(stage_max[stage_idx + 1 :]) if stage_idx + 1 < NUM_STAGES else 0
                next_remaining = remaining - c
                if next_remaining < next_min or next_remaining > next_max:
                    continue

                chosen.append(c)
                if rec(stage_idx + 1, next_remaining):
                    return True
                chosen.pop()

            return False

        ok = rec(0, total_attn)
        return chosen if ok else None

    def _sample_op_indices_for_attn_range(
        group: str,
        min_attn_total: int,
        max_attn_total: int,
        *,
        max_tries: int = 10_000,
    ) -> Optional[List[List[int]]]:
        for _ in range(max_tries):
            target_total = rng.randint(min_attn_total, max_attn_total)
            counts = _sample_counts_for_total(target_total)
            if counts is None:
                continue
            op_indices: List[List[int]] = []
            for stage_idx, attn_count in enumerate(counts):
                pool = stage_paths_by_attn[stage_idx][attn_count]
                op_indices.append(rng.choice(pool))
            # Safety check.
            total = _count_resatt_from_op_indices(op_indices)
            if min_attn_total <= total <= max_attn_total:
                return op_indices
        return None

    def _fill_group(
        group: str,
        n_target: int,
        *,
        sampler,
        max_tries: int = 1_000_000,
    ) -> None:
        tries = 0
        while len(groups[group]) < n_target and tries < max_tries:
            tries += 1
            op_indices = sampler()
            if op_indices is None:
                continue
            arch_info = _build_arch_info(op_indices, group)
            arch_str = arch_info["arch_string"]
            if arch_str in seen_strings:
                continue
            seen_strings.add(arch_str)
            groups[group].append(arch_info)

        if len(groups[group]) < n_target:
            raise RuntimeError(
                f"Không thể sinh đủ nhóm '{group}': {len(groups[group])}/{n_target} "
                f"sau {tries} lần thử."
            )

    print("=" * 60)
    print("  SINH 1200 MẪU STRATIFIED")
    print("=" * 60)
    print(f"  Mục tiêu: {TARGETS}")

    def _sample_random_op_indices() -> List[List[int]]:
        op_indices: List[List[int]] = []
        for stage_idx in range(NUM_STAGES):
            depth = STAGE_DEPTHS[stage_idx]
            path = rng.choice(RESTRICTED_PATHS[depth])
            op_indices.append(list(path))
        return op_indices

    def _sample_pure_conv_op_indices() -> Optional[List[List[int]]]:
        op_indices: List[List[int]] = []
        for stage_idx in range(NUM_STAGES):
            pool = stage_paths_by_attn[stage_idx].get(0)
            if not pool:
                return None
            op_indices.append(rng.choice(pool))
        return op_indices

    # Build each group by construction (avoids rare-event rejection sampling).
    _fill_group("pure_conv", TARGETS["pure_conv"], sampler=_sample_pure_conv_op_indices)
    print(f"  ✅ pure_conv  : {len(groups['pure_conv'])}/{TARGETS['pure_conv']}")

    _fill_group(
        "hybrid",
        TARGETS["hybrid"],
        sampler=lambda: _sample_op_indices_for_attn_range("hybrid", 3, 5),
    )
    print(f"  ✅ hybrid     : {len(groups['hybrid'])}/{TARGETS['hybrid']}")

    # For heavy_attn, cap max by what's achievable from stage path buckets.
    max_total_attn = sum(stage_max)
    _fill_group(
        "heavy_attn",
        TARGETS["heavy_attn"],
        sampler=lambda: _sample_op_indices_for_attn_range(
            "heavy_attn", 6, max_total_attn
        ),
    )
    print(f"  ✅ heavy_attn : {len(groups['heavy_attn'])}/{TARGETS['heavy_attn']}")

    _fill_group("random", TARGETS["random"], sampler=_sample_random_op_indices)
    print(f"  ✅ random     : {len(groups['random'])}/{TARGETS['random']}")

    # ── Báo cáo kết quả ─────────────────────────────────────────────────
    for name, target in TARGETS.items():
        actual = len(groups[name])
        status = "✅" if actual >= target else "⚠️ THIẾU"
        print(f"  {status} {name:12s}: {actual}/{target}")

    # ── Merge và shuffle ─────────────────────────────────────────────────
    all_samples = []
    for group_name, archs in groups.items():
        all_samples.extend(archs)

    rng.shuffle(all_samples)

    # Gán arch_id sau khi shuffle
    for i, s in enumerate(all_samples):
        s["arch_id"] = i

    # ── Thống kê ResAtt distribution ────────────────────────────────────
    print(f"\n  Total: {len(all_samples)} mẫu")
    print(f"\n  ResAtt distribution:")
    resatt_counts = [s["n_resatt"] for s in all_samples]
    for n in range(0, 13):
        count = resatt_counts.count(n)
        if count > 0:
            bar = "█" * (count // 10)
            print(f"    {n:2d} ResAtt: {count:4d} mẫu {bar}")

    if resatt_counts:
        pct_with_attn = sum(1 for x in resatt_counts if x > 0) / len(resatt_counts) * 100
        print(f"\n  Có ResAtt: {pct_with_attn:.1f}%")
        print(f"  Không ResAtt: {100 - pct_with_attn:.1f}%")

    return all_samples


def save_fixed_set(samples: List[Dict], output_path: str) -> None:
    """
    Lưu 1200 mẫu ra JSON để tất cả devices đo cùng 1 set.

    QUAN TRỌNG: Tất cả devices phải load file này,
    không được generate lại riêng.
    """
    # Convert arch_tuples sang serializable format
    serializable = []
    for s in samples:
        record = {
            "arch_id":     s["arch_id"],
            "arch_string": s["arch_string"],
            "op_indices":  s["op_indices"],   # [[4,2,0,1], ...]
            "group":       s["group"],
            "n_resatt":    s["n_resatt"],
            "complexity":  s["complexity"],
            # arch_tuples không JSON-serializable trực tiếp
            # → reconstruct từ arch_string khi load
        }
        serializable.append(record)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "version":     "1.0",
            "total":       len(serializable),
            "description": "1200 stratified HyTra architectures for MHLP",
            "sampling":    {
                "pure_conv":  250,
                "heavy_attn": 250,
                "hybrid":     450,
                "random":     250,
            },
            "architectures": serializable,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved {len(serializable)} mẫu → {output_path}")
    print(f"   Copy file này sang TẤT CẢ devices trước khi đo!")


def load_fixed_set(json_path: str) -> List[Dict]:
    """Load file JSON và reconstruct arch_tuples từ arch_string."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for record in data["architectures"]:
        arch_tuples = _arch_string_to_tuples(record["arch_string"])
        samples.append({
            **record,
            "arch_tuples": arch_tuples,
        })

    print(f"✅ Loaded {len(samples)} mẫu từ {json_path}")
    return samples


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PHẦN B: ĐO LATENCY TRÊN TỪNG DEVICE                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def measure_on_device(
    samples: List[Dict],
    device_name: str,
    hardware_label: str,
    *,
    input_size: int = 32,      # CIFAR size — không phải 224!
    batch_size: int = 1,
    warmup_iters: int = 20,
    measure_iters: int = 50,
    save_every: int = 50,
    output_path: str,
    resume: bool = True,
) -> List[Dict]:
    """
    Đo latency của tất cả mẫu trong samples trên device hiện tại.

    Args:
        samples:        List arch dicts từ load_fixed_set()
        device_name:    "cuda" | "mps" | "cpu"
        hardware_label: Tên thiết bị lưu vào CSV ("rtx4060", "macbook_m1", ...)
        input_size:     32 cho CIFAR, 224 cho ImageNet
        warmup_iters:   Số lần warmup (giảm xuống 20 vì CIFAR nhỏ hơn)
        measure_iters:  Số lần đo (giảm xuống 50 vì đo nhiều device)
        save_every:     Lưu checkpoint mỗi N mẫu
        output_path:    File JSON kết quả cho device này
        resume:         Tiếp tục từ checkpoint nếu bị interrupt

    Returns:
        List dicts với latency stats
    """
    # ── Kiểm tra device ──────────────────────────────────────────────────
    if device_name == "cuda":
        if not torch.cuda.is_available():
            print("⚠️  CUDA không khả dụng, fallback → cpu")
            device_name = "cpu"
        else:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  GPU: {gpu_name}")

    elif device_name == "mps":
        if not (hasattr(torch.backends, "mps") and
                torch.backends.mps.is_available()):
            print("⚠️  MPS không khả dụng, fallback → cpu")
            device_name = "cpu"
        else:
            print("  Device: Apple Silicon MPS")

    device = torch.device(device_name)

    # ── Resume logic ─────────────────────────────────────────────────────
    completed_ids = set()
    results = []

    if resume and os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            existing = json.load(f)
        results = existing.get("measurements", [])
        completed_ids = {r["arch_id"] for r in results}
        print(f"  Resume: {len(completed_ids)} mẫu đã đo, "
              f"còn {len(samples) - len(completed_ids)} mẫu")

    remaining = [s for s in samples if s["arch_id"] not in completed_ids]

    print(f"\n  Bắt đầu đo {len(remaining)} mẫu...")
    print(f"  Input size: {input_size}×{input_size} (batch={batch_size})")
    print(f"  Warmup: {warmup_iters}, Measure: {measure_iters}")
    print(f"  Output: {output_path}")
    print("-" * 60)

    input_tensor = torch.randn(
        batch_size, 3, input_size, input_size, device=device
    )

    failures = []
    t_start_total = time.time()

    for i, sample in enumerate(remaining):
        arch_id = sample["arch_id"]
        arch_tuples = sample["arch_tuples"]

        try:
            # Build model
            model = HyTraProxyModel(
                arch_tuples, input_size=input_size
            ).to(device).eval()

            # Measure
            stats = measure_latency_full(
                model, input_tensor, device_name,
                warmup_iters=warmup_iters,
                measure_iters=measure_iters,
            )

            results.append({
                "arch_id":      arch_id,
                "arch_string":  sample["arch_string"],
                "hardware":     hardware_label,
                "device_used":  device_name,
                "input_size":   input_size,
                "latency_ms":   stats["median"],   # dùng median làm giá trị chính
                "stats":        stats,
                "group":        sample["group"],
                "n_resatt":     sample["n_resatt"],
            })

            # Cleanup
            del model
            if device_name == "cuda":
                torch.cuda.empty_cache()

            # Progress log
            elapsed = time.time() - t_start_total
            avg_per_sample = elapsed / (i + 1)
            remaining_count = len(remaining) - i - 1
            eta_min = remaining_count * avg_per_sample / 60

            print(f"  [{i+1:4d}/{len(remaining)}] "
                  f"arch#{arch_id:4d} | "
                  f"lat={stats['median']:7.2f}ms | "
                  f"resatt={sample['n_resatt']:2d} | "
                  f"ETA={eta_min:.0f}min")

        except Exception as e:
            print(f"  ❌ arch#{arch_id}: {e}")
            failures.append({"arch_id": arch_id, "error": str(e)})

        # Save checkpoint
        if (i + 1) % save_every == 0:
            _save_checkpoint(results, failures, hardware_label, output_path)
            print(f"  💾 Checkpoint saved ({len(results)} mẫu)")

    # Final save
    _save_checkpoint(results, failures, hardware_label, output_path)

    elapsed_total = time.time() - t_start_total
    print(f"\n{'='*60}")
    print(f"  ✅ HOÀN TẤT đo trên {hardware_label}")
    print(f"  Thành công: {len(results)} mẫu")
    print(f"  Thất bại  : {len(failures)} mẫu")
    print(f"  Thời gian : {elapsed_total/60:.1f} phút")
    print(f"  Kết quả   : {output_path}")

    return results


def _save_checkpoint(results, failures, hardware_label, output_path):
    """Lưu checkpoint để resume nếu bị interrupt."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "hardware":     hardware_label,
            "total":        len(results),
            "measurements": results,
            "failures":     failures,
        }, f, indent=2, ensure_ascii=False)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PHẦN C: AGGREGATE TẤT CẢ DEVICES → CSV CUỐI CÙNG                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def aggregate_all_devices(
    result_files: List[str],
    output_csv: str,
    output_json: str,
) -> None:
    """
    Gộp kết quả từ tất cả devices thành 1 CSV duy nhất cho MHLP.

    CSV format (giống pipeline gốc):
        architecture_string, hardware_type, latency_ms

    Args:
        result_files: List đường dẫn các file JSON từng device
        output_csv:   File CSV đầu ra
        output_json:  File JSON metadata
    """
    import csv

    all_rows = []
    summary = {}

    for fpath in result_files:
        if not os.path.exists(fpath):
            print(f"⚠️  Không tìm thấy: {fpath}")
            continue

        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)

        hardware = data["hardware"]
        measurements = data["measurements"]

        print(f"  {hardware:20s}: {len(measurements):4d} mẫu")
        summary[hardware] = len(measurements)

        for m in measurements:
            all_rows.append({
                "architecture_string": m["arch_string"],
                "hardware_type":       m["hardware"],
                "latency_ms":          m["latency_ms"],
            })

    # Lưu CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["architecture_string", "hardware_type", "latency_ms"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    # Lưu JSON metadata
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({
            "total_rows":  len(all_rows),
            "devices":     summary,
            "description": "MHLP latency dataset — 1200 archs × N devices",
        }, f, indent=2)

    print(f"\n✅ Aggregate hoàn tất")
    print(f"   CSV : {output_csv} ({len(all_rows)} dòng)")
    print(f"   Devices: {list(summary.keys())}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CLI                                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate 1200 stratified samples + measure latency",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--phase", required=True,
                   choices=["generate", "measure", "aggregate"],
                   help=(
                       "generate: sinh 1200 mẫu (chạy 1 lần trên server) | "
                       "measure: đo latency (chạy trên từng device) | "
                       "aggregate: gộp tất cả → CSV"
                   ))

    # Generate phase
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fixed-set", type=str,
                   default=os.path.join(_SCRIPT_DIR, "arch_1200_fixed.json"),
                   help="Path file JSON chứa 1200 mẫu cố định")

    # Measure phase
    p.add_argument("--device", type=str, default="cuda",
                   choices=["cuda", "mps", "cpu"])
    p.add_argument("--device-name", type=str, default=None,
                   help="Label thiết bị: rtx4060, rtx3050, rtx5060, "
                        "macbook_m1, windows_cpu, raspi4")
    p.add_argument("--input-size", type=int, default=32,
                   help="Input image size (32 cho CIFAR, 224 cho ImageNet)")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--n-samples", type=int, default=1200,
                   help="Số mẫu cần đo (Raspi dùng 600)")
    p.add_argument("--output-dir", type=str,
                   default=os.path.join(_SCRIPT_DIR, "latency_raw"),
                   help="Thư mục lưu kết quả từng device")
    p.add_argument("--no-resume", action="store_true",
                   help="Đo lại từ đầu, không resume")

    # Aggregate phase
    p.add_argument("--output-csv", type=str,
                   default=os.path.join(_SCRIPT_DIR, "hytra_latency_dataset_1200.csv"))
    p.add_argument("--output-json", type=str,
                   default=os.path.join(_SCRIPT_DIR, "hytra_latency_dataset_1200.json"))

    return p.parse_args()


def main():
    args = parse_args()

    # Canonicalize paths so the script works from any cwd.
    args.fixed_set = _resolve_path(args.fixed_set)
    args.output_dir = _resolve_path(args.output_dir)
    args.output_csv = _resolve_path(args.output_csv)
    args.output_json = _resolve_path(args.output_json)

    # ── PHASE: generate ──────────────────────────────────────────────────
    if args.phase == "generate":
        print("\n🔧 PHASE: GENERATE 1200 MẪU STRATIFIED")
        samples = generate_stratified_1200(seed=args.seed)
        save_fixed_set(samples, args.fixed_set)
        print(f"\n📋 BƯỚC TIẾP THEO:")
        print(f"   1. Copy file '{args.fixed_set}' sang tất cả devices")
        print(f"   2. Chạy trên RTX 4060:")
        print(f"      python generate_1200_samples.py --phase measure "
              f"--device cuda --device-name rtx4060")
        print(f"   3. Chạy tương tự trên các device khác")
        print(f"   4. Sau khi tất cả xong, chạy aggregate")

    # ── PHASE: measure ───────────────────────────────────────────────────
    elif args.phase == "measure":
        if args.device_name is None:
            print("❌ Cần chỉ định --device-name")
            print("   Ví dụ: --device-name rtx4060")
            sys.exit(1)

        if not os.path.exists(args.fixed_set):
            print(f"❌ Không tìm thấy file: {args.fixed_set}")
            print(f"   Chạy phase generate trước!")
            sys.exit(1)

        print(f"\n📏 PHASE: ĐO LATENCY trên {args.device_name.upper()}")

        # Load fixed set
        samples = load_fixed_set(args.fixed_set)

        # Giới hạn số mẫu nếu cần (Raspi)
        if args.n_samples < len(samples):
            print(f"  Giới hạn: {args.n_samples}/{len(samples)} mẫu")
            samples = samples[:args.n_samples]

        # Output path
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(
            args.output_dir,
            f"latency_{args.device_name}.json"
        )

        # Measure
        measure_on_device(
            samples=samples,
            device_name=args.device,
            hardware_label=args.device_name,
            input_size=args.input_size,
            warmup_iters=args.warmup,
            measure_iters=args.runs,
            output_path=output_path,
            resume=not args.no_resume,
        )

    # ── PHASE: aggregate ─────────────────────────────────────────────────
    elif args.phase == "aggregate":
        print("\n📦 PHASE: AGGREGATE TẤT CẢ DEVICES")

        raw_dir = os.path.join(os.path.dirname(args.fixed_set), "latency_raw")

        # Tự động tìm tất cả file JSON trong thư mục raw
        result_files = []
        if os.path.exists(raw_dir):
            for fname in sorted(os.listdir(raw_dir)):
                if fname.startswith("latency_") and fname.endswith(".json"):
                    result_files.append(os.path.join(raw_dir, fname))

        if not result_files:
            print(f"❌ Không tìm thấy file kết quả trong {raw_dir}")
            sys.exit(1)

        print(f"  Tìm thấy {len(result_files)} device files:")
        for f in result_files:
            print(f"    - {os.path.basename(f)}")

        aggregate_all_devices(
            result_files=result_files,
            output_csv=args.output_csv,
            output_json=args.output_json,
        )


if __name__ == "__main__":
    main()