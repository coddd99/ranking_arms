import argparse
import os
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import csv


METRICS = [
    "best_fixed_is_stepbest_rate",
    "top_share_mean_per_window",
    "top_arm_switch_count",
    "tail_mean_reward", "tail_mean_hit1", "tail_mean_hitk",
    "full_mean_reward", "full_mean_hit1", "full_mean_hitk",
]

META_KEYS = ["T", "A", "window", "best_fixed_action_idx"]


def load_rows(check_dir: str):
    rows = []
    for fn in os.listdir(check_dir):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(check_dir, fn)
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        
        if "bandit" not in d or "seed" not in d:
            continue
        rows.append(d)
    return rows


def summarize(rows):
    by_bandit = defaultdict(list)
    for r in rows:
        by_bandit[r["bandit"]].append(r)

    summary = {}
    warnings = []

    for bandit, lst in by_bandit.items():
        # meta 일관성 검사
        meta = {}
        for k in META_KEYS:
            vals = [x.get(k, None) for x in lst]
            uniq = sorted(set(vals))
            meta[k] = uniq[0] if len(uniq) == 1 else uniq
            if len(uniq) != 1:
                warnings.append(f"[WARN] bandit={bandit} meta '{k}' mismatch: {uniq}")

        seeds = sorted([int(x["seed"]) for x in lst])

        # metric 통계
        stats = {}
        for m in METRICS:
            arr = np.array([x.get(m, np.nan) for x in lst], dtype=float)
            valid = arr[~np.isnan(arr)]
            if len(valid) == 0:
                continue
            stats[m] = {
                "mean": float(valid.mean()),
                "std": float(valid.std(ddof=0)),
                "n": int(len(valid)),
            }

        summary[bandit] = {
            "n_runs": len(lst),
            "seeds": seeds,
            "meta": meta,
            "metrics": stats,
        }

    return summary, warnings


def print_table(summary):
    cols = [
        ("full_mean_reward",  "fullR"),
        ("tail_mean_reward",  "tailR"),
        ("full_mean_hitk",    "full@k"),
        ("tail_mean_hitk",    "tail@k"),
        ("top_share_mean_per_window", "topShare"),
        ("top_arm_switch_count",      "switch"),
    ]

    header = f"{'bandit':45s} {'n':>3s} " + " ".join([f"{c[1]:>12s}" for c in cols])
    print(header)
    print("-" * len(header))

    for bandit in sorted(summary.keys()):
        s = summary[bandit]
        parts = [f"{bandit[:45]:45s}", f"{s['n_runs']:3d}"]
        for key, _ in cols:
            if key in s["metrics"]:
                mu = s["metrics"][key]["mean"]
                sd = s["metrics"][key]["std"]
                parts.append(f"{mu:.4f}±{sd:.4f}")
            else:
                parts.append("NA")
        print(" ".join(parts))


def save_json(summary, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def _fmt4(x) -> str:
    # csv에 4째자리 반올림해서 "문자열"로 저장
    try:
        return f"{float(x):.4f}"
    except Exception:
        return ""


def save_csv_full_tail(summary, out_full_path: str, out_tail_path: str):
    """
    2개 CSV 생성:
      - aggregate_full.csv : full_mean_hit1/hitk/reward 의 mean/std
      - aggregate_tail.csv : tail_mean_hit1/hitk/reward 의 mean/std
    컬럼명은 hitk 그대로 유지, 값은 소수점 4자리 반올림
    """

    def build_rows(prefix: str):
        rows = []
        for bandit, s in summary.items():
            metrics = s.get("metrics", {})
            row = {"bandit": bandit}

            for base in [f"{prefix}_mean_hit1", f"{prefix}_mean_hitk", f"{prefix}_mean_reward"]:
                if base in metrics:
                    row[f"{base}_mean"] = _fmt4(metrics[base]["mean"])
                    row[f"{base}_std"] = _fmt4(metrics[base]["std"])
                else:
                    row[f"{base}_mean"] = ""
                    row[f"{base}_std"] = ""
            rows.append(row)

        rows.sort(key=lambda r: r["bandit"])
        return rows

    def write_csv(rows, path, prefix: str):
        fieldnames = [
            "bandit",
            f"{prefix}_mean_hit1_mean", f"{prefix}_mean_hit1_std",
            f"{prefix}_mean_hitk_mean", f"{prefix}_mean_hitk_std",
            f"{prefix}_mean_reward_mean", f"{prefix}_mean_reward_std",
        ]
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    full_rows = build_rows("full")
    tail_rows = build_rows("tail")

    write_csv(full_rows, out_full_path, "full")
    write_csv(tail_rows, out_tail_path, "tail")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--check-dir",
        type=str,
        required=True,
        help="run_bandit_checks 폴더 경로 (예: .outputs/20251218_143012/run_bandit_checks)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    check_dir = args.check_dir

    if not os.path.isdir(check_dir):
        raise SystemExit(f"--check-dir not found or not a directory: {check_dir}")

    rows = load_rows(check_dir)
    if not rows:
        raise SystemExit(f"No json found in: {check_dir}")

    summary, warnings = summarize(rows)

    # 콘솔 출력
    print_table(summary)
    if warnings:
        print("\n".join(warnings))

    out_json = os.path.join(check_dir, "aggregate_summary.json")
    out_full_csv = os.path.join(check_dir, "aggregate_full.csv")
    out_tail_csv = os.path.join(check_dir, "aggregate_tail.csv")

    save_json(summary, out_json)
    save_csv_full_tail(summary, out_full_csv, out_tail_csv)

    print(f"\nSaved:\n- {out_json}\n- {out_full_csv}\n- {out_tail_csv}")


if __name__ == "__main__":
    main()
