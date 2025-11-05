#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_hybrid.py
------------------
Evaluate hybrid recommender outputs with top-K ranking metrics.

Defaults (project-relative):
  GT:       data/test/clean_opentable.pkl
  RECS:     results/hybrid_recs_opentable.csv
  OUT DIR:  results/

Usage examples:
  python evaluate_hybrid.py
  python evaluate_hybrid.py --gt data/test/clean_opentable.pkl --recs results/hybrid_recs_opentable.csv --k 10
  python evaluate_hybrid.py --k 20 --pos-col rating_norm --pos-thresh 0.5

Notes:
- We treat a (user_id, business_id) pair as positive if GT value in --pos-col >= --pos-thresh.
  If --pos-col doesn't exist, it falls back to column 'rating' and threshold 4.0.
- Users with no positives in GT are excluded from Recall/Precision denominators by default.
"""

import os, json, argparse, math
from typing import Dict, Set, Tuple, List
import numpy as np
import pandas as pd
from scipy import sparse

def log(msg: str):
    print(msg, flush=True)

def load_ground_truth(gt_path: str, pos_col: str, pos_thresh: float) -> pd.DataFrame:
    df = pd.read_pickle(gt_path)
    # Try to be robust to column names
    if pos_col not in df.columns:
        fallback = "rating" if "rating" in df.columns else None
        if fallback is None:
            raise KeyError(f"Column '{pos_col}' not found and no 'rating' fallback in GT.")
        log(f"[WARN] '{pos_col}' not in GT. Falling back to '{fallback}' >= 4.0")
        pos_col = fallback
        pos_thresh = 4.0
    # Ensure types as strings to match recs csv that may be numeric
    df["user_id"] = df["user_id"].astype(str)
    df["business_id"] = df["business_id"].astype(str)
    # Keep only positives
    df_pos = df[df[pos_col] >= pos_thresh][["user_id", "business_id"]].drop_duplicates()
    return df_pos

def load_recs(recs_path: str) -> pd.DataFrame:
    df = pd.read_csv(recs_path)
    # Coerce to str for safe join with GT
    df["user_id"] = df["user_id"].astype(str)
    df["business_id"] = df["business_id"].astype(str)
    # Ensure sorted by (user, rank)
    df = df.sort_values(["user_id", "rank"], kind="mergesort").reset_index(drop=True)
    return df

def group_to_sets(df: pd.DataFrame, key: str, val: str) -> Dict[str, Set[str]]:
    return df.groupby(key)[val].apply(set).to_dict()

def group_to_list(df: pd.DataFrame, key: str, val: str, k: int) -> Dict[str, List[str]]:
    # Assumes df sorted by rank already
    top = df.groupby(key).head(k)
    return top.groupby(key)[val].apply(list).to_dict()

def dcg_at_k(recommended: List[str], positives: Set[str], k: int) -> float:
    dcg = 0.0
    for i, item in enumerate(recommended[:k], start=1):
        if item in positives:
            dcg += 1.0 / math.log2(i + 1)
    return dcg

def idcg_at_k(positives: Set[str], k: int) -> float:
    # Ideal DCG with all positives ranked first (binary relevance)
    p = min(len(positives), k)
    return sum(1.0 / math.log2(i + 1) for i in range(1, p + 1))

def average_precision_at_k(recommended: List[str], positives: Set[str], k: int) -> float:
    hits = 0
    ap = 0.0
    for i, item in enumerate(recommended[:k], start=1):
        if item in positives:
            hits += 1
            ap += hits / i
    denom = min(len(positives), k)
    return ap / denom if denom > 0 else 0.0

def evaluate(gt_df: pd.DataFrame, recs_df: pd.DataFrame, k: int) -> Tuple[pd.DataFrame, Dict]:
    gt_sets = group_to_sets(gt_df, "user_id", "business_id")
    rec_lists = group_to_list(recs_df, "user_id", "business_id", k)

    rows = []
    users_eval = 0
    recall_sum = precision_sum = ndcg_sum = map_sum = hit_sum = 0.0

    for uid, recs in rec_lists.items():
        positives = gt_sets.get(uid, set())
        if len(positives) == 0:
            # skip users with no positives in GT
            continue

        users_eval += 1
        rec_k = recs[:k]
        rec_set = set(rec_k)
        hits = len(rec_set & positives)

        # metrics
        recall = hits / len(positives) if len(positives) > 0 else 0.0
        precision = hits / k
        ndcg = 0.0
        idcg = idcg_at_k(positives, k)
        if idcg > 0:
            ndcg = dcg_at_k(rec_k, positives, k) / idcg
        ap = average_precision_at_k(rec_k, positives, k)
        hit = 1.0 if hits > 0 else 0.0

        recall_sum += recall
        precision_sum += precision
        ndcg_sum += ndcg
        map_sum += ap
        hit_sum += hit

        rows.append({
            "user_id": uid,
            "n_pos": len(positives),
            "hits@k": hits,
            "recall@k": recall,
            "precision@k": precision,
            "ndcg@k": ndcg,
            "map@k": ap,
            "hit@k": hit,
        })

    if users_eval == 0:
        raise RuntimeError("No evaluable users (no positives in ground truth after filtering).")

    df_user = pd.DataFrame(rows)
    summary = {
        "k": k,
        "users_in_recs": int(recs_df["user_id"].nunique()),
        "users_with_gt": int(len(gt_sets)),
        "users_evaluated": int(users_eval),
        "recall@k": float(recall_sum / users_eval),
        "precision@k": float(precision_sum / users_eval),
        "ndcg@k": float(ndcg_sum / users_eval),
        "map@k": float(map_sum / users_eval),
        "hit_rate@k": float(hit_sum / users_eval),
    }
    return df_user, summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=str, default="data/test/clean_opentable.pkl", help="Ground-truth pickle path")
    parser.add_argument("--recs", type=str, default="results/hybrid_recs_opentable.csv", help="Recommendations CSV path")
    parser.add_argument("--outdir", type=str, default="results", help="Output directory for metrics")
    parser.add_argument("--k", type=int, default=10, help="Top-K")
    parser.add_argument("--pos-col", type=str, default="rating_norm", help="Column used to define positives")
    parser.add_argument("--pos-thresh", type=float, default=0.5, help="Threshold for positive label")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    log(f"[INFO] Load GT: {args.gt}")
    gt_df = load_ground_truth(args.gt, args.pos_col, args.pos_thresh)

    log(f"[INFO] Load RECS: {args.recs}")
    recs_df = load_recs(args.recs)

    log(f"[INFO] Evaluate @K={args.k}")
    df_user, summary = evaluate(gt_df, recs_df, args.k)

    # Save
    user_path = os.path.join(args.outdir, f"eval_per_user_k{args.k}.csv")
    json_path = os.path.join(args.outdir, f"eval_summary_k{args.k}.json")
    df_user.to_csv(user_path, index=False, encoding="utf-8")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Console summary
    log("\n=== Evaluation Summary ===")
    for k, v in summary.items():
        log(f"{k}: {v}")
    log(f"\n[OK] Saved per-user metrics → {user_path}")
    log(f"[OK] Saved summary         → {json_path}")

if __name__ == "__main__":
    main()
