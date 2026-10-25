"""
Reproducible leakage-free ablation run.

Usage:
    python3 -m ml_pipeline.run_ablation                # label-blind account sample (fast)
    python3 -m ml_pipeline.run_ablation --full         # full IBM HI-Small window (2022/09/01-09/10)
    python3 -m ml_pipeline.run_ablation --test-graph train   # strictest: test features from training graph
"""

import argparse
import json
import time

from data.ibm_aml_loader import load_ibm_aml_dataset
from ml_pipeline.train_models import AMLModelEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Use the full dataset instead of the cached sample")
    parser.add_argument("--test-graph", choices=["cumulative", "train"], default="cumulative")
    parser.add_argument("--json", help="Optional path to write the metrics JSON")
    args = parser.parse_args()

    start = time.time()
    df = load_ibm_aml_dataset(force_full=args.full)
    print(f"Rows: {len(df):,}  laundering: {int(df['Is Laundering'].sum()):,} "
          f"({df['Is Laundering'].mean() * 100:.3f}%)")

    engine = AMLModelEngine()
    metrics = engine.train_and_evaluate(df, test_graph=args.test_graph)

    split = metrics["split"]
    print(f"Split: temporal, train ends {split['train_end']}, test starts {split['test_start']}, "
          f"test graph = {split['test_graph']}")
    print(f"Test rows: {metrics['test_sample_size']:,}  test positive rate: {split['test_positive_rate'] * 100:.3f}%")
    print()
    print(f"{'Stage':45s} {'PR-AUC':>8s} {'ROC-AUC':>8s} {'Lift':>8s}")
    for s in metrics["ablation_study"]:
        print(f"{s['stage']:45s} {s['pr_auc']:8.4f} {s['roc_auc']:8.4f} {s['incremental_pr_auc_lift']:+8.4f}")
    print()
    for name in ["tabular_baseline", "graph_random_forest", "graph_gradient_boosting"]:
        m = metrics[name]
        print(f"{name:25s} PR-AUC {m['pr_auc']:.4f}  ROC-AUC {m['roc_auc']:.4f}  "
              f"P@100 {m.get('precision_at_k', {}).get('P@100', 0.0):.2f}")
    print(f"PR-AUC lift (graph RF vs tabular): {metrics['pr_auc_lift_pct']}%")
    print(f"Elapsed: {time.time() - start:.0f}s")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({k: v for k, v in metrics.items() if k != "feature_importance"}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
