"""
One-off offline build of the artifacts the dashboard serves.

Reads the full IBM HI-Small window once (~2-4 minutes, a few GB of RAM) and writes:
- data/precomputed_metrics.json : leakage-free full-dataset evaluation (strict temporal split,
                                  training-period graph features only), ablation and SHAP
- data/HI-Small_Trans_sample.csv: light demo sample (~1% of accounts plus the transactions
                                  between members of the largest detected rings)

Ring members are found by label-blind cycle detection, so the sample never selects on labels.

Usage:
    python3 -m ml_pipeline.build_artifacts            # requires data/HI-Small_Trans.csv
"""

import json
import os
import time

import numpy as np
import pandas as pd

from data.ibm_aml_loader import (
    IBM_DATASET_PATH, IBM_SAMPLE_PATH, PRECOMPUTED_METRICS_PATH,
    _read_raw_window, _standardize_columns
)
from neo4j_engine.algorithms import build_aml_digraph, detect_laundering_cycles
from ml_pipeline.train_models import AMLModelEngine


def build_demo_sample(df: pd.DataFrame, account_fraction: float = 0.01, ring_count: int = 150,
                      random_state: int = 42) -> pd.DataFrame:
    accounts = pd.Index(pd.unique(pd.concat([df["Account"], df["Account.1"]])))
    rng = np.random.default_rng(random_state)
    selected = accounts[rng.random(len(accounts)) < account_fraction]
    ego_mask = df["Account"].isin(selected) | df["Account.1"].isin(selected)

    cycles = detect_laundering_cycles(build_aml_digraph(df), max_results=ring_count)
    ring_accounts = {a for c in cycles for a in c["accounts"]}
    ring_mask = df["Account"].isin(ring_accounts) & df["Account.1"].isin(ring_accounts)
    print(f"  Rings: {len(cycles)} across {len(ring_accounts)} accounts, "
          f"{int(ring_mask.sum()):,} ring transactions")

    return _standardize_columns(df[ego_mask | ring_mask].reset_index(drop=True))


def main():
    if not os.path.exists(IBM_DATASET_PATH):
        raise SystemExit(f"Full dataset not found at {IBM_DATASET_PATH}")

    start = time.time()
    print("Loading full dataset window...")
    df = _standardize_columns(_read_raw_window(IBM_DATASET_PATH))
    print(f"  {len(df):,} rows, {int(df['Is Laundering'].sum()):,} laundering")

    print("Running strict temporal evaluation...")
    engine = AMLModelEngine()
    metrics = engine.train_and_evaluate(df, test_graph="train", score_all=False)
    metrics["dataset"] = {
        "rows": int(len(df)),
        "laundering": int(df["Is Laundering"].sum()),
        "scope": "IBM HI-Small, 2022/09/01-2022/09/10 (full window)"
    }
    with open(PRECOMPUTED_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=1, default=str)
    print(f"  Saved {PRECOMPUTED_METRICS_PATH}")

    print("Building demo sample...")
    sample = build_demo_sample(df)
    sample.to_csv(IBM_SAMPLE_PATH, index=False)
    print(f"  Saved {IBM_SAMPLE_PATH} ({len(sample):,} rows, {int(sample['Is Laundering'].sum()):,} laundering)")
    print(f"Done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
