"""
IBM Transactions for Anti-Money Laundering (AML) — High-Performance Dataset Loader & Benchmark Generator
Dataset: IBM HI-Small_Trans.csv (~5M transactions, 5,177 labelled laundering transactions)
Source:  https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
License: CDLA-Sharing-1.0

Features:
- Reads the real IBM HI-Small_Trans.csv with automatic column duplication resolution ('Account', 'Account.1')
- Label-blind account sampling that keeps the true laundering rate and account neighbourhoods
- Fast-path cached sample support (HI-Small_Trans_sample.csv) for instantaneous sub-second startup
- High-fidelity synthetic benchmark generator with configurable rings and fan-in patterns for unit tests
"""

import os
import random
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
IBM_DATASET_PATH = os.path.join(_DATA_DIR, "HI-Small_Trans.csv")
IBM_SAMPLE_PATH = os.path.join(_DATA_DIR, "HI-Small_Trans_sample.csv")
SYNTHETIC_SAMPLE_PATH = os.path.join(_DATA_DIR, "sample_aml_data.csv")
PRECOMPUTED_METRICS_PATH = os.path.join(_DATA_DIR, "precomputed_metrics.json")

PAYMENT_FORMATS = ["Wire", "ACH", "Credit Card", "Cheque", "Cash"]
CURRENCIES = ["USD", "EUR", "GBP", "CHF"]


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes types and ensures downstream compatibility."""
    df = df.copy()
    df["Amount Paid"] = pd.to_numeric(df.get("Amount Paid", df.get("Amount Received", 0.0)), errors="coerce").fillna(0.0)
    df["Amount Received"] = pd.to_numeric(df.get("Amount Received", df["Amount Paid"]), errors="coerce").fillna(0.0)
    df["Is Laundering"] = pd.to_numeric(df.get("Is Laundering", 0), errors="coerce").fillna(0).astype(int)

    df["Account"] = df["Account"].astype(str)
    df["Account.1"] = df["Account.1"].astype(str)
    df["From Bank"] = df.get("From Bank", "BNK_001").astype(str)
    df["To Bank"] = df.get("To Bank", "BNK_002").astype(str)

    if "Payment Format" not in df.columns:
        df["Payment Format"] = "Wire"
    if "Payment Currency" not in df.columns:
        df["Payment Currency"] = "USD"
    if "Receiving Currency" not in df.columns:
        df["Receiving Currency"] = "USD"
    if "Timestamp" not in df.columns:
        df["Timestamp"] = "2026-09-01 10:00:00"

    if "Typology" not in df.columns:
        df["Typology"] = df["Is Laundering"].apply(lambda x: "LAUNDERING" if x == 1 else "NORMAL")

    return df


# The IBM simulator stops generating normal traffic after 2022/09/10; the remaining few hundred
# rows are ~60% laundering, so any time-based feature would trivially detect them.
SIMULATION_END = "2022/09/11"


def _read_raw_window(raw_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path, dtype={"Account": str, "Account.1": str})
    return df[df["Timestamp"] < SIMULATION_END]


def create_account_sample_from_raw(
    raw_path: str = IBM_DATASET_PATH,
    output_path: str = IBM_SAMPLE_PATH,
    account_fraction: float = 0.01,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Label-blind sample: picks a random fraction of accounts and keeps every transaction
    they send or receive. Laundering and normal accounts are sampled at the same rate, so
    the true ~0.1% laundering rate and each sampled account's full neighbourhood are preserved.
    For the dashboard demo sample (which also keeps whole detected rings) use
    `python3 -m ml_pipeline.build_artifacts`.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}")

    print(f"Creating label-blind account sample from {raw_path}...")
    df = _read_raw_window(raw_path)
    accounts = pd.Index(pd.unique(pd.concat([df["Account"], df["Account.1"]])))
    rng = np.random.default_rng(random_state)
    selected = accounts[rng.random(len(accounts)) < account_fraction]
    mask = df["Account"].isin(selected) | df["Account.1"].isin(selected)

    df_sample = _standardize_columns(df[mask].reset_index(drop=True))
    df_sample.to_csv(output_path, index=False)
    print(f"Account sample saved to {output_path} ({len(df_sample):,} rows, "
          f"{int(df_sample['Is Laundering'].sum()):,} laundering)")
    return df_sample


def load_ibm_aml_dataset(
    filepath: Optional[str] = None,
    sample_size: int = 30000,
    random_state: int = 42,
    force_full: bool = False
) -> pd.DataFrame:
    """
    Loads IBM HI-Small AML dataset. Uses pre-sampled cache if available for instant startup.
    """
    # 1. Check if specific filepath was given
    if filepath and os.path.exists(filepath):
        if os.path.getsize(filepath) > 50_000_000 and not force_full:
            # It's the 475MB raw file: check if sample cache exists
            if os.path.exists(IBM_SAMPLE_PATH):
                print(f"  Fast-loading real IBM HI-Small dataset from sample cache: {IBM_SAMPLE_PATH}")
                df = pd.read_csv(IBM_SAMPLE_PATH)
                return _standardize_columns(df)
            else:
                return create_account_sample_from_raw(raw_path=filepath, random_state=random_state)
        else:
            df = pd.read_csv(filepath)
            return _standardize_columns(df)

    # 2. Check sample cache
    if os.path.exists(IBM_SAMPLE_PATH) and not force_full:
        print(f"  Loading real IBM HI-Small dataset from cache: {IBM_SAMPLE_PATH}")
        df = pd.read_csv(IBM_SAMPLE_PATH)
        return _standardize_columns(df)

    # 3. Check raw CSV
    if os.path.exists(IBM_DATASET_PATH):
        if force_full:
            print(f"  Loading full 5M+ IBM AML dataset from {IBM_DATASET_PATH}...")
            return _standardize_columns(_read_raw_window(IBM_DATASET_PATH))
        else:
            return create_account_sample_from_raw(random_state=random_state)

    # 4. Fallback to synthetic sample if real CSV not downloaded
    if os.path.exists(SYNTHETIC_SAMPLE_PATH):
        print(f"  Loading synthetic benchmark sample from {SYNTHETIC_SAMPLE_PATH}")
        df = pd.read_csv(SYNTHETIC_SAMPLE_PATH)
        return _standardize_columns(df)

    # 5. Generate benchmark on the fly
    return generate_ibm_aml_benchmark()


def load_aml_data(filepath: Optional[str] = None, sample_size: int = 30000) -> pd.DataFrame:
    """Entry point for API and services."""
    return load_ibm_aml_dataset(filepath=filepath, sample_size=sample_size)


def generate_ibm_aml_benchmark(
    num_accounts: int = 150,
    num_normal_tx: int = 1200,
    num_cycles: int = 6,
    num_fan_in_patterns: int = 4,
    random_seed: int = 42,
    output_path: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Generates a realistic AML benchmark adhering to IBM HI-Small schema.
    Used for unit testing and fast isolated verification.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    accounts = [f"ACC_{i:04d}" for i in range(1, num_accounts + 1)]
    banks = [f"BNK_{i:03d}" for i in range(1, max(3, num_accounts // 10) + 1)]
    acc_bank_map = {acc: random.choice(banks) for acc in accounts}

    base_time = datetime(2026, 9, 1, 8, 0, 0)
    tx_list = []

    # 1. Normal transactions
    for _ in range(num_normal_tx):
        src, dst = random.sample(accounts, 2)
        amount = round(float(np.random.lognormal(mean=6.5, sigma=1.2)), 2)
        tx_time = base_time + timedelta(
            days=random.randint(0, 14),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        tx_list.append({
            "Timestamp": tx_time.strftime("%Y/%m/%d %H:%M:%S"),
            "From Bank": acc_bank_map[src],
            "Account": src,
            "To Bank": acc_bank_map[dst],
            "Account.1": dst,
            "Amount Received": amount,
            "Receiving Currency": "USD",
            "Amount Paid": amount,
            "Payment Currency": "USD",
            "Payment Format": random.choice(PAYMENT_FORMATS),
            "Is Laundering": 0,
            "Typology": "NORMAL"
        })

    # 2. Circular layering rings (length 3, 4, 5)
    used_in_cycles = set()
    for c_idx in range(num_cycles):
        cycle_len = random.choice([3, 4, 5])
        avail_accs = [a for a in accounts if a not in used_in_cycles]
        if len(avail_accs) < cycle_len:
            avail_accs = accounts
        ring_nodes = random.sample(avail_accs, cycle_len)
        used_in_cycles.update(ring_nodes)

        base_amount = random.uniform(15000.0, 95000.0)
        ring_start_time = base_time + timedelta(days=random.randint(1, 10), hours=random.randint(0, 12))

        for step in range(cycle_len):
            u = ring_nodes[step]
            v = ring_nodes[(step + 1) % cycle_len]
            step_amount = round(base_amount * (1.0 - (step * 0.015)), 2)
            step_time = ring_start_time + timedelta(hours=step * 2, minutes=random.randint(5, 30))
            tx_list.append({
                "Timestamp": step_time.strftime("%Y/%m/%d %H:%M:%S"),
                "From Bank": acc_bank_map[u],
                "Account": u,
                "To Bank": acc_bank_map[v],
                "Account.1": v,
                "Amount Received": step_amount,
                "Receiving Currency": "USD",
                "Amount Paid": step_amount,
                "Payment Currency": "USD",
                "Payment Format": "Wire",
                "Is Laundering": 1,
                "Typology": f"CYCLE_RING_{cycle_len}"
            })

    # 3. Fan-in Smurfing / Structuring patterns
    for _ in range(num_fan_in_patterns):
        mule = random.choice(accounts)
        smurfs = [a for a in random.sample(accounts, 4) if a != mule]
        fan_in_time = base_time + timedelta(days=random.randint(2, 12))
        for smurf in smurfs:
            struct_amt = round(random.uniform(8500.0, 9900.0), 2)
            tx_list.append({
                "Timestamp": (fan_in_time + timedelta(minutes=random.randint(10, 180))).strftime("%Y/%m/%d %H:%M:%S"),
                "From Bank": acc_bank_map[smurf],
                "Account": smurf,
                "To Bank": acc_bank_map[mule],
                "Account.1": mule,
                "Amount Received": struct_amt,
                "Receiving Currency": "USD",
                "Amount Paid": struct_amt,
                "Payment Currency": "USD",
                "Payment Format": "Cash",
                "Is Laundering": 1,
                "Typology": "FAN_IN_STRUCTURING"
            })

    df = pd.DataFrame(tx_list)
    df = df.sort_values("Timestamp").reset_index(drop=True)

    if output_path:
        df.to_csv(output_path, index=False)

    return df
