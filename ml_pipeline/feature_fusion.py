"""
Hybrid Feature Fusion Engine
Fuses transaction-level attributes with Neo4j/GDS graph topological indicators:
- Transaction features: Amount, Payment Format, Currency, Cross-bank/currency, Self-transfer, Timing/Hour
- Graph features: PageRank, Betweenness Centrality, In/Out Degrees, Flow Asymmetry,
  Louvain Community ID, Cycle Membership flag, Min Cycle Length.
"""

from typing import Tuple
import numpy as np
import pandas as pd

# Payment formats present in IBM HI-Small, mapped to their one-hot column
PAYMENT_FORMAT_COLUMNS = {
    "wire": "is_wire",
    "ach": "is_ach",
    "cash": "is_cash",
    "cheque": "is_cheque",
    "credit card": "is_credit_card",
    "reinvestment": "is_reinvestment",
    "bitcoin": "is_bitcoin",
}

CURRENCIES = [
    "US Dollar", "Euro", "Canadian Dollar", "Swiss Franc", "Yuan", "Rupee", "Yen", "Shekel",
    "UK Pound", "Ruble", "Bitcoin", "Australian Dollar", "Mexican Peso", "Saudi Riyal", "Brazil Real",
]
CURRENCY_COLUMNS = ["cur_" + c.lower().replace(" ", "_") for c in CURRENCIES]

TABULAR_FEATURE_COLUMNS = [
    "amount",
    "log_amount",
    "hour_of_day",
    *PAYMENT_FORMAT_COLUMNS.values(),
    *CURRENCY_COLUMNS,
    "cross_currency",
    "cross_bank",
    "self_transfer",
]

FEATURE_COLUMNS = [
    # Transaction level
    *TABULAR_FEATURE_COLUMNS,
    # Sender Graph Topology
    "src_pagerank",
    "src_betweenness",
    "src_in_degree",
    "src_out_degree",
    "src_flow_asymmetry",
    "src_in_cycle",
    "src_cycle_count",
    "src_velocity_per_day",
    # Receiver Graph Topology
    "dst_pagerank",
    "dst_betweenness",
    "dst_in_degree",
    "dst_out_degree",
    "dst_flow_asymmetry",
    "dst_in_cycle",
    "dst_cycle_count",
    # Relational Synergy
    "same_community",
    "pagerank_ratio"
]


def build_fused_feature_matrix(
    df: pd.DataFrame,
    account_features_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Merges transaction records with account-level graph features.
    Vectorized column-wise joins so it scales to the full multi-million row dataset.
    Returns:
    - X: Fused feature matrix
    - y: Labels (Is Laundering: 0 or 1)
    - fused_df: Full annotated DataFrame with transaction metadata preserved
    """
    n = len(df)
    df = df.reset_index(drop=True)

    def col(name, default):
        return df[name] if name in df.columns else pd.Series([default] * n, index=df.index)

    src = col("Account", "").astype(str)
    dst = (df["Account.1"] if "Account.1" in df.columns else src).astype(str)
    amount_raw = df["Amount Paid"] if "Amount Paid" in df.columns else col("Amount Received", 0.0)
    amount = pd.to_numeric(amount_raw, errors="coerce").fillna(0.0).astype(float)
    ts_str = col("Timestamp", "2026-09-01 12:00:00").astype(str)

    hour = pd.to_numeric(ts_str.str[11:13], errors="coerce")
    hour = hour.where(ts_str.str[10:11].isin([" ", "T", "/"]), 12).fillna(12).astype(int)

    fmt = col("Payment Format", "Wire").astype(str).str.strip().str.lower()
    pay_cur = col("Payment Currency", "US Dollar").astype(str)
    recv_cur = col("Receiving Currency", "US Dollar").astype(str)
    from_bank = col("From Bank", "BNK_001").astype(str)
    to_bank = col("To Bank", "BNK_002").astype(str)

    label_raw = df["Is Laundering"] if "Is Laundering" in df.columns else col("is_laundering", 0)
    is_laundering = pd.to_numeric(label_raw, errors="coerce").fillna(0).astype(int)
    typology = col("Typology", "NORMAL").astype(str)
    typology = typology.where(~((is_laundering == 1) & (typology == "NORMAL")), "LAUNDERING")

    # Per-account feature lookups (missing accounts fall back to the defaults below)
    if "account_id" in account_features_df.columns and len(account_features_df) > 0:
        unique_feats = account_features_df.drop_duplicates("account_id")
        feats = unique_feats.set_index(unique_feats["account_id"].astype(str))
    else:
        feats = pd.DataFrame()

    def lookup(keys, name, default):
        if name not in feats.columns:
            return pd.Series(default, index=keys.index, dtype=float)
        return keys.map(feats[name]).fillna(default).astype(float)

    src_pr = lookup(src, "pagerank", 0.001)
    dst_pr = lookup(dst, "pagerank", 0.001)
    src_comm = lookup(src, "community_id", -1).astype(int)
    dst_comm = lookup(dst, "community_id", -2).astype(int)

    fused_df = pd.DataFrame({
        "tx_index": np.arange(n),
        "timestamp": ts_str,
        "account_from": src,
        "account_to": dst,
        "from_bank": from_bank,
        "to_bank": to_bank,
        "typology": typology,
        "is_laundering": is_laundering,
        "currency": pay_cur,
        # Features
        "amount": amount,
        "log_amount": np.log1p(amount.clip(lower=0.0)),
        "hour_of_day": hour,
        **{c: (fmt == f).astype(int) for f, c in PAYMENT_FORMAT_COLUMNS.items()},
        **{c: (pay_cur == cur).astype(int) for cur, c in zip(CURRENCIES, CURRENCY_COLUMNS)},
        "cross_currency": (pay_cur != recv_cur).astype(int),
        "cross_bank": (from_bank != to_bank).astype(int),
        "self_transfer": (src == dst).astype(int),
        "src_pagerank": src_pr,
        "src_betweenness": lookup(src, "betweenness", 0.0),
        "src_in_degree": lookup(src, "in_degree", 0).astype(int),
        "src_out_degree": lookup(src, "out_degree", 0).astype(int),
        "src_flow_asymmetry": lookup(src, "flow_asymmetry", 0.0),
        "src_in_cycle": lookup(src, "in_cycle", 0).astype(int),
        "src_cycle_count": lookup(src, "cycle_count", 0).astype(int),
        "src_velocity_per_day": lookup(src, "tx_velocity_per_day", 1.0),
        "dst_pagerank": dst_pr,
        "dst_betweenness": lookup(dst, "betweenness", 0.0),
        "dst_in_degree": lookup(dst, "in_degree", 0).astype(int),
        "dst_out_degree": lookup(dst, "out_degree", 0).astype(int),
        "dst_flow_asymmetry": lookup(dst, "flow_asymmetry", 0.0),
        "dst_in_cycle": lookup(dst, "in_cycle", 0).astype(int),
        "dst_cycle_count": lookup(dst, "cycle_count", 0).astype(int),
        "same_community": ((src_comm == dst_comm) & (src_comm >= 0)).astype(int),
        "pagerank_ratio": src_pr / (dst_pr + 1e-6)
    })

    X = fused_df[FEATURE_COLUMNS].copy()
    y = fused_df["is_laundering"].rename("is_laundering")

    return X, y, fused_df
