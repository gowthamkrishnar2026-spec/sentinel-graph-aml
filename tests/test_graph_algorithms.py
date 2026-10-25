"""
Unit tests for AML Graph Algorithms & Topological Feature Extraction
"""

import pytest
import pandas as pd
import networkx as nx
from neo4j_engine.algorithms import (
    build_aml_digraph,
    detect_laundering_cycles,
    extract_graph_topological_features
)

@pytest.fixture
def sample_ring_data():
    """Generates a closed 4-node circular layering ring + 2 normal transfers."""
    data = [
        # 4-node closed cycle: ACC_01 -> ACC_02 -> ACC_03 -> ACC_04 -> ACC_01
        {"Account": "ACC_01", "From Bank": "BNK_01", "Account.1": "ACC_02", "To Bank": "BNK_02", "Amount Paid": 15000.0, "Timestamp": "2026-09-01 10:00:00", "Payment Format": "Wire", "Is Laundering": 1, "Typology": "CYCLE_RING_4"},
        {"Account": "ACC_02", "From Bank": "BNK_02", "Account.1": "ACC_03", "To Bank": "BNK_03", "Amount Paid": 14500.0, "Timestamp": "2026-09-01 11:00:00", "Payment Format": "Wire", "Is Laundering": 1, "Typology": "CYCLE_RING_4"},
        {"Account": "ACC_03", "From Bank": "BNK_03", "Account.1": "ACC_04", "To Bank": "BNK_04", "Amount Paid": 14000.0, "Timestamp": "2026-09-01 12:00:00", "Payment Format": "Wire", "Is Laundering": 1, "Typology": "CYCLE_RING_4"},
        {"Account": "ACC_04", "From Bank": "BNK_04", "Account.1": "ACC_01", "To Bank": "BNK_01", "Amount Paid": 13800.0, "Timestamp": "2026-09-01 13:00:00", "Payment Format": "Wire", "Is Laundering": 1, "Typology": "CYCLE_RING_4"},
        # Normal transfer
        {"Account": "ACC_NORM_1", "From Bank": "BNK_01", "Account.1": "ACC_NORM_2", "To Bank": "BNK_02", "Amount Paid": 500.0, "Timestamp": "2026-09-01 14:00:00", "Payment Format": "ACH", "Is Laundering": 0, "Typology": "NORMAL"},
    ]
    return pd.DataFrame(data)

def test_build_aml_digraph(sample_ring_data):
    G = build_aml_digraph(sample_ring_data)
    assert G.number_of_nodes() == 6
    assert G.number_of_edges() == 5
    assert G.has_edge("ACC_01", "ACC_02")
    assert G["ACC_01"]["ACC_02"]["weight"] == 15000.0

def test_detect_laundering_cycles(sample_ring_data):
    G = build_aml_digraph(sample_ring_data)
    cycles = detect_laundering_cycles(G, min_len=3, max_len=5)
    assert len(cycles) == 1
    c = cycles[0]
    assert c["length"] == 4
    assert set(c["accounts"]) == {"ACC_01", "ACC_02", "ACC_03", "ACC_04"}
    assert c["total_volume"] == (15000.0 + 14500.0 + 14000.0 + 13800.0)

def test_extract_graph_topological_features(sample_ring_data):
    feat_df, cycles, G = extract_graph_topological_features(sample_ring_data)
    assert len(feat_df) == 6
    assert "pagerank" in feat_df.columns
    assert "betweenness" in feat_df.columns
    assert "community_id" in feat_df.columns
    assert "in_cycle" in feat_df.columns

    # Verify cycle flag is set for ring accounts
    ring_accs = feat_df[feat_df["account_id"].isin(["ACC_01", "ACC_02", "ACC_03", "ACC_04"])]
    assert (ring_accs["in_cycle"] == 1).all()

    norm_acc = feat_df[feat_df["account_id"] == "ACC_NORM_1"].iloc[0]
    assert norm_acc["in_cycle"] == 0


def test_cycle_detection_ignores_labels(sample_ring_data):
    # Cycle flags are model features, so they must not depend on the laundering label
    relabeled = sample_ring_data.copy()
    relabeled["Is Laundering"] = 1 - relabeled["Is Laundering"]
    original = detect_laundering_cycles(build_aml_digraph(sample_ring_data))
    flipped = detect_laundering_cycles(build_aml_digraph(relabeled))
    assert [c["accounts"] for c in original] == [c["accounts"] for c in flipped]
