"""
Unit tests for Feature Fusion and Matrix Construction
"""

import pytest
import pandas as pd
from data.ibm_aml_loader import generate_ibm_aml_benchmark
from neo4j_engine.algorithms import extract_graph_topological_features
from ml_pipeline.feature_fusion import build_fused_feature_matrix, FEATURE_COLUMNS

@pytest.fixture
def benchmark_data():
    return generate_ibm_aml_benchmark(num_accounts=40, num_normal_tx=120, num_cycles=2)

def test_feature_fusion_matrix(benchmark_data):
    account_features_df, cycles, G = extract_graph_topological_features(benchmark_data)
    X, y, fused_df = build_fused_feature_matrix(benchmark_data, account_features_df)

    assert len(X) == len(benchmark_data)
    assert len(y) == len(benchmark_data)
    assert list(X.columns) == FEATURE_COLUMNS
    assert not X.isnull().values.any(), "Feature matrix must not contain NaN values"
    assert y.isin([0, 1]).all(), "Labels must be binary"

def test_feature_fusion_missing_accounts(benchmark_data):
    # Pass an empty account features dataframe to simulate cold-start / missing accounts
    empty_acc_df = pd.DataFrame(columns=["account_id", "pagerank", "betweenness", "community_id", "in_cycle", "cycle_count", "in_degree", "out_degree", "degree_ratio", "in_volume", "out_volume", "total_volume", "flow_asymmetry", "min_cycle_length", "min_time_delta_hrs", "avg_time_delta_hrs", "tx_velocity_per_day"])
    X, y, fused_df = build_fused_feature_matrix(benchmark_data, empty_acc_df)

    assert len(X) == len(benchmark_data)
    assert not X.isnull().values.any(), "Missing accounts should default to zeros without NaNs"
