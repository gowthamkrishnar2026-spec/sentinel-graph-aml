"""
Unit tests for Machine Learning Pipeline, Evaluation, Ablation Study & Precision@K
"""

import pytest
from data.ibm_aml_loader import generate_ibm_aml_benchmark
from neo4j_engine.algorithms import extract_graph_topological_features
from ml_pipeline.train_models import AMLModelEngine
from ml_pipeline.evaluation import evaluate_aml_predictions, compute_precision_at_k
import numpy as np

@pytest.fixture
def trained_engine():
    df = generate_ibm_aml_benchmark(num_accounts=60, num_normal_tx=200, num_cycles=3)
    account_features_df, cycles, G = extract_graph_topological_features(df)
    engine = AMLModelEngine()
    engine.train_and_evaluate(df, account_features_df)
    return engine

def test_model_training_and_metrics(trained_engine):
    metrics = trained_engine.metrics
    assert "tabular_baseline" in metrics
    assert "graph_random_forest" in metrics
    assert "graph_gradient_boosting" in metrics
    assert "ablation_study" in metrics
    assert "feature_importance" in metrics

    # PR-AUC of graph model should be high
    graph_rf_pr = metrics["graph_random_forest"]["pr_auc"]
    assert 0.0 <= graph_rf_pr <= 1.0

def test_ablation_study(trained_engine):
    ablation = trained_engine.ablation_results
    assert len(ablation) == 5
    assert ablation[0]["stage"] == "Tabular Baseline"
    assert ablation[-1]["stage"] == "+ Multi-Hop Cycle Ring Flags (Full Model)"

    # Verify each stage produces valid PR-AUC
    for stage in ablation:
        assert 0.0 <= stage["pr_auc"] <= 1.0
        assert 0.0 <= stage["roc_auc"] <= 1.0

def test_precision_at_k():
    y_true = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    p_at_k = compute_precision_at_k(y_true, y_prob, k_values=[2, 4, 10])

    assert p_at_k["P@2"] == 1.0   # top 2 are both 1
    assert p_at_k["P@4"] == 0.75  # 3 of top 4 are 1
    assert p_at_k["P@10"] == 0.3  # 3 of 10 are 1


def test_temporal_split_has_no_overlap(trained_engine):
    split = trained_engine.split_info
    assert split["method"] == "temporal"
    assert split["train_end"] <= split["test_start"]
