"""
AML Model Trainer, Comparative Evaluation, Ablation Study & SHAP Explainability Engine
Implements:
- Model 1: Purely Tabular ML Baseline (Random Forest)
- Model 2: Graph-Augmented Random Forest (balanced weights, multi-threaded)
- Model 3: Graph-Augmented HistGradientBoosting (class-balanced sample weights)
- Ablation Study: Quantifies incremental PR-AUC lift of each graph feature stage
- SHAP Explainability: Explains individual fraud risk scores for forensic investigators
"""

import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from neo4j_engine.algorithms import extract_graph_topological_features
from ml_pipeline.feature_fusion import FEATURE_COLUMNS, TABULAR_FEATURE_COLUMNS, build_fused_feature_matrix
from ml_pipeline.evaluation import evaluate_aml_predictions

TABULAR_ONLY_COLUMNS = TABULAR_FEATURE_COLUMNS

# CPU cores used for model training; keep low so laptops stay responsive
N_JOBS = int(os.getenv("AML_N_JOBS", "2"))

# Correct column names mapped to FEATURE_COLUMNS
ABLATION_STAGES = [
    {
        "stage": "Tabular Baseline",
        "description": "Amount, hour of day, payment format & currency encodings, cross-currency, cross-bank, self-transfer",
        "features": TABULAR_ONLY_COLUMNS
    },
    {
        "stage": "+ PageRank Centrality",
        "description": "Tabular + PageRank structural influence score & ratio",
        "features": TABULAR_ONLY_COLUMNS + ["src_pagerank", "dst_pagerank", "pagerank_ratio"]
    },
    {
        "stage": "+ Betweenness Centrality",
        "description": "Tabular + PageRank + Betweenness money-mule bridge score",
        "features": TABULAR_ONLY_COLUMNS + ["src_pagerank", "dst_pagerank", "pagerank_ratio", "src_betweenness", "dst_betweenness"]
    },
    {
        "stage": "+ Community & Flow Asymmetry",
        "description": "Tabular + Centrality + Louvain community match + In/Out flow asymmetry & degrees",
        "features": TABULAR_ONLY_COLUMNS + [
            "src_pagerank", "dst_pagerank", "pagerank_ratio",
            "src_betweenness", "dst_betweenness",
            "same_community",
            "src_flow_asymmetry", "dst_flow_asymmetry",
            "src_in_degree", "src_out_degree",
            "dst_in_degree", "dst_out_degree",
            "src_velocity_per_day"
        ]
    },
    {
        "stage": "+ Multi-Hop Cycle Ring Flags (Full Model)",
        "description": "All features + Directed cycle detection (3-5 hops) ring flags",
        "features": FEATURE_COLUMNS
    }
]


class AMLModelEngine:
    def __init__(self):
        self.rf_model = None
        self.gb_model = None
        self.tabular_baseline_model = None
        self.feature_names = FEATURE_COLUMNS
        self.metrics: Dict[str, Any] = {}
        self.feature_importances: List[Dict[str, Any]] = []
        self.ablation_results: List[Dict[str, Any]] = []
        self.shap_summary: Dict[str, Any] = {}
        self.fused_df: pd.DataFrame = None
        self.split_info: Dict[str, Any] = {}

    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        account_features_df: Optional[pd.DataFrame] = None,
        test_size: float = 0.25,
        random_state: int = 42,
        test_graph: str = "cumulative",
        max_train_negatives: Optional[int] = 300000,
        score_all: bool = True
    ) -> Dict[str, Any]:
        """
        Runs a leakage-free evaluation: temporal train/test split, graph features rebuilt
        from data available at each point in time, model training, comparative evaluation,
        ablation study, and SHAP feature impact analysis.

        - Train rows are the earliest (1 - test_size) of transactions by timestamp; test rows
          are the latest test_size.
        - Training features come from a graph of training-period transactions only.
        - test_graph="cumulative": test features come from the graph of all transactions up to
          the end of the test period (labels are never read, as at deployment time).
          test_graph="train": test features reuse the training-period graph (strictest).
        - max_train_negatives: optional random undersampling of training negatives for speed.
          The test set is never resampled, so metrics reflect the true class balance.
        - account_features_df: features for scoring the full dataset (investigator leads);
          defaults to the cumulative graph. It is not used for evaluation.
        - score_all: set False to skip scoring every row (offline evaluation runs).
        """
        ts = pd.to_datetime(df["Timestamp"], errors="coerce") if "Timestamp" in df.columns else pd.Series(pd.NaT, index=df.index)
        ordered = df.assign(_ts=ts).sort_values("_ts", kind="stable").drop(columns="_ts")
        n_train = int(round(len(ordered) * (1.0 - test_size)))
        train_df = ordered.iloc[:n_train]
        test_df = ordered.iloc[n_train:]

        train_feats, _, _ = extract_graph_topological_features(train_df)
        if test_graph == "train":
            test_feats = train_feats
        else:
            test_feats = account_features_df if account_features_df is not None else extract_graph_topological_features(df)[0]

        X_train, y_train, _ = build_fused_feature_matrix(train_df, train_feats)
        X_test, y_test, _ = build_fused_feature_matrix(test_df, test_feats)

        if max_train_negatives is not None and (y_train == 0).sum() > max_train_negatives:
            rng = np.random.default_rng(random_state)
            neg_idx = np.flatnonzero(y_train.values == 0)
            keep = np.sort(np.concatenate([
                np.flatnonzero(y_train.values == 1),
                rng.choice(neg_idx, size=max_train_negatives, replace=False)
            ]))
            X_train, y_train = X_train.iloc[keep], y_train.iloc[keep]

        self.split_info = {
            "method": "temporal",
            "test_graph": test_graph,
            "train_end": str(ordered["Timestamp"].iloc[n_train - 1]) if n_train > 0 else None,
            "test_start": str(ordered["Timestamp"].iloc[n_train]) if n_train < len(ordered) else None,
            "train_positive_rate": round(float(y_train.mean()), 5) if len(y_train) else 0.0,
            "test_positive_rate": round(float(y_test.mean()), 5) if len(y_test) else 0.0
        }

        # 1. Baseline: Tabular-only Random Forest
        X_train_tab = X_train[TABULAR_ONLY_COLUMNS]
        X_test_tab = X_test[TABULAR_ONLY_COLUMNS]
        self.tabular_baseline_model = RandomForestClassifier(
            n_estimators=80, max_depth=8, class_weight="balanced", random_state=random_state, n_jobs=N_JOBS
        )
        self.tabular_baseline_model.fit(X_train_tab, y_train)
        tab_probs = self.tabular_baseline_model.predict_proba(X_test_tab)[:, 1] if len(self.tabular_baseline_model.classes_) > 1 else np.zeros(len(X_test_tab))
        tabular_metrics = evaluate_aml_predictions(y_test.values, tab_probs)

        # 2. Graph-Augmented Random Forest
        self.rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=10, class_weight="balanced", random_state=random_state, n_jobs=N_JOBS
        )
        self.rf_model.fit(X_train, y_train)
        rf_probs = self.rf_model.predict_proba(X_test)[:, 1] if len(self.rf_model.classes_) > 1 else np.zeros(len(X_test))
        rf_metrics = evaluate_aml_predictions(y_test.values, rf_probs)

        # 3. Graph-Augmented Gradient Boosting
        neg_count = max(1, (y_train == 0).sum())
        pos_count = max(1, (y_train == 1).sum())
        pos_weight = float(neg_count / pos_count)
        sample_weights = np.where(y_train == 1, pos_weight, 1.0)

        self.gb_model = HistGradientBoostingClassifier(
            max_iter=80, random_state=random_state
        )
        self.gb_model.fit(X_train, y_train, sample_weight=sample_weights)
        gb_probs = self.gb_model.predict_proba(X_test)[:, 1] if len(self.gb_model.classes_) > 1 else np.zeros(len(X_test))
        gb_metrics = evaluate_aml_predictions(y_test.values, gb_probs)

        # 4. Feature Importance
        rf_importances = self.rf_model.feature_importances_
        sorted_indices = np.argsort(rf_importances)[::-1]
        self.feature_importances = [
            {
                "feature": self.feature_names[i],
                "importance": round(float(rf_importances[i]), 4),
                "is_graph_feature": self.feature_names[i] not in TABULAR_ONLY_COLUMNS
            }
            for i in sorted_indices
        ]

        # 5. Ablation Study Execution
        self.ablation_results = self._run_ablation_study(X_train, X_test, y_train, y_test, random_state)

        # 6. SHAP Explainability Approximation
        self.shap_summary = self._compute_shap_insights(X_test, y_test)

        # 7. Generate Predictions & Risk Scores for entire dataset (triage, not evaluation)
        if score_all:
            scoring_feats = account_features_df if account_features_df is not None else test_feats
            X, _, fused_df = build_fused_feature_matrix(df, scoring_feats)
            self.fused_df = fused_df
            all_rf_probs = self.rf_model.predict_proba(X)[:, 1] if len(self.rf_model.classes_) > 1 else np.zeros(len(X))
            all_gb_probs = self.gb_model.predict_proba(X)[:, 1] if len(self.gb_model.classes_) > 1 else np.zeros(len(X))

            # Ensemble risk score: 50% RF + 50% GB
            fused_df["risk_score"] = np.round((all_rf_probs + all_gb_probs) / 2.0, 4)
            fused_df["predicted_label"] = (fused_df["risk_score"] >= 0.5).astype(int)
            # Rows from the test period were never seen in training; the triage queue ranks only these
            test_start = self.split_info.get("test_start")
            fused_df["in_test_period"] = (fused_df["timestamp"] >= test_start).astype(int) if test_start else 1

        self.metrics = {
            "tabular_baseline": tabular_metrics,
            "graph_random_forest": rf_metrics,
            "graph_gradient_boosting": gb_metrics,
            "feature_importance": self.feature_importances,
            "ablation_study": self.ablation_results,
            "shap_summary": self.shap_summary,
            "split": self.split_info,
            "test_sample_size": len(y_test),
            "train_sample_size": len(y_train),
            "pr_auc_lift_pct": round(((rf_metrics["pr_auc"] - tabular_metrics["pr_auc"]) / max(0.01, tabular_metrics["pr_auc"])) * 100, 1)
        }

        return self.metrics

    def _run_ablation_study(self, X_train, X_test, y_train, y_test, random_state: int) -> List[Dict[str, Any]]:
        """
        Incrementally adds feature categories and calculates the exact PR-AUC and ROC-AUC lift.
        """
        results = []
        prev_pr_auc = 0.0

        for i, stage in enumerate(ABLATION_STAGES):
            cols = [c for c in stage["features"] if c in X_train.columns]
            X_tr = X_train[cols]
            X_te = X_test[cols]

            model = RandomForestClassifier(
                n_estimators=60, max_depth=8, class_weight="balanced", random_state=random_state, n_jobs=N_JOBS
            )
            model.fit(X_tr, y_train)
            probs = model.predict_proba(X_te)[:, 1] if len(model.classes_) > 1 else np.zeros(len(X_te))
            metrics = evaluate_aml_predictions(y_test.values, probs)

            pr_auc = metrics["pr_auc"]
            roc_auc = metrics["roc_auc"]
            lift = round(pr_auc - prev_pr_auc, 4) if i > 0 else 0.0
            prev_pr_auc = pr_auc

            results.append({
                "stage_num": i + 1,
                "stage": stage["stage"],
                "description": stage["description"],
                "feature_count": len(cols),
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "f1_score": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "incremental_pr_auc_lift": lift
            })

        return results

    def _compute_shap_insights(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Calculates SHAP values or feature impact contributions for the graph-augmented model.
        """
        try:
            import shap
            sample_X = X_test.head(80)
            explainer = shap.TreeExplainer(self.rf_model)
            shap_values = explainer.shap_values(sample_X)

            if isinstance(shap_values, list) and len(shap_values) > 1:
                vals = shap_values[1]
            elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                vals = shap_values[:, :, 1]
            else:
                vals = shap_values

            mean_abs_shap = np.mean(np.abs(vals), axis=0)
            shap_ranking = [
                {
                    "feature": self.feature_names[i],
                    "mean_shap_value": round(float(mean_abs_shap[i]), 4),
                    "impact": "High Positive Risk Influence" if mean_abs_shap[i] > np.median(mean_abs_shap) else "Moderate Context"
                }
                for i in np.argsort(mean_abs_shap)[::-1]
            ]

            return {
                "status": "computed",
                "ranking": shap_ranking,
                "top_driving_feature": shap_ranking[0]["feature"] if shap_ranking else "src_in_cycle"
            }
        except Exception:
            ranking = [
                {
                    "feature": item["feature"],
                    "mean_shap_value": item["importance"],
                    "impact": "High Positive Risk Influence" if i < 5 else "Moderate Context"
                }
                for i, item in enumerate(self.feature_importances[:12])
            ]
            return {
                "status": "approximated",
                "ranking": ranking,
                "top_driving_feature": ranking[0]["feature"] if ranking else "src_in_cycle"
            }

    def get_top_prioritized_leads(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Returns highest-risk transactions prioritized for AML forensic investigators,
        drawn from the test period so the queue shows out-of-sample behaviour.
        """
        if self.fused_df is None:
            return []

        pool = self.fused_df
        if "in_test_period" in pool.columns and pool["in_test_period"].any():
            pool = pool[pool["in_test_period"] == 1]
        leads = pool.sort_values(by="risk_score", ascending=False).head(limit)
        return leads.to_dict(orient="records")

    def explain_account_lead(self, transaction_index: int) -> Dict[str, Any]:
        """
        Returns forensic risk breakdown explaining why a specific transaction was flagged.
        """
        if self.fused_df is None:
            return {"error": "Pipeline not initialized"}

        rows = self.fused_df[self.fused_df["tx_index"] == transaction_index]
        if rows.empty:
            if 0 <= transaction_index < len(self.fused_df):
                row = self.fused_df.iloc[transaction_index]
            else:
                return {"error": f"Transaction {transaction_index} not found"}
        else:
            row = rows.iloc[0]

        risk_factors = []

        if row.get("src_in_cycle", 0) == 1 or row.get("dst_in_cycle", 0) == 1:
            risk_factors.append({
                "factor": "Circular Laundering Ring Participation",
                "severity": "CRITICAL",
                "details": "Account detected in a closed directed multi-hop transfer loop (lengths 3-5)."
            })

        if row.get("src_betweenness", 0) > 0.02 or row.get("dst_betweenness", 0) > 0.02:
            risk_factors.append({
                "factor": "High Betweenness Centrality (Money Mule Bridge)",
                "severity": "HIGH",
                "details": "Account acts as a topological intermediary connecting disparate subgraphs."
            })

        if row.get("src_pagerank", 0) > 0.005 or row.get("dst_pagerank", 0) > 0.005:
            risk_factors.append({
                "factor": "High PageRank Network Influence",
                "severity": "MEDIUM",
                "details": "Account has high relative structural weight in the financial flow topology."
            })

        if row.get("amount", 0) > 10000:
            risk_factors.append({
                "factor": "High-Value Transaction Threshold",
                "severity": "MEDIUM",
                "details": f"Transaction amount ${row.get('amount', 0):,.2f} exceeds threshold limits."
            })

        return {
            "transaction_index": transaction_index,
            "risk_score": float(row.get("risk_score", 0.0)),
            "predicted_label": int(row.get("predicted_label", 0)),
            "actual_label": int(row.get("is_laundering", row.get("Is Laundering", 0))),
            "from_account": str(row.get("account_from", row.get("Account", ""))),
            "to_account": str(row.get("account_to", row.get("Account.1", ""))),
            "amount": float(row.get("amount", 0.0)),
            "risk_factors": risk_factors
        }
