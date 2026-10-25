"""
FastAPI Backend Application for AML Forensic Analytics
Serves graph topology, cycle detection, ML predictions, ablation studies, SHAP explanations,
SAR generation, and live Neo4j Bolt synchronization.

Uses the real IBM HI-Small AML dataset (5,078,345 transactions, 5,177 labelled laundering transactions).
"""

import json
import os
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data.ibm_aml_loader import load_aml_data, load_ibm_aml_dataset, PRECOMPUTED_METRICS_PATH
from neo4j_engine.graph_service import AMLGraphService
from ml_pipeline.train_models import AMLModelEngine
from ml_pipeline.sar_generator import generate_sar_report

# Global State
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.getenv("DATA_PATH", os.path.join(BASE_DIR, "data", "HI-Small_Trans.csv"))
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")

graph_service = AMLGraphService()
model_engine = AMLModelEngine()
aml_data = None
precomputed_metrics = {}


def evaluation_metrics() -> dict:
    """Full-dataset evaluation from ml_pipeline.build_artifacts when available, else the live sample run."""
    return precomputed_metrics or model_engine.metrics


def evaluation_scope() -> str:
    if precomputed_metrics:
        rows = precomputed_metrics.get("dataset", {}).get("rows", 0)
        return f"Full IBM HI-Small ({rows:,} tx), strict temporal split"
    return "Live demo sample, temporal split"


def graph_lift_pct(metrics: dict) -> float:
    tab = metrics.get("tabular_baseline", {}).get("pr_auc", 0.0)
    graph = metrics.get("graph_gradient_boosting", {}).get("pr_auc", 0.0)
    return round((graph - tab) / max(tab, 1e-6) * 100, 1) if metrics else 0.0


def init_system(sample_size: int = 30000, force_full: bool = False):
    global aml_data, precomputed_metrics
    print("=" * 60)
    print("  SENTINEL GRAPH // AML NETWORK FORENSICS")
    print("  IBM HI-Small AML Dataset — Authentic Transaction Topology")
    print("=" * 60)

    print("\n[1/3] Loading IBM AML Dataset...")
    aml_data = load_aml_data(DATA_PATH, sample_size=sample_size)
    print(f"  Loaded {len(aml_data):,} transactions")

    print("\n[2/3] Building graph topology & running GDS feature extraction...")
    graph_service.load_dataset(aml_data, sync_to_neo4j=False)
    print(f"  Graph: {graph_service.G.number_of_nodes():,} nodes, {graph_service.G.number_of_edges():,} edges")
    print(f"  Cycles detected: {len(graph_service.cycles)}")

    print("\n[3/3] Training ML Classifiers & executing Ablation Study...")
    model_engine.train_and_evaluate(aml_data, graph_service.account_features)

    if os.path.exists(PRECOMPUTED_METRICS_PATH):
        with open(PRECOMPUTED_METRICS_PATH) as f:
            precomputed_metrics = json.load(f)
    print(f"  Evaluation: {evaluation_scope()}")
    print(f"  PR-AUC lift (graph GB vs tabular): {graph_lift_pct(evaluation_metrics()):+}%")

    print("\n" + "=" * 60)
    print("  AML FORENSIC PLATFORM READY")
    print("  Dashboard: http://localhost:8000")
    print("=" * 60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_system()
    yield


app = FastAPI(
    title="Fraud Detection & AML Network Analysis API",
    description="Enterprise Anti-Money Laundering Forensic Analytics using Neo4j, Graph Data Science, and Machine Learning on the IBM HI-Small AML Dataset.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/overview")
def get_overview():
    total_tx = len(aml_data) if aml_data is not None else 0
    laundering_tx = int(aml_data["Is Laundering"].sum()) if aml_data is not None else 0
    unique_accs = graph_service.G.number_of_nodes() if graph_service.G else 0

    unique_banks = 0
    if aml_data is not None:
        src_banks = set(aml_data["From Bank"].dropna().unique()) if "From Bank" in aml_data.columns else set()
        dst_banks = set(aml_data["To Bank"].dropna().unique()) if "To Bank" in aml_data.columns else set()
        all_banks = src_banks | dst_banks
        unique_banks = len(all_banks)

    # Recheck Neo4j connectivity
    graph_service.reconnect()

    return {
        "status": "online",
        "system_name": "Anti-Money Laundering (AML) Network Forensics",
        "dataset": "IBM HI-Small Transactions for AML",
        "researchers": "Aakash R (23MIA1011), Gowthamkrishna R (23MIA1087)",
        "neo4j_live_connected": graph_service.is_neo4j_connected,
        "gds_available": graph_service.gds_available,
        "engine_mode": "Neo4j 5 Bolt + GDS" if graph_service.is_neo4j_connected else "In-memory graph engine",
        "total_accounts": unique_accs,
        "total_transactions": total_tx,
        "laundering_transactions": laundering_tx,
        "clean_transactions": total_tx - laundering_tx,
        "imbalance_ratio_pct": round((laundering_tx / max(1, total_tx)) * 100, 2),
        "detected_laundering_rings": len(graph_service.cycles),
        "unique_banks": unique_banks,
        "pr_auc_lift_pct": graph_lift_pct(evaluation_metrics()),
        "evaluation_scope": evaluation_scope()
    }


@app.get("/api/graph")
def get_graph(max_nodes: int = Query(150, ge=10, le=500)):
    return graph_service.get_graph_visualization_payload(max_nodes=max_nodes)


@app.get("/api/cycles")
def get_cycles():
    # Ground-truth labels are attached for display only; detection itself never reads them
    G = graph_service.G
    cycles = []
    for c in graph_service.cycles:
        accs = c["accounts"]
        hops = [(accs[i], accs[(i + 1) % len(accs)]) for i in range(len(accs))]
        labelled = sum(1 for u, v in hops if G is not None and G.has_edge(u, v) and G[u][v].get("is_laundering", 0))
        cycles.append({**c, "laundering_hops": labelled, "confirmed_laundering": labelled > 0})
    return {
        "count": len(cycles),
        "confirmed_count": sum(c["confirmed_laundering"] for c in cycles),
        "cycles": cycles
    }


@app.get("/api/leads")
def get_prioritized_leads(limit: int = Query(50, ge=5, le=200)):
    leads = model_engine.get_top_prioritized_leads(limit=limit)
    return {
        "total_leads": len(leads),
        "leads": leads
    }


@app.get("/api/leads/{index}/explain")
def explain_lead(index: int):
    return model_engine.explain_account_lead(index)


@app.get("/api/metrics")
def get_model_metrics():
    return {
        "metrics": evaluation_metrics(),
        "evaluation_scope": evaluation_scope()
    }


@app.get("/api/ablation")
def get_ablation_study():
    return {
        "stages": evaluation_metrics().get("ablation_study", []),
        "summary": "Incremental PR-AUC change as each graph feature group is added (Random Forest per stage).",
        "evaluation_scope": evaluation_scope()
    }


@app.get("/api/shap")
def get_shap_explainability():
    return {
        "shap": evaluation_metrics().get("shap_summary", {}),
        "evaluation_scope": evaluation_scope()
    }


@app.get("/api/account/{account_id}")
def get_account_forensic_profile(account_id: str):
    profile = graph_service.get_account_profile(account_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found.")
    return profile


@app.get("/api/sar/{account_id}")
def get_suspicious_activity_report(account_id: str):
    profile = graph_service.get_account_profile(account_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found.")

    leads = model_engine.get_top_prioritized_leads(limit=100)
    # Account risk = highest model risk score among the transactions it sends or receives
    scored = model_engine.fused_df
    risk_score = 0.0
    if scored is not None:
        involved = scored[(scored["account_from"] == account_id) | (scored["account_to"] == account_id)]
        if not involved.empty:
            risk_score = float(involved["risk_score"].max())

    sar = generate_sar_report(
        account_id=account_id,
        account_profile=profile,
        leads=leads,
        cycles=graph_service.cycles,
        risk_score=risk_score
    )
    return sar


class CypherQueryRequest(BaseModel):
    query: str


@app.post("/api/cypher")
def run_cypher_query(req: CypherQueryRequest):
    return graph_service.execute_cypher(req.query)


@app.post("/api/neo4j/sync")
def sync_data_to_neo4j():
    global aml_data
    if aml_data is None:
        raise HTTPException(status_code=400, detail="AML data not loaded")

    graph_service.reconnect()
    if not graph_service.is_neo4j_connected:
        return {
            "status": "warning",
            "message": f"Neo4j Bolt server is currently not reachable at {graph_service.neo4j_uri}. Make sure Neo4j is started (docker compose up -d neo4j). Running with in-memory graph engine.",
            "neo4j_connected": False,
            "synced_rows": 0
        }
    return graph_service.sync_dataframe_to_neo4j(aml_data)


class ReloadRequest(BaseModel):
    sample_size: Optional[int] = 30000
    force_full: Optional[bool] = False


@app.post("/api/reload-dataset")
def reload_dataset(req: Optional[ReloadRequest] = None):
    """Reload the IBM dataset, recalculate graph features, and re-train ML models."""
    global aml_data
    sample_size = req.sample_size if req else 30000
    force_full = req.force_full if req else False

    aml_data = load_ibm_aml_dataset(sample_size=sample_size, force_full=force_full)
    graph_service.load_dataset(aml_data, sync_to_neo4j=False)
    model_engine.train_and_evaluate(aml_data, graph_service.account_features)

    return {
        "status": "success",
        "message": f"Reloaded IBM AML dataset with {len(aml_data):,} transactions across {graph_service.G.number_of_nodes():,} accounts.",
        "total_transactions": len(aml_data),
        "total_accounts": graph_service.G.number_of_nodes(),
        "detected_rings": len(graph_service.cycles),
        "pr_auc_lift_pct": graph_lift_pct(evaluation_metrics())
    }


# Mount static frontend
if os.path.exists(DASHBOARD_DIR):
    app.mount("/", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")
