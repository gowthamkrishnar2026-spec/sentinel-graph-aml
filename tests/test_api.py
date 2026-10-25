"""
Unit tests for FastAPI REST Endpoints
"""

import pytest
from fastapi.testclient import TestClient
from api.app import app, init_system

@pytest.fixture(scope="module")
def client():
    init_system()
    with TestClient(app) as c:
        yield c

def test_get_overview(client):
    res = client.get("/api/overview")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert data["total_accounts"] > 0
    assert data["total_transactions"] > 0
    assert "detected_laundering_rings" in data

def test_get_graph(client):
    res = client.get("/api/graph?max_nodes=50")
    assert res.status_code == 200
    data = res.json()
    assert "nodes" in data
    assert "links" in data
    assert len(data["nodes"]) > 0

def test_get_cycles(client):
    res = client.get("/api/cycles")
    assert res.status_code == 200
    data = res.json()
    assert "cycles" in data
    assert isinstance(data["cycles"], list)

def test_get_leads(client):
    res = client.get("/api/leads?limit=10")
    assert res.status_code == 200
    data = res.json()
    assert "leads" in data
    assert len(data["leads"]) > 0

def test_get_metrics(client):
    res = client.get("/api/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "metrics" in data
    assert "tabular_baseline" in data["metrics"]
    assert "graph_random_forest" in data["metrics"]

def test_get_ablation(client):
    res = client.get("/api/ablation")
    assert res.status_code == 200
    data = res.json()
    assert "stages" in data
    assert len(data["stages"]) == 5

def test_get_shap(client):
    res = client.get("/api/shap")
    assert res.status_code == 200
    data = res.json()
    assert "shap" in data

def test_get_account_and_sar(client):
    # Get a known account from graph
    graph_res = client.get("/api/graph?max_nodes=10").json()
    acc_id = graph_res["nodes"][0]["id"]

    acc_res = client.get(f"/api/account/{acc_id}")
    assert acc_res.status_code == 200

    sar_res = client.get(f"/api/sar/{acc_id}")
    assert sar_res.status_code == 200
    sar = sar_res.json()
    assert sar["primary_suspect_account"] == acc_id
    assert "threat_level" in sar

def test_cypher_endpoint(client):
    res = client.post("/api/cypher", json={"query": "MATCH (n) RETURN count(n) LIMIT 5"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
