"""
Unit tests for Suspicious Activity Report (SAR) Automated Forensic Generator
"""

import pytest
from ml_pipeline.sar_generator import generate_sar_report

def test_generate_sar_report():
    account_id = "ACC_0001"
    account_profile = {
        "account_id": account_id,
        "bank": "BNK_001",
        "pagerank": 0.015,
        "betweenness": 0.065,
        "community_id": 2,
        "in_cycle": 1,
        "flow_asymmetry": 0.02,
        "in_degree": 4,
        "out_degree": 4,
        "inflow_transactions": [
            {"Timestamp": "2026-09-01 10:00:00", "Account": "ACC_0002", "Account.1": account_id, "Amount Paid": 12000.0, "Payment Format": "Wire", "Is Laundering": 1}
        ],
        "outflow_transactions": [
            {"Timestamp": "2026-09-01 10:30:00", "Account": account_id, "Account.1": "ACC_0003", "Amount Paid": 11800.0, "Payment Format": "Wire", "Is Laundering": 1}
        ]
    }
    cycles = [
        {"cycle_id": "RING_001", "accounts": ["ACC_0001", "ACC_0002", "ACC_0003"], "length": 3, "total_volume": 35800.0}
    ]

    sar = generate_sar_report(
        account_id=account_id,
        account_profile=account_profile,
        leads=[],
        cycles=cycles,
        risk_score=0.94
    )

    assert sar["primary_suspect_account"] == account_id
    assert sar["threat_level"] == "CRITICAL RISK"
    assert sar["ml_risk_score"] == 0.94
    assert len(sar["suspicion_indicators"]) >= 2
    assert "CIRC-01" in [ind["code"] for ind in sar["suspicion_indicators"]]
    assert len(sar["laundering_rings_detected"]) == 1
    assert "forensic_narrative" in sar
    assert len(sar["forensic_narrative"]) > 50
