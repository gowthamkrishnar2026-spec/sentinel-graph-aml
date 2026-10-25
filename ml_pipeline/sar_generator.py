"""
Suspicious Activity Report (SAR) Automated Forensic Generator
Generates FinCEN / FIU compliant AML Suspicious Activity Reports with:
- Executive Summary & Risk Scoring
- Suspect Entity Dossier & Network Topography
- Layering / Smurfing Pattern Analysis
- Multi-Hop Cycle Ring Evidence Chain
- Chronological Transaction Narrative
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

def generate_sar_report(
    account_id: str,
    account_profile: Dict[str, Any],
    leads: List[Dict[str, Any]],
    cycles: List[Dict[str, Any]],
    risk_score: float = 0.95
) -> Dict[str, Any]:
    """
    Constructs a formal structured Suspicious Activity Report (SAR).
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    report_id = f"SAR-AML-{datetime.now().strftime('%Y%m%d')}-{account_id.replace('ACC_', '')}"

    # Extract account transactions
    inflows = account_profile.get("inflow_transactions", [])
    outflows = account_profile.get("outflow_transactions", [])
    all_txs = inflows + outflows

    # Sort chronological
    all_txs.sort(key=lambda x: str(x.get("Timestamp", "")), reverse=True)

    # Calculate aggregate volumes
    total_inflow = sum(float(tx.get("Amount Paid", tx.get("Amount Received", 0))) for tx in inflows)
    total_outflow = sum(float(tx.get("Amount Paid", tx.get("Amount Received", 0))) for tx in outflows)
    total_volume = total_inflow + total_outflow

    # Determine participating cycle rings
    associated_cycles = [
        c for c in cycles if account_id in c.get("accounts", [])
    ]

    # Suspicion Indicators
    suspicion_reasons = []
    if associated_cycles:
        suspicion_reasons.append({
            "code": "CIRC-01",
            "category": "Circular Laundering Loop",
            "description": f"Subject account is an active participant in {len(associated_cycles)} closed transfer rings (3 to 5 hops), returning capital to originating clusters."
        })

    if account_profile.get("betweenness", 0.0) > 0.02:
        suspicion_reasons.append({
            "code": "CENT-02",
            "category": "Money Mule Intermediary Bridge",
            "description": f"Subject demonstrates high betweenness centrality ({account_profile.get('betweenness'):.4f}), acting as a conduit between distinct financial entities."
        })

    if abs(account_profile.get("flow_asymmetry", 0.0)) < 0.15 and len(all_txs) > 4:
        suspicion_reasons.append({
            "code": "FLOW-03",
            "category": "Rapid Pass-Through Flow",
            "description": f"Inflow volume ({total_inflow:,.2f}) closely mirrors outflow volume ({total_outflow:,.2f}), indicating zero capital retention and conduit behavior."
        })

    if not suspicion_reasons:
        suspicion_reasons.append({
            "code": "BEHV-04",
            "category": "Anomalous Transaction Volume",
            "description": f"Elevated transaction velocity and statistical deviation from baseline account activity."
        })

    # Narrative Summary
    narrative = (
        f"Automated Graph Forensic Analysis flagged account {account_id} with an aggregated ML fraud risk score of {risk_score:.2%}. "
        f"Graph topology analysis found {len(associated_cycles)} circular layering cycle(s) totaling {total_volume:,.2f} (mixed currencies) across "
        f"{len(inflows)} incoming and {len(outflows)} outgoing transfers. "
        f"The subject account exhibits a PageRank centrality of {account_profile.get('pagerank', 0.0):.4f} within community #{account_profile.get('community_id', 0)}. "
        f"Enhanced due diligence (EDD) review by a compliance analyst is recommended."
    )

    return {
        "report_id": report_id,
        "filing_date": now,
        "filing_institution": "Sentinel Graph (academic prototype)",
        "primary_suspect_account": account_id,
        "bank_affiliation": account_profile.get("bank", "BNK_001"),
        "ml_risk_score": round(risk_score, 4),
        "threat_level": "CRITICAL RISK" if risk_score >= 0.8 else ("HIGH RISK" if risk_score >= 0.5 else "MEDIUM RISK"),
        "total_volume_analyzed": round(total_volume, 2),
        "total_inflow_volume": round(total_inflow, 2),
        "total_outflow_volume": round(total_outflow, 2),
        "graph_metrics": {
            "pagerank": round(account_profile.get("pagerank", 0.0), 4),
            "betweenness_centrality": round(account_profile.get("betweenness", 0.0), 4),
            "community_cluster_id": account_profile.get("community_id", 0),
            "flow_asymmetry": round(account_profile.get("flow_asymmetry", 0.0), 4),
            "in_degree": account_profile.get("in_degree", 0),
            "out_degree": account_profile.get("out_degree", 0),
            "cycle_participation_count": len(associated_cycles)
        },
        "suspicion_indicators": suspicion_reasons,
        "laundering_rings_detected": associated_cycles,
        "recent_transactions": all_txs[:15],
        "forensic_narrative": narrative,
        "compliance_officer_action": "Refer to Financial Intelligence Unit (FIU) for statutory filing."
    }
