"""
Graph Service with Dual-Engine Architecture:
- Direct Neo4j 5 Bolt Driver + GDS Plugin (Live enterprise graph database)
- Embedded High-Speed In-Memory GDS Engine (instant local / unit test operation)
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import networkx as nx

from neo4j_engine.algorithms import (
    build_aml_digraph,
    detect_laundering_cycles,
    extract_graph_topological_features
)

logger = logging.getLogger("AML_GraphService")


class AMLGraphService:
    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None
    ):
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "password123")
        self.driver = None
        self.is_neo4j_connected = False
        self.gds_available = False

        # In-memory graph state
        self.G: Optional[nx.DiGraph] = None
        self.df: Optional[pd.DataFrame] = None
        self.account_features: Optional[pd.DataFrame] = None
        self.cycles: List[Dict[str, Any]] = []

        self._init_neo4j_if_available(retries=1, delay=0.5)

    def _init_neo4j_if_available(self, retries: int = 1, delay: float = 0.5):
        """Attempts connection to live Neo4j instance."""
        for attempt in range(retries):
            try:
                import neo4j
                self.driver = neo4j.GraphDatabase.driver(
                    self.neo4j_uri,
                    auth=(self.neo4j_user, self.neo4j_password),
                    connection_timeout=3
                )
                with self.driver.session() as session:
                    res = session.run("RETURN 1 AS ping").single()
                    if res and res["ping"] == 1:
                        self.is_neo4j_connected = True
                        logger.info(f"Connected to live Neo4j at {self.neo4j_uri}")

                        try:
                            gds_res = session.run("RETURN gds.version() AS gds_ver").single()
                            if gds_res:
                                self.gds_available = True
                                logger.info(f"Neo4j GDS Plugin Active: {gds_res['gds_ver']}")
                        except Exception:
                            self.gds_available = False
                        break
            except Exception as e:
                self.is_neo4j_connected = False
                if attempt < retries - 1:
                    time.sleep(delay)

    def reconnect(self) -> bool:
        """Forces an attempt to reconnect to Neo4j."""
        self._init_neo4j_if_available(retries=1, delay=0.2)
        return self.is_neo4j_connected

    def load_dataset(self, df: pd.DataFrame, sync_to_neo4j: bool = True) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Loads the dataset into the in-memory graph, computes all topological metrics,
        and optionally synchronizes into live Neo4j database.
        """
        self.df = df
        self.account_features, self.cycles, self.G = extract_graph_topological_features(df)

        if sync_to_neo4j and self.is_neo4j_connected:
            try:
                self.sync_dataframe_to_neo4j(df)
            except Exception as e:
                logger.warning(f"Neo4j sync deferred: {e}")

        return self.account_features, self.cycles

    def sync_dataframe_to_neo4j(self, df: pd.DataFrame, batch_size: int = 2000):
        """
        Synchronizes transaction records and account nodes into live Neo4j.
        """
        if not self.is_neo4j_connected or not self.driver:
            self.reconnect()

        if not self.is_neo4j_connected or not self.driver:
            return {
                "status": "warning",
                "synced_rows": 0,
                "message": f"Neo4j is not connected at {self.neo4j_uri}. Start Neo4j via: docker compose up -d neo4j"
            }

        try:
            with self.driver.session() as session:
                # 1. Setup Constraints & Indexes
                session.run("CREATE CONSTRAINT account_id_unique IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE")
                session.run("CREATE INDEX account_bank_idx IF NOT EXISTS FOR (a:Account) ON (a.bank)")
                session.run("CREATE INDEX transfer_timestamp_idx IF NOT EXISTS FOR ()-[r:TRANSFER]-() ON (r.timestamp)")

                # 2. Prepare records
                records = []
                for row in df.itertuples():
                    row_dict = row._asdict() if hasattr(row, "_asdict") else {}
                    src = str(getattr(row, "Account", row_dict.get("Account", "")))
                    dst = str(row_dict.get("Account.1", getattr(row, "_4", getattr(row, "Account_1", ""))))
                    if not dst:
                        dst = src

                    records.append({
                        "src": src,
                        "src_bank": str(getattr(row, "From_Bank", row_dict.get("From Bank", "BNK_UNKNOWN"))),
                        "dst": dst,
                        "dst_bank": str(getattr(row, "To_Bank", row_dict.get("To Bank", "BNK_UNKNOWN"))),
                        "amount": float(getattr(row, "Amount_Paid", row_dict.get("Amount Paid", row_dict.get("Amount Received", 0.0)))),
                        "currency": str(getattr(row, "Payment_Currency", row_dict.get("Payment Currency", "USD"))),
                        "format": str(getattr(row, "Payment_Format", row_dict.get("Payment Format", "Wire"))),
                        "timestamp": str(getattr(row, "Timestamp", row_dict.get("Timestamp", ""))),
                        "is_laundering": int(getattr(row, "Is_Laundering", row_dict.get("Is Laundering", 0))),
                        "typology": str(getattr(row, "Typology", row_dict.get("Typology", "NORMAL")))
                    })

                query = """
                UNWIND $batch AS row
                MERGE (s:Account {id: row.src})
                  ON CREATE SET s.bank = row.src_bank
                MERGE (d:Account {id: row.dst})
                  ON CREATE SET d.bank = row.dst_bank
                CREATE (s)-[r:TRANSFER {
                    amount: row.amount,
                    currency: row.currency,
                    payment_format: row.format,
                    timestamp: row.timestamp,
                    is_laundering: row.is_laundering,
                    typology: row.typology
                }]->(d)
                """

                for i in range(0, len(records), batch_size):
                    batch = records[i:i + batch_size]
                    session.run(query, batch=batch)

            logger.info(f"Successfully synchronized {len(df)} transactions to Neo4j.")
            return {
                "status": "success",
                "synced_rows": len(df),
                "message": f"Successfully synchronized {len(df):,} transactions and {self.G.number_of_nodes():,} accounts into live Neo4j."
            }
        except Exception as e:
            return {
                "status": "error",
                "synced_rows": 0,
                "message": f"Failed syncing to Neo4j: {str(e)}"
            }

    def get_graph_visualization_payload(self, max_nodes: int = 150) -> Dict[str, Any]:
        """
        Formats graph nodes and links for frontend interactive force visualization.
        Prioritizes cycle-participating nodes and high-centrality nodes.
        """
        if self.G is None or self.account_features is None:
            return {"nodes": [], "links": [], "communities": []}

        feat_map = self.account_features.set_index("account_id").to_dict(orient="index")

        # Members of the reported rings first, then their direct counterparties, then hubs,
        # so the drawn subgraph stays connected around the rings an investigator cares about
        degree = lambda n: self.G.in_degree(n) + self.G.out_degree(n)
        ring_nodes = list(dict.fromkeys(a for c in self.cycles for a in c["accounts"] if self.G.has_node(a)))
        ring_set = set(ring_nodes)
        neighbours = {m for n in ring_nodes for m in nx.all_neighbors(self.G, n)} - ring_set
        hubs = [n for n in sorted(self.G.nodes(), key=degree, reverse=True)[:max_nodes] if n not in ring_set]

        selected_nodes = list(dict.fromkeys(ring_nodes + sorted(neighbours, key=degree, reverse=True) + hubs))[:max_nodes]
        node_ids = set(selected_nodes)

        nodes = []
        for n in selected_nodes:
            attrs = feat_map.get(n, {})
            nodes.append({
                "id": n,
                "label": n,
                "bank": self.G.nodes[n].get("bank", "BNK_001") if self.G.has_node(n) else "BNK_001",
                "pagerank": attrs.get("pagerank", 0.0),
                "betweenness": attrs.get("betweenness", 0.0),
                "community": attrs.get("community_id", 0),
                "inCycle": n in ring_set,
                "totalVolume": attrs.get("total_volume", 0.0),
                "inDegree": attrs.get("in_degree", 0),
                "outDegree": attrs.get("out_degree", 0),
                "rings": attrs.get("participating_rings", "")
            })

        links = []
        for u, v, data in self.G.edges(data=True):
            if u in node_ids and v in node_ids:
                links.append({
                    "source": u,
                    "target": v,
                    "amount": round(data.get("weight", 0.0), 2),
                    "count": data.get("count", 1),
                    "isLaundering": data.get("is_laundering", 0)
                })

        communities = sorted(list(set(n["community"] for n in nodes)))

        return {
            "nodes": nodes,
            "links": links,
            "communities": communities,
            "totalAccounts": self.G.number_of_nodes(),
            "totalTransfers": self.G.number_of_edges(),
            "cycleCount": len(self.cycles),
            "neo4jConnected": self.is_neo4j_connected,
            "gdsAvailable": self.gds_available
        }

    def get_detected_cycles(self) -> List[Dict[str, Any]]:
        return self.cycles

    def get_account_profile(self, account_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves deep forensic dossier for a specific account."""
        if self.account_features is None or self.df is None:
            return None

        row = self.account_features[self.account_features["account_id"] == account_id]
        if row.empty:
            return None
        profile = row.iloc[0].to_dict()

        in_txs = self.df[self.df["Account.1"] == account_id].tail(20).to_dict(orient="records")
        out_txs = self.df[self.df["Account"] == account_id].tail(20).to_dict(orient="records")

        in_neighbors = list(self.G.predecessors(account_id)) if self.G and self.G.has_node(account_id) else []
        out_neighbors = list(self.G.successors(account_id)) if self.G and self.G.has_node(account_id) else []

        profile["bank"] = self.G.nodes[account_id].get("bank", "Unknown") if self.G and self.G.has_node(account_id) else "Unknown"
        profile["inflow_transactions"] = in_txs
        profile["outflow_transactions"] = out_txs
        profile["counterparties"] = {
            "senders": in_neighbors,
            "receivers": out_neighbors
        }
        return profile

    def execute_cypher(self, query: str) -> Dict[str, Any]:
        """
        Executes Cypher on live Neo4j instance if connected. Otherwise the query text is matched to one of
        the dashboard templates (cycles, structuring, Louvain, PageRank, betweenness) and the equivalent
        result is computed in memory; arbitrary Cypher is not parsed in that mode.
        """
        if not self.is_neo4j_connected:
            self.reconnect()

        if self.is_neo4j_connected and self.driver:
            try:
                with self.driver.session() as session:
                    res = session.run(query)
                    records = [r.data() for r in res]
                    return {
                        "status": "success",
                        "mode": "Neo4j Live Bolt Server",
                        "records": records,
                        "count": len(records)
                    }
            except Exception as e:
                return {
                    "status": "error",
                    "mode": "Neo4j Live Bolt Server",
                    "error": str(e)
                }
        else:
            q_lower = query.lower()
            if "cycle" in q_lower or "*3..5" in q_lower or "ring" in q_lower:
                return {
                    "status": "success",
                    "mode": "In-memory graph engine (Neo4j not connected)",
                    "records": self.cycles[:20],
                    "count": len(self.cycles[:20]),
                    "message": "Executed via Embedded Graph Data Science Engine (Neo4j not connected)"
                }
            elif "mule" in q_lower or "10000" in q_lower:
                # Structuring fan-in: collectors receiving just-below-threshold transfers from 3+ senders
                records = []
                if self.df is not None:
                    amounts = pd.to_numeric(self.df["Amount Paid"], errors="coerce")
                    near = self.df[(amounts >= 7500.0) & (amounts < 10000.0)].assign(_amt=amounts)
                    grouped = near.groupby("Account.1").agg(
                        mule_count=("Account", "nunique"), total_collected=("_amt", "sum")
                    ).reset_index()
                    grouped = grouped[grouped["mule_count"] >= 3].sort_values("total_collected", ascending=False).head(15)
                    records = [
                        {"aggregator": r["Account.1"], "mule_count": int(r["mule_count"]),
                         "total_collected": round(float(r["total_collected"]), 2)}
                        for _, r in grouped.iterrows()
                    ]
                return {
                    "status": "success",
                    "mode": "In-memory graph engine (Neo4j not connected)",
                    "records": records,
                    "count": len(records)
                }
            elif "louvain" in q_lower or "community" in q_lower:
                counts = (
                    self.account_features["community_id"].value_counts().head(15)
                    if self.account_features is not None else pd.Series(dtype=int)
                )
                records = [{"communityId": int(k), "accounts_in_community": int(v)} for k, v in counts.items()]
                return {
                    "status": "success",
                    "mode": "In-memory graph engine (Neo4j not connected)",
                    "records": records,
                    "count": len(records)
                }
            elif "pagerank" in q_lower:
                top_pr = (
                    self.account_features.sort_values("pagerank", ascending=False)
                    .head(15)
                    .to_dict(orient="records")
                    if self.account_features is not None else []
                )
                return {
                    "status": "success",
                    "mode": "In-memory graph engine (Neo4j not connected)",
                    "records": top_pr,
                    "count": len(top_pr)
                }
            elif "betweenness" in q_lower:
                top_bw = (
                    self.account_features.sort_values("betweenness", ascending=False)
                    .head(15)
                    .to_dict(orient="records")
                    if self.account_features is not None else []
                )
                return {
                    "status": "success",
                    "mode": "In-memory graph engine (Neo4j not connected)",
                    "records": top_bw,
                    "count": len(top_bw)
                }
            else:
                top_accs = (
                    self.account_features.head(15).to_dict(orient="records")
                    if self.account_features is not None else []
                )
                return {
                    "status": "success",
                    "mode": "In-memory graph engine (Neo4j not connected)",
                    "records": top_accs,
                    "count": len(top_accs),
                    "note": "Running on the in-memory graph engine."
                }
