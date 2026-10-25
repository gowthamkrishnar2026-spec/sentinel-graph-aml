"""
Graph Algorithms & Topological Feature Extraction for AML Analysis
Implements:
- Multi-hop Cycle Detection (Layering rings of lengths 3, 4, 5)
- PageRank Centrality (Structural influence)
- Betweenness Centrality (Intermediary / Money mule bridge detection)
- Louvain Community Detection (Laundering ring clusters)
- Degree Flow Asymmetry & Transaction Velocity

Optimized for real IBM HI-Small dataset:
- Vectorized edge aggregation and bulk insertion
- Single-pass degree and flow computation
- High-speed sampled betweenness centrality
- Vectorized transaction velocity profiling
"""

import collections
from typing import Dict, List, Tuple, Any, Optional
import networkx as nx
import numpy as np
import pandas as pd


def build_aml_digraph(df: pd.DataFrame) -> nx.DiGraph:
    """
    Constructs a directed weighted transaction graph from the AML DataFrame.
    Uses vectorized groupby aggregation and bulk edge addition.
    """
    G = nx.DiGraph()

    df = df.copy()
    df["Account"] = df["Account"].astype(str)
    df["Account.1"] = df["Account.1"].astype(str)
    df["Amount Paid"] = pd.to_numeric(df["Amount Paid"], errors="coerce").fillna(0.0)
    df["Is Laundering"] = pd.to_numeric(df["Is Laundering"], errors="coerce").fillna(0).astype(int)

    # Group by (source, destination)
    edge_agg = df.groupby(["Account", "Account.1"]).agg(
        total_weight=("Amount Paid", "sum"),
        tx_count=("Amount Paid", "count"),
        max_laundering=("Is Laundering", "max")
    ).reset_index()

    # Bulk add edges
    edges = [
        (
            str(row[1]),
            str(row[2]),
            {
                "weight": float(row[3]),
                "count": int(row[4]),
                "is_laundering": int(row[5])
            }
        )
        for row in edge_agg.itertuples()
    ]
    G.add_edges_from(edges)

    # Node bank attributes from first occurrence
    bank_col_src = "From Bank" if "From Bank" in df.columns else "bank"
    bank_col_dst = "To Bank" if "To Bank" in df.columns else "bank"

    if bank_col_src in df.columns:
        bank_map_src = df.drop_duplicates("Account").set_index("Account")[bank_col_src].to_dict()
    else:
        bank_map_src = {}

    if bank_col_dst in df.columns:
        bank_map_dst = df.drop_duplicates("Account.1").set_index("Account.1")[bank_col_dst].to_dict()
    else:
        bank_map_dst = {}

    for node in G.nodes():
        bank = bank_map_src.get(node) or bank_map_dst.get(node, "BNK_001")
        G.nodes[node]["bank"] = str(bank)

    return G


def detect_laundering_cycles(
    G: nx.DiGraph,
    min_len: int = 3,
    max_len: int = 5,
    max_results: int = 150,
    expansion_budget: int = 2000,
    random_state: int = 42
) -> List[Dict[str, Any]]:
    """
    Finds directed cycles of length between min_len and max_len using bounded DFS.
    Matches Cypher pattern: (a:Account)-[:TRANSFER*3..5]->(a).

    Label-blind: seeds are every node with both inflow and outflow, visited in a
    seeded random order. Edge laundering labels are never read, so cycle flags
    are safe to use as model features. Each start node gets a fixed DFS
    expansion budget so high-degree hubs cannot stall the search.
    """
    detected = []
    seen_canonical = set()

    rng = np.random.default_rng(random_state)
    nodes = [n for n in G.nodes() if G.in_degree(n) >= 1 and G.out_degree(n) >= 1]
    nodes = [nodes[i] for i in rng.permutation(len(nodes))]
    node_to_idx = {n: i for i, n in enumerate(sorted(G.nodes()))}

    def dfs(start_node, current_node, path, depth, budget):
        if depth > max_len or budget[0] <= 0:
            return
        start_idx = node_to_idx[start_node]

        for neighbor in G.successors(current_node):
            if len(detected) >= max_results or budget[0] <= 0:
                return
            budget[0] -= 1

            if neighbor == start_node and depth >= min_len:
                # Cycle found
                cycle_nodes = list(path)
                min_node = min(cycle_nodes)
                min_idx = cycle_nodes.index(min_node)
                canonical = tuple(cycle_nodes[min_idx:] + cycle_nodes[:min_idx])

                if canonical not in seen_canonical:
                    seen_canonical.add(canonical)
                    total_vol = 0.0
                    for i in range(len(cycle_nodes)):
                        u_node = cycle_nodes[i]
                        v_node = cycle_nodes[(i + 1) % len(cycle_nodes)]
                        total_vol += G[u_node][v_node].get("weight", 0.0)

                    detected.append({
                        "cycle_id": f"RING_{len(detected) + 1:04d}",
                        "accounts": cycle_nodes,
                        "length": len(cycle_nodes),
                        "total_volume": round(total_vol, 2),
                        "risk_level": "CRITICAL" if len(cycle_nodes) <= 4 else "HIGH"
                    })
            elif depth < max_len and neighbor not in path and node_to_idx[neighbor] > start_idx:
                path.append(neighbor)
                dfs(start_node, neighbor, path, depth + 1, budget)
                path.pop()

    for n in nodes:
        if len(detected) >= max_results:
            break
        dfs(n, n, [n], 1, [expansion_budget])

    detected.sort(key=lambda x: x["total_volume"], reverse=True)
    return detected


def extract_graph_topological_features(
    df: pd.DataFrame,
    G: Optional[nx.DiGraph] = None,
    max_cycles: int = 20000,
    max_reported_cycles: int = 150
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], nx.DiGraph]:
    """
    Executes full Graph Data Science suite:
    - PageRank
    - Sampled Betweenness Centrality
    - Louvain Community Detection
    - Flow and Degree Asymmetry
    - Multi-hop Cycle Participation
    - Vectorized Transaction Velocity Profiling
    """
    if G is None:
        G = build_aml_digraph(df)

    nodes = list(G.nodes())
    if not nodes:
        return pd.DataFrame(), [], G

    n_nodes = len(nodes)

    # 1. PageRank
    try:
        pagerank_scores = nx.pagerank(G, weight="weight", alpha=0.85, max_iter=60)
    except Exception:
        pagerank_scores = {n: 1.0 / n_nodes for n in nodes}

    # 2. Betweenness Centrality (sampled for performance on large graphs)
    try:
        if n_nodes > 3000:
            k = min(40, n_nodes)
            betweenness_scores = nx.betweenness_centrality(G, normalized=True, k=k, seed=42)
        else:
            betweenness_scores = nx.betweenness_centrality(G, normalized=True)
    except Exception:
        betweenness_scores = {n: 0.0 for n in nodes}

    # 3. Louvain Community Detection
    G_undirected = nx.Graph()
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1.0)
        if G_undirected.has_edge(u, v):
            G_undirected[u][v]["weight"] += w
        else:
            G_undirected.add_edge(u, v, weight=w)

    for n in G.nodes():
        if not G_undirected.has_node(n):
            G_undirected.add_node(n)

    try:
        communities = nx.community.louvain_communities(G_undirected, weight="weight", seed=42)
        community_map = {}
        for comm_id, comm_nodes in enumerate(communities):
            for node in comm_nodes:
                community_map[node] = comm_id
    except Exception:
        community_map = {n: 0 for n in nodes}

    # 4. Multi-hop Cycle Detection (membership uses the large set, API reports the top rings)
    cycles = detect_laundering_cycles(G, min_len=3, max_len=5, max_results=max_cycles)

    cycle_membership = {}
    for c in cycles:
        for acc in c["accounts"]:
            if acc not in cycle_membership:
                cycle_membership[acc] = {"count": 0, "min_length": c["length"], "rings": []}
            cycle_membership[acc]["count"] += 1
            cycle_membership[acc]["min_length"] = min(cycle_membership[acc]["min_length"], c["length"])
            cycle_membership[acc]["rings"].append(c["cycle_id"])

    # 5. Single-pass volume aggregation across edges
    in_vol_map = collections.defaultdict(float)
    out_vol_map = collections.defaultdict(float)
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 0.0)
        out_vol_map[u] += w
        in_vol_map[v] += w

    # 6. Vectorized Velocity calculation
    df_tx = df.copy()
    df_tx["ts"] = pd.to_datetime(df_tx["Timestamp"], errors="coerce")

    tx_from = df_tx[["Account", "ts"]].rename(columns={"Account": "acc"})
    tx_to = df_tx[["Account.1", "ts"]].rename(columns={"Account.1": "acc"})
    all_tx = pd.concat([tx_from, tx_to], ignore_index=True).dropna(subset=["acc", "ts"]).sort_values(["acc", "ts"])

    all_tx["prev_acc"] = all_tx["acc"].shift(1)
    all_tx["prev_ts"] = all_tx["ts"].shift(1)
    same_acc = all_tx["acc"] == all_tx["prev_acc"]
    all_tx["delta_hrs"] = np.where(same_acc, (all_tx["ts"] - all_tx["prev_ts"]).dt.total_seconds() / 3600.0, np.nan)

    vel_agg = all_tx.groupby("acc").agg(
        tx_count=("ts", "count"),
        min_delta_hrs=("delta_hrs", "min"),
        avg_delta_hrs=("delta_hrs", "mean"),
        min_ts=("ts", "min"),
        max_ts=("ts", "max")
    ).reset_index()

    time_span_days = ((vel_agg["max_ts"] - vel_agg["min_ts"]).dt.total_seconds() / 86400.0).clip(lower=1.0)
    vel_agg["tx_velocity_per_day"] = (vel_agg["tx_count"] / time_span_days).round(2)
    vel_agg["min_delta_hrs"] = vel_agg["min_delta_hrs"].fillna(48.0).round(2)
    vel_agg["avg_delta_hrs"] = vel_agg["avg_delta_hrs"].fillna(48.0).round(2)
    vel_map = vel_agg.set_index("acc").to_dict(orient="index")

    # 7. Build feature records
    account_stats = []
    for acc in nodes:
        in_vol = in_vol_map[acc]
        out_vol = out_vol_map[acc]
        in_deg = G.in_degree(acc)
        out_deg = G.out_degree(acc)

        vol_total = in_vol + out_vol
        deg_total = in_deg + out_deg
        flow_asymmetry = (out_vol - in_vol) / (vol_total + 1e-6)
        degree_ratio = (out_deg - in_deg) / (deg_total + 1e-6)

        c_info = cycle_membership.get(acc, {"count": 0, "min_length": 0, "rings": []})
        v_info = vel_map.get(acc, {"min_delta_hrs": 48.0, "avg_delta_hrs": 48.0, "tx_velocity_per_day": 0.5})

        account_stats.append({
            "account_id": acc,
            "pagerank": float(pagerank_scores.get(acc, 0.0)),
            "betweenness": float(betweenness_scores.get(acc, 0.0)),
            "community_id": int(community_map.get(acc, 0)),
            "in_degree": in_deg,
            "out_degree": out_deg,
            "degree_ratio": round(degree_ratio, 4),
            "in_volume": round(in_vol, 2),
            "out_volume": round(out_vol, 2),
            "total_volume": round(vol_total, 2),
            "flow_asymmetry": round(flow_asymmetry, 4),
            "in_cycle": 1 if c_info["count"] > 0 else 0,
            "cycle_count": c_info["count"],
            "min_cycle_length": c_info["min_length"],
            "participating_rings": ",".join(c_info["rings"]),
            "min_time_delta_hrs": float(v_info["min_delta_hrs"]),
            "avg_time_delta_hrs": float(v_info["avg_delta_hrs"]),
            "tx_velocity_per_day": float(v_info["tx_velocity_per_day"])
        })

    features_df = pd.DataFrame(account_stats)
    return features_df, cycles[:max_reported_cycles], G
