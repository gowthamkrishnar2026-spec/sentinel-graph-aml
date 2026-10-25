// ============================================================================
// Neo4j Anti-Money Laundering (AML) Cypher Queries & Graph Data Science (GDS)
// Project: Fraud Detection & AML Network Analysis Using Neo4j
// Authors: Aakash R (23MIA1011), Gowthamkrishna R (23MIA1087)
// ============================================================================

// 1. Schema Definition & Constraints
CREATE CONSTRAINT account_id_unique IF NOT EXISTS
FOR (a:Account) REQUIRE a.id IS UNIQUE;

CREATE INDEX account_bank_idx IF NOT EXISTS
FOR (a:Account) ON (a.bank);

CREATE INDEX transfer_timestamp_idx IF NOT EXISTS
FOR ()-[r:TRANSFER]-() ON (r.timestamp);

// 2. Data Ingestion from IBM Synthetic AML CSV format
// LOAD CSV WITH HEADERS FROM 'file:///HI-Small_Trans.csv' AS row
// MERGE (src:Account {id: row.Account})
//   ON CREATE SET src.bank = row.`From Bank`
// MERGE (dst:Account {id: row.`Account.1`})
//   ON CREATE SET dst.bank = row.`To Bank`
// CREATE (src)-[:TRANSFER {
//   amount: toFloat(row.`Amount Paid`),
//   currency: row.`Payment Currency`,
//   payment_format: row.`Payment Format`,
//   timestamp: row.Timestamp,
//   is_laundering: toInteger(row.`Is Laundering`),
//   typology: row.Typology
// }]->(dst);

// 3. Multi-Hop Circular Path / Laundering Ring Detection (Lengths 3 to 5)
MATCH path = (a:Account)-[:TRANSFER*3..5]->(a)
WITH a, path, relationships(path) AS txs
RETURN a.id AS start_account,
       [n IN nodes(path) | n.id] AS account_chain,
       length(path) AS cycle_length,
       [t IN txs | t.amount] AS amounts,
       reduce(total = 0.0, t IN txs | total + t.amount) AS total_volume
ORDER BY total_volume DESC;

// 4. Structuring / Smurfing Detection (Fan-In below $10,000 threshold)
MATCH (mule:Account)-[r:TRANSFER]->(collector:Account)
WHERE r.amount >= 7500.0 AND r.amount < 10000.0
WITH collector, count(DISTINCT mule) AS mule_count,
     sum(r.amount) AS total_collected,
     collect(mule.id) AS mule_accounts
WHERE mule_count >= 3
OPTIONAL MATCH (collector)-[out:TRANSFER]->(destination:Account)
WHERE out.amount >= 0.8 * total_collected
RETURN collector.id AS aggregator,
       mule_count,
       total_collected,
       mule_accounts,
       destination.id AS offshore_dispersal,
       out.amount AS dispersed_amount;

// 5. Neo4j Graph Data Science (GDS): Project Graph Catalog
// CALL gds.graph.project.cypher(
//   'amlNetwork',
//   'MATCH (a:Account) RETURN id(a) AS id, labels(a) AS labels',
//   'MATCH (s:Account)-[r:TRANSFER]->(t:Account) RETURN id(s) AS source, id(t) AS target, type(r) AS type, r.amount AS amount'
// );

// 6. GDS PageRank Centrality
// CALL gds.pageRank.stream('amlNetwork', {
//   dampingFactor: 0.85,
//   maxIterations: 20
// })
// YIELD nodeId, score
// RETURN gds.util.asNode(nodeId).id AS account_id, score AS pagerank
// ORDER BY pagerank DESC LIMIT 25;

// 7. GDS Betweenness Centrality (Bridge / Money Mule Intermediaries)
// CALL gds.betweenness.stream('amlNetwork')
// YIELD nodeId, score
// RETURN gds.util.asNode(nodeId).id AS account_id, score AS betweenness
// ORDER BY betweenness DESC LIMIT 25;

// 8. GDS Louvain Community Detection (Laundering Ring Clusters)
// CALL gds.louvain.stream('amlNetwork')
// YIELD nodeId, communityId
// RETURN gds.util.asNode(nodeId).id AS account_id, communityId
// ORDER BY communityId ASC;

// 9. Account Forensic 2-Hop Subgraph Neighborhood
// MATCH path = (a:Account {id: $account_id})-[r:TRANSFER*1..2]-(connected:Account)
// RETURN path LIMIT 50;
