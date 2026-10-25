# Sentinel Graph Dashboard Guide

How every part of the dashboard works, what each number means, where it comes from in the code, and why it was designed that way. Each section starts with a **high-level** explanation (what it is and why it matters) followed by the **low-level** details (exact data source, formula, thresholds and code location).

---

## 1. The big picture

### High level

The dashboard is an investigator's workbench for anti-money laundering (AML). It answers four questions:

1. **What does the money network look like?** (Network Graph, Rings)
2. **Which transactions should an analyst review first?** (Alert Queue)
3. **How good is the detector, honestly measured?** (Model Benchmark, Ablation Study, SHAP)
4. **What would I report about a suspicious account?** (Account Inspector, SAR report)

### Two sources of data, on purpose

The dashboard uses two datasets, and it is important to know which panel uses which:

| Data | Size | Used by | Why |
| --- | --- | --- | --- |
| **Demo sample** `data/HI-Small_Trans_sample.csv` | 102,907 transactions | KPI cards (except lift), Network Graph, Rings, Alert Queue, Account Inspector, SAR, Cypher Studio | Small enough to load, analyse and train on in about 30 seconds on a laptop |
| **Full-data evaluation** `data/precomputed_metrics.json` | 5,077,237 transactions | Lift KPI, Model Benchmark, Ablation Study, SHAP | Honest metrics need the full dataset at the real 0.1% laundering rate; computing that takes 2 minutes and several GB of memory, so it is done once offline |

Every evaluation tab shows a blue caption saying exactly which data it came from, for example: *"Full IBM HI-Small (5,077,237 tx), strict temporal split · test period from 2022/09/08 03:10 · 1,269,309 test transactions · 0.114% laundering"*.

### Low level: what happens when the server starts

`./run.sh` starts `uvicorn api.app:app`. On startup `init_system()` in `api/app.py` runs three steps:

1. **Load data.** `load_aml_data()` in `data/ibm_aml_loader.py` reads the 102,907-row demo sample.
2. **Build the graph.** `graph_service.load_dataset()` calls `extract_graph_topological_features()` in `neo4j_engine/algorithms.py`. This builds the account network and computes all account features and rings (Section 3).
3. **Train the live models.** `model_engine.train_and_evaluate()` in `ml_pipeline/train_models.py` trains three models on the earliest 75% of the sample and scores every transaction. These scores drive the Alert Queue and SAR.
4. **Load the saved evaluation.** If `data/precomputed_metrics.json` exists, it is loaded into `precomputed_metrics` and served by the evaluation tabs instead of the live sample metrics.

The browser then loads `dashboard/index.html`, and `dashboard/app.js` calls the REST API (Section 12) to fill each panel.

### How the demo sample is built (and why it is fair)

`python3 -m ml_pipeline.build_artifacts` creates the sample from the full data:

1. Drop every transaction after 10 September 2022 (`SIMULATION_END`). After that date the IBM simulator stops generating normal traffic and about 60% of rows are laundering, which would let any model "cheat" using the date.
2. Pick a random 1% of accounts and keep **every** transaction they send or receive. Random selection ignores labels, so the laundering rate stays realistic.
3. Run label-free ring detection on the full graph, take the 150 largest rings, and add every transaction **between** their members. Without this, random sampling would break almost every ring apart and the Rings view would be empty.

Result: 102,907 transactions, 271 labelled laundering (0.26%). The rate is higher than the real 0.089% only because complete rings were added; the rings were still found without using labels.

---

## 2. Header bar

| Element | What it shows | How it works |
| --- | --- | --- |
| **Sentinel Graph** logo and subtitle | Product name | Static HTML |
| **IBM HI-Small · 102,907 TX** badge | Number of transactions loaded in the live sample | `loadOverview()` sets it from `/api/overview` → `total_transactions` |
| **Reload IBM Data** | Re-runs the whole startup pipeline without restarting the server | `POST /api/reload-dataset` → reloads the sample, rebuilds the graph, retrains the live models (about 30 s). Useful after rebuilding the artifacts |
| **Sync to Neo4j** | Copies the sample into a running Neo4j database | `POST /api/neo4j/sync` → `sync_dataframe_to_neo4j()` writes `Account` nodes and `TRANSFER` relationships in batches of 2,000 with `MERGE` (safe to repeat). If Neo4j is not running, it shows a warning and nothing changes |
| **Export SAR Dossier** | Opens the SAR report for the sender of the highest-risk alert | Takes `state.leads[0].account_from` and calls `viewSARReport()` |
| **Status dot** | Which graph engine is active | "In-memory graph engine" (NetworkX, default) or "Neo4j 5 Bolt + GDS" when the Docker Neo4j is running |

**Why an in-memory engine at all?** Neo4j needs Docker and about 1.5 GB of memory. NetworkX runs the same algorithms inside Python, so the project works on any laptop and in the tests. Neo4j is an optional extra.

---

## 3. How the graph and account features are computed

This section explains the variables used everywhere else.

### Building the graph (`build_aml_digraph`)

- Every **account** becomes a **node**.
- All transfers from account A to account B are combined into **one directed edge A → B** with:
  - `weight` = total amount sent (sum of `Amount Paid`)
  - `count` = number of transfers
  - `is_laundering` = 1 if any of those transfers is labelled laundering. **Used only for display and validation, never by detection or models.**
- Each node stores its `bank`.

### Account features (`extract_graph_topological_features`)

| Variable | Formula / method | Meaning | Why it helps |
| --- | --- | --- | --- |
| `pagerank` | NetworkX PageRank on amount-weighted edges, damping 0.85, max 60 iterations | Share of "money importance" flowing into the account, counting how important its senders are | Laundered funds collect in a few destination accounts |
| `betweenness` | Fraction of shortest paths passing through the account. Exact on small graphs; on graphs over 3,000 nodes, estimated from 40 random source nodes | How much the account acts as a bridge | Money mules connect otherwise separate groups |
| `community_id` | Louvain community detection on the undirected, amount-weighted graph (seed 42) | Which tightly connected cluster the account is in | Rings tend to form their own clusters |
| `in_degree` / `out_degree` | Number of distinct senders / receivers | Breadth of connections | Many senders into one account suggests structuring |
| `in_volume` / `out_volume` / `total_volume` | Sum of incoming / outgoing / all edge weights | Money in and out | Scale of activity |
| `flow_asymmetry` | (out_volume − in_volume) / (total_volume + 10⁻⁶), from −1 to +1 | Near 0: money passes straight through. +1: only sends. −1: only receives | Pass-through accounts are a layering sign |
| `degree_ratio` | (out_degree − in_degree) / (total degree + 10⁻⁶) | Sender-heavy vs receiver-heavy by counterparties | Context for the inspector |
| `in_cycle` | 1 if the account is in any detected 3–5 hop ring | Ring member flag | Circular flow is a classic laundering pattern |
| `cycle_count` | Number of detected rings containing the account | How deeply involved | More rings means more suspicion |
| `min_cycle_length` | Shortest ring the account is in | Tightness of the loop | Short loops are more suspicious |
| `participating_rings` | Comma-separated ring IDs | Which rings | Navigation |
| `min_time_delta_hrs` / `avg_time_delta_hrs` | Smallest / average gap in hours between the account's consecutive transactions (48 if only one transaction) | Rhythm of activity | Rapid back-to-back transfers are suspicious |
| `tx_velocity_per_day` | Transactions ÷ active days (at least 1 day) | Activity rate | Bursts of activity |

### Ring detection (`detect_laundering_cycles`)

**High level:** a ring is money that leaves an account and comes back to it after 3, 4 or 5 transfers (A → B → C → A). Launderers use loops to make funds look like ordinary business payments.

**Low level:**

1. Candidate start accounts are those with at least one incoming and one outgoing edge, visited in a fixed random order (seed 42).
2. From each start, a depth-first search follows outgoing edges up to 5 hops. To avoid finding the same loop several times, it only moves to accounts whose sorted position is higher than the start's.
3. A loop is recorded when the search returns to the start after 3–5 hops. It is stored in a canonical form (rotated to begin at its smallest account ID) so each loop is counted once.
4. Each start has a budget of 2,000 edge expansions, so very busy accounts cannot stall the search.
5. For each ring: `total_volume` = sum of the edge weights around the loop; `risk_level` = **CRITICAL** for 3–4 hops, **HIGH** for 5 hops.
6. Up to 20,000 rings are used for the account features; the 150 with the highest volume are shown in the dashboard.

**Why label-free matters:** an earlier version started its search from accounts already labelled as laundering. That leaked the answer into the model's inputs and inflated every result. The current version never reads the label, and a unit test (`test_cycle_detection_ignores_labels`) checks that flipping all labels does not change the rings found.

---

## 4. KPI cards

| Card | Value | Source | Notes |
| --- | --- | --- | --- |
| **Monitored Accounts** | 17,877 | Number of nodes in the sample graph (`G.number_of_nodes()`) | Subtext: number of distinct banks seen in `From Bank` / `To Bank` (3,679) |
| **Total Transactions** | 102,907 | Rows in the demo sample | |
| **Labelled Laundering TX** | 271 | Rows in the sample with `Is Laundering = 1` | Subtext: 271 ÷ 102,907 = 0.26%. Ground-truth labels, shown for context only |
| **Detected Rings (Cycles)** | 150 | Number of rings reported by `detect_laundering_cycles` | Label-free |
| **Graph ML PR-AUC Lift** | +170.9% | (PR-AUC of graph gradient boosting − PR-AUC of tabular baseline) ÷ PR-AUC of tabular baseline, from the full-data evaluation: (0.1211 − 0.0447) ÷ 0.0447 | Computed by `graph_lift_pct()` in `api/app.py`. Subtext shows the data scope |

---

## 5. Network Graph tab

### High level

An interactive picture of the money network around the detected rings. Analysts can see loops (triangles, squares), hubs (stars) and bridges, then click any account to inspect it.

### Low level

**Which accounts are drawn** (`get_graph_visualization_payload`, up to 200 nodes):

1. members of the 150 reported rings (197 accounts in the current sample)
2. then their direct counterparties, busiest first
3. then the busiest accounts overall, to fill the remaining slots

Edges are drawn only between accounts that are both on screen. This order keeps the picture connected around the rings instead of showing 200 unrelated accounts.

**Layout:** D3 force simulation:

- links pull connected accounts 55 px apart
- each node repels others (strength −120)
- a gentle pull toward the centre (`forceX`/`forceY`, strength 0.06) stops disconnected pieces drifting away
- collision radius 16 px prevents overlap

The layout is computed up front (300 steps) and then zoomed to fit, so the graph appears complete and steady.

**Encoding:**

| Visual | Rule |
| --- | --- |
| Red node, radius 9 | Ring member (`inCycle`) |
| Amber node, radius 10 | Bridge: betweenness > 0.02 |
| Blue node, radius 5–13 | Other account; radius = 5 + min(8, PageRank × 200) |
| Node label | Shown for ring members, bridges and PageRank > 0.005 (others hidden to avoid clutter) |
| Thin grey line | Transfer |
| Thick red line with arrow | Transfer labelled as laundering (ground truth, for validation) |
| Black outline | Account currently open in the inspector |

**Controls:**

- **Drag** a node to move it; **scroll** to zoom; **drag the background** to pan.
- **Hover** a node to see its ID, bank, PageRank, betweenness and ring membership.
- **All Nodes:** show everything.
- **Laundering Rings:** fade all but ring members and labelled laundering edges.
- **High Centrality:** fade all but bridges (betweenness > 0.02).

### Account Inspector (right panel)

Opened by clicking a graph node, an account link in the Alert Queue, or **Inspect Origin** on a ring card. From another tab, the dashboard switches to this tab automatically. Data comes from `GET /api/account/{id}`.

| Field | Meaning |
| --- | --- |
| Account ID and bank | Identity |
| PageRank, Betweenness | Network importance and bridge score (Section 3) |
| Total Volume | Money in + out across the account's edges (mixed currencies) |
| In / Out Degree | Distinct senders / receivers |
| Flow Asymmetry | −1 to +1 (Section 3) |
| Cycle Rings | Number of rings; hover to see their IDs |
| **Generate SAR Report** | Opens the SAR for this account (Section 11) |

The badge in the panel header turns red when the account is in a ring.

---

## 6. Rings tab

### High level

A catalogue of the detected circular money paths, largest first. Each card answers: who is in the loop, how much money went around, and whether the ground-truth labels agree it is laundering.

### Low level

- **Header Cypher line** shows the equivalent Neo4j query: `MATCH path = (a:Account)-[:TRANSFER*3..5]->(a)`.
- **Big red number:** sum of `total_volume` over the 150 rings. Amounts are in each transaction's own currency, so this is a scale indicator, not a currency total.
- **Validation caption:** "41 of 150 rings contain labelled laundering transfers". It is computed in `GET /api/cycles` by checking each hop's edge label. **Labels are added only after detection, for display.**

Each card shows (the first 30 rings, by volume):

| Item | Meaning |
| --- | --- |
| `RING_0083` | Ring ID (numbered in the order it was found) |
| **3/4/5-Hop Ring** | Ring length |
| **CONFIRMED** | At least one hop carries a laundering label. Pink card |
| **UNLABELLED** | No hop is labelled. White card. This does not mean "innocent": it may be a missed or legitimate loop |
| Path | Accounts in order; ↻ marks the return to the start |
| Volume | Sum of the ring's edge amounts |
| Labelled hops | How many of the ring's hops are labelled laundering, for example 3/3 |
| **Inspect Origin** | Opens the first account of the ring in the Account Inspector |

**Why show CONFIRMED vs UNLABELLED?** It lets you show, in the demo, that a purely structural, label-free method finds real laundering (41 of 150 rings) without claiming every loop is criminal.

---

## 7. Alert Queue tab

### High level

The investigator's to-do list: the 60 riskiest transactions, ranked by the model. Ranking matters more than a yes/no decision because analysts can only review a limited number of alerts per day.

### Low level

- **Which transactions:** only those from the **latest 25% of the sample** (from 8 September 2022), which the live model never trained on (`in_test_period = 1`). This makes the queue an honest preview: currently **38 of the top 60** are labelled laundering.
- **Risk score** = average of the graph-augmented Random Forest and graph-augmented gradient-boosting probabilities: `(rf_prob + gb_prob) / 2`.

| Column | Meaning |
| --- | --- |
| **Risk** | Risk score as a percentage. Red ≥ 80%, amber ≥ 50%, blue ≥ 30%, grey below |
| **Sender / Receiver** | Account IDs; click to open the Account Inspector |
| **Bank** | Sender's and receiver's bank codes |
| **Amount** | Amount paid, with its currency |
| **Format** | Payment type: Wire, ACH, Cash, Cheque, Credit Card, Reinvestment or Bitcoin |
| **Graph Indicators** | **In Cycle**: sender or receiver is a ring member. **Bridge**: sender's betweenness > 0.02. **Same Cluster**: both accounts in the same Louvain community |
| **Label** | Ground truth from the dataset (LAUNDERING / NORMAL), shown for validation |
| **Action** | Opens the SAR for the sender |

**Search** filters by account, bank or typology as you type. The **filter** menu offers:

- **All**
- **High Risk:** risk ≥ 85%
- **Circular Rings:** the In Cycle flag is set
- **Below 10K (structuring):** amount under 10,000, a classic way to avoid reporting thresholds

**Note on identical scores:** many top alerts share the same score (for example 98.1%). Tree models give the same probability to transactions whose features fall into the same leaves, which is common for transfers inside the same ring.

---

## 8. Model Benchmark tab

### High level

Compares a transaction-only model with the graph-augmented model on the full dataset, measured fairly. The question it answers: *do network features genuinely improve detection?* Yes: PR-AUC rises from 0.045 to 0.121, and 61 of the top 100 alerts are laundering instead of 17.

### How the evaluation was done (low level)

Produced by `ml_pipeline/build_artifacts.py` → `train_and_evaluate(test_graph="train")`:

1. **Data:** all 5,077,237 transactions from 1 to 10 September 2022.
2. **Temporal split:** sorted by time; the earliest 75% for training (to 8 September 03:10), the latest 25% for testing (1,269,309 transactions, 1,449 laundering, 0.114%).
3. **Training features** come from a graph built only from training-period transactions.
4. **Test features** also come only from the training-period graph (`test_graph="train"`), the strictest setting: no future information at all. Accounts never seen in training get default values.
5. **Training data:** all 3,073 laundering transactions plus 300,000 random normal ones (303,073 rows), to keep training fast. The test set is not resampled.
6. **Models:**
   - **Tabular Random Forest:** 80 trees, depth 8, balanced class weights, 28 transaction features.
   - **Graph Random Forest:** 100 trees, depth 10, balanced class weights, all 45 features.
   - **Graph Gradient Boosting:** scikit-learn `HistGradientBoostingClassifier`, 80 iterations, all 45 features, each laundering example weighted by (number of normal ÷ number of laundering).

### Model cards

| Metric | Tabular RF | Graph GB (best model) | Meaning |
| --- | --- | --- | --- |
| ROC-AUC | 0.9208 | 0.9746 | Probability that a random laundering transaction is ranked above a random normal one. Looks high for every model when classes are this imbalanced, so it is a secondary metric |
| PR-AUC | 0.0447 | 0.1211 | Area under the precision-recall curve (average precision). **The main metric.** Random guessing scores 0.00114 |
| Precision@50 | 0.10 | 0.82 | Fraction of the top 50 alerts that are laundering |
| Precision@100 | 0.17 | 0.61 | Fraction of the top 100 alerts that are laundering |

**Why not accuracy?** With 0.114% laundering, a model that always says "normal" is 99.886% accurate and catches nothing.

### Precision-Recall Curve

- **x-axis, recall:** share of all laundering transactions caught.
- **y-axis, precision:** share of flagged transactions that really are laundering.
- **Solid blue:** graph-augmented gradient boosting. **Dashed grey:** tabular baseline.
- **Reading it:** the graph model keeps high precision at low recall: its very top alerts are mostly correct. Precision falls as it reaches further down the ranking. The tabular curve drops almost immediately.
- **How it is drawn:** for 41 recall levels (0, 0.025, …, 1.0), the plotted precision is the best precision achievable at that recall or higher (`evaluate_aml_predictions` in `ml_pipeline/evaluation.py`). This standard "interpolated" curve makes the area under the line match PR-AUC.

### Confusion matrix (score ≥ 0.5) and Precision@K

- **Matrix:** counts for the graph gradient-boosting model when every transaction with score ≥ 0.5 is flagged:

  | | Actually laundering | Actually normal |
  | --- | ---: | ---: |
  | **Flagged** | 1,379 true positives | 155,308 false positives |
  | **Not flagged** | 70 false negatives | 1,112,552 true negatives |

  It catches 95% of laundering at that threshold, but with many false alerts. The 0.5 threshold is not tuned: the class weighting pushes scores up. That is why the dashboard emphasises ranking metrics (PR-AUC, Precision@K) over a single threshold.
- **Precision @ top audit ranks:** P@10 = 80%, P@25 = 84%, P@50 = 82%, P@100 = 61%. These are the numbers that matter to an investigation team.

---

## 9. Ablation Study tab

### High level

Adds one group of network features at a time to see what each contributes. Each stage uses the same Random Forest (60 trees, depth 8, balanced weights), the same full-data training set and the same test set.

### Low level

| Stage | Adds | # features | PR-AUC | Change |
| --- | --- | ---: | ---: | ---: |
| 1 | Tabular baseline: amount, log amount, hour, 7 payment formats, 15 currencies, cross-currency, cross-bank, self-transfer | 28 | 0.0407 | Baseline |
| 2 | + PageRank (sender, receiver, ratio) | 31 | 0.0521 | +0.0114 |
| 3 | + Betweenness (sender, receiver) | 33 | 0.0581 | +0.0060 |
| 4 | + Same community, flow asymmetry, degrees, sender velocity | 41 | 0.0429 | −0.0152 |
| 5 | + Ring flags and counts (full model) | 45 | 0.0884 | **+0.0455** |

- **Lift badges** are the change in PR-AUC from the previous stage (green up, red down).
- **Chart bars:** grey for the baseline, light blue for the intermediate stages, dark blue for the full model.
- **Why stage 4 drops:** Louvain community IDs computed on the training graph do not line up with later data. Under the strict setting they add noise. This is reported as a finding, not hidden.
- **Why stage 1 here (0.0407) differs slightly from the tabular card (0.0447):** the ablation uses a smaller forest (60 trees, depth 8) than the benchmark model (80 trees).

---

## 10. SHAP tab

### High level

Explains *which* features the graph-augmented Random Forest relies on, so the model is not a black box.

### Low level

- **Left panel, Random Forest Feature Importance:** each feature's share of the total impurity (Gini) reduction across all trees, top 10. Fast, global, built into Random Forest. Features tagged **GRAPH** are network features.
- **Right panel, SHAP Interpretation:** mean absolute SHAP value per feature from `shap.TreeExplainer`, computed on 80 test transactions. A SHAP value is how much a feature pushed one prediction up or down from the average. The mean of its absolute value measures overall influence.
- **Top Risk Driver:** the feature ranked first by SHAP: `is_ach` (ACH payment format), a known characteristic of the IBM dataset.
- **Graph share sentence:** the share of the top-8 SHAP total that comes from graph features (about 19%), computed in the browser by `graphShare()`.
- If the `shap` package is not installed, the right panel falls back to Random Forest importances and says so.
- **Feature descriptions:** every feature name in both panels has a one-line plain-English explanation underneath, from `FEATURE_INFO` in `dashboard/app.js`.
- **What each feature means:** a glossary card at the bottom of the tab lists all 45 model inputs, grouped into transaction and graph features (the full definitions are in Section 14).

**Why two panels?** Impurity importance is quick but can favour features with many distinct values. SHAP is fairer and more widely accepted, but slower. Showing both is a standard cross-check.

---

## 11. SAR (Suspicious Activity Report)

### High level

A structured report a compliance analyst could file with a Financial Intelligence Unit, generated in one click for any account. It turns model output into an auditable explanation.

### Low level

`GET /api/sar/{account}` → `generate_sar_report()` in `ml_pipeline/sar_generator.py`.

| Field | How it is computed |
| --- | --- |
| Filing ID | `SAR-AML-<today's date>-<account>` |
| ML risk score | Highest risk score among the transactions the account sends or receives |
| Threat level | CRITICAL ≥ 0.8; HIGH ≥ 0.5; otherwise MEDIUM |
| Analyzed volume | Sum of the account's last 20 incoming and last 20 outgoing transactions (mixed currencies) |
| Cycle rings | Number of the 150 reported rings containing the account |

**Suspicion indicators:**

| Code | Rule |
| --- | --- |
| **CIRC-01** Circular Laundering Loop | The account is in at least one reported ring |
| **CENT-02** Money Mule Bridge | Betweenness > 0.02 |
| **FLOW-03** Rapid Pass-Through | Absolute flow asymmetry < 0.15 and more than 4 transactions |
| **BEHV-04** Anomalous Volume | Added only when none of the above apply |

The report also contains:

- **Narrative:** a plain-English paragraph built from the numbers above.
- **Transaction ledger:** the 15 most recent transactions.
- **Download SAR (JSON):** saves the full report as a file.

---

## 12. Cypher Studio tab

### High level

Shows the graph queries an analyst would run in Neo4j's query language, Cypher, and their results.

### Low level

Five templates:

| Template | What the Cypher does |
| --- | --- |
| 1. Laundering Cycles (3–5 hops) | Finds paths that return to their start in 3–5 transfers |
| 2. Smurfing Fan-In (< 10K) | Finds collector accounts receiving 7,500–10,000 transfers from 3 or more different senders |
| 3. PageRank Centrality | Top 15 accounts by PageRank (Neo4j GDS) |
| 4. Betweenness (Mule Bridge) | Top 15 accounts by betweenness (Neo4j GDS) |
| 5. Louvain Communities | Community sizes (Neo4j GDS) |

**Two modes:**

- **Neo4j running (Docker):** the query text is sent to Neo4j over the Bolt protocol and real Cypher results come back.
- **Neo4j not running (default):** Cypher is **not parsed**. The server recognises which template the text matches (keywords such as `*3..5`, `mule`/`10000`, `louvain`, `pagerank`, `betweenness`) and returns the equivalent result computed in memory. Custom queries that match no template return the first 15 accounts' features. The badge above the results always says which mode was used.

**Table View / JSON View** switch between a table and the raw response.

---

## 13. REST API reference

| Endpoint | Returns | Used by |
| --- | --- | --- |
| `GET /api/overview` | Counts, rates, engine mode, lift, evaluation scope | Header, KPI cards |
| `GET /api/graph?max_nodes=200` | Nodes and links for the graph | Network Graph |
| `GET /api/cycles` | 150 rings with validation labels | Rings |
| `GET /api/leads?limit=60` | Top risk transactions from the test period | Alert Queue, Export SAR |
| `GET /api/leads/{index}/explain` | Plain-language risk factors for one transaction | API only |
| `GET /api/metrics` | Model metrics, curves, confusion matrix, feature importance, split | Model Benchmark |
| `GET /api/ablation` | Ablation stages | Ablation Study |
| `GET /api/shap` | SHAP ranking | SHAP |
| `GET /api/account/{id}` | Account features, last 20 in/out transactions, counterparties, bank | Account Inspector |
| `GET /api/sar/{id}` | SAR report | SAR window |
| `POST /api/cypher` | Query results | Cypher Studio |
| `POST /api/neo4j/sync` | Sync status | Sync to Neo4j |
| `POST /api/reload-dataset` | Reload status | Reload IBM Data |

Interactive documentation for all endpoints: [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 14. The 45 model features

**Transaction features (28)** from `ml_pipeline/feature_fusion.py`:

| Feature | Definition |
| --- | --- |
| `amount` | Amount paid, in the original currency |
| `log_amount` | log(1 + amount); reduces the effect of very large values |
| `hour_of_day` | Hour from the timestamp (0–23) |
| `is_wire`, `is_ach`, `is_cash`, `is_cheque`, `is_credit_card`, `is_reinvestment`, `is_bitcoin` | 1 if the payment format matches |
| `cur_us_dollar` … `cur_brazil_real` (15) | 1 if the payment currency matches |
| `cross_currency` | 1 if paid and received currencies differ |
| `cross_bank` | 1 if the sending and receiving banks differ |
| `self_transfer` | 1 if sender and receiver are the same account |

**Network features (17)**, looked up for both accounts of each transaction:

| Feature | Definition |
| --- | --- |
| `src_pagerank`, `dst_pagerank` | PageRank of sender / receiver |
| `src_betweenness`, `dst_betweenness` | Betweenness of sender / receiver |
| `src_in_degree`, `src_out_degree`, `dst_in_degree`, `dst_out_degree` | Distinct counterparties |
| `src_flow_asymmetry`, `dst_flow_asymmetry` | Flow asymmetry |
| `src_in_cycle`, `dst_in_cycle` | Ring membership |
| `src_cycle_count`, `dst_cycle_count` | Number of rings |
| `src_velocity_per_day` | Sender's transactions per day |
| `same_community` | 1 if both accounts are in the same Louvain community |
| `pagerank_ratio` | src_pagerank ÷ (dst_pagerank + 10⁻⁶) |

Accounts missing from the feature table get defaults (PageRank 0.001, others 0), which happens for new accounts in the strict test setting.

---

## 15. Thresholds at a glance

| Threshold | Value | Used in |
| --- | --- | --- |
| Ring length | 3 to 5 hops | Ring detection, Cypher template |
| Ring risk level | CRITICAL for 3–4 hops, HIGH for 5 hops | Rings, Cypher output |
| Bridge | betweenness > 0.02 | Graph colour, High Centrality filter, Alert Queue badge, SAR CENT-02, lead explanations |
| Label shown on graph | ring member, bridge, or PageRank > 0.005 | Network Graph |
| Risk badge colours | ≥ 0.8 red, ≥ 0.5 amber, ≥ 0.3 blue | Alert Queue |
| SAR threat level | ≥ 0.8 CRITICAL, ≥ 0.5 HIGH | SAR |
| Pass-through | absolute flow asymmetry < 0.15 with more than 4 transactions | SAR FLOW-03 |
| Structuring | amount below 10,000 (filter); 7,500–10,000 from 3+ senders (Cypher template) | Alert Queue filter, Cypher Studio |
| Confusion matrix cut-off | score ≥ 0.5 | Model Benchmark |

---

## 16. Questions you may be asked

**Why is PR-AUC only 0.12? Isn't that bad?**
At a 0.114% laundering rate, random guessing gives 0.001, so 0.12 is about 100 times better than chance. It also nearly triples the tabular baseline (0.045).

**Why not report accuracy or F1 at 0.5?**
Accuracy is meaningless here (99.9% for predicting "never laundering"). F1 at 0.5 depends on an arbitrary threshold. Investigations work from a ranked list, so PR-AUC and Precision@K are the relevant measures.

**How do you know the graph features are not leaking the answer?**
Ring detection never reads labels (and a unit test checks this). The split is by time. Training graph features use only training-period data. In the strict setting, test features also come only from the training-period graph.

**Why does the Alert Queue use the sample but the benchmark use the full data?**
The queue needs a live, interactive model on a laptop. The benchmark needs the full dataset to be statistically meaningful. Each panel states its source.

**Why is the demo sample's laundering rate 0.26%, not 0.089%?**
Complete rings were added so the ring views have something to show. They were selected without labels, but rings are where laundering concentrates. The sample is never used for reported metrics.

**What does an UNLABELLED ring mean?**
The ring has no transfer labelled laundering. It could be a legitimate loop or laundering the labels missed. The system flags structure; an analyst decides.

**Why does Community & Flow lower PR-AUC in the ablation?**
Louvain community numbers are arbitrary and not stable between the training graph and later data, so in the strict setting they add noise. With features recomputed on the full graph (the cumulative setting), the same stage helps.

**Is Cypher Studio real Cypher?**
With Neo4j running, yes. Without it, the server recognises the template and returns the equivalent precomputed result; the badge states which mode was used.
