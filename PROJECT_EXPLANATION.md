# Sentinel Graph: Project Explanation

**Anti-Money Laundering Network Forensics Using Graph Analytics and Machine Learning**

Aakash R (23MIA1011), Gowthamkrishna R (23MIA1087)
CSE4074 Forensic Analytics, Fall 2026-27, Vellore Institute of Technology, Chennai
Professor: Dr. S. Johanan Joysingh

This document follows the order of the presentation. Each numbered section corresponds to one or two slides.

---

## 1. Problem statement

Money laundering disguises the origin of illegal funds by moving them through many accounts until they appear legitimate. Banks must detect it and report it to regulators, but:

- **Laundering is rare.** In the dataset used here, about 1 in 1,000 transactions is laundering.
- **Single transactions look normal.** A launderer's individual transfers are often ordinary amounts at ordinary times.
- **The evidence is in the pattern.** Laundering shows up in *how accounts are connected*: money moving in circles, through bridge accounts, or collected from many small deposits.

Traditional rule-based and tabular machine-learning systems examine each transaction in isolation, so they miss these network patterns.

## 2. Objectives

1. Represent bank transfers as a **directed graph** of accounts.
2. Detect **circular layering rings** (money that returns to its origin within 3 to 5 transfers) **without using labels**.
3. Compute **network features** for each account and combine them with transaction features.
4. Measure, with a **leakage-free evaluation**, how much network features improve detection over a strong tabular baseline.
5. Provide an **investigator dashboard** and an automatic **Suspicious Activity Report (SAR)**.

## 3. Dataset

**IBM Transactions for Anti Money Laundering, HI-Small** (Altman et al., 2023; Kaggle; CDLA-Sharing-1.0 licence).

| Property | Value |
| --- | --- |
| Transactions | 5,078,345 |
| Labelled laundering transactions | 5,177 |
| Sending accounts | about 497,000 |
| Period | 1 to 18 September 2022 |
| Fields | timestamp, sending bank and account, receiving bank and account, amounts, currencies, payment format, laundering label |

**Data cleaning decision.** After 10 September the simulator produces only a few hundred transactions, about 60% of them laundering. Keeping them would let a model learn "late date means laundering". The project uses **1 to 10 September only** (5,077,237 transactions, 4,522 laundering, 0.089%).

## 4. System architecture

```
IBM HI-Small CSV
      │
      ▼
Data loader ── drops post-10-Sep artifact, label-blind sampling
      │
      ▼
Graph builder ── accounts = nodes, transfers = weighted directed edges
      │
      ▼
Graph analytics ── PageRank · betweenness · Louvain · flow · cycles · velocity
      │
      ▼
Feature fusion ── 28 transaction features + 17 network features = 45 per transaction
      │
      ▼
Model training ── temporal split · 3 models · ablation · SHAP
      │
      ▼
FastAPI server ──► D3.js dashboard · SAR reports · Cypher console (optional Neo4j)
```

| Layer | Technology |
| --- | --- |
| Graph analytics | NetworkX; Neo4j 5 with Graph Data Science (optional, via Docker) |
| Machine learning | scikit-learn (Random Forest, HistGradientBoosting), SHAP |
| Backend | Python, FastAPI |
| Frontend | HTML, CSS, D3.js |
| Deployment | Docker Compose |

## 5. Graph construction

Each **account** is a node. All transfers from account A to account B are combined into one **directed edge** A → B, which stores the total amount and the number of transfers. Aggregation is done with a single pandas `groupby`. The 3.2-million-row training window, for example, becomes a graph of about 513,000 accounts and 940,000 edges in seconds.

## 6. Network features

| Feature | What it measures | Why it matters for laundering |
| --- | --- | --- |
| PageRank | How much money flows toward an account, weighted by the importance of its senders | Collection points for laundered funds score high |
| Betweenness centrality | How often an account lies on shortest paths between others | Money mules act as bridges between groups |
| Louvain community | Which tightly connected cluster the account belongs to | Laundering rings form their own clusters |
| In-degree / out-degree | Number of distinct senders / receivers | Many senders into one account suggests structuring |
| Flow asymmetry | Outflow minus inflow, relative to total volume | Pass-through accounts send out what they receive |
| Ring membership and count | Whether, and in how many, 3 to 5 hop cycles the account appears | Circular flows are a classic layering technique |
| Transaction velocity | Transactions per day and shortest gap between transactions | Rapid successive transfers are suspicious |

Betweenness is approximated from 40 sampled source nodes on large graphs, which keeps it fast.

## 7. Label-free cycle detection

A cycle is a path A → B → C → A that returns to its start. The detector:

1. Starts from every account that both sends and receives money, in a fixed random order.
2. Follows outgoing edges with a depth-first search up to 5 hops.
3. Records a cycle when the search returns to the start after 3 to 5 hops.
4. Stores each cycle once, however many accounts it was reached from.
5. Limits the search effort per start account, so very busy accounts cannot stall it.

**It never reads the laundering label.** An earlier version started its search from accounts already labelled as laundering, which leaked the answer into the model's features. Removing that leak was essential for an honest evaluation.

**Validation.** In the demo sample, 41 of the 150 rings found without labels contain at least one labelled laundering transfer.

## 8. Transaction features and feature fusion

Each transaction becomes one row of 45 numbers:

- **28 transaction features:** amount, log amount, hour of day, 7 payment-format flags, 15 currency flags, and flags for cross-currency, cross-bank and self-transfers.
- **17 network features:** the sender's and receiver's network features from Section 6, whether both are in the same community, and the ratio of their PageRanks.

## 9. Models

| Model | Input | Purpose |
| --- | --- | --- |
| Tabular Random Forest | 28 transaction features | Baseline: what a non-graph system can do |
| Graph-augmented Random Forest | All 45 features | Measures the value of network features |
| Graph-augmented Gradient Boosting | All 45 features | Strongest model |

All models compensate for class imbalance: the Random Forests use balanced class weights, and gradient boosting uses per-sample weights. Training uses all laundering transactions plus 300,000 randomly chosen normal transactions, to keep training fast.

## 10. Evaluation methodology

Fraud models are easy to over-score by accident. The project uses the following safeguards:

| Risk | Safeguard |
| --- | --- |
| Labels leaking into features | Cycle detection never reads labels |
| Learning from the future | Temporal split: train on the earliest 75%, test on the latest 25% |
| Future transactions shaping training features | Training-period graph features are computed only from training-period transactions |
| Unrealistic class balance | The test set keeps the real 0.114% laundering rate |
| Simulator artifact | Transactions after 10 September are excluded |

**Why not accuracy?** With 0.1% laundering, a model that always predicts "normal" is 99.9% accurate and useless. The project reports:

- **PR-AUC:** ranking quality focused on the rare class. Random guessing scores about 0.001.
- **ROC-AUC:** overall ranking quality.
- **Precision@K:** the fraction of the top K alerts that are real laundering. This is what an investigator experiences.

## 11. Results

Test set: 1,269,309 transactions from 8 to 10 September 2022, of which 1,449 are laundering.

| Model | PR-AUC | ROC-AUC | P@10 | P@50 | P@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tabular Random Forest | 0.0447 | 0.9208 | 0.10 | 0.10 | 0.17 |
| Graph-augmented Random Forest | 0.0716 | 0.9665 | 1.00 | 0.62 | 0.39 |
| Graph-augmented Gradient Boosting | **0.1211** | **0.9746** | 0.80 | **0.82** | **0.61** |

**Key findings:**

- Network features **improve PR-AUC by 171%** (0.045 → 0.121) over a 28-feature tabular baseline.
- In the 100 highest-risk alerts, the graph model finds **61** laundering transactions, against **17** for the tabular model: about 3.6 times more productive investigator time.
- These numbers are low in absolute terms, as expected at a 0.1% positive rate. They are reported honestly, without the inflated scores a leaky evaluation would give.

## 12. Ablation study

Feature groups are added one at a time to a Random Forest, measuring PR-AUC on the same test set:

| Stage | Features added | Features | PR-AUC | Change |
| --- | --- | ---: | ---: | ---: |
| 1 | Tabular baseline | 28 | 0.0407 | baseline |
| 2 | + PageRank | 31 | 0.0521 | +0.0114 |
| 3 | + Betweenness | 33 | 0.0581 | +0.0060 |
| 4 | + Community and flow asymmetry | 41 | 0.0429 | −0.0152 |
| 5 | + Ring membership (full model) | 45 | 0.0884 | **+0.0455** |

- **Ring membership is the single most valuable group,** even though it is detected without labels.
- **Community features reduced performance in this strict setting.** Louvain cluster IDs computed on the training graph do not line up with clusters in later data. This is reported as a limitation rather than hidden.

## 13. Explainability

The project ranks features by their mean absolute SHAP value (TreeExplainer) on the graph-augmented Random Forest, computed over 80 test transactions. If the `shap` package is not installed, it falls back to Random Forest feature importances.

| Rank | Feature | Mean \|SHAP\| | Type |
| ---: | --- | ---: | --- |
| 1 | `is_ach` (ACH payment format) | 0.1995 | Transaction |
| 2 | `is_cheque` | 0.0546 | Transaction |
| 3 | `amount` | 0.0502 | Transaction |
| 4 | `src_velocity_per_day` | 0.0439 | Network |
| 5 | `log_amount` | 0.0416 | Transaction |
| 6 | `is_credit_card` | 0.0327 | Transaction |
| 7 | `src_out_degree` | 0.0243 | Network |
| 8 | `pagerank_ratio` | 0.0207 | Network |
| 9 | `dst_in_degree` | 0.0200 | Network |
| 10 | `src_in_degree` | 0.0185 | Network |

- **ACH payment format** is the strongest single signal, a known characteristic of the IBM dataset.
- **Network features** make up 5 of the top 10 and about 19% of the attribution among the top 8, which is consistent with their measured lift in Sections 11 and 12.

Each alert in the dashboard also lists plain-language risk factors: ring participation, bridge position, high PageRank and large amount.

## 14. Dashboard

| Tab | What the investigator sees |
| --- | --- |
| Network Graph | Interactive force-directed graph; ring accounts highlighted; click an account to inspect it |
| Rings | Detected 3 to 5 hop rings, with path, volume and how many hops carry labelled laundering |
| Alert Queue | Transactions ranked by risk, with search, filters and network indicators |
| Model Benchmark | Model comparison, precision-recall curves, confusion matrix and Precision@K |
| Ablation Study | Stage-by-stage table and chart |
| SHAP | Feature importance ranking and interpretation |
| Cypher Studio | Graph queries against Neo4j, or equivalent built-in results when Neo4j is not running |

Evaluation tabs show the full-data results and state where they come from. The interactive views use a 102,907-row demo sample, which keeps every detected ring intact and starts in about 30 seconds on a laptop.

## 15. Suspicious Activity Report

For any account, the system generates a SAR containing:

- a report ID and threat level
- coded indicators: `CIRC-01` ring participation, `CENT-02` bridge position, `FLOW-03` pass-through flow, `BEHV-04` unusual volume
- the rings the account belongs to and its recent transactions
- a plain-English narrative for compliance officers

The report can be downloaded as JSON.

## 16. Limitations

- **The data is simulated.** IBM HI-Small is synthetic; real bank data would be noisier.
- **Low absolute precision.** At a 0.5 score threshold the model raises many false alerts. It is designed for ranking alerts, not automatic blocking.
- **Mixed currencies.** Amounts are in their original currency and are not converted.
- **Community features** are unstable across time windows (Section 12).
- **Static graph features.** Features are computed once per period rather than updated with every transaction.

## 17. Future work

- Convert all amounts to one currency.
- Compute features over sliding time windows.
- Try graph neural networks (for example GraphSAGE or GIN) that learn from the network directly.
- Evaluate on the larger IBM HI-Medium and LI variants.
- Use Neo4j Graph Data Science for all feature computation at production scale.

## 18. How to run

See the [README](README.md). In short:

```bash
pip install -r requirements.txt
./run.sh
```

Then open [http://localhost:8000](http://localhost:8000).

## 19. References

1. E. Altman, J. Blanuša, L. von Niederhäusern, B. Egressy, A. Anghel, K. Atasu. *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models.* NeurIPS Datasets and Benchmarks, 2023.
2. IBM Transactions for Anti Money Laundering (AML), Kaggle: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
3. L. Page, S. Brin, R. Motwani, T. Winograd. *The PageRank Citation Ranking: Bringing Order to the Web.* Stanford InfoLab, 1999.
4. V. Blondel, J.-L. Guillaume, R. Lambiotte, E. Lefebvre. *Fast unfolding of communities in large networks.* Journal of Statistical Mechanics, 2008.
5. S. Lundberg, S.-I. Lee. *A Unified Approach to Interpreting Model Predictions.* NeurIPS, 2017.
6. Financial Action Task Force (FATF). *Money Laundering and Terrorist Financing Typologies.*
