# Sentinel Graph: Anti-Money Laundering Network Forensics Using Graph Analytics and Machine Learning

Developed as part of the course on "Forensic Analytics" at Vellore Institute of Technology, Chennai.

by Aakash R (23MIA1011), Gowthamkrishna R (23MIA1087)

Subject Code: CSE4074

Semester: Fall 2026-27

Professor: Dr. S. Johanan Joysingh

---

Sentinel Graph is an anti-money laundering (AML) investigation prototype. It models bank transfers as a directed network, detects circular money paths of 3 to 5 hops, computes account-level network features, and uses them to rank transactions for investigator review. A web dashboard supports graph exploration, ring inspection, risk triage, model evaluation, feature explanations, account dossiers, Suspicious Activity Report (SAR) generation and optional Neo4j Cypher queries.

The full project explanation, structured to follow the presentation, is in **[PROJECT_EXPLANATION.md](PROJECT_EXPLANATION.md)**. Every dashboard panel, number and threshold is explained in **[DASHBOARD_GUIDE.md](DASHBOARD_GUIDE.md)**.

## Results

Evaluated on the full IBM HI-Small dataset window (1 to 10 September 2022, 5,077,237 transactions). Transactions are split by time: the earliest 75% are used for training and the latest 25% (1,269,309 transactions, 0.114% laundering) for testing. Graph features for test transactions come only from the training-period graph, so no future information is used.

| Model | Features | PR-AUC | ROC-AUC | Precision@100 |
| --- | ---: | ---: | ---: | ---: |
| Tabular Random Forest (baseline) | 28 | 0.0447 | 0.9208 | 0.17 |
| Graph-augmented Random Forest | 45 | 0.0716 | 0.9665 | 0.39 |
| Graph-augmented Gradient Boosting | 45 | **0.1211** | **0.9746** | **0.61** |

A random ranking would score a PR-AUC of about 0.001 at this laundering rate. Among its 100 highest-risk alerts, the graph gradient-boosting model finds 61 labelled laundering transfers, compared with 17 for the tabular baseline.

## Features

- **Graph construction** from raw transfers into a weighted directed account network (NetworkX, optionally Neo4j).
- **Label-free cycle detection** for 3 to 5 hop circular layering rings.
- **Account network features:** PageRank, betweenness, Louvain community, in/out degree, flow asymmetry, ring membership and transaction velocity.
- **Three models compared:** tabular Random Forest, graph-augmented Random Forest and graph-augmented gradient boosting.
- **Five-stage ablation study** measuring what each feature group adds.
- **Feature attribution** with SHAP (TreeExplainer), falling back to Random Forest importances when SHAP is unavailable.
- **SAR generator** producing a structured report for any account.
- **Investigator dashboard** built with D3.js and served by FastAPI.

## Project structure

| Path | Purpose |
| --- | --- |
| `api/app.py` | FastAPI server and REST endpoints; serves the dashboard |
| `neo4j_engine/algorithms.py` | Graph construction, cycle detection and account features |
| `neo4j_engine/graph_service.py` | In-memory graph state, optional Neo4j connection, account dossiers, Cypher console |
| `neo4j_engine/cypher_queries.cyp` | Reference Cypher queries for Neo4j |
| `ml_pipeline/feature_fusion.py` | Joins transaction and account features into the 45-column model input |
| `ml_pipeline/train_models.py` | Temporal split, model training, ablation study and SHAP |
| `ml_pipeline/evaluation.py` | PR-AUC, ROC-AUC, Precision@K, confusion matrix and curves |
| `ml_pipeline/sar_generator.py` | Suspicious Activity Report generation |
| `ml_pipeline/build_artifacts.py` | One-time full-data evaluation and demo sample builder |
| `ml_pipeline/run_ablation.py` | Command-line ablation run |
| `data/ibm_aml_loader.py` | Dataset loading, label-blind sampling and synthetic test data |
| `data/HI-Small_Trans_sample.csv` | Lightweight demo sample (102,907 transactions, 150 intact rings) |
| `data/precomputed_metrics.json` | Saved full-data evaluation shown in the dashboard |
| `dashboard/` | Dashboard HTML, CSS and JavaScript |
| `tests/` | Unit and API tests |

## Quick start

Requires Python 3.10 or later. Neo4j and Docker are optional.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run.sh
```

Open [http://localhost:8000](http://localhost:8000). Startup takes about 30 seconds. Model training uses 2 CPU cores by default (set `AML_N_JOBS` to change it). Stop the server with `Ctrl+C`.

### With Docker and Neo4j

```bash
docker compose up --build
```

| Service | Address |
| --- | --- |
| Dashboard | [http://localhost:8000](http://localhost:8000) |
| API documentation | [http://localhost:8000/docs](http://localhost:8000/docs) |
| Neo4j Browser | [http://localhost:7474](http://localhost:7474) (user `neo4j`, password `password123`) |

Neo4j is limited to 1 CPU and 1.5 GB of memory, and the app to 2 CPUs and 3 GB. Stop everything with `docker compose down`.

## Data

The repository includes everything the dashboard needs: the demo sample and the saved full-data evaluation. The full dataset is only required to rebuild them.

1. Download `HI-Small_Trans.csv` from the [IBM Transactions for Anti Money Laundering dataset on Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml).
2. Place it at `data/HI-Small_Trans.csv`.
3. Rebuild the artifacts (reads about 5 million rows once; takes 2 to 4 minutes):

```bash
python3 -m ml_pipeline.build_artifacts
```

To repeat the strict full-data ablation on its own:

```bash
python3 -m ml_pipeline.run_ablation --full --test-graph train
```

Both commands are memory and CPU intensive. Close other heavy applications first.

## Evaluation safeguards

- **Label-free cycle detection.** The search never reads the laundering label, so ring features are safe to use as model inputs.
- **Temporal split.** Models are trained on earlier transactions and tested on later ones.
- **No future graph information.** Training-period graph features are built only from training-period transactions.
- **Real class balance.** The test set keeps the real 0.114% laundering rate. Only training negatives are undersampled, for speed.
- **Simulator artifact removed.** Transactions after 10 September 2022 are excluded; in that tail the IBM simulator produces almost no normal traffic and about 60% of rows are laundering.
- **Demo sample kept separate.** The dashboard's sample is chosen without labels and is used for exploration only. All reported metrics come from the full-data evaluation.

## Tests

```bash
python3 -m pytest -q tests
```

## Dataset licence

The IBM AML dataset is distributed under the CDLA-Sharing-1.0 licence. The full dataset is not included in this repository.
