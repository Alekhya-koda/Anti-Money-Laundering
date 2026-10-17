# AML Alert System: Account Risk Ranking

CS611 Machine Learning Engineering, Group 9 (Zhou Jie, Ho Wee Loong, Joyce Teo Hua Sze, Abinav Shajil, Alekhya Kodavatiganti, Zachary Artemus Kondas)

Banks in Singapore must monitor transactions under MAS Notice 626. Fixed rules ("flag anything above X") create many false alerts and miss new laundering patterns, and investigators can only review a limited number of cases a day. This project ranks **bank accounts** by laundering risk every morning, gives each alert **three plain-English reasons** (MAS FEAT transparency), and also offers a **real-time API** for one-off checks.

Data: IBM synthetic AML transactions (Altman et al., NeurIPS 2023), **HI-Small** variant, 5.08M transactions, Sep 1 - Sep 18 2022 ([Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)).

## Architecture

```
 data/raw/HI-Small_Trans.csv  (stands in for the bank's daily transaction file)
        |
        |  DAG "aml_data_pipeline", daily, one run per day ({{ ds }})
        v
 bronze  ->  silver (types, dedup, data quality report)
                |
                +--> gold feature store (24 account features, PySpark)
                +--> gold graph features (pagerank, loops, 2-hop; networkx)
                +--> gold label store (labels arrive 2 days late)

 DAG "aml_training_pipeline", run by hand (first models, or after a drift alert)
   model_train_check (train / valid snapshots ready?)
     +--> model_train_logistic_regression --+
     +--> model_train_lightgbm -------------+--> model_select --> MLflow registry: model versions,
     +--> model_train_lightgbm_graph -------+                     aliases champion / challenger / baseline,
                                                                  chart: champion and challenger vs the LR baseline

 DAG "aml_inference_pipeline", daily, waits for that day's aml_data_pipeline run
   model_inference (loads the models by alias from the MLflow registry)
     - champion -> alert list (top 500 + reasons)
     - challenger -> shadow scores (the LR baseline is only compared at training time)
   model_monitor
     - P@K / PR-AUC when labels arrive, data quality
     - feature, score and attribution drift (PSI)
     - big drift -> ERROR log "DRIFT ALERT" + drift_alerts.csv, a person decides whether to retrain

 FastAPI service (separate container): same feature store + same champion alias -> /accounts/{acct}, /predict, /alerts
```

| Service | URL | What |
|---|---|---|
| Airflow | http://localhost:8081 (admin / admin) | runs the data, training and inference DAGs |
| MLflow | http://localhost:5001 | experiments, model registry, aliases |
| API | http://localhost:8000/docs | real-time scoring (Swagger UI) |

## How to run

Needs Docker Desktop (give it at least 8 GB memory).

1. Download the data (only the 3 HI-Small files, not the whole Kaggle dataset)
   ```bash
   pip install kagglehub
   python scripts/download_data.py        # -> data/raw/HI-Small_Trans.csv, HI-Small_accounts.csv, HI-Small_Patterns.txt
   ```
2. Build and start everything
   ```bash
   docker compose up -d --build
   ```
3. Open Airflow at http://localhost:8081 (admin / admin). There are three DAGs, switch them on in this order:
   - `aml_data_pipeline`: has `catchup=True` from 2022-09-01 to 2022-09-10, so it replays the 10 days one by one (about 8 min).
   - `aml_training_pipeline`: has no schedule, switching it on runs nothing. When the data pipeline is done,
     train the first models with train date 2022-09-09: Trigger DAG w/ config -> `snapshotdate` = 2022-09-09.
     This makes v1 (LR, baseline), v2 (LightGBM, champion) and v3 (LightGBM + graph, challenger).
   - `aml_inference_pipeline`: switch it on after the first models exist. It catches up on the 10 days; a model only
     scores snapshots from its train date on, so the alert lists start with 2022-09-09.

   From the terminal:
   ```bash
   docker compose exec airflow-scheduler airflow dags unpause aml_data_pipeline
   docker compose exec airflow-scheduler airflow dags unpause aml_training_pipeline
   docker compose exec airflow-scheduler airflow dags trigger aml_training_pipeline -c '{"snapshotdate": "2022-09-09"}'
   docker compose exec airflow-scheduler airflow dags unpause aml_inference_pipeline
   ```
4. Drift: the monitor writes `DRIFT ALERT` as an ERROR in the log of `run_model_monitor` and adds a row to
   `datamart/gold/model_monitoring/drift_alerts.csv`. Nothing is retrained automatically. If you decide a new model is
   needed, trigger `aml_training_pipeline` again (e.g. `snapshotdate` = 2022-09-10). The new model becomes challenger
   and runs in shadow until you promote it (step 7).
5. Outputs:
   - alert lists: `datamart/gold/alert_list/alert_list_2022_09_09.csv`
   - model quality per day: `datamart/gold/model_monitoring/model_performance.csv`
   - drift reports and alerts: `datamart/gold/model_monitoring/drift_report_*.csv`, `drift_alerts.csv`
   - champion / challenger vs baseline at each training: `datamart/gold/model_training/model_comparison_*.png`
   - models: MLflow UI -> Models -> `aml-account-risk`
6. Try the API
   ```bash
   curl localhost:8000/health
   curl "localhost:8000/alerts?snapshot_date=2022-09-09&top=5"
   curl "localhost:8000/accounts/811D80C30?snapshot_date=2022-09-09"
   curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
        -d '{"features": {"out_cnt": 40, "out_uniq_cp": 35, "out_ach_share": 1.0}}'
   ```
7. Promote a model: in MLflow give the version the `champion` alias, then `curl -X POST localhost:8000/reload`.
   The next batch run picks it up automatically, the API switches without a restart.

Run one job by hand (same command the DAG uses):
```bash
docker compose exec airflow-scheduler bash -c "cd /opt/airflow/aml && python3 scripts/gold_feature_store.py --snapshotdate 2022-09-09"
```
Each model has its own training script, so it can be changed and run on its own:
```bash
docker compose exec airflow-scheduler bash -c "cd /opt/airflow/aml && python3 scripts/model_train_lightgbm.py --snapshotdate 2022-09-09"
```
Note that a training script registers a new MLflow version every time it runs, and `model_select.py` moves
the aliases, so normally they are run through `aml_training_pipeline`.

Tests (also run by GitHub Actions on every push / PR, see `.github/workflows/ci.yml`):
```bash
pip install -r requirements-api.txt pytest httpx
pytest -q tests
```

## Project structure

```
dags/
  aml_data_dag.py               aml_data_pipeline: ingest -> feature store + label store (daily)
  aml_training_dag.py           aml_training_pipeline: train the 3 models -> select (by hand)
  aml_inference_dag.py          aml_inference_pipeline: batch inference -> monitor (daily)
scripts/
  bronze_transactions.py        raw csv -> bronze (one day)
  silver_transactions.py        bronze -> silver + data quality report
  gold_feature_store.py         4-day account features for a snapshot (PySpark)
  gold_graph_features.py        graph features for a snapshot (networkx)
  gold_label_store.py           labels for a snapshot (2-day window)
  model_train_check.py          checks that the train / valid snapshots are ready
  model_train_logistic_regression.py   LR baseline (24 base features), grid search, MLflow
  model_train_lightgbm.py              LightGBM (24 base features), random search, MLflow
  model_train_lightgbm_graph.py        LightGBM (24 base + 6 graph features), random search, MLflow
  model_select.py               picks champion / challenger / baseline, draws them against the LR baseline
  model_inference.py            batch scoring, alert list with reason codes
  model_monitor.py              model quality, data quality, drift (PSI), drift alert
  download_data.py              gets the HI-Small files with kagglehub
  report_figures.py             tables and charts for the report / slides (run after the DAG)
  utils/                        processing functions + shared model helpers (metrics, PSI, reasons)
api/app.py                      FastAPI real-time inference
tests/                          unit tests (metrics, PSI, reason codes, model selection, API)
reports/                        final report (PDF + editable DOCX + HTML source), figures, result tables
AML_EDA.ipynb                   first EDA on Colab (DuckDB)
Dockerfile                      Airflow 2.6.1 + Java 17 + PySpark + LightGBM (from lab 5)
Dockerfile.api                  small python:3.9 image for the API
docker-compose.yaml             mlflow, airflow-init / webserver / scheduler, api
```

## Data design

**Snapshots.** One snapshot per morning. For snapshot date D:

| | window | why |
|---|---|---|
| features | D-4 to D-1 | only the past |
| label | D to D+1 | account *sends* at least one laundering payment in the next 2 days |

Labels are only complete 2 days later, so the DAG builds the features of `ds+1` but the labels of `ds-1`.
For model development we use three snapshots whose label windows do not overlap:

| snapshot | features | labels | use |
|---|---|---|---|
| 2022-09-05 | Sep 1-4 | Sep 5-6 | train |
| 2022-09-07 | Sep 3-6 | Sep 7-8 | tune / select (valid) |
| 2022-09-09 | Sep 5-8 | Sep 9-10 | test, evaluated once by the monitor when its labels arrive |

**Alert universe.** Self-transfers and cross-currency payments are excluded (about 0% laundering in the EDA).
**Amounts.** `amt_pct` = percentile of the amount within its currency, computed inside the 4-day window only (the EDA version used the whole period, which leaks future amounts).
**Why only up to Sep 10.** After Sep 10 normal volume collapses (about 480k/day to a few hundred) while laundering chains continue, 59% of transactions after Sep 10 are illicit. Using this tail would make the task look easy.

## Models

| model | features | role |
|---|---|---|
| Logistic regression (arcsinh + scaling) | 24 base | baseline, compared at every training |
| LightGBM | 24 base | champion candidate |
| LightGBM | 24 base + 6 graph | challenger candidate |

- Tuned on the valid snapshot (grid for LR, 20-try random search for LightGBM with early stopping on valid PR-AUC). All tries are saved in MLflow (`tuning_trials.csv`).
- **Class imbalance (0.2% positives):** class weights (`scale_pos_weight`, `class_weight="balanced"`) are part of the search, and they make the ranking worse on valid, so the chosen models do not use them. We handle imbalance in the evaluation instead: Precision@K / Recall@K (K = 500, 2000, 5000) and PR-AUC, never accuracy.
- **Selection rule:** the simplest model within 10% of the best valid PR-AUC becomes champion (the graph features cost an extra daily job). The next best runs as challenger in shadow mode. LR is the baseline: it is not scored every day, but every training run draws champion and challenger against it (`datamart/gold/model_training/model_comparison_<date>.png`).
- **Retraining:** never automatic. The monitor raises a drift alert (ERROR log + `drift_alerts.csv`) and a person decides whether to run `aml_training_pipeline`. A retrained model only becomes **challenger**; a person promotes it to champion after looking at its shadow results.

## Results (test snapshot 2022-09-09, 534 laundering accounts among 252,987)

| model | role | PR-AUC | P@500 | R@2000 | R@5000 |
|---|---|---|---|---|---|
| Rule: rank by out_cnt | - | 0.030 | 4.2% | 6.2% | 7.9% |
| Logistic regression | baseline (v1) | 0.042 | 5.0% | 10.1% | 13.3% |
| LightGBM | champion (v2) | 0.068 | 9.4% | 15.0% | 20.2% |
| LightGBM + graph | challenger (v3, shadow) | 0.081 | 13.8% | 17.0% | 21.7% |

The champion's top 500 held 47 laundering accounts (9 of the top 10). The shadow challenger with graph features
did better on this day (+4.2 points P@500, paired bootstrap 95% CI 2.2 to 6.2), so the plan is to promote it after a
few more labelled days. Full numbers with intervals: `reports/tables/test_results.csv`, report: `reports/AML_Final_Report.pdf`.
To rebuild the charts and tables after a run:
```bash
docker compose exec airflow-scheduler bash -c "cd /opt/airflow/aml && python3 scripts/report_figures.py"
```

## Notes

- `AML_EDA.ipynb` was our first exploration on Colab. Its baseline cell ranks with `nlargest`, which breaks ties by row order; on `out_big_cnt` (few non-zero values) this gave a fake 100% recall at K=5000. The pipeline breaks ties randomly (`utils/model_utils.py`) and the monitor recomputes the rule baselines.
- The data is synthetic and not Singapore-specific; ACH carries most laundering in this simulator, so the model leans on `out_ach_share`.
- Illicit senders with no activity in the 4-day feature window cannot be scored (about 25% of them in the test window).
- The Fernet key and admin password in `docker-compose.yaml` are lab defaults for local use only.
