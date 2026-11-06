# AML Alert System

CS611 Machine Learning Engineering, Group 9

Zhou Jie, Ho Wee Loong, Joyce Teo Hua Sze, Abinav Shajil, Alekhya Kodavatiganti, Zachary Artemus Kondas

## How to run

You need Docker Desktop with at least 8 GB of memory.

### 1. Download the data

We only use the three HI-Small files from the Kaggle dataset. They are saved to `data/raw/`.

```bash
pip install kagglehub
python scripts/download_data.py
```

### 2. Start the services

```bash
docker compose up -d --build
```

- Airflow: http://localhost:8081 (admin / admin)
- MLflow: http://localhost:5001
- API docs: http://localhost:8000/docs

### 3. Run the pipelines

In Airflow, turn on the three DAGs in this order:

1. `aml_data_pipeline` runs once per day from 2022-09-01 to 2022-09-10 (catchup), about 7 minutes in total.
2. `aml_training_pipeline` has no schedule. After the data pipeline has finished, trigger it with `snapshotdate` set to `2022-09-09`. This registers v1 (logistic regression, baseline), v2 (LightGBM, champion) and v3 (LightGBM + graph features, challenger).
3. `aml_inference_pipeline` scores each day with the registered models. Turn it on after training. The first alert list is for 2022-09-09, because a model doesn't score dates before its train date.

### 4. Check the outputs

- Alert lists: `datamart/gold/alert_list/`
- Daily model performance: `datamart/gold/model_monitoring/model_performance.csv`
- Drift reports and alerts: `datamart/gold/model_monitoring/`
- Model comparison chart from each training run: `datamart/gold/model_training/model_comparison_*.png`
- Registered models: MLflow UI, Models tab, `aml-account-risk`

### 5. Call the API

```bash
curl localhost:8000/health
curl "localhost:8000/alerts?snapshot_date=2022-09-09&top=5"
curl "localhost:8000/accounts/811D80C30?snapshot_date=2022-09-09"
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"features": {"out_cnt": 40, "out_uniq_cp": 35, "out_ach_share": 1.0}}'
```

### 6. Retrain and promote

The monitor doesn't retrain anything by itself. When drift is too large it logs `DRIFT ALERT` in the `run_model_monitor` task and adds a row to `drift_alerts.csv`.

To retrain, trigger `aml_training_pipeline` again with a later date, e.g. `2022-09-10`. The two new models get the `shadow_1` and `shadow_2` aliases and are scored next to the champion every day. The old challenger is no longer scored.

If a shadow model beats the champion on several labelled days (P@500, PR-AUC, Recall@2000), promote it:

```bash
docker compose exec airflow-scheduler bash -c "cd /opt/airflow/aml && python3 scripts/model_promote.py \
  --version 6 --by 'Zhou Jie' --reason 'v6 beat v2 on 3 labelled days'"
```

The promoted version becomes `champion` and the other shadow model becomes `challenger`. The script logs the change to `datamart/gold/model_training/promotion_log.csv` and tells the API to reload. To roll back, run the same command with an earlier champion version, e.g. `--version 2`.
