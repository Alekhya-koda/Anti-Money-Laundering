# AML Alert System: Account Risk Ranking

CS611 Machine Learning Engineering, Group 9 (Zhou Jie, Ho Wee Loong, Joyce Teo Hua Sze, Abinav Shajil, Alekhya Kodavatiganti, Zachary Artemus Kondas)

## How to run

Needs Docker Desktop (give it at least 8 GB memory).

| Service | URL | What |
|---|---|---|
| Airflow | http://localhost:8081 (admin / admin) | runs the data, training and inference DAGs |
| MLflow | http://localhost:5001 | experiments, model registry, aliases |
| API | http://localhost:8000/docs | real-time scoring (Swagger UI) |

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
   - `aml_data_pipeline`: has `catchup=True` from 2022-09-01 to 2022-09-10, so it replays the 10 days one by one (about 7 min).
   - `aml_training_pipeline`: has no schedule, switching it on runs nothing. When the data pipeline is done,
     train the first models with train date 2022-09-09: Trigger DAG w/ config -> `snapshotdate` = 2022-09-09.
     This makes v1 (LR, baseline), v2 (LightGBM, champion) and v3 (LightGBM + graph, challenger).
   - `aml_inference_pipeline`: switch it on after the first models exist. It catches up on the 10 days; a model only
     scores snapshots from its train date on, so the alert lists start with 2022-09-09.

4. Drift: the monitor writes `DRIFT ALERT` as an ERROR in the log of `run_model_monitor` and adds a row to
   `datamart/gold/model_monitoring/drift_alerts.csv`. Nothing is retrained automatically. If you decide a new model is
   needed, trigger `aml_training_pipeline` again (e.g. `snapshotdate` = 2022-09-10). Its two new models get the
   aliases `shadow_1` and `shadow_2` and run in shadow next to the champion until you promote one of them (step 7).
   The old challenger stops: it was trained on the same old data as the champion.
5. Outputs:
   - alert lists: `datamart/gold/alert_list/alert_list_2022_09_09.csv`
   - model quality per day: `datamart/gold/model_monitoring/model_performance.csv`
   - drift reports and alerts: `datamart/gold/model_monitoring/drift_report_*.csv`, `drift_alerts.csv`
   - the models vs the LR baseline at each training: `datamart/gold/model_training/model_comparison_*.png`
   - who promoted or rolled back which model and why: `datamart/gold/model_training/promotion_log.csv`
   - models: MLflow UI -> Models -> `aml-account-risk`
6. Try the API
   ```bash
   curl localhost:8000/health
   curl "localhost:8000/alerts?snapshot_date=2022-09-09&top=5"
   curl "localhost:8000/accounts/811D80C30?snapshot_date=2022-09-09"
   curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
        -d '{"features": {"out_cnt": 40, "out_uniq_cp": 35, "out_ach_share": 1.0}}'
   ```
7. Promote a model, when a shadow model beats the champion on several labelled days (P@500, PR-AUC, Recall@2000).
   One command does every step:
   ```bash
   docker compose exec airflow-scheduler bash -c "cd /opt/airflow/aml && python3 scripts/model_promote.py \
     --version 6 --by 'Zhou Jie' --reason 'v6 beat v2 on 3 labelled days'"
   ```
   It checks that the version is a shadow model, gives it the `champion` alias and the other shadow model the
   `challenger` alias, removes `shadow_1` / `shadow_2`, records who, when and why (tags on the model version and a
   row in `datamart/gold/model_training/promotion_log.csv`) and calls `POST /reload` on the API. It is one champion
   and one challenger again; the next batch run picks them up automatically.
   Rollback is the same command with an older champion, e.g. `--version 2 --reason "..."`.
   Moving the aliases by hand in the MLflow UI still works, but leaves no record. The API checks the champion alias
   every minute (`CHAMPION_CHECK_SECONDS`), so it follows the change even without `/reload`.