from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# the project folder is mounted at /opt/airflow/aml, scripts use paths relative to it
run_script = 'cd /opt/airflow/aml && python3 scripts/'

# one run = one day of transactions ({{ ds }}), we replay Sep 1 - Sep 10 2022 as if it was production
#   - bronze / silver ingest the transactions of the day
#   - features for tomorrow morning use the last 4 days up to today        -> ds + 1
#   - labels need 2 days of future data, so today the labels of yesterday's
#     snapshot are complete                                                 -> ds - 1
#   - the model scores tomorrow morning's snapshot                          -> ds + 1
with DAG(
    'aml_daily_pipeline',
    default_args=default_args,
    description='AML account risk: ingest, feature/label store, train, batch inference, monitor',
    schedule_interval='@daily',
    start_date=datetime(2022, 9, 1),
    end_date=datetime(2022, 9, 10),
    catchup=True,
    max_active_runs=1,
) as dag:

    # --- ingestion ---
    dep_check_source_transactions = BashOperator(
        task_id='dep_check_source_transactions',
        bash_command='test -f /opt/airflow/aml/data/raw/HI-Small_Trans.csv',
    )

    bronze_transactions = BashOperator(
        task_id='run_bronze_transactions',
        bash_command=run_script + 'bronze_transactions.py --snapshotdate "{{ ds }}"',
    )

    silver_transactions = BashOperator(
        task_id='run_silver_transactions',
        bash_command=run_script + 'silver_transactions.py --snapshotdate "{{ ds }}"',
    )

    dep_check_source_transactions >> bronze_transactions >> silver_transactions


    # --- feature store ---
    gold_feature_store = BashOperator(
        task_id='run_gold_feature_store',
        bash_command=run_script + 'gold_feature_store.py --snapshotdate "{{ macros.ds_add(ds, 1) }}"',
    )

    gold_graph_features = BashOperator(
        task_id='run_gold_graph_features',
        bash_command=run_script + 'gold_graph_features.py --snapshotdate "{{ macros.ds_add(ds, 1) }}"',
    )

    feature_store_completed = DummyOperator(task_id="feature_store_completed")

    silver_transactions >> gold_feature_store >> feature_store_completed
    silver_transactions >> gold_graph_features >> feature_store_completed


    # --- label store ---
    gold_label_store = BashOperator(
        task_id='run_gold_label_store',
        bash_command=run_script + 'gold_label_store.py --snapshotdate "{{ macros.ds_add(ds, -1) }}"',
    )

    label_store_completed = DummyOperator(task_id="label_store_completed")

    silver_transactions >> gold_label_store >> label_store_completed


    # --- model training (first models, weekly retrain, or retrain when the monitor found drift) ---
    # the check exits with 99 when there is nothing to train today,
    # Airflow then marks it skipped and the training tasks after it are skipped too
    model_train_check = BashOperator(
        task_id='run_model_train_check',
        bash_command=run_script + 'model_train_check.py --snapshotdate "{{ macros.ds_add(ds, 1) }}"',
        skip_on_exit_code=99,
    )

    feature_store_completed >> model_train_check
    label_store_completed >> model_train_check

    # one task per candidate model. The executor runs one task at a time, priority_weight keeps the
    # order LR -> LightGBM -> LightGBM + graph so the MLflow version numbers come out the same every time
    model_train_logistic_regression = BashOperator(
        task_id='run_model_train_logistic_regression',
        bash_command=run_script + 'model_train.py --snapshotdate "{{ macros.ds_add(ds, 1) }}" --model logistic_regression',
        priority_weight=3,
        weight_rule='absolute',
    )

    model_train_lightgbm = BashOperator(
        task_id='run_model_train_lightgbm',
        bash_command=run_script + 'model_train.py --snapshotdate "{{ macros.ds_add(ds, 1) }}" --model lightgbm',
        priority_weight=2,
        weight_rule='absolute',
    )

    model_train_lightgbm_graph = BashOperator(
        task_id='run_model_train_lightgbm_graph',
        bash_command=run_script + 'model_train.py --snapshotdate "{{ macros.ds_add(ds, 1) }}" --model lightgbm_graph',
        priority_weight=1,
        weight_rule='absolute',
    )

    # compares the 3 new models and moves the champion / challenger / baseline aliases in MLflow
    model_select = BashOperator(
        task_id='run_model_select',
        bash_command=run_script + 'model_select.py --snapshotdate "{{ macros.ds_add(ds, 1) }}"',
    )

    model_train_check >> model_train_logistic_regression >> model_select
    model_train_check >> model_train_lightgbm >> model_select
    model_train_check >> model_train_lightgbm_graph >> model_select


    # --- model inference (champion = production, challenger / baseline = shadow) ---
    # none_failed: also runs on the days the training was skipped, then it uses the current champion
    model_inference = BashOperator(
        task_id='run_model_inference',
        bash_command=run_script + 'model_inference.py --snapshotdate "{{ macros.ds_add(ds, 1) }}" --topk 500',
        trigger_rule='none_failed',
    )

    model_select >> model_inference


    # --- model monitoring ---
    model_monitor = BashOperator(
        task_id='run_model_monitor',
        bash_command=run_script + 'model_monitor.py --snapshotdate "{{ macros.ds_add(ds, 1) }}"',
    )

    model_inference >> model_monitor
