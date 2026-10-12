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


    # --- model training (first model, weekly retrain, or retrain when the monitor found drift) ---
    model_train = BashOperator(
        task_id='run_model_train',
        bash_command=run_script + 'model_train.py --snapshotdate "{{ macros.ds_add(ds, 1) }}"',
    )

    feature_store_completed >> model_train
    label_store_completed >> model_train


    # --- model inference (champion = production, challenger / baseline = shadow) ---
    model_inference = BashOperator(
        task_id='run_model_inference',
        bash_command=run_script + 'model_inference.py --snapshotdate "{{ macros.ds_add(ds, 1) }}" --topk 500',
    )

    model_train >> model_inference


    # --- model monitoring ---
    model_monitor = BashOperator(
        task_id='run_model_monitor',
        bash_command=run_script + 'model_monitor.py --snapshotdate "{{ macros.ds_add(ds, 1) }}"',
    )

    model_inference >> model_monitor
