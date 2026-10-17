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

# data pipeline: one run = one day of transactions ({{ ds }}), we replay Sep 1 - Sep 10 2022 as if it was production
#   - bronze / silver ingest the transactions of the day (+ data quality report)
#   - features for tomorrow morning use the last 4 days up to today        -> ds + 1
#   - labels need 2 days of future data, so today the labels of yesterday's
#     snapshot are complete                                                 -> ds - 1
# the other 2 DAGs only read what this one writes:
#   aml_training_pipeline  (dags/aml_training_dag.py, run by hand)  trains models from the feature + label store
#   aml_inference_pipeline (dags/aml_inference_dag.py, daily)       scores the feature store, then monitors
with DAG(
    'aml_data_pipeline',
    default_args=default_args,
    description='AML account risk: ingest transactions, build the feature store and the label store',
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


    # --- done for the day, aml_inference_pipeline waits for this task ---
    data_pipeline_completed = DummyOperator(task_id="data_pipeline_completed")

    feature_store_completed >> data_pipeline_completed
    label_store_completed >> data_pipeline_completed
