from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# the project folder is mounted at /opt/airflow/aml, scripts use paths relative to it
run_script = 'cd /opt/airflow/aml && python3 scripts/'

# inference pipeline: every morning score the new snapshot with the models in the MLflow registry
#   champion = production (alert list), challenger / baseline = shadow (scores kept for the monitor)
# then monitor model quality, drift and data quality. Big drift -> ERROR log "DRIFT ALERT",
# no automatic retraining: a person decides whether to run aml_training_pipeline
# Same schedule as aml_data_pipeline: the run of day ds waits for that day's data run, then scores ds + 1.
# Switch this DAG on after the first models are trained. A model only scores snapshots from its
# train date on, so the catchup does not score the old snapshots that were used to train it.
with DAG(
    'aml_inference_pipeline',
    default_args=default_args,
    description='AML account risk: batch inference with the registered models, then monitoring',
    schedule_interval='@daily',
    start_date=datetime(2022, 9, 1),
    end_date=datetime(2022, 9, 10),
    catchup=True,
    max_active_runs=1,
) as dag:

    # --- wait for the feature store and label store of the same day ---
    # reschedule: the sensor frees the executor between checks (we run one task at a time)
    wait_for_data_pipeline = ExternalTaskSensor(
        task_id='wait_for_data_pipeline',
        external_dag_id='aml_data_pipeline',
        external_task_id='data_pipeline_completed',
        allowed_states=['success'],
        failed_states=['failed', 'upstream_failed', 'skipped'],
        mode='reschedule',
        poke_interval=60,
        timeout=6 * 60 * 60,
    )


    # --- model inference (champion = production, challenger / baseline = shadow) ---
    model_inference = BashOperator(
        task_id='run_model_inference',
        bash_command=run_script + 'model_inference.py --snapshotdate "{{ macros.ds_add(ds, 1) }}" --topk 500',
    )

    wait_for_data_pipeline >> model_inference


    # --- model monitoring ---
    # big drift -> ERROR log "DRIFT ALERT" + drift_alerts.csv, a person decides whether to retrain
    model_monitor = BashOperator(
        task_id='run_model_monitor',
        bash_command=run_script + 'model_monitor.py --snapshotdate "{{ macros.ds_add(ds, 1) }}"',
    )

    model_inference >> model_monitor
