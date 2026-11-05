from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.external_task import ExternalTaskSensor

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# the project folder is mounted at /opt/airflow/aml, scripts use paths relative to it
run_script = "cd /opt/airflow/aml && python3 scripts/"
# dates for --snapshotdate, Airflow renders the {{ }} templates at run time
tomorrow = '"{{ macros.ds_add(ds, 1) }}"'

# every morning: score the new snapshot (ds + 1) with the registered models, then monitor
# champion -> alert list, challenger / shadow_1 / shadow_2 -> shadow scores for the monitor
# big drift only logs a "DRIFT ALERT", a person decides whether to run aml_training_pipeline
# a model only scores snapshots from its train date on, so catchup skips the days it trained on
with DAG(
    "aml_inference_pipeline",
    default_args=default_args,
    description="AML account risk: batch inference with the registered models, then monitoring",
    schedule_interval="@daily",
    start_date=datetime(2022, 9, 1),
    end_date=datetime(2022, 9, 10),
    catchup=True,
    max_active_runs=1,
) as dag:

    # reschedule mode frees the only executor slot between checks
    wait_for_data_pipeline = ExternalTaskSensor(
        task_id="wait_for_data_pipeline",
        external_dag_id="aml_data_pipeline",
        external_task_id="data_pipeline_completed",
        allowed_states=["success"],
        failed_states=["failed", "upstream_failed", "skipped"],
        mode="reschedule",
        poke_interval=60,
        timeout=6 * 60 * 60,
    )

    model_inference = BashOperator(
        task_id="run_model_inference",
        bash_command=run_script + "model_inference.py --snapshotdate " + tomorrow + " --topk 500",
    )

    model_monitor = BashOperator(
        task_id="run_model_monitor",
        bash_command=run_script + "model_monitor.py --snapshotdate " + tomorrow,
    )

    wait_for_data_pipeline >> model_inference >> model_monitor
