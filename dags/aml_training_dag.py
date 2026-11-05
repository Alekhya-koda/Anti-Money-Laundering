from datetime import datetime, timedelta

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# the project folder is mounted at /opt/airflow/aml, scripts use paths relative to it
run_script = "cd /opt/airflow/aml && python3 scripts/"
# dates for --snapshotdate, Airflow renders the {{ }} templates at run time
train_date = '"{{ params.snapshotdate }}"'

# not scheduled: a person runs it for the first models or after a drift alert, e.g.
#   airflow dags trigger aml_training_pipeline -c '{"snapshotdate": "2022-09-09"}'
# snapshotdate = train date, train snapshot = date - 4 days, valid snapshot = date - 2 days
# first run sets champion / challenger, a retrain sets shadow_1 / shadow_2 (README step 7)
with DAG(
    "aml_training_pipeline",
    default_args=default_args,
    description="AML account risk: train the 3 candidate models, set the MLflow aliases (manual)",
    schedule_interval=None,
    start_date=datetime(2022, 9, 1),
    catchup=False,
    max_active_runs=1,
    params={
        "snapshotdate": Param(
            "2022-09-09", type="string", description="model train date, YYYY-MM-DD"
        )
    },
) as dag:

    # fails the run early if the train / valid snapshots are not ready
    model_train_check = BashOperator(
        task_id="run_model_train_check",
        bash_command=run_script + "model_train_check.py --snapshotdate " + train_date,
    )

    # one task runs at a time, priority_weight keeps the order LR -> LightGBM -> LightGBM + graph
    # so the MLflow version numbers come out the same every time
    model_train_logistic_regression = BashOperator(
        task_id="run_model_train_logistic_regression",
        bash_command=run_script
        + "model_train_logistic_regression.py --snapshotdate "
        + train_date,
        priority_weight=3,
        weight_rule="absolute",
    )

    model_train_lightgbm = BashOperator(
        task_id="run_model_train_lightgbm",
        bash_command=run_script + "model_train_lightgbm.py --snapshotdate " + train_date,
        priority_weight=2,
        weight_rule="absolute",
    )

    model_train_lightgbm_graph = BashOperator(
        task_id="run_model_train_lightgbm_graph",
        bash_command=run_script + "model_train_lightgbm_graph.py --snapshotdate " + train_date,
        priority_weight=1,
        weight_rule="absolute",
    )

    model_select = BashOperator(
        task_id="run_model_select",
        bash_command=run_script + "model_select.py --snapshotdate " + train_date,
    )

    model_train_check >> model_train_logistic_regression >> model_select
    model_train_check >> model_train_lightgbm >> model_select
    model_train_check >> model_train_lightgbm_graph >> model_select
