from airflow import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# the project folder is mounted at /opt/airflow/aml, scripts use paths relative to it
run_script = 'cd /opt/airflow/aml && python3 scripts/'

# training is not scheduled: a person runs this DAG when it is needed, e.g. for the first models,
# or after aml_inference_pipeline logged a DRIFT ALERT and the person decided a new model is worth it.
# Trigger it in the Airflow UI (Trigger DAG w/ config) or from the command line:
#   airflow dags trigger aml_training_pipeline -c '{"snapshotdate": "2022-09-09"}'
# snapshotdate = model train date: train snapshot = snapshotdate - 4 days, valid snapshot = snapshotdate - 2 days
# The first run sets champion / challenger / baseline. Later runs only set challenger and baseline,
# the new model runs in shadow until a person promotes it (MLflow alias "champion" + POST /reload on the API)
with DAG(
    'aml_training_pipeline',
    default_args=default_args,
    description='AML account risk: train the 3 candidate models, pick champion / challenger / baseline (manual)',
    schedule_interval=None,
    start_date=datetime(2022, 9, 1),
    catchup=False,
    max_active_runs=1,
    params={"snapshotdate": Param("2022-09-09", type="string", description="model train date, YYYY-MM-DD")},
) as dag:

    # stops the run with an error if the train / valid snapshots have no features or labels yet
    model_train_check = BashOperator(
        task_id='run_model_train_check',
        bash_command=run_script + 'model_train_check.py --snapshotdate "{{ params.snapshotdate }}"',
    )

    # one task per candidate model. The executor runs one task at a time, priority_weight keeps the
    # order LR -> LightGBM -> LightGBM + graph so the MLflow version numbers come out the same every time
    model_train_logistic_regression = BashOperator(
        task_id='run_model_train_logistic_regression',
        bash_command=run_script + 'model_train_logistic_regression.py --snapshotdate "{{ params.snapshotdate }}"',
        priority_weight=3,
        weight_rule='absolute',
    )

    model_train_lightgbm = BashOperator(
        task_id='run_model_train_lightgbm',
        bash_command=run_script + 'model_train_lightgbm.py --snapshotdate "{{ params.snapshotdate }}"',
        priority_weight=2,
        weight_rule='absolute',
    )

    model_train_lightgbm_graph = BashOperator(
        task_id='run_model_train_lightgbm_graph',
        bash_command=run_script + 'model_train_lightgbm_graph.py --snapshotdate "{{ params.snapshotdate }}"',
        priority_weight=1,
        weight_rule='absolute',
    )

    # compares the 3 new models and moves the champion / challenger / baseline aliases in MLflow
    model_select = BashOperator(
        task_id='run_model_select',
        bash_command=run_script + 'model_select.py --snapshotdate "{{ params.snapshotdate }}"',
    )

    model_train_check >> model_train_logistic_regression >> model_select
    model_train_check >> model_train_lightgbm >> model_select
    model_train_check >> model_train_lightgbm_graph >> model_select
