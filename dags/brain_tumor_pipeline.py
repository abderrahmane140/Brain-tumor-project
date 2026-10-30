import logging
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

PY = "/opt/airflow/ml-venv/bin/python"
SCRIPT = "/opt/airflow/project/src/pipeline.py"

def on_failure(context):
    ti = context["task_instance"]
    logging.error("TASK FAILED: %s (run %s)", ti.task_id, context["run_id"])

default_args = {
    "owner": "brain-tumor-lab",
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
    "on_failure_callback": on_failure,
}

def task(name):
    return BashOperator(
        task_id=name,
        bash_command=f"{PY} {SCRIPT} {name}",
        execution_timeout=timedelta(hours=2),
    )

with DAG(
    dag_id="brain_tumor_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,          # manual trigger
    catchup=False,
    default_args=default_args,
    tags=["brain-tumor"],
) as dag:
    check  = task("check_datasets")
    valid  = task("validate_data")
    cls    = task("train_classification")
    det    = task("train_detection")
    evalu  = task("evaluate_log_mlflow")
    save   = task("save_artifacts")

    check >> valid >> [cls, det] >> evalu >> save