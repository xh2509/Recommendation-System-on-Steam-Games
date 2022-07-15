from datetime import datetime, timedelta
from textwrap import dedent
import time

from airflow import DAG

from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'xiaohang',
    'depends_on_past': False,
    'email': ['xh2509@columbia.edu'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(seconds=30),
}


count = 0

with DAG(
    'Project',
    default_args=default_args,
    description='Project DAG',
    schedule_interval='0 2 * * *',
    start_date=datetime(2021, 12, 5),
    catchup=False,
    tags=['example'],
) as dag:

    t1 = BashOperator(
        task_id='run data collection file',
        bash_command='python /home/xh2509/data.py',
    )

    t2 = BashOperator(
        task_id='run modeling file',
        bash_command='python /home/xh2509/model.py',
    )

    t1 >> t2
