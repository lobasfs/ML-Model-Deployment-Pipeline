"""
Airflow DAG для автоматического переобучения модели
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
import os


default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email': ['ml-team@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'model_retraining_pipeline',
    default_args=default_args,
    description='Automated model retraining pipeline',
    schedule_interval='0 2 * * 0',  # Every Sunday at 2 AM
    catchup=False,
    tags=['ml', 'retraining'],
)


def check_data_quality(**context):
    """Проверка качества новых данных"""
    import pandas as pd
    from great_expectations.dataset import PandasDataset

    # Загрузка новых данных
    data_path = context['params']['data_path']
    df = pd.read_csv(data_path)

    # Валидация с Great Expectations
    ge_df = PandasDataset(df)

    # Проверки
    results = []
    results.append(ge_df.expect_column_values_to_not_be_null('ID'))
    results.append(ge_df.expect_column_values_to_be_between('LIMIT_BAL', 0, 1000000))

    # Проверка на успешность
    all_passed = all(r.success for r in results)

    if not all_passed:
        raise ValueError("Data quality checks failed!")

    return True


def check_drift(**context):
    """Проверка дрифта данных"""
    from check_drift import DriftDetector
    import pandas as pd

    reference_data = pd.read_csv(context['params']['reference_data'])
    current_data = pd.read_csv(context['params']['current_data'])

    detector = DriftDetector(reference_data, current_data)
    detector.setup_column_mapping()

    drift_metrics = detector.generate_drift_report()
    drift_score = detector.calculate_drift_score(drift_metrics)

    # Push to XCom
    context['task_instance'].xcom_push(key='drift_score', value=drift_score)

    return drift_score > context['params']['drift_threshold']


def prepare_training_data(**context):
    """Подготовка данных для обучения"""
    import pandas as pd
    from src.data.preprocessing import DataPreprocessor

    # Загрузка всех доступных данных
    data_path = context['params']['data_path']
    df = pd.read_csv(data_path)

    # Препроцессинг
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.full_pipeline(data_path)

    # Сохранение для следующего таска
    output_path = '/tmp/processed_data.pkl'
    import joblib
    joblib.dump(processed_data, output_path)

    context['task_instance'].xcom_push(key='processed_data_path', value=output_path)


def train_model(**context):
    """Обучение новой модели"""
    import joblib
    from src.models.lstm_model import CreditDefaultLSTM
    from src.training.trainer import ModelTrainer

    # Загрузка подготовленных данных
    data_path = context['task_instance'].xcom_pull(
        task_ids='prepare_data',
        key='processed_data_path'
    )
    data = joblib.load(data_path)

    # Создание и обучение модели
    model = CreditDefaultLSTM(
        input_size=data['input_size'],
        hidden_size=64,
        num_layers=2
    )

    trainer = ModelTrainer(model)
    trained_model = trainer.train(
        data['X_train'],
        data['y_train'],
        epochs=20,
        verbose=True
    )

    # Сохранение модели
    output_path = '/tmp/new_model.pth'
    trainer.save_model(output_path)

    context['task_instance'].xcom_push(key='model_path', value=output_path)


def evaluate_model(**context):
    """Оценка новой модели"""
    import joblib
    import torch
    from src.models.lstm_model import CreditDefaultLSTM
    from src.utils.metrics import ModelMetrics

    # Загрузка модели и данных
    model_path = context['task_instance'].xcom_pull(
        task_ids='train_model',
        key='model_path'
    )
    data_path = context['task_instance'].xcom_pull(
        task_ids='prepare_data',
        key='processed_data_path'
    )

    # Загрузка
    checkpoint = torch.load(model_path)
    data = joblib.load(data_path)

    model = CreditDefaultLSTM(input_size=data['input_size'])
    model.load_state_dict(checkpoint['model_state_dict'])

    # Оценка
    metrics = ModelMetrics.evaluate_model(
        model,
        data['X_test'],
        data['y_test']
    )

    ModelMetrics.print_metrics(metrics)

    # Push metrics to XCom
    context['task_instance'].xcom_push(key='metrics', value=metrics)

    # Проверка порога качества
    if metrics['roc_auc'] < 0.75:
        raise ValueError(f"Model quality too low: ROC-AUC = {metrics['roc_auc']}")


def compare_with_production(**context):
    """Сравнение с production моделью"""
    import joblib

    new_metrics = context['task_instance'].xcom_pull(
        task_ids='evaluate_model',
        key='metrics'
    )

    # Загрузка метрик production модели
    prod_metrics_path = context['params']['prod_metrics_path']
    with open(prod_metrics_path, 'r') as f:
        import json
        prod_metrics = json.load(f)

    # Сравнение
    improvement = new_metrics['roc_auc'] - prod_metrics['roc_auc']

    print(f"Production ROC-AUC: {prod_metrics['roc_auc']:.4f}")
    print(f"New model ROC-AUC: {new_metrics['roc_auc']:.4f}")
    print(f"Improvement: {improvement:.4f}")

    context['task_instance'].xcom_push(key='improvement', value=improvement)

    # Новая модель должна быть лучше или сопоставима
    if improvement < -0.02:  # Допускаем падение до 2%
        raise ValueError("New model is significantly worse than production")

    return improvement > 0


def deploy_model(**context):
    """Развертывание новой модели"""
    from src.deployment.onnx_converter import ONNXConverter
    import torch
    from src.models.lstm_model import CreditDefaultLSTM
    import shutil

    # Загрузка обученной модели
    model_path = context['task_instance'].xcom_pull(
        task_ids='train_model',
        key='model_path'
    )

    checkpoint = torch.load(model_path)
    data_path = context['task_instance'].xcom_pull(
        task_ids='prepare_data',
        key='processed_data_path'
    )

    import joblib
    data = joblib.load(data_path)

    model = CreditDefaultLSTM(input_size=data['input_size'])
    model.load_state_dict(checkpoint['model_state_dict'])

    # Конвертация в ONNX
    converter = ONNXConverter(model)
    onnx_path = converter.convert(
        input_size=data['input_size'],
        output_path='/tmp/new_model.onnx'
    )

    # Копирование в production
    prod_model_path = context['params']['prod_model_path']

    # Backup старой модели
    backup_path = prod_model_path + '.backup'
    shutil.copy(prod_model_path, backup_path)

    # Развертывание новой
    shutil.copy(onnx_path, prod_model_path)

    print(f"✓ Model deployed to: {prod_model_path}")


# Tasks
t1 = PythonOperator(
    task_id='check_data_quality',
    python_callable=check_data_quality,
    params={
        'data_path': '/data/new_data.csv'
    },
    dag=dag,
)

t2 = PythonOperator(
    task_id='check_drift',
    python_callable=check_drift,
    params={
        'reference_data': '/data/reference_data.csv',
        'current_data': '/data/new_data.csv',
        'drift_threshold': 0.1
    },
    dag=dag,
)

t3 = PythonOperator(
    task_id='prepare_data',
    python_callable=prepare_training_data,
    params={
        'data_path': '/data/new_data.csv'
    },
    dag=dag,
)

t4 = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

t5 = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

t6 = PythonOperator(
    task_id='compare_with_production',
    python_callable=compare_with_production,
    params={
        'prod_metrics_path': '/models/production_metrics.json'
    },
    dag=dag,
)

t7 = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    params={
        'prod_model_path': '/models/model_quantized.onnx'
    },
    dag=dag,
)

t8 = SlackWebhookOperator(
    task_id='notify_success',
    http_conn_id='slack_webhook',
    message='✓ Model retraining completed successfully!',
    dag=dag,
)

# Dependencies
t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7 >> t8
