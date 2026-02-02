"""
Скрипт для проверки data drift используя Evidently AI
"""

import argparse
import pandas as pd
import json
from pathlib import Path
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.test_suite import TestSuite
from evidently.tests import *


class DriftDetector:
    """Класс для детектирования дрифта данных"""

    def __init__(self, reference_data: pd.DataFrame, current_data: pd.DataFrame):
        self.reference_data = reference_data
        self.current_data = current_data
        self.column_mapping = None

    def setup_column_mapping(self, target_col: str = 'target',
                             prediction_col: str = None):
        """Настройка маппинга колонок"""
        self.column_mapping = ColumnMapping(
            target=target_col,
            prediction=prediction_col,
            numerical_features=self.reference_data.select_dtypes(
                include=['int64', 'float64']
            ).columns.tolist(),
            categorical_features=self.reference_data.select_dtypes(
                include=['object', 'category']
            ).columns.tolist()
        )

    def generate_drift_report(self, output_path: str = 'reports/drift_report.html'):
        """Генерация отчета о дрифте"""

        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ])

        report.run(
            reference_data=self.reference_data,
            current_data=self.current_data,
            column_mapping=self.column_mapping
        )

        # Сохранение HTML отчета
        report.save_html(output_path)

        # Получение метрик в JSON
        drift_metrics = report.as_dict()

        return drift_metrics

    def run_drift_tests(self, drift_threshold: float = 0.1):
        """Запуск тестов на дрифт"""

        test_suite = TestSuite(tests=[
            TestNumberOfColumns(),
            TestNumberOfRows(),
            TestColumnsType(),
            TestNumberOfDriftedColumns(threshold=drift_threshold),
            TestShareOfDriftedColumns(threshold=0.3),
        ])

        test_suite.run(
            reference_data=self.reference_data,
            current_data=self.current_data,
            column_mapping=self.column_mapping
        )

        results = test_suite.as_dict()

        return results

    def calculate_drift_score(self, drift_metrics: dict) -> float:
        """Вычисление общего drift score"""

        # Извлекаем метрики дрифта по фичам
        dataset_drift = drift_metrics['metrics'][0]['result']
        drift_by_columns = dataset_drift.get('drift_by_columns', {})

        # Подсчет доли дрифтованных колонок
        total_columns = len(drift_by_columns)
        drifted_columns = sum(
            1 for col_data in drift_by_columns.values()
            if col_data.get('drift_detected', False)
        )

        drift_score = drifted_columns / total_columns if total_columns > 0 else 0

        return drift_score

    def detect_concept_drift(self, model, test_size: int = 1000):
        """Детектирование concept drift (изменение целевой переменной)"""
        import numpy as np
        from sklearn.metrics import accuracy_score

        # Предсказания на референсных данных
        ref_sample = self.reference_data.sample(min(test_size, len(self.reference_data)))
        X_ref = ref_sample.drop(columns=['target'])
        y_ref = ref_sample['target']

        # Предсказания на текущих данных
        cur_sample = self.current_data.sample(min(test_size, len(self.current_data)))
        X_cur = cur_sample.drop(columns=['target'])
        y_cur = cur_sample['target']

        # Сравнение точности
        ref_predictions = model.predict(X_ref)
        cur_predictions = model.predict(X_cur)

        ref_accuracy = accuracy_score(y_ref, ref_predictions)
        cur_accuracy = accuracy_score(y_cur, cur_predictions)

        accuracy_drop = ref_accuracy - cur_accuracy

        concept_drift_detected = accuracy_drop > 0.05  # 5% threshold

        return {
            'concept_drift_detected': concept_drift_detected,
            'reference_accuracy': ref_accuracy,
            'current_accuracy': cur_accuracy,
            'accuracy_drop': accuracy_drop
        }


def main():
    parser = argparse.ArgumentParser(description='Data Drift Detection')
    parser.add_argument('--reference', type=str, required=True,
                        help='Path to reference dataset')
    parser.add_argument('--current', type=str, required=True,
                        help='Path to current dataset')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Drift detection threshold')
    parser.add_argument('--output', type=str, default='reports/drift_report.html',
                        help='Output path for drift report')

    args = parser.parse_args()

    # Загрузка данных
    print("Loading datasets...")
    reference_data = pd.read_csv(args.reference)
    current_data = pd.read_csv(args.current)

    print(f"Reference data: {reference_data.shape}")
    print(f"Current data: {current_data.shape}")

    # Создание детектора
    detector = DriftDetector(reference_data, current_data)
    detector.setup_column_mapping(target_col='default.payment.next.month')

    # Генерация отчета
    print("\nGenerating drift report...")
    drift_metrics = detector.generate_drift_report(args.output)

    # Запуск тестов
    print("Running drift tests...")
    test_results = detector.run_drift_tests(args.threshold)

    # Вычисление drift score
    drift_score = detector.calculate_drift_score(drift_metrics)

    print(f"\n{'='*60}")
    print("DRIFT DETECTION RESULTS")
    print(f"{'='*60}")
    print(f"Drift Score: {drift_score:.3f}")
    print(f"Threshold: {args.threshold}")
    print(f"Drift Detected: {'YES' if drift_score > args.threshold else 'NO'}")
    print(f"\nReport saved to: {args.output}")

    # Сохранение результатов
    results = {
        'drift_score': drift_score,
        'drift_detected': drift_score > args.threshold,
        'threshold': args.threshold,
        'report_path': args.output
    }

    with open('reports/drift_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Return drift detected for CI/CD
    return drift_score > args.threshold


if __name__ == '__main__':
    import sys
    drift_detected = main()
    sys.exit(0 if not drift_detected else 1)
