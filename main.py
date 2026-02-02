"""
Главный скрипт для обучения, конвертации и бенчмаркинга модели
"""

import argparse
import json
from pathlib import Path
import sys

from src.models.lstm_model import CreditDefaultLSTM
from src.data.preprocessing import DataPreprocessor
from src.training.trainer import ModelTrainer
from src.deployment.onnx_converter import ONNXConverter
from src.deployment.validator import ConversionValidator
from src.deployment.optimizer import ModelOptimizer
from src.benchmarking.performance import PerformanceBenchmark, LoadTester
from src.utils.metrics import ModelMetrics
from src.utils.config import ConfigManager


def setup_directories():
    """Создание необходимых директорий"""
    dirs = ['models', 'data', 'reports', 'logs']
    for d in dirs:
        Path(d).mkdir(exist_ok=True)


def train_pipeline(config):
    """Пайплайн обучения модели"""
    print("\n" + "="*70)
    print("ЭТАП 1: ОБУЧЕНИЕ МОДЕЛИ")
    print("="*70)

    # 1. Подготовка данных
    print("\n[1/4] Подготовка данных...")
    preprocessor = DataPreprocessor()
    data = preprocessor.full_pipeline(config['data_path'])
    preprocessor.save_scaler('models/scaler.pkl')

    print(f"  ✓ Train samples: {len(data['X_train'])}")
    print(f"  ✓ Test samples: {len(data['X_test'])}")
    print(f"  ✓ Features: {data['input_size']}")

    # 2. Создание модели
    print("\n[2/4] Создание LSTM модели...")
    model = CreditDefaultLSTM(
        input_size=data['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    )
    print(f"  ✓ Модель создана: {sum(p.numel() for p in model.parameters())} параметров")

    # 3. Обучение
    print("\n[3/4] Обучение модели...")
    trainer = ModelTrainer(model)
    trainer.train(
        data['X_train'], data['y_train'],
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        lr=config['training']['learning_rate']
    )
    trainer.save_model('models/model.pth')

    # 4. Оценка качества
    print("\n[4/4] Оценка качества модели...")
    metrics = ModelMetrics.evaluate_model(
        model, data['X_test'], data['y_test']
    )
    ModelMetrics.print_metrics(metrics)

    return model, data, metrics


def deployment_pipeline(model, data, config):
    """Пайплайн подготовки к развертыванию"""
    print("\n" + "="*70)
    print("ЭТАП 2: ПОДГОТОВКА К РАЗВЕРТЫВАНИЮ")
    print("="*70)

    # 1. Конвертация в ONNX
    print("\n[1/4] Конвертация в ONNX...")
    converter = ONNXConverter(model)
    onnx_path = converter.convert(
        input_size=data['input_size'],
        output_path='models/model.onnx'
    )
    converter.verify_model(onnx_path)

    # 2. Валидация конвертации
    print("\n[2/4] Валидация конвертации...")
    validator = ConversionValidator(model, onnx_path)
    validation_results = validator.validate(data['X_test'])

    # 3. Оптимизация (квантизация)
    print("\n[3/4] Оптимизация модели...")
    optimizer = ModelOptimizer(onnx_path)
    quantization_results = optimizer.quantize('models/model_quantized.onnx')

    # 4. Проверка метрик после квантизации
    print("\n[4/4] Проверка качества квантизованной модели...")
    quantized_metrics = ModelMetrics.evaluate_onnx(
        'models/model_quantized.onnx',
        data['X_test'],
        data['y_test']
    )
    ModelMetrics.print_metrics(quantized_metrics)

    return {
        'onnx_path': onnx_path,
        'quantized_path': 'models/model_quantized.onnx',
        'validation': validation_results,
        'quantization': quantization_results,
        'quantized_metrics': quantized_metrics
    }


def benchmark_pipeline(model, data, deployment_results):
    """Пайплайн бенчмаркинга"""
    print("\n" + "="*70)
    print("ЭТАП 3: БЕНЧМАРКИНГ И НАГРУЗОЧНОЕ ТЕСТИРОВАНИЕ")
    print("="*70)

    # 1. Сравнение производительности
    print("\n[1/4] Сравнение производительности PyTorch vs ONNX...")
    benchmark = PerformanceBenchmark(
        pytorch_model=model,
        onnx_path=deployment_results['onnx_path']
    )
    performance_results = benchmark.compare_performance(
        data['X_test'][:1000],
        num_runs=100
    )

    # 2. Нагрузочное тестирование
    print("\n[2/4] Нагрузочное тестирование...")
    load_tester = LoadTester(deployment_results['onnx_path'])
    batch_results = load_tester.test_batch_sizes(data['X_test'])

    # 3. Тестирование параллельных запросов
    print("\n[3/4] Тестирование параллельных запросов...")
    concurrent_results = load_tester.test_concurrent_requests(data['X_test'])

    # 4. Профилирование памяти
    print("\n[4/4] Профилирование памяти...")
    memory_results = load_tester.memory_profiling(data['X_test'])

    return {
        'performance': performance_results,
        'batch_testing': batch_results,
        'concurrent_testing': concurrent_results,
        'memory_profiling': memory_results
    }


def generate_report(training_metrics, deployment_results, benchmark_results, config):
    """Генерация итогового отчета"""
    print("\n" + "="*70)
    print("ГЕНЕРАЦИЯ ОТЧЕТА")
    print("="*70)

    report = {
        'model_info': {
            'architecture': 'LSTM',
            'input_size': config['model'].get('input_size', 'auto'),
            'hidden_size': config['model']['hidden_size'],
            'num_layers': config['model']['num_layers'],
            'dropout': config['model']['dropout']
        },
        'training': {
            'epochs': config['training']['epochs'],
            'batch_size': config['training']['batch_size'],
            'learning_rate': config['training']['learning_rate']
        },
        'metrics': {
            'original_model': training_metrics,
            'quantized_model': deployment_results['quantized_metrics']
        },
        'deployment': {
            'validation': deployment_results['validation'],
            'quantization': deployment_results['quantization']
        },
        'performance': benchmark_results
    }

    # Сохранение отчета
    report_path = 'reports/benchmark_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"✓ Отчет сохранен: {report_path}")

    # Генерация markdown отчета
    generate_markdown_report(report)

    return report


def generate_markdown_report(report):
    """Генерация отчета в формате Markdown"""
    md_content = f"""# ML Model Deployment Report

## Model Information
- **Architecture**: {report['model_info']['architecture']}
- **Hidden Size**: {report['model_info']['hidden_size']}
- **Number of Layers**: {report['model_info']['num_layers']}
- **Dropout**: {report['model_info']['dropout']}

## Training Configuration
- **Epochs**: {report['training']['epochs']}
- **Batch Size**: {report['training']['batch_size']}
- **Learning Rate**: {report['training']['learning_rate']}

## Model Performance Metrics

### Original Model
- **Accuracy**: {report['metrics']['original_model']['accuracy']:.4f}
- **Precision**: {report['metrics']['original_model']['precision']:.4f}
- **Recall**: {report['metrics']['original_model']['recall']:.4f}
- **F1-Score**: {report['metrics']['original_model']['f1']:.4f}
- **ROC-AUC**: {report['metrics']['original_model']['roc_auc']:.4f}

### Quantized Model
- **Accuracy**: {report['metrics']['quantized_model']['accuracy']:.4f}
- **Precision**: {report['metrics']['quantized_model']['precision']:.4f}
- **Recall**: {report['metrics']['quantized_model']['recall']:.4f}
- **F1-Score**: {report['metrics']['quantized_model']['f1']:.4f}
- **ROC-AUC**: {report['metrics']['quantized_model']['roc_auc']:.4f}

## Deployment Optimization

### Model Size
- **Original ONNX**: {report['deployment']['quantization']['original_size_mb']:.2f} MB
- **Quantized ONNX**: {report['deployment']['quantization']['quantized_size_mb']:.2f} MB
- **Compression**: {report['deployment']['quantization']['compression_percent']:.1f}%

### Inference Performance
- **PyTorch**: {report['performance']['performance']['pytorch']['mean_ms']:.2f} ms
- **ONNX**: {report['performance']['performance']['onnx']['mean_ms']:.2f} ms
- **Speedup**: {report['performance']['performance']['speedup']:.2f}x

## Recommendations for Production

### Optimal Configuration
Based on load testing results:
"""

    # Добавляем рекомендации по batch size
    batch_results = report['performance']['batch_testing']
    best_batch = max(batch_results, key=lambda x: x['throughput_samples_sec'])

    md_content += f"""
- **Recommended Batch Size**: {best_batch['batch_size']}
- **Expected Throughput**: {best_batch['throughput_samples_sec']:.1f} samples/sec
- **Latency**: {best_batch['latency_ms']:.2f} ms

### Infrastructure Requirements
- **CPU**: Recommended 4+ cores for production
- **Memory**: ~{report['performance']['memory_profiling'][-1]['memory_after_mb']:.0f} MB per instance
- **Concurrent Requests**: Tested up to {report['performance']['concurrent_testing'][-1]['concurrent_requests']} parallel requests
"""

    with open('reports/REPORT.md', 'w', encoding='utf-8') as f:
        f.write(md_content)

    print("✓ Markdown отчет сохранен: reports/REPORT.md")


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='ML Model Training and Deployment Pipeline')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--data', type=str, default='data/UCI_Credit_Card.csv',
                        help='Path to data file')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training and use existing model')

    args = parser.parse_args()

    # Создание директорий
    setup_directories()

    # Загрузка конфигурации
    config_manager = ConfigManager(args.config)
    if Path(args.config).exists():
        config = config_manager.load_config()
    else:
        config = config_manager.create_default_config()
        config_manager.save_config(config)
        print(f"✓ Создан конфигурационный файл: {args.config}")

    config['data_path'] = args.data

    try:
        # Этап 1: Обучение
        if not args.skip_training:
            model, data, training_metrics = train_pipeline(config)
        else:
            print(" Пропуск обучения, загрузка существующей модели...")
            raise NotImplementedError("Загрузка существующей модели еще не реализована")

        # Этап 2: Развертывание
        deployment_results = deployment_pipeline(model, data, config)

        # Этап 3: Бенчмаркинг
        benchmark_results = benchmark_pipeline(model, data, deployment_results)

        # Генерация отчета
        report = generate_report(training_metrics, deployment_results,
                                 benchmark_results, config)

        print("\n" + "="*70)
        print("✓ PIPELINE УСПЕШНО ЗАВЕРШЕН")
        print("="*70)
        print("\nСозданные файлы:")
        print("  - models/model.pth (PyTorch модель)")
        print("  - models/model.onnx (ONNX модель)")
        print("  - models/model_quantized.onnx (Квантизованная модель)")
        print("  - models/scaler.pkl (Scaler для препроцессинга)")
        print("  - reports/benchmark_report.json")
        print("  - reports/REPORT.md")
        print("\n" + "="*70)

    except Exception as e:
        print(f"\n✗ ОШИБКА: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
