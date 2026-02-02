"""
Модуль оптимизации моделей (квантизация, pruning)
"""

from pathlib import Path
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


class ModelOptimizer:
    """Класс для оптимизации ONNX моделей"""

    def __init__(self, onnx_path):
        self.onnx_path = onnx_path

    def quantize(self, output_path='model_quantized.onnx',
                 weight_type=QuantType.QUInt8):
        """Динамическая квантизация модели"""

        quantize_dynamic(
            self.onnx_path,
            output_path,
            weight_type=weight_type
        )

        # Сравнение размеров
        original_size = Path(self.onnx_path).stat().st_size / 1024 / 1024
        quantized_size = Path(output_path).stat().st_size / 1024 / 1024
        compression = (1 - quantized_size / original_size) * 100

        results = {
            'original_size_mb': original_size,
            'quantized_size_mb': quantized_size,
            'compression_percent': compression,
            'output_path': output_path
        }

        self.print_quantization_results(results)
        return results

    def print_quantization_results(self, results):
        """Печать результатов квантизации"""
        print(f"\n{'='*60}")
        print("КВАНТИЗАЦИЯ МОДЕЛИ")
        print(f"{'='*60}")
        print(f"Исходный размер: {results['original_size_mb']:.2f} MB")
        print(f"Квантизованный размер: {results['quantized_size_mb']:.2f} MB")
        print(f"Сжатие: {results['compression_percent']:.1f}%")
        print(f"✓ Сохранено: {results['output_path']}")
