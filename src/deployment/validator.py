"""
Модуль валидации конвертации
"""

import torch
import numpy as np
import onnxruntime as ort


class ConversionValidator:
    """Класс для валидации конвертации PyTorch -> ONNX"""

    def __init__(self, pytorch_model, onnx_path):
        self.pytorch_model = pytorch_model
        self.pytorch_model.eval()
        self.pytorch_model.cpu()
        self.ort_session = ort.InferenceSession(onnx_path)

    def validate(self, test_data, tolerance=1e-5, num_samples=100):
        """Валидация корректности конвертации"""

        test_input = torch.FloatTensor(test_data[:num_samples])

        # PyTorch инференс
        with torch.no_grad():
            pytorch_output = self.pytorch_model(test_input).numpy()

        # ONNX инференс
        ort_inputs = {self.ort_session.get_inputs()[0].name: test_input.numpy()}
        onnx_output = self.ort_session.run(None, ort_inputs)[0]

        # Вычисление метрик
        max_diff = np.abs(pytorch_output - onnx_output).max()
        mean_diff = np.abs(pytorch_output - onnx_output).mean()

        results = {
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'tolerance': tolerance,
            'is_valid': max_diff < tolerance
        }

        self.print_results(results)
        return results

    def print_results(self, results):
        """Печать результатов валидации"""
        print(f"\n{'='*60}")
        print("ВАЛИДАЦИЯ КОНВЕРТАЦИИ")
        print(f"{'='*60}")
        print(f"Максимальная разница: {results['max_diff']:.2e}")
        print(f"Средняя разница: {results['mean_diff']:.2e}")
        print(f"Tolerance: {results['tolerance']:.2e}")

        if results['is_valid']:
            print("✓ Конвертация корректна!")
        else:
            print("✗ Конвертация имеет значительные отличия")
