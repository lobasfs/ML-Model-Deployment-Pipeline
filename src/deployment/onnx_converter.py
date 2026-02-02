"""
Модуль конвертации модели в ONNX
"""

import torch
import onnx


class ONNXConverter:
    """Класс для конвертации PyTorch моделей в ONNX"""

    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.model.cpu()

    def convert(self, input_size, seq_len=1, output_path='model.onnx',
                opset_version=11):
        """Конвертация модели в ONNX формат"""

        dummy_input = torch.randn(1, seq_len, input_size)

        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        print(f"✓ Модель сконвертирована в ONNX: {output_path}")
        return output_path

    def verify_model(self, onnx_path):
        """Проверка корректности ONNX модели"""
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX модель валидна")
        return True
