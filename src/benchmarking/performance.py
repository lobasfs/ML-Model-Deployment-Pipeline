"""
Модуль для бенчмаркинга и нагрузочного тестирования
"""

import time
import numpy as np
import torch
import onnxruntime as ort
from typing import List, Dict


class PerformanceBenchmark:
    """Класс для бенчмаркинга производительности моделей"""

    def __init__(self, pytorch_model=None, onnx_path=None):
        self.pytorch_model = pytorch_model
        self.onnx_session = None

        if pytorch_model:
            self.pytorch_model.eval()
            self.pytorch_model.cpu()

        if onnx_path:
            self.onnx_session = ort.InferenceSession(onnx_path)

    def benchmark_pytorch(self, test_data, num_runs=100):
        """Бенчмарк PyTorch модели"""
        if self.pytorch_model is None:
            raise ValueError("PyTorch модель не загружена")

        test_input = torch.FloatTensor(test_data)
        times = []

        for _ in range(num_runs):
            start = time.time()
            with torch.no_grad():
                _ = self.pytorch_model(test_input)
            times.append(time.time() - start)

        return {
            'mean_ms': np.mean(times) * 1000,
            'std_ms': np.std(times) * 1000,
            'min_ms': np.min(times) * 1000,
            'max_ms': np.max(times) * 1000
        }

    def benchmark_onnx(self, test_data, num_runs=100):
        """Бенчмарк ONNX модели"""
        if self.onnx_session is None:
            raise ValueError("ONNX модель не загружена")

        input_name = self.onnx_session.get_inputs()[0].name
        test_input = {input_name: test_data.astype(np.float32)}
        times = []

        for _ in range(num_runs):
            start = time.time()
            _ = self.onnx_session.run(None, test_input)
            times.append(time.time() - start)

        return {
            'mean_ms': np.mean(times) * 1000,
            'std_ms': np.std(times) * 1000,
            'min_ms': np.min(times) * 1000,
            'max_ms': np.max(times) * 1000
        }

    def compare_performance(self, test_data, num_runs=100):
        """Сравнение производительности PyTorch vs ONNX"""

        pytorch_results = self.benchmark_pytorch(test_data, num_runs)
        onnx_results = self.benchmark_onnx(test_data, num_runs)

        speedup = pytorch_results['mean_ms'] / onnx_results['mean_ms']

        results = {
            'pytorch': pytorch_results,
            'onnx': onnx_results,
            'speedup': speedup,
            'samples': len(test_data)
        }

        self.print_comparison(results)
        return results

    def print_comparison(self, results):
        """Печать результатов сравнения"""
        print(f"\n{'='*60}")
        print("BENCHMARK ИНФЕРЕНСА (CPU)")
        print(f"{'='*60}")
        print(f"Образцов: {results['samples']}")
        print(f"\nPyTorch:")
        print(f"  Mean: {results['pytorch']['mean_ms']:.2f} ms")
        print(f"  Std:  {results['pytorch']['std_ms']:.2f} ms")
        print(f"\nONNX:")
        print(f"  Mean: {results['onnx']['mean_ms']:.2f} ms")
        print(f"  Std:  {results['onnx']['std_ms']:.2f} ms")
        print(f"\nУскорение: {results['speedup']:.2f}x")


class LoadTester:
    """Класс для нагрузочного тестирования"""

    def __init__(self, onnx_path):
        self.onnx_session = ort.InferenceSession(onnx_path)
        self.input_name = self.onnx_session.get_inputs()[0].name

    def test_batch_sizes(self, test_data,
                         batch_sizes: List[int] = [1, 8, 32, 128, 512],
                         num_runs: int = 50):
        """Тестирование на разных размерах батчей"""

        results = []

        print(f"\n{'='*60}")
        print("НАГРУЗОЧНОЕ ТЕСТИРОВАНИЕ")
        print(f"{'='*60}")
        print(f"{'Batch':>8} | {'Latency':>10} | {'Throughput':>18}")
        print(f"{'-'*8}-+-{'-'*10}-+-{'-'*18}")

        for batch_size in batch_sizes:
            batch_data = test_data[:batch_size]
            test_input = {self.input_name: batch_data.astype(np.float32)}

            times = []
            for _ in range(num_runs):
                start = time.time()
                _ = self.onnx_session.run(None, test_input)
                times.append(time.time() - start)

            mean_time = np.mean(times) * 1000
            throughput = batch_size / (mean_time / 1000)

            result = {
                'batch_size': batch_size,
                'latency_ms': mean_time,
                'latency_std_ms': np.std(times) * 1000,
                'throughput_samples_sec': throughput
            }
            results.append(result)

            print(f"{batch_size:8d} | {mean_time:8.2f} ms | "
                  f"{throughput:10.1f} samp/sec")

        return results

    def test_concurrent_requests(self, test_data,
                                 num_concurrent: List[int] = [1, 5, 10, 20],
                                 duration_sec: int = 10):
        """Тестирование с параллельными запросами"""
        import threading

        results = []

        print(f"\n{'='*60}")
        print("ТЕСТИРОВАНИЕ ПАРАЛЛЕЛЬНЫХ ЗАПРОСОВ")
        print(f"{'='*60}")

        for num_threads in num_concurrent:
            request_count = [0]
            stop_flag = [False]

            def worker():
                test_input = {self.input_name: test_data[:32].astype(np.float32)}
                while not stop_flag[0]:
                    _ = self.onnx_session.run(None, test_input)
                    request_count[0] += 1

            threads = [threading.Thread(target=worker)
                       for _ in range(num_threads)]

            start = time.time()
            for t in threads:
                t.start()

            time.sleep(duration_sec)
            stop_flag[0] = True

            for t in threads:
                t.join()

            elapsed = time.time() - start
            rps = request_count[0] / elapsed

            result = {
                'concurrent_requests': num_threads,
                'total_requests': request_count[0],
                'duration_sec': elapsed,
                'requests_per_sec': rps
            }
            results.append(result)

            print(f"Threads: {num_threads:2d} | "
                  f"Requests: {request_count[0]:5d} | "
                  f"RPS: {rps:.1f}")

        return results

    def memory_profiling(self, test_data, batch_sizes: List[int] = [1, 32, 128]):
        """Профилирование использования памяти"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        results = []

        print(f"\n{'='*60}")
        print("ПРОФИЛИРОВАНИЕ ПАМЯТИ")
        print(f"{'='*60}")

        for batch_size in batch_sizes:
            batch_data = test_data[:batch_size]
            test_input = {self.input_name: batch_data.astype(np.float32)}

            # Измерение до
            mem_before = process.memory_info().rss / 1024 / 1024

            # Инференс
            for _ in range(10):
                _ = self.onnx_session.run(None, test_input)

            # Измерение после
            mem_after = process.memory_info().rss / 1024 / 1024
            mem_increase = mem_after - mem_before

            result = {
                'batch_size': batch_size,
                'memory_before_mb': mem_before,
                'memory_after_mb': mem_after,
                'memory_increase_mb': mem_increase
            }
            results.append(result)

            print(f"Batch {batch_size:3d}: {mem_before:.1f} MB → "
                  f"{mem_after:.1f} MB (Δ{mem_increase:+.1f} MB)")

        return results
