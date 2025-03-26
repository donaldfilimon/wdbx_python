# wdbx/performance.py
import time
from typing import Dict, Any, List, Tuple
from wdbx.constants import logger

class PerformanceAnalyzer:
    """
    Analyzes and reports on system performance.
    """
    def __init__(self, wdbx: Any) -> None:
        self.wdbx = wdbx
        self.latency_samples: List[Tuple[str, float]] = []
        self.throughput_samples: List[Tuple[str, float]] = []
        self.start_time = time.time()
    
    def measure_latency(self, operation: str, *args, **kwargs) -> float:
        start = time.time()
        if hasattr(self.wdbx, operation):
            getattr(self.wdbx, operation)(*args, **kwargs)
        else:
            raise ValueError(f"Unknown operation: {operation}")
        latency = time.time() - start
        self.latency_samples.append((operation, latency))
        return latency
    
    def measure_throughput(self, operation: str, count: int, *args, **kwargs) -> float:
        start = time.time()
        for _ in range(count):
            if hasattr(self.wdbx, operation):
                getattr(self.wdbx, operation)(*args, **kwargs)
            else:
                raise ValueError(f"Unknown operation: {operation}")
        duration = time.time() - start
        throughput = count / duration if duration > 0 else 0
        self.throughput_samples.append((operation, throughput))
        return throughput
    
    def calculate_system_latency(self) -> float:
        L_api = 0.01
        L_model = 0.1
        L_db = 0.03
        L_moderation = 0.02
        return L_api + L_model + L_db + L_moderation
    
    def calculate_throughput(self, num_requests: int) -> float:
        L_total = self.calculate_system_latency()
        return num_requests / L_total if L_total > 0 else 0
    
    def calculate_scaling_throughput(self, base_throughput: float, base_gpus: int, scaled_gpus: int) -> float:
        return base_throughput * (scaled_gpus / base_gpus)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        latency_by_op = {}
        throughput_by_op = {}
        for op, lat in self.latency_samples:
            latency_by_op.setdefault(op, []).append(lat)
        for op, thr in self.throughput_samples:
            throughput_by_op.setdefault(op, []).append(thr)
        avg_latency = {op: sum(samples)/len(samples) for op, samples in latency_by_op.items()}
        avg_throughput = {op: sum(samples)/len(samples) for op, samples in throughput_by_op.items()}
        return {
            "avg_latency": avg_latency,
            "avg_throughput": avg_throughput,
            "system_latency": self.calculate_system_latency(),
            "system_throughput": self.calculate_throughput(10),
            "uptime": time.time() - self.start_time
        }
