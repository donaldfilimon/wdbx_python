# wdbx/shard_manager.py
import time
import uuid
import threading
from typing import Dict, List
from wdbx.data_structures import ShardInfo
from config import SHARD_COUNT, NETWORK_OVERHEAD

class ShardManager:
    """
    Manages sharded storage across multiple simulated nodes.
    """
    def __init__(self, num_shards: int = SHARD_COUNT) -> None:
        self.num_shards = num_shards
        self.shards: Dict[int, ShardInfo] = {}
        self.shard_assignment: Dict[str, int] = {}
        self.lock = threading.RLock()
        self._initialize_shards()
    
    def _initialize_shards(self) -> None:
        for i in range(self.num_shards):
            host = "127.0.0.1"
            port = 9000 + i
            self.shards[i] = ShardInfo(
                shard_id=i,
                host=host,
                port=port
            )
    
    def get_shard_for_block(self, block_id: str) -> int:
        with self.lock:
            if block_id in self.shard_assignment:
                return self.shard_assignment[block_id]
            shard_id = min(self.shards.keys(), key=lambda x: self.shards[x].load)
            self.shard_assignment[block_id] = shard_id
            self.shards[shard_id].block_count += 1
            self.shards[shard_id].load = min(1.0, self.shards[shard_id].block_count / 1000)
            self.shards[shard_id].last_updated = time.time()
            return shard_id
    
    def calculate_latency(self, shard_id: int, retrieval_size: int) -> float:
        with self.lock:
            if shard_id not in self.shards:
                raise ValueError(f"Invalid shard ID: {shard_id}")
            return self.shards[shard_id].calculate_latency(retrieval_size)
    
    def get_optimal_shards(self, retrieval_size: int, count: int = 3) -> List[int]:
        with self.lock:
            latencies = [(shard_id, self.calculate_latency(shard_id, retrieval_size)) for shard_id in self.shards]
            latencies.sort(key=lambda x: x[1])
            return [shard_id for shard_id, _ in latencies[:count]]
    
    def update_shard_status(self, shard_id: int, status: str) -> None:
        with self.lock:
            if shard_id not in self.shards:
                raise ValueError(f"Invalid shard ID: {shard_id}")
            self.shards[shard_id].status = status
            self.shards[shard_id].last_updated = time.time()
    
    def get_shard_info(self, shard_id: int) -> ShardInfo:
        with self.lock:
            if shard_id not in self.shards:
                raise ValueError(f"Invalid shard ID: {shard_id}")
            return self.shards[shard_id]
    
    def get_all_shard_info(self) -> Dict[int, ShardInfo]:
        with self.lock:
            return self.shards.copy()
