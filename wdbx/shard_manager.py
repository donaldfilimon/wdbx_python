# wdbx/shard_manager.py
import time
import uuid
import threading
from typing import Dict, List, Optional, Tuple
from wdbx.data_structures import ShardInfo
from config import SHARD_COUNT, NETWORK_OVERHEAD, MAX_LOAD

class ShardManager:
    """
    Manages sharded storage across multiple simulated nodes.
    Handles shard allocation, load balancing, and status tracking.
    """
    def __init__(self, num_shards: int = SHARD_COUNT) -> None:
        self.num_shards = num_shards
        self.shards: Dict[int, ShardInfo] = {}
        self.shard_assignment: Dict[str, int] = {}
        self.lock = threading.RLock()
        self._initialize_shards()
        self._last_rebalance = time.time()

    def _initialize_shards(self) -> None:
        for i in range(self.num_shards):
            host = "127.0.0.1"
            port = 9000 + i
            self.shards[i] = ShardInfo(
                shard_id=i,
                host=host,
                port=port,
                status="active"
            )

    def get_shard_for_block(self, block_id: str) -> int:
        with self.lock:
            if block_id in self.shard_assignment:
                return self.shard_assignment[block_id]

            # Find least loaded shard that is active
            available_shards = [s for s in self.shards.items() if s[1].status == "active"]
            if not available_shards:
                raise RuntimeError("No active shards available")

            shard_id = min(available_shards, key=lambda x: x[1].load)[0]

            self.shard_assignment[block_id] = shard_id
            self.shards[shard_id].block_count += 1
            self.shards[shard_id].load = min(1.0, self.shards[shard_id].block_count / MAX_LOAD)
            self.shards[shard_id].last_updated = time.time()

            # Trigger rebalance if needed
            if self._should_rebalance():
                self._rebalance_shards()

            return shard_id

    def calculate_latency(self, shard_id: int, retrieval_size: int) -> float:
        with self.lock:
            if shard_id not in self.shards:
                raise ValueError(f"Invalid shard ID: {shard_id}")
            shard = self.shards[shard_id]
            if shard.status != "active":
                return float('inf')
            return shard.calculate_latency(retrieval_size)

    def get_optimal_shards(self, retrieval_size: int, count: int = 3) -> List[int]:
        with self.lock:
            active_shards = [(sid, self.calculate_latency(sid, retrieval_size)) 
                           for sid, shard in self.shards.items() 
                           if shard.status == "active"]
            if not active_shards:
                raise RuntimeError("No active shards available")
            active_shards.sort(key=lambda x: x[1])
            return [shard_id for shard_id, _ in active_shards[:min(count, len(active_shards))]]

    def update_shard_status(self, shard_id: int, status: str) -> None:
        with self.lock:
            if shard_id not in self.shards:
                raise ValueError(f"Invalid shard ID: {shard_id}")
            self.shards[shard_id].status = status
            self.shards[shard_id].last_updated = time.time()
            if status != "active":
                self._handle_shard_failure(shard_id)

    def get_shard_info(self, shard_id: int) -> ShardInfo:
        with self.lock:
            if shard_id not in self.shards:
                raise ValueError(f"Invalid shard ID: {shard_id}")
            return self.shards[shard_id].copy()

    def get_all_shard_info(self) -> Dict[int, ShardInfo]:
        with self.lock:
            return {k: v.copy() for k, v in self.shards.items()}

    def _should_rebalance(self) -> bool:
        now = time.time()
        if now - self._last_rebalance < 300:  # Don't rebalance more often than every 5 minutes
            return False
        loads = [s.load for s in self.shards.values() if s.status == "active"]
        if not loads:
            return False
        return max(loads) - min(loads) > 0.2  # Rebalance if load difference > 20%

    def _rebalance_shards(self) -> None:
        self._last_rebalance = time.time()
        active_shards = [sid for sid, s in self.shards.items() if s.status == "active"]
        if len(active_shards) <= 1:
            return

        # Find blocks to move from most loaded to least loaded shards
        assignments = [(block_id, shard_id) for block_id, shard_id in self.shard_assignment.items()
                      if shard_id in active_shards]
        if not assignments:
            return

        assignments.sort(key=lambda x: self.shards[x[1]].load, reverse=True)
        target_load = sum(self.shards[sid].load for sid in active_shards) / len(active_shards)

        for block_id, old_shard in assignments:
            if self.shards[old_shard].load <= target_load:
                break

            new_shard = min(active_shards, key=lambda x: self.shards[x].load)
            if new_shard == old_shard:
                continue

            # Move block to new shard
            self.shard_assignment[block_id] = new_shard
            self.shards[old_shard].block_count -= 1
            self.shards[new_shard].block_count += 1
            self.shards[old_shard].load = min(1.0, self.shards[old_shard].block_count / MAX_LOAD)
            self.shards[new_shard].load = min(1.0, self.shards[new_shard].block_count / MAX_LOAD)

    def _handle_shard_failure(self, failed_shard_id: int) -> None:
        # Reassign blocks from failed shard to active shards
        blocks_to_reassign = [block_id for block_id, shard_id in self.shard_assignment.items()
                            if shard_id == failed_shard_id]

        active_shards = [sid for sid, s in self.shards.items() 
                        if s.status == "active" and sid != failed_shard_id]

        if not active_shards:
            raise RuntimeError("No active shards available for failover")

        for block_id in blocks_to_reassign:
            new_shard = min(active_shards, key=lambda x: self.shards[x].load)
            self.shard_assignment[block_id] = new_shard
            self.shards[new_shard].block_count += 1
            self.shards[new_shard].load = min(1.0, self.shards[new_shard].block_count / MAX_LOAD)
