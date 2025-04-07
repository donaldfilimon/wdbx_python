# wdbx/shard_manager.py
import hashlib
import time
from typing import Dict, List, Optional

from ..core.constants import SHARD_COUNT, logger
from ..core.data_structures import ShardInfo

# Default values for shard management
NETWORK_OVERHEAD = 0.001  # seconds per operation
MAX_LOAD = 1000  # maximum blocks per shard


class ShardManager:
    """
    Manages the distribution of data across shards based on a consistent hashing scheme.

    Attributes:
        num_shards (int): The total number of shards.
        shards (Dict[int, ShardInfo]): Dictionary mapping shard IDs to ShardInfo objects.
    """

    def __init__(self, num_shards: int = SHARD_COUNT):
        """Initialize the ShardManager.

        Args:
            num_shards (int): The number of shards to manage.
        """
        if num_shards <= 0:
            raise ValueError("Number of shards must be positive")
        self.num_shards = num_shards
        self.shards: Dict[int, ShardInfo] = {}
        self._initialize_shards()
        logger.info(f"ShardManager initialized with {self.num_shards} shards.")

    def _initialize_shards(self) -> None:
        """Initialize the shard information dictionary."""
        # Placeholder: In a real system, this would involve discovering or
        # configuring shard hosts, ports, capacities, etc.
        for i in range(self.num_shards):
            # Example initialization - replace with actual configuration loading
            self.shards[i] = ShardInfo(
                shard_id=str(i),
                node_id=f"node_{i % 3}",  # Assign shards to 3 example nodes
                item_count=0,
                size_bytes=0
            )

    def _get_shard_id(self, key: str) -> int:
        """Determine the shard ID for a given key using consistent hashing."""
        # Use SHA-1 for hashing (consider SHA-256 or others for production)
        hash_bytes = hashlib.sha1(key.encode("utf-8")).digest()
        # Convert the first 4 bytes of the hash to an integer
        hash_int = int.from_bytes(hash_bytes[:4], "big")
        return hash_int % self.num_shards

    def get_shard_for_key(self, key: str) -> ShardInfo:
        """Get the ShardInfo object responsible for a given key."""
        shard_id = self._get_shard_id(key)
        if shard_id not in self.shards:
            # This should ideally not happen if shards are initialized correctly
            logger.error(f"Calculated shard ID {shard_id} not found in shard map.")
            # Fallback or raise error
            raise KeyError(f"Shard ID {shard_id} not found.")
        return self.shards[shard_id]

    def get_all_shards(self) -> List[ShardInfo]:
        """Get information about all managed shards."""
        return list(self.shards.values())

    def add_shard(self, shard_info: ShardInfo) -> None:
        """Add or update information for a shard."""
        try:
            shard_id_int = int(shard_info.shard_id)
            if 0 <= shard_id_int < self.num_shards:
                self.shards[shard_id_int] = shard_info
                logger.info(f"Added/Updated shard info for shard ID {shard_id_int}")
            else:
                logger.warning(f"Invalid shard ID {shard_id_int} for current configuration.")
        except ValueError:
            logger.error(f"Cannot add shard: shard_id '{shard_info.shard_id}' is not a valid integer.")

    def remove_shard(self, shard_id: int) -> bool:
        """Remove a shard from management (e.g., due to failure)."""
        if shard_id in self.shards:
            del self.shards[shard_id]
            logger.info(f"Removed shard ID {shard_id} from management.")
            # Note: Data redistribution logic would be needed here in a real system.
            return True
        logger.warning(f"Cannot remove shard: Shard ID {shard_id} not found.")
        return False

    def update_shard_stats(self, shard_id: int, item_count: Optional[int] = None, size_bytes: Optional[int] = None) -> bool:
        """Update statistics for a specific shard."""
        if shard_id in self.shards:
            shard = self.shards[shard_id]
            updated = False
            if item_count is not None:
                shard.item_count = item_count
                updated = True
            if size_bytes is not None:
                shard.size_bytes = size_bytes
                updated = True
            if updated:
                # shard.last_updated = time.time() # Assuming ShardInfo has this
                logger.debug(f"Updated stats for shard ID {shard_id}")
            return True
        logger.warning(f"Cannot update stats: Shard ID {shard_id} not found.")
        return False

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
                return float("inf")
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
