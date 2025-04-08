# wdbx/shard_manager.py
import hashlib
import os
import threading
import time
from typing import Dict, List, Optional

from ..core.constants import SHARD_COUNT, logger
from ..core.data_structures import ShardInfo

# Default values for shard management
NETWORK_OVERHEAD = 0.001  # seconds per operation
MAX_LOAD = 1000  # maximum blocks per shard
DEFAULT_REBALANCE_INTERVAL = 300  # seconds (5 minutes)


class ShardManager:
    """
    Manages the distribution of data across shards based on a consistent hashing scheme.

    Attributes:
        num_shards (int): The total number of shards.
        shards (Dict[int, ShardInfo]): Dictionary mapping shard IDs to ShardInfo objects.
        lock (threading.RLock): Lock for thread-safe operations.
        shard_assignment (Dict[str, int]): Maps block IDs to shard IDs.
        _last_rebalance (float): Timestamp of the last rebalance operation.
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
        self.lock = threading.RLock()
        self.shard_assignment: Dict[str, int] = {}
        self._last_rebalance = time.time()
        self._initialize_shards()
        logger.info(f"ShardManager initialized with {self.num_shards} shards.")

    def _initialize_shards(self) -> None:
        """Initialize the shard information dictionary."""
        with self.lock:
            # Placeholder: In a real system, this would involve discovering or
            # configuring shard hosts, ports, capacities, etc.
            for i in range(self.num_shards):
                # Example initialization - replace with actual configuration loading
                self.shards[i] = ShardInfo(
                    shard_id=str(i),
                    node_id=f"node_{i % 3}",  # Assign shards to 3 example nodes
                    item_count=0,
                    size_bytes=0,
                    status="active",
                    load=0.0,
                    block_count=0,
                    last_updated=time.time(),
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
        with self.lock:
            shard_id = self._get_shard_id(key)
            if shard_id not in self.shards:
                # This should ideally not happen if shards are initialized correctly
                logger.error(f"Calculated shard ID {shard_id} not found in shard map.")
                # Fallback or raise error
                raise KeyError(f"Shard ID {shard_id} not found.")
            return self.shards[shard_id]

    def get_all_shards(self) -> List[ShardInfo]:
        """Get information about all managed shards."""
        with self.lock:
            return list(self.shards.values())

    def add_shard(self, shard_info: ShardInfo) -> None:
        """Add or update information for a shard."""
        with self.lock:
            try:
                shard_id_int = int(shard_info.shard_id)
                if 0 <= shard_id_int < self.num_shards:
                    self.shards[shard_id_int] = shard_info
                    logger.info(f"Added/Updated shard info for shard ID {shard_id_int}")
                else:
                    logger.warning(f"Invalid shard ID {shard_id_int} for current configuration.")
            except ValueError:
                logger.error(
                    f"Cannot add shard: shard_id '{shard_info.shard_id}' is not a valid integer."
                )

    def remove_shard(self, shard_id: int) -> bool:
        """Remove a shard from management (e.g., due to failure)."""
        with self.lock:
            if shard_id in self.shards:
                old_status = self.shards[shard_id].status
                self.shards[shard_id].status = "removed"

                # If it was active, handle failover
                if old_status == "active":
                    self._handle_shard_failure(shard_id)

                logger.info(f"Removed shard ID {shard_id} from management.")
                return True

            logger.warning(f"Cannot remove shard: Shard ID {shard_id} not found.")
            return False

    def update_shard_stats(
        self, shard_id: int, item_count: Optional[int] = None, size_bytes: Optional[int] = None
    ) -> bool:
        """Update statistics for a specific shard."""
        with self.lock:
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
                    shard.last_updated = time.time()
                    logger.debug(f"Updated stats for shard ID {shard_id}")
                return True

            logger.warning(f"Cannot update stats: Shard ID {shard_id} not found.")
            return False

    def get_shard_for_block(self, block_id: str) -> int:
        """
        Determine which shard should store a block, using load balancing.

        Args:
            block_id: ID of the block to assign

        Returns:
            Shard ID where the block should be stored

        Raises:
            RuntimeError: If no active shards are available
        """
        with self.lock:
            if block_id in self.shard_assignment:
                return self.shard_assignment[block_id]

            # Find least loaded shard that is active
            available_shards = [
                (sid, sinfo) for sid, sinfo in self.shards.items() if sinfo.status == "active"
            ]

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
        """
        Calculate estimated latency for retrieving data from a shard.

        Args:
            shard_id: ID of the shard
            retrieval_size: Size of data to retrieve in bytes

        Returns:
            Estimated latency in seconds

        Raises:
            ValueError: If shard ID is invalid
        """
        with self.lock:
            if shard_id not in self.shards:
                raise ValueError(f"Invalid shard ID: {shard_id}")

            shard = self.shards[shard_id]
            if shard.status != "active":
                return float("inf")

            # Basic latency model based on load and data size
            base_latency = NETWORK_OVERHEAD * (1.0 + shard.load)
            size_factor = retrieval_size / 1024 / 1024 * 0.01  # 0.01s per MB
            return base_latency + size_factor

    def get_optimal_shards(self, retrieval_size: int, count: int = 3) -> List[int]:
        """
        Get the optimal shards for retrieving data, ordered by estimated latency.

        Args:
            retrieval_size: Size of data to retrieve in bytes
            count: Number of shards to return

        Returns:
            List of shard IDs ordered by estimated latency

        Raises:
            RuntimeError: If no active shards are available
        """
        with self.lock:
            active_shards = [
                (sid, self.calculate_latency(sid, retrieval_size))
                for sid, shard in self.shards.items()
                if shard.status == "active"
            ]

            if not active_shards:
                raise RuntimeError("No active shards available")

            active_shards.sort(key=lambda x: x[1])
            return [shard_id for shard_id, _ in active_shards[: min(count, len(active_shards))]]

    def update_shard_status(self, shard_id: int, status: str) -> None:
        """
        Update the status of a shard.

        Args:
            shard_id: ID of the shard
            status: New status ('active', 'inactive', 'maintenance', 'failed')

        Raises:
            ValueError: If shard ID is invalid
        """
        with self.lock:
            if shard_id not in self.shards:
                raise ValueError(f"Invalid shard ID: {shard_id}")

            old_status = self.shards[shard_id].status
            self.shards[shard_id].status = status
            self.shards[shard_id].last_updated = time.time()

            logger.info(f"Updated shard {shard_id} status from {old_status} to {status}")

            if status != "active" and old_status == "active":
                self._handle_shard_failure(shard_id)

    def get_shard_info(self, shard_id: int) -> ShardInfo:
        """
        Get information about a specific shard.

        Args:
            shard_id: ID of the shard

        Returns:
            Copy of the ShardInfo object

        Raises:
            ValueError: If shard ID is invalid
        """
        with self.lock:
            if shard_id not in self.shards:
                raise ValueError(f"Invalid shard ID: {shard_id}")
            # Return a copy to prevent external modification
            return (
                self.shards[shard_id].copy()
                if hasattr(self.shards[shard_id], "copy")
                else self.shards[shard_id]
            )

    def get_all_shard_info(self) -> Dict[int, ShardInfo]:
        """
        Get information about all shards.

        Returns:
            Dictionary mapping shard IDs to ShardInfo objects
        """
        with self.lock:
            # Return copies to prevent external modification
            return {k: v.copy() if hasattr(v, "copy") else v for k, v in self.shards.items()}

    def _should_rebalance(self) -> bool:
        """
        Determine if shards should be rebalanced based on load distribution.

        Returns:
            True if rebalance should be performed, False otherwise
        """
        now = time.time()
        # Don't rebalance more often than every DEFAULT_REBALANCE_INTERVAL seconds
        if now - self._last_rebalance < DEFAULT_REBALANCE_INTERVAL:
            return False

        with self.lock:
            loads = [s.load for s in self.shards.values() if s.status == "active"]
            if not loads:
                return False

            # Rebalance if load difference > 20%
            return max(loads) - min(loads) > 0.2

    def _rebalance_shards(self) -> None:
        """
        Rebalance blocks across shards to equalize load.
        """
        with self.lock:
            self._last_rebalance = time.time()
            logger.info("Starting shard rebalance operation")

            active_shards = [sid for sid, s in self.shards.items() if s.status == "active"]
            if len(active_shards) <= 1:
                logger.debug("Not enough active shards for rebalancing")
                return

            # Find blocks to move from most loaded to least loaded shards
            assignments = [
                (block_id, shard_id)
                for block_id, shard_id in self.shard_assignment.items()
                if shard_id in active_shards
            ]

            if not assignments:
                logger.debug("No blocks to rebalance")
                return

            # Sort by load, highest first
            assignments.sort(key=lambda x: self.shards[x[1]].load, reverse=True)

            # Calculate target load (average across active shards)
            total_load = sum(self.shards[sid].load for sid in active_shards)
            target_load = total_load / len(active_shards)

            logger.info(f"Rebalancing shards to target load of {target_load:.2f}")
            moves = 0

            for block_id, old_shard in assignments:
                # Stop if high-loaded shard is at or below target
                if self.shards[old_shard].load <= target_load:
                    break

                # Find least loaded shard
                new_shard = min(active_shards, key=lambda x: self.shards[x].load)
                if new_shard == old_shard:
                    continue

                # Move block to new shard
                self.shard_assignment[block_id] = new_shard
                self.shards[old_shard].block_count -= 1
                self.shards[new_shard].block_count += 1

                # Update load calculations
                self.shards[old_shard].load = min(
                    1.0, self.shards[old_shard].block_count / MAX_LOAD
                )
                self.shards[new_shard].load = min(
                    1.0, self.shards[new_shard].block_count / MAX_LOAD
                )

                moves += 1

                # Don't move too many blocks at once (prevent thrashing)
                if moves >= 100:  # limit moves per rebalance
                    break

            logger.info(f"Rebalance complete: moved {moves} blocks between shards")

    def _handle_shard_failure(self, failed_shard_id: int) -> None:
        """
        Handle failure of a shard by reassigning its blocks to other active shards.

        Args:
            failed_shard_id: ID of the failed shard

        Raises:
            RuntimeError: If no active shards are available for failover
        """
        with self.lock:
            logger.warning(f"Handling shard failure for shard ID {failed_shard_id}")

            # Reassign blocks from failed shard to active shards
            blocks_to_reassign = [
                block_id
                for block_id, shard_id in self.shard_assignment.items()
                if shard_id == failed_shard_id
            ]

            active_shards = [
                sid
                for sid, s in self.shards.items()
                if s.status == "active" and sid != failed_shard_id
            ]

            if not active_shards:
                raise RuntimeError("No active shards available for failover")

            logger.info(
                f"Reassigning {len(blocks_to_reassign)} blocks from failed shard {failed_shard_id}"
            )

            for block_id in blocks_to_reassign:
                # Find least loaded active shard
                new_shard = min(active_shards, key=lambda x: self.shards[x].load)

                # Reassign block
                self.shard_assignment[block_id] = new_shard
                self.shards[new_shard].block_count += 1
                self.shards[new_shard].load = min(
                    1.0, self.shards[new_shard].block_count / MAX_LOAD
                )

            logger.info(f"Failover complete for shard {failed_shard_id}")

    def export_config(self, filepath: str) -> bool:
        """
        Export shard configuration to a file.

        Args:
            filepath: Path to save configuration

        Returns:
            True if successful, False otherwise
        """
        try:
            import json

            # Normalize path for Windows
            filepath = os.path.normpath(filepath)

            with self.lock:
                # Create config dict with current state
                config = {
                    "num_shards": self.num_shards,
                    "shards": {
                        str(sid): {
                            "shard_id": s.shard_id,
                            "node_id": s.node_id,
                            "status": s.status,
                            "item_count": s.item_count,
                            "size_bytes": s.size_bytes,
                            "block_count": s.block_count,
                            "load": s.load,
                            "last_updated": s.last_updated,
                        }
                        for sid, s in self.shards.items()
                    },
                    "timestamp": time.time(),
                }

                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(filepath), exist_ok=True)

                # Write config to file
                with open(filepath, "w") as f:
                    json.dump(config, f, indent=2)

                logger.info(f"Shard configuration exported to {filepath}")
                return True

        except Exception as e:
            logger.error(f"Failed to export shard configuration: {str(e)}")
            return False

    @classmethod
    def import_config(cls, filepath: str) -> Optional["ShardManager"]:
        """
        Import shard configuration from a file.

        Args:
            filepath: Path to load configuration from

        Returns:
            ShardManager instance or None if import fails
        """
        try:
            import json

            # Normalize path for Windows
            filepath = os.path.normpath(filepath)

            with open(filepath) as f:
                config = json.load(f)

            # Create ShardManager with specified number of shards
            manager = cls(num_shards=config["num_shards"])

            # Clear default shards
            with manager.lock:
                manager.shards.clear()

                # Add shards from config
                for sid_str, shard_data in config["shards"].items():
                    sid = int(sid_str)
                    manager.shards[sid] = ShardInfo(
                        shard_id=shard_data["shard_id"],
                        node_id=shard_data["node_id"],
                        status=shard_data["status"],
                        item_count=shard_data["item_count"],
                        size_bytes=shard_data["size_bytes"],
                        block_count=shard_data.get("block_count", 0),
                        load=shard_data.get("load", 0.0),
                        last_updated=shard_data.get("last_updated", time.time()),
                    )

            logger.info(f"Shard configuration imported from {filepath}")
            return manager

        except Exception as e:
            logger.error(f"Failed to import shard configuration: {str(e)}")
            return None
