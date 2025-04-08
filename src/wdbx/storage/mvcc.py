# wdbx/mvcc.py
import threading
import time
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.constants import logger


class MVCCTransaction:
    """
    Represents a transaction using multiversion concurrency control.

    MVCC allows for concurrent access to database records without locking
    by maintaining multiple versions of data.
    """

    def __init__(self, transaction_id: Optional[str] = None) -> None:
        self.transaction_id = transaction_id or str(uuid.uuid4())
        self.start_time = time.time()
        self.version = int(self.start_time * 1000)
        self.read_set: Set[str] = set()
        self.write_set: Set[str] = set()
        self.status = "active"  # One of: active, committed, aborted
        self.locks: Set[str] = set()
        self.end_time: Optional[float] = None

    def read(self, key: str) -> None:
        """Mark a key as read by this transaction"""
        self.read_set.add(key)

    def write(self, key: str) -> None:
        """Mark a key as written by this transaction"""
        self.write_set.add(key)

    def acquire_lock(self, key: str) -> bool:
        """
        Attempt to acquire a lock on the specified key.

        Returns:
            bool: True if lock was acquired, False otherwise
        """
        if self.status != "active":
            logger.warning(
                f"Transaction {
                    self.transaction_id} attempted to acquire lock while {
                    self.status}"
            )
            return False
        self.locks.add(key)
        return True

    def release_locks(self) -> None:
        """Release all locks held by this transaction"""
        self.locks.clear()

    def commit(self) -> None:
        """Mark transaction as committed and release all locks"""
        self.status = "committed"
        self.end_time = time.time()
        self.release_locks()
        logger.debug(
            f"Transaction {
                self.transaction_id} committed after {
                self.end_time -
                self.start_time:.3f}s"
        )

    def abort(self) -> None:
        """Mark transaction as aborted and release all locks"""
        self.status = "aborted"
        self.end_time = time.time()
        self.release_locks()
        logger.debug(
            f"Transaction {
                self.transaction_id} aborted after {
                self.end_time -
                self.start_time:.3f}s"
        )

    def is_active(self) -> bool:
        """Check if transaction is still active"""
        return self.status == "active"

    def is_committed(self) -> bool:
        """Check if transaction is committed"""
        return self.status == "committed"

    def is_aborted(self) -> bool:
        """Check if transaction is aborted"""
        return self.status == "aborted"

    def conflicts_with(self, other: "MVCCTransaction") -> bool:
        """
        Determine if this transaction conflicts with another transaction.

        Conflicts occur when:
        - Both transactions write to the same key (write-write conflict)
        - This transaction reads a key that another transaction writes (read-write conflict)

        Returns:
            bool: True if there is a conflict, False otherwise
        """
        # Write-write conflict
        if self.write_set.intersection(other.write_set):
            return True
        # Read-write conflict
        if self.read_set.intersection(other.write_set):
            return True
        # Write-read conflict (depending on isolation level, this might be relevant)
        if self.write_set.intersection(other.read_set):
            return True
        return False


class MVCCManager:
    """
    Manages MVCC transactions and versioned data.

    This class provides functionality for:
    - Starting, committing, and aborting transactions
    - Reading and writing versioned data
    - Cleaning up old versions of data
    """

    def __init__(self) -> None:
        self.transactions: Dict[str, MVCCTransaction] = {}
        # Key -> list of (version, value, transaction_id)
        self.versions: Dict[str, List[Tuple[int, Any, str]]] = defaultdict(list)
        self.locks: Dict[str, str] = {}  # key -> transaction_id
        self.lock = threading.RLock()
        self.last_cleanup_time = time.time()

    def start_transaction(self, transaction_id: Optional[str] = None) -> MVCCTransaction:
        """
        Start a new MVCC transaction.

        Args:
            transaction_id: Optional custom transaction ID

        Returns:
            MVCCTransaction: The newly created transaction
        """
        with self.lock:
            transaction = MVCCTransaction(transaction_id)
            self.transactions[transaction.transaction_id] = transaction
            logger.debug(f"Started transaction {transaction.transaction_id}")
            return transaction

    def get_transaction(self, transaction_id: str) -> Optional[MVCCTransaction]:
        """
        Get a transaction by ID.

        Args:
            transaction_id: The transaction ID to look up

        Returns:
            Optional[MVCCTransaction]: The transaction if found, None otherwise
        """
        return self.transactions.get(transaction_id)

    def read(self, transaction_id: str, key: str) -> Optional[Any]:
        """
        Read a value for a key using MVCC principles.

        Args:
            transaction_id: The ID of the transaction performing the read
            key: The key to read

        Returns:
            Optional[Any]: The value at the key visible to this transaction or None if not found

        Raises:
            ValueError: If the transaction is invalid or inactive
        """
        with self.lock:
            transaction = self.transactions.get(transaction_id)
            if not transaction:
                raise ValueError(f"Invalid transaction: {transaction_id}")
            if not transaction.is_active():
                raise ValueError(
                    f"Transaction {transaction_id} is not active (status: {
                        transaction.status})"
                )

            transaction.read(key)
            versions = self.versions.get(key, [])

            # Find versions that are visible to this transaction:
            # - Created by a committed transaction before this transaction started, or
            # - Created by this transaction itself
            valid_versions = [
                (v, val, tid)
                for v, val, tid in versions
                if (
                    v <= transaction.version
                    and (
                        tid == transaction_id
                        or self.transactions.get(tid, MVCCTransaction()).is_committed()
                    )
                )
                or tid == transaction_id
            ]

            if not valid_versions:
                logger.debug(
                    f"Transaction {transaction_id} read None for key {key} (no valid versions)"
                )
                return None

            # Return the most recent valid version
            valid_versions.sort(reverse=True, key=lambda x: x[0])
            logger.debug(
                f"Transaction {transaction_id} read value for key {key} (version: {
                    valid_versions[0][0]})"
            )
            return valid_versions[0][1]

    def write(self, transaction_id: str, key: str, value: Any) -> bool:
        """
        Write a value for a key using MVCC principles.

        Args:
            transaction_id: The ID of the transaction performing the write
            key: The key to write to
            value: The value to write

        Returns:
            bool: True if write was successful, False if it failed (e.g., due to lock contention)

        Raises:
            ValueError: If the transaction is invalid or inactive
        """
        with self.lock:
            transaction = self.transactions.get(transaction_id)
            if not transaction:
                raise ValueError(f"Invalid transaction: {transaction_id}")
            if not transaction.is_active():
                raise ValueError(
                    f"Transaction {transaction_id} is not active (status: {
                        transaction.status})"
                )

            # Check if another transaction holds the lock
            if key in self.locks and self.locks[key] != transaction_id:
                logger.debug(
                    f"Transaction {transaction_id} failed to write to {key}: locked by {
                        self.locks[key]}"
                )
                return False

            # Acquire lock for this key
            if not transaction.acquire_lock(key):
                logger.debug(f"Transaction {transaction_id} failed to acquire lock for key {key}")
                return False

            self.locks[key] = transaction_id
            transaction.write(key)

            # Add new version
            self.versions[key].append((transaction.version, value, transaction_id))
            logger.debug(
                f"Transaction {transaction_id} wrote value for key {key} (version: {
                    transaction.version})"
            )
            return True

    def commit(self, transaction_id: str) -> bool:
        """
        Commit a transaction, making its changes visible to other transactions.

        Args:
            transaction_id: The ID of the transaction to commit

        Returns:
            bool: True if commit was successful, False otherwise
        """
        with self.lock:
            transaction = self.transactions.get(transaction_id)
            if not transaction:
                logger.warning(f"Attempt to commit unknown transaction: {transaction_id}")
                return False

            if not transaction.is_active():
                logger.warning(
                    f"Attempt to commit {
                        transaction.status} transaction: {transaction_id}"
                )
                return False

            transaction.commit()

            # Release locks
            for key in list(transaction.locks):
                if key in self.locks and self.locks[key] == transaction_id:
                    del self.locks[key]

            logger.info(
                f"Transaction {transaction_id} committed: {len(transaction.write_set)} writes, {len(transaction.read_set)} reads"
            )

            # Automatic cleanup if needed
            self._auto_cleanup()
            return True

    def abort(self, transaction_id: str) -> bool:
        """
        Abort a transaction, discarding all of its changes.

        Args:
            transaction_id: The ID of the transaction to abort

        Returns:
            bool: True if abort was successful, False otherwise
        """
        with self.lock:
            transaction = self.transactions.get(transaction_id)
            if not transaction:
                logger.warning(f"Attempt to abort unknown transaction: {transaction_id}")
                return False

            if not transaction.is_active():
                logger.warning(
                    f"Attempt to abort {
                        transaction.status} transaction: {transaction_id}"
                )
                return False

            transaction.abort()

            # Remove versions created by this transaction
            for key in transaction.write_set:
                self.versions[key] = [
                    (v, val, tid) for v, val, tid in self.versions[key] if tid != transaction_id
                ]
                if not self.versions[key]:
                    del self.versions[key]

            # Release locks
            for key in list(transaction.locks):
                if key in self.locks and self.locks[key] == transaction_id:
                    del self.locks[key]

            logger.info(
                f"Transaction {transaction_id} aborted: {len(transaction.write_set)} writes rolled back"
            )
            return True

    def _auto_cleanup(self, cleanup_interval: float = 3600.0) -> None:
        """
        Automatically clean up old versions periodically.

        Args:
            cleanup_interval: Interval between cleanups in seconds
        """
        now = time.time()
        if now - self.last_cleanup_time > cleanup_interval:
            self.cleanup_old_versions()
            self.last_cleanup_time = now

    def get_active_transaction_count(self) -> int:
        """
        Get the number of active transactions.

        Returns:
            int: The number of active transactions
        """
        with self.lock:
            return sum(1 for tx in self.transactions.values() if tx.is_active())

    def cleanup_old_versions(self, max_age: float = 3600.0) -> int:
        """
        Clean up old versions of data that are no longer needed.

        Args:
            max_age: Maximum age in seconds for versions to keep

        Returns:
            int: Number of versions removed
        """
        with self.lock:
            current_time = time.time()
            min_version = int((current_time - max_age) * 1000)
            removed_count = 0

            # Clean up transactions table
            old_transactions = [
                tid
                for tid, tx in self.transactions.items()
                if not tx.is_active() and tx.end_time and (current_time - tx.end_time > max_age)
            ]
            for tid in old_transactions:
                del self.transactions[tid]

            # Clean up versions for each key
            for key in list(self.versions.keys()):
                versions = self.versions[key]
                if not versions:
                    del self.versions[key]
                    continue

                # Find the latest committed version
                committed_versions = [
                    v
                    for v in versions
                    if self.transactions.get(v[2], MVCCTransaction()).is_committed()
                ]
                if not committed_versions:
                    continue

                latest_committed = max(committed_versions, key=lambda x: x[0])

                # Keep versions that are:
                # 1. Newer than min_version, or
                # 2. The latest committed version, or
                # 3. Created by active transactions
                new_versions = [
                    v
                    for v in versions
                    if v[0] >= min_version
                    or v == latest_committed
                    or (v[2] in self.transactions and self.transactions[v[2]].is_active())
                ]

                removed_count += len(versions) - len(new_versions)

                if not new_versions:
                    del self.versions[key]
                else:
                    self.versions[key] = new_versions

            logger.debug(
                f"Cleaned up {removed_count} old versions, current keys: {len(self.versions)}"
            )
            return removed_count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the MVCC manager state.

        Returns:
            Dict[str, Any]: Statistics including transaction counts, version counts, etc.
        """
        with self.lock:
            active_txns = sum(1 for tx in self.transactions.values() if tx.is_active())
            committed_txns = sum(1 for tx in self.transactions.values() if tx.is_committed())
            aborted_txns = sum(1 for tx in self.transactions.values() if tx.is_aborted())
            total_versions = sum(len(versions) for versions in self.versions.values())

            return {
                "transactions": {
                    "active": active_txns,
                    "committed": committed_txns,
                    "aborted": aborted_txns,
                    "total": len(self.transactions),
                },
                "keys": len(self.versions),
                "versions": total_versions,
                "locks": len(self.locks),
                "last_cleanup": self.last_cleanup_time,
            }
