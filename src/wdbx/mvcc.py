"""
Multiversion Concurrency Control (MVCC) for WDBX.

This module provides transaction management with MVCC for concurrency control.
"""

import logging
import time
import uuid
from typing import Any, Dict, Optional, Set

logger = logging.getLogger("wdbx.mvcc")


class MVCCTransaction:
    """
    Represents a transaction with MVCC.

    Attributes:
        transaction_id: Unique identifier for the transaction
        version: Transaction version/timestamp
        read_set: Set of keys read during the transaction
        write_set: Set of keys written during the transaction
        status: Transaction status (active, committed, aborted)
    """

    # Transaction status constants
    STATUS_ACTIVE = "active"
    STATUS_COMMITTED = "committed"
    STATUS_ABORTED = "aborted"

    def __init__(self):
        """Initialize a new transaction."""
        self.transaction_id = str(uuid.uuid4())
        self.version = time.time()
        self.read_set: Set[str] = set()
        self.write_set: Set[str] = set()
        self.status = self.STATUS_ACTIVE

        logger.debug(f"Transaction {self.transaction_id} started with version {self.version}")

    def read(self, key: str) -> None:
        """
        Record a read operation.

        Args:
            key: Key being read
        """
        if not self.is_active():
            logger.warning(f"Attempted read on inactive transaction {self.transaction_id}")
            return

        self.read_set.add(key)

    def write(self, key: str) -> None:
        """
        Record a write operation.

        Args:
            key: Key being written
        """
        if not self.is_active():
            logger.warning(f"Attempted write on inactive transaction {self.transaction_id}")
            return

        self.write_set.add(key)

    def commit(self) -> None:
        """Commit the transaction."""
        if not self.is_active():
            logger.warning(f"Cannot commit inactive transaction {self.transaction_id}")
            return

        self.status = self.STATUS_COMMITTED
        logger.debug(f"Transaction {self.transaction_id} committed")

    def abort(self) -> None:
        """Abort the transaction."""
        if not self.is_active():
            logger.warning(f"Cannot abort inactive transaction {self.transaction_id}")
            return

        self.status = self.STATUS_ABORTED
        logger.debug(f"Transaction {self.transaction_id} aborted")

    def is_active(self) -> bool:
        """
        Check if the transaction is active.

        Returns:
            True if active, False otherwise
        """
        return self.status == self.STATUS_ACTIVE

    def is_committed(self) -> bool:
        """
        Check if the transaction is committed.

        Returns:
            True if committed, False otherwise
        """
        return self.status == self.STATUS_COMMITTED

    def is_aborted(self) -> bool:
        """
        Check if the transaction is aborted.

        Returns:
            True if aborted, False otherwise
        """
        return self.status == self.STATUS_ABORTED


class MVCCManager:
    """
    Manager for MVCC transactions.

    Handles transaction creation, validation, and conflict detection.
    """

    def __init__(self):
        """Initialize the MVCC manager."""
        self.transactions: Dict[str, MVCCTransaction] = {}
        self.global_version = time.time()
        self.committed_versions: Dict[str, Dict[float, Any]] = {}

    def start_transaction(self) -> MVCCTransaction:
        """
        Start a new transaction.

        Returns:
            A new MVCC transaction
        """
        transaction = MVCCTransaction()
        self.transactions[transaction.transaction_id] = transaction
        return transaction

    def get_transaction(self, transaction_id: str) -> Optional[MVCCTransaction]:
        """
        Get a transaction by ID.

        Args:
            transaction_id: ID of the transaction

        Returns:
            Transaction if found, None otherwise
        """
        return self.transactions.get(transaction_id)

    def commit_transaction(self, transaction_id: str) -> bool:
        """
        Commit a transaction.

        Args:
            transaction_id: ID of the transaction to commit

        Returns:
            True if successful, False otherwise
        """
        transaction = self.get_transaction(transaction_id)
        if not transaction or not transaction.is_active():
            return False

        # Check for write-write conflicts (optimistic concurrency control)
        for key in transaction.write_set:
            if key in self.committed_versions:
                # Check if another transaction committed a write to this key after our transaction started
                conflicting_versions = [
                    v for v in self.committed_versions[key].keys() if v > transaction.version
                ]
                if conflicting_versions:
                    logger.warning(
                        f"Write-write conflict detected for key {key} in transaction {transaction_id}"
                    )
                    transaction.abort()
                    return False

        # Commit the transaction
        transaction.commit()
        self.global_version = max(self.global_version, time.time())

        # Record committed writes
        for key in transaction.write_set:
            if key not in self.committed_versions:
                self.committed_versions[key] = {}
            self.committed_versions[key][transaction.version] = transaction_id

        return True

    def abort_transaction(self, transaction_id: str) -> bool:
        """
        Abort a transaction.

        Args:
            transaction_id: ID of the transaction to abort

        Returns:
            True if successful, False otherwise
        """
        transaction = self.get_transaction(transaction_id)
        if not transaction or not transaction.is_active():
            return False

        transaction.abort()
        return True

    def cleanup_transactions(self, max_age: float = 3600) -> int:
        """
        Clean up old transactions.

        Args:
            max_age: Maximum age of transactions to keep (in seconds)

        Returns:
            Number of transactions cleaned up
        """
        current_time = time.time()
        to_remove = []

        for tid, transaction in self.transactions.items():
            if not transaction.is_active() and current_time - transaction.version > max_age:
                to_remove.append(tid)

        for tid in to_remove:
            del self.transactions[tid]

        return len(to_remove)
