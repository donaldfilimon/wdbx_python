# wdbx/mvcc.py
import time
import uuid
from collections import defaultdict
import threading
from typing import Any, Dict, Optional
from wdbx.constants import logger

class MVCCTransaction:
    """
    Represents a transaction using multiversion concurrency control.
    """
    def __init__(self, transaction_id: Optional[str] = None) -> None:
        self.transaction_id = transaction_id or str(uuid.uuid4())
        self.start_time = time.time()
        self.version = int(self.start_time * 1000)
        self.read_set = set()
        self.write_set = set()
        self.status = "active"
        self.locks = set()
    
    def read(self, key: str) -> None:
        self.read_set.add(key)
    
    def write(self, key: str) -> None:
        self.write_set.add(key)
    
    def acquire_lock(self, key: str) -> bool:
        self.locks.add(key)
        return True
    
    def release_locks(self) -> None:
        self.locks.clear()
    
    def commit(self) -> None:
        self.status = "committed"
        self.release_locks()
    
    def abort(self) -> None:
        self.status = "aborted"
        self.release_locks()
    
    def is_active(self) -> bool:
        return self.status == "active"
    
    def is_committed(self) -> bool:
        return self.status == "committed"
    
    def conflicts_with(self, other: "MVCCTransaction") -> bool:
        if self.write_set.intersection(other.write_set):
            return True
        if self.read_set.intersection(other.write_set):
            return True
        return False


class MVCCManager:
    """
    Manages MVCC transactions and versioned data.
    """
    def __init__(self) -> None:
        self.transactions: Dict[str, MVCCTransaction] = {}
        self.versions = defaultdict(list)  # key -> list of (version, value, transaction_id)
        self.locks: Dict[str, str] = {}
        self.lock = threading.RLock()
    
    def start_transaction(self) -> MVCCTransaction:
        with self.lock:
            transaction = MVCCTransaction()
            self.transactions[transaction.transaction_id] = transaction
            return transaction
    
    def read(self, transaction_id: str, key: str) -> Optional[Any]:
        with self.lock:
            transaction = self.transactions.get(transaction_id)
            if not transaction or not transaction.is_active():
                raise ValueError(f"Invalid or inactive transaction: {transaction_id}")
            transaction.read(key)
            versions = self.versions.get(key, [])
            valid_versions = [(v, val, tid) for v, val, tid in versions if v <= transaction.version or tid == transaction_id]
            if not valid_versions:
                return None
            valid_versions.sort(reverse=True)
            return valid_versions[0][1]
    
    def write(self, transaction_id: str, key: str, value: Any) -> bool:
        with self.lock:
            transaction = self.transactions.get(transaction_id)
            if not transaction or not transaction.is_active():
                raise ValueError(f"Invalid or inactive transaction: {transaction_id}")
            if key in self.locks and self.locks[key] != transaction_id:
                return False
            if not transaction.acquire_lock(key):
                return False
            self.locks[key] = transaction_id
            transaction.write(key)
            self.versions[key].append((transaction.version, value, transaction_id))
            return True
    
    def commit(self, transaction_id: str) -> bool:
        with self.lock:
            transaction = self.transactions.get(transaction_id)
            if not transaction or not transaction.is_active():
                return False
            transaction.commit()
            for key in list(transaction.locks):
                if key in self.locks and self.locks[key] == transaction_id:
                    del self.locks[key]
            return True
    
    def abort(self, transaction_id: str) -> bool:
        with self.lock:
            transaction = self.transactions.get(transaction_id)
            if not transaction or not transaction.is_active():
                return False
            transaction.abort()
            for key in transaction.write_set:
                self.versions[key] = [(v, val, tid) for v, val, tid in self.versions[key] if tid != transaction_id]
                if not self.versions[key]:
                    del self.versions[key]
            for key in list(transaction.locks):
                if key in self.locks and self.locks[key] == transaction_id:
                    del self.locks[key]
            return True
    
    def cleanup_old_versions(self, max_age: float = 3600.0) -> int:
        with self.lock:
            current_time = time.time()
            min_version = int((current_time - max_age) * 1000)
            removed_count = 0
            for key in list(self.versions.keys()):
                versions = self.versions[key]
                if not versions:
                    continue
                latest_version = max(versions, key=lambda x: x[0])
                new_versions = [v for v in versions if v[0] >= min_version or v == latest_version]
                removed_count += len(versions) - len(new_versions)
                if not new_versions:
                    del self.versions[key]
                else:
                    self.versions[key] = new_versions
            return removed_count
