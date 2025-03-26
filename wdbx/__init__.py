# wdbx/__init__.py
from re import I, M
from numpy import ma
import cli as cli
from wdbx.data_structures import EmbeddingVector, Block
from wdbx.vector_store import VectorStore, VectorOperations
from wdbx.blockchain import BlockChainManager
from wdbx.mvcc import MVCCManager
from wdbx.neural_backtracking import NeuralBacktracker
from wdbx.shard_manager import ShardManager
from wdbx.persona import PersonaManager

class WDBX:
    """
    Core implementation of the Wide Distributed Block Exchange (WDBX) system.
    """
    def __init__(self, vector_dimension: int, num_shards: int) -> None:
            import time
            self.vector_dimension = vector_dimension
            self.num_shards = num_shards
            self.vector_store = VectorStore(dimension=vector_dimension)
            self.shard_manager = ShardManager(num_shards=num_shards)
            self.block_chain_manager = BlockChainManager()
            self.mvcc_manager = MVCCManager()
            self.neural_backtracker = NeuralBacktracker(
                block_chain_manager=self.block_chain_manager,
                vector_store=self.vector_store
            )
            self.stats = {
                "blocks_created": 0,
                "vectors_stored": 0,
                "transactions_processed": 0,
                "traces_created": 0,
                "start_time": time.time()
            }

    def store_embedding(self, embedding_vector: EmbeddingVector) -> str:
        transaction = self.mvcc_manager.start_transaction()
        import uuid
        vector_id = str(uuid.uuid4())
        try:
            embedding_vector.metadata["vector_id"] = vector_id
            if not self.vector_store.add(vector_id, embedding_vector):
                self.mvcc_manager.abort(transaction.transaction_id)
                raise ValueError(f"Failed to store vector {vector_id}")
            self.mvcc_manager.commit(transaction.transaction_id)
            self.stats["vectors_stored"] += 1
            self.stats["transactions_processed"] += 1
            return vector_id
        except Exception as e:
            self.mvcc_manager.abort(transaction.transaction_id)
            raise e

    def create_conversation_block(self, data: dict, embeddings: list, chain_id: str = None, context_references: list = None) -> str:
        transaction = self.mvcc_manager.start_transaction()
        try:
            for embedding in embeddings:
                vector_id = self.store_embedding(embedding)
                embedding.metadata.setdefault("vector_ids", []).append(vector_id)
            block = self.block_chain_manager.create_block(
                data=data,
                embeddings=embeddings,
                chain_id=chain_id,
                context_references=context_references
            )
            self.shard_manager.get_shard_for_block(block.id)
            self.mvcc_manager.commit(transaction.transaction_id)
            self.stats["blocks_created"] += 1
            return block.id
        except Exception as e:
            self.mvcc_manager.abort(transaction.transaction_id)
            raise e

    def search_similar_vectors(self, query_vector: any, top_k: int = 10) -> list:
        return self.vector_store.search_similar(query_vector, top_k=top_k)

    def create_neural_trace(self, query_vector: any) -> str:
        trace_id = self.neural_backtracker.trace_activation(query_vector)
        self.stats["traces_created"] += 1
        return trace_id

    def get_conversation_context(self, block_ids: list) -> dict:
        blocks = []
        embeddings = []
        chains = set()
        for block_id in block_ids:
            block = self.block_chain_manager.get_block(block_id)
            if block:
                blocks.append(block)
                embeddings.extend(block.embeddings)
                for cid, head in self.block_chain_manager.chain_heads.items():
                    chain_blocks = self.block_chain_manager.get_chain(cid)
                    if any(b.id == block_id for b in chain_blocks):
                        chains.add(cid)
                        break
        context_blocks = []
        for block in blocks:
            context_blocks.extend(self.block_chain_manager.get_context_blocks(block.id))
        aggregated_embedding = None
        if embeddings:
            from wdbx.vector_store import VectorOperations
            vectors = [e.vector for e in embeddings]
            aggregated_embedding = VectorOperations.average_vectors(vectors)
        return {
            "blocks": blocks,
            "context_blocks": context_blocks,
            "chains": list(chains),
            "aggregated_embedding": aggregated_embedding
        }

    def get_system_stats(self) -> dict:
            import time
            uptime = time.time() - self.stats["start_time"]
            return {
                **self.stats,
                "uptime": uptime,
                "blocks_per_second": self.stats["blocks_created"] / uptime if uptime > 0 else 0,
                "vectors_per_second": self.stats["vectors_stored"] / uptime if uptime > 0 else 0,
                "shard_count": self.num_shards,
                "vector_dimension": self.vector_dimension
            }