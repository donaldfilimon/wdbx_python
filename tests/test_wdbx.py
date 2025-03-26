# wdbx/tests/test_wdbx.py
import unittest
import numpy as np
import time
import uuid
from wdbx import WDBX
from wdbx.data_structures import EmbeddingVector, Block
from wdbx.vector_store import VectorStore, VectorOperations
from wdbx.blockchain import BlockChainManager
from wdbx.mvcc import MVCCManager
from wdbx.neural_backtracking import NeuralBacktracker
from wdbx.shard_manager import ShardManager
from wdbx.persona import PersonaManager


class TestEmbeddingVector(unittest.TestCase):
    def test_create_embedding(self):
        # Test creating an embedding vector
        vector = np.random.rand(128).astype(np.float32)
        metadata = {"description": "Test vector", "source": "test"}
        embedding = EmbeddingVector(vector=vector, metadata=metadata)

        self.assertEqual(embedding.vector.shape, (128,))
        self.assertEqual(embedding.metadata["description"], "Test vector")
        self.assertEqual(embedding.metadata["source"], "test")

    def test_normalize(self):
        # Test vector normalization
        vector = np.array([3.0, 4.0])
        embedding = EmbeddingVector(vector=vector)
        normalized = embedding.normalize()

        # The norm of [3, 4] is 5, so normalized vector should be [0.6, 0.8]
        self.assertAlmostEqual(normalized[0], 0.6, places=5)
        self.assertAlmostEqual(normalized[1], 0.8, places=5)

    def test_serialization(self):
        # Test serialization and deserialization
        vector = np.random.rand(64).astype(np.float32)
        metadata = {"name": "test", "timestamp": 123456789}
        embedding = EmbeddingVector(vector=vector, metadata=metadata)

        serialized = embedding.serialize()
        deserialized = EmbeddingVector.deserialize(serialized)

        self.assertTrue(np.allclose(embedding.vector, deserialized.vector))
        self.assertEqual(embedding.metadata, deserialized.metadata)


class TestBlock(unittest.TestCase):
    def test_create_block(self):
        # Test creating a block
        block_id = str(uuid.uuid4())
        timestamp = time.time()
        data = {"message": "Hello world"}
        embeddings = [
            EmbeddingVector(vector=np.random.rand(32), metadata={"type": "query"}),
            EmbeddingVector(vector=np.random.rand(32), metadata={"type": "response"})
        ]

        block = Block(
            id=block_id,
            timestamp=timestamp,
            data=data,
            embeddings=embeddings,
            previous_hash="prev_hash_123"
        )

        self.assertEqual(block.id, block_id)
        self.assertEqual(block.data["message"], "Hello world")
        self.assertEqual(len(block.embeddings), 2)
        self.assertEqual(block.previous_hash, "prev_hash_123")
        self.assertTrue(block.hash.startswith("0"))  # Mining should produce hash with leading zeros

    def test_validate(self):
        # Test block validation
        block = Block(
            id="test_block",
            timestamp=time.time(),
            data={"test": "data"},
            embeddings=[EmbeddingVector(vector=np.random.rand(16))],
            previous_hash="prev_hash"
        )

        # Block should be valid initially
        self.assertTrue(block.validate())

        # Tamper with data
        block.data["test"] = "modified"
        # Block should now be invalid
        self.assertFalse(block.validate())


class TestVectorStore(unittest.TestCase):
    def setUp(self):
        self.vector_store = VectorStore(dimension=64)

    def test_add_vector(self):
        # Test adding a vector
        vector = EmbeddingVector(vector=np.random.rand(64), metadata={"test": "data"})
        vector_id = "test_vector_1"

        success = self.vector_store.add(vector_id, vector)
        self.assertTrue(success)

        # Test retrieving the vector
        retrieved = self.vector_store.get(vector_id)
        self.assertIsNotNone(retrieved)
        self.assertTrue(np.allclose(vector.vector, retrieved.vector))
        self.assertEqual(vector.metadata, retrieved.metadata)

    def test_update_vector(self):
        # Add a vector first
        vector = EmbeddingVector(vector=np.random.rand(64), metadata={"test": "data"})
        vector_id = "test_vector_2"
        self.vector_store.add(vector_id, vector)

        # Update the vector
        updated_vector = EmbeddingVector(vector=np.random.rand(64), metadata={"test": "updated"})
        success = self.vector_store.update(vector_id, updated_vector)

        self.assertTrue(success)
        retrieved = self.vector_store.get(vector_id)
        self.assertTrue(np.allclose(updated_vector.vector, retrieved.vector))
        self.assertEqual(retrieved.metadata["test"], "updated")

    def test_delete_vector(self):
        # Add a vector first
        vector = EmbeddingVector(vector=np.random.rand(64), metadata={"test": "data"})
        vector_id = "test_vector_3"
        self.vector_store.add(vector_id, vector)

        # Delete the vector
        success = self.vector_store.delete(vector_id)
        self.assertTrue(success)

        # Vector should no longer be retrievable
        retrieved = self.vector_store.get(vector_id)
        self.assertIsNone(retrieved)

    def test_search_similar(self):
        # Add some vectors
        vectors = []
        vector_ids = []

        # Create vectors with specific patterns for predictable search results
        base_vector = np.array([1.0, 0.0, 0.0, 0.0, 0.0] * 12 + [0.0, 0.0, 0.0, 0.0])

        for i in range(5):
            # Create variations of the base vector
            v = base_vector.copy()
            v[i] = 0.8  # Make each vector a bit different
            vector = EmbeddingVector(vector=v, metadata={"index": i})
            vector_id = f"test_vector_{i}"
            vectors.append(vector)
            vector_ids.append(vector_id)
            self.vector_store.add(vector_id, vector)

        # Search with a vector similar to the first one
        query_vector = base_vector.copy()
        query_vector[0] = 0.9

        results = self.vector_store.search_similar(query_vector, top_k=3)

        self.assertEqual(len(results), 3)
        # The most similar vector should be the first one
        self.assertEqual(results[0][0], vector_ids[0])


class TestBlockChainManager(unittest.TestCase):
    def setUp(self):
        self.bcm = BlockChainManager()

    def test_create_block(self):
        # Test creating a block with no previous chain
        data = {"message": "First block"}
        embeddings = [EmbeddingVector(vector=np.random.rand(32))]

        block = self.bcm.create_block(data=data, embeddings=embeddings)

        self.assertIsNotNone(block)
        self.assertIn(block.id, self.bcm.blocks)

        # Get the chain_id for this block
        chain_id = next(iter(self.bcm.chain_heads))
        self.assertEqual(self.bcm.chain_heads[chain_id], block.id)

        # Create another block in the same chain
        data2 = {"message": "Second block"}
        embeddings2 = [EmbeddingVector(vector=np.random.rand(32))]

        block2 = self.bcm.create_block(
            data=data2,
            embeddings=embeddings2,
            chain_id=chain_id
        )

        self.assertIsNotNone(block2)
        self.assertEqual(self.bcm.chain_heads[chain_id], block2.id)
        self.assertEqual(block2.previous_hash, block.hash)

    def test_get_chain(self):
        # Create a chain with 3 blocks
        chain_id = None
        blocks = []

        for i in range(3):
            data = {"index": i}
            embeddings = [EmbeddingVector(vector=np.random.rand(32))]

            block = self.bcm.create_block(
                data=data,
                embeddings=embeddings,
                chain_id=chain_id
            )

            blocks.append(block)
            if chain_id is None:
                chain_id = next(cid for cid, bid in self.bcm.chain_heads.items() if bid == block.id)

        # Get the chain
        chain_blocks = self.bcm.get_chain(chain_id)

        # Chain should be in reverse order (newest first)
        self.assertEqual(len(chain_blocks), 3)
        self.assertEqual(chain_blocks[0].id, blocks[2].id)
        self.assertEqual(chain_blocks[1].id, blocks[1].id)
        self.assertEqual(chain_blocks[2].id, blocks[0].id)

    def test_validate_chain(self):
        # Create a valid chain
        chain_id = None

        for i in range(3):
            data = {"index": i}
            embeddings = [EmbeddingVector(vector=np.random.rand(32))]

            block = self.bcm.create_block(
                data=data,
                embeddings=embeddings,
                chain_id=chain_id
            )

            if chain_id is None:
                chain_id = next(cid for cid, bid in self.bcm.chain_heads.items() if bid == block.id)

        # Chain should be valid
        self.assertTrue(self.bcm.validate_chain(chain_id))

        # Tamper with a block
        block_id = self.bcm.get_chain(chain_id)[1].id
        self.bcm.blocks[block_id].data["index"] = 999

        # Chain should now be invalid
        self.assertFalse(self.bcm.validate_chain(chain_id))


class TestMVCCManager(unittest.TestCase):
    def setUp(self):
        self.mvcc = MVCCManager()

    def test_transactions(self):
        # Start a transaction
        tx = self.mvcc.start_transaction()
        self.assertIsNotNone(tx)
        self.assertTrue(tx.is_active())

        # Write some data
        success = self.mvcc.write(tx.transaction_id, "key1", "value1")
        self.assertTrue(success)

        # Read the data
        value = self.mvcc.read(tx.transaction_id, "key1")
        self.assertEqual(value, "value1")

        # Commit the transaction
        success = self.mvcc.commit(tx.transaction_id)
        self.assertTrue(success)
        self.assertTrue(tx.is_committed())

    def test_transaction_isolation(self):
        # Start two transactions
        tx1 = self.mvcc.start_transaction()
        tx2 = self.mvcc.start_transaction()

        # Write data in tx1
        self.mvcc.write(tx1.transaction_id, "key2", "value_tx1")

        # tx2 shouldn't see tx1's uncommitted data
        value = self.mvcc.read(tx2.transaction_id, "key2")
        self.assertIsNone(value)

        # Commit tx1
        self.mvcc.commit(tx1.transaction_id)

        # Now tx2 should see tx1's committed data (since tx2's version is higher)
        value = self.mvcc.read(tx2.transaction_id, "key2")
        self.assertEqual(value, "value_tx1")

    def test_transaction_abort(self):
        # Start a transaction
        tx = self.mvcc.start_transaction()

        # Write some data
        self.mvcc.write(tx.transaction_id, "key3", "temp_value")

        # Abort the transaction
        success = self.mvcc.abort(tx.transaction_id)
        self.assertTrue(success)

        # Start a new transaction
        tx2 = self.mvcc.start_transaction()

        # The aborted transaction's data should not be visible
        value = self.mvcc.read(tx2.transaction_id, "key3")
        self.assertIsNone(value)


class TestWDBX(unittest.TestCase):
    def setUp(self):
        self.wdbx = WDBX(vector_dimension=128, num_shards=4)

    def test_store_embedding(self):
        # Create and store an embedding
        vector = np.random.rand(128).astype(np.float32)
        embedding = EmbeddingVector(vector=vector, metadata={"source": "test"})

        vector_id = self.wdbx.store_embedding(embedding)

        self.assertIsNotNone(vector_id)
        # Check that the vector is in the store
        stored_vector = self.wdbx.vector_store.get(vector_id)
        self.assertIsNotNone(stored_vector)
        self.assertTrue(np.allclose(vector, stored_vector.vector))

    def test_create_conversation_block(self):
        # Create embeddings
        embeddings = [
            EmbeddingVector(vector=np.random.rand(128), metadata={"role": "user"}),
            EmbeddingVector(vector=np.random.rand(128), metadata={"role": "assistant"})
        ]

        # Create a conversation block
        data = {
            "user_input": "Hello, how are you?",
            "response": "I'm doing well, thank you!",
            "timestamp": time.time()
        }

        block_id = self.wdbx.create_conversation_block(data=data, embeddings=embeddings)

        self.assertIsNotNone(block_id)
        # Check that the block is in the blockchain
        block = self.wdbx.block_chain_manager.get_block(block_id)
        self.assertIsNotNone(block)
        self.assertEqual(block.data, data)
        self.assertEqual(len(block.embeddings), 2)

    def test_search_similar_vectors(self):
        # Store some vectors
        vectors = []
        vector_ids = []

        for i in range(5):
            # Create vectors that will have different similarities
            v = np.zeros(128)
            v[i*10:(i+1)*10] = 1.0  # Set a different region to 1 for each vector
            vector = EmbeddingVector(vector=v, metadata={"index": i})
            vector_id = self.wdbx.store_embedding(vector)
            vectors.append(v)
            vector_ids.append(vector_id)

        # Search with a vector similar to the third one
        query_vector = np.zeros(128)
        query_vector[20:30] = 1.0  # Similar to the third vector

        results = self.wdbx.search_similar_vectors(query_vector, top_k=3)

        self.assertEqual(len(results), 3)
        # The most similar vector should be the third one (index 2)
        self.assertEqual(results[0][0], vector_ids[2])

    def test_conversation_context(self):
        # Create a chain of conversation blocks
        chain_id = None
        block_ids = []

        for i in range(3):
            embeddings = [
                EmbeddingVector(vector=np.random.rand(128), metadata={"role": "user"}),
                EmbeddingVector(vector=np.random.rand(128), metadata={"role": "assistant"})
            ]

            data = {
                "user_input": f"Message {i}",
                "response": f"Response {i}",
                "timestamp": time.time()
            }

            block_id = self.wdbx.create_conversation_block(
                data=data,
                embeddings=embeddings,
                chain_id=chain_id,
                context_references=block_ids
            )

            block_ids.append(block_id)
            if chain_id is None:
                # Find the chain ID for the first block
                block = self.wdbx.block_chain_manager.get_block(block_id)
                for cid, head in self.wdbx.block_chain_manager.chain_heads.items():
                    if head == block_id:
                        chain_id = cid
                        break

        # Get conversation context for the latest block
        context = self.wdbx.get_conversation_context([block_ids[-1]])

        self.assertEqual(len(context["blocks"]), 1)
        self.assertEqual(len(context["context_blocks"]), 2)  # Should have 2 previous blocks as context
        self.assertIn(chain_id, context["chains"])
        self.assertIsNotNone(context["aggregated_embedding"])


class TestPersonaManager(unittest.TestCase):
    def setUp(self):
        self.wdbx = WDBX(vector_dimension=128, num_shards=4)
        self.persona_manager = PersonaManager(self.wdbx)

    def test_determine_persona(self):
        # Test technical input
        technical_input = "Can you explain how the algorithm works?"
        persona = self.persona_manager.determine_optimal_persona(technical_input)
        self.assertEqual(persona, "Aviva")

        # Test emotional input
        emotional_input = "I feel sad today, can you help me feel better?"
        persona = self.persona_manager.determine_optimal_persona(emotional_input)
        self.assertEqual(persona, "Abbey")

        # Test mixed input
        mixed_input = "I'm worried about my coding project."
        persona = self.persona_manager.determine_optimal_persona(mixed_input)
        self.assertIsInstance(persona, dict)
        self.assertIn("Abbey", persona)
        self.assertIn("Aviva", persona)

    def test_process_user_input(self):
        # Test processing a user input
        user_input = "Tell me about the WDBX system"
        context = {"chain_id": None, "block_ids": []}

        response, block_id = self.persona_manager.process_user_input(user_input, context)

        self.assertIsNotNone(response)
        self.assertIsNotNone(block_id)

        # Check that a block was created
        block = self.wdbx.block_chain_manager.get_block(block_id)
        self.assertIsNotNone(block)
        self.assertEqual(block.data["user_input"], user_input)

        # Check that the response is in the block
        self.assertEqual(block.data["response"], response)


class TestVectorOperations(unittest.TestCase):
    def test_average_vectors(self):
        # Test averaging vectors
        vectors = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
            np.array([7.0, 8.0, 9.0])
        ]

        avg = VectorOperations.average_vectors(vectors)

        self.assertEqual(avg.shape, (3,))
        self.assertEqual(avg[0], 4.0)
        self.assertEqual(avg[1], 5.0)
        self.assertEqual(avg[2], 6.0)

    def test_cluster_vectors(self):
        # Test clustering vectors
        vectors = [
            np.array([1.0, 1.0]),
            np.array([1.1, 1.1]),
            np.array([9.0, 9.0]),
            np.array([9.1, 9.1])
        ]

        clusters, centers = VectorOperations.cluster_vectors(vectors, n_clusters=2)

        self.assertEqual(len(clusters), 4)
        self.assertEqual(len(centers), 2)

        # The first two vectors should be in one cluster, the second two in another
        self.assertEqual(clusters[0], clusters[1])
        self.assertEqual(clusters[2], clusters[3])
        self.assertNotEqual(clusters[0], clusters[2])


class TestShardManager(unittest.TestCase):
    def setUp(self):
        self.shard_manager = ShardManager(num_shards=4)

    def test_shard_assignment(self):
        # Test assigning blocks to shards
        block_ids = [f"block_{i}" for i in range(10)]

        shard_assignments = {}
        for block_id in block_ids:
            shard_id = self.shard_manager.get_shard_for_block(block_id)
            shard_assignments[block_id] = shard_id

        # Each block should have a shard assigned
        self.assertEqual(len(shard_assignments), 10)

        # The same block should always get the same shard
        for block_id in block_ids:
            shard_id = self.shard_manager.get_shard_for_block(block_id)
            self.assertEqual(shard_id, shard_assignments[block_id])

        # Shards should be somewhat balanced (all shards should have some blocks)
        used_shards = set(shard_assignments.values())
        self.assertTrue(len(used_shards) > 1)  # At least some load balancing

    def test_optimal_shards(self):
        # Test getting optimal shards for retrieval
        # First assign some blocks to create load
        for i in range(20):
            block_id = f"load_block_{i}"
            shard_id = i % 4  # Distribute across shards
            self.shard_manager.shard_assignment[block_id] = shard_id
            self.shard_manager.shards[shard_id].block_count += 1
            self.shard_manager.shards[shard_id].load = min(1.0, self.shard_manager.shards[shard_id].block_count / 10)

        # Get optimal shards
        optimal_shards = self.shard_manager.get_optimal_shards(retrieval_size=1000, count=2)

        self.assertEqual(len(optimal_shards), 2)
        # The shards with the lowest load should be returned first
        shard_loads = [(shard_id, self.shard_manager.shards[shard_id].load) for shard_id in range(4)]
        shard_loads.sort(key=lambda x: x[1])
        self.assertEqual(optimal_shards[0], shard_loads[0][0])


class TestNeuralBacktracker(unittest.TestCase):
    def setUp(self):
        self.wdbx = WDBX(vector_dimension=64, num_shards=4)
        self.backtracker = self.wdbx.neural_backtracker

    def test_trace_activation(self):
        # Store some vectors and create blocks
        block_ids = []
        for i in range(5):
            vector = np.zeros(64)
            vector[i*10:(i+1)*10] = 1.0
            embedding = EmbeddingVector(vector=vector, metadata={"index": i})

            data = {"message": f"Block {i}"}

            block_id = self.wdbx.create_conversation_block(
                data=data,
                embeddings=[embedding]
            )
            block_ids.append(block_id)

        # Create a trace with a vector similar to the third block
        query_vector = np.zeros(64)
        query_vector[20:30] = 1.0  # Similar to the third vector

        trace_id = self.backtracker.trace_activation(query_vector)

        self.assertIsNotNone(trace_id)
        self.assertIn(trace_id, self.backtracker.activation_traces)

        # The third block should have the highest activation
        activations = self.backtracker.activation_traces[trace_id]
        self.assertIn(block_ids[2], activations)

        max_block_id = max(activations.items(), key=lambda x: x[1])[0]
        self.assertEqual(max_block_id, block_ids[2])


# wdbx/tests/__init__.py
import wdbx

def test_wdbx():
    # Initialize the WDBX environment
    db = wdbx.WDBX(vector_dimension=64, num_shards=4)

    # Test basic functionality
    assert db is not None

    # Test data operations
    import numpy as np
    from wdbx.data_structures import EmbeddingVector
    test_vector = np.random.rand(64).astype(np.float32)
    embedding = EmbeddingVector(vector=test_vector, metadata={"key": "value"})
    vector_id = db.store_embedding(embedding)
    
    # Test querying
    similar_vectors = db.search_similar_vectors(test_vector, top_k=1)
    assert len(similar_vectors) > 0
    assert similar_vectors[0][0] == vector_id