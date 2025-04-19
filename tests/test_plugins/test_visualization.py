"""
Unit tests for the visualization plugin.
"""

import pytest
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import visualization module
try:
    from wdbx_plugins.visualization import (
        cmd_export_all_visualizations,
        cmd_histogram,
        cmd_pca_visualization,
        cmd_plot,
        cmd_similarity_matrix,
        cmd_tsne_visualization,
        cmd_vector_heatmap,
        cmd_visualization_config,
        get_vectors_for_query,
        register_commands,
    )
except ImportError:
    # Skip tests if module not available
    visualization_available = False
else:
    visualization_available = True

# Skip entire test class if dependencies not available

try:
    import numpy as np

    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

try:
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend for testing
    import matplotlib.pyplot as plt

    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False

try:
    pass

    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


class MockWDBX:
    """Mock WDBX instance for testing."""

    def __init__(self):
        self.vector_dimension = 128
        self.vector_store = MagicMock()
        self.block_chain_manager = MagicMock()
        # Create a mock vector store
        self.vectors = {
            f"test_vector_{i}": np.random.rand(self.vector_dimension).astype(np.float32)
            for i in range(5)
        }

    def search_similar_vectors(self, query_vector, top_k=10):
        """Mock implementation of search_similar_vectors."""
        # Return list of tuples (id, similarity_score)
        return [(f"test_vector_{i}", 0.9 - (i * 0.1)) for i in range(min(5, top_k))]

    def create_embedding_from_text(self, text):
        """Mock implementation of create_embedding_from_text."""
        # Return a random vector of the correct dimension
        return np.random.rand(self.vector_dimension).astype(np.float32)

    def get_vector_by_id(self, vector_id):
        """Mock implementation of get_vector_by_id."""
        return self.vectors.get(vector_id)

    def retrieve_vector(self, vector_id):
        """Mock implementation of retrieve_vector."""
        return self.vectors.get(vector_id)


@pytest.mark.skipif(not visualization_available, reason="Visualization plugin not available")
class TestVisualizationPlugin(unittest.TestCase):
    """Test the visualization plugin functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.db = MockWDBX()
        self.plugin_registry = {}
        # Create a temporary directory for visualizations
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name

        # Register commands in the plugin registry
        if visualization_available:
            register_commands(self.plugin_registry)

            # Configure visualization to use temp directory
            cmd_visualization_config(self.db, f"output_dir={self.output_dir}")

    def tearDown(self):
        """Tear down test fixtures."""
        # Close all matplotlib figures
        if HAVE_MATPLOTLIB:
            plt.close("all")

        # Clean up temporary directory
        self.temp_dir.cleanup()

    @pytest.mark.skipif(not HAVE_NUMPY, reason="NumPy not available")
    def test_get_vectors_for_query(self):
        """Test getting vectors for a query."""
        vector_ids, vectors = get_vectors_for_query(self.db, "test query", max_vectors=3)

        # Check that we got the expected number of vectors
        self.assertEqual(len(vector_ids), 3)
        self.assertEqual(len(vectors), 3)

        # Check that the vector IDs match the expected format
        for i, vec_id in enumerate(vector_ids):
            self.assertEqual(vec_id, f"test_vector_{i}")

        # Check that the vectors have the correct shape
        for vec in vectors:
            self.assertEqual(vec.shape, (self.db.vector_dimension,))

    def test_register_commands(self):
        """Test that commands are registered correctly."""
        self.assertIn("plot", self.plugin_registry)
        self.assertIn("histogram", self.plugin_registry)
        self.assertIn("similarity", self.plugin_registry)
        self.assertIn("pca", self.plugin_registry)
        self.assertIn("tsne", self.plugin_registry)
        self.assertIn("heatmap", self.plugin_registry)
        self.assertIn("vis", self.plugin_registry)
        self.assertIn("vis:config", self.plugin_registry)
        self.assertIn("vis:export", self.plugin_registry)

    @pytest.mark.skipif(
        not (HAVE_MATPLOTLIB and HAVE_NUMPY), reason="Matplotlib and/or NumPy not available"
    )
    def test_plot_command(self):
        """Test the plot command."""
        # Mock the save_figure function to avoid creating files
        with patch("wdbx_plugins.visualization.save_figure") as mock_save:
            mock_save.return_value = os.path.join(self.output_dir, "test_plot.png")

            # Add default truncate_labels config value if missing
            with patch.dict("wdbx_plugins.visualization.vis_config", {"truncate_labels": True}):
                # Call the plot command
                cmd_plot(self.db, "test query")

                # Check that save_figure was called at least once
                self.assertTrue(mock_save.called, "save_figure should have been called")

    @pytest.mark.skipif(
        not (HAVE_MATPLOTLIB and HAVE_NUMPY), reason="Matplotlib and/or NumPy not available"
    )
    def test_histogram_command(self):
        """Test the histogram command."""
        # Mock the save_figure function to avoid creating files
        with patch("wdbx_plugins.visualization.save_figure") as mock_save:
            mock_save.return_value = os.path.join(self.output_dir, "test_histogram.png")

            # Call the histogram command
            cmd_histogram(self.db, "test query")

            # Check that save_figure was called at least once
            self.assertTrue(mock_save.called, "save_figure should have been called")

    @pytest.mark.skipif(
        not (HAVE_MATPLOTLIB and HAVE_NUMPY), reason="Matplotlib and/or NumPy not available"
    )
    def test_similarity_matrix_command(self):
        """Test the similarity matrix command."""
        # Mock the save_figure function to avoid creating files
        with patch("wdbx_plugins.visualization.save_figure") as mock_save:
            mock_save.return_value = os.path.join(self.output_dir, "test_similarity.png")

            # Call the similarity matrix command
            cmd_similarity_matrix(self.db, "test query")

            # Check that save_figure was called at least once
            self.assertTrue(mock_save.called, "save_figure should have been called")

    @pytest.mark.skipif(
        not (HAVE_MATPLOTLIB and HAVE_NUMPY and HAVE_SKLEARN),
        reason="Matplotlib, NumPy, and/or scikit-learn not available",
    )
    def test_pca_visualization_command(self):
        """Test the PCA visualization command."""
        # Mock the save_figure function to avoid creating files
        with patch("wdbx_plugins.visualization.save_figure") as mock_save:
            mock_save.return_value = os.path.join(self.output_dir, "test_pca.png")

            # Call the PCA visualization command
            cmd_pca_visualization(self.db, "test query")

            # Check that save_figure was called at least once
            self.assertTrue(mock_save.called, "save_figure should have been called")

    @pytest.mark.skipif(
        not (HAVE_MATPLOTLIB and HAVE_NUMPY and HAVE_SKLEARN),
        reason="Matplotlib, NumPy, and/or scikit-learn not available",
    )
    def test_tsne_visualization_command(self):
        """Test the t-SNE visualization command."""
        # Mock the save_figure function to avoid creating files
        with patch("wdbx_plugins.visualization.save_figure") as mock_save:
            mock_save.return_value = os.path.join(self.output_dir, "test_tsne.png")

            # Call the t-SNE visualization command with necessary config values
            with patch.dict(
                "wdbx_plugins.visualization.vis_config",
                {"output_dir": self.output_dir, "dpi": 100, "max_vectors": 5},
            ):
                cmd_tsne_visualization(self.db, "test query")

            # Check that save_figure was called at least once
            self.assertTrue(mock_save.called, "save_figure should have been called")

    @pytest.mark.skipif(
        not (HAVE_MATPLOTLIB and HAVE_NUMPY), reason="Matplotlib and/or NumPy not available"
    )
    def test_vector_heatmap_command(self):
        """Test the vector heatmap command."""
        # Mock the save_figure function to avoid creating files
        with patch("wdbx_plugins.visualization.save_figure") as mock_save:
            mock_save.return_value = os.path.join(self.output_dir, "test_heatmap.png")

            # Call the vector heatmap command
            cmd_vector_heatmap(self.db, "test query")

            # Check that save_figure was called at least once
            self.assertTrue(mock_save.called, "save_figure should have been called")

    @pytest.mark.skipif(
        not (HAVE_MATPLOTLIB and HAVE_NUMPY), reason="Matplotlib and/or NumPy not available"
    )
    def test_export_all_visualizations_command(self):
        """Test the export all visualizations command."""
        # Mock all the individual visualization commands
        with (
            patch("wdbx_plugins.visualization.cmd_plot") as mock_plot,
            patch("wdbx_plugins.visualization.cmd_histogram") as mock_histogram,
            patch("wdbx_plugins.visualization.cmd_similarity_matrix") as mock_similarity,
            patch("wdbx_plugins.visualization.cmd_pca_visualization") as mock_pca,
            patch("wdbx_plugins.visualization.cmd_tsne_visualization") as mock_tsne,
            patch("wdbx_plugins.visualization.cmd_vector_heatmap") as mock_heatmap,
        ):

            # Call the export all visualizations command
            cmd_export_all_visualizations(self.db, "test query")

            # Check that all visualization commands were called
            mock_plot.assert_called_once()
            mock_histogram.assert_called_once()
            mock_similarity.assert_called_once()
            mock_pca.assert_called_once()
            mock_tsne.assert_called_once()
            mock_heatmap.assert_called_once()


if __name__ == "__main__":
    unittest.main()
