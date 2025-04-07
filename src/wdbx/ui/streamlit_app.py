"""
Streamlit-based UI for WDBX.

This provides a simple web interface for interacting with WDBX,
allowing users to visualize embeddings, search for similar vectors,
and manage the vector store.
"""

import builtins
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Add the parent directory to sys.path to allow importing WDBX
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
sys.path.append(str(project_root))

try:
    from wdbx import WDBX, EmbeddingVector, WDBXConfig
except ImportError:
    st.error("Failed to import WDBX modules. Please ensure WDBX is installed.")
    st.stop()

# Constants
DEFAULT_DIMENSION = 128
DEFAULT_NUM_SHARDS = 2
DEFAULT_DATA_DIR = "./wdbx_data"

# Page configuration
st.set_page_config(
    page_title="WDBX Vector Explorer",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Helper Functions

# Check if a WDBX instance was provided


def get_provided_wdbx() -> Optional[WDBX]:
    """Get WDBX instance if it was provided by the CLI."""
    return getattr(builtins, "_WDBX_INSTANCE", None)


@st.cache_data
def reduce_dimensions(vectors: np.ndarray, method: str = "PCA", n_components: int = 3):
    """Reduce vector dimensions for visualization."""
    if method == "PCA":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components)
    elif method == "TSNE":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=n_components, random_state=42)
    elif method == "UMAP":
        try:
            import umap
            reducer = umap.UMAP(n_components=n_components, random_state=42)
        except ImportError:
            st.warning("UMAP not installed. Falling back to PCA.")
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components)
    else:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components)

    return reducer.fit_transform(vectors)


def initialize_wdbx() -> WDBX:
    """Initialize WDBX instance with configured parameters."""
    config = WDBXConfig(
        vector_dimension=st.session_state.dimension,
        num_shards=st.session_state.num_shards,
        data_dir=st.session_state.data_dir
    )

    try:
        db = WDBX(config=config)
        return db
    except Exception as e:
        st.error(f"Failed to initialize WDBX: {e}")
        st.stop()


def get_embedding_dataframe(db: WDBX) -> pd.DataFrame:
    """Get embeddings from WDBX as a DataFrame."""
    embeddings = []

    for vector_id in db.vector_store.list_ids():
        embedding = db.vector_store.get(vector_id)
        if embedding:
            metadata = embedding.metadata or {}
            row = {
                "id": vector_id,
                "vector": embedding.vector,
                **metadata
            }
            embeddings.append(row)

    if not embeddings:
        return pd.DataFrame()

    df = pd.DataFrame(embeddings)
    return df


def visualize_embeddings(df: pd.DataFrame, dim_reduction: str = "PCA"):
    """Visualize embeddings using the specified dimension reduction technique."""
    if df.empty:
        st.warning("No embeddings to visualize.")
        return

    # Stack all vectors into a 2D array
    vectors = np.stack(df["vector"].values)

    # Reduce dimensions for visualization
    reduced_vecs = reduce_dimensions(vectors, method=dim_reduction)

    # Create a visualization dataframe
    viz_df = pd.DataFrame()
    viz_df["id"] = df["id"]
    viz_df["x"] = reduced_vecs[:, 0]
    viz_df["y"] = reduced_vecs[:, 1]
    if reduced_vecs.shape[1] >= 3:
        viz_df["z"] = reduced_vecs[:, 2]

    # Add metadata columns
    for col in df.columns:
        if col not in ["id", "vector"]:
            viz_df[col] = df[col]

    # Create visualization
    if reduced_vecs.shape[1] >= 3:
        fig = px.scatter_3d(
            viz_df, x="x", y="y", z="z",
            color="id" if "type" not in viz_df.columns else "type",
            hover_data=viz_df.columns,
            title=f"Vector Embeddings ({dim_reduction})"
        )
    else:
        fig = px.scatter(
            viz_df, x="x", y="y",
            color="id" if "type" not in viz_df.columns else "type",
            hover_data=viz_df.columns,
            title=f"Vector Embeddings ({dim_reduction})"
        )

    st.plotly_chart(fig, use_container_width=True)

    # Display data table
    with st.expander("Show embedding data"):
        st.dataframe(viz_df.drop(columns=["vector"] if "vector" in viz_df.columns else []))


def add_sample_embeddings(db: WDBX, count: int = 10):
    """Add sample embeddings to the database."""
    for i in range(count):
        vector = np.random.randn(st.session_state.dimension).astype(np.float32)
        embedding = EmbeddingVector(
            vector=vector,
            metadata={
                "description": f"Sample vector {i}",
                "timestamp": 1649926800 + i,
                "source": "streamlit_app",
                "type": f"sample_{i % 3}"  # Create some groups for visualization
            }
        )
        vector_id = f"sample_{i:03d}"
        db.vector_store.add(vector_id, embedding)

    st.success(f"Added {count} sample embeddings.")

# Main App UI


def main():
    st.title("WDBX Vector Explorer")

    # Check if a WDBX instance was provided via CLI
    provided_wdbx = get_provided_wdbx()

    # Initialize session state
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
        st.session_state.dimension = DEFAULT_DIMENSION
        st.session_state.num_shards = DEFAULT_NUM_SHARDS
        st.session_state.data_dir = DEFAULT_DATA_DIR
        # Use provided WDBX instance if available
        if provided_wdbx:
            st.session_state.wdbx = provided_wdbx
            st.session_state.initialized = True
            st.session_state.dimension = provided_wdbx.vector_dimension
            st.session_state.num_shards = len(provided_wdbx.vector_store.shards)
            # Get data directory from provided WDBX if possible
            if hasattr(provided_wdbx, "data_dir"):
                st.session_state.data_dir = provided_wdbx.data_dir

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        # If using provided WDBX, show info but disable inputs
        if provided_wdbx and st.session_state.initialized:
            st.info("Using WDBX instance provided by CLI")
            st.text(f"Vector dimension: {st.session_state.dimension}")
            st.text(f"Number of shards: {st.session_state.num_shards}")
            st.text(f"Data directory: {st.session_state.data_dir}")
        else:
            # Allow configuration when no WDBX instance was provided
            st.session_state.dimension = st.number_input(
                "Vector Dimension",
                min_value=2,
                max_value=4096,
                value=st.session_state.dimension
            )
            st.session_state.num_shards = st.number_input(
                "Number of Shards",
                min_value=1,
                max_value=32,
                value=st.session_state.num_shards
            )
            st.session_state.data_dir = st.text_input(
                "Data Directory",
                value=st.session_state.data_dir
            )

            if st.button("Initialize WDBX"):
                st.session_state.wdbx = initialize_wdbx()
                st.session_state.initialized = True
                st.success("WDBX initialized successfully!")

        # Sample data section always available
        if st.session_state.initialized:
            if st.button("Add Sample Embeddings"):
                sample_count = st.number_input(
                    "Number of samples", min_value=1, max_value=100, value=10)
                add_sample_embeddings(st.session_state.wdbx, count=sample_count)

    # Main content area
    if not st.session_state.initialized:
        st.info("Please initialize WDBX using the sidebar options.")
    else:
        # Get the dataframe of embeddings
        db = st.session_state.wdbx
        df = get_embedding_dataframe(db)

        # Show basic stats
        st.subheader("Vector Store Statistics")
        st.write(f"Total vectors: {len(df)}")
        st.write(f"Vector dimension: {st.session_state.dimension}")
        st.write(f"Number of shards: {st.session_state.num_shards}")

        # Visualization section
        st.subheader("Vector Visualization")
        dim_reduction = st.selectbox(
            "Dimension Reduction Technique",
            options=["PCA", "TSNE", "UMAP"],
            index=0
        )

        visualize_embeddings(df, dim_reduction)

        # Search section
        st.subheader("Vector Search")
        search_type = st.radio("Search Type", ["Random Vector", "Existing Vector"])

        if search_type == "Random Vector":
            random_seed = st.number_input("Random Seed", value=42, min_value=0)
            np.random.seed(random_seed)
            search_vector = np.random.randn(st.session_state.dimension).astype(np.float32)
        else:
            if df.empty:
                st.warning("No existing vectors to search from.")
                return

            selected_id = st.selectbox("Select Vector ID", options=df["id"].tolist())
            selected_row = df[df["id"] == selected_id].iloc[0]
            search_vector = selected_row["vector"]

        top_k = st.slider(
            "Number of Results", min_value=1, max_value=min(
                100, len(df)), value=min(
                5, len(df)))
        threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.0)

        if st.button("Search"):
            results = db.vector_store.search_similar(
                search_vector, top_k=top_k, threshold=threshold)

            if not results:
                st.warning("No results found.")
            else:
                # Create results dataframe
                results_df = pd.DataFrame({
                    "Vector ID": [r[0] for r in results],
                    "Similarity": [r[1] for r in results]
                })

                st.dataframe(results_df)

                # Visualize search results
                result_ids = results_df["Vector ID"].tolist()
                result_df = df[df["id"].isin(result_ids)]

                st.subheader("Search Results Visualization")
                visualize_embeddings(result_df, dim_reduction)


if __name__ == "__main__":
    main()
