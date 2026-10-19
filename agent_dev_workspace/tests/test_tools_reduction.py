import pytest
import numpy as np
import matplotlib.figure

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools


def _setup_session_with_dtm():
    """Helper function to load sample fixture data and build the global DTM."""
    session = SessionState()
    
    # Load dataset tool from premade_tools
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load sample_fixture.csv
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    return session


def test_reduce_dimensions_pca_known_answer():
    """Test PCA against Known Answer #7 in TEST_FIXTURE.md (deterministic across environments)."""
    session = _setup_session_with_dtm()
    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_tool.invoke({"method": "pca", "n_components": 2})

    # 1. Verify pending_figure is set and is a Figure
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)

    # 2. Check explained_variance_ratio ~ [0.7532, 0.1965]
    expected_evr = [0.7532, 0.1965]
    assert np.allclose(res["explained_variance_ratio"], expected_evr, atol=1e-3)

    # 3. Check 8x2 coordinates against Known Answer #7
    coords = np.array(res["coordinates"])
    expected_coords = np.array([
        [ 2.3669,  0.8871],
        [ 1.7039,  0.2642],
        [ 1.0408, -0.3587],
        [-2.0759,  1.0129],
        [-1.4554,  0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [ 0.0894, -1.4822],
    ])

    assert np.allclose(coords, expected_coords, atol=1e-3)


def test_reduce_dimensions_reproducibility():
    """Test that running t-SNE and UMAP with the same random_state produces identical coordinates."""
    for method, kwargs in [("tsne", {"perplexity": 2}), ("umap", {"n_neighbors": 3})]:
        session1 = _setup_session_with_dtm()
        reduce_tool1 = next(t for t in make_reduction_tools(session1) if t.name == "reduce_dimensions_tool")
        res1 = reduce_tool1.invoke({"method": method, "random_state": 42, **kwargs})

        session2 = _setup_session_with_dtm()
        reduce_tool2 = next(t for t in make_reduction_tools(session2) if t.name == "reduce_dimensions_tool")
        res2 = reduce_tool2.invoke({"method": method, "random_state": 42, **kwargs})

        coords1 = np.array(res1["coordinates"])
        coords2 = np.array(res2["coordinates"])

        # Identical seeds must produce 100% identical coordinate output
        assert np.allclose(coords1, coords2, atol=1e-5), f"{method} failed reproducibility check for identical random_state"


def test_reduce_dimensions_shape_and_artifacts():
    """Test output shape and artifact storage for PCA, t-SNE, and UMAP."""
    methods_config = [
        ("pca", {}),
        ("tsne", {"perplexity": 2}),
        ("umap", {"n_neighbors": 3}),
    ]

    for method, kwargs in methods_config:
        session = _setup_session_with_dtm()
        reduce_tool = next(t for t in make_reduction_tools(session) if t.name == "reduce_dimensions_tool")

        res = reduce_tool.invoke({"method": method, "n_components": 2, **kwargs})

        # Check returned coordinates shape
        coords = np.array(res["coordinates"])
        assert coords.shape == (8, 2), f"{method} returned shape {coords.shape}, expected (8, 2)"

        # Check full artifact storage in session.artifacts
        artifact_key = f"reduced_{method}"
        assert artifact_key in session.artifacts
        assert session.artifacts[artifact_key].shape == (8, 2)


def test_reduce_dimensions_duplicate_rows_pca_tsne():
    """Test that exact duplicate documents (rows 5 & 6) map to identical coordinates in PCA and t-SNE."""
    for method, kwargs in [("pca", {}), ("tsne", {"perplexity": 2})]:
        session = _setup_session_with_dtm()
        reduce_tool = next(t for t in make_reduction_tools(session) if t.name == "reduce_dimensions_tool")

        res = reduce_tool.invoke({"method": method, "random_state": 42, **kwargs})
        coords = np.array(res["coordinates"])

        # Row 5 and Row 6 are exact duplicate texts ("always gamma delta"), must map to identical 2D points
        assert np.allclose(coords[5], coords[6], atol=1e-4), f"{method} failed duplicate document coordinate equality"


def test_reduce_dimensions_plotting():
    """Test that session.pending_figure is set to a Figure for all reduction methods."""
    methods_config = [
        ("pca", {}),
        ("tsne", {"perplexity": 2}),
        ("umap", {"n_neighbors": 3}),
    ]

    for method, kwargs in methods_config:
        session = _setup_session_with_dtm()
        reduce_tool = next(t for t in make_reduction_tools(session) if t.name == "reduce_dimensions_tool")

        reduce_tool.invoke({"method": method, **kwargs})

        # Must assign matplotlib.figure.Figure on session.pending_figure every time
        assert session.pending_figure is not None
        assert isinstance(session.pending_figure, matplotlib.figure.Figure)


def test_reduce_dimensions_no_dtm():
    """Test reduce_dimensions_tool returns error if DTM not built."""
    session = SessionState()
    reduce_tool = next(t for t in make_reduction_tools(session) if t.name == "reduce_dimensions_tool")

    res = reduce_tool.invoke({"method": "pca"})
    assert res.get("status") == "error"
    assert "No document-term matrix built" in res.get("message", "")


def test_binarize_labels_known_fixture():
    """Test binarize_labels_tool on sample_fixture.csv."""
    session = _setup_session_with_dtm()
    reduction_tools = make_reduction_tools(session)
    binarize_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    res = binarize_tool.invoke({})

    assert res.get("status") == "success"
    assert res.get("n_samples") == 8
    assert res.get("n_classes") == 2
    assert res.get("classes") == ["catA", "catB"]
    assert res.get("matrix_shape") == [8, 2]

    # Verify artifacts
    assert "binarized_labels" in session.artifacts
    matrix = session.artifacts["binarized_labels"]
    assert matrix.shape == (8, 2)

    # Check exact row mappings: rows 0,1,2,7 are catA [1, 0]; rows 3,4,5,6 are catB [0, 1]
    expected_catA = [1, 0]
    expected_catB = [0, 1]

    for idx in [0, 1, 2, 7]:
        assert list(matrix[idx]) == expected_catA, f"Row {idx} expected catA binary vector [1, 0]"

    for idx in [3, 4, 5, 6]:
        assert list(matrix[idx]) == expected_catB, f"Row {idx} expected catB binary vector [0, 1]"


def test_binarize_labels_dataframe_unmodified():
    """Test that binarize_labels_tool does not mutate or add columns to session.dataframe."""
    session = _setup_session_with_dtm()
    reduction_tools = make_reduction_tools(session)
    binarize_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    df_shape_before = session.dataframe.shape
    df_cols_before = list(session.dataframe.columns)

    binarize_tool.invoke({})

    assert session.dataframe.shape == df_shape_before
    assert list(session.dataframe.columns) == df_cols_before


def test_binarize_labels_missing_dataframe():
    """Test binarize_labels_tool returns error when dataframe is missing."""
    session = SessionState()
    reduction_tools = make_reduction_tools(session)
    binarize_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    res = binarize_tool.invoke({})
    assert "error" in res or res.get("status") == "error"


def test_binarize_labels_missing_category_column():
    """Test binarize_labels_tool returns error when category_name column is missing."""
    import pandas as pd
    session = SessionState()
    session.dataframe = pd.DataFrame({"text": ["hello", "world"]})
    reduction_tools = make_reduction_tools(session)
    binarize_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    res = binarize_tool.invoke({})
    assert "error" in res or res.get("status") == "error"
