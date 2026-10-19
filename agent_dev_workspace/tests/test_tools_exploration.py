import pytest
import numpy as np
import matplotlib.figure

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools


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


def test_cosine_similarity_known_answers():
    """Test cosine_similarity_tool against Known Answer #16 in TEST_FIXTURE.md."""
    session = _setup_session_with_dtm()
    exploration_tools = make_exploration_tools(session)
    cosine_sim_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # 1. Doc 0 vs Doc 1 (both catA, near-identical text): expected ~0.9847
    res_0_1 = cosine_sim_tool.invoke({"doc_index_1": 0, "doc_index_2": 1})
    assert res_0_1["similarity"] == pytest.approx(0.9847, abs=1e-3)

    # 2. Doc 0 vs Doc 3 (catA vs catB, share only "always"): expected ~0.0909
    res_0_3 = cosine_sim_tool.invoke({"doc_index_1": 0, "doc_index_2": 3})
    assert res_0_3["similarity"] == pytest.approx(0.0909, abs=1e-3)

    # 3. Doc 3 vs Doc 4 (both catB, near-identical text): expected ~0.9847
    res_3_4 = cosine_sim_tool.invoke({"doc_index_1": 3, "doc_index_2": 4})
    assert res_3_4["similarity"] == pytest.approx(0.9847, abs=1e-3)

    # 4. Doc 0 vs Doc 7 (catA vs empty text row): expected exactly 0.0
    res_0_7 = cosine_sim_tool.invoke({"doc_index_1": 0, "doc_index_2": 7})
    assert res_0_7["similarity"] == 0.0


def test_cosine_similarity_by_text():
    """Test cosine_similarity_tool using raw text strings transformed by stored vectorizer."""
    session = _setup_session_with_dtm()
    exploration_tools = make_exploration_tools(session)
    cosine_sim_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    text_1 = "always alpha alpha alpha beta"
    text_2 = "always alpha alpha beta"

    res = cosine_sim_tool.invoke({"text_1": text_1, "text_2": text_2})
    assert res["similarity"] == pytest.approx(0.9847, abs=1e-3)


def test_cosine_similarity_priority():
    """Test priority when both document indices and raw text strings are supplied."""
    session = _setup_session_with_dtm()
    exploration_tools = make_exploration_tools(session)
    cosine_sim_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # Indices 0 vs 3 yield ~0.0909.
    # If text_1 and text_2 were used (identical strings), it would yield 1.0.
    # Indices must take precedence over text arguments.
    res = cosine_sim_tool.invoke({
        "doc_index_1": 0,
        "doc_index_2": 3,
        "text_1": "always alpha",
        "text_2": "always alpha"
    })

    assert res["similarity"] == pytest.approx(0.0909, abs=1e-3)


def test_cosine_similarity_errors():
    """Test error handling for invalid arguments and missing session state."""
    session = _setup_session_with_dtm()
    exploration_tools = make_exploration_tools(session)
    cosine_sim_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # 1. Out of bounds document index
    res_oob = cosine_sim_tool.invoke({"doc_index_1": 0, "doc_index_2": 99})
    assert res_oob.get("status") == "error"
    assert "out of bounds" in res_oob.get("message", "").lower()

    # 2. Incomplete parameter pairs (only doc_index_1 provided)
    res_incomplete = cosine_sim_tool.invoke({"doc_index_1": 0})
    assert res_incomplete.get("status") == "error"

    # 3. Missing DTM state in session
    session_no_dtm = SessionState()
    exploration_tools_no_dtm = make_exploration_tools(session_no_dtm)
    cosine_sim_tool_no_dtm = next(t for t in exploration_tools_no_dtm if t.name == "cosine_similarity_tool")
    
    res_no_dtm = cosine_sim_tool_no_dtm.invoke({"doc_index_1": 0, "doc_index_2": 1})
    assert res_no_dtm.get("status") == "error"


def test_feature_correlation_matrix_known_answers():
    """Test feature_correlation_matrix_tool against Known Answer #19 in TEST_FIXTURE.md."""
    session = _setup_session_with_dtm()
    exploration_tools = make_exploration_tools(session)
    feat_corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    res = feat_corr_tool.invoke({"top_k": 20})

    # 1. Check term ordering (descending variance): alpha, gamma, delta, beta, always
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["terms"] == expected_terms
    assert res["n_terms"] == 5

    # 2. Check pending_figure is created and set on session
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)

    # 3. Check 5x5 Pearson correlation matrix against Known Answer #19 numbers
    corr_matrix = np.array(res["correlation_matrix"])
    expected_corr = np.array([
        [1.0000, -0.5718, -0.6882,  0.8885, 0.2601],
        [-0.5718, 1.0000,  0.8307, -0.6435, 0.3140],
        [-0.6882, 0.8307,  1.0000, -0.7746, 0.3780],
        [0.8885, -0.6435, -0.7746,  1.0000, 0.2928],
        [0.2601,  0.3140,  0.3780,  0.2928, 1.0000]
    ])

    assert np.allclose(corr_matrix, expected_corr, atol=1e-3)


def test_feature_correlation_matrix_no_dtm():
    """Test feature_correlation_matrix_tool returns error if DTM not built."""
    session = SessionState()
    exploration_tools = make_exploration_tools(session)
    feat_corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    res = feat_corr_tool.invoke({"top_k": 20})
    assert res.get("status") == "error"
    assert "No document-term matrix built" in res.get("message", "")
