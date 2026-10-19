import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool_known_answer_4():
    # 1. Create a single session instance
    session = SessionState()

    # 2. Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # 3. Load sample fixture dataset
    load_result = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert load_result["n_documents"] == 8

    # 4. Instantiate tools_dtm.py tools using the same session
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    # 5. Invoke build_dtm_tool with default arguments
    res = build_dtm_tool.invoke({})

    # 6. Check returned metrics against Known Answer 4 in TEST_FIXTURE.md
    assert res["n_features"] == 5
    assert res["non_zero"] == 21
    assert res["total_elements"] == 40
    assert res["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)

    # 7. Check session state mutation
    assert session.feature_matrix is not None
    assert session.feature_matrix.shape == (8, 5)
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]
    assert "count_vectorizer" in session.artifacts


def test_term_frequency_tool_known_answer_14():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    # Build DTM first
    build_dtm_tool.invoke({})

    # Invoke term_frequency_tool
    res = term_frequency_tool.invoke({})

    assert res["n_terms"] == 5
    # Assert against Known Answer 14: alpha=6, always=7, beta=3, delta=4, gamma=7
    expected_top_terms = {
        "always": 7,
        "gamma": 7,
        "alpha": 6,
        "delta": 4,
        "beta": 3,
    }
    assert res["top_terms"] == expected_top_terms


def test_term_frequency_tool_without_dtm():
    session = SessionState()

    dtm_tools = make_tools(session)
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    # Call term_frequency_tool before build_dtm_tool
    res = term_frequency_tool.invoke({})

    assert res["status"] == "error"
    assert "No document-term matrix built" in res["message"]


def test_dtm_heatmap_tool_known_answer_21():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    build_dtm_tool.invoke({})

    # 1. Full Matrix Slice Test (defaults n_terms=20, n_documents=20)
    res_full = dtm_heatmap_tool.invoke({"n_terms": 20, "n_documents": 20})

    assert res_full["n_documents_sliced"] == 8
    assert res_full["n_terms_sliced"] == 5
    assert res_full["terms"] == ["alpha", "always", "beta", "delta", "gamma"]

    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0],
    ]
    assert res_full["matrix_slice"] == expected_full_matrix
    assert session.pending_figure is not None

    # 2. Corner Slice Test (n_terms=2, n_documents=3)
    res_corner = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})

    assert res_corner["n_documents_sliced"] == 3
    assert res_corner["n_terms_sliced"] == 2
    assert res_corner["terms"] == ["alpha", "always"]
    assert res_corner["matrix_slice"] == [[3, 1], [2, 1], [1, 1]]


def test_dtm_heatmap_tool_without_dtm():
    session = SessionState()

    dtm_tools = make_tools(session)
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    res = dtm_heatmap_tool.invoke({})

    assert res["status"] == "error"
    assert "No document-term matrix built" in res["message"]
