import pytest
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool_known_answer_5():
    session = SessionState()

    # 1. Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 2. Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 3. Instantiate variance_filter_tool
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # 4. Invoke variance_filter_tool with threshold=0.15
    res = variance_filter_tool.invoke({"threshold": 0.15})

    # 5. Assert against Known Answer 5 in TEST_FIXTURE.md
    assert res["n_features_before"] == 5
    assert res["n_features_kept"] == 4
    assert res["n_features_removed"] == 1
    assert res["kept_terms"] == ["alpha", "beta", "delta", "gamma"]
    assert res["removed_terms"] == ["always"]

    # 6. Verify stored full_report structure for visualize_result_tool
    result_id = res["result_id"]
    stored_entry = session.results_store[result_id]
    full_report = stored_entry["full_report"]

    assert isinstance(full_report, pd.DataFrame)
    assert list(full_report.columns[:2]) == ["term", "variance"]

    # Explicitly check terms are sorted descending by variance
    variances_list = full_report["variance"].tolist()
    assert variances_list == sorted(variances_list, reverse=True)
    assert full_report["term"].tolist() == ["alpha", "gamma", "delta", "beta", "always"]

    # Check exact variance values
    variances_dict = dict(zip(full_report["term"], full_report["variance"]))
    assert variances_dict["alpha"] == pytest.approx(1.1875, abs=1e-3)
    assert variances_dict["gamma"] == pytest.approx(1.109375, abs=1e-3)
    assert variances_dict["delta"] == pytest.approx(0.25, abs=1e-3)
    assert variances_dict["beta"] == pytest.approx(0.234375, abs=1e-3)
    assert variances_dict["always"] == pytest.approx(0.109375, abs=1e-3)


def test_variance_filter_tool_without_dtm():
    session = SessionState()

    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    res = variance_filter_tool.invoke({"threshold": 0.15})

    assert res["status"] == "error"
    assert "No document-term matrix built" in res["message"]


def test_pearson_filter_tool_known_answer_6():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    # 1. Test default threshold=0.0 against Known Answer 6
    res = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    assert res["target_class"] == "catB"
    assert res["n_features_before"] == 5
    assert res["n_features_kept"] == 5

    corrs = res["correlations"]
    assert corrs["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert corrs["gamma"] == pytest.approx(0.8307, abs=1e-3)
    assert corrs["always"] == pytest.approx(0.3780, abs=1e-3)
    assert corrs["alpha"] == pytest.approx(-0.6882, abs=1e-3)
    assert corrs["beta"] == pytest.approx(-0.7746, abs=1e-3)

    # 2. Check full_report sorting by absolute correlation descending
    result_id = res["result_id"]
    stored_entry = session.results_store[result_id]
    full_report = stored_entry["full_report"]

    assert isinstance(full_report, pd.DataFrame)
    assert list(full_report.columns[:2]) == ["term", "pearson_r"]

    # Terms sorted by absolute correlation descending
    expected_order = ["delta", "gamma", "beta", "alpha", "always"]
    assert full_report["term"].tolist() == expected_order

    # 3. Test custom threshold filtering by absolute value (|r| >= 0.7)
    res_thresh = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.7})
    assert res_thresh["n_features_kept"] == 3
    assert set(res_thresh["kept_terms"]) == {"delta", "gamma", "beta"}
    assert res_thresh["n_features_removed"] == 2
    assert set(res_thresh["removed_terms"]) == {"alpha", "always"}


def test_pearson_filter_tool_invalid_target_class():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    res = pearson_filter_tool.invoke({"target_class": "non_existent_category"})

    assert res["status"] == "error"
    assert "not found in session.labels" in res["message"]


def test_spearman_filter_tool_known_answer_6():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # 1. Test default threshold=0.0 against Known Answer 6
    res = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    assert res["target_class"] == "catB"
    assert res["n_features_before"] == 5
    assert res["n_features_kept"] == 5

    corrs = res["correlations"]
    assert corrs["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert corrs["gamma"] == pytest.approx(0.9363, abs=1e-3)
    assert corrs["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert corrs["alpha"] == pytest.approx(-0.7500, abs=1e-3)
    assert corrs["always"] == pytest.approx(0.3780, abs=1e-3)

    # 2. Check full_report sorting by absolute correlation descending
    result_id = res["result_id"]
    stored_entry = session.results_store[result_id]
    full_report = stored_entry["full_report"]

    assert isinstance(full_report, pd.DataFrame)
    assert list(full_report.columns[:2]) == ["term", "spearman_r"]

    # Terms sorted by absolute correlation descending
    expected_order = ["delta", "gamma", "beta", "alpha", "always"]
    assert full_report["term"].tolist() == expected_order

    # 3. Test custom threshold filtering by absolute value (|r| >= 0.76)
    # Threshold 0.76 preserves positive (delta=1.0, gamma=0.9363) and negative (beta=-0.7746) correlations
    res_thresh = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.76})
    assert res_thresh["n_features_kept"] == 3
    assert set(res_thresh["kept_terms"]) == {"delta", "gamma", "beta"}
    assert res_thresh["n_features_removed"] == 2
    assert set(res_thresh["removed_terms"]) == {"alpha", "always"}


def test_spearman_filter_tool_errors():
    session = SessionState()

    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # 1. Without DTM built
    res_no_dtm = spearman_filter_tool.invoke({"target_class": "catB"})
    assert res_no_dtm["status"] == "error"
    assert "No document-term matrix" in res_no_dtm["message"]

    # 2. Invalid target class
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    res_bad_cat = spearman_filter_tool.invoke({"target_class": "non_existent_category"})
    assert res_bad_cat["status"] == "error"
    assert "not found in session.labels" in res_bad_cat["message"]
