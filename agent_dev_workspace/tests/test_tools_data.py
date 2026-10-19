import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_check_missing_tool_known_answer_2():
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

    # 4. Instantiate tools_data.py tools using the same session
    data_tools = make_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    # 5. Invoke check_missing_tool
    res = check_missing_tool.invoke({"text_column": "text"})

    # 6. Assert against Known Answer 2 in TEST_FIXTURE.md
    assert res["n_documents"] == 8
    assert res["missing_count"] == 1
    assert res["missing_pct"] == pytest.approx(12.5, abs=1e-3)
    assert res["missing_indices"] == [7]
    assert res["text_column"] == "text"
    assert res["result_id"].startswith("check_missing_")

    # 7. Non-destructive check: assert session.dataframe length remains unchanged
    assert len(session.dataframe) == 8


def test_check_duplicates_tool_inspection_known_answer_3():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_result = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert load_result["n_documents"] == 8

    data_tools = make_tools(session)
    check_duplicates_tool = next(t for t in data_tools if t.name == "check_duplicates_tool")

    res = check_duplicates_tool.invoke({"drop": False})

    assert res["n_documents_before"] == 8
    assert res["n_documents_after"] == 8
    assert res["duplicate_count"] == 1
    assert res["duplicate_indices"] == [6]
    assert res["dropped"] is False

    assert len(session.dataframe) == 8


def test_check_duplicates_tool_drop_known_answer_3():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_result = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert load_result["n_documents"] == 8

    data_tools = make_tools(session)
    check_duplicates_tool = next(t for t in data_tools if t.name == "check_duplicates_tool")

    res = check_duplicates_tool.invoke({"drop": True})

    assert res["n_documents_before"] == 8
    assert res["n_documents_after"] == 6
    assert res["duplicate_count"] == 1
    assert res["dropped"] is True

    # Assert remaining rows in session.dataframe (rows 5 and 6 "always gamma delta" should be gone)
    remaining_texts = session.dataframe["text"].tolist()
    assert len(remaining_texts) == 6
    assert "always gamma delta" not in remaining_texts


def test_sample_data_tool_known_answer_13():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_result = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert load_result["n_documents"] == 8

    data_tools = make_tools(session)
    sample_data_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    res = sample_data_tool.invoke({"n": 4, "random_state": 42})

    assert res["n_documents_before"] == 8
    assert res["n_sampled"] == 4
    assert res["sample_indices"] == [1, 5, 0, 7]
    assert res["random_state"] == 42

    # Assert session.dataframe was updated in-place to 4 rows
    assert len(session.dataframe) == 4


def test_sample_data_tool_invalid_parameters():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    sample_data_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    # 1. Provide both n and frac simultaneously
    res1 = sample_data_tool.invoke({"n": 4, "frac": 0.5})
    assert res1["status"] == "error"
    assert "specify either 'n' or 'frac'" in res1["message"]

    # 2. n greater than total document count (n=20 on 8 rows)
    res2 = sample_data_tool.invoke({"n": 20})
    assert res2["status"] == "error"
    assert "Invalid sampling parameters" in res2["message"]

    # 3. Negative frac
    res3 = sample_data_tool.invoke({"frac": -0.5})
    assert res3["status"] == "error"
    assert "Invalid sampling parameters" in res3["message"]


def test_tokenize_tool_known_answer_10():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_result = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert load_result["n_documents"] == 8

    data_tools = make_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    res = tokenize_tool.invoke({"text_column": "text"})

    assert res["n_documents"] == 8
    assert res["total_tokens"] == 27
    assert res["text_column"] == "text"

    # Known Answer 10 token lists per row
    expected_unigrams = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        [],
    ]

    assert "unigrams" in session.dataframe.columns
    actual_unigrams = session.dataframe["unigrams"].tolist()
    assert actual_unigrams == expected_unigrams


def test_list_files_tool_known_answer_11():
    session = SessionState()

    data_tools = make_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    res = list_files_tool.invoke({"subdirectory": "newdataset"})

    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]
    assert res["subdirectory"] == "newdataset"


def test_list_files_tool_invalid_directory():
    session = SessionState()

    data_tools = make_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    # 1. Non-existent directory
    res1 = list_files_tool.invoke({"subdirectory": "non_existent_folder_xyz"})
    assert res1["status"] == "error"
    assert "not found" in res1["message"]

    # 2. Out of repository root directory traversal
    res2 = list_files_tool.invoke({"subdirectory": "../../"})
    assert res2["status"] == "error"
    assert "Access denied" in res2["message"]


def test_describe_data_tool_known_answer_18():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_result = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert load_result["n_documents"] == 8

    data_tools = make_tools(session)
    describe_data_tool = next(t for t in data_tools if t.name == "describe_data_tool")

    res = describe_data_tool.invoke({"text_column": "text"})

    # 1. Assert overall stats against Known Answer 18
    overall = res["overall_stats"]
    assert overall["count"] == 8
    assert overall["mean"] == pytest.approx(19.875, abs=1e-3)
    assert overall["std"] == pytest.approx(9.4330, abs=1e-3)
    assert overall["min"] == 0.0
    assert overall["25%"] == pytest.approx(17.75, abs=1e-3)
    assert overall["50%"] == pytest.approx(20.5, abs=1e-3)
    assert overall["75%"] == pytest.approx(25.25, abs=1e-3)
    assert overall["max"] == 30.0

    # 2. Assert by-category stats against Known Answer 18
    by_cat = res["by_category_stats"]
    assert "catA" in by_cat
    assert "catB" in by_cat

    catA = by_cat["catA"]
    assert catA["count"] == 4
    assert catA["mean"] == pytest.approx(17.25, abs=1e-3)
    assert catA["std"] == pytest.approx(12.5, abs=1e-3)
    assert catA["min"] == 0.0
    assert catA["25%"] == pytest.approx(12.75, abs=1e-3)
    assert catA["50%"] == pytest.approx(20.0, abs=1e-3)
    assert catA["75%"] == pytest.approx(24.5, abs=1e-3)
    assert catA["max"] == 29.0

    catB = by_cat["catB"]
    assert catB["count"] == 4
    assert catB["mean"] == pytest.approx(22.5, abs=1e-3)
    assert catB["std"] == pytest.approx(5.7446, abs=1e-3)
    assert catB["min"] == 18.0
    assert catB["25%"] == pytest.approx(18.0, abs=1e-3)
    assert catB["50%"] == pytest.approx(21.0, abs=1e-3)
    assert catB["75%"] == pytest.approx(25.5, abs=1e-3)
    assert catB["max"] == 30.0

    # 3. Assert plot side-effect on session.pending_figure
    assert session.pending_figure is not None

    # 4. Read-only check: dataset length unchanged
    assert len(session.dataframe) == 8


def test_describe_data_tool_errors():
    session = SessionState()

    data_tools = make_tools(session)
    describe_data_tool = next(t for t in data_tools if t.name == "describe_data_tool")

    # 1. No dataset loaded
    res1 = describe_data_tool.invoke({"text_column": "text"})
    assert res1["status"] == "error"
    assert "No dataset loaded" in res1["message"]

    # 2. Invalid text column
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    res2 = describe_data_tool.invoke({"text_column": "non_existent_column"})
    assert res2["status"] == "error"
    assert "not found" in res2["message"]


def test_inspect_data_tool():
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load student-built tools from tools_data.py
    data_tools = make_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")
    sample_data_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    # Load sample fixture (8 documents)
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    # Test 1: Default inspection on initial loaded fixture (8 documents)
    res1 = inspect_data_tool.invoke({})
    assert res1["n_documents"] == 8
    assert res1["n_inspected"] == 8
    assert "text" in res1["columns"]
    assert "category_name" in res1["columns"]
    assert len(res1["preview_rows"]) == 8

    # Test 2: Custom n parameter (n=3)
    res2 = inspect_data_tool.invoke({"n": 3})
    assert res2["n_documents"] == 8
    assert res2["n_inspected"] == 3
    assert len(res2["preview_rows"]) == 3

    # Test 3: Inspection after tokenize_tool adds "unigrams" column
    tokenize_tool.invoke({"text_column": "text"})
    res3 = inspect_data_tool.invoke({})
    assert "unigrams" in res3["columns"]
    assert "unigrams" in res3["preview_rows"][0]
    assert isinstance(res3["preview_rows"][0]["unigrams"], list)

    # Test 4: Inspection after sample_data_tool modifies dataframe (n=4, random_state=42)
    sample_data_tool.invoke({"n": 4, "random_state": 42})
    res4 = inspect_data_tool.invoke({})
    assert res4["n_documents"] == 4
    assert res4["n_inspected"] == 4
    assert len(res4["preview_rows"]) == 4
