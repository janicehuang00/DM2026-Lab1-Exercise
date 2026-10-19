from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools


def test_mine_patterns_parameter_validation():
    session = SessionState()
    premade, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # Missing or non-positive min_sup for fpgrowth returns error dictionary
    res_none_sup = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": None
    })
    assert "error" in res_none_sup
    assert "min_sup" in res_none_sup["error"]

    res_zero_sup = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 0
    })
    assert "error" in res_zero_sup
    assert "min_sup" in res_zero_sup["error"]

    # Missing or non-positive k for topk returns error dictionary
    res_none_k = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": None
    })
    assert "error" in res_none_k
    assert "k" in res_none_k["error"]

    res_neg_k = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": -1
    })
    assert "error" in res_neg_k
    assert "k" in res_neg_k["error"]


def test_mine_patterns_known_answer_17():
    session = SessionState()
    premade, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # 1. Variance filtering keeps 0 terms
    res_var = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_var["n_surviving_terms"] == 0
    assert res_var["patterns"] == []
    assert res_var["note"] == "no terms survived filtering"

    # 2. Term frequency filtering keeps 0 terms
    res_tf = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "term_frequency",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_tf["n_surviving_terms"] == 0
    assert res_tf["patterns"] == []
    assert res_tf["note"] == "no terms survived filtering"

    # 3. TF-IDF filtering catA: keeps alpha, support 3 at min_sup=1
    res_tfidf_cata = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_tfidf_cata["n_surviving_terms"] == 1
    assert len(res_tfidf_cata["patterns"]) == 1
    assert res_tfidf_cata["patterns"][0]["pattern"] == ["alpha"]
    assert res_tfidf_cata["patterns"][0]["support"] == 3

    # catA at min_sup=4 finds 0 patterns
    res_tfidf_cata_4 = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4
    })
    assert res_tfidf_cata_4["total_patterns"] == 0
    assert res_tfidf_cata_4["patterns"] == []

    # 4. TF-IDF filtering catB: keeps gamma, support 4 at min_sup=1
    res_tfidf_catb = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_tfidf_catb["n_surviving_terms"] == 1
    assert len(res_tfidf_catb["patterns"]) == 1
    assert res_tfidf_catb["patterns"][0]["pattern"] == ["gamma"]
    assert res_tfidf_catb["patterns"][0]["support"] == 4


def test_mine_patterns_known_answer_20():
    session = SessionState()
    premade, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # 1. algorithm="topk"
    res_topk_cata = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert len(res_topk_cata["patterns"]) == 1
    assert res_topk_cata["patterns"][0]["pattern"] == ["alpha"]
    assert res_topk_cata["patterns"][0]["support"] == 3

    res_topk_catb = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert len(res_topk_catb["patterns"]) == 1
    assert res_topk_catb["patterns"][0]["pattern"] == ["gamma"]
    assert res_topk_catb["patterns"][0]["support"] == 4

    # 2. algorithm="maxfpgrowth" returns 0 patterns for both categories
    res_max_cata = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_max_cata["total_patterns"] == 0
    assert res_max_cata["patterns"] == []

    res_max_catb = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_max_catb["total_patterns"] == 0
    assert res_max_catb["patterns"] == []
