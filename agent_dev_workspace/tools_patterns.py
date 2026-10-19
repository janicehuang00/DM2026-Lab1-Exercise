import os
import tempfile
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Literal
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool


def make_tools(session):
    """
    Factory function to create tools bound to the provided session.
    """

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"],
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Mines frequent itemsets/patterns for a specific category using PAMI.

        Args:
            category_name: The category label string to filter documents by.
            filtering_method: Method to filter terms ('variance', 'tfidf', or 'term_frequency').
            algorithm: Mining algorithm to run ('fpgrowth', 'topk', or 'maxfpgrowth').
            min_sup: Minimum support threshold (required for 'fpgrowth' and 'maxfpgrowth').
            k: Number of top patterns to return (required for 'topk').

        Returns:
            A summary dictionary with mined patterns, support counts, and status info.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        # Parameter validation for algorithms
        if algorithm in ["fpgrowth", "maxfpgrowth"]:
            if min_sup is None or min_sup <= 0:
                return {
                    "error": f"Algorithm '{algorithm}' requires a positive integer 'min_sup'."
                }
        elif algorithm == "topk":
            if k is None or k <= 0:
                return {
                    "error": "Algorithm 'topk' requires a positive integer 'k'."
                }

        result_id: str = session.next_result_id("mine_patterns")

        # Step 1: Filter dataframe by category_name
        df_cat = session.dataframe[
            session.dataframe["category_name"] == category_name
        ]
        if df_cat.empty:
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_surviving_terms": 0,
                "n_patterns": 0,
                "patterns": [],
                "note": f"No documents found for category '{category_name}'.",
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # Preserve all category rows (fill missing/empty text with empty string)
        texts: List[str] = df_cat["text"].fillna("").astype(str).tolist()

        # Build fresh per-category DTM with stop_words='english'
        vectorizer = CountVectorizer(stop_words="english")
        try:
            dtm = vectorizer.fit_transform(texts)
        except ValueError:
            # Vectorizer raises ValueError if vocabulary is completely empty
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_surviving_terms": 0,
                "n_patterns": 0,
                "patterns": [],
                "note": "No vocabulary terms found after English stopword removal.",
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        feature_names: List[str] = vectorizer.get_feature_names_out().tolist()

        if len(feature_names) == 0:
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_surviving_terms": 0,
                "n_patterns": 0,
                "patterns": [],
                "note": "No vocabulary terms after English stopword removal.",
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        dtm_dense = dtm.toarray()

        # Step 2: Apply chosen filtering_method
        surviving_indices: List[int] = []
        if filtering_method == "variance":
            variances = np.var(dtm_dense, axis=0)
            low_pct = np.percentile(variances, 5)
            high_pct = np.percentile(variances, 95)
            surviving_indices = [
                i for i, v in enumerate(variances) if low_pct < v < high_pct
            ]
        elif filtering_method == "term_frequency":
            counts = np.sum(dtm_dense, axis=0)
            low_pct = np.percentile(counts, 5)
            high_pct = np.percentile(counts, 95)
            surviving_indices = [
                i for i, c in enumerate(counts) if low_pct < c < high_pct
            ]
        elif filtering_method == "tfidf":
            tfidf_transformer = TfidfTransformer()
            tfidf_matrix = tfidf_transformer.fit_transform(dtm)
            mean_tfidf = np.array(tfidf_matrix.mean(axis=0)).flatten()
            cutoff = np.percentile(mean_tfidf, 20)
            surviving_indices = [
                i for i, score in enumerate(mean_tfidf) if score >= cutoff
            ]

        # Step 3: Handle zero terms surviving
        if not surviving_indices:
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_surviving_terms": 0,
                "n_patterns": 0,
                "patterns": [],
                "note": "no terms survived filtering",
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        surviving_terms: List[str] = [feature_names[i] for i in surviving_indices]
        filtered_dtm = dtm_dense[:, surviving_indices]

        # Step 4: Convert to Transactional DB using pandas -> DF2DB
        df_trans = pd.DataFrame(filtered_dtm, columns=surviving_terms)

        temp_db_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt")
        temp_db_path = temp_db_file.name
        temp_db_file.close()

        try:
            from PAMI.extras.convert import DF2DB
            converter = DF2DB.DF2DB(df_trans)
            converter.convert2TransactionalDatabase(temp_db_path, '>=', 1)

            # Step 5: Execute requested mining algorithm
            if algorithm == "fpgrowth":
                from PAMI.frequentPattern.basic import FPGrowth
                obj = FPGrowth.FPGrowth(temp_db_path, minSup=min_sup)
            elif algorithm == "topk":
                from PAMI.frequentPattern.topk import FAE
                obj = FAE.FAE(temp_db_path, k=k)
            elif algorithm == "maxfpgrowth":
                from PAMI.frequentPattern.maximal import MaxFPGrowth
                obj = MaxFPGrowth.MaxFPGrowth(temp_db_path, minSup=min_sup)

            obj.mine()
            patterns_df = obj.getPatternsAsDataFrame()

            raw_patterns: List[Dict[str, Any]] = []
            if patterns_df is not None and not patterns_df.empty:
                for _, row in patterns_df.iterrows():
                    pat_str = str(row.iloc[0])
                    sup_val = int(row.iloc[1])
                    pat_items = pat_str.strip().split()
                    raw_patterns.append({"pattern": pat_items, "support": sup_val})

            # Sort patterns descending by support
            raw_patterns.sort(key=lambda x: x["support"], reverse=True)

            # Store full list in session.artifacts
            artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
            session.artifacts[artifact_key] = raw_patterns

            # Limit top 10 patterns in summary
            top_patterns = raw_patterns[:10]

            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_surviving_terms": len(surviving_terms),
                "total_patterns": len(raw_patterns),
                "patterns": top_patterns,
            }
            if len(raw_patterns) > 10:
                summary["note"] = f"Showing top 10 patterns out of {len(raw_patterns)} total found."
            elif len(raw_patterns) == 0:
                summary["note"] = "No patterns found matching criteria."

            session.store_result(
                "mine_patterns_tool",
                {
                    "category_name": category_name,
                    "filtering_method": filtering_method,
                    "algorithm": algorithm,
                    "min_sup": min_sup,
                    "k": k,
                },
                summary,
                full_report=pd.DataFrame(raw_patterns) if raw_patterns else None,
            )
            return summary

        finally:
            if os.path.exists(temp_db_path):
                os.remove(temp_db_path)

    return [mine_patterns_tool]
