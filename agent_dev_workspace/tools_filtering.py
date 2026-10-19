from typing import List, Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def variance_filter_tool(
        threshold: float = 0.0,
    ) -> dict:
        """Flag low-variance terms in session.feature_matrix for feature selection.

        Calculates the population variance for each term across all documents in the
        document-term matrix. Flags terms with variance below the threshold as removed
        and terms with variance at or above the threshold as kept. Passes a full sorted
        report to store_result for bar-chart visualization.

        Args:
            threshold: Variance threshold for feature selection. Terms with variance
                       below this threshold are flagged for removal. Defaults to 0.0.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "No document-term matrix built in session. Call build_dtm_tool first.",
            }

        # Calculate population variance (ddof=0) for each column in feature_matrix
        dense_matrix = session.feature_matrix.toarray()
        variances = np.var(dense_matrix, axis=0, ddof=0)
        feature_names = list(session.feature_names)

        # Create full_report DataFrame required by visualize_result_tool
        df_report = pd.DataFrame({
            "term": feature_names,
            "variance": [round(float(v), 6) for v in variances],
        }).sort_values(by="variance", ascending=False).reset_index(drop=True)

        kept_terms = []
        removed_terms = []

        for term, var in zip(feature_names, variances):
            if float(var) >= threshold:
                kept_terms.append(term)
            else:
                removed_terms.append(term)

        result_id = session.next_result_id("variance_filter")

        # Keep term lists concise in summary (full lists if <= 50 terms, otherwise capped sample)
        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "n_features_before": len(feature_names),
            "n_features_kept": len(kept_terms),
            "n_features_removed": len(removed_terms),
            "kept_terms": kept_terms if len(kept_terms) <= 50 else kept_terms[:50],
            "removed_terms": removed_terms if len(removed_terms) <= 50 else removed_terms[:50],
        }

        session.store_result(
            "variance_filter_tool",
            {"threshold": threshold},
            summary,
            full_report=df_report,
        )
        return summary

    @tool
    def pearson_filter_tool(
        target_class: str,
        threshold: float = 0.0,
    ) -> dict:
        """Calculate Pearson correlation between terms in session.feature_matrix and target_class.

        Computes the Pearson correlation coefficient r between each term's document counts
        and a binary indicator vector for target_class. Filters terms by absolute correlation
        |r| >= threshold.

        Args:
            target_class: Category label string in session.labels to compute correlation against.
            threshold: Minimum absolute correlation threshold |r|. Defaults to 0.0 (keeps all).
        """
        if session.feature_matrix is None or session.feature_names is None or session.labels is None:
            return {
                "status": "error",
                "message": "No document-term matrix or labels in session. Call build_dtm_tool first.",
            }

        labels = np.asarray(session.labels)
        unique_labels = set(labels)

        if target_class not in unique_labels:
            return {
                "status": "error",
                "message": f"Target class '{target_class}' not found in session.labels.",
            }

        # Binary indicator vector y for target class
        y = (labels == target_class).astype(float)
        y_std = np.std(y)

        dense_matrix = session.feature_matrix.toarray()
        feature_names = list(session.feature_names)
        pearson_rs = []

        for col_idx in range(dense_matrix.shape[1]):
            x = dense_matrix[:, col_idx]
            x_std = np.std(x)

            if x_std == 0 or y_std == 0:
                r = 0.0
            else:
                r = float(np.corrcoef(x, y)[0, 1])
                if np.isnan(r):
                    r = 0.0

            pearson_rs.append(r)

        # Create full_report sorted by absolute correlation descending
        df_report = pd.DataFrame({
            "term": feature_names,
            "pearson_r": [round(float(r), 4) for r in pearson_rs],
            "abs_r": [abs(float(r)) for r in pearson_rs],
        }).sort_values(by=["abs_r", "pearson_r"], ascending=[False, False]).drop(columns=["abs_r"]).reset_index(drop=True)

        kept_terms = []
        removed_terms = []

        for term, r in zip(feature_names, pearson_rs):
            if abs(float(r)) >= threshold:
                kept_terms.append(term)
            else:
                removed_terms.append(term)

        # Include top correlations (capped at top 50 by absolute correlation) in summary
        top_report = df_report.head(50)
        correlations_summary = dict(zip(top_report["term"], top_report["pearson_r"]))

        result_id = session.next_result_id("pearson_filter")

        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "n_features_before": len(feature_names),
            "n_features_kept": len(kept_terms),
            "n_features_removed": len(removed_terms),
            "kept_terms": kept_terms if len(kept_terms) <= 50 else kept_terms[:50],
            "removed_terms": removed_terms if len(removed_terms) <= 50 else removed_terms[:50],
            "correlations": correlations_summary,
        }

        session.store_result(
            "pearson_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=df_report,
        )
        return summary

    @tool
    def spearman_filter_tool(
        target_class: str,
        threshold: float = 0.0,
    ) -> dict:
        """Calculate Spearman rank correlation between terms in session.feature_matrix and target_class.

        Computes the Spearman rank correlation coefficient r between each term's document counts
        and a binary indicator vector for target_class. Filters terms by absolute correlation
        |r| >= threshold.

        Args:
            target_class: Category label string in session.labels to compute correlation against.
            threshold: Minimum absolute correlation threshold |r|. Defaults to 0.0 (keeps all).
        """
        if session.feature_matrix is None or session.feature_names is None or session.labels is None:
            return {
                "status": "error",
                "message": "No document-term matrix or labels in session. Call build_dtm_tool first.",
            }

        labels = np.asarray(session.labels)
        unique_labels = set(labels)

        if target_class not in unique_labels:
            return {
                "status": "error",
                "message": f"Target class '{target_class}' not found in session.labels.",
            }

        # Binary indicator vector y for target class
        y = (labels == target_class).astype(float)

        dense_matrix = session.feature_matrix.toarray()
        feature_names = list(session.feature_names)
        spearman_rs = []

        for col_idx in range(dense_matrix.shape[1]):
            x = dense_matrix[:, col_idx]

            # Calculate Spearman rank correlation using scipy.stats.spearmanr
            res = spearmanr(x, y)
            r = getattr(res, "statistic", getattr(res, "correlation", 0.0))

            if np.isnan(r):
                r = 0.0

            spearman_rs.append(float(r))

        # Create full_report sorted by absolute correlation descending
        df_report = pd.DataFrame({
            "term": feature_names,
            "spearman_r": [round(float(r), 4) for r in spearman_rs],
            "abs_r": [abs(float(r)) for r in spearman_rs],
        }).sort_values(by=["abs_r", "spearman_r"], ascending=[False, False]).drop(columns=["abs_r"]).reset_index(drop=True)

        kept_terms = []
        removed_terms = []

        for term, r in zip(feature_names, spearman_rs):
            if abs(float(r)) >= threshold:
                kept_terms.append(term)
            else:
                removed_terms.append(term)

        # Include top correlations (capped at top 50 by absolute correlation) in summary
        top_report = df_report.head(50)
        correlations_summary = dict(zip(top_report["term"], top_report["spearman_r"]))

        result_id = session.next_result_id("spearman_filter")

        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "n_features_before": len(feature_names),
            "n_features_kept": len(kept_terms),
            "n_features_removed": len(removed_terms),
            "kept_terms": kept_terms if len(kept_terms) <= 50 else kept_terms[:50],
            "removed_terms": removed_terms if len(removed_terms) <= 50 else removed_terms[:50],
            "correlations": correlations_summary,
        }

        session.store_result(
            "spearman_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=df_report,
        )
        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
