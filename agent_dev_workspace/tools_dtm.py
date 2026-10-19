from typing import List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for server/GUI environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def build_dtm_tool(
        max_features: Optional[int] = None,
        ngram_range: List[int] = [1, 1],
        min_df: int = 1,
        max_df: float = 1.0,
    ) -> dict:
        """Build a document-term matrix (DTM) from text in session.dataframe.

        Fits a CountVectorizer on session.dataframe['text'], converts the text into a sparse
        document-term matrix, updates session state (feature_matrix, feature_names, labels,
        and vectorizer artifact), and reports matrix dimension and sparsity metrics.

        Args:
            max_features: Maximum number of terms to keep in the vocabulary, sorted by term frequency.
            ngram_range: The lower and upper boundary of the range of n-values for different n-grams to be extracted.
            min_df: Ignore terms that have a document frequency strictly lower than the given threshold.
            max_df: Ignore terms that have a document frequency strictly higher than the given threshold.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"status": "error", "message": "No dataset loaded in session.dataframe."}

        # Convert list ngram_range to tuple for CountVectorizer
        ngram_tuple = (ngram_range[0], ngram_range[1])

        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_tuple,
            min_df=min_df,
            max_df=max_df,
        )

        texts = session.dataframe["text"].fillna("").tolist()
        feature_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out().tolist()

        session.feature_matrix = feature_matrix
        session.feature_names = feature_names

        # Set session labels if label/category column exists
        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)
        elif "label" in session.dataframe.columns:
            session.set_labels(session.dataframe["label"].values)

        # Store vectorizer artifact for downstream tools (e.g. cosine_similarity_tool)
        session.artifacts["count_vectorizer"] = vectorizer

        n_documents, n_features = feature_matrix.shape
        non_zero = int(feature_matrix.nnz)
        total_elements = int(n_documents * n_features)
        sparsity_pct = float(100.0 * (1.0 - non_zero / total_elements)) if total_elements > 0 else 0.0

        result_id = session.next_result_id("dtm")

        summary = {
            "result_id": result_id,
            "n_documents": n_documents,
            "n_features": n_features,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
            "vocabulary_sample": feature_names[:10],
        }

        session.store_result("build_dtm_tool", {"max_features": max_features, "ngram_range": ngram_range}, summary)
        return summary

    @tool
    def term_frequency_tool(
        top_n: Optional[int] = 10,
    ) -> dict:
        """Aggregate term frequencies across all documents in session.feature_matrix.

        Sums occurrences of each term across all documents in the document-term matrix.
        Returns the top_n most frequent terms in the summary and stores the complete
        term frequency report for visualization.

        Args:
            top_n: Number of top terms to return in summary. Defaults to 10.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "No document-term matrix built in session. Call build_dtm_tool first.",
            }

        # Calculate sum of occurrences per term column
        counts = np.asarray(session.feature_matrix.sum(axis=0)).ravel()
        feature_names = list(session.feature_names)

        # Create sorted DataFrame for full_report
        df_report = pd.DataFrame({
            "term": feature_names,
            "frequency": [int(c) for c in counts],
        }).sort_values(by=["frequency", "term"], ascending=[False, True]).reset_index(drop=True)

        # Slice top_n for summary
        if top_n is not None and top_n > 0:
            df_summary = df_report.head(top_n)
        else:
            df_summary = df_report

        top_terms = dict(zip(df_summary["term"], df_summary["frequency"]))
        result_id = session.next_result_id("term_frequency")

        summary = {
            "result_id": result_id,
            "n_terms": len(df_report),
            "top_n": top_n,
            "top_terms": top_terms,
        }

        session.store_result(
            "term_frequency_tool",
            {"top_n": top_n},
            summary,
            full_report=df_report,
        )
        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: int = 20,
        n_documents: int = 20,
    ) -> dict:
        """Render a heatmap visualization of a slice of session.feature_matrix.

        Slices the document-term matrix to the first n_documents rows and n_terms columns in
        their existing positional order. Generates an annotated heatmap figure and sets it on
        session.pending_figure. Returns the sliced count matrix as a nested list.

        Args:
            n_terms: Number of terms (columns) to include from session.feature_names. Defaults to 20.
            n_documents: Number of documents (rows) to include from session.dataframe. Defaults to 20.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "No document-term matrix built in session. Call build_dtm_tool first.",
            }

        n_total_docs, n_total_terms = session.feature_matrix.shape

        effective_n_docs = min(n_documents, n_total_docs)
        effective_n_terms = min(n_terms, n_total_terms)

        # Slice the sparse matrix and convert slice to dense array
        sparse_slice = session.feature_matrix[0:effective_n_docs, 0:effective_n_terms]
        dense_slice = sparse_slice.toarray()

        terms_slice = list(session.feature_names)[0:effective_n_terms]
        doc_indices = list(range(effective_n_docs))

        # Build annotated heatmap figure using Seaborn
        fig, ax = plt.subplots(figsize=(min(12, max(6, effective_n_terms * 0.8)), min(10, max(4, effective_n_docs * 0.5))))
        sns.heatmap(
            dense_slice,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=terms_slice,
            yticklabels=doc_indices,
            ax=ax,
            cbar=True,
        )
        ax.set_title(f"DTM Slice Heatmap ({effective_n_docs} Docs x {effective_n_terms} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Document Index")
        plt.tight_layout()

        # Set pending figure for automated notebook rendering
        session.pending_figure = fig

        matrix_slice = dense_slice.tolist()
        result_id = session.next_result_id("dtm_heatmap")

        summary = {
            "result_id": result_id,
            "n_documents_sliced": effective_n_docs,
            "n_terms_sliced": effective_n_terms,
            "terms": terms_slice,
            "doc_indices": doc_indices,
            "matrix_slice": matrix_slice,
        }

        session.store_result(
            "dtm_heatmap_tool",
            {"n_terms": n_terms, "n_documents": n_documents},
            summary,
        )
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
