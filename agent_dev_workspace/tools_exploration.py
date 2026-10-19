from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def cosine_similarity_tool(
        doc_index_1: Optional[int] = None,
        doc_index_2: Optional[int] = None,
        text_1: Optional[str] = None,
        text_2: Optional[str] = None,
    ) -> dict:
        """Compute cosine similarity between two documents in session or two text strings.

        Calculates the cosine similarity between two document vectors. If doc_index_1 and
        doc_index_2 are provided, extracts row vectors directly from session.feature_matrix.
        If text_1 and text_2 are provided, transforms the text inputs using the count_vectorizer
        stored in session.artifacts. Returns 0.0 if either document vector has zero norm.

        Args:
            doc_index_1: Optional row index of the first document in session.feature_matrix.
            doc_index_2: Optional row index of the second document in session.feature_matrix.
            text_1: Optional raw text string for the first document.
            text_2: Optional raw text string for the second document.
        """
        # Case A: Document index lookup takes precedence if both indices provided
        if doc_index_1 is not None and doc_index_2 is not None:
            if session.feature_matrix is None:
                return {
                    "status": "error",
                    "message": "No document-term matrix built in session. Call build_dtm_tool first.",
                }

            n_documents = session.feature_matrix.shape[0]
            if not (0 <= doc_index_1 < n_documents) or not (0 <= doc_index_2 < n_documents):
                return {
                    "status": "error",
                    "message": f"Document index out of bounds. Must be between 0 and {n_documents - 1}.",
                }

            vec1 = session.feature_matrix[doc_index_1]
            vec2 = session.feature_matrix[doc_index_2]

            # Extract text snippets if session.dataframe is available
            text_preview_1 = None
            text_preview_2 = None
            if session.dataframe is not None and "text" in session.dataframe.columns:
                text_preview_1 = str(session.dataframe["text"].iloc[doc_index_1])[:50]
                text_preview_2 = str(session.dataframe["text"].iloc[doc_index_2])[:50]

        # Case B: Raw text strings provided
        elif text_1 is not None and text_2 is not None:
            vectorizer = session.artifacts.get("count_vectorizer")
            if vectorizer is None:
                return {
                    "status": "error",
                    "message": "No fitted count_vectorizer found in session.artifacts. Call build_dtm_tool first.",
                }

            vec1 = vectorizer.transform([text_1])
            vec2 = vectorizer.transform([text_2])
            text_preview_1 = str(text_1)[:50]
            text_preview_2 = str(text_2)[:50]

        else:
            return {
                "status": "error",
                "message": "Please provide either both 'doc_index_1' and 'doc_index_2', or both 'text_1' and 'text_2'.",
            }

        # Check vector norms for zero-vector / empty-document edge case
        norm1 = np.linalg.norm(vec1.toarray())
        norm2 = np.linalg.norm(vec2.toarray())

        if norm1 == 0.0 or norm2 == 0.0:
            similarity = 0.0
        else:
            sim_matrix = cosine_similarity(vec1, vec2)
            similarity = float(sim_matrix[0, 0])

        result_id = session.next_result_id("cosine_similarity")

        summary = {
            "result_id": result_id,
            "doc_index_1": doc_index_1,
            "doc_index_2": doc_index_2,
            "similarity": round(similarity, 4),
            "text_preview_1": text_preview_1,
            "text_preview_2": text_preview_2,
        }

        session.store_result(
            "cosine_similarity_tool",
            {
                "doc_index_1": doc_index_1,
                "doc_index_2": doc_index_2,
                "text_1": text_1,
                "text_2": text_2,
            },
            summary,
        )
        return summary

    @tool
    def feature_correlation_matrix_tool(
        top_k: int = 20,
    ) -> dict:
        """Compute feature correlation matrix and plot a heatmap for top-variance terms.

        Identifies the top_k terms with the highest variance across all documents in
        session.feature_matrix, computes their pairwise Pearson correlation matrix, and renders
        an annotated heatmap stored in session.pending_figure.

        Args:
            top_k: Maximum number of highest-variance terms to include in correlation matrix.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "No document-term matrix built in session. Call build_dtm_tool first.",
            }

        dense_dtm = session.feature_matrix.toarray()
        feature_names = np.array(session.feature_names)

        # 1. Compute variance for each term across all documents
        variances = np.var(dense_dtm, axis=0)

        # 2. Rank terms by variance descending and take top_k
        top_indices = np.argsort(variances)[::-1][:top_k]

        selected_terms = feature_names[top_indices].tolist()
        selected_dtm = dense_dtm[:, top_indices]

        # 3. Compute Pearson correlation matrix across top terms
        # np.corrcoef expects variables in rows, so we transpose selected_dtm (docs x terms -> terms x docs)
        corr_matrix = np.corrcoef(selected_dtm.T)

        # Handle edge cases (e.g. single term selected)
        if corr_matrix.ndim == 0:
            corr_matrix = np.array([[float(corr_matrix)]])

        # Replace any potential NaNs (e.g., zero variance term) with 0.0
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Round correlation matrix for JSON output
        corr_list = [[round(float(val), 4) for val in row] for row in corr_matrix]

        # 4. Generate annotated heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            xticklabels=selected_terms,
            yticklabels=selected_terms,
            ax=ax,
            cbar=True,
            vmin=-1,
            vmax=1,
        )
        ax.set_title(f"Feature Correlation Matrix (Top {len(selected_terms)} Terms by Variance)")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("feature_correlation")

        summary = {
            "result_id": result_id,
            "terms": selected_terms,
            "correlation_matrix": corr_list,
            "n_terms": len(selected_terms),
        }

        session.store_result(
            "feature_correlation_matrix_tool",
            {"top_k": top_k},
            summary,
        )

        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
