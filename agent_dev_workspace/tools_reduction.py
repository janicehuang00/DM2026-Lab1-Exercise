from typing import List, Optional, Literal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        random_state: Optional[int] = 42,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        max_summary_rows: int = 20,
    ) -> dict:
        """Reduce dimensionality of the document-term matrix using PCA, t-SNE, or UMAP.

        Applies the selected dimensionality reduction method to session.feature_matrix,
        stores the reduced coordinates in session.artifacts, and builds a 2D scatter plot
        colored by category on session.pending_figure.

        Args:
            method: Reduction method to use ('pca', 'tsne', or 'umap'). Default is 'pca'.
            n_components: Target number of dimensions (default 2).
            random_state: Random seed for reproducible output (default 42).
            perplexity: Perplexity parameter for t-SNE (default 30.0; use smaller value for tiny datasets).
            n_neighbors: Number of neighbors parameter for UMAP (default 15; use smaller value for tiny datasets).
            max_summary_rows: Maximum number of coordinate rows to include in the returned summary dict.
        """
        if session.feature_matrix is None:
            return {
                "status": "error",
                "message": "No document-term matrix built in session. Call build_dtm_tool first.",
            }

        dense_dtm = session.feature_matrix.toarray()
        method_lower = method.lower()

        # 1. Fit & transform using chosen method
        explained_variance_ratio = None

        if method_lower == "pca":
            pca = PCA(n_components=n_components, random_state=random_state)
            coords = pca.fit_transform(dense_dtm)
            explained_variance_ratio = [round(float(v), 4) for v in pca.explained_variance_ratio_]

        elif method_lower == "tsne":
            # Use default init="pca" for t-SNE matching sklearn default
            tsne = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
            )
            coords = tsne.fit_transform(dense_dtm)

        elif method_lower == "umap":
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                random_state=random_state,
            )
            coords = reducer.fit_transform(dense_dtm)

        else:
            return {
                "status": "error",
                "message": f"Unsupported method '{method}'. Supported methods are 'pca', 'tsne', 'umap'.",
            }

        # Store full coordinate matrix in artifacts (never sliced, complete dataset preserved)
        session.artifacts[f"reduced_{method_lower}"] = coords

        # 2. Build required 2D scatter plot colored by category
        category_labels = None
        if session.dataframe is not None:
            if "category_name" in session.dataframe.columns:
                category_labels = session.dataframe["category_name"].tolist()
            elif "category" in session.dataframe.columns:
                category_labels = session.dataframe["category"].tolist()

        if category_labels is None and session.labels is not None:
            category_labels = list(session.labels)

        if category_labels is None:
            category_labels = ["default"] * coords.shape[0]

        # Extract 2D coordinates for plotting
        x_coords = coords[:, 0]
        y_coords = coords[:, 1] if coords.shape[1] > 1 else np.zeros_like(x_coords)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x=x_coords,
            y=y_coords,
            hue=category_labels,
            palette="Set2",
            s=80,
            ax=ax,
        )
        ax.set_title(f"Dimensionality Reduction ({method_lower.upper()})")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        plt.tight_layout()

        # ALWAYS set session.pending_figure
        session.pending_figure = fig

        # 3. Format coordinates for summary: slice up to max_summary_rows for large datasets
        sliced_coords = coords[:max_summary_rows]
        coords_list = [[round(float(v), 4) for v in row] for row in sliced_coords]

        result_id = session.next_result_id("reduce_dimensions")

        summary = {
            "result_id": result_id,
            "method": method_lower,
            "n_components": n_components,
            "n_documents": coords.shape[0],
            "coordinates": coords_list,
            "showing_first_n": len(coords_list),
        }

        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        session.store_result(
            "reduce_dimensions_tool",
            {
                "method": method,
                "n_components": n_components,
                "random_state": random_state,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors,
            },
            summary,
        )

        return summary

    @tool
    def binarize_labels_tool() -> dict:
        """One-hot encodes category labels into a binary matrix using pd.get_dummies.

        Stores the resulting 2D binary numpy array in session.artifacts['binarized_labels']
        and class names in session.artifacts['binarized_classes'].

        Returns:
            Summary dict with result_id, n_samples, n_classes, class names, and matrix shape.
        """
        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            return {"status": "error", "message": "No dataframe or category_name column found in session."}

        cat_series = session.dataframe["category_name"]
        binarized_df = pd.get_dummies(cat_series)

        matrix = binarized_df.values.astype(int)
        classes = list(binarized_df.columns)

        session.artifacts["binarized_labels"] = matrix
        session.artifacts["binarized_classes"] = classes

        res_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": res_id,
            "status": "success",
            "n_samples": matrix.shape[0],
            "n_classes": len(classes),
            "classes": classes,
            "matrix_shape": list(matrix.shape),
            "sample": matrix[:5].tolist(),
        }

        session.store_result("binarize_labels_tool", {}, summary)
        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
