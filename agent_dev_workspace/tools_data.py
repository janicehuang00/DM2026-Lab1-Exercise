from typing import List, Optional, Any
import os
import pathlib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for server/GUI environments
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def inspect_data_tool(
        n: Optional[int] = None,
    ) -> dict:
        """Peek at the first few rows and column structure of session.dataframe.

        Read-only inspection tool. Retrieves a preview of the working dataset, truncating
        long text fields and large token lists so summary context remains compact.

        Args:
            n: Number of rows to inspect. If None, defaults to min(10, total_rows).
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session.dataframe."}

        df = session.dataframe
        n_documents = len(df)

        if n_documents == 0:
            return {
                "status": "error",
                "message": "session.dataframe is empty.",
            }

        if n is None:
            n_rows = min(10, n_documents)
        else:
            n_rows = max(1, min(n, n_documents))

        columns = list(df.columns)
        head_df = df.head(n_rows)

        preview_rows = []
        for _, row in head_df.iterrows():
            row_dict = {}
            for col in columns:
                val = row[col]
                if isinstance(val, (list, tuple)):
                    if len(val) > 10:
                        row_dict[col] = list(val[:10]) + [f"... ({len(val)} items total)"]
                    else:
                        row_dict[col] = list(val)
                elif pd.isna(val):
                    row_dict[col] = None
                elif isinstance(val, str):
                    if len(val) > 150:
                        row_dict[col] = val[:150] + "..."
                    else:
                        row_dict[col] = val
                else:
                    row_dict[col] = val
            preview_rows.append(row_dict)

        result_id = session.next_result_id("inspect_data")

        summary = {
            "result_id": result_id,
            "n_documents": n_documents,
            "columns": columns,
            "n_inspected": n_rows,
            "preview_rows": preview_rows,
        }

        session.store_result("inspect_data_tool", {"n": n}, summary)
        return summary

    @tool
    def check_missing_tool(text_column: str = "text") -> dict:
        """Check for missing, NaN, or whitespace-only text values in session.dataframe.

        Inspects the specified text column in session.dataframe, counts how many rows
        are missing, NaN, or empty strings (including pure whitespace), calculates the
        percentage of missing documents, and records the row indices of missing entries.

        Args:
            text_column: Name of the text column in session.dataframe to check.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session.dataframe."}

        if text_column not in session.dataframe.columns:
            return {
                "status": "error",
                "message": f"Column '{text_column}' not found in session.dataframe."
            }

        df = session.dataframe
        col = df[text_column]

        # Identify missing: either pd.isna or string stripped to empty
        is_missing = col.isna() | (col.astype(str).str.strip() == "")
        missing_indices = df.index[is_missing].tolist()

        n_documents = len(df)
        missing_count = len(missing_indices)
        missing_pct = float(100.0 * missing_count / n_documents) if n_documents > 0 else 0.0

        result_id = session.next_result_id("check_missing")

        summary = {
            "result_id": result_id,
            "n_documents": n_documents,
            "missing_count": missing_count,
            "missing_pct": round(missing_pct, 4),
            "missing_indices": missing_indices,
            "text_column": text_column,
        }

        session.store_result("check_missing_tool", {"text_column": text_column}, summary)
        return summary

    @tool
    def check_duplicates_tool(
        drop: bool = False,
    ) -> dict:
        """Check for (and optionally drop) duplicate documents in session.dataframe.

        Inspects session.dataframe across all columns to find exact duplicate rows.
        Calculates the count, percentage, and row indices of duplicates. If drop=True,
        removes all copies of duplicated rows using keep=False matching Master's
        X.drop_duplicates(keep=False, inplace=True) behavior.

        Args:
            drop: If True, removes all copies of duplicate rows from session.dataframe.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session.dataframe."}

        df = session.dataframe
        n_documents_before = len(df)

        # Identify extra duplicate copies across all columns (no subset)
        dup_mask = df.duplicated(keep="first")
        duplicate_indices = df.index[dup_mask].tolist()
        duplicate_count = len(duplicate_indices)
        duplicate_pct = float(100.0 * duplicate_count / n_documents_before) if n_documents_before > 0 else 0.0

        dropped = False
        n_documents_after = n_documents_before

        if drop and duplicate_count > 0:
            # Drop every copy across all columns (no subset) matching keep=False
            df.drop_duplicates(keep=False, inplace=True)
            df.reset_index(drop=True, inplace=True)
            dropped = True
            n_documents_after = len(df)

        result_id = session.next_result_id("check_duplicates")

        summary = {
            "result_id": result_id,
            "n_documents_before": n_documents_before,
            "n_documents_after": n_documents_after,
            "duplicate_count": duplicate_count,
            "duplicate_pct": round(duplicate_pct, 4),
            "duplicate_indices": duplicate_indices,
            "dropped": dropped,
        }

        session.store_result("check_duplicates_tool", {"drop": drop}, summary)
        return summary

    @tool
    def sample_data_tool(
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> dict:
        """Subsample the working DataFrame in session.dataframe.

        WARNING: This tool replaces session.dataframe in-place with the subsampled dataset.
        Downstream pipeline steps will operate on this reduced sample.

        Draws a random sample of rows from session.dataframe using pandas DataFrame.sample.
        Specify either n (row count) or frac (fraction of rows), but not both.

        Args:
            n: Number of items from axis to return. Cannot be combined with frac.
            frac: Fraction of axis items to return. Cannot be combined with n.
            random_state: Seed for the random number generator for reproducible sampling.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session.dataframe."}

        if n is not None and frac is not None:
            return {
                "status": "error",
                "message": "Please specify either 'n' or 'frac', but not both.",
            }

        df = session.dataframe
        n_documents_before = len(df)

        if n is None and frac is None:
            n = min(5, n_documents_before)

        try:
            sampled_df = df.sample(n=n, frac=frac, random_state=random_state)
        except ValueError as e:
            return {
                "status": "error",
                "message": f"Invalid sampling parameters: {str(e)}",
            }

        sample_indices = sampled_df.index.tolist()
        n_sampled = len(sampled_df)

        # Update session.dataframe with sampled rows and reset index
        session.dataframe = sampled_df.reset_index(drop=True)

        result_id = session.next_result_id("sample_data")

        summary = {
            "result_id": result_id,
            "n_documents_before": n_documents_before,
            "n_sampled": n_sampled,
            "sample_indices": sample_indices,
            "random_state": random_state,
        }

        session.store_result(
            "sample_data_tool",
            {"n": n, "frac": frac, "random_state": random_state},
            summary,
        )
        return summary

    @tool
    def tokenize_tool(
        text_column: str = "text",
    ) -> dict:
        """Tokenize document text into unigrams in session.dataframe.

        Iterates through the specified text column in session.dataframe, converts text to
        lowercase, and tokenizes it using nltk.word_tokenize. Stores the resulting token lists
        as a new 'unigrams' column in session.dataframe.

        Args:
            text_column: Name of the text column in session.dataframe to tokenize.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session.dataframe."}

        if text_column not in session.dataframe.columns:
            return {
                "status": "error",
                "message": f"Column '{text_column}' not found in session.dataframe.",
            }

        df = session.dataframe
        token_lists = []

        for text in df[text_column]:
            if pd.isna(text) or not str(text).strip():
                token_lists.append([])
            else:
                lowercased_text = str(text).lower()
                tokens = nltk.word_tokenize(lowercased_text)
                token_lists.append(tokens)

        # Store token lists in session.dataframe under 'unigrams' column
        session.dataframe["unigrams"] = token_lists

        n_documents = len(df)
        total_tokens = sum(len(t) for t in token_lists)

        result_id = session.next_result_id("tokenize")

        summary = {
            "result_id": result_id,
            "n_documents": n_documents,
            "total_tokens": total_tokens,
            "text_column": text_column,
            "sample_tokens": token_lists[:3],
        }

        session.store_result("tokenize_tool", {"text_column": text_column}, summary)
        return summary

    @tool
    def list_files_tool(
        subdirectory: str = "",
    ) -> dict:
        """List files and subdirectories within a repository folder.

        Explores the specified subdirectory relative to the repository root and separates
        entries into subdirectories and files. Prevents directory traversal outside the repo root.

        Args:
            subdirectory: Path relative to repository root to list. Defaults to root ("").
        """
        repo_root = pathlib.Path.cwd().resolve()
        target_path = (repo_root / subdirectory).resolve()

        # Prevent directory traversal attacks outside repository root
        if not target_path.is_relative_to(repo_root):
            return {
                "status": "error",
                "message": f"Access denied: Path '{subdirectory}' is outside repository root.",
            }

        if not target_path.exists() or not target_path.is_dir():
            return {
                "status": "error",
                "message": f"Directory '{subdirectory}' not found.",
            }

        directories = []
        files = []

        for item in target_path.iterdir():
            # Skip hidden files/directories and cache folders
            if item.name.startswith(".") or item.name == "__pycache__":
                continue
            if item.is_dir():
                directories.append(item.name)
            elif item.is_file():
                files.append(item.name)

        directories.sort()
        files.sort()

        result_id = session.next_result_id("list_files")

        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files,
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def describe_data_tool(
        text_column: str = "text",
    ) -> dict:
        """Compute text length statistics and build box plot by category for session.dataframe.

        Read-only inspection tool. Calculates character counts (text_length) on the specified text
        column and returns standard .describe() summary statistics overall and grouped by category.
        Generates a boxplot figure of text_length by category and sets it on session.pending_figure.

        Args:
            text_column: Name of the text column in session.dataframe to compute length statistics on.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session.dataframe."}

        if text_column not in session.dataframe.columns:
            return {
                "status": "error",
                "message": f"Column '{text_column}' not found in session.dataframe.",
            }

        df = session.dataframe

        # Calculate character count text_length without modifying session.dataframe permanently
        text_length = df[text_column].apply(lambda x: len(str(x)) if pd.notna(x) and str(x) != "" else 0)

        # Helper function to convert .describe() Series to clean rounded JSON dict
        def format_describe(desc_series: pd.Series) -> dict:
            return {
                "count": int(desc_series.get("count", 0)),
                "mean": round(float(desc_series.get("mean", 0.0)), 4),
                "std": round(float(desc_series.get("std", 0.0)), 4),
                "min": float(desc_series.get("min", 0.0)),
                "25%": float(desc_series.get("25%", 0.0)),
                "50%": float(desc_series.get("50%", 0.0)),
                "75%": float(desc_series.get("75%", 0.0)),
                "max": float(desc_series.get("max", 0.0)),
            }

        overall_stats = format_describe(text_length.describe())

        # Group by category column if available
        category_col = None
        for col in ["category_name", "category", "label"]:
            if col in df.columns:
                category_col = col
                break

        by_category_stats = {}
        if category_col is not None:
            temp_df = df.copy()
            temp_df["_temp_text_length"] = text_length
            grouped = temp_df.groupby(category_col)["_temp_text_length"]

            for cat_val, grp_series in grouped:
                by_category_stats[str(cat_val)] = format_describe(grp_series.describe())

            # Build boxplot figure grouped by category_col
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=temp_df, x=category_col, y="_temp_text_length", ax=ax)
            ax.set_title("Text Length by Category")
            ax.set_xlabel(category_col)
            ax.set_ylabel("Text Length (characters)")
            plt.tight_layout()

            # Attach figure to session.pending_figure (read-only side effect for display)
            session.pending_figure = fig

        result_id = session.next_result_id("describe_data")

        summary = {
            "result_id": result_id,
            "text_column": text_column,
            "overall_stats": overall_stats,
            "by_category_stats": by_category_stats,
        }

        session.store_result("describe_data_tool", {"text_column": text_column}, summary)
        return summary

    return [
        inspect_data_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        tokenize_tool,
        list_files_tool,
        describe_data_tool,
    ]
