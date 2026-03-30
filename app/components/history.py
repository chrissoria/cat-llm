"""
Run history — auto-saves results and provides a history browser.
Stores metadata in ~/Library/Application Support/CatLLM/history.json
and result files in a results/ subdirectory.
"""

import json
import os
import platform
import shutil
from datetime import datetime

import pandas as pd


def _storage_dir():
    if platform.system() == "Darwin":
        return os.path.join(os.path.expanduser("~"), "Library", "Application Support", "CatLLM")
    return os.path.join(os.path.expanduser("~"), ".catllm")


def _results_dir():
    d = os.path.join(_storage_dir(), "results")
    os.makedirs(d, exist_ok=True)
    return d


def _history_path():
    return os.path.join(_storage_dir(), "history.json")


def _load_history():
    path = _history_path()
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return []


def _save_history(entries):
    d = _storage_dir()
    os.makedirs(d, exist_ok=True)
    with open(_history_path(), "w") as f:
        json.dump(entries, f, indent=2)


def save_run(results_dict, settings=None):
    """Auto-save a run's results to disk and add to history.

    Args:
        results_dict: The st.session_state.results dict from a function page.
        settings: The current settings dict (to check auto_save_results).
    """
    if settings and not settings.get("auto_save_results", True):
        return

    task_type = results_dict.get("task_type", "unknown")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"catllm_{task_type}_{timestamp}.csv"
    filepath = os.path.join(_results_dir(), filename)

    # Save the dataframe if present
    df = results_dict.get("df")
    row_count = 0
    if df is not None and isinstance(df, pd.DataFrame):
        df.to_csv(filepath, index=False)
        row_count = len(df)
    elif task_type == "extract":
        # Extract results have counts_df
        counts_df = results_dict.get("counts_df")
        if counts_df is not None and isinstance(counts_df, pd.DataFrame):
            counts_df.to_csv(filepath, index=False)
            row_count = len(counts_df)
        else:
            # Save categories as a simple CSV
            cats = results_dict.get("categories", [])
            if cats:
                pd.DataFrame({"category": cats}).to_csv(filepath, index=False)
                row_count = len(cats)
            else:
                return
    elif task_type == "explore":
        cats = results_dict.get("categories", [])
        if cats:
            pd.DataFrame({"category": cats}).to_csv(filepath, index=False)
            row_count = len(cats)
        else:
            return
    else:
        return

    # Build models string
    models = results_dict.get("models_list", [])
    model_str = ", ".join(models) if models else "unknown"

    # Add to history
    entry = {
        "timestamp": timestamp,
        "task_type": task_type,
        "model": model_str,
        "row_count": row_count,
        "categories": results_dict.get("categories", []),
        "filename": filename,
        "filepath": filepath,
        "status": results_dict.get("status", ""),
    }

    history = _load_history()
    history.insert(0, entry)

    # Keep last 100 entries
    if len(history) > 100:
        # Delete old files
        for old in history[100:]:
            try:
                os.remove(old["filepath"])
            except OSError:
                pass
        history = history[:100]

    _save_history(history)


def get_history():
    """Return list of history entries (newest first)."""
    return _load_history()


def load_run(entry):
    """Load a saved run's results from disk. Returns a DataFrame or None."""
    filepath = entry.get("filepath", "")
    if os.path.isfile(filepath):
        try:
            return pd.read_csv(filepath)
        except Exception:
            return None
    return None


def delete_run(index):
    """Delete a history entry by index."""
    history = _load_history()
    if 0 <= index < len(history):
        entry = history.pop(index)
        try:
            os.remove(entry["filepath"])
        except OSError:
            pass
        _save_history(history)


def clear_history():
    """Delete all history entries and result files."""
    history = _load_history()
    for entry in history:
        try:
            os.remove(entry["filepath"])
        except OSError:
            pass
    _save_history([])


def render_history_sidebar():
    """Render history browser in the sidebar."""
    import streamlit as st

    history = get_history()
    if not history:
        st.sidebar.caption("No previous runs.")
        return

    for i, entry in enumerate(history[:10]):
        ts = entry.get("timestamp", "").replace("_", " ")
        task = entry.get("task_type", "?")
        rows = entry.get("row_count", 0)
        model = entry.get("model", "?")
        # Shorten model name
        short_model = model.split(",")[0].split("/")[-1][:20]

        label = f"{task} | {short_model} | {rows} rows"
        col_load, col_del = st.sidebar.columns([4, 1])
        with col_load:
            if st.button(label, key=f"hist_load_{i}", use_container_width=True,
                         help=f"{ts}\nModel: {model}"):
                df = load_run(entry)
                if df is not None:
                    st.session_state.results = {
                        "df": df,
                        "task_type": entry.get("task_type", "classify"),
                        "status": f"Loaded from history ({ts})",
                        "categories": entry.get("categories", []),
                        "models_list": [entry.get("model", "")],
                    }
                    st.rerun()
        with col_del:
            if st.button("\u2715", key=f"hist_del_{i}", help="Delete this run"):
                delete_run(i)
                st.rerun()

    if len(history) > 10:
        st.sidebar.caption(f"+ {len(history) - 10} more runs")

    if st.sidebar.button("Clear History", key="hist_clear_all"):
        clear_history()
        st.rerun()
