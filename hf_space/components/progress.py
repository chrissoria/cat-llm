"""
Progress tracking with ETA estimation.
"""

import time
import streamlit as st


class ProgressTracker:
    """Wraps st.progress + status text with ETA calculation."""

    def __init__(self, label_prefix="Processing"):
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.start_time = time.time()
        self.label_prefix = label_prefix

    def get_callback(self):
        """Return a callback function with (current_idx, total, label) signature."""
        start_time = self.start_time
        progress_bar = self.progress_bar
        status_text = self.status_text
        prefix = self.label_prefix

        def callback(current_idx, total, label=None):
            progress = current_idx / total if total > 0 else 0
            progress_bar.progress(min(progress, 1.0))

            elapsed = time.time() - start_time
            if current_idx > 0:
                avg_time = elapsed / current_idx
                eta_seconds = avg_time * (total - current_idx)
                eta_str = f" | ETA: {eta_seconds:.0f}s" if eta_seconds < 60 else f" | ETA: {eta_seconds/60:.1f}m"
            else:
                eta_str = ""

            label_str = f" ({label})" if label else ""
            status_text.text(
                f"{prefix} {current_idx + 1} of {total}{label_str} ({progress * 100:.0f}%){eta_str}"
            )

        return callback

    def complete(self, message):
        """Mark progress as complete."""
        self.progress_bar.progress(1.0)
        self.status_text.text(message)
