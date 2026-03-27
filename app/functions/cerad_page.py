"""
CERAD scoring page (cat-cog only): shape selector, image upload, score display.
"""

import sys
import time
import tempfile
import streamlit as st

import catllm
from config import render_single_model_selector, get_model_source
from components.file_upload import render_image_upload


def render(domain_id, domain_panel):
    """Render the CERAD scoring page."""
    st.markdown("---")
    st.markdown("### CERAD Drawing Score")
    st.markdown("Score hand-drawn images for the CERAD cognitive assessment.")

    shape = st.selectbox("Shape", options=["circle", "diamond", "rectangles", "cube"], key="cerad_shape")

    # Image upload
    input_data, description, filename = render_image_upload(key_prefix="cerad_")

    st.markdown("### Model Selection")
    model, api_key, model_source, key_error = render_single_model_selector(key_prefix="cerad_")

    if st.button("Score Drawing", type="primary", use_container_width=True):
        if input_data is None:
            st.error("Please upload image(s) first")
            return

        if key_error:
            st.error(key_error)
            return

        try:
            with st.spinner("Scoring..."):
                result_df = catllm.cerad_drawn_score(
                    shape=shape,
                    image_input=input_data,
                    api_key=api_key,
                    user_model=model,
                    model_source=model_source,
                )

            st.success("Scoring complete!")
            st.dataframe(result_df, use_container_width=True)

            # Download
            with tempfile.NamedTemporaryFile(mode="w", suffix="_cerad_scores.csv", delete=False) as f:
                result_df.to_csv(f.name, index=False)
                with open(f.name, "rb") as fread:
                    st.download_button("Download Scores (CSV)", data=fread, file_name="cerad_scores.csv", mime="text/csv")

            # Code
            with st.expander("See the Code"):
                st.code(f"""import catllm

result = catllm.cerad_drawn_score(
    shape="{shape}",
    image_input="path/to/images/",
    api_key="YOUR_API_KEY",
    user_model="{model}"
)
print(result)
""", language="python")

        except Exception as e:
            st.error(f"Error: {str(e)}")
