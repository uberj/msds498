import streamlit as st
from input_ui import draw_inputs
import logging
from model_loader import load_model
import pdb
import pandas as pd
import numpy as np
import shap
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import time
import pandas as pd
import streamlit as st
import logging
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import plotly.graph_objects as go
from llm_explainer import explain_prediction_with_llm
from utils import analyze_prediction, type_out_text

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set the page layout to wide
st.set_page_config(layout="wide")

# Load the model, pipeline, input schema, and metadata
model, pipeline, input_schema, metadata = load_model()

# Load secrets
anthropic_api_key = st.secrets["ANTHROPIC_API_KEY"]

# Load the AI doctor system prompt from a text file
with open("ai_doctor_system_prompt.txt", "r") as file:
    ai_doctor_system_prompt = file.read()

st.title("Heart Disease Risk Diagnosis Dashboard")

st.markdown("""
_:warning: This is not a real medical diagnosis tool. It is a demo for a proof of concept.:warning:_

Enter a patient's report to get a risk assessment for heart disease. Our AI doctor will explain the results in a way that is easy to understand.

_Click:_ **Use Sample Patient** _to see an example of how the AI doctor would explain the results._
""")

st.divider()

# Initialize SHAP JavaScript
shap.initjs()

# Create two columns for layout
col1, vline, col2 = st.columns((1, 0.05, 1))  # Adjust the width of the columns to include space for the line

# Draw inputs in the left column
with col1:
    inputs, missing_inputs = draw_inputs(metadata, input_schema, model)
    st.divider()

# Add a vertical line between the columns
with vline:
    st.markdown(
        f"""
        <style>
        .vertical-line {{
            border-left: 1px solid grey;
            height: 100vh;  /* Use viewport height to ensure it spans the column */
            position: relative;  /* Use relative positioning */
            margin: auto;  /* Center the line vertically */
        }}
        </style>
        <div class="vertical-line"></div>
        """,
        unsafe_allow_html=True
    )
# Check for missing inputs and make predictions
input_data = {col: [st.session_state[col]] for col in inputs}

data_ready = len(missing_inputs) == 0
with col1:
    if data_ready:
        st.subheader("AI Medical Dr. Explanation ✨")
        loading_placeholder = st.empty()
        loading_placeholder.markdown("Loading...")

with col2:
    if data_ready:
        risk_prediction, shap_values = analyze_prediction(pipeline, input_data, metadata)
    else:
        st.write("Please fill in all required inputs to get a prediction. Missing inputs:")
        for missing_input in missing_inputs:
            # Look up the title and description in the metadata
            title = metadata[missing_input].get('title', missing_input)
            description = metadata[missing_input].get('description', 'No description available.')
            st.write(f"- **{title}**: {description}")

# Use the typing effect function to display the explanation
with col1:
    if data_ready:
        try:
            pass
            # Use an expander to make the info collapsible
            prompt, llm_explanation = explain_prediction_with_llm(
                input_data, shap_values, metadata, anthropic_api_key, ai_doctor_system_prompt, risk_prediction
            )
            with st.expander("View AI Prompt", expanded=False):
                st.info(prompt, icon="ℹ️")
            loading_placeholder.empty()
            type_out_text(llm_explanation.text, speed=0.05)  # Adjust the speed as needed
        except Exception as e:
            loading_placeholder.empty()
            st.error("Sorry, the AI doctor is offline. Please try again later.")
            logging.error(f"Error in explain_prediction_with_llm: {e}")
