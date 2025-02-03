import streamlit as st
from input_ui import draw_inputs
import logging
from model_loader import load_model  # Import the load_model function
import pdb
import pandas as pd
import numpy as np
from tooltip_utils import generate_tooltip_html
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
ai_doctor_system_prompt = st.secrets["AI_DOCTOR_SYSTEM_PROMPT"]

def analyze_prediction(pipeline, input_data, metadata):
    try:
        input_df = pd.DataFrame(input_data)

        # Run the model and get the prediction probabilities
        prediction_proba = pipeline.predict_proba(input_df)

        # Extract the positive class probability
        positive_class_proba = prediction_proba[0][1]  # Assuming binary classification

        # Convert to percentage
        positive_class_percentage = positive_class_proba * 100

        # Determine color based on probability
        color = get_color_from_probability(positive_class_proba)

        # Create a gauge chart using Plotly
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(positive_class_percentage),
            title={'text': "Heart Disease Risk"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': color},
                   'threshold': {
                       'line': {'color': "black", 'width': 4},
                       'thickness': 0.75,
                       'value': positive_class_percentage}}))

        st.plotly_chart(fig)

        logging.info("Prediction probabilities displayed successfully.")

        # Extract the preprocessing steps from the pipeline
        # Assuming 'classifier' is the last step in the pipeline
        preprocessing_pipeline = pipeline[:-1]  # Exclude the last step (classifier)

        # Transform the input data using the preprocessing steps
        preprocessed_input = preprocessing_pipeline.transform(input_df)

        # Extract the XGBoost model from the pipeline
        xgboost_model = pipeline.named_steps["classifier"]

        # Initialize the SHAP TreeExplainer for the XGBoost model
        explainer = shap.TreeExplainer(xgboost_model)

        # Compute SHAP values
        shap_values = explainer(preprocessed_input)

        # Map column names to titles from metadata
        feature_names = input_df.columns
        feature_titles = [
            metadata[col]["title"] if col in metadata and "title" in metadata[col] else col
            for col in feature_names
        ]

        # Update the feature names in the SHAP values
        shap_values.feature_names = feature_titles
        shap_values.data = input_df.values

        # Render the SHAP force plot
        st.subheader("Risk factor contributions")
        force_plot_html = shap.force_plot(
            explainer.expected_value,
            shap_values[0, :].values,
            preprocessed_input.iloc[0, :],
            matplotlib=False,
            feature_names=feature_titles,
            link="logit",
        )
        # Convert the force plot to an HTML string
        force_plot_html = (
            f"<head>{shap.getjs()}</head><body>{force_plot_html.html()}</body>"
        )
        components.html(force_plot_html)

        # Draw the SHAP waterfall plot
        st.subheader("Risk factor contributions")
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        st.error("Failed to generate prediction.")
        raise e
    return positive_class_proba, shap_values


def get_color_from_probability(probability):
    # Interpolate between green (0, 255, 0) and red (255, 0, 0)
    red = int(255 * probability)
    green = int(255 * (1 - probability))
    blue = 0
    return f"#{red:02x}{green:02x}{blue:02x}"


# Add a function to simulate typing effect
def type_out_text(text, speed=0.05):
    placeholder = st.empty()
    typed_text = ""
    for word in text.split(" "):
        typed_text += word + " "
        placeholder.markdown(typed_text)
        time.sleep(speed)

st.title("Heart Disease Prediction Diagnosis Dashboard")

st.markdown("""
This is a dashboard for predicting heart disease based on a variety of features. 
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
        st.subheader("AI Explanation ✨")
        loading_placeholder = st.empty()
        loading_placeholder.markdown("Loading...")

with col2:
    if data_ready:
        positive_class_proba, shap_values = analyze_prediction(pipeline, input_data, metadata)
    else:
        st.write("Please fill in all required inputs to get a prediction. Missing inputs:")
        for missing_input in missing_inputs:
            st.write(f"- {missing_input}")

# Use the typing effect function to display the explanation
with col1:
    if data_ready:
        prompt, llm_explanation = explain_prediction_with_llm(input_data, shap_values, metadata, anthropic_api_key, ai_doctor_system_prompt, positive_class_proba)
        loading_placeholder.empty()
        type_out_text(llm_explanation.text, speed=0.05)  # Adjust the speed as needed
