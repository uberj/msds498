import streamlit as st
from input_ui import draw_inputs
import pdb
import pandas as pd
import numpy as np
import mlflow.pyfunc
import yaml
import logging  # Import the logging module
from tooltip_utils import generate_tooltip_html  # Import the new function
import shap  # Import SHAP
import plotly.express as px  # Import Plotly Express for plotting
import matplotlib.pyplot as plt  # Import Matplotlib for creating figures

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the page layout to wide
st.set_page_config(layout="wide")

# Load the MLflow model
model_path = "./best_model"
try:
    model = mlflow.pyfunc.load_model(model_path)
    pipeline = mlflow.sklearn.load_model(model_path)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    st.error("Failed to load the model.")
    raise e

# Get the input schema from the model
try:
    input_schema = model.metadata.get_input_schema()
    logging.info("Input schema retrieved successfully.")
except Exception as e:
    logging.error(f"Error retrieving input schema: {e}")
    st.error("Failed to retrieve input schema.")
    raise e

st.title("ML Model Prediction Dashboard")

# Load the metadata from YAML
try:
    with open('data/heart_disease_metadata.yaml', 'r') as f:
        metadata = yaml.safe_load(f)
    logging.info("Metadata loaded successfully.")
except FileNotFoundError as e:
    logging.error("Metadata YAML file not found.")
    st.error("Metadata YAML file not found.")
    raise e
except Exception as e:
    logging.error(f"An error occurred while loading the YAML file: {e}")
    st.error(f"An error occurred while loading the YAML file: {e}")
    raise e

def analyze_prediction(pipeline, input_data, metadata):
    # Convert input data to DataFrame
    input_df = pd.DataFrame(input_data)

    # Extract the preprocessing steps from the pipeline
    # Assuming 'classifier' is the last step in the pipeline
    preprocessing_pipeline = pipeline[:-1]  # Exclude the last step (classifier)

    # Transform the input data using the preprocessing steps
    preprocessed_input = preprocessing_pipeline.transform(input_df)

    # Extract the XGBoost model from the pipeline
    xgboost_model = pipeline.named_steps['classifier']

    # Initialize the SHAP TreeExplainer for the XGBoost model
    explainer = shap.TreeExplainer(xgboost_model)

    # Compute SHAP values
    shap_values = explainer(preprocessed_input)

    # Map column names to titles from metadata
    feature_names = input_df.columns
    feature_titles = [metadata[col]['title'] if col in metadata and 'title' in metadata[col] else col for col in feature_names]

    # Update the feature names in the SHAP values
    shap_values.feature_names = feature_titles

    # Draw the SHAP waterfall plot
    st.subheader("SHAP Waterfall Plot")
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

def get_color_from_probability(probability):
    # Interpolate between green (0, 255, 0) and red (255, 0, 0)
    red = int(255 * probability)
    green = int(255 * (1 - probability))
    blue = 0
    return f"#{red:02x}{green:02x}{blue:02x}"

inputs, valid_inputs = draw_inputs(metadata, input_schema, model)
if valid_inputs:
    try:
        # Prepare the input data for the model
        input_data = {col: [inputs[col]] for col in inputs}

        # Convert input data to DataFrame
        input_df = pd.DataFrame(input_data)

        # Run the model and get the prediction probabilities
        prediction_proba = pipeline.predict_proba(input_df)

        # Extract the positive class probability
        positive_class_proba = prediction_proba[0][1]  # Assuming binary classification

        # Convert to percentage
        positive_class_percentage = positive_class_proba * 100

        # Determine color based on probability
        color = get_color_from_probability(positive_class_proba)

        # Display the prediction probability for the positive class
        st.markdown(
            f"<div style='text-align: center; color: {color}; font-size: 24px;'>"
            f"Probability of Heart Disease: {positive_class_percentage:.2f}%"
            f"</div>",
            unsafe_allow_html=True
        )
        logging.info("Prediction probabilities displayed successfully.")
        analyze_prediction(pipeline, input_data, metadata)
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        st.error("Failed to generate prediction.")
        raise e
else:
    st.write("Please fill in all required inputs to get a prediction.")

