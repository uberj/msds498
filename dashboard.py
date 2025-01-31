import streamlit as st
import pdb
import pandas as pd
import numpy as np
import mlflow.pyfunc
import yaml
import logging  # Import the logging module
from tooltip_utils import generate_tooltip_html  # Import the new function

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

# Filter out the 'id' column from metadata
filtered_metadata = {col: col_metadata for col, col_metadata in metadata.items() if col != 'id'}
boolean_cols = [col for col, col_metadata in filtered_metadata.items() if col_metadata['type'] == 'boolean']    

# Initialize session state with default values if not already set
for col in input_schema:
    if col.name not in st.session_state:
        # Set default values based on the type of input
        if col.type == 'float' or col.type == 'int':
            st.session_state[col.name] = 0  # or any other default numeric value
        elif col.type == 'str':
            st.session_state[col.name] = ''  # default empty string for text inputs
        elif col.name in boolean_cols:  # Check if the column is a boolean
            st.session_state[col.name] = False  # Default to False for boolean inputs
        else:
            st.session_state[col.name] = None  # or any other appropriate default

# Add a button to populate the form with sample data
if st.button('Use Sample Data'):
    try:
        sample_data = model.input_example.sample(1).iloc[0].to_dict()
        for col, col_metadata in filtered_metadata.items():
            value = sample_data.get(col, None)
            if value is not None:
                if col_metadata['type'] == 'numeric':
                    st.session_state[col] = float(value)
                elif col_metadata['type'] == 'categorical':
                    st.session_state[col] = value
                elif col_metadata['type'] == 'boolean':
                    st.session_state[col] = bool(value)
        logging.info("Sample data loaded into session state.")
    except Exception as e:
        logging.error(f"Error loading sample data: {e}")
        st.error("Failed to load sample data.")
        raise e

# Initialize inputs dictionary
inputs = {}

# Group inputs by type
numeric_cols = [col for col, col_metadata in filtered_metadata.items() if col_metadata['type'] == 'numeric']
categorical_cols = [col for col, col_metadata in filtered_metadata.items() if col_metadata['type'] == 'categorical']
boolean_cols = [col for col, col_metadata in filtered_metadata.items() if col_metadata['type'] == 'boolean']

# Display numeric inputs
num_cols = st.columns(3)  # Adjust the number of columns as needed
for i, col in enumerate(numeric_cols):
    col_metadata = filtered_metadata[col]
    with num_cols[i % 3]:  # Cycle through columns
        try:
            inputs[col] = st.number_input(
                col_metadata['title'],
                min_value=col_metadata['min'],
                max_value=col_metadata['max'],
                value=st.session_state[col],  # Use session state value
                key=col,
                help=col_metadata['description']  # Use the help parameter for tooltips
            )
            logging.debug(f"Number input for {col} set with value {st.session_state[col]}.")
        except Exception as e:
            logging.error(f"Error setting number input for {col}: {e}")
            st.error(f"Error setting number input for {col}.")
            raise e

# Display categorical inputs
cat_cols = st.columns(3)  # Adjust the number of columns as needed
for i, col in enumerate(categorical_cols):
    col_metadata = filtered_metadata[col]
    with cat_cols[i % 3]:  # Cycle through columns
        try:
            inputs[col] = st.selectbox(
                col_metadata['title'],
                options=col_metadata['values'],
                index=col_metadata['values'].index(st.session_state[col]) if st.session_state[col] in col_metadata['values'] else 0,
                key=col,
                help=col_metadata['description']  # Use the help parameter for tooltips
            )
            logging.debug(f"Selectbox for {col} set with value {st.session_state[col]}.")
        except Exception as e:
            logging.error(f"Error setting selectbox for {col}: {e}")
            st.error(f"Error setting selectbox for {col}.")
            raise e

# Display boolean inputs
bool_cols = st.columns(3)  # Adjust the number of columns as needed
for i, col in enumerate(boolean_cols):
    col_metadata = filtered_metadata[col]
    with bool_cols[i % 3]:  # Cycle through columns
        try:
            inputs[col] = st.checkbox(
                col_metadata['title'],
                value=st.session_state[col],  # Use session state or default to False
                key=col,
                help=col_metadata['description']  # Use the help parameter for tooltips
            )
            logging.debug(f"Checkbox for {col} set with value {st.session_state[col]}.")
        except Exception as e:
            logging.error(f"Error setting checkbox for {col}: {e}")
            st.error(f"Error setting checkbox for {col}.")
            raise e

# Validate inputs
valid_inputs = True
for col, col_metadata in filtered_metadata.items():
    if inputs[col] is None and not col_metadata.get('missing_allowed', False):
        valid_inputs = False
        logging.warning(f"Input for {col} is None and missing is not allowed.")
        break
    # Add more validation logic as needed for other types

if valid_inputs:
    try:
        # Prepare the input data for the model
        input_data = {col: [inputs[col]] for col in inputs}

        # Convert input data to DataFrame
        input_df = pd.DataFrame(input_data)

        # Run the model and get the prediction probabilities
        prediction_proba = pipeline.predict_proba(input_df)

        # Display the prediction probabilities
        st.write("Model Prediction Probabilities:", prediction_proba)
        logging.info("Prediction probabilities displayed successfully.")
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        st.error("Failed to generate prediction.")
        raise e
else:
    st.write("Please fill in all required inputs to get a prediction.")

