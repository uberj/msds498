import streamlit as st
import pandas as pd
import numpy as np
import mlflow.pyfunc

# Load the MLflow model
model_path = "./best_model"
model = mlflow.pyfunc.load_model(model_path)
pipeline = mlflow.sklearn.load_model(model_path)

# Get the input schema from the model
input_schema = model.metadata.get_input_schema()

# Define the form inputs based on the model's input schema

st.title("ML Model Prediction Dashboard")

# Add a button to populate the form with sample data
if st.button('Use Sample Data'):
    sample_data = model.input_example.sample(1).iloc[0].to_dict()
    for col in input_schema:
        value = sample_data[col.name]
        if isinstance(value, float) and np.isnan(value):
            st.session_state[col.name] = ''
        else:
            st.session_state[col.name] = str(value)
else:
    for col in input_schema:
        if col.name not in st.session_state:
            st.session_state[col.name] = ''

with st.form(key='input_form'):
    inputs = {}
    for col in input_schema:
        if col.type == 'long':
            inputs[col.name] = st.number_input(col.name, value=int(st.session_state[col.name]), key=col.name)
        elif col.type == 'double':
            inputs[col.name] = st.number_input(col.name, value=float(st.session_state[col.name]), key=col.name)
        elif col.type == 'boolean':
            inputs[col.name] = st.checkbox(col.name, value=st.session_state[col.name].lower() == 'true', key=col.name)
        else:
            inputs[col.name] = st.text_input(col.name, value=st.session_state[col.name], key=col.name)
    
    submit_button = st.form_submit_button(label='Run Model')

if submit_button:
    # Prepare the input data for the model
    input_data = {}
    for col in input_schema:
        value = st.session_state[col.name]
        if value == '':
            if col.type == mlflow.types.DataType.long:
                input_data[col.name] = [0]
            elif col.type == mlflow.types.DataType.double:
                input_data[col.name] = [np.nan]
            elif col.type == mlflow.types.DataType.boolean:
                input_data[col.name] = [False]
            else:
                input_data[col.name] = ['']
        else:
            if col.type == mlflow.types.DataType.long:
                input_data[col.name] = [int(value)]
            elif col.type == mlflow.types.DataType.double:
                input_data[col.name] = [float(value)]
            elif col.type == mlflow.types.DataType.boolean:
                input_data[col.name] = [value.lower() == 'true']
            else:
                input_data[col.name] = [value]

    # Convert input data to DataFrame
    input_df = pd.DataFrame(input_data)

    # Run the model and get the prediction probabilities
    prediction_proba = pipeline.predict_proba(input_df)

    # Display the prediction probabilities
    st.write("Model Prediction Probabilities:", prediction_proba)
