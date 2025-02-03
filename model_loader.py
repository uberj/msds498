import mlflow.pyfunc
import yaml
import logging
import streamlit as st


@st.cache_resource
def load_model(
    model_path="./best_model", metadata_path="data/heart_disease_metadata.yaml"
):
    try:
        model = mlflow.pyfunc.load_model(model_path)
        pipeline = mlflow.sklearn.load_model(model_path)
        input_schema = model.metadata.get_input_schema()
        logging.info("Model and input schema loaded successfully.")

        # Load the metadata from YAML
        with open(metadata_path, "r") as f:
            metadata = yaml.safe_load(f)
        logging.info("Metadata loaded successfully.")

        return model, pipeline, input_schema, metadata
    except FileNotFoundError as e:
        logging.error("Metadata YAML file not found.")
        st.error("Metadata YAML file not found.")
        raise e
    except Exception as e:
        logging.error(f"Error loading model, input schema, or metadata: {e}")
        st.error("Failed to load the model, input schema, or metadata.")
        raise e
