import streamlit as st
import numpy as np
import pdb
import logging

def draw_inputs(metadata, input_schema, model):
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
                    if col_metadata['type'] == 'float':
                        st.session_state[col] = float(value)
                    elif col_metadata['type'] == 'integer':
                        st.session_state[col] = int(value) if value not in (np.nan, None) else None
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
    numeric_cols = [col for col, col_metadata in filtered_metadata.items() if col_metadata['type'] in ['float', 'integer']]
    categorical_cols = [col for col, col_metadata in filtered_metadata.items() if col_metadata['type'] == 'categorical']
    boolean_cols = [col for col, col_metadata in filtered_metadata.items() if col_metadata['type'] == 'boolean']

    # Display numeric inputs
    num_cols = st.columns(3)  # Adjust the number of columns as needed
    for i, col in enumerate(numeric_cols):
        col_metadata = filtered_metadata[col]
        with num_cols[i % 3]:  # Cycle through columns
            try:
                if col_metadata['type'] == 'integer':
                    inputs[col] = st.number_input(
                        label=col_metadata['title'],
                        min_value=int(col_metadata.get('min')),
                        max_value=int(col_metadata.get('max')),
                        step=1,
                        key=col,
                        format="%d",
                        help=col_metadata['description']  # Use the help parameter for tooltips
                    )
                elif col_metadata['type'] == 'float':
                    inputs[col] = st.number_input(
                        label=col_metadata['title'],
                        min_value=col_metadata.get('min'),
                        max_value=col_metadata.get('max'),
                        step=0.01,
                        key=col,
                        format="%.2f",
                        help=col_metadata['description']  # Use the help parameter for tooltips
                    )
                else:
                    # Handle other types, e.g., string
                    inputs[col] = st.text_input(
                        label=col_metadata['title'],
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
    return inputs, valid_inputs