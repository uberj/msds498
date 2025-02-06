import streamlit as st
import logging
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import plotly.graph_objects as go
import pandas as pd
import time

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
            title={'text': "Heart Disease Risk", 'font': {'size': 44}},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': color, 'thickness': 0.2},
                   'threshold': {
                       'line': {'color': "black", 'width': 4},
                       'thickness': 0.75,
                       'value': positive_class_percentage}}))

        # Update layout to make the indicator 1/3 the size
        fig.update_layout(
            autosize=False,  # Disable autosize to use fixed width and height
        )

        # Create three columns and place the chart in the middle column
        _, col2, _ = st.columns((0.2, 0.6, 0.2))
        with col2:
            st.plotly_chart(fig, use_container_width=True)  # Use container width to help center

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

        st.subheader("Factor Ranks")
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values, max_display=12, ax=ax)
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