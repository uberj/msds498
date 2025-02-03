import anthropic
import pdb
import pandas as pd
import shap
import mlflow.pyfunc
import yaml


def explain_prediction_with_llm(input_data, shap_values, metadata, api_key, system, positive_class_proba) -> str:
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Map column names to titles from metadata
    feature_names = input_df.columns
    feature_titles = [
        metadata[col]["title"] if col in metadata and "title" in metadata[col] else col
        for col in feature_names
    ]

    # Prepare the prompt for Claude
    prompt = "\n\nHuman: Explain the prediction for the following input data:\n"
    for feature, value in input_data.items():
        title = (
            metadata[feature]["title"]
            if feature in metadata and "title" in metadata[feature]
            else feature
        )
        prompt += f"- {title}: {value}\n"

    prompt += f"\nThe model predicts a probability of heart disease of {round(positive_class_proba, 3)}. \nThe SHAP values for each feature are:\n"
    for feature, shap_value in zip(feature_titles, shap_values[0].values):
        prompt += f"- {feature}: {shap_value:.4f}\n"

    prompt += "\n Please provide a detailed explanation of how these features contribute to the prediction."

    # Initialize the Anthropic client
    client = anthropic.Client(api_key=api_key)

    # Call the Claude API
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=300,
        system=system,
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    )

    # Return the explanation
    return prompt, response.content[0]


if __name__ == "__main__":
    # Load the model
    model_path = "./best_model"
    model = mlflow.pyfunc.load_model(model_path)
    pipeline = mlflow.sklearn.load_model(model_path)

    # Get the input schema from the model
    input_schema = model.metadata.get_input_schema()

    # Load metadata
    with open("data/heart_disease_metadata.yaml", "r") as f:
        metadata = yaml.safe_load(f)

    example_input = model.input_example.sample(1).iloc[0].to_dict()
    preprocessing_pipeline = pipeline[:-1]  # Exclude the last step (classifier)

    # Transform the input data using the preprocessing steps
    preprocessed_input = preprocessing_pipeline.transform(pd.DataFrame([example_input]))

    # Extract the XGBoost model from the pipeline
    xgboost_model = pipeline.named_steps["classifier"]

    # Initialize the SHAP TreeExplainer for the XGBoost model
    explainer = shap.TreeExplainer(xgboost_model)

    # Compute SHAP values
    shap_values = explainer(preprocessed_input)
    # Load the API key from the .secrets file
    with open(".streamlit/secrets.toml", "r") as f:
        for line in f:
            if line.startswith("ANTHROPIC_API_KEY"):
                api_key = line.strip().split("=")[1].strip('"')

    # Define the system parameter
    system = """
You are a world-class heart specialist. 
Respond only with short explanations of the prediction of heart disease 
for a patient with the given features. Give recommendations for how the patient can improve their heart health.
"""

    # Run the model and get the prediction probabilities
    prediction_proba = pipeline.predict_proba(pd.DataFrame([example_input]))

    # Extract the positive class probability
    positive_class_proba = prediction_proba[0][1]  # Assuming binary classification
    # Get explanation
    prompt, explanation = explain_prediction_with_llm(
        example_input, shap_values, metadata, api_key, system, positive_class_proba
    )
    print("Prompt:\n", prompt)
    print("Explanation:\n", explanation.text)
