import anthropic
import pdb
import pandas as pd
import shap
import mlflow.pyfunc
import yaml


def explain_prediction_with_llm(model, input_data, metadata, api_key) -> str:
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Extract the preprocessing steps from the pipeline
    # Assuming 'classifier' is the last step in the pipeline
    preprocessing_pipeline = model[:-1]  # Exclude the last step (classifier)

    # Transform the input data using the preprocessing steps
    preprocessed_input = preprocessing_pipeline.transform(input_df)

    # Extract the XGBoost model from the pipeline
    xgboost_model = model.named_steps["classifier"]

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

    # Prepare the prompt for Claude
    prompt = "\n\nHuman: Explain the prediction for the following input data:\n"
    for feature, value in input_data.items():
        title = (
            metadata[feature]["title"]
            if feature in metadata and "title" in metadata[feature]
            else feature
        )
        prompt += f"- {title}: {value}\n"

    probability = model.predict_proba(input_df)[0][1]
    prompt += f"\nThe model predicts a probability of heart disease of {probability}. \nThe SHAP values for each feature are:\n"
    for feature, shap_value in zip(feature_titles, shap_values[0].values):
        prompt += f"- {feature}: {shap_value:.4f}\n"

    prompt += "\n Please provide a detailed explanation of how these features contribute to the prediction."

    # Initialize the Anthropic client
    client = anthropic.Client(api_key=api_key)

    # Call the Claude API
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=300,
        system="You are a world-class heart specialist. Respond only with short explainations of the prediction of hearth disease for a patient with the given features.",
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
    # Load the API key from the .secrets file
    with open(".secrets", "r") as f:
        for line in f:
            if line.startswith("ANTHROPIC_API_KEY"):
                api_key = line.strip().split("=")[1]

    # Get explanation
    prompt, explanation = explain_prediction_with_llm(
        pipeline, example_input, metadata, api_key
    )
    print("Prompt:\n", prompt)
    print("Explanation:\n", explanation.text)
