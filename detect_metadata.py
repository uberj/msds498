import pandas as pd
import yaml

def load_heart_disease_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"An error occurred while loading the data: {e}")

def save_metadata_to_yaml(data, file_path):
    # Define titles and descriptions for each column
    titles_and_descriptions = {
        'age': ('Age', 'Age of the patient'),
        'sex': ('Sex', 'Sex of the patient (Male/Female)'),
        'cp': ('Chest Pain Type', 'Chest pain type'),
        'trestbps': ('Resting Blood Pressure', 'Resting blood pressure (in mm Hg)'),
        'chol': ('Serum Cholesterol', 'Serum cholesterol in mg/dl'),
        'fbs': ('Fasting Blood Sugar', 'Fasting blood sugar > 120 mg/dl (True/False)'),
        'restecg': ('Resting ECG Results', 'Resting electrocardiographic results'),
        'thalch': ('Max Heart Rate Achieved', 'Maximum heart rate achieved'),
        'exang': ('Exercise Induced Angina', 'Exercise induced angina (True/False)'),
        'oldpeak': ('ST Depression', 'ST depression induced by exercise relative to rest'),
        'slope': ('Slope of ST Segment', 'Slope of the peak exercise ST segment'),
        'ca': ('Number of Major Vessels', 'Number of major vessels (0-3) colored by fluoroscopy'),
        'thal': ('Thalassemia', 'Thalassemia (normal, fixed defect, reversible defect)'),
        'dataset': ('Hospital', 'No description available')
    }

    metadata = {}
    for col in data.columns:
        if col == 'num':  # Skip the 'num' column
            continue
        title, description = titles_and_descriptions.get(col, (col, 'No description available'))
        is_missing_allowed = bool(data[col].isnull().any())  # Check if the column has missing values

        if pd.api.types.is_bool_dtype(data[col]) or set(data[col].dropna().unique()) <= {True, False}:
            metadata[col] = {
                'type': 'boolean',
                'title': title,
                'description': description,
                'missing_allowed': is_missing_allowed
            }
        elif pd.api.types.is_numeric_dtype(data[col]):
            metadata[col] = {
                'type': 'numeric',
                'min': float(data[col].min()),
                'max': float(data[col].max()),
                'title': title,
                'description': description,
                'missing_allowed': is_missing_allowed
            }
        else:
            metadata[col] = {
                'type': 'categorical',
                'values': data[col].dropna().astype(str).unique().tolist(),
                'title': title,
                'description': description,
                'missing_allowed': is_missing_allowed
            }
    
    try:
        with open(file_path, 'w') as f:
            yaml.dump(metadata, f)
        print(f"Metadata successfully saved to {file_path}")
    except Exception as e:
        raise Exception(f"An error occurred while saving metadata to YAML: {e}")

if __name__ == "__main__":
    try:
        # Load the heart disease dataset
        heart_disease_data = load_heart_disease_data('data/heart_disease_uci.csv')
        
        if heart_disease_data is not None:
            # Save the metadata to a YAML file
            save_metadata_to_yaml(heart_disease_data, 'data/heart_disease_metadata.yaml')
    except Exception as e:
        print(e) 