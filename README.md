# Capstone Project

This repository contains the code and resources for a machine learning capstone project. The project involves data exploration, model training, and deploying a Streamlit dashboard for making predictions using the trained model.

## Repository Structure

- `0_eda.ipynb`: Jupyter notebook for exploratory data analysis (EDA).
- `1_train_model.ipynb`: Jupyter notebook for training the machine learning model.
- `dashboard.py`: Streamlit app for making predictions using the trained model.
- `best_model/`: Directory containing the saved MLflow model.
- `data/`: Directory containing the dataset.

## Setup
## Setup
1. **Install Python 3.11**:
    Ensure you have Python 3.11 installed. You can download it from the official [Python website](https://www.python.org/downloads/).

    Alternatively, you can use `conda` to install Python 3.11:
    ```bash
    conda install python=3.11
    ```

2. **Install Dependencies**:
    Ensure you have the required dependencies installed. You can install them using:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run Exploratory Data Analysis (EDA)**:
    Open and run the `0_eda.ipynb` notebook to perform EDA on the dataset.

4. **Train the Model**:
    Open and run the `1_train_model.ipynb` notebook to train the machine learning model. The trained model will be saved in the `best_model/` directory.

5. **Run the Streamlit Dashboard**:
    To run the Streamlit dashboard, execute the following command in your terminal:
    ```bash
    streamlit run dashboard.py
    ```
    This will start the Streamlit app, and you can access it in your web browser.

## Using the Dashboard

- **Load Sample Data**: Click the "Use Sample Data" button to populate the form with sample data.
- **Input Data**: Fill in the form with the required input data.
- **Run Model**: Click the "Run Model" button to get the prediction probabilities from the model.

The dashboard will display the prediction probabilities based on the input data.

## Notes

- Ensure that the `best_model/` directory contains the trained MLflow model before running the dashboard.
- The `data/` directory should contain the dataset used for EDA and model training.

Feel free to explore and modify the code to suit your needs. If you encounter any issues, please open an issue in this repository.
