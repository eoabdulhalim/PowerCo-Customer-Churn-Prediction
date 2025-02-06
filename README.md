# PowerCo Customer Churn Prediction: EDA & Random Forest Model

## Overview

This project is a part of **The Forage Job Simulation offered by BCG X Company**, It focuses on predicting customer churn for BCG's Client PowerCo company using machine learning. It involves an **Exploratory Data Analysis (EDA)** to understand the data and uncover patterns, followed by the development of a **Random Forest model** to predict customer churn. I added an extra feature a **full deployment pipeline** using **Streamlit** for building an interactive web application that showcases the model's predictions.

## Repository Structure
The repository is organized as follows:
```plaintext
├── eda_notebook.ipynb            # Jupyter notebook for Exploratory Data Analysis
├── model_training_notebook.ipynb # Jupyter notebook for training the Random Forest model
├── model_production.py           # Python script for model pipeline
├── app.py                        # Streamlit app for model deployment
├── requirements.txt              # List of dependencies required to run the project
└── README.md                     # Project README file (this file)
```

1. **`BCGX_Data Science_EDA_notebook.ipynb`**  
   This Jupyter notebook includes the full **Exploratory Data Analysis (EDA)**. It focuses on data cleaning, and visualization allowing insights into the customer churn dataset.

2. **`model_training_notebook.ipynb`**  
   This notebook contains the code for building the **Random Forest model** for predicting customer churn. The notebook covers feature engineering, and the steps of model training, hyperparameter tuning, and model evaluation using metrics such as accuracy, precision, recall, and F1-score.
   I used **MLFlow** for version control and experiment tracking to select the best model.
   
3. **`model_production.py`**  
   This Python script contains all the essential functions for data preprocessing, feature engineering, and prediction. It also includes the complete **model pipeline** that integrates preprocessing, feature engineering, and generating predictions for unseen data.

4. **`app.py`**  
   This script is used for **deployment with Streamlit**, providing a simple web interface for users to input data and receive churn predictions in real time. It showcases how the trained model can be deployed for production use.

## Dependencies

To install all dependencies, you can create a `virtual environment` and install the required libraries via `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Project Flow

### 1. **Exploratory Data Analysis (EDA)**

The **EDA** phase involves understanding the dataset, identifying key trends, and preparing the data for modeling. This notebook includes:

- Loading and cleaning the dataset
- Visualizing key features using histograms, boxplots, and correlation matrices

### 2. **Model Training and Evaluation**

In the **Model Training** notebook, a **Random Forest Classifier** is used to predict customer churn. The steps followed include:

- Data Preprocessing: Splitting the dataset into training and test sets, encoding categorical features, and handling imbalanced target
- Feature engineering and selection
- Model Training: Using the Random Forest algorithm to train the model
- Hyperparameter Tuning: Finding the optimal hyperparameters using techniques like GridSearchCV
- Model Evaluation: Using MLFlow in assessing the model's performance using metrics such as accuracy, precision, recall, and F1-score
- Saving the trained model using **Joblib** for future use.

### 3. **Model Production Pipeline**

The `model_production.py` script includes:

- A pipeline for **preprocessing** the input data
- Function for applying feature engineering for the new data
- Function for encoding categorical features for the new data
- Function for combining all transformations into one pipeline and generating the final result as a data frame to download by the user feature engineering for the new data

The model is saved using **Joblib**, ensuring it can be loaded for making predictions in the deployment phase.

### 4. **Deployment with Streamlit**

The `app.py` file is a Streamlit-based web application that allows users to interact with the model. Users can:

- Input new customer data through a web upload option
- Submit the data to the model to receive a churn prediction for the full data
- Download the prediction results into a CSV file

To run the Streamlit app locally, use the following command:

```bash
streamlit run app.py
```

## How to Use

### Step 1: Run EDA Notebook

Start by analyzing the data through the **EDA notebook** (`eda_notebook.ipynb`). This will give you a clear understanding of the data and correlations between different features.

### Step 2: Train and Evaluate Model

Run the **Model Training notebook** (`model_training_notebook.ipynb`) to:
Make necessary adjustments to prepare the data for modeling.

- Data Preprocessing: Splitting the dataset into training and test sets, encoding categorical features, and handling imbalanced target
- Feature engineering and selection
- Train the **Random Forest model**
- Tune hyperparameters to improve performance
- Evaluate the model using metrics like accuracy, precision, recall, and F1-score

### Step 3: Deploy with Streamlit

After training the model, deploy it using **Streamlit** by running the `app.py` file. This will allow you to make predictions for new customers in a real-time interactive environment.

### Step 4: Use Model in Production

You can use the `model_production.py` script to integrate the trained model into a production pipeline or a backend API for predictions.

## Example Usage

- **Input Data (Streamlit UI)**: The user will upload CSV file containing new data with the same structure of the data which the model trained on, usage patterns, account history, etc.
- **Output**: The model will predict whether the customer is likely to churn or not, you will receive a CSV file ready to download containing Customer ID, Channel Sales, Prediction.

