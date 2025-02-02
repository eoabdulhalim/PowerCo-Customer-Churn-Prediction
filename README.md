# PowerCo-Customer-Churn-Prediction

Here's a professional README for your project repository:

---

# Customer Churn Prediction: EDA & Random Forest Model

## Overview

This project focuses on predicting customer churn for a PowerCo company using machine learning techniques. It involves an **Exploratory Data Analysis (EDA)** to understand the data and uncover patterns, followed by the development of a **Random Forest model** to predict customer churn. The project also includes a **full deployment pipeline** using **Streamlit** for building an interactive web application that showcases the model's predictions.

## Project Structure

The repository is organized into the following files:

1. **`eda_notebook.ipynb`**  
   This Jupyter notebook includes the full **Exploratory Data Analysis (EDA)**. It focuses on data cleaning, visualization, and feature engineering, allowing insights into the customer churn dataset.

2. **`model_training_notebook.ipynb`**  
   This notebook contains the code for building the **Random Forest model** for predicting customer churn. The notebook covers the steps of model training, hyperparameter tuning, and model evaluation using metrics such as accuracy, precision, recall, and F1-score.

3. **`model_production.py`**  
   This Python script contains all the essential functions for model training, evaluation, and prediction. It also includes the complete **model pipeline** that integrates preprocessing, model training, and generating predictions for unseen data.

4. **`app.py`**  
   This script is used for **deployment with Streamlit**, providing a simple web interface for users to input data and receive churn predictions in real time. It showcases how the trained model can be deployed for production use.

## Dependencies

This project requires the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `streamlit`
- `joblib`
- `mlflow` (Optional, for model tracking)
- `pickle` (Optional, for model serialization)
  
To install all dependencies, you can create a `virtual environment` and install the required libraries via `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Project Flow

### 1. **Exploratory Data Analysis (EDA)**

The **EDA** phase involves understanding the dataset, identifying key trends, and preparing the data for modeling. This notebook includes:

- Loading and cleaning the dataset
- Visualizing key features using histograms, boxplots, and correlation matrices
- Identifying any missing or outlier values
- Feature engineering and selection

### 2. **Model Training and Evaluation**

In the **Model Training** notebook, a **Random Forest Classifier** is used to predict customer churn. The steps followed include:

- Data Preprocessing: Splitting the dataset into training and test sets, handling missing values, and scaling the features
- Model Training: Using the Random Forest algorithm to train the model
- Hyperparameter Tuning: Finding the optimal hyperparameters using techniques like GridSearchCV
- Model Evaluation: Assessing the model's performance using metrics such as accuracy, precision, recall, and F1-score
- Saving the trained model using **Joblib** for future use.

### 3. **Model Production Pipeline**

The `model_production.py` script includes:

- A pipeline for **preprocessing** the input data
- Functions for **training** the Random Forest model
- Methods for **evaluating** the model's performance
- **Saving and loading** the trained model for production use

The model is saved using **Joblib** or **Pickle**, ensuring it can be loaded for making predictions in the deployment phase.

### 4. **Deployment with Streamlit**

The `app.py` file is a Streamlit-based web application that allows users to interact with the model. Users can:

- Input customer data through a web form
- Submit the data to the model to receive a churn prediction
- View the prediction results and associated probabilities

To run the Streamlit app locally, use the following command:

```bash
streamlit run app.py
```

## How to Use

### Step 1: Run EDA Notebook

Start by analyzing the data through the **EDA notebook** (`eda_notebook.ipynb`). This will give you a clear understanding of the data, missing values, and correlations between different features. Make necessary adjustments to prepare the data for modeling.

### Step 2: Train and Evaluate Model

Run the **Model Training notebook** (`model_training_notebook.ipynb`) to:

- Train the **Random Forest model**
- Tune hyperparameters to improve performance
- Evaluate the model using metrics like accuracy, precision, recall, and F1-score

### Step 3: Deploy with Streamlit

After training the model, deploy it using **Streamlit** by running the `app.py` file. This will allow you to make predictions for new customers in a real-time interactive environment.

### Step 4: Use Model in Production

You can use the `model_production.py` script to integrate the trained model into a production pipeline or a backend API for predictions.

## Example Usage

- **Input Data (Streamlit UI)**: The user will input features like customer details, usage patterns, account history, etc.
- **Output**: The model will predict whether the customer is likely to churn or not, along with the probability of churn.

## Future Improvements

- **Model Improvements**: Experiment with other algorithms such as Gradient Boosting or XGBoost to compare results.
- **Hyperparameter Optimization**: Implement more advanced hyperparameter tuning techniques such as RandomizedSearchCV or Bayesian Optimization.
- **Model Versioning**: Use **MLflow** for better model tracking and versioning to keep track of the performance of different models.
- **Model Interpretability**: Use tools like SHAP or LIME to explain model predictions and improve user trust in the deployed solution.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to customize the README further based on specific needs or additional features that your project may include! Let me know if you need further adjustments or details!
