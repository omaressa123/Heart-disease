# ❤️ Heart Disease Prediction App

## Project Overview
This project provides a Streamlit web application for predicting the likelihood of heart disease based on various patient parameters. It leverages a RandomForestClassifier model trained on the `heart.csv` dataset. The application offers an interactive user interface to input patient data and receive instant predictions, along with a probability score.

## Features
- **Interactive Input:** User-friendly sliders and select boxes for entering patient information.
- **Real-time Prediction:** Get instant predictions on the likelihood of heart disease.
- **Probability Score:** Displays the probability of heart disease for better insights.
- **Clear Interface:** A clean and intuitive interface for easy navigation and understanding.
- **Data Preprocessing:** Handles categorical feature encoding seamlessly.

## Dataset
The model is trained on the `heart.csv` dataset, which contains 14 features related to heart health and a target variable indicating the presence or absence of heart disease.

## Model
The prediction model used is a **RandomForestClassifier** from the scikit-learn library. It has been trained and optimized to provide accurate predictions for heart disease.

## Setup and Installation

To get this project up and running on your local machine, follow these steps:

1.  **Clone the repository (if applicable):**
    ```bash
    # If this project is in a git repository
    git clone <repository_url>
    cd heart-disease-prediction
    ```

2.  **Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    *   **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Application

After setting up the environment and installing dependencies, you can run the Streamlit application:

```bash
streamlit run streamlit_app.py
```

This command will open the application in your default web browser. You can then interact with the input fields to get predictions.

## Project Structure

```
.
├── app.py                      # (Original, potentially unused, but kept for context)
├── heart-disease.ipynb         # Jupyter Notebook with EDA and model training
├── heart.csv                   # Dataset used for training and prediction
├── random_forest_model.pkl     # Trained RandomForestClassifier model
├── requirements.txt            # Python dependencies
├── streamlit_app.py            # Streamlit web application
└── README.md                   # Project README file
```

## Logo

![Heart Disease Prediction Logo](https://raw.githubusercontent.com/streamlit/streamlit/develop/docs/images/streamlit-logo-secondary-dark.png)

*(Note: This is a placeholder logo from Streamlit. You can replace it with a custom project logo.)*
