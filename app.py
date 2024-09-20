import streamlit as st
import pandas as pd
from preprocessing import handle_missing_values, preprocess_data
from models import algorithm_suggestions, get_predictions
from visualization import unique_visualizations
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

def get_model(task_type):
    """
    Returns the model based on the task type.
    
    Args:
    - task_type (str): The type of task, either "regression" or "classification".
    
    Returns:
    - model: The corresponding model.
    """
    if task_type == "regression":
        st.sidebar.subheader("Select a Regression Model")
        model_choice = st.sidebar.selectbox("Choose Model", ("Linear Regression", "Random Forest", "LightGBM"))

        if model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Random Forest":
            model = RandomForestRegressor()
        elif model_choice == "LightGBM":
            model = LGBMRegressor()

    elif task_type == "classification":
        st.sidebar.subheader("Select a Classification Model")
        model_choice = st.sidebar.selectbox("Choose Model", ("Random Forest", "LightGBM"))

        if model_choice == "Random Forest":
            model = RandomForestClassifier()
        elif model_choice == "LightGBM":
            model = LGBMClassifier()

    else:
        st.write("Error: Invalid task type selected.")
        model = None

    return model

# Set page configuration
st.set_page_config(page_title="AutoML Web App", layout="wide")


def determine_task_type(df, target_column):
    if pd.api.types.is_numeric_dtype(df[target_column]):
        if df[target_column].nunique() > 10:
            st.write("This seems like a **Regression** task.")
            return 'regression'
        else:
            st.write("This could be a **Classification** task.")
            return 'classification'
    else:
        st.write("This could be a **Classification** task.")
        return 'classification'

def main():
    st.title("AutoML Task: Regression or Classification")
    st.sidebar.title("Navigation")
    st.sidebar.subheader("Upload a CSV file")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("**Data Preview:**")
        st.write(df.head())

        df = handle_missing_values(df)

        st.write("**Dataset Information:**")
        st.write(df.describe())
        st.write("**Data Types:**")
        st.write(df.dtypes)

        target_column = st.selectbox("Select the target column", df.columns)
        
        task_type = determine_task_type(df, target_column)
        
        algorithm_suggestions(df, target_column, task_type)

        if st.checkbox("Show Unique Visualizations"):
            unique_visualizations(df, target_column)

        X, y = preprocess_data(df, target_column)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = get_model(task_type)
        model.fit(X_train, y_train)
        
        get_predictions(model, df, target_column)
        # In app.py
        st.markdown("""
            <style>
            .reportview-container {
                background: #f0f2f6;
                color: #0e0e0e;
            }
            </style>
            """, unsafe_allow_html=True)



if __name__ == '__main__':
    main()
