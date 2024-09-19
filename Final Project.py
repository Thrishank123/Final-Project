import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# Set page configuration
st.set_page_config(page_title="AutoML Web App", layout="wide")

# Function to handle missing values
def handle_missing_values(df):
    st.sidebar.subheader("Missing Value Handling")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.select_dtypes(include=['object']).columns

    if df.isnull().sum().sum() > 0:
        st.sidebar.write("Dataset has missing values. Options to handle:")
        option = st.sidebar.radio("Select an option:", ("Fill Numeric with Mean", "Fill Numeric with Median", "Drop Rows with Missing Values", "Fill Non-Numeric with Mode", "Drop Columns with Missing Values"))

        if option == "Fill Numeric with Mean":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif option == "Fill Numeric with Median":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif option == "Drop Rows with Missing Values":
            df = df.dropna()
        elif option == "Fill Non-Numeric with Mode":
            for col in non_numeric_cols:
                df[col] = df[col].fillna(df[col].mode()[0])
        elif option == "Drop Columns with Missing Values":
            df = df.dropna(axis=1)

        st.write("Missing values handled.")
    else:
        st.sidebar.write("No missing values detected.")

    return df

# Function to encode categorical data
def encode_categorical_data(df):
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

# Function to determine the task type
def determine_task_type(df):
    target_column = st.selectbox("Select the target column:", df.columns)

    task_type = None
    if pd.api.types.is_numeric_dtype(df[target_column]):
        if df[target_column].nunique() > 10:
            st.write("This seems like a **Regression** task.")
            task_type = 'regression'
        else:
            st.write("This could be a **Classification** task.")
            task_type = 'classification'
    else:
        st.write("This could be a **Classification** task.")
        task_type = 'classification'

    return target_column, task_type

# Function to preprocess the data
def preprocess_data(df, target_column):
    df_processed = df.copy()
    df_processed = encode_categorical_data(df_processed)
    df_processed = df_processed.fillna(df_processed.mean())

    if target_column in df_processed.columns:
        X = df_processed.drop(columns=[target_column])
        y = df_processed[target_column]
    else:
        st.write(f"Error: The target column '{target_column}' is not present after preprocessing.")
        st.stop()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# Function for algorithm suggestions
def algorithm_suggestions(df, target_column, task_type):
    st.subheader("Algorithm Suggestions and Comparison")

    X, y = preprocess_data(df, target_column)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = []

    if task_type == 'regression':
        st.write("**Regression Algorithms**")

        # Linear Regression
        lin_reg = LinearRegression()
        lin_reg.fit(X_train, y_train)
        y_pred_lr = lin_reg.predict(X_test)
        lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
        results.append({"Algorithm": "Linear Regression", "Metric (RMSE)": lr_rmse})

        # Random Forest Regressor
        rf_reg = RandomForestRegressor()
        rf_reg.fit(X_train, y_train)
        y_pred_rf = rf_reg.predict(X_test)
        rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
        results.append({"Algorithm": "Random Forest", "Metric (RMSE)": rf_rmse})

    elif task_type == 'classification':
        st.write("**Classification Algorithms**")

        # Logistic Regression
        log_reg = LogisticRegression(max_iter=1000)
        log_reg.fit(X_train, y_train)
        y_pred_lr = log_reg.predict(X_test)
        lr_acc = accuracy_score(y_test, y_pred_lr)
        results.append({"Algorithm": "Logistic Regression", "Metric (Accuracy)": lr_acc})

        # Random Forest Classifier
        rf_clf = RandomForestClassifier()
        rf_clf.fit(X_train, y_train)
        y_pred_rf = rf_clf.predict(X_test)
        rf_acc = accuracy_score(y_test, y_pred_rf)
        results.append({"Algorithm": "Random Forest", "Metric (Accuracy)": rf_acc})

    st.write(pd.DataFrame(results))

# Function to visualize unique plots for each feature
def unique_visualizations(df, target_column):
    st.subheader("Unique Visualizations for Each Feature")

    for col in df.columns:
        if col != target_column:
            st.write(f"**Visualizing {col} vs {target_column}**")
            fig, ax = plt.subplots()
            if pd.api.types.is_numeric_dtype(df[col]):
                sns.scatterplot(x=df[col], y=df[target_column], ax=ax)
            else:
                sns.boxplot(x=df[col], y=df[target_column], ax=ax)
            st.pyplot(fig)

# Function to get predictions on new data
# Function to get predictions on new data
def get_predictions(model, df, target_column):
    st.sidebar.subheader("Make Predictions on New Data")

    user_input = {}
    for col in df.columns:
        if col != target_column:
            # Check if the column is numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                user_input[col] = st.sidebar.number_input(f"Enter value for {col}", value=float(df[col].mean()))
            else:
                # For non-numeric columns, create a dropdown with unique values
                unique_values = df[col].unique()
                user_input[col] = st.sidebar.selectbox(f"Select value for {col}", unique_values)

    # Convert input into a DataFrame
    input_df = pd.DataFrame([user_input])

    # Encode the categorical data (if any)
    input_df = encode_categorical_data(input_df)
    
    # Scale the data
    input_df = StandardScaler().fit_transform(input_df)

    # Make prediction using the model
    pred = model.predict(input_df)
    st.sidebar.write(f"Prediction for the input: {pred[0]}")


# Main function
def main():
    st.title("AutoML Task Suggestion: Regression, Classification, or Clustering")
    
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

        target_column, task_type = determine_task_type(df)

        if task_type != 'clustering':
            algorithm_suggestions(df, target_column, task_type)

        if st.checkbox("Show Unique Visualizations"):
            unique_visualizations(df, target_column)

        # Train the model for prediction on new data
        X, y = preprocess_data(df, target_column)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if task_type == 'regression':
            model = LinearRegression()
        else:
            model = LogisticRegression(max_iter=1000)
        
        model.fit(X_train, y_train)

        # Provide predictions on new data
        get_predictions(model, df, target_column)

if __name__ == '__main__':
    main()
