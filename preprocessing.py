from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import streamlit as st
import pandas as pd

# Strategy Pattern for Missing Value Handling
def fill_numeric_with_mean(df, numeric_cols):
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    return df

def fill_numeric_with_median(df, numeric_cols):
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df

def drop_rows(df):
    return df.dropna()

def fill_non_numeric_with_mode(df, non_numeric_cols):
    for col in non_numeric_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df

def drop_columns(df):
    return df.dropna(axis=1)

# Main function for handling missing values using Strategy Pattern
def handle_missing_values(df):
    st.sidebar.subheader("Missing Value Handling")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.select_dtypes(include=['object']).columns

    strategies = {
        "Fill Numeric with Mean": fill_numeric_with_mean,
        "Fill Numeric with Median": fill_numeric_with_median,
        "Drop Rows with Missing Values": drop_rows,
        "Fill Non-Numeric with Mode": fill_non_numeric_with_mode,
        "Drop Columns with Missing Values": drop_columns,
    }

    if df.isnull().sum().sum() > 0:
        st.sidebar.write("Dataset has missing values. Options to handle:")
        option = st.sidebar.radio("Select an option:", list(strategies.keys()))

        # Apply the selected strategy
        if option in strategies:
            if 'Numeric' in option:
                df = strategies[option](df, numeric_cols)
            else:
                df = strategies[option](df, non_numeric_cols)

        st.write("Missing values handled.")
    else:
        st.sidebar.write("No missing values detected.")

    return df

def preprocess_data(df, target_column):
    df_processed = df.copy()
    numeric_cols = []
    categorical_cols = []
    columns = df.columns
    
    # Separate categorical and numeric columns
    for column in columns:
        if df[column].dtype == "object" and column != target_column:
            categorical_cols.append(column)
        else:
            numeric_cols.append(column)

    # Handle missing values using strategy
    df_processed = handle_missing_values(df_processed)

    # Apply Label Encoding before one-hot encoding to avoid key errors
    le = LabelEncoder()
    for col in categorical_cols:
        df_processed[col] = le.fit_transform(df_processed[col])

    # One-hot encoding for the categorical columns, excluding target_column
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)

    # Handle missing values for numeric columns
    df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].mean())

    # Ensure target column is available and split data
    if target_column in df_processed.columns:
        X = df_processed.drop(columns=[target_column])  # Drop target column from features
        y = df_processed[target_column]  # Assign target column to 'y'
    else:
        st.write(f"Error: The target column '{target_column}' is not present after preprocessing.")
        st.stop()

    # Standard Scaling for numeric columns
    if any(col in X.columns for col in numeric_cols):
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    else:
        st.write("Warning: No numeric columns found for scaling.")

    st.write("Available columns after preprocessing:", X.columns)

    return X, y
