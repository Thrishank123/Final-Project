import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

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
    sns.set_style("whitegrid")
