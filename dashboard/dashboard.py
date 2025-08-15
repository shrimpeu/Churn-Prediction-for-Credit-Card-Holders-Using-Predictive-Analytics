import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# Load Data
# =========================
@st.cache_data
def load_data(choice):
    if choice == "Train":
        return pd.read_csv(r"D:\Projects\Github Repo\Churn-Prediction-for-Credit-Card-Holders-Using-Predictive-Analytics\data\processed\train.csv")
    else:
        return pd.read_csv(r"D:\Projects\Github Repo\Churn-Prediction-for-Credit-Card-Holders-Using-Predictive-Analytics\data\processed\test.csv")

# =========================
# Sidebar Navigation
# =========================
st.sidebar.title("Navigation")
dataset_choice = st.sidebar.selectbox("Select Dataset", ["Train", "Test"])
page = st.sidebar.radio("Go to", ["Overview", "EDA", "Model Results"])

df = load_data(dataset_choice)

# =========================
# Overview
# =========================
if page == "Overview":
    st.title("Churn Prediction – Credit Card Holders")
    st.markdown("""
    This dashboard presents:
    - **EDA Insights** on customer data
    - **Model evaluation results**
    - **Key takeaways** on why the model could not predict churn
    """)
    st.write(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    st.write(df.head())

# =========================
# EDA Section
# =========================
elif page == "EDA":
    st.header("Exploratory Data Analysis")
    
    # Show class distribution
    st.subheader("Churn vs Non-Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="AttritionFlag", palette="Set2", ax=ax)
    ax.set_title("Class Distribution")
    st.pyplot(fig)

    # Select a categorical column to plot
    cat_cols = ['Gender', 'MaritalStatus', 'EducationLevel', 'CardType', 'Country']
    choice = st.selectbox("Select Categorical Variable", cat_cols)

    fig, ax = plt.subplots()
    sns.countplot(data=df, x=choice, hue="AttritionFlag", palette="Set2", ax=ax)
    ax.set_title(f"Churn by {choice}")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

# =========================
# Model Results Section
# =========================
elif page == "Model Results":
    st.header("Model Performance Summary")
    st.markdown("""
    - **Accuracy:** ~95% (misleading due to class imbalance)
    - **Precision / Recall / F1-score:** All 0.0
    - **ROC-AUC:** ~0.51–0.52 → close to random guessing
    
    **Interpretation:**  
    The model predicts only the majority class (non-churn).  
    No strong features in the dataset separate churn from non-churn customers.
    """)
