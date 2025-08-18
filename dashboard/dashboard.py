import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import gdown
from pathlib import Path

# =========================
# Paths
# =========================
# Local
# BASE_DIR = Path(__file__).resolve().parent.parent
# DATA_PATHS = {
#     "Train": BASE_DIR / "data" / "processed" / "train.csv",
#     "Train after SMOTE": BASE_DIR / "data" / "processed" / "train_after_smote.csv",
#     "Test": BASE_DIR / "data" / "processed" / "test.csv",
# }
# CLEANED_PATH = BASE_DIR / "data" / "processed" / "credit_card_attrition_cleaned.csv"

# Google Drive
DATA_PATHS = {
    "Train": "https://drive.google.com/uc?id=1ojNA2VOJAQazz1OQ5GTGW0gazz9AUwQm",
    "Train after SMOTE": "https://drive.google.com/uc?id=1UKu6h-0nbyRFntrrXuKdQOuImAM5kkfQ",
    "Test": "https://drive.google.com/uc?id=1FrMfClLs9vcpFXWpbzUGv1HG2y7AuvpA",
}
CLEANED_PATH = "https://drive.google.com/uc?id=1w6qehlWNA48RTGpDVS_owEw3JedfI6EG"

# =========================
# Load Data
# =========================
@st.cache_data
# Local
# def load_data(path):
#     return pd.read_csv(path)

# Google Drive
def load_data(path, filename=None):
    """
    Loads CSV from Google Drive or local path.
    If path is a Google Drive link, downloads it using gdown.
    filename: local filename to save downloaded file (required for Drive links)
    """
    if path.startswith("https://drive.google.com"):
        if filename is None:
            # Default to the last part of the URL as filename
            filename = path.split("=")[-1] + ".csv"
        gdown.download(path, filename, quiet=True)
        df = pd.read_csv(filename)
    else:
        df = pd.read_csv(path)
    return df

# =========================
# Sidebar Navigation
# =========================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "EDA", "Model Results"])

# =========================
# Overview
# =========================
if page == "Overview":
    dataset_choice = st.selectbox("Select Dataset to View", list(DATA_PATHS.keys()))
    df = load_data(DATA_PATHS[dataset_choice], filename=f"{dataset_choice}.csv")

    st.title("Churn Prediction – Credit Card Holders")
    st.markdown("""
    This dashboard presents:
    - **EDA Insights** on customer data (cleaned dataset)
    - **Model evaluation results**
    - **Key takeaways** on why the model could not predict churn
    """)
    st.write(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(50))

    # --- Class Distribution ---
    st.subheader("Churn vs Non-Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="AttritionFlag", palette="Set2", ax=ax)
    ax.set_title("Class Distribution")
    st.pyplot(fig)

# =========================
# EDA Section
# =========================
elif page == "EDA":
    df = load_data(CLEANED_PATH, filename="credit_card_attrition_cleaned.csv")

    if "AttritionFlag" not in df.columns:
        st.error("'AttritionFlag' column not found in dataset.")
    else:
        # Make target variable string for hue coloring
        df["AttritionFlag"] = df["AttritionFlag"].astype(str)

        st.title("Exploratory Data Analysis")
        st.markdown("""
        This section replicates the EDA from my notebook, showing:
        - Bivariate analysis of features and the target variable
        - Correlation analysis
        - Feature Engineering
        """)
        # Separate features by type
        continuous_features = ['Age', 'Income', 'CreditLimit', 'TotalSpend', 'Tenure', 'TotalTransactions']
        boolean_features = ['Is_Female', 'MaritalStatus_Divorced', 'MaritalStatus_Married', 
                    'MaritalStatus_Single', 'MaritalStatus_Widowed',
                    'EducationLevel_Bachelor', 'EducationLevel_High School',
                    'EducationLevel_Master', 'EducationLevel_PhD',
                    'CardType_Black', 'CardType_Gold', 'CardType_Platinum', 'CardType_Silver']
        engineered_features = [f'Feature_{i}' for i in range(50)] + ['Country_FE']

        # --- Bivariate Analysis ---
        st.subheader("1. Bivariate Analysis")

        # Continuous Features 
        st.write("1.1. Comparing continuous features against AttritionFlag.")
        rows, cols = 2, 3

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), sharex=True, sharey=False)
        axes = axes.flatten()

        for i, col in enumerate(continuous_features):
            sns.boxplot(ax=axes[i], x='AttritionFlag', y=col, data=df)
            axes[i].set_title(col, fontsize=10)
            axes[i].set_xlabel("")  # remove individual x-labels
            axes[i].set_ylabel("")  # remove individual y-labels

        # Shared axis labels
        fig.text(0.5, 0.04, 'AttritionFlag', ha='center', fontsize=12)
        fig.text(0.04, 0.5, 'Value', va='center', rotation='vertical', fontsize=12)

        plt.tight_layout(rect=[0.05, 0.05, 1, 1])

        st.pyplot(fig)
        st.markdown("""
            *For Continuous features `Age`, `Income`, `CreditLimit`, `TotalSpend`, `Tenure`, `TotalTransactions`,  there are no significant differences in the distributions of continuous features between customers who churned (AttritionFlag = 1) and those who did not (AttritionFlag = 0). This suggests that these continuous features may have limited predictive power for distinguishing churn in this dataset.*
        """) 

        # Boolean Features
        st.write("1.2. Comparing boolean features against AttritionFlag.")
        n_features = len(boolean_features)
        n_cols = 3
        n_rows = math.ceil(n_features / n_cols)

        fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), sharey=True, sharex=True)
        axes2= axes2.flatten()  # Easier indexing

        for i, col in enumerate(boolean_features):
            sns.countplot(x=col, hue='AttritionFlag', data=df, palette='Set2', ax=axes2[i])
            axes2[i].set_title(f'{col} by AttritionFlag')
            axes2[i].set_xlabel("")  # Remove X-axis label
            axes2[i].set_ylabel('Count')
            axes2[i].legend(title='AttritionFlag', labels=['No Churn (0)', 'Churn (1)'])

        # Remove empty subplots but keep last one centered if applicable
        if n_features % n_cols != 0:
            empty_plots = n_cols - (n_features % n_cols)
            for j in range(1, empty_plots + 1):
                fig2.delaxes(axes2[-j])  # Remove unused axes

        plt.tight_layout()
        st.pyplot(fig2)
        st.markdown("""
                    *Same with continuous variables, the boolean features do not show significant differences between Churn and No Churn.*
        """) 

        # Engineered Features
        st.write("1.3. Comparing engineered features against AttritionFlag.")

        n_features = len(engineered_features)
        n_cols = 3
        n_rows = math.ceil(n_features / n_cols)

        fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), sharey=False, sharex=True)
        axes3 = axes3.flatten()

        for i, col in enumerate(engineered_features):
            sns.boxplot(x='AttritionFlag', y=col, data=df, ax=axes3[i])
            axes3[i].set_title(f'{col} vs AttritionFlag')
            axes3[i].set_xlabel('')  # Remove X label
            axes3[i].set_ylabel(col)

        # Remove extra empty subplots
        if n_features % n_cols != 0:
            empty_plots = n_cols - (n_features % n_cols)
            for j in range(1, empty_plots + 1):
                fig3.delaxes(axes3[-j])

        plt.tight_layout()
        st.pyplot(fig3)
        st.markdown("""
            The bivariate analysis of the engineered features (`Feature_0` to `eature_49` and `Country_FE`) against the target variable AttritionFlag shows that the mean values of each feature are very similar between employees who stayed (`0`) and those who left (`1`).*
        """) 

        # --- Correlation Analysis ---
        st.subheader("2. Correlation Analysis")

        st.markdown("2.1. Correlation Analysis with target variable `Attrition Flag`")
        corr_method = st.selectbox("Correlation Method for 2.1.", ["pearson", "spearman"])
        corr_values = df.corr(method=corr_method)[["AttritionFlag"]].sort_values(by="AttritionFlag", ascending=False)

        fig4, ax4 = plt.subplots(figsize=(3, len(corr_values) * 0.3))
        sns.heatmap(corr_values, annot=True, cmap="coolwarm", ax=ax4)
        st.pyplot(fig4)

        st.markdown("2.2. Correlation Analysis with all features")
        corr_method1 = st.selectbox("Select correlation method for 2.2.", ["pearson", "spearman"])

        fig5, ax5 = plt.subplots(figsize=(20,16))
        corr_matrix = df.corr(method=corr_method1)

        # Plot heatmap
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, linewidths=0.5, ax=ax5)
        ax5.set_title(f"{corr_method1.capitalize()} Correlation Matrix", fontsize=16)
        ax5.set_xticklabels(ax5.get_xticklabels(), rotation=90)
        ax5.set_yticklabels(ax5.get_yticklabels(), rotation=0)

        # Show in Streamlit
        st.pyplot(fig5)

        st.markdown("""
        **Observation:**  
            *There are no notable correlations between feature-feature and feature-target relationships. The few negative correlations observed are expected, as they originate from individual columns that were one-hot encoded.*
        """)

        st.markdown("2.3. Correlation Analysis with all features including newly engineered features")

        train_df = load_data(DATA_PATHS["Train"], filename="Train.csv")

        corr_method2 = st.selectbox("Select correlation method for 2.3.", ["pearson", "spearman"])
        corr_matrix2 = train_df.corr(method=corr_method2)

        fig, ax = plt.subplots(figsize=(20,16))
        sns.heatmap(corr_matrix2, cmap='coolwarm', center=0, linewidths=0.5, ax=ax)
        ax.set_title(f'{corr_method2.capitalize()} Correlation Matrix')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        st.pyplot(fig)
        st.markdown(
            """
            There are no significant correlations between the newly engineered features and the target variable `Attrition_Flag`.  
            The observed correlations in the graph are expected, since the engineered features are derived from the original features.
            """
        )

# =========================
# Model Results
# =========================
elif page == "Model Results":
    st.header("Model Performance Summary")

    # Logistic Regression
    st.subheader("Logistic Regression")
    st.markdown("""
    - **Accuracy:** ~95% (misleading due to class imbalance)  
    - **Precision / Recall / F1-score (Churn class):** 0.0  
    - **ROC-AUC:** ~0.51 → essentially random guessing  
    
    **Interpretation:**  
    Logistic Regression failed to detect churners and predicted only the majority class (non-churn).  
    """)
    st.image(r"..\reports\figures\Model_Training_Evaluation\logistic_regression_conf_matrix.png", caption="Logistic Regression – Evaluation Results")

    # Random Forest
    st.subheader("Random Forest")
    st.markdown("""
    - **Accuracy:** ~95% (also misleading)  
    - **Precision / Recall / F1-score (Churn class):** 0.0  
    - **ROC-AUC:** ~0.52  
    
    **Interpretation:**  
    Despite being more complex, Random Forest still failed to capture useful patterns.  
    The lack of signal in features prevented the model from distinguishing churners.  
    """)
    st.image(r"..\reports\figures\Model_Training_Evaluation\random_forest_conf_matrix.png", caption="Random Forest – Evaluation Results")

    # XGBoost
    st.subheader("XGBoost")
    st.markdown("""
    - **Accuracy:** ~95% (due to imbalance)  
    - **Precision / Recall / F1-score (Churn class):** 0.0  
    - **ROC-AUC:** ~0.50
    
    **Interpretation:**  
    XGBoost, a more advanced boosting model, still performed no better than chance.  
    This confirms that the dataset lacks predictive power for churn.  
    """)
    st.image(r"..\reports\figures\Model_Training_Evaluation\xgboost_conf_matrix.png", caption="XGBoost – Evaluation Results")

    # General Conclusion
    st.markdown("""
    ---
    **Overall Conclusion:**  
    All models (Logistic Regression, Random Forest, XGBoost) failed to predict churn effectively.  
    The consistently poor performance indicates that the available features do not contain enough predictive information about customer churn.  
    Additional behavioral or engagement-related data would likely be required to improve performance.  
    """)

