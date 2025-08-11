# Credit Card Customer Churn Prediction

## 📌 Project Overview
A major bank is experiencing customer attrition in its credit card division.  
This project aims to build a predictive model to identify customers likely to close their credit card accounts, enabling proactive retention strategies.

Using a synthetic dataset containing customer information, transaction history, and demographic details, we will:
- Perform data exploration and preprocessing.
- Engineer features to capture key customer behaviors.
- Train multiple classification models to predict churn.
- Evaluate models using relevant business and statistical metrics.
- Provide insights and actionable recommendations for retention.

---

## 🎯 Objectives
1. **Data Understanding & Cleaning**  
   - Explore dataset and summarize key statistics.  
   - Handle missing values, outliers, duplicates, high cardinality, and imbalance.
   
2. **Feature Engineering**  
   - Create meaningful features to improve model performance.  
   - Apply dimensionality reduction if necessary.

3. **Model Development & Evaluation**  
   - Train and compare at least two classification models.  
   - Handle class imbalance via SMOTE or class weights.  
   - Perform hyperparameter tuning and cross-validation.  
   - Evaluate using Accuracy, Precision, Recall, F1-score, ROC-AUC, and confusion matrix.

4. **Insights & Recommendations**  
   - Identify key drivers of attrition.  
   - Provide actionable strategies for the bank.

5. **(Optional Bonus)**  
   - Build a Streamlit dashboard to visualize attrition risk.  
   - Deploy model demo.

---

## 🗂 Project Structure
creditcard-churn/
├─ data/
│  ├─ raw/              # Original dataset (excluded from repo if large)
│  └─ processed/        # Cleaned/processed data
├─ notebooks/           # Jupyter notebooks for EDA & modeling
├─ src/                 # Python scripts for data prep, features, models
├─ models/              # Saved trained models
├─ reports/             # Plots, presentations, summaries
├─ requirements.txt     # Project dependencies
└─ README.md            # Project documentation

---

## ⚙️ Setup Instructions

### 1. Clone Repository
Using **GitHub Desktop**:
1. Create the repo on GitHub and clone it locally.
2. Open the project folder in your code editor.

### 2. Create Virtual Environment
python -m venv .venv
# Activate:
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

### 3. Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

### 4. Link Environment to Jupyter
python -m ipykernel install --user --name creditcard-env --display-name "creditcard-env"

---

## 📊 Deliverables
- Cleaned dataset and data dictionary.
- EDA and insights.
- Engineered features and preprocessing pipeline.
- Trained models with evaluation results.
- Feature importance analysis and key churn drivers.
- Recommendations for retention.
- (Optional) Interactive dashboard.

---

## 📈 Evaluation Metrics
- **Accuracy**
- **Precision, Recall, F1-score**
- **ROC-AUC**
- **Confusion Matrix**
- **Business relevance of recommendations**

---

## 🛠 Tools & Libraries
- **Python**: pandas, numpy, matplotlib, seaborn
- **Modeling**: scikit-learn, XGBoost, LightGBM
- **Imbalance Handling**: imbalanced-learn
- **Interpretability**: shap
- **Visualization**: matplotlib, seaborn, plotly
- **Dashboard (optional)**: Streamlit

---

## 📅 Timeline (5 Days)
| Day | Task |
|-----|------|
| 1 | Setup, initial EDA |
| 2 | Data cleaning, handling imbalance |
| 3 | Feature engineering, baseline models |
| 4 | Model tuning, evaluation |
| 5 | Insights, recommendations, presentation, optional dashboard |

---

## 👤 Author
Mark Francis Masadre
*Contact: masadremarkfrancis@gmail.com*  

---

## 📊 Results Summary (To be filled after project completion)
- **Best Model**: _Model Name_
- **Key Metrics**: _Accuracy, Precision, Recall, F1, ROC-AUC_
- **Top Drivers of Churn**: _List features_
- **Business Impact**: _Brief description_

---

## 🚀 Next Steps
- Expand dataset with additional behavioral variables.
- Deploy model to production environment.
- Integrate with customer retention campaign systems.
