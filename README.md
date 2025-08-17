# Credit Card Customer Churn Prediction

## Project Overview
A major bank is experiencing customer attrition in its credit card division.  
This project aimed to build a predictive model to identify customers likely to close their credit card accounts, enabling proactive retention strategies.

We explored a synthetic dataset containing customer information, transaction history, and demographic details. The project involved:
- Data exploration and preprocessing.  
- Feature engineering to capture customer behaviors.  
- Training and evaluating classification models.  
- Providing insights and recommendations for churn reduction.  

---

## Objectives
1. **Data Understanding & Cleaning**  
   - Explored dataset and summarized key statistics.  
   - Handled missing values, outliers, duplicates, and imbalance.  
   
2. **Feature Engineering**  
   - Created features to improve model performance.  
   - Considered dimensionality reduction.  

3. **Model Development & Evaluation**  
   - Trained Logistic Regression, Random Forest, and XGBoost.  
   - Addressed class imbalance with resampling techniques.  
   - Evaluated using Accuracy, Precision, Recall, F1-score, ROC-AUC, and confusion matrix.  

4. **Insights & Recommendations**  
   - Identified dataset limitations and possible improvements.  
   - Suggested business strategies for retention.  

5. **Dashboard**  
   - Built an interactive Streamlit dashboard to show the **EDA process and model performance**.  
   - [👉 View Dashboard](https://your-dashboard-link.streamlit.app)  

---

## Notebooks
- `notebooks/dev/` → Development notebooks (no outputs, used for experimentation).  
- `notebooks/reports/` → Final notebooks with outputs (recommended for review).  

---

## Results Summary
- **Models Tested**: Logistic Regression, Random Forest, XGBoost  
- **Key Metrics**:  
  - Accuracy appeared high.  
  - Precision, Recall, and F1-score for churn were very low.  
  - ROC-AUC ~0.5, suggesting near-random performance.  
- **Finding**: The dataset lacked predictive power — models were unable to reliably predict churn.  
- **Takeaway**: Richer behavioral and transactional features are needed for meaningful churn prediction.  

---

## Project Structure
```
creditcard-churn/
├─ data/
│  ├─ raw/                  # Original dataset (excluded from repo if large)
│  └─ processed/            # Cleaned/processed data
├─ notebooks/
│  ├─ dev/                  # Development notebooks (no outputs)
│  └─ reports/              # Final notebooks with outputs
├─ reports/                 # Plots, presentations, summaries
├─ dashboard/               # Streamlit dashboard folder
│  └─ dashboard.py          # Main Streamlit app file
├─ requirements.txt         # Project dependencies
└─ README.md                # Project documentation
```

---

## Evaluation Metrics
- **Accuracy**
- **Precision, Recall, F1-score**
- **ROC-AUC**
- **Confusion Matrix**
- **Business relevance of recommendations**

---

## Tools & Libraries
- **Python**: pandas, numpy, matplotlib, seaborn  
- **Modeling**: scikit-learn, XGBoost, LightGBM  
- **Imbalance Handling**: imblearn.over_sampling  
- **Visualization**: matplotlib, seaborn, plotly  
- **Dashboard**: Streamlit  

---

## Author
**Mark Francis Masadre**  
📧 masadremarkfrancis@gmail.com  

---

## Next Steps
- Expand dataset with behavioral and transactional features.  
- Explore additional external data sources.  
- Revisit modeling with richer signals.  
- Deploy improved churn prediction models in production.  
