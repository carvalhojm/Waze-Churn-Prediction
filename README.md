# WAZE CHURN PREDICTION – Capstone Project  

## 📌 *Project Overview*  
This repository contains the final capstone project developed as part of the **Google Advanced Data Analytics Certificate** on Coursera.  

The goal of this project is to predict whether a Waze user will churn (stop using the app) or remain active, by applying different machine learning models. The full workflow of a data science project was followed, including:  
- Data discovery and understanding  
- Exploratory data analysis (EDA)  
- Hypothesis testing  
- Regression modeling  
- Machine learning model development and evaluation  

The complete document to access the integrated analysis is: [`waze_case_study.ipynb`](https://github.com/carvalhojm/waze-data-scientist-project/blob/main/waze_case_study.ipynb)

---

## 🎯 *Objectives*  
- Build predictive models to classify users as **churned** or **retained**.  
- Identify the most important behavioral factors that drive churn.  
- Provide insights and actionable recommendations for Waze to improve user retention.  

---

## 📂 *Repository Structure*  
```plaintext
├── code/                        # Python scripts for each stage
│   ├── 01_data_discovering.py   # Initial data understanding
│   ├── 02_data_analysis.py      # Exploratory data analysis
│   ├── 03_hypothesis_test.py    # Hypothesis testing
│   ├── 04_regression_model.py   # Regression modeling
│   ├── 05_ML_model_comments.py  # ML model building & evaluation
│   └── code_with_comments/      # Versions of scripts with detailed comments
│
├── data/                        
│   └── waze_dataset.csv         # Dataset provided for the capstone
│
├── waze_case_study.ipynb        # Jupyter notebook with integrated analysis
├── README.md                    # Project documentation
├── .requirements.txt            # Python dependencies
├── .gitignore                   # Files ignored by Git
└── .gitkeep                     # Placeholder for empty directories
```

---

## 📊 *Data Description*
The dataset (waze_dataset.csv) contains simulated information on Waze user behavior.
- Target variable: `label` (categorical: retained, churned).

⚠️ *Note: This dataset is for educational use only and was provided as part of the Coursera project.*

## 🔎 *Data Analysis and Machine Learning Models*

- Checked for missing values and outliers.
- Explored user activity patterns and compared churned vs. retained groups.
- Performed hypothesis tests to validate assumptions about churn behavior.
- Feature engineering applied to create useful variables for models
- The following models were implemented and compared: Logistic Regression, Random Fores and XGBoost
- Key metrics in model evaluation: Recall and Precision.
- Cross-validation applied to validate results.

## 📈 *Results & Insights*

- Best performing model: XGBoost (highest Recall Score 20%).
- Key predictors of churn: `km_per_hour`, `n_days_after_onboarding`, `percent_sessions_in_last_month`.

Logistic Regression provided interpretability, while Random Forest and XGBoost offered stronger predictive performance.

Initially, `activity_days` and drivers who drive professionally for longer distances include more direct indicators of turnover rate.

Despite all the effort, the scores were lower than ideally desired. Even using techniques to increase recall, the other indicators dropped even further.

## 💡 *Business Recommendations*

Based on the analysis, Waze could:
- Collect more personal information about users could help improve the model.
- Use the current model by adjusting the threshold to increase recall in low-investment campaigns: banners and email marketing.

## 🛠️ *Tech Stack*

- Programming Language: Python 3.13.5
- Libraries: pandas, numpy, scipy, matplotlib, seaborn, scikit-learn, imblearn, xgboost
- Environment: Jupyter Notebook / VS Code

*The versions of the libraries used are available at: `requirements.txt`*