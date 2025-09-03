# WAZE CHURN PREDICTION – Capstone Project  

## 📌 *Project Overview*  
This repository contains the final capstone project developed as part of the **Google Advanced Data Analytics Certificate** on Coursera.  

The goal of this project is to predict whether a Waze user will churn (stop using the app) or remain active, by applying different machine learning models. The full workflow of a data science project was followed, including:  
- Data discovery and understanding  
- Exploratory data analysis (EDA)  
- Hypothesis testing  
- Regression modeling  
- Machine learning model development and evaluation  

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
├── .gitignore                   # Files ignored by Git
└── .gitkeep                     # Placeholder for empty directories

---

## 📊 Data Description

The dataset (waze_dataset.csv) contains simulated information on Waze user behavior.
Features include: activity_days, driving_days, total_sessions, km_driven, inactive_days, and others.

Target variable: label (retained, churned).
⚠️ Note: This dataset is for educational use only and was provided as part of the Coursera project.