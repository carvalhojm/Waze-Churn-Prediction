# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import sklearn.metrics as metrics

# %%
df = pd.read_csv("../data/waze_dataset.csv")

# %%
# exploring data

print(df.shape)
print()
df.info()

# %%
df.head()

# %%
# droping "ID" column
df = df.drop("ID", axis=1)

# %%
# checking the class balance of the dependent (target) variable
print(df["label"].value_counts())
print()
df["label"].value_counts(normalize=True)

# %%
df.describe()

# %%
# creating features
df["km_per_driving_day"] = df["driven_km_drives"] / df["driving_days"]
df["km_per_driving_day"].describe()

# %%
# fixing df["km_per_driving_day"] column
df.loc[df["km_per_driving_day"]== np.inf, "km_per_driving_day"] = 0
df["km_per_driving_day"].describe()

# %%
# creating "professional_driver" new columns

df["professional_driver"] = np.where((df["drives"] >= 60) & (df["driving_days"] >= 15), 1, 0)
df["professional_driver"].head()


# %%
print(df["professional_driver"].value_counts())
print()
df.groupby(["professional_driver"])["label"].value_counts(normalize=True)

# The churn rate for professional drivers is 7.6%, while the churn rate for non-professionals is 19.9%. This seems like it could add predictive signal to the model.

# %%
# preparing variables
df.info()

# %%
# cleaning data
df = df.dropna()
df.info()

# %%
# imputing outliers
for column in ["sessions", "drives", "total_sessions", "total_navigations_fav1",
               "total_navigations_fav2", "driven_km_drives", "duration_minutes_drives"]:
    threshold = df[column].quantile(0.95)
    df.loc[df[column] > threshold, column] = threshold

df.describe()

# %%
df["label2"] = np.where(df["label"] == "churned", 1, 0)
df[["label", "label2"]].tail()

# %%
# Determining whether assumptions have been met
# No multicollinearity assumption:
corr_heatmap = df.select_dtypes(include=[np.number]).corr(method="pearson")
corr_heatmap

# %%
# ploting a correlation heatmap
plt.figure(figsize=(15,10))
sns.heatmap(corr_heatmap, vmin= -1, vmax= 1, annot= True, cmap="coolwarm")
plt.title("Correlation heatmap indicates many low correlated variables", fontsize= 18)
plt.show()


# %%
df["device2"] = np.where(df["device"] == "Android", 0, 1)
df[["device","device2"]].tail()

# %%
# building the model
# Isolating predictor variables
X = df.drop(columns= ["label", "label2", "device", "sessions", "driving_days"])

# Isolating target variable
y = df["label2"]

# %%
# Spliting the data
# Set the function's stratify parameter to y to ensure that the minority class appears in both train and test sets in the same proportion that it does in the overall dataset.
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
X_train.head()

# %%
# fiting the model

model = LogisticRegression(penalty=None, max_iter=4000)
model.fit(X_train, y_train)

# %%
model.coef_

# %%
# creating a series whose index is the column names and whose values are the coefficients in model.coef_

pd.Series(model.coef_[0], index=X.columns)

# %%
# intercept 
model.intercept_

# %%
# checking final assumption

# checking the linear relationship beatween X and the estimated log odds

training_probabilities = model.predict_proba(X_train)
training_probabilities

# %%
logit_data = X_train.copy()

# creating new "logit" column
logit_data["logit"] = [np.log(prob[1] / prob[0]) for prob in training_probabilities]
logit_data.head()

# %%
# testing the same operation with a diferent function
logit_data_test = X_train.copy()
logit_data_test["logit"] = model.decision_function(X_train)
logit_data_test.head()

# %%
sns.regplot(x="activity_days", y="logit", data=logit_data, scatter_kws={"s":2, "alpha": 0.5})
plt.title("Log-odds: activity days")
plt.show()

# %%
# Results and Evaluation
# generation predictions on X_test

y_preds = model.predict(X_test)
y_preds

# %%
# Scoring the model (accuracy) on the test data

model.score(X_test, y_test)

# %%
# Displaying results with a confusion matrix
cm = metrics.confusion_matrix(y_test, y_preds)

disp = metrics.ConfusionMatrixDisplay(confusion_matrix= cm, display_labels= ["retained", "churned"])
disp.plot()

# %%
# Calculating precision manually
precision = round((cm[1,1] / (cm[0,1] + cm[1,1])) * 100, 2)
print(precision, "%")

# %%
# Calculating recall manually
recall = round((cm[1,1] / (cm[1,0] + cm[1,1])) * 100, 2)
print(recall, "%")

# %%
# Analyzing the results 
print("Accuracy:", round(metrics.accuracy_score(y_test, y_preds) * 100,1),"%") 
print("Precision:", round(metrics.precision_score(y_test, y_preds)* 100,1),"%")
print("Recall:", round(metrics.recall_score(y_test, y_preds) * 100,1),"%")
print("F1 Score:", round(metrics.f1_score(y_test, y_preds) * 100,1),"%")

# %%
# Creating a classification report
target_labels = ["retained", "churned"]
print(metrics.classification_report(y_test, y_preds, target_names=target_labels))

# %%

# improving churned user identification. 
from imblearn.combine import SMOTEENN

smote_enn = SMOTEENN(random_state=42)
X_res, y_res = smote_enn.fit_resample(X_train, y_train)

# %%
# training new model with balanced data
model2 = LogisticRegression(max_iter=5000)
model2.fit(X_res, y_res)

# %%
pd.Series(model2.coef_[0], index=X.columns)

# %%
# intercept 
model2.intercept_

# %%
y_pred = model2.predict(X_test)
y_pred

# %%
# creating a new confusion matrix
cm = metrics.confusion_matrix(y_test, y_pred)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["retained","churned"])
disp.plot()

# %% 
# evaluating metrics
print(metrics.classification_report(y_test, y_pred, target_names=target_labels))

# %%
# Adjusting the threshold to analyze changes in scores and user churn predictions.
y_prob = model2.predict_proba(X_test)[:, 1]  
# changing the default threshold from 0.5 to 0.3
y_pred_threshold = (y_prob >= 0.3).astype(int)  

print(metrics.classification_report(y_test, y_pred_threshold, target_names=target_labels))

# %%
# evaluating precision and recall of different thresholds
precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_prob)

plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.title("Precision vs Recall for Differente Thresholds")
plt.xlabel("threshold")
plt.ylabel("score")
plt.legend()
plt.show()


# %%
# Generating a bar chart of the coefficients of the two models for a visual representation of the importance of the features of each model.

# first model
# Creating a list of (column_name, coefficient) tuples
features_importance = list(zip(X_train.columns, model.coef_[0]))
# sort the list by coefficient value
features_importance = sorted(features_importance, key=lambda x:x[1], reverse = True)
features_importance

# %%
# second model
features_importance2 = list(zip(X_res.columns, model2.coef_[0]))
features_importance2 = sorted(features_importance2, key=lambda x:x[1], reverse=True)
features_importance2

# %%
fig, axes = plt.subplots(2 , 1, figsize=(10,12))
sns.barplot(x=[x[1] for x in features_importance],
            y=[x[0] for x in features_importance],
            hue= [x[0] for x in features_importance],
            orient="h", legend=False, ax=axes[0])
axes[0].set_title("Feature importance - Fisrt model (53% churn precision & 9% recall)", fontsize=15)
axes[0].set_xlabel("churn probability", fontsize=12)

sns.barplot(x=[x[1] for x in features_importance2],
            y=[x[0] for x in features_importance2],
            hue= [x[0] for x in features_importance],
            orient="h", legend=False, ax=axes[1])
axes[1].set_title("Feature importance - Second model (24% churn precision & 89% recall)", fontsize=15)
axes[1].set_xlabel("churn probability", fontsize=12)
plt.tight_layout()
plt.show()
