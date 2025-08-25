# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import sklearn.metrics as metrics
# from sklearn.metrics import classification_report, accuracy_score, precision_score, \
# recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# %%
df = pd.read_csv("data/waze_dataset.csv")

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

# Based on the data dictionary, we can assume that the observations are independent for this project.

# No extreme outliers has already been addressed before.

# No multicollinearity assumption:

corr_heatmap = df.select_dtypes(include=[np.number]).corr(method="pearson")
corr_heatmap
# %%
# ploting a correlation heatmap
plt.figure(figsize=(15,10))
sns.heatmap(corr_heatmap, vmin= -1, vmax= 1, annot= True, cmap="coolwarm")
plt.title("Correlation heatmap indicates many low correlated variables", fontsize= 18)
plt.show()

# Setting the limit to 0.7, two variables presented multicollinearity:
# - sessions and drives: 1.0
# - driving_days and activity_days: 0.95

# %%
df["device2"] = np.where(df["device"] == "Android", 0, 1)
df[["device","device2"]].tail()

# %%
# another way of making it with OneHotEncoder
# df["device2"] = OneHotEncoder(drop="first").fit_transform(df[["device"]]).toarray()
# df[["device","device2"]].tail()
# with pandas
# device_dummies = pd.get_dummies(df["device"], prefix="device", drop_first=True)
# df = pd.concat([df, device_dummies], axis=1)

# %%
# building the model

# droping columns that are not usefull:
# - label (this is the target)
# - label2 (this is the target)
# - device (this is the non-binary-encoded categorical variable)
# - sessions (this had high multicollinearity)
# - driving_days (this had high multicollinearity)

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

# As seen in the graph, as activity_days increases, the log-odds decrease approximately linearly, meaning that more days of activity are associated with a lower chance of the positive class.

# The scattered pattern of the individual observations confirms consistency with the linearity hypothesis.
# CONCERTAR ESSE FINAL
# As days of activity increase, the probability of being satisfied decreases dramatically. In other words, very active users have a lower chance of being in the positive class (e.g., "satisfied").

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

# - The model is heavily biased toward the "retained" class, likely because it has higher numbers than churns.
# - True Retained (TN): 2,889
# - False Churns (FP): 52
# - False Retained (FN): 576
# - True Churns (TP): 58
# - 
# - The model correctly predicts almost all retained users, but misses most churned users.
# - Since the project's goal is to identify churned users, the results aren't very interesting. We'll run scorecards for a more in-depth analysis.

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

# Analyzing results: 
# - The model hits the vast majority of its predictions, with an accuracy of 82.4% considering retained and churned users.
# - However, when isolating churned users, the accuracy is 53%, indicating that only half of those identified were actually churned. When we look at the total number of churned customers the model was able to correctly identify, the number drops to 9%, or 9 out of 100.
# - That's why the F1 Score is so low, at 16%, because the balance between accuracy and accuracy for churn is very low.

# Final considerations about the model
# The model achieved 82% overall accuracy, meaning it gets most of its predictions right.
# However, performance is uneven across groups:
# - For customers who remain (retained), the model performs very well (98% accuracy).
# - For customers who churn (cancel), the model has low detection power, identifying only 9% of actual cases.
# 
# This is because there are many more customers who remain than those who cancel.
# The model prioritizes correct predictions in the majority class, but fails to predict churn well—precisely the customers most important to the retention strategy.
# 
# Despite its good overall accuracy, the model is not reliable for predicting churn.
# It can create a false sense of security, as it practically only predicts that customers will remain.
# 
# To improve predictive results with the available data, it would be worthwhile to use some technique to balance the data.

# %%

# Several techniques were tested to improve churned user identification. 
# The best alternative found was to use a combination of SMOTE and undersampling. SMOTE creates synthetic data from the minority class, in this case churned users, and then undersamples the retained users. This balances data loss and duplication.

from imblearn.combine import SMOTEENN

smote_enn = SMOTEENN(random_state=42)
X_res, y_res = smote_enn.fit_resample(X_train, y_train)

# %%
# training new model with balanced data
model2 = LogisticRegression(max_iter=5000)
model2.fit(X_res, y_res)

# %%
# creating a series whose index is the column names and whose values are the coefficients in model.coef_

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

# - New model results:
# - True Retained (TN): 1,614
# - False Churns (FP): 1327
# - False Retained (FN): 129
# - True Churns (TP): 505

# The new model still remained biased, but now toward the "churned" side. Even with a large increase in false positives, the model reduced false negatives and managed to accurately predict the number of true churns.

# %% 
# evaluating metrics
print(metrics.classification_report(y_test, y_pred, target_names=target_labels))

# Previously, churned user recall was at 9%, a very low value. With the new model, recall increased to 80%, and the F1 Score increased from 0.16 to 0.41 (it didn't increase further due to the high rate of false positives).
# Even with a drop in retained user scores, the project's focus is churn prediction. It's more interesting to have a more targeted view of users who will churn, even if slightly distorted. Ultimately, it would be better to spend money offering a promotion to someone who would stay than to lose a real customer.
# Of course, more research is needed, as well as an assessment of the project's financial viability.

# %%
# Adjusting the threshold to analyze changes in scores and user churn predictions.

y_prob = model2.predict_proba(X_test)[:, 1]  
# changing the default threshold from 0.5 to 0.3
y_pred_threshold = (y_prob >= 0.3).astype(int)  

print(metrics.classification_report(y_test, y_pred_threshold, target_names=target_labels))

# Overall scores decreased, but churned user recall increased from 80% to 89%.
# Model results should be discussed based on the business's primary need. A balance would be ideal.

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

# The graph shows how precision and recall change as the decision threshold is adjusted, being inversely proportional.
# - If the main objective is to detect as many churners as possible, a low threshold of 0.3 is ideal, even if it misses those who wouldn't otherwise leave.
# - If the main objective is to avoid false alarms, a high threshold of 0.6-0.7 would be more effective, but would miss more real churners.


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

# Since there is a significant difference in churn identification accuracy between the two models, the results are expected to be inconsistent.
# In the first model, the device would be the most decisive factor in increasing the likelihood of churn, while activity days and professional drivers would be the factors that decrease the likelihood of churn.
# In the second model, there are no significant results that increase the likelihood of churn, while professional drivers, diversion, and activity days would decrease the likelihood of churn.
# Both results contradict the indicators found previously during the project. However, in terms of future importance, the first graph would be more reliable, since the Precision Score for churn users is 53% compared to 24% in the second model.

# %%
# FINAL CONSIDERATIONS

# According to the feature_importance graph, activity_days was by far the most important feature in the model. It showed 
# a negative correlation with user churn. This was not surprising, as this variable showed a strong correlation with 
# driving_days, which, according to the EDA, had a negative correlation with churn.

# Km_per_driving_day was expected to have a higher predictive value in the model. In the model, when they looked at 
# feature_importance, this attribute ended up appearing as of little relevance (second least important).

# In a multiple logistic regression model, variables interact with each other, and these interactions can result in 
# seemingly counterintuitive relationships. This is both a strength and a weakness of predictive models, as capturing 
# these interactions typically makes a model more predictive while also making it more difficult to explain.

# As in the case of km_per_driving_day, something that appears predictive on its own may lose relevance when combined 
# with other variables because the model is already explaining the same effect through another pathway.

# The second model created, despite lowering the overall prediction scores, managed to increase churn recall to 89%. 
# However, it was generating many false positives. However, in an emergency scenario, this model could be used despite 
# its flaws, since it's better to please a percentage of users who wouldn't otherwise leave the app with coupon campaigns
# or exclusive features than to let several churned users slip through undetected.

# New features could be designed to generate a better predictive signal. In this model's case, one of the designed 
# features (professional_driver) was the third most predictive predictor. It could also be useful to scale the predictor 
# variables and/or rebuild the model with different combinations of predictor variables to reduce noise from non-predictive 
# features.

# Even though the best possible model solution was obtained with the available data, other solutions could be explored in 
# the future. How to test more robust machine learning algorithms for imbalanced data (next step of the work). It would also 
# be helpful to have more information about each user's driving level (such as driving times, geographic locations, etc.), 
# with more granular data that would improve future regressions.