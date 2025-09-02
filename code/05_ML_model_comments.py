# Final consideration before building the ML model:
# The goal of this project is to predict whether a customer will churn or continue using the app, allowing Waze to adopt proactive retention measures. However, there are ethical implications: a false negative would cause the company to miss the opportunity to intervene and potentially retain the user, while a false positive could generate a negative experience by annoying loyal users with unnecessary actions. Despite these risks, if the measures are applied in a balanced and effective manner, the benefits outweigh the potential problems, making model development feasible and recommended, provided it is accompanied by ongoing impact analysis.

# %%
import numpy as np
import pandas as pd
# setting to show all columns
pd.set_option("display.max_columns", None)

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, precision_recall_curve

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance

import pickle

# %%
df0 = pd.read_csv("../data/waze_dataset.csv")

# %%
df0.head()
# %%
# feature engineering
df = df0.copy()

# %%
df.info()

# %%
# creating new columns
df['km_per_driving_day'] = df['driven_km_drives'] / df['driving_days']
df['km_per_driving_day'].describe()
# %%
df.loc[df['km_per_driving_day']==np.inf, 'km_per_driving_day'] = 0
df['km_per_driving_day'].describe()

# %%
df["percent_sessions_in_last_month"] = df['sessions'] / df['total_sessions']
df["percent_sessions_in_last_month"].describe()

# %%
df["professional_driver"] = np.where((df["drives"] >= 60) & (df["driving_days"] >= 15), 1, 0)
df["professional_driver"].value_counts()

# %%
df["total_sessions_per_day"] = df["total_sessions"] / df["n_days_after_onboarding"]
df["total_sessions_per_day"].describe()

# %%
df["km_per_hour"] = df["driven_km_drives"] / (df["duration_minutes_drives"] / 60)
df["km_per_hour"].describe()

# %%
df["km_per_drive"] = df["driven_km_drives"] / df["drives"]
df["km_per_drive"].describe()

# %%
df.loc[df["km_per_drive"]==np.inf, "km_per_drive"] = 0
df["km_per_drive"].describe()

# %%
df["percent_of_sessions_to_favorite"] = (
    df["total_navigations_fav1"] + df["total_navigations_fav2"]) / df["total_sessions"]
df["percent_of_sessions_to_favorite"].describe()

# %%
# checking new columns
df.head(10)
# %%
# droping missing values
df = df.dropna(subset=["label"])

# %%
# variable encoding
df["device2"] = np.where(df["device"]=="Android", 0, 1)
df[["device", "device2"]].tail()

# %%
# target encoding
df["label2"] = np.where(df["label"]=="retained", 0, 1)
df[["label","label2"]].tail()

# %%
# feature selection
df = df.drop(["ID"], axis=1)
df.head()

# %%
# evaluation metric
round(df["label"].value_counts(normalize=True),3)

# As seen before, approximately 18% of the users in this dataset churned. This is an unbalanced dataset, but not extremely so for advanced ML models. It can be modeled without any class rebalancing.

# %%
# modeling workflow and model selection process
# the data will be split into train/validation/test sets (60/20/20)

# defining X and y variables
X = df.copy()
X = X.drop(columns=["label","label2", "device"])

y = df["label2"]

# %%
# splitting into train and test sets
X_tr, X_test, y_tr, y_test = train_test_split(X, y, stratify=y,
                                              test_size=0.2, random_state=42)

# splitting into train and validate sets
X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, stratify=y_tr,
                                              test_size=0.25, random_state=42)

# %%
# verifing the number of samples in the partitioned data
for i in [X_train, X_val, X_test]:
    print(len(i))

# the test and validation sets have the same size, as they should be.

# %%
# MODELING

# random forest

rf = RandomForestClassifier(random_state=42)

# defining hyperparameters
cv_params = {'max_depth': [None],
             'max_features': [1.0],
             'max_samples': [1.0],
             'min_samples_leaf': [2],
             'min_samples_split': [2],
             'n_estimators': [300],
             }

# defining a list of scoring metrics
scoring = ["accuracy", "precision", "recall", "f1"]

# instantiating the GridSearchCV object
rf_cv = GridSearchCV(rf, cv_params, scoring= scoring, cv=4, refit="recall")

# %%
# fitting the model

rf_cv.fit(X_train, y_train)

# %%
# examining the best average score across all the validation folds
rf_cv.best_score_

# %%
# examining the best combination of hyperparameters
rf_cv.best_params_

# %%
# making a fuction to output all scores
def make_results(model_name:str, model_object, metric:str):
    '''
    Arguments:
        model_name (string): what you want the model to be called in the output table
        model_object: a fit GridSearchCV object
        metric (string): precision, recall, f1, or accuracy

    Returns a pandas df with the F1, recall, precision, and accuracy scores
    for the model with the best mean 'metric' score across all validation folds.
    '''

    # Create dictionary that maps input metric to actual metric name in GridSearchCV
    metric_dict = {'precision': 'mean_test_precision',
                   'recall': 'mean_test_recall',
                   'f1': 'mean_test_f1',
                   'accuracy': 'mean_test_accuracy',
                   }

    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(model_object.cv_results_)

    # Isolate the row of the df with the max(metric) score
    best_estimator_results = cv_results.iloc[cv_results[metric_dict[metric]].idxmax(), :]

    # Extract accuracy, precision, recall, and f1 score from that row
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy

    # Create table of results
    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision],
                          'recall': [recall],
                          'F1': [f1],
                          'accuracy': [accuracy],
                          },
                         )

    return table
# %%
results = make_results("Random Forest cv", rf_cv, "recall")
results

# Besides accuracy, the scores aren't as good. However, compared to the logistic regression model, the recall was ~0.09, meaning this model has 33% better recall and roughly the same accuracy, and was trained with less data.

# %%
# XGBOOST

# making a xgboost model to compare results 

xgb = XGBClassifier(objective="binary:logistic", random_state=42)

# defining hyperparameters
cv_params = {'max_depth': [6, 12],
             'min_child_weight': [3, 5],
             'learning_rate': [0.01, 0.1],
             'n_estimators': [300]
             }

# defining a list of scoring metrics
scoring = ["accuracy", "precision", "recall", "f1"]

# instantiating the GridSearchCV object
xgb_cv = GridSearchCV(xgb, cv_params, scoring=scoring, cv=5, refit="recall")

# %%
# fitting the model
xgb_cv.fit(X_train, y_train)

# %%
# getting the best score grom this model
xgb_cv.best_score_

# %%
# examing the best parameters
xgb_cv.best_params_

# %%
# outputing scores

xgb_cv_results = make_results("XGBoost cv", xgb_cv, "recall")

results = pd.concat([results, xgb_cv_results], axis=0)
results

# This model fit the data even better than the random forest model. The recall score is nearly double the recall score from the logistic regression model from the previous course, and it's almost 50% better than the random forest model's recall score, while maintaining a similar accuracy and precision score. 

# %%
# model selection
# predicting on validation data with the two model

# random forest:
rf_val_preds = rf_cv.best_estimator_.predict(X_val)

# %%
def get_test_scores(model_name:str, preds, y_test_data):
    '''
    Generate a table of test scores.

    In:
        model_name (string): Your choice: how the model will be named in the output table
        preds: numpy array of test predictions
        y_test_data: numpy array of y_test data

    Out:
        table: a pandas df of precision, recall, f1, and accuracy scores for your model
    '''
    accuracy = accuracy_score(y_test_data, preds)
    precision = precision_score(y_test_data, preds)
    recall = recall_score(y_test_data, preds)
    f1 = f1_score(y_test_data, preds)

    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision],
                          'recall': [recall],
                          'F1': [f1],
                          'accuracy': [accuracy]
                          })

    return table

# %%
rf_val_scores = get_test_scores("RandomForest val", rf_val_preds, y_val)

results = pd.concat([results, rf_val_scores], axis=0)
results
# Notice that the scores went down from the training scores across all metrics, but only by very little. This means that the model did not overfit the training data.

# %%
# XGBoost

xgb_val_preds = xgb_cv.best_estimator_.predict(X_val)

xgb_val_scores = get_test_scores("XGBoost val", xgb_val_preds, y_val)

results = pd.concat([results, xgb_val_scores], axis=0)
results

# Just like with the random forest model, the XGBoost model's validation scores were lower, but only very slightly. It is still the clear champion.

# %%
# using the champion model to predict on test data (XGBoost)

# This serves to give a final indication of how to expect the model to behave on new future data.

xgb_test_preds = xgb_cv.best_estimator_.predict(X_test)

xgb_test_scores = get_test_scores("XGBoost test", xgb_test_preds, y_test)

results = pd.concat([results, xgb_test_scores], axis=0)
results

# The scores performed better than with the validation data. 
# The recall score, the most important metric, increased even further.

# %%
# confusion matrix

# building a confusion matrix of the champion model's predictions on the test data to check the results.

cm = confusion_matrix(y_test, xgb_test_preds, labels= xgb_cv.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= ["retained", "churned"])
disp.plot();

# The model predicted three times as many false negatives than it did false positives, and it correctly identified only 16.6% of the users who actually churned.

# %%
# feature of importance 
plot_importance(xgb_cv.best_estimator_);

# The XGBoost model made more use of many of the features than did the logistic regression model from the previous course, which weighted a single feature (activity_days) very heavily in its final prediction.
# This highlights the importance of feature engineering in creating the new columns. Engineered features accounted for six of the top 10 features (and three of the top five).

# %%
# adjusting threshold to test different model predictions results

# ploting precision-recall curve
display = PrecisionRecallDisplay.from_estimator(
    xgb_cv.best_estimator_, X_test, y_test, name="XGboost"
)
plt.title("Precision-recall curve, XGBoost model")
plt.show()

# As recall increases, precision decreases. But what if you determined that false positives aren't much of a problem? For example, in the case of this Waze project, a false positive could just mean that a user who will not actually churn gets an email and a banner notification on their phone. It's very low risk.

# %%
# getting predicted probabilities on the test data

predicted_probabilities = xgb_cv.best_estimator_.predict_proba(X_test)
predicted_probabilities

# %%
# create a list of just the second column values (probability of target)
probs = [i[1] for i in predicted_probabilities]

# create an array of new predictions that assigns a 1 to any value >= 0.4
new_preds = np.array([1 if x >= 0.4 else 0 for x in probs])
new_preds

# %%
precision, recall, thresholds = precision_recall_curve(y_test, probs)

plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.title("Precision vs Recall for Differente Thresholds")
plt.xlabel("threshold")
plt.ylabel("score")
plt.legend()
plt.show()

# %%
# checking evaluation metrics for the new 0.4 threshold
get_test_scores("XGB, threshold = 0.4", new_preds, y_test)

# %%
# comparing results
results

# Recall and F1 score increased significantly, while precision and accuracy decreased
 
# %%
# Making a function that determines the threshold based on the desired recall

def threshold_finder(y_test_data, probabilities, desired_recall):
    '''
    Find the decision threshold that most closely yields a desired recall score.

    Inputs:
        y_test_data: Array of true y values
        probabilities: The results of the `predict_proba()` model method
        desired_recall: The recall that you want the model to have

    Outputs:
        threshold: The decision threshold that most closely yields the desired recall
        recall: The exact recall score associated with `threshold`
    '''
    probs = [x[1] for x in probabilities]  # Isolate second column of `probabilities`
    thresholds = np.arange(0, 1, 0.001)    # Set a grid of 1,000 thresholds to test

    scores = []
    for threshold in thresholds:
        # Create a new array of {0, 1} predictions based on new threshold
        preds = np.array([1 if x >= threshold else 0 for x in probs])
        # Calculate recall score for that threshold
        recall = recall_score(y_test_data, preds)
        # Append the threshold and its corresponding recall score as a tuple to `scores`
        scores.append((threshold, recall))

    distances = []
    for idx, score in enumerate(scores):
        # Calculate how close each actual score is to the desired score
        distance = abs(score[1] - desired_recall)
        # Append the (index#, distance) tuple to `distances`
        distances.append((idx, distance))

    # Sort `distances` by the second value in each of its tuples (least to greatest)
    sorted_distances = sorted(distances, key=lambda x: x[1], reverse=False)
    # Identify the tuple with the actual recall closest to desired recall
    best = sorted_distances[0]
    # Isolate the index of the threshold with the closest recall score
    best_idx = best[0]
    # Retrieve the threshold and actual recall score closest to desired recall
    threshold, recall = scores[best_idx]

    return threshold, recall

# %%
# get the predicted probabilities from the champion model
probabilities = xgb_cv.best_estimator_.predict_proba(X_test)

# %%
# desired recall = 0.5
threshold_finder(y_test, probabilities, 0.5)

# %%
probs = [x[1] for x in probabilities]
new_preds = np.array([1 if x >= 0.194 else 0 for x in probs])

# %%
get_test_scores('XGB, threshold = 0.194', new_preds, y_test)

# %%

# CONCLUSIONS

# Whether this model is recommended for churn prediction would depend on the company's objective.
# 
# If it were used to guide relevant business decisions, then no, as it is not a strong enough predictor, as evidenced by its low recall score. However, if the model is only being used to guide additional exploratory efforts or lightweight retention marketing campaigns such as banners and emails, it may have value.
# 
# Dividing the data into three parts (train, validation, and test) has its advantages and disadvantages. This results in less data available to train the model than splitting it into just two parts, especially on a small dataset like this. However, performing model selection on a separate validation set allows the champion model to be tested in isolation on the previously unseen test set, more realistically simulating new data input. This provides another step in model verification, providing a better estimate of future performance.
# 
# As I observed when comparing the scores of the models developed at this stage with the logistic regression model,
# tree-based models are often better predictors, as was the case in this case study.
# 
# Despite all the effort, the scores were lower than ideally desired. Even using techniques to increase recall, the other indicators dropped even further.
# 
# It would be extremely important for the development of an effective machine learning model to have more information about users, including personal information such as age, job, geographic and location information, more granular data such as reports of alerts on the route, reading alerts from other users, and how many different destinations they enter into the app. Cross-referencing this data with current data in different combinations across more feature engineering stages could help the model become more predictive.
