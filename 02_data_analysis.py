# %%
# seting up envirioment 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# loading data

df = pd.read_csv("data/waze_dataset.csv")

# %%
df.head(10)

# %%
print(df.size)
print()
print(df.shape)

# %%
df.describe()

# %%
df.info()

# %% 
# Sessions column
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

sns.boxplot(x=df["sessions"], fliersize=1, ax=ax1) 
ax1.set_xlabel("sessions", fontsize=14)
ax1.set_title('Sessions Boxplot', fontsize=18)

median = df["sessions"].median()
ax2 = sns.histplot(x=df["sessions"]) 
ax2.axvline(median, color="r", linestyle="--")
ax2.text(75,1000, "median= 56.0", color= "r")
ax2.set_xlabel('sessions')
ax2.set_ylabel('count', fontsize=14)
ax2.set_title('Sessions Histogram', fontsize=18)

plt.show()


# %%
# making a fuction to help ploting

def grapher(column_name, spacing, median_txt=True, **kwargs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4), constrained_layout=True)

    sns.boxplot(x=df[column_name], fliersize=1, ax=ax1)
    ax1.set_xlabel(column_name, fontsize= 14)
    ax1.set_title(column_name + " boxplot", fontsize=18)

    median= round(df[column_name].median(), 1)
    ax2 = sns.histplot(x=df[column_name], **kwargs)
    ax2.axvline(median, color="r", linestyle= "--")
    if median_txt==True:
        ax2.text(median + spacing, ax2.get_ylim()[1] * 0.8, f"median={median}", color="r", 
                 ha="left", va="top", fontsize=13)
    else:
        print("Median:", median)
    ax2.set_xlabel(column_name, fontsize= 14)
    ax2.set_ylabel("count", fontsize= 14)
    ax2.set_title(column_name + " histogram", fontsize=18)

    plt.show()

# %% 
# drives column
grapher("drives", 30)

# %%
# total_sessions column

grapher("total_sessions", 50)

# %%
# n_days_after_onboarding column

grapher("n_days_after_onboarding", 50, median_txt=False)

# %%
# driven_km_drives

grapher("driven_km_drives", 1000)

# %%
# duration_minutes_drives column

grapher("duration_minutes_drives", 1000)

# %%
# activity_days column

grapher("activity_days", 50, median_txt=False, discrete=True)

# %%
# driving_days column

grapher("driving_days", 1, discrete=True)

# %%
# device column

fig = plt.figure(figsize=(3,3))
data= df["device"].value_counts()
plt.pie(data,
        labels=[f"{data.index[0]}: {data.values[0]}",
                f"{data.index[1]}: {data.values[1]}"],
                autopct= "%.1f%%")
plt.title("Users by device")
plt.show()

# %%
# label column

fig = plt.figure(figsize=(3,3))
data = df["label"].value_counts()
plt.pie(data,
        labels = [f"{idx}: {val}" for idx, val in data.items()],
        autopct= "%.1f%%")
plt.title("Retained vs Churned users")
plt.show()

# %%
# driving days vs activity dasy

plt.figure(figsize=(12,4))
label=["driving_days","activity_days"]
plt.hist([df["driving_days"], df["activity_days"]],
         bins=range(0,33), label=label)
plt.xlabel("days")
plt.ylabel("count")
plt.legend()
plt.title("Driving days vs Acitivy days")
plt.show()

# %%
print(df["driving_days"].max())
print(df["activity_days"].max())

# %%
sns.scatterplot(data= df, x= "driving_days", y= "activity_days")
plt.title("Driving days vs Activity days")
plt.plot([0,31], [0,31], color="r", linestyle= "--")
plt.show()

# %%
# reteintion by device
plt.figure(figsize=(5,4))
sns.histplot(data= df, x= "device", hue="label",
            multiple="dodge", shrink=0.9)
plt.title("Retention by device histogram")
plt.show()

# %%
# retention by kilometers driven per driving day
df["km_per_driving_day"] = df["driven_km_drives"] / df["driving_days"]
df["km_per_driving_day"].describe()

# %%
# cleaning data
df.loc[df["km_per_driving_day"]== np.inf, "km_per_driving_day"] = 0
df["km_per_driving_day"].describe()

# %%
# making the plot

plt.figure(figsize=(12,5))
sns.histplot(data=df, x="km_per_driving_day",
             bins= range(0,1201,15), hue="label",
             multiple="fill")
plt.ylabel("%", rotation= 0)
plt.title("Churn rate by mean km per driving day")
plt.show()

# %%
# churn rate per mumber of driving days
plt.figure(figsize=(12,5))
sns.histplot(data=df, x= "driving_days", bins= range(0,32,1),
             hue="label", multiple="fill")
plt.ylabel("%", rotation= 0)
plt.title("Churn rate by driving days")

# %%
# proportion of sessions that occurred in the last month

df["percent_sessions_in_last_month"] = df["sessions"] / df["total_sessions"]
df.head()

# %%
df["percent_sessions_in_last_month"].mean()

# %%
grapher("percent_sessions_in_last_month", 50, hue=df["label"], multiple="layer", median_txt=False)

# %%
# checking the median value of the n_days_after_onboarding

df["n_days_after_onboarding"].mean()

# %%
data = df.loc[df["percent_sessions_in_last_month"]>=0.4]
plt.figure(figsize=(5,3))
sns.histplot(x=data["n_days_after_onboarding"])
plt.title("Num. days after onboarding for users with >=40% sessions in last month")
plt.show()

# %%
# cleaning outliers

def outlier_imputer(column_name, percentile):
    threshold = df[column_name].quantile(percentile)
    df.loc[df[column_name] > threshold, column_name] = threshold
    print("{:>25} | percentile: {} | threshold: {}".format(column_name, percentile, threshold))

# %%

for i in ["sessions", "drives", "total_sessions", "driven_km_drives", "duration_minutes_drives"]:
    outlier_imputer(i, 0.95)

# %%
df.describe()

# %%
# checking changes
grapher("sessions", 20)

# %% 
grapher("drives", 20)

# %%
df["monthly_drives_per_session_ratio"] = (df["drives"]/df["sessions"])
df.head()

# %%
grapher("monthly_drives_per_session_ratio", 1, median_txt=False)