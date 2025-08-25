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

# Half of users open the app at least 56 times a month.
# However, some users open the app more than 700 times.
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

# The "drives" data follows a similar distribution to the "sessions" variable. 
# Half users had at least 48 drives, but some drives has more than 400 drives.
# %%
# total_sessions column

grapher("total_sessions", 50)

# Based on previous session data, the median was 48, while the current total 
# sessions are ~160, indicating a high proportion of sessions in the last month.
# %%
# n_days_after_onboarding column

grapher("n_days_after_onboarding", 50, median_txt=False)

# The total user tenure has a uniform distribution with values ranging from 
# near-zero to 3,500 days, approximately 9.5 years.
# %%
# driven_km_drives

grapher("driven_km_drives", 1000)

# Half the users driving under 3,495 kilometers. As discovered in previous 
# analysis and the box plot, there are many outliers that will be excluded.
# %%
# duration_minutes_drives column

grapher("duration_minutes_drives", 1000)

# Half of the users drove less than ~1,478 minutes (~25 hours), but some users clocked 
# over 250 hours over the month, which represents more than 8 hours a day
# %%
# activity_days column

grapher("activity_days", 50, median_txt=False, discrete=True)

# These graphs offer some insights:
# - Within the last month, users opened the app a median of 16 times. 
# - The box plot reveals a centered distribution. '
# - The histogram shows a nearly uniform distribution of ~500 people opening the app 
# on each count of days. 
# - However, there are ~250 people who didn't open the app at all and ~250 people who 
# opened the app every day of the month.
# - This distribution does not reflect the distribution of "sessions," showing that 
# they are not directly related.
# %%
# driving_days column

grapher("driving_days", 1, discrete=True)

# There were almost twice as many users (~1,000 vs ~500) who didn't drive at all during the month. 
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

# There are nearly twice as many iPhone users as Android users represented in this data.
# %%
# label column

fig = plt.figure(figsize=(3,3))
data = df["label"].value_counts()
plt.pie(data,
        labels = [f"{idx}: {val}" for idx, val in data.items()],
        autopct= "%.1f%%")
plt.title("Retained vs Churned users")
plt.show()

# Less than 18% of the users churned.
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

# There are fewer people who didn't use the app than people who didn't drive during the month.
# People probably open the app more than they use it to drive—perhaps to check travel times or route information, 
# to update settings, or even by mistake.
# %%
print(df["driving_days"].max())
print(df["activity_days"].max())

# The number of days in the month is not the same between variables. Although it's possible 
# that not a single user drove all 31 days of the month, it's highly unlikel

# %%
sns.scatterplot(data= df, x= "driving_days", y= "activity_days")
plt.title("Driving days vs Activity days")
plt.plot([0,31], [0,31], color="r", linestyle= "--")
plt.show()

# Checking whether the number of drives is equal to or greater than the active days.
# %%
# reteintion by device
plt.figure(figsize=(5,4))
sns.histplot(data= df, x= "device", hue="label",
            multiple="dodge", shrink=0.9)
plt.title("Retention by device histogram")
plt.show()

# The proportion of churned users to retained users is consistent between device types.
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

# Disregarding rows where the distance is greater than 1,200 km.
# Confirming what was found before, the churn rate tends to increase as the mean daily distance driven increases.
# %%
# churn rate per mumber of driving days
plt.figure(figsize=(12,5))
sns.histplot(data=df, x= "driving_days", bins= range(0,32,1),
             hue="label", multiple="fill")
plt.ylabel("%", rotation= 0)
plt.title("Churn rate by driving days")

# The churn rate is highest for people who didn't use Waze much during the last month. 
# The more times they used the app, the less likely they were to churn.
# While 40% of the users who didn't use the app at all last month churned, nobody who used the app 30 days churned.
# When people who don't use the app churn, it might be the result of dissatisfaction in the past, or it might be 
# indicative of a lesser need for a navigational app, like using public transportation.

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

# Half of the people in the dataset had 40% or more of their sessions in just the last month, yet the overall 
# median time since onboarding is almost five years.
# %%
data = df.loc[df["percent_sessions_in_last_month"]>=0.4]
plt.figure(figsize=(5,3))
sns.histplot(x=data["n_days_after_onboarding"])
plt.title("Num. days after onboarding for users with >=40% sessions in last month")
plt.show()

# Further investigation is needed to understand why so many long-time users have suddenly used the app so much in the last month.

# %%
# cleaning outliers

# crating a fuuction to set a threshold based on a 95th percentile of the distribution.
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

# %%

# CONCLUSION:

# - Almost all variables were either very skewed to the right or evenly distributed.
# - Several variables had highly unlikely or perhaps even impossible outliers, such 
# as `km_driven_driven`. Some of the monthly variables may also be problematic, such 
# as `activity_days` and `driving_days`, because one has a maximum value of 31, while 
# the other has a maximum value of 30.
# - Further investigation would be necessary to determine why so many long-term users 
# started using the app so frequently only in the last month. Were there any changes 
# in the last month that might have motivated this type of behavior?
# - Less than 18% of users churned, and ~82% were retained.
# - Distance driven per driving day had a positive correlation with user churn. The 
# farther a user drove on each driving day, the more likely they were to churn. On 
# the other hand, number of driving days had a negative correlation with churn. Users 
# who drove more days of the last month were less likely to churn.
# - Users of all tenures from brand new to ~10 years were relatively evenly represented 
# in the data.