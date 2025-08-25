# Do drivers who open the application using an iPhone have the same number of drives on average as drivers who use Android devices?

# %%
import numpy as np
import pandas as pd
from scipy import stats

# %%
df = pd.read_csv("data/waze_dataset.csv")

# %%
df.head()
# %%
# crating a map dictionary
map_dictionary = {"iPhone":2, "Android":1}

# %%
# crating a new device type column
df["device_type"] = df["device"].map(map_dictionary)
df.head()

# %%
df.groupby(["device_type"])["drives"].mean()

# the mean number os drives for iPhone users is about 66, 
# and the Android users is about 67
# %%
# Hypothesis testing
# the goal is to oconduct a two-sample t-test.

# Steps for constructing a hypothesis test:

# 1. State the null hypothesis and the alternative hypothesis
# 𝐻0: There is no difference in average number of drives between drivers who use iPhone devices and drivers who use Androids, the difference in mean values occurred by chance.
# 𝐻𝐴: There is a difference in average number of drives between drivers who use iPhone devices and drivers who use Androids, a statistically significant difference that the events occurred for some reason.

# 2. Choose a signficance level
# The choosen significante level is 5%

# %%
# 3. find the p-value

# isolating "drives" columns for user device type
iphone = df[df["device_type"] == 2]["drives"]
android = df[df["device_type"] == 1]["drives"]

# performing the t-test to find p-value
stats.ttest_ind(a=iphone,b=android, alternative="two-sided", equal_var=False)

# %%
statistic, pvalue = stats.ttest_ind(a=iphone,b=android, alternative="two-sided", equal_var=False)
pvalue_percent = round(pvalue * 100, 2)
print("p-value:",pvalue_percent, "%")

# the p-value is ~14.3%
# %%
# 4. Reject or fail to reject the null hypothesis
# Since the p-value was 14%, or greater than the chosen significance level, this indicates 
# that the hypothesis was rejected. There was no statistically significant difference between 
# Android and iPhone users.

# %%
# CONCLUSIONS
# The key business insight is that drivers who use iPhone devices on average have a similar 
# number of drives as those who use Androids.

# One potential next step is to explore what other factors influence the variation in the 
# number of drives, and run additonal hypothesis tests to learn more about user behavior. 
# Further, temporary changes in marketing or user interface for the Waze app may provide more 
# data to investigate churn.