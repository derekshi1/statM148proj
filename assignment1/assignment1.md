# TASK 1

## 1. Total Rows: 54,960,961
## 2. Unique IDs: 1,430,445
## 3. Time Range: 2020-11-03 03:31:30+00:00 (November 3rd, 2020 @ 3:30UTC) to 2023-01-23 12:29:56+00:00 (January 23rd, 2023 @ 12:29UTC)

# TASK 2

## 1. Total duplicates found: 4,515,888
## 2. Proportion of duplicates: 8.22%
## 3. Rows remaining after cleaning: 50,445,073

# TASK 3
I sampled 10% of the data on seed(42), to get 143,000 users, then proceeded with summary stats
I used median over mean in the case of large outliers (bots) which would skew results

## 1. Typical Journey Characteristics

**Median Actions per Journey:** 24.0 actions

![The number of actions per Journney shows a steady decrease in countas actions increase](/Users/derek/Documents/statM148proj/number_of_actions.png)

**Median Journey Duration:** 140,758.61 minutes (~97.75 days)
**Median between Actions:** 88 seconds (Median time between actions)
![Actions are concentrated heavily in the first 100 seconds, and have a significant drop off followed by a steady decline thereafter](/Users/derek/Documents/statM148proj/time_between_actions.png)

Insight: While the total "life" of a user ID in the system spans roughly 3 months, the 88-second median between actions suggests that actual engagement occurs in smaller sessions.

## 2. Common User Actions

| Event Name | Total Occurrences | Stage |
|:---|:---|:---|
| **Browse Products** | 1,944,325 | First Purchase |
| **View Cart** | 595,620 | First Purchase |
| **Application Web View** | 590,662 | Apply for Credit |
![Note: the count of users per action is on the 1e6 scale, showing a dominant number of actions in browse_products](/Users/derek/Documents/statM148proj/top10frequentactions.png)

## 3. Success Case Analysis
A "Success" is defined as any journey containing at least one **order_shipped** event (Event ID: 28). This segment represents **19.53%** (279,363 users) of the total population.

The following table highlights the difference in behavior between users who convert and those who do not.

| Metric | Non-Successful Users (80.47%) | Successful Users (19.53%) |
|:---|:---:|:---:|
| **Median Actions** | 21.0 | **41.0** |
| **Average Actions** | 31.0 | **54.0** |
| **Median Duration** | 159.0 Days | **18.0 Days** |
| **Average Duration** | 154.0 Days | **48.0 Days** |

## Key findings
1. **Intensity of Engagement:** Successful users are nearly twice as active (41 vs 21 median actions) as those who do not purchase.
2. **The "Conversion Window":**  Successful users convert relatively quickly (Median: **18 days**), whereas non-successful users tend to linger in the "browsing" state for over **154 dayss** 
3. **Progression:** The high frequency of "Application Web View" in the apply for credit stage suggests that navigating the credit application is a standard prerequisite for the successful "Order Shipped" outcome.
