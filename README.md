# OTT-Subscription-Churn-Customer-Analysis
This repository contains a detailed analysis of customer churn data using Python

![Banner Image](https://github.com/baabhishek/OTT-Subscription-Churn-Customer-Analysis/blob/main/banner-image-41.4-01.png)

# Customer Churn Analysis

This repository contains a detailed analysis of customer churn data using Python. The analysis covers data cleaning and preprocessing, exploratory data analysis -> EDA, and advanced relationship analysis using various visualization techniques.
<p align="center">
  <img src="https://github.com/user-attachments/assets/b129795b-1e81-4e5f-b92c-b142217d1261" alt="Image Description">
</p>

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Data Preprocessing](#data_preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Advanced Relationship Analysis](#advanced-relationship-analysis)
- [Visualizations](#visualizations)
- [License](#license)

## Introduction

Customer churn analysis helps businesses understand why customers leave and provide insights to improve customer retention strategies. This project analyzes customer churn data without applying machine learning algorithms, focusing on basic python concepts instead of data exploration and visualization.

## Dataset:

The dataset used in this project contains the following columns:
- `year`: Year of data collection
- `customer_id`: Unique identifier for each customer
- `phone_no`: Customer's phone number
- `gender`: Gender of the customer
- `age`: Age of the customer
- `no_of_days_subscribed`: Number of days the customer has been subscribed
- `multi_screen`: Whether the customer has a multi-screen subscription
- `mail_subscribed`: Whether the customer is subscribed to the mailing list
- `weekly_mins_watched`: Number of minutes watched per week
- `minimum_daily_mins`: Minimum daily minutes watched
- `maximum_daily_mins`: Maximum daily minutes watched
- `weekly_max_night_mins`: Maximum minutes watched at night per week
- `videos_watched`: Number of videos watched
- `maximum_days_inactive`: Maximum number of days the customer was inactive
- `customer_support_calls`: Number of customer support calls made
- `churn`: Whether the customer churned (1) or not (0)

## Installation along with the following libraries:

| Anaconda | Jupyter | Python | Pandas | Matplotlib | NumPy  | Seaborn
|----------|---------|--------|--------|-------------|-------|---------|
| <img src="https://cdn.simpleicons.org/anaconda/44A833.svg" title="Anaconda" alt="Anaconda" width="50" height="50"/> | <img src="https://cdn.simpleicons.org/jupyter/F37626.svg" title="Jupyter" alt="Jupyter" width="50" height="50"/> | <img src="https://skillicons.dev/icons?i=python" title="Python" alt="Python" width="49" height="49"/> | <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg" title="Pandas" alt="Pandas" width="45" height="45"/> | <img src="https://upload.wikimedia.org/wikipedia/commons/0/01/Created_with_Matplotlib-logo.svg" title="Matplotlib" alt="Matplotlib" width="50" height="50"/> | <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" title="NumPy" alt="NumPy" width="50" height="50"/> | <img src="https://user-images.githubusercontent.com/315810/92159303-30d41100-edfb-11ea-8107-1c5352202571.png" title="Seaborn" alt="Seaborn" width="50" height="50"/>

![Beige Minimalist Personal Business LinkedIn Background Photo](https://github.com/user-attachments/assets/30bb8c3a-c04c-492a-9092-d752d12a7978)

### Exploratory Data Analysis (EDA):

- Gender Distribution:

```python
import matplotlib.pyplot as plt

plt.pie(df['gender'].value_counts(),
       labels=df['gender'].value_counts(dropna=False).index,
       autopct="%.2f%%")
plt.title('Gender Distribution in %')
plt.show()
```
<img width="361" alt="Screenshot 2567-07-18 at 3 36 23 PM" src="https://github.com/user-attachments/assets/6cbff6e7-028b-454a-b255-33fa5682a24c">

- Distribution of age

```python
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

```
<img width="671" alt="Screenshot 2567-07-18 at 3 53 23 PM" src="https://github.com/user-attachments/assets/a31f14b8-3624-413e-b346-1b2145b93d31">

- Churn rate by gender
```python
plt.figure(figsize=(10, 6))
sns.barplot(x='gender', y='churn', data=df)
plt.title('Churn Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Churn Rate')
plt.show()
```
<img width="739" alt="Screenshot 2567-07-18 at 3 56 52 PM" src="https://github.com/user-attachments/assets/27760bfc-4427-4d8c-99c5-fdf925b35385">

-Churn rate by multi_screen
```python
plt.figure(figsize=(10, 6))
sns.barplot(x='multi_screen', y='churn', data=df)
plt.title('Churn Rate by Multi Screen Subscription')
plt.xlabel('Multi Screen Subscription')
plt.ylabel('Churn Rate')
plt.show()
```
<img width="708" alt="Screenshot 2567-07-18 at 3 58 15 PM" src="https://github.com/user-attachments/assets/2fe1d1b2-05e2-4cd8-8d1b-ea280ae63594">

-Correlation Heatmap of Dataset Variables
```python
plt.figure(figsize=(12, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()
```
<img width="740" alt="Screenshot 2567-07-18 at 4 00 23 PM" src="https://github.com/user-attachments/assets/588d5050-6083-4e5a-9893-717cfb2eb169">
-Churn by Weekly Minutes Watched

```python
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='weekly_mins_watched', hue='churn', kde=True)
plt.title('Weekly Minutes Watched by Churn')
plt.xlabel('Weekly Minutes Watched')
plt.ylabel('Frequency')
plt.show()
```
<img width="699" alt="Screenshot 2567-07-18 at 4 05 42 PM" src="https://github.com/user-attachments/assets/220022a8-dd4a-473c-8ba7-43d12c464dc1">

-Churn by Customer Support Calls

```python
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='customer_support_calls', hue='churn')
plt.title('Churn by Customer Support Calls')
plt.xlabel('Customer Support Calls')
plt.ylabel('Count')
plt.show()
```
<img width="689" alt="Screenshot 2567-07-18 at 4 07 42 PM" src="https://github.com/user-attachments/assets/a255d395-ce36-41dc-a481-b36353f92bb5">

-Pair Plot for Selected Features

```python
selected_features = ['age', 'no_of_days_subscribed', 'weekly_mins_watched', 'churn']
sns.pairplot(df[selected_features], hue='churn')
plt.show()
```
<img width="700" alt="Screenshot 2567-07-18 at 4 09 26 PM" src="https://github.com/user-attachments/assets/40fd8607-2cf9-47e5-ad19-f0da5f00f3a8">

-Box Plot continuous variables for churned vs. non-churned customers 
->understand the distribution and central tendency of these continuous variables

```python
continuous_features = ['age', 'no_of_days_subscribed', 'weekly_mins_watched', 'minimum_daily_mins', 'maximum_daily_mins', 'weekly_max_night_mins']

plt.figure(figsize=(15, 10))
for i, feature in enumerate(continuous_features, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x='churn', y=feature, data=df)
    plt.title(f'{feature} by Churn')
    plt.xlabel('Churn')
    plt.ylabel(feature)
plt.tight_layout()
plt.show()
```
<img width="801" alt="Screenshot 2567-07-18 at 4 12 54 PM" src="https://github.com/user-attachments/assets/792f4150-08c3-4261-8f90-d95fdc74330f">

- Violin plots combine the benefits of box plots and KDE plots
- the distribution of the data across different churn categories.

```python
plt.figure(figsize=(15, 10))
for i, feature in enumerate(continuous_features, 1):
    plt.subplot(2, 3, i)
    sns.violinplot(x='churn', y=feature, data=df, inner='quartile')
    plt.title(f'{feature} by Churn')
    plt.xlabel('Churn')
    plt.ylabel(feature)
plt.tight_layout()
plt.show()
```
<img width="790" alt="Screenshot 2567-07-18 at 4 15 11 PM" src="https://github.com/user-attachments/assets/8baf6041-9dca-4eb4-87eb-4c9a1e90759b">

- The relationship between categorical variables such as multi_screen, mail_subscribed, and churn.

```python
categorical_features = ['multi_screen', 'mail_subscribed']

plt.figure(figsize=(10, 5))
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(1, 2, i)
    sns.countplot(x=feature, hue='churn', data=df)
    plt.title(f'{feature} by Churn')
    plt.xlabel(feature)
    plt.ylabel('Count')
plt.tight_layout()
plt.show()
```

<img width="800" alt="Screenshot 2567-07-18 at 4 17 10 PM" src="https://github.com/user-attachments/assets/3a156b16-c5a7-4339-a845-6d9bb7e6b584">

-Using pair plots to visualize pairwise relationships between features for churned and non-churned customers.

```python
selected_features = ['age', 'no_of_days_subscribed', 'weekly_mins_watched', 'minimum_daily_mins', 'maximum_daily_mins', 'churn']
sns.pairplot(df[selected_features], hue='churn', palette='Set1', diag_kind='kde')
plt.show()
```
<img width="742" alt="Screenshot 2567-07-18 at 4 19 54 PM" src="https://github.com/user-attachments/assets/1e1dac8b-5106-445b-bdd3-6771f3d45f41">

- Correlation with Churn
- Analyze the correlation of each feature with churn.


```python
# Calculate correlation with churn
correlation_with_churn = df.corr()['churn'].sort_values(ascending=False)
print(correlation_with_churn)

# Visualize the correlation
plt.figure(figsize=(10, 6))
sns.barplot(x=correlation_with_churn.index, y=correlation_with_churn.values, palette='viridis')
plt.title('Correlation with Churn')
plt.xlabel('Features')
plt.ylabel('Correlation with Churn')
plt.xticks(rotation=90)
plt.show()
```
<img width="869" alt="Screenshot 2567-07-18 at 4 27 47 PM" src="https://github.com/user-attachments/assets/eb5be8c3-9484-4f9a-aa41-05445337e9d3">

<div align="center">
  <img src="https://github.com/user-attachments/assets/be2a94a6-630c-4806-a85f-38a00d75a93c" alt="giphy" width="500" height="500">
</div>







  





