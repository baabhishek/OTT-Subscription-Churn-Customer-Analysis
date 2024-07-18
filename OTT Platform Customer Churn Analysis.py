#!/usr/bin/env python
# coding: utf-8

# ## OTT Platform Customer Churn Analysis Using Python:

# In[16]:


# import library 

import pandas as pd
import numpy as np               
import seaborn as sns             
import matplotlib.pyplot as plt


# In[17]:


df= pd.read_csv('datasets/customer_churn_data.csv')
df


# In[18]:


#checking the row x columns of dataframe forther understanding of data
df.shape


# In[19]:


df.columns


# #### Information of dataframe:-

# In[20]:


df.info()


# In[21]:


#Null values: (Manually Checking Null/Missing Values)

#gender- (2000- 1976)= 24 columns
#max_day_inactive- (2000- 1972)= 28 columns
#churn - (2000- 1965)= 35 columns


# In[22]:


#Lets verify this:

col_missing_values=df.isna().sum()
print(col_missing_values)


# In[23]:


# Total Missing Values are:

total_missing_values = df.isna().sum().sum()
print(f"Total missing values in the DataFrame: {total_missing_values}")


# - **Columns having missing/ NaN values**
# - gender
# - maximum_days_inactive
# - churn

# In[24]:


# Lets check the descriptive statistical information
df.describe()


# ###### From above data:
# - All data entries are from the year 2015.
# - **Important**: The std dev of a column is 0, If the std of a column is 0, then all the data in that column is same and we can **delete** that column.
# ----
# - The dataset includes 2000 unique customer IDs.
# - The average age of customers is approximately 38.69 years.
# - Ages range from min 18 to max 82 years, with a standard deviation of 10.21 years, showing a fairly broad age distribution.
# ----
# - On average, customers are subscribed for about 99.75 days ~approx 3 months 10 days.
# - The subscription duration varies significantly, from 1 to 243 days, with a standard deviation of around 39.76 days.
# ----
# - Customers watch an average of 270.18 minutes weekly.
# - The watching time ranges from 0 to 526.20 minutes, indicating some customers watch very little or no content while others watch a lot.
# - The standard deviation is 80.55 minutes, indicating considerable variation in watching time.
# ----
# - The minimum daily watching time averages 10.20 minutes.
# - This metric varies from 0 to 20 minutes, with a standard deviation of about 2.79 minutes.
# ----
# - The maximum daily watching time averages 30.62 minutes.
# - This ranges from 0 to 59.64 minutes, with a standard deviation of approximately 9.13 minutes.
# ----
# - On average, customers watch about 100.42 minutes at night per week.
# - This metric ranges from 42 to 175 minutes, with a standard deviation of 19.53 minutes.
# ----
# - Customers watch an average of 4.48 videos.
# - The number of videos watched ranges from 0 to 19, with a standard deviation of 2.49.
# ----
# - On average, customers are inactive for about 3.25 days.
# - The inactivity period ranges from 0 to 6 days, with a standard deviation of 0.81 days.
# ----
# - Customers make an average of 1.55 support calls.
# - The number of support calls ranges from 0 to 9, with a standard deviation of 1.32.
# ----
# - The churn rate is about 13.33% (mean of 0.1333), indicating the proportion of customers who have churned.
# 
# 

# In[25]:


# As Year column standard dev is 0 so no use of so lets delete this 

df.drop(['year'],axis=1,inplace=True)
df.drop(['phone_no'],axis=1,inplace=True)
df


# In[26]:


#Lets check the numerical statistical information, where we can get some more information:
df.describe(include='O')


# - In (df.describe())>> gender column having some null or missing value are there.
# - Male, NO Multi screen category and not subscribe most frequently showing in quantify the no.

# In[27]:


df['churn'].unique()
# 1.0 => customers that unsubscribe/ subscription lapse
# 0.0 => Customers that stay/ continue subscription


# ######  Check for duplicates

# In[28]:


df


# In[29]:


df.duplicated()


# In[30]:


df[df.duplicated()]


# - There is no Duplicate duplicates row from the above code its looks like that.

# ###### Lets check unique values are there in each column:

# In[31]:


# Get the unique values for each column

col_unique_values = df.nunique()
col_unique_values


# In[32]:


# Get the overall unique values in the DataFrame
total_unique_values = df.nunique().sum()
print(f"Total unique values in the DataFrame: {total_unique_values}")


# - Lets check each column unique values: as multiple column are there >
# - Instead of manually checking the unique value lets run a for loop::

# In[33]:


for i in df.columns:
    unique_values = df[i].unique()
    print(f"Unique values for column '{i}': {unique_values}")
    print()  


# - From the above loop we can concludes there are 4 categorical columns:
#     - Gender
#     - Multi_screen
#     - Mail_subscribed
#     - Churn  

# - **Lets understand each column unique value to understand futher analysis**

# In[34]:


df['gender'].value_counts(dropna=False, normalize=True) 
# here data are in float value ensure to check in %age wise:
df_gender = df['gender'].value_counts(dropna=False, normalize=True) * 100
df_gender = df_gender.apply(lambda x: f"{x:.2f}%")

print(df_gender)


# In[35]:


df['multi_screen'].value_counts()


# In[36]:


df['multi_screen'].value_counts(dropna=False)


# In[37]:


df_multiscreen= df['multi_screen'].value_counts(dropna=False, normalize=True) *100
df_multiscreen = df_multiscreen.apply(lambda x: f"{x:.2f}%")

print(df_multiscreen)


# In[38]:


df['mail_subscribed'].value_counts(dropna=False)


# In[39]:


df_mailsubscrib=df['mail_subscribed'].value_counts(dropna=False, normalize=True) * 100
df_mailsubscrib = df_mailsubscrib.apply(lambda x: f"{x:.2f}%")

print(df_mailsubscrib)


# In[40]:


df['churn'].value_counts(dropna=False)


# In[41]:


df['churn'].value_counts(dropna=False, normalize=True) * 100


# In[42]:


df_gender=df['gender'].value_counts(dropna=False,normalize=True)*100
df_gender= df_gender.apply(lambda x: f"{x:.2f}%")
df_gender


# ### Data Cleaning: 

# #### Handling null values:
# 
# - Fill the values
#     - For the numerical columns, we can fill the missing values with the mean/median of that columm
#     - For the categorical columns, we can fill the missing values with mode, least frequent value, or a new category
# - Delete the row(s)
#     - we should avoid deleting the rows as much as possible because it reduces the data
# - Delete the column(s)
#     - if any columns has more than 30-40% values missing, then we can delete that column
#     - this 30-40% value is NOT a hard rule. It can vary depending on the column, data, use case and requirement

# In[43]:


#filling the missing value from NaN to Female as more than 52% already male are there 
# so we can assumed based on probabilities that it can be Female as we can't make 3rd
# catagories as Null, so will check mode for female categories.

df.gender.fillna('Female',inplace=True)


# In[44]:


col_missing_values=df.isna().sum()
print(col_missing_values)


# In[45]:


med_value = int(df.maximum_days_inactive.median())
print(med_value)
#Replacing the median value where NaN/Null value  
df.maximum_days_inactive.fillna(3, inplace=True)


# In[46]:


med_val_churn= int(df.churn.median())
#filling the missing values in the Churn column
df.churn.fillna(0, inplace=True)


# In[65]:


df.isnull().sum()


# In[48]:


df


# - **Data Analysis** 

# In[49]:


# Churn rate Analysis 

churn_rate = df['churn'].mean() * 100
print(f'Churn Rate: {churn_rate:.2f}%')


# In[50]:


# Lets understand the weekly watched analysis by Gender

df.groupby('gender')['weekly_mins_watched'].mean()


# In[51]:


df.sort_values(by='age', ascending=False)


# In[52]:


#Lets filter out how many are having subscription 
subsc= df[df['churn'] == 0]
subsc


# In[53]:


#Lets filter out how many are not having subscription expired 
unsubsc= df[df['churn'] == 1]
unsubsc


# In[54]:


#comparison of Subscriber and unsubsriber 
df['churn'].value_counts(dropna=False, normalize=True) * 100


# In[55]:


df['total_minutes_watched'] = df['weekly_mins_watched'] + df['weekly_max_night_mins']
print(df.head())
df


# In[131]:


# Average age of churned vs. non-churned customers
print(df.groupby('churn')['age'].mean())


# In[130]:


# Average number of days subscribed for churned vs. non-churned customers
print(df.groupby('churn')['no_of_days_subscribed'].mean())


# In[129]:


# Average weekly minutes watched for churned vs. non-churned customers
print(df.groupby('churn')['weekly_mins_watched'].mean())


# In[64]:


df.fillna({'gender': df['gender'].mode().iloc[0], 
           'maximum_days_inactive': df['maximum_days_inactive'].median(),
          'churn': df['churn'].value_counts().idxmin()})


# In[ ]:





# In[128]:


# Lets see the average weekly min watched by gender:

pivot_table = df.pivot_table(index='gender', columns='churn', values='weekly_mins_watched', aggfunc='mean')
print(pivot_table)


# #### Vizuallization:

# In[57]:


sns.countplot(x='gender', data=df)
plt.title('Gender Distribution')


# In[58]:


plt.pie(df['gender'].value_counts(),
       labels=df['gender'].value_counts(dropna=False).index,
       autopct="%.2f%%")
plt.title('Gender Distribution in %')
plt.show()


# In[59]:


df_gender=df['gender'].value_counts(dropna=False,normalize=True)*100
df_gender= df_gender.apply(lambda x: f"{x:.2f}%")
df_gender


# In[133]:


#Gender and Churn
#Exploring the churn rate by gender

plt.figure(figsize=(10, 6))
sns.countplot(x='gender', hue='churn', data=df, palette='Set2')
plt.title('Churn Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()


# In[66]:


# Distribution of age

plt.figure(figsize=(10, 6))
sns.histplot(df['age'], kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[67]:


# Churn rate by gender
plt.figure(figsize=(10, 6))
sns.barplot(x='gender', y='churn', data=df)
plt.title('Churn Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Churn Rate')
plt.show()


# In[68]:


# Churn rate by multi_screen
plt.figure(figsize=(10, 6))
sns.barplot(x='multi_screen', y='churn', data=df)
plt.title('Churn Rate by Multi Screen Subscription')
plt.xlabel('Multi Screen Subscription')
plt.ylabel('Churn Rate')
plt.show()


# In[69]:


# Correlation heatmap
plt.figure(figsize=(12, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()


# In[70]:


#Churn by Weekly Minutes Watched
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='weekly_mins_watched', hue='churn', kde=True)
plt.title('Weekly Minutes Watched by Churn')
plt.xlabel('Weekly Minutes Watched')
plt.ylabel('Frequency')
plt.show()


# In[71]:


# Churn By Cust. supports call
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='customer_support_calls', hue='churn')
plt.title('Churn by Customer Support Calls')
plt.xlabel('Customer Support Calls')
plt.ylabel('Count')
plt.show()


# In[76]:


#Pair Plot for Selected Features
selected_features = ['age', 'no_of_days_subscribed', 'weekly_mins_watched', 'churn']
sns.pairplot(df[selected_features], hue='churn')
plt.show()


# In[77]:


#Box Plot continuous variables for churned vs. non-churned customers 
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


# In[78]:


# Violin plots combine the benefits of box plots and KDE plots
#the distribution of the data across different churn categories.

plt.figure(figsize=(15, 10))
for i, feature in enumerate(continuous_features, 1):
    plt.subplot(2, 3, i)
    sns.violinplot(x='churn', y=feature, data=df, inner='quartile')
    plt.title(f'{feature} by Churn')
    plt.xlabel('Churn')
    plt.ylabel(feature)
plt.tight_layout()
plt.show()



# In[79]:


#Analysis of Categorical Variables
#Let's analyze the relationship between categorical variables such as multi_screen, mail_subscribed, and churn.
# Count Plots- Count plots can help us understand the distribution of 
# categorical variables across churned and non-churned customers.

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


# In[80]:


#Stacked Bar Plots
#Stacked bar plots show the proportion of churn within each category.

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for i, feature in enumerate(categorical_features):
    churn_counts = df.groupby([feature, 'churn']).size().unstack()
    churn_counts.plot(kind='bar', stacked=True, ax=axes[i])
    axes[i].set_title(f'{feature} by Churn')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Count')
plt.tight_layout()
plt.show()


# In[82]:


#Using pair plots to visualize pairwise relationships between features for churned and non-churned customers.

selected_features = ['age', 'no_of_days_subscribed', 'weekly_mins_watched', 'minimum_daily_mins', 'maximum_daily_mins', 'churn']
sns.pairplot(df[selected_features], hue='churn', palette='Set1', diag_kind='kde')
plt.show()


# In[83]:


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


# In[89]:


#Subscription Duration Analysis
#Analyzing the duration of subscriptions can provide insights into 
#how long customers typically remain subscribed

plt.figure(figsize=(10, 6))
df['no_of_days_subscribed'].plot(kind='hist', bins=20, color='lightgreen', edgecolor='black')
plt.title('Subscription Duration Distribution')
plt.xlabel('Number of Days Subscribed')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[91]:


#Multi-Screen Subscription Analysis
#Checking the proportion of customers who have multi-screen subscriptions.
plt.figure(figsize=(6, 6))
plt.pie(df['multi_screen'].value_counts(), labels=['No', 'Yes'], autopct='%1.1f%%')
plt.title('Multi-Screen Subscription Distribution')
plt.show()


# In[93]:


#Mail Subscription Analysis
#Analyzing the proportion of customers who have subscribed to email notifications.


plt.figure(figsize=(6, 6))
plt.pie(df['mail_subscribed'].value_counts(), labels=['No', 'Yes'], autopct='%1.1f%%')
plt.title('Mail Subscription Distribution')
plt.show()


# In[103]:


#Churn Rate Analysis
#Understanding the churn rate in your customer base.

plt.figure(figsize=(6, 6))
plt.pie(df['churn'].value_counts(), labels=['Not Churned', 'Churned'], autopct='%1.1f%%')
plt.title('Churn Rate')
plt.show()


# In[95]:


#Relationship Between Maximum Days Inactive and Churn
#Exploring the relationship between the number of maximum days a customer was inactive and whether they churned.

plt.figure(figsize=(10, 6))
sns.boxplot(x='churn', y='maximum_days_inactive', data=df)
plt.title('Maximum Days Inactive vs Churn')
plt.xlabel('Churn')
plt.ylabel('Maximum Days Inactive')
plt.show()


# In[96]:


#Customer Support Calls and Churn
#Analyzing the relationship between the number of customer support calls and churn.

plt.figure(figsize=(10, 6))
sns.boxplot(x='churn', y='customer_support_calls', data=df)
plt.title('Customer Support Calls vs Churn')
plt.xlabel('Churn')
plt.ylabel('Customer Support Calls')
plt.show()


# In[97]:


#Videos Watched and Churn
#Examining the relationship between the number of videos watched and churn.

plt.figure(figsize=(10, 6))
sns.boxplot(x='churn', y='videos_watched', data=df)
plt.title('Videos Watched vs Churn')
plt.xlabel('Churn')
plt.ylabel('Videos Watched')
plt.show()


# In[105]:


#Monthly Engagement Analysis
#Analyzing the monthly engagement of users by looking at the number of videos watched.

plt.figure(figsize=(10, 6))
sns.lineplot(x='no_of_days_subscribed', y='videos_watched', hue='churn', data=df, palette='coolwarm')
plt.title('Monthly Engagement: Videos Watched vs Subscription Duration')
plt.xlabel('Number of Days Subscribed')
plt.ylabel('Videos Watched')
plt.legend(title='Churn')
plt.show()


# In[101]:


#Weekly Maximum Night Minutes and Churn
#Examining the relationship between the maximum night minutes watched in a week and churn.

plt.figure(figsize=(10, 6))
sns.boxplot(x='churn', y='weekly_max_night_mins', data=df)
plt.title('Weekly Maximum Night Minutes vs Churn')
plt.xlabel('Churn')
plt.ylabel('Weekly Maximum Night Minutes')
plt.show()


# In[ ]:




