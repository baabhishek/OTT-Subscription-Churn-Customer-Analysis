## OTT Platform Customer Churn Analysis Using Python:


```python
# import library 

import pandas as pd
import numpy as np               
import seaborn as sns             
import matplotlib.pyplot as plt

```


```python
df= pd.read_csv('datasets/customer_churn_data.csv')
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>customer_id</th>
      <th>phone_no</th>
      <th>gender</th>
      <th>age</th>
      <th>no_of_days_subscribed</th>
      <th>multi_screen</th>
      <th>mail_subscribed</th>
      <th>weekly_mins_watched</th>
      <th>minimum_daily_mins</th>
      <th>maximum_daily_mins</th>
      <th>weekly_max_night_mins</th>
      <th>videos_watched</th>
      <th>maximum_days_inactive</th>
      <th>customer_support_calls</th>
      <th>churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015</td>
      <td>100198</td>
      <td>409-8743</td>
      <td>Female</td>
      <td>36</td>
      <td>62</td>
      <td>no</td>
      <td>no</td>
      <td>148.35</td>
      <td>12.2</td>
      <td>16.81</td>
      <td>82</td>
      <td>1</td>
      <td>4.0</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015</td>
      <td>100643</td>
      <td>340-5930</td>
      <td>Female</td>
      <td>39</td>
      <td>149</td>
      <td>no</td>
      <td>no</td>
      <td>294.45</td>
      <td>7.7</td>
      <td>33.37</td>
      <td>87</td>
      <td>3</td>
      <td>3.0</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015</td>
      <td>100756</td>
      <td>372-3750</td>
      <td>Female</td>
      <td>65</td>
      <td>126</td>
      <td>no</td>
      <td>no</td>
      <td>87.30</td>
      <td>11.9</td>
      <td>9.89</td>
      <td>91</td>
      <td>1</td>
      <td>4.0</td>
      <td>5</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015</td>
      <td>101595</td>
      <td>331-4902</td>
      <td>Female</td>
      <td>24</td>
      <td>131</td>
      <td>no</td>
      <td>yes</td>
      <td>321.30</td>
      <td>9.5</td>
      <td>36.41</td>
      <td>102</td>
      <td>4</td>
      <td>3.0</td>
      <td>3</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015</td>
      <td>101653</td>
      <td>351-8398</td>
      <td>Female</td>
      <td>40</td>
      <td>191</td>
      <td>no</td>
      <td>no</td>
      <td>243.00</td>
      <td>10.9</td>
      <td>27.54</td>
      <td>83</td>
      <td>7</td>
      <td>3.0</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>2015</td>
      <td>997132</td>
      <td>385-7387</td>
      <td>Female</td>
      <td>54</td>
      <td>75</td>
      <td>no</td>
      <td>yes</td>
      <td>182.25</td>
      <td>11.3</td>
      <td>20.66</td>
      <td>97</td>
      <td>5</td>
      <td>4.0</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>2015</td>
      <td>998086</td>
      <td>383-9255</td>
      <td>Male</td>
      <td>45</td>
      <td>127</td>
      <td>no</td>
      <td>no</td>
      <td>273.45</td>
      <td>9.3</td>
      <td>30.99</td>
      <td>116</td>
      <td>3</td>
      <td>3.0</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>2015</td>
      <td>998474</td>
      <td>353-2080</td>
      <td>NaN</td>
      <td>53</td>
      <td>94</td>
      <td>no</td>
      <td>no</td>
      <td>128.85</td>
      <td>15.6</td>
      <td>14.60</td>
      <td>110</td>
      <td>16</td>
      <td>5.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>2015</td>
      <td>998934</td>
      <td>359-7788</td>
      <td>Male</td>
      <td>40</td>
      <td>94</td>
      <td>no</td>
      <td>no</td>
      <td>178.05</td>
      <td>10.4</td>
      <td>20.18</td>
      <td>100</td>
      <td>6</td>
      <td>NaN</td>
      <td>3</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>2015</td>
      <td>999961</td>
      <td>414-1496</td>
      <td>Male</td>
      <td>37</td>
      <td>73</td>
      <td>no</td>
      <td>no</td>
      <td>326.70</td>
      <td>10.3</td>
      <td>37.03</td>
      <td>89</td>
      <td>6</td>
      <td>3.0</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>2000 rows × 16 columns</p>
</div>




```python
#checking the row x columns of dataframe forther understanding of data
df.shape

```




    (2000, 16)




```python
df.columns
```




    Index(['year', 'customer_id', 'phone_no', 'gender', 'age',
           'no_of_days_subscribed', 'multi_screen', 'mail_subscribed',
           'weekly_mins_watched', 'minimum_daily_mins', 'maximum_daily_mins',
           'weekly_max_night_mins', 'videos_watched', 'maximum_days_inactive',
           'customer_support_calls', 'churn'],
          dtype='object')



#### Information of dataframe:-


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2000 entries, 0 to 1999
    Data columns (total 16 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   year                    2000 non-null   int64  
     1   customer_id             2000 non-null   int64  
     2   phone_no                2000 non-null   object 
     3   gender                  1976 non-null   object 
     4   age                     2000 non-null   int64  
     5   no_of_days_subscribed   2000 non-null   int64  
     6   multi_screen            2000 non-null   object 
     7   mail_subscribed         2000 non-null   object 
     8   weekly_mins_watched     2000 non-null   float64
     9   minimum_daily_mins      2000 non-null   float64
     10  maximum_daily_mins      2000 non-null   float64
     11  weekly_max_night_mins   2000 non-null   int64  
     12  videos_watched          2000 non-null   int64  
     13  maximum_days_inactive   1972 non-null   float64
     14  customer_support_calls  2000 non-null   int64  
     15  churn                   1965 non-null   float64
    dtypes: float64(5), int64(7), object(4)
    memory usage: 250.1+ KB



```python
#Null values: (Manually Checking Null/Missing Values)

#gender- (2000- 1976)= 24 columns
#max_day_inactive- (2000- 1972)= 28 columns
#churn - (2000- 1965)= 35 columns
```


```python
#Lets verify this:

col_missing_values=df.isna().sum()
print(col_missing_values)
```

    year                       0
    customer_id                0
    phone_no                   0
    gender                    24
    age                        0
    no_of_days_subscribed      0
    multi_screen               0
    mail_subscribed            0
    weekly_mins_watched        0
    minimum_daily_mins         0
    maximum_daily_mins         0
    weekly_max_night_mins      0
    videos_watched             0
    maximum_days_inactive     28
    customer_support_calls     0
    churn                     35
    dtype: int64



```python
# Total Missing Values are:

total_missing_values = df.isna().sum().sum()
print(f"Total missing values in the DataFrame: {total_missing_values}")
```

    Total missing values in the DataFrame: 87


- **Columns having missing/ NaN values**
- gender
- maximum_days_inactive
- churn


```python
# Lets check the descriptive statistical information
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>customer_id</th>
      <th>age</th>
      <th>no_of_days_subscribed</th>
      <th>weekly_mins_watched</th>
      <th>minimum_daily_mins</th>
      <th>maximum_daily_mins</th>
      <th>weekly_max_night_mins</th>
      <th>videos_watched</th>
      <th>maximum_days_inactive</th>
      <th>customer_support_calls</th>
      <th>churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2000.0</td>
      <td>2000.000000</td>
      <td>2000.00000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>1972.000000</td>
      <td>2000.000000</td>
      <td>1965.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2015.0</td>
      <td>554887.157500</td>
      <td>38.69050</td>
      <td>99.750000</td>
      <td>270.178425</td>
      <td>10.198700</td>
      <td>30.620780</td>
      <td>100.415500</td>
      <td>4.482500</td>
      <td>3.250507</td>
      <td>1.547000</td>
      <td>0.133333</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.0</td>
      <td>261033.690318</td>
      <td>10.20641</td>
      <td>39.755386</td>
      <td>80.551627</td>
      <td>2.785519</td>
      <td>9.129165</td>
      <td>19.529454</td>
      <td>2.487728</td>
      <td>0.809084</td>
      <td>1.315164</td>
      <td>0.340021</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2015.0</td>
      <td>100198.000000</td>
      <td>18.00000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>42.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2015.0</td>
      <td>328634.750000</td>
      <td>32.00000</td>
      <td>73.000000</td>
      <td>218.212500</td>
      <td>8.400000</td>
      <td>24.735000</td>
      <td>87.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2015.0</td>
      <td>567957.500000</td>
      <td>37.00000</td>
      <td>99.000000</td>
      <td>269.925000</td>
      <td>10.200000</td>
      <td>30.590000</td>
      <td>101.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2015.0</td>
      <td>773280.250000</td>
      <td>44.00000</td>
      <td>127.000000</td>
      <td>324.675000</td>
      <td>12.000000</td>
      <td>36.797500</td>
      <td>114.000000</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2015.0</td>
      <td>999961.000000</td>
      <td>82.00000</td>
      <td>243.000000</td>
      <td>526.200000</td>
      <td>20.000000</td>
      <td>59.640000</td>
      <td>175.000000</td>
      <td>19.000000</td>
      <td>6.000000</td>
      <td>9.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



###### From above data:
- All data entries are from the year 2015.
- **Important**: The std dev of a column is 0, If the std of a column is 0, then all the data in that column is same and we can **delete** that column.
----
- The dataset includes 2000 unique customer IDs.
- The average age of customers is approximately 38.69 years.
- Ages range from min 18 to max 82 years, with a standard deviation of 10.21 years, showing a fairly broad age distribution.
----
- On average, customers are subscribed for about 99.75 days ~approx 3 months 10 days.
- The subscription duration varies significantly, from 1 to 243 days, with a standard deviation of around 39.76 days.
----
- Customers watch an average of 270.18 minutes weekly.
- The watching time ranges from 0 to 526.20 minutes, indicating some customers watch very little or no content while others watch a lot.
- The standard deviation is 80.55 minutes, indicating considerable variation in watching time.
----
- The minimum daily watching time averages 10.20 minutes.
- This metric varies from 0 to 20 minutes, with a standard deviation of about 2.79 minutes.
----
- The maximum daily watching time averages 30.62 minutes.
- This ranges from 0 to 59.64 minutes, with a standard deviation of approximately 9.13 minutes.
----
- On average, customers watch about 100.42 minutes at night per week.
- This metric ranges from 42 to 175 minutes, with a standard deviation of 19.53 minutes.
----
- Customers watch an average of 4.48 videos.
- The number of videos watched ranges from 0 to 19, with a standard deviation of 2.49.
----
- On average, customers are inactive for about 3.25 days.
- The inactivity period ranges from 0 to 6 days, with a standard deviation of 0.81 days.
----
- Customers make an average of 1.55 support calls.
- The number of support calls ranges from 0 to 9, with a standard deviation of 1.32.
----
- The churn rate is about 13.33% (mean of 0.1333), indicating the proportion of customers who have churned.




```python
# As Year column standard dev is 0 so no use of so lets delete this 

df.drop(['year'],axis=1,inplace=True)
df.drop(['phone_no'],axis=1,inplace=True)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customer_id</th>
      <th>gender</th>
      <th>age</th>
      <th>no_of_days_subscribed</th>
      <th>multi_screen</th>
      <th>mail_subscribed</th>
      <th>weekly_mins_watched</th>
      <th>minimum_daily_mins</th>
      <th>maximum_daily_mins</th>
      <th>weekly_max_night_mins</th>
      <th>videos_watched</th>
      <th>maximum_days_inactive</th>
      <th>customer_support_calls</th>
      <th>churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100198</td>
      <td>Female</td>
      <td>36</td>
      <td>62</td>
      <td>no</td>
      <td>no</td>
      <td>148.35</td>
      <td>12.2</td>
      <td>16.81</td>
      <td>82</td>
      <td>1</td>
      <td>4.0</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100643</td>
      <td>Female</td>
      <td>39</td>
      <td>149</td>
      <td>no</td>
      <td>no</td>
      <td>294.45</td>
      <td>7.7</td>
      <td>33.37</td>
      <td>87</td>
      <td>3</td>
      <td>3.0</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100756</td>
      <td>Female</td>
      <td>65</td>
      <td>126</td>
      <td>no</td>
      <td>no</td>
      <td>87.30</td>
      <td>11.9</td>
      <td>9.89</td>
      <td>91</td>
      <td>1</td>
      <td>4.0</td>
      <td>5</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>101595</td>
      <td>Female</td>
      <td>24</td>
      <td>131</td>
      <td>no</td>
      <td>yes</td>
      <td>321.30</td>
      <td>9.5</td>
      <td>36.41</td>
      <td>102</td>
      <td>4</td>
      <td>3.0</td>
      <td>3</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>101653</td>
      <td>Female</td>
      <td>40</td>
      <td>191</td>
      <td>no</td>
      <td>no</td>
      <td>243.00</td>
      <td>10.9</td>
      <td>27.54</td>
      <td>83</td>
      <td>7</td>
      <td>3.0</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>997132</td>
      <td>Female</td>
      <td>54</td>
      <td>75</td>
      <td>no</td>
      <td>yes</td>
      <td>182.25</td>
      <td>11.3</td>
      <td>20.66</td>
      <td>97</td>
      <td>5</td>
      <td>4.0</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>998086</td>
      <td>Male</td>
      <td>45</td>
      <td>127</td>
      <td>no</td>
      <td>no</td>
      <td>273.45</td>
      <td>9.3</td>
      <td>30.99</td>
      <td>116</td>
      <td>3</td>
      <td>3.0</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>998474</td>
      <td>NaN</td>
      <td>53</td>
      <td>94</td>
      <td>no</td>
      <td>no</td>
      <td>128.85</td>
      <td>15.6</td>
      <td>14.60</td>
      <td>110</td>
      <td>16</td>
      <td>5.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>998934</td>
      <td>Male</td>
      <td>40</td>
      <td>94</td>
      <td>no</td>
      <td>no</td>
      <td>178.05</td>
      <td>10.4</td>
      <td>20.18</td>
      <td>100</td>
      <td>6</td>
      <td>NaN</td>
      <td>3</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>999961</td>
      <td>Male</td>
      <td>37</td>
      <td>73</td>
      <td>no</td>
      <td>no</td>
      <td>326.70</td>
      <td>10.3</td>
      <td>37.03</td>
      <td>89</td>
      <td>6</td>
      <td>3.0</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>2000 rows × 14 columns</p>
</div>




```python
#Lets check the numerical statistical information, where we can get some more information:
df.describe(include='O')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>multi_screen</th>
      <th>mail_subscribed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1976</td>
      <td>2000</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Male</td>
      <td>no</td>
      <td>no</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1053</td>
      <td>1802</td>
      <td>1430</td>
    </tr>
  </tbody>
</table>
</div>



- In (df.describe())>> gender column having some null or missing value are there.
- Male, NO Multi screen category and not subscribe most frequently showing in quantify the no.


```python
df['churn'].unique()
# 1.0 => customers that unsubscribe/ subscription lapse
# 0.0 => Customers that stay/ continue subscription
```




    array([ 0.,  1., nan])



######  Check for duplicates


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customer_id</th>
      <th>gender</th>
      <th>age</th>
      <th>no_of_days_subscribed</th>
      <th>multi_screen</th>
      <th>mail_subscribed</th>
      <th>weekly_mins_watched</th>
      <th>minimum_daily_mins</th>
      <th>maximum_daily_mins</th>
      <th>weekly_max_night_mins</th>
      <th>videos_watched</th>
      <th>maximum_days_inactive</th>
      <th>customer_support_calls</th>
      <th>churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100198</td>
      <td>Female</td>
      <td>36</td>
      <td>62</td>
      <td>no</td>
      <td>no</td>
      <td>148.35</td>
      <td>12.2</td>
      <td>16.81</td>
      <td>82</td>
      <td>1</td>
      <td>4.0</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100643</td>
      <td>Female</td>
      <td>39</td>
      <td>149</td>
      <td>no</td>
      <td>no</td>
      <td>294.45</td>
      <td>7.7</td>
      <td>33.37</td>
      <td>87</td>
      <td>3</td>
      <td>3.0</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100756</td>
      <td>Female</td>
      <td>65</td>
      <td>126</td>
      <td>no</td>
      <td>no</td>
      <td>87.30</td>
      <td>11.9</td>
      <td>9.89</td>
      <td>91</td>
      <td>1</td>
      <td>4.0</td>
      <td>5</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>101595</td>
      <td>Female</td>
      <td>24</td>
      <td>131</td>
      <td>no</td>
      <td>yes</td>
      <td>321.30</td>
      <td>9.5</td>
      <td>36.41</td>
      <td>102</td>
      <td>4</td>
      <td>3.0</td>
      <td>3</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>101653</td>
      <td>Female</td>
      <td>40</td>
      <td>191</td>
      <td>no</td>
      <td>no</td>
      <td>243.00</td>
      <td>10.9</td>
      <td>27.54</td>
      <td>83</td>
      <td>7</td>
      <td>3.0</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>997132</td>
      <td>Female</td>
      <td>54</td>
      <td>75</td>
      <td>no</td>
      <td>yes</td>
      <td>182.25</td>
      <td>11.3</td>
      <td>20.66</td>
      <td>97</td>
      <td>5</td>
      <td>4.0</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>998086</td>
      <td>Male</td>
      <td>45</td>
      <td>127</td>
      <td>no</td>
      <td>no</td>
      <td>273.45</td>
      <td>9.3</td>
      <td>30.99</td>
      <td>116</td>
      <td>3</td>
      <td>3.0</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>998474</td>
      <td>NaN</td>
      <td>53</td>
      <td>94</td>
      <td>no</td>
      <td>no</td>
      <td>128.85</td>
      <td>15.6</td>
      <td>14.60</td>
      <td>110</td>
      <td>16</td>
      <td>5.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>998934</td>
      <td>Male</td>
      <td>40</td>
      <td>94</td>
      <td>no</td>
      <td>no</td>
      <td>178.05</td>
      <td>10.4</td>
      <td>20.18</td>
      <td>100</td>
      <td>6</td>
      <td>NaN</td>
      <td>3</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>999961</td>
      <td>Male</td>
      <td>37</td>
      <td>73</td>
      <td>no</td>
      <td>no</td>
      <td>326.70</td>
      <td>10.3</td>
      <td>37.03</td>
      <td>89</td>
      <td>6</td>
      <td>3.0</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>2000 rows × 14 columns</p>
</div>




```python
df.duplicated()
```




    0       False
    1       False
    2       False
    3       False
    4       False
            ...  
    1995    False
    1996    False
    1997    False
    1998    False
    1999    False
    Length: 2000, dtype: bool




```python
df[df.duplicated()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customer_id</th>
      <th>gender</th>
      <th>age</th>
      <th>no_of_days_subscribed</th>
      <th>multi_screen</th>
      <th>mail_subscribed</th>
      <th>weekly_mins_watched</th>
      <th>minimum_daily_mins</th>
      <th>maximum_daily_mins</th>
      <th>weekly_max_night_mins</th>
      <th>videos_watched</th>
      <th>maximum_days_inactive</th>
      <th>customer_support_calls</th>
      <th>churn</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



- There is no Duplicate duplicates row from the above code its looks like that.

###### Lets check unique values are there in each column:


```python
# Get the unique values for each column

col_unique_values = df.nunique()
col_unique_values
```




    customer_id               1999
    gender                       2
    age                         63
    no_of_days_subscribed      204
    multi_screen                 2
    mail_subscribed              2
    weekly_mins_watched       1260
    minimum_daily_mins         149
    maximum_daily_mins        1260
    weekly_max_night_mins      111
    videos_watched              19
    maximum_days_inactive        7
    customer_support_calls      10
    churn                        2
    dtype: int64




```python
# Get the overall unique values in the DataFrame
total_unique_values = df.nunique().sum()
print(f"Total unique values in the DataFrame: {total_unique_values}")
```

    Total unique values in the DataFrame: 5090


- Lets check each column unique values: as multiple column are there >
- Instead of manually checking the unique value lets run a for loop::


```python
for i in df.columns:
    unique_values = df[i].unique()
    print(f"Unique values for column '{i}': {unique_values}")
    print()  
```

    Unique values for column 'customer_id': [100198 100643 100756 ... 998474 998934 999961]
    
    Unique values for column 'gender': ['Female' nan 'Male']
    
    Unique values for column 'age': [36 39 65 24 40 31 54 61 34 30 23 21 44 45 59 57 32 50 28 37 63 33 35 52
     48 55 41 43 38 26 29 27 56 49 47 42 67 46 64 66 22 51 25 62 53 19 76 60
     58 75 20 74 77 82 80 71 73 18 70 72 69 68 79]
    
    Unique values for column 'no_of_days_subscribed': [ 62 149 126 131 191  65  59  50 205  63 114 107  84 142 137 100  99 115
     194 104 170  76  94  81 119 138 124  48 106 161  89 105  67 129  56  95
      37  80 190 168 166  42  96  77  54  41  53  98  90 146  51  97  55  74
      86  75  27 163  44  92  45 140 122 132 103 134 109  21 167  35 177 118
      87 135 144 148  70  16  83  72  30  93  91 110 128  66  38  64 172  82
      39 108 162  10 156 101  68 111 112 127  61 158 192 136 116  49  40 130
     125   9  57  88 151 113  58  79 179  17 117 152 159 155  43 123   5 186
     139  19 147  34  73 121  78 153 171  85  29 102 165  36 150  31 201 174
     141 157 178  18 120  22 176 182 189 154   1 143  24   2   3 181 164 193
     210  71  13  69 195 173  52 133  46 184  47 145 197  25 199  60  32  33
      28 169  12 232  26 160  11 225 224  20 212 185 215  15  23   6 243 217
     209   7 200 180 196 208]
    
    Unique values for column 'multi_screen': ['no' 'yes']
    
    Unique values for column 'mail_subscribed': ['no' 'yes']
    
    Unique values for column 'weekly_mins_watched': [148.35 294.45  87.3  ... 182.25 128.85 178.05]
    
    Unique values for column 'minimum_daily_mins': [12.2  7.7 11.9  9.5 10.9 12.7 10.2  5.6  7.8 12.3  8.4  7.3 11.1 12.1
      7.2 11.4 13.7 18.2 10.7  9.1 13.4  9.2 14.7  8.7 15.  11.3  7.6 11.
      5.4  5.2  4.9 14.3 10.1  6.7 12.9  8.9  8.8 10.6 11.5 12.   7.5 10.
     13.1 10.4 14.1  7.4  8.3 12.5 14.6 13.3  9.9  9.6  9.4 13.6 11.6  9.7
      8.  11.8 10.8  4.2  7.9 12.6 13.2  8.2  6.8  9.8 14.8  8.1 14.5 10.5
      5.5 10.3  8.6  8.5  3.9 13.9  1.3  9.3  6.4 13.   6.6 11.2  0.  11.7
      6.2 14.4  5.8  5.9 12.4 16.4  6.3  5.1 16.7  6.1 15.5 14.2 16.9 18.
      5.3 15.4  4.1 12.8  7.  17.5  9.   6.9  6.5  5.  16.5  6.  15.1  5.7
      4.4 15.6 15.3  4.5 14.  15.7  7.1 13.5 13.8 16.  15.2 14.9  3.8  4.8
      2.2 17.6  4.7 15.9  3.5  4.6 16.3  3.7 15.8 18.9  2.  20.  17.9  4.
     17.2 16.1 17.1 17.3 18.4  2.7  3.6 16.2 17. ]
    
    Unique values for column 'maximum_daily_mins': [16.81 33.37  9.89 ... 20.66 14.6  20.18]
    
    Unique values for column 'weekly_max_night_mins': [ 82  87  91 102  83 111 106  88  64  58 100  79 134  96 130 117 124  95
     101 131 103  50 107 125  81 128  70 109 104  72 115  97  74 123  93  84
     108  66  76 110  98  92 121  71  86  77 119 135  94  73  78 114  68 155
      99  89  80 127 116 137  75 105  57 157 142 113  49 112  85 118  61 151
     136 146  90  63  67  53 144  69  60 122 126  62 138 141 120 129 133 139
      59 132  55 143  54 153 140 147  65 145  42  56 152 150 148 158 154  51
      46 175  44]
    
    Unique values for column 'videos_watched': [ 1  3  4  7  6  9  5  2  8 10 14  0 11 13 18 15 12 19 16]
    
    Unique values for column 'maximum_days_inactive': [ 4.  3. nan  2.  5.  1.  0.  6.]
    
    Unique values for column 'customer_support_calls': [1 2 5 3 4 0 7 8 6 9]
    
    Unique values for column 'churn': [ 0.  1. nan]
    


- From the above loop we can concludes there are 4 categorical columns:
    - Gender
    - Multi_screen
    - Mail_subscribed
    - Churn  

- **Lets understand each column unique value to understand futher analysis**


```python
df['gender'].value_counts(dropna=False, normalize=True) 
# here data are in float value ensure to check in %age wise:
df_gender = df['gender'].value_counts(dropna=False, normalize=True) * 100
df_gender = df_gender.apply(lambda x: f"{x:.2f}%")

print(df_gender)
```

    Male      52.65%
    Female    46.15%
    NaN        1.20%
    Name: gender, dtype: object



```python
df['multi_screen'].value_counts()
```




    no     1802
    yes     198
    Name: multi_screen, dtype: int64




```python
df['multi_screen'].value_counts(dropna=False)
```




    no     1802
    yes     198
    Name: multi_screen, dtype: int64




```python
df_multiscreen= df['multi_screen'].value_counts(dropna=False, normalize=True) *100
df_multiscreen = df_multiscreen.apply(lambda x: f"{x:.2f}%")

print(df_multiscreen)
```

    no     90.10%
    yes     9.90%
    Name: multi_screen, dtype: object



```python
df['mail_subscribed'].value_counts(dropna=False)
```




    no     1430
    yes     570
    Name: mail_subscribed, dtype: int64




```python
df_mailsubscrib=df['mail_subscribed'].value_counts(dropna=False, normalize=True) * 100
df_mailsubscrib = df_mailsubscrib.apply(lambda x: f"{x:.2f}%")

print(df_mailsubscrib)
```

    no     71.50%
    yes    28.50%
    Name: mail_subscribed, dtype: object



```python
df['churn'].value_counts(dropna=False)
```




    0.0    1703
    1.0     262
    NaN      35
    Name: churn, dtype: int64




```python
df['churn'].value_counts(dropna=False, normalize=True) * 100
```




    0.0    85.15
    1.0    13.10
    NaN     1.75
    Name: churn, dtype: float64




```python
df_gender=df['gender'].value_counts(dropna=False,normalize=True)*100
df_gender= df_gender.apply(lambda x: f"{x:.2f}%")
df_gender
```




    Male      52.65%
    Female    46.15%
    NaN        1.20%
    Name: gender, dtype: object



### Data Cleaning: 

#### Handling null values:

- Fill the values
    - For the numerical columns, we can fill the missing values with the mean/median of that columm
    - For the categorical columns, we can fill the missing values with mode, least frequent value, or a new category
- Delete the row(s)
    - we should avoid deleting the rows as much as possible because it reduces the data
- Delete the column(s)
    - if any columns has more than 30-40% values missing, then we can delete that column
    - this 30-40% value is NOT a hard rule. It can vary depending on the column, data, use case and requirement


```python
#filling the missing value from NaN to Female as more than 52% already male are there 
# so we can assumed based on probabilities that it can be Female as we can't make 3rd
# catagories as Null, so will check mode for female categories.

df.gender.fillna('Female',inplace=True)
```


```python
col_missing_values=df.isna().sum()
print(col_missing_values)
```

    customer_id                0
    gender                     0
    age                        0
    no_of_days_subscribed      0
    multi_screen               0
    mail_subscribed            0
    weekly_mins_watched        0
    minimum_daily_mins         0
    maximum_daily_mins         0
    weekly_max_night_mins      0
    videos_watched             0
    maximum_days_inactive     28
    customer_support_calls     0
    churn                     35
    dtype: int64



```python
med_value = int(df.maximum_days_inactive.median())
print(med_value)
#Replacing the median value where NaN/Null value  
df.maximum_days_inactive.fillna(3, inplace=True)
```

    3



```python
med_val_churn= int(df.churn.median())
#filling the missing values in the Churn column
df.churn.fillna(0, inplace=True)
```


```python
df.isnull().sum()
```




    customer_id               0
    gender                    0
    age                       0
    no_of_days_subscribed     0
    multi_screen              0
    mail_subscribed           0
    weekly_mins_watched       0
    minimum_daily_mins        0
    maximum_daily_mins        0
    weekly_max_night_mins     0
    videos_watched            0
    maximum_days_inactive     0
    customer_support_calls    0
    churn                     0
    total_minutes_watched     0
    dtype: int64




```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customer_id</th>
      <th>gender</th>
      <th>age</th>
      <th>no_of_days_subscribed</th>
      <th>multi_screen</th>
      <th>mail_subscribed</th>
      <th>weekly_mins_watched</th>
      <th>minimum_daily_mins</th>
      <th>maximum_daily_mins</th>
      <th>weekly_max_night_mins</th>
      <th>videos_watched</th>
      <th>maximum_days_inactive</th>
      <th>customer_support_calls</th>
      <th>churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100198</td>
      <td>Female</td>
      <td>36</td>
      <td>62</td>
      <td>no</td>
      <td>no</td>
      <td>148.35</td>
      <td>12.2</td>
      <td>16.81</td>
      <td>82</td>
      <td>1</td>
      <td>4.0</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100643</td>
      <td>Female</td>
      <td>39</td>
      <td>149</td>
      <td>no</td>
      <td>no</td>
      <td>294.45</td>
      <td>7.7</td>
      <td>33.37</td>
      <td>87</td>
      <td>3</td>
      <td>3.0</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100756</td>
      <td>Female</td>
      <td>65</td>
      <td>126</td>
      <td>no</td>
      <td>no</td>
      <td>87.30</td>
      <td>11.9</td>
      <td>9.89</td>
      <td>91</td>
      <td>1</td>
      <td>4.0</td>
      <td>5</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>101595</td>
      <td>Female</td>
      <td>24</td>
      <td>131</td>
      <td>no</td>
      <td>yes</td>
      <td>321.30</td>
      <td>9.5</td>
      <td>36.41</td>
      <td>102</td>
      <td>4</td>
      <td>3.0</td>
      <td>3</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>101653</td>
      <td>Female</td>
      <td>40</td>
      <td>191</td>
      <td>no</td>
      <td>no</td>
      <td>243.00</td>
      <td>10.9</td>
      <td>27.54</td>
      <td>83</td>
      <td>7</td>
      <td>3.0</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>997132</td>
      <td>Female</td>
      <td>54</td>
      <td>75</td>
      <td>no</td>
      <td>yes</td>
      <td>182.25</td>
      <td>11.3</td>
      <td>20.66</td>
      <td>97</td>
      <td>5</td>
      <td>4.0</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>998086</td>
      <td>Male</td>
      <td>45</td>
      <td>127</td>
      <td>no</td>
      <td>no</td>
      <td>273.45</td>
      <td>9.3</td>
      <td>30.99</td>
      <td>116</td>
      <td>3</td>
      <td>3.0</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>998474</td>
      <td>Female</td>
      <td>53</td>
      <td>94</td>
      <td>no</td>
      <td>no</td>
      <td>128.85</td>
      <td>15.6</td>
      <td>14.60</td>
      <td>110</td>
      <td>16</td>
      <td>5.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>998934</td>
      <td>Male</td>
      <td>40</td>
      <td>94</td>
      <td>no</td>
      <td>no</td>
      <td>178.05</td>
      <td>10.4</td>
      <td>20.18</td>
      <td>100</td>
      <td>6</td>
      <td>3.0</td>
      <td>3</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>999961</td>
      <td>Male</td>
      <td>37</td>
      <td>73</td>
      <td>no</td>
      <td>no</td>
      <td>326.70</td>
      <td>10.3</td>
      <td>37.03</td>
      <td>89</td>
      <td>6</td>
      <td>3.0</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>2000 rows × 14 columns</p>
</div>



- **Data Analysis** 


```python
# Churn rate Analysis 

churn_rate = df['churn'].mean() * 100
print(f'Churn Rate: {churn_rate:.2f}%')

```

    Churn Rate: 13.10%



```python
# Lets understand the weekly watched analysis by Gender

df.groupby('gender')['weekly_mins_watched'].mean()
```




    gender
    Female    269.628881
    Male      270.672650
    Name: weekly_mins_watched, dtype: float64




```python
df.sort_values(by='age', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customer_id</th>
      <th>gender</th>
      <th>age</th>
      <th>no_of_days_subscribed</th>
      <th>multi_screen</th>
      <th>mail_subscribed</th>
      <th>weekly_mins_watched</th>
      <th>minimum_daily_mins</th>
      <th>maximum_daily_mins</th>
      <th>weekly_max_night_mins</th>
      <th>videos_watched</th>
      <th>maximum_days_inactive</th>
      <th>customer_support_calls</th>
      <th>churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>682</th>
      <td>417761</td>
      <td>Female</td>
      <td>82</td>
      <td>122</td>
      <td>yes</td>
      <td>no</td>
      <td>346.35</td>
      <td>11.0</td>
      <td>39.25</td>
      <td>57</td>
      <td>2</td>
      <td>3.0</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>832</th>
      <td>490698</td>
      <td>Female</td>
      <td>80</td>
      <td>62</td>
      <td>no</td>
      <td>no</td>
      <td>373.05</td>
      <td>13.2</td>
      <td>42.28</td>
      <td>78</td>
      <td>2</td>
      <td>4.0</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1426</th>
      <td>742373</td>
      <td>Female</td>
      <td>79</td>
      <td>82</td>
      <td>no</td>
      <td>no</td>
      <td>310.50</td>
      <td>9.1</td>
      <td>35.19</td>
      <td>108</td>
      <td>8</td>
      <td>3.0</td>
      <td>3</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>577</th>
      <td>364327</td>
      <td>Male</td>
      <td>77</td>
      <td>184</td>
      <td>no</td>
      <td>no</td>
      <td>354.60</td>
      <td>13.8</td>
      <td>40.19</td>
      <td>94</td>
      <td>4</td>
      <td>4.0</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>280</th>
      <td>224478</td>
      <td>Male</td>
      <td>76</td>
      <td>153</td>
      <td>no</td>
      <td>no</td>
      <td>290.70</td>
      <td>8.5</td>
      <td>32.95</td>
      <td>108</td>
      <td>3</td>
      <td>3.0</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1334</th>
      <td>709479</td>
      <td>Male</td>
      <td>19</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>217.20</td>
      <td>13.8</td>
      <td>24.62</td>
      <td>79</td>
      <td>3</td>
      <td>4.0</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>277</th>
      <td>223234</td>
      <td>Female</td>
      <td>19</td>
      <td>121</td>
      <td>no</td>
      <td>yes</td>
      <td>297.60</td>
      <td>5.8</td>
      <td>33.73</td>
      <td>77</td>
      <td>3</td>
      <td>2.0</td>
      <td>3</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>921</th>
      <td>531048</td>
      <td>Male</td>
      <td>18</td>
      <td>120</td>
      <td>no</td>
      <td>no</td>
      <td>225.90</td>
      <td>6.4</td>
      <td>25.60</td>
      <td>123</td>
      <td>2</td>
      <td>2.0</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1253</th>
      <td>678217</td>
      <td>Female</td>
      <td>18</td>
      <td>64</td>
      <td>no</td>
      <td>no</td>
      <td>218.25</td>
      <td>8.9</td>
      <td>24.74</td>
      <td>91</td>
      <td>8</td>
      <td>3.0</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1509</th>
      <td>779479</td>
      <td>Male</td>
      <td>18</td>
      <td>149</td>
      <td>no</td>
      <td>no</td>
      <td>178.80</td>
      <td>12.2</td>
      <td>20.26</td>
      <td>119</td>
      <td>6</td>
      <td>4.0</td>
      <td>4</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>2000 rows × 14 columns</p>
</div>




```python
#Lets filter out how many are having subscription 
subsc= df[df['churn'] == 0]
subsc
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customer_id</th>
      <th>gender</th>
      <th>age</th>
      <th>no_of_days_subscribed</th>
      <th>multi_screen</th>
      <th>mail_subscribed</th>
      <th>weekly_mins_watched</th>
      <th>minimum_daily_mins</th>
      <th>maximum_daily_mins</th>
      <th>weekly_max_night_mins</th>
      <th>videos_watched</th>
      <th>maximum_days_inactive</th>
      <th>customer_support_calls</th>
      <th>churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100198</td>
      <td>Female</td>
      <td>36</td>
      <td>62</td>
      <td>no</td>
      <td>no</td>
      <td>148.35</td>
      <td>12.2</td>
      <td>16.81</td>
      <td>82</td>
      <td>1</td>
      <td>4.0</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100643</td>
      <td>Female</td>
      <td>39</td>
      <td>149</td>
      <td>no</td>
      <td>no</td>
      <td>294.45</td>
      <td>7.7</td>
      <td>33.37</td>
      <td>87</td>
      <td>3</td>
      <td>3.0</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>101595</td>
      <td>Female</td>
      <td>24</td>
      <td>131</td>
      <td>no</td>
      <td>yes</td>
      <td>321.30</td>
      <td>9.5</td>
      <td>36.41</td>
      <td>102</td>
      <td>4</td>
      <td>3.0</td>
      <td>3</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>101653</td>
      <td>Female</td>
      <td>40</td>
      <td>191</td>
      <td>no</td>
      <td>no</td>
      <td>243.00</td>
      <td>10.9</td>
      <td>27.54</td>
      <td>83</td>
      <td>7</td>
      <td>3.0</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>103051</td>
      <td>Female</td>
      <td>54</td>
      <td>59</td>
      <td>no</td>
      <td>no</td>
      <td>239.25</td>
      <td>10.2</td>
      <td>27.12</td>
      <td>106</td>
      <td>4</td>
      <td>3.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1994</th>
      <td>996524</td>
      <td>Female</td>
      <td>60</td>
      <td>141</td>
      <td>no</td>
      <td>yes</td>
      <td>310.35</td>
      <td>9.3</td>
      <td>35.17</td>
      <td>124</td>
      <td>11</td>
      <td>3.0</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>997132</td>
      <td>Female</td>
      <td>54</td>
      <td>75</td>
      <td>no</td>
      <td>yes</td>
      <td>182.25</td>
      <td>11.3</td>
      <td>20.66</td>
      <td>97</td>
      <td>5</td>
      <td>4.0</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>998086</td>
      <td>Male</td>
      <td>45</td>
      <td>127</td>
      <td>no</td>
      <td>no</td>
      <td>273.45</td>
      <td>9.3</td>
      <td>30.99</td>
      <td>116</td>
      <td>3</td>
      <td>3.0</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>998474</td>
      <td>Female</td>
      <td>53</td>
      <td>94</td>
      <td>no</td>
      <td>no</td>
      <td>128.85</td>
      <td>15.6</td>
      <td>14.60</td>
      <td>110</td>
      <td>16</td>
      <td>5.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>998934</td>
      <td>Male</td>
      <td>40</td>
      <td>94</td>
      <td>no</td>
      <td>no</td>
      <td>178.05</td>
      <td>10.4</td>
      <td>20.18</td>
      <td>100</td>
      <td>6</td>
      <td>3.0</td>
      <td>3</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>1738 rows × 14 columns</p>
</div>




```python
#Lets filter out how many are not having subscription expired 
unsubsc= df[df['churn'] == 1]
unsubsc
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customer_id</th>
      <th>gender</th>
      <th>age</th>
      <th>no_of_days_subscribed</th>
      <th>multi_screen</th>
      <th>mail_subscribed</th>
      <th>weekly_mins_watched</th>
      <th>minimum_daily_mins</th>
      <th>maximum_daily_mins</th>
      <th>weekly_max_night_mins</th>
      <th>videos_watched</th>
      <th>maximum_days_inactive</th>
      <th>customer_support_calls</th>
      <th>churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>100756</td>
      <td>Female</td>
      <td>65</td>
      <td>126</td>
      <td>no</td>
      <td>no</td>
      <td>87.30</td>
      <td>11.9</td>
      <td>9.89</td>
      <td>91</td>
      <td>1</td>
      <td>4.0</td>
      <td>5</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>101953</td>
      <td>Female</td>
      <td>31</td>
      <td>65</td>
      <td>no</td>
      <td>no</td>
      <td>193.65</td>
      <td>12.7</td>
      <td>21.95</td>
      <td>111</td>
      <td>6</td>
      <td>4.0</td>
      <td>4</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>103225</td>
      <td>Female</td>
      <td>40</td>
      <td>50</td>
      <td>no</td>
      <td>no</td>
      <td>196.65</td>
      <td>5.6</td>
      <td>22.29</td>
      <td>88</td>
      <td>9</td>
      <td>3.0</td>
      <td>5</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>107251</td>
      <td>Male</td>
      <td>39</td>
      <td>115</td>
      <td>no</td>
      <td>no</td>
      <td>367.50</td>
      <td>13.7</td>
      <td>41.65</td>
      <td>124</td>
      <td>8</td>
      <td>4.0</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>108519</td>
      <td>Female</td>
      <td>45</td>
      <td>76</td>
      <td>no</td>
      <td>no</td>
      <td>395.10</td>
      <td>11.4</td>
      <td>44.78</td>
      <td>101</td>
      <td>5</td>
      <td>4.0</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1926</th>
      <td>968500</td>
      <td>Male</td>
      <td>36</td>
      <td>101</td>
      <td>no</td>
      <td>no</td>
      <td>134.55</td>
      <td>13.5</td>
      <td>15.25</td>
      <td>93</td>
      <td>11</td>
      <td>4.0</td>
      <td>5</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1936</th>
      <td>971483</td>
      <td>Female</td>
      <td>37</td>
      <td>208</td>
      <td>no</td>
      <td>no</td>
      <td>489.75</td>
      <td>10.7</td>
      <td>55.51</td>
      <td>102</td>
      <td>6</td>
      <td>3.0</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1940</th>
      <td>971989</td>
      <td>Female</td>
      <td>33</td>
      <td>125</td>
      <td>yes</td>
      <td>no</td>
      <td>280.95</td>
      <td>9.6</td>
      <td>31.84</td>
      <td>112</td>
      <td>2</td>
      <td>3.0</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1959</th>
      <td>979909</td>
      <td>Male</td>
      <td>29</td>
      <td>144</td>
      <td>no</td>
      <td>no</td>
      <td>417.75</td>
      <td>11.6</td>
      <td>47.35</td>
      <td>90</td>
      <td>5</td>
      <td>4.0</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>999961</td>
      <td>Male</td>
      <td>37</td>
      <td>73</td>
      <td>no</td>
      <td>no</td>
      <td>326.70</td>
      <td>10.3</td>
      <td>37.03</td>
      <td>89</td>
      <td>6</td>
      <td>3.0</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>262 rows × 14 columns</p>
</div>




```python
#comparison of Subscriber and unsubsriber 
df['churn'].value_counts(dropna=False, normalize=True) * 100
```




    0.0    86.9
    1.0    13.1
    Name: churn, dtype: float64




```python
df['total_minutes_watched'] = df['weekly_mins_watched'] + df['weekly_max_night_mins']
print(df.head())
df
```

       customer_id  gender  age  no_of_days_subscribed multi_screen  \
    0       100198  Female   36                     62           no   
    1       100643  Female   39                    149           no   
    2       100756  Female   65                    126           no   
    3       101595  Female   24                    131           no   
    4       101653  Female   40                    191           no   
    
      mail_subscribed  weekly_mins_watched  minimum_daily_mins  \
    0              no               148.35                12.2   
    1              no               294.45                 7.7   
    2              no                87.30                11.9   
    3             yes               321.30                 9.5   
    4              no               243.00                10.9   
    
       maximum_daily_mins  weekly_max_night_mins  videos_watched  \
    0               16.81                     82               1   
    1               33.37                     87               3   
    2                9.89                     91               1   
    3               36.41                    102               4   
    4               27.54                     83               7   
    
       maximum_days_inactive  customer_support_calls  churn  total_minutes_watched  
    0                    4.0                       1    0.0                 230.35  
    1                    3.0                       2    0.0                 381.45  
    2                    4.0                       5    1.0                 178.30  
    3                    3.0                       3    0.0                 423.30  
    4                    3.0                       1    0.0                 326.00  





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customer_id</th>
      <th>gender</th>
      <th>age</th>
      <th>no_of_days_subscribed</th>
      <th>multi_screen</th>
      <th>mail_subscribed</th>
      <th>weekly_mins_watched</th>
      <th>minimum_daily_mins</th>
      <th>maximum_daily_mins</th>
      <th>weekly_max_night_mins</th>
      <th>videos_watched</th>
      <th>maximum_days_inactive</th>
      <th>customer_support_calls</th>
      <th>churn</th>
      <th>total_minutes_watched</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100198</td>
      <td>Female</td>
      <td>36</td>
      <td>62</td>
      <td>no</td>
      <td>no</td>
      <td>148.35</td>
      <td>12.2</td>
      <td>16.81</td>
      <td>82</td>
      <td>1</td>
      <td>4.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>230.35</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100643</td>
      <td>Female</td>
      <td>39</td>
      <td>149</td>
      <td>no</td>
      <td>no</td>
      <td>294.45</td>
      <td>7.7</td>
      <td>33.37</td>
      <td>87</td>
      <td>3</td>
      <td>3.0</td>
      <td>2</td>
      <td>0.0</td>
      <td>381.45</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100756</td>
      <td>Female</td>
      <td>65</td>
      <td>126</td>
      <td>no</td>
      <td>no</td>
      <td>87.30</td>
      <td>11.9</td>
      <td>9.89</td>
      <td>91</td>
      <td>1</td>
      <td>4.0</td>
      <td>5</td>
      <td>1.0</td>
      <td>178.30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>101595</td>
      <td>Female</td>
      <td>24</td>
      <td>131</td>
      <td>no</td>
      <td>yes</td>
      <td>321.30</td>
      <td>9.5</td>
      <td>36.41</td>
      <td>102</td>
      <td>4</td>
      <td>3.0</td>
      <td>3</td>
      <td>0.0</td>
      <td>423.30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>101653</td>
      <td>Female</td>
      <td>40</td>
      <td>191</td>
      <td>no</td>
      <td>no</td>
      <td>243.00</td>
      <td>10.9</td>
      <td>27.54</td>
      <td>83</td>
      <td>7</td>
      <td>3.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>326.00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>997132</td>
      <td>Female</td>
      <td>54</td>
      <td>75</td>
      <td>no</td>
      <td>yes</td>
      <td>182.25</td>
      <td>11.3</td>
      <td>20.66</td>
      <td>97</td>
      <td>5</td>
      <td>4.0</td>
      <td>2</td>
      <td>0.0</td>
      <td>279.25</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>998086</td>
      <td>Male</td>
      <td>45</td>
      <td>127</td>
      <td>no</td>
      <td>no</td>
      <td>273.45</td>
      <td>9.3</td>
      <td>30.99</td>
      <td>116</td>
      <td>3</td>
      <td>3.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>389.45</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>998474</td>
      <td>Female</td>
      <td>53</td>
      <td>94</td>
      <td>no</td>
      <td>no</td>
      <td>128.85</td>
      <td>15.6</td>
      <td>14.60</td>
      <td>110</td>
      <td>16</td>
      <td>5.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>238.85</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>998934</td>
      <td>Male</td>
      <td>40</td>
      <td>94</td>
      <td>no</td>
      <td>no</td>
      <td>178.05</td>
      <td>10.4</td>
      <td>20.18</td>
      <td>100</td>
      <td>6</td>
      <td>3.0</td>
      <td>3</td>
      <td>0.0</td>
      <td>278.05</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>999961</td>
      <td>Male</td>
      <td>37</td>
      <td>73</td>
      <td>no</td>
      <td>no</td>
      <td>326.70</td>
      <td>10.3</td>
      <td>37.03</td>
      <td>89</td>
      <td>6</td>
      <td>3.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>415.70</td>
    </tr>
  </tbody>
</table>
<p>2000 rows × 15 columns</p>
</div>




```python
# Average age of churned vs. non-churned customers
print(df.groupby('churn')['age'].mean())
```

    churn
    0.0    38.643843
    1.0    39.000000
    Name: age, dtype: float64



```python
# Average number of days subscribed for churned vs. non-churned customers
print(df.groupby('churn')['no_of_days_subscribed'].mean())

```

    churn
    0.0     99.711162
    1.0    100.007634
    Name: no_of_days_subscribed, dtype: float64



```python
# Average weekly minutes watched for churned vs. non-churned customers
print(df.groupby('churn')['weekly_mins_watched'].mean())
```

    churn
    0.0    265.085731
    1.0    303.961260
    Name: weekly_mins_watched, dtype: float64



```python
df.fillna({'gender': df['gender'].mode().iloc[0], 
           'maximum_days_inactive': df['maximum_days_inactive'].median(),
          'churn': df['churn'].value_counts().idxmin()})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customer_id</th>
      <th>gender</th>
      <th>age</th>
      <th>no_of_days_subscribed</th>
      <th>multi_screen</th>
      <th>mail_subscribed</th>
      <th>weekly_mins_watched</th>
      <th>minimum_daily_mins</th>
      <th>maximum_daily_mins</th>
      <th>weekly_max_night_mins</th>
      <th>videos_watched</th>
      <th>maximum_days_inactive</th>
      <th>customer_support_calls</th>
      <th>churn</th>
      <th>total_minutes_watched</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100198</td>
      <td>Female</td>
      <td>36</td>
      <td>62</td>
      <td>no</td>
      <td>no</td>
      <td>148.35</td>
      <td>12.2</td>
      <td>16.81</td>
      <td>82</td>
      <td>1</td>
      <td>4.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>230.35</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100643</td>
      <td>Female</td>
      <td>39</td>
      <td>149</td>
      <td>no</td>
      <td>no</td>
      <td>294.45</td>
      <td>7.7</td>
      <td>33.37</td>
      <td>87</td>
      <td>3</td>
      <td>3.0</td>
      <td>2</td>
      <td>0.0</td>
      <td>381.45</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100756</td>
      <td>Female</td>
      <td>65</td>
      <td>126</td>
      <td>no</td>
      <td>no</td>
      <td>87.30</td>
      <td>11.9</td>
      <td>9.89</td>
      <td>91</td>
      <td>1</td>
      <td>4.0</td>
      <td>5</td>
      <td>1.0</td>
      <td>178.30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>101595</td>
      <td>Female</td>
      <td>24</td>
      <td>131</td>
      <td>no</td>
      <td>yes</td>
      <td>321.30</td>
      <td>9.5</td>
      <td>36.41</td>
      <td>102</td>
      <td>4</td>
      <td>3.0</td>
      <td>3</td>
      <td>0.0</td>
      <td>423.30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>101653</td>
      <td>Female</td>
      <td>40</td>
      <td>191</td>
      <td>no</td>
      <td>no</td>
      <td>243.00</td>
      <td>10.9</td>
      <td>27.54</td>
      <td>83</td>
      <td>7</td>
      <td>3.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>326.00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>997132</td>
      <td>Female</td>
      <td>54</td>
      <td>75</td>
      <td>no</td>
      <td>yes</td>
      <td>182.25</td>
      <td>11.3</td>
      <td>20.66</td>
      <td>97</td>
      <td>5</td>
      <td>4.0</td>
      <td>2</td>
      <td>0.0</td>
      <td>279.25</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>998086</td>
      <td>Male</td>
      <td>45</td>
      <td>127</td>
      <td>no</td>
      <td>no</td>
      <td>273.45</td>
      <td>9.3</td>
      <td>30.99</td>
      <td>116</td>
      <td>3</td>
      <td>3.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>389.45</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>998474</td>
      <td>Female</td>
      <td>53</td>
      <td>94</td>
      <td>no</td>
      <td>no</td>
      <td>128.85</td>
      <td>15.6</td>
      <td>14.60</td>
      <td>110</td>
      <td>16</td>
      <td>5.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>238.85</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>998934</td>
      <td>Male</td>
      <td>40</td>
      <td>94</td>
      <td>no</td>
      <td>no</td>
      <td>178.05</td>
      <td>10.4</td>
      <td>20.18</td>
      <td>100</td>
      <td>6</td>
      <td>3.0</td>
      <td>3</td>
      <td>0.0</td>
      <td>278.05</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>999961</td>
      <td>Male</td>
      <td>37</td>
      <td>73</td>
      <td>no</td>
      <td>no</td>
      <td>326.70</td>
      <td>10.3</td>
      <td>37.03</td>
      <td>89</td>
      <td>6</td>
      <td>3.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>415.70</td>
    </tr>
  </tbody>
</table>
<p>2000 rows × 15 columns</p>
</div>




```python

```


```python
# Lets see the average weekly min watched by gender:

pivot_table = df.pivot_table(index='gender', columns='churn', values='weekly_mins_watched', aggfunc='mean')
print(pivot_table)

```

    churn          0.0         1.0
    gender                        
    Female  264.395788  303.112500
    Male    265.700598  304.772015


#### Vizuallization:


```python
sns.countplot(x='gender', data=df)
plt.title('Gender Distribution')
```




    Text(0.5, 1.0, 'Gender Distribution')




    
![png](output_61_1.png)
    



```python
plt.pie(df['gender'].value_counts(),
       labels=df['gender'].value_counts(dropna=False).index,
       autopct="%.2f%%")
plt.title('Gender Distribution in %')
plt.show()
```


    
![png](output_62_0.png)
    



```python
df_gender=df['gender'].value_counts(dropna=False,normalize=True)*100
df_gender= df_gender.apply(lambda x: f"{x:.2f}%")
df_gender
```




    Male      52.65%
    Female    47.35%
    Name: gender, dtype: object




```python
#Gender and Churn
#Exploring the churn rate by gender

plt.figure(figsize=(10, 6))
sns.countplot(x='gender', hue='churn', data=df, palette='Set2')
plt.title('Churn Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

```


    
![png](output_64_0.png)
    



```python
# Distribution of age

plt.figure(figsize=(10, 6))
sns.histplot(df['age'], kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
```


    
![png](output_65_0.png)
    



```python
# Churn rate by gender
plt.figure(figsize=(10, 6))
sns.barplot(x='gender', y='churn', data=df)
plt.title('Churn Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Churn Rate')
plt.show()
```


    
![png](output_66_0.png)
    



```python
# Churn rate by multi_screen
plt.figure(figsize=(10, 6))
sns.barplot(x='multi_screen', y='churn', data=df)
plt.title('Churn Rate by Multi Screen Subscription')
plt.xlabel('Multi Screen Subscription')
plt.ylabel('Churn Rate')
plt.show()
```


    
![png](output_67_0.png)
    



```python
# Correlation heatmap
plt.figure(figsize=(12, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()
```

    /var/folders/1f/0g92ck1d38116469ltht60r80000gn/T/ipykernel_58205/2866692320.py:3: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
      corr = df.corr()



    
![png](output_68_1.png)
    



```python
#Churn by Weekly Minutes Watched
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='weekly_mins_watched', hue='churn', kde=True)
plt.title('Weekly Minutes Watched by Churn')
plt.xlabel('Weekly Minutes Watched')
plt.ylabel('Frequency')
plt.show()
```


    
![png](output_69_0.png)
    



```python
# Churn By Cust. supports call
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='customer_support_calls', hue='churn')
plt.title('Churn by Customer Support Calls')
plt.xlabel('Customer Support Calls')
plt.ylabel('Count')
plt.show()

```


    
![png](output_70_0.png)
    



```python
#Pair Plot for Selected Features
selected_features = ['age', 'no_of_days_subscribed', 'weekly_mins_watched', 'churn']
sns.pairplot(df[selected_features], hue='churn')
plt.show()

```


    
![png](output_71_0.png)
    



```python
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

```


    
![png](output_72_0.png)
    



```python
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


```


    
![png](output_73_0.png)
    



```python
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

```


    
![png](output_74_0.png)
    



```python
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

```


    
![png](output_75_0.png)
    



```python
#Using pair plots to visualize pairwise relationships between features for churned and non-churned customers.

selected_features = ['age', 'no_of_days_subscribed', 'weekly_mins_watched', 'minimum_daily_mins', 'maximum_daily_mins', 'churn']
sns.pairplot(df[selected_features], hue='churn', palette='Set1', diag_kind='kde')
plt.show()
```


    
![png](output_76_0.png)
    



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

    churn                     1.000000
    customer_support_calls    0.204774
    weekly_mins_watched       0.162876
    maximum_daily_mins        0.162874
    total_minutes_watched     0.158556
    minimum_daily_mins        0.066646
    maximum_days_inactive     0.044778
    age                       0.011777
    weekly_max_night_mins     0.006917
    no_of_days_subscribed     0.002517
    videos_watched           -0.019314
    customer_id              -0.051440
    Name: churn, dtype: float64


    /var/folders/1f/0g92ck1d38116469ltht60r80000gn/T/ipykernel_58205/1506240649.py:2: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
      correlation_with_churn = df.corr()['churn'].sort_values(ascending=False)



    
![png](output_77_2.png)
    



```python
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

```


    
![png](output_78_0.png)
    



```python
#Multi-Screen Subscription Analysis
#Checking the proportion of customers who have multi-screen subscriptions.
plt.figure(figsize=(6, 6))
plt.pie(df['multi_screen'].value_counts(), labels=['No', 'Yes'], autopct='%1.1f%%')
plt.title('Multi-Screen Subscription Distribution')
plt.show()

```


    
![png](output_79_0.png)
    



```python
#Mail Subscription Analysis
#Analyzing the proportion of customers who have subscribed to email notifications.


plt.figure(figsize=(6, 6))
plt.pie(df['mail_subscribed'].value_counts(), labels=['No', 'Yes'], autopct='%1.1f%%')
plt.title('Mail Subscription Distribution')
plt.show()

```


    
![png](output_80_0.png)
    



```python
#Churn Rate Analysis
#Understanding the churn rate in your customer base.

plt.figure(figsize=(6, 6))
plt.pie(df['churn'].value_counts(), labels=['Not Churned', 'Churned'], autopct='%1.1f%%')
plt.title('Churn Rate')
plt.show()

```


    
![png](output_81_0.png)
    



```python
#Relationship Between Maximum Days Inactive and Churn
#Exploring the relationship between the number of maximum days a customer was inactive and whether they churned.

plt.figure(figsize=(10, 6))
sns.boxplot(x='churn', y='maximum_days_inactive', data=df)
plt.title('Maximum Days Inactive vs Churn')
plt.xlabel('Churn')
plt.ylabel('Maximum Days Inactive')
plt.show()

```


    
![png](output_82_0.png)
    



```python
#Customer Support Calls and Churn
#Analyzing the relationship between the number of customer support calls and churn.

plt.figure(figsize=(10, 6))
sns.boxplot(x='churn', y='customer_support_calls', data=df)
plt.title('Customer Support Calls vs Churn')
plt.xlabel('Churn')
plt.ylabel('Customer Support Calls')
plt.show()

```


    
![png](output_83_0.png)
    



```python
#Videos Watched and Churn
#Examining the relationship between the number of videos watched and churn.

plt.figure(figsize=(10, 6))
sns.boxplot(x='churn', y='videos_watched', data=df)
plt.title('Videos Watched vs Churn')
plt.xlabel('Churn')
plt.ylabel('Videos Watched')
plt.show()

```


    
![png](output_84_0.png)
    



```python
#Monthly Engagement Analysis
#Analyzing the monthly engagement of users by looking at the number of videos watched.

plt.figure(figsize=(10, 6))
sns.lineplot(x='no_of_days_subscribed', y='videos_watched', hue='churn', data=df, palette='coolwarm')
plt.title('Monthly Engagement: Videos Watched vs Subscription Duration')
plt.xlabel('Number of Days Subscribed')
plt.ylabel('Videos Watched')
plt.legend(title='Churn')
plt.show()

```


    
![png](output_85_0.png)
    



```python
#Weekly Maximum Night Minutes and Churn
#Examining the relationship between the maximum night minutes watched in a week and churn.

plt.figure(figsize=(10, 6))
sns.boxplot(x='churn', y='weekly_max_night_mins', data=df)
plt.title('Weekly Maximum Night Minutes vs Churn')
plt.xlabel('Churn')
plt.ylabel('Weekly Maximum Night Minutes')
plt.show()

```


    
![png](output_86_0.png)
    



```python

```
