# Time Series Data Preparation and Processing

![Milestone 1](https://s3.ap-southeast-1.amazonaws.com/www.jiahao.io/manning/project1_milestone_1.png)

## Importing Necessary Libraries and Functions

The data (data/sales.csv) that we are using is a daily retail sales dataset modified from the [M5 competition data](https://www.kaggle.com/c/m5-forecasting-accuracy/data). The challenge is to forecast the sales of the respective stores by each category for the next 28 days.

Instruction:<br>
We have written the code to import the libraries so you can just run it. If you need other libraries while working on this notebook, please feel free to add the library to this cell below.


```python
RunningInCOLAB = 'google.colab' in str(get_ipython())

# import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

%matplotlib inline

# suppress pandas SettingWithCopyWarning 
pd.options.mode.chained_assignment = None
```

## Previewing the Sales Data

Let us first have a preview of the data to understand what we will be working with.

Instruction:<br>
Read in the data *sales.csv* from the data folder into a pandas dataframe and preview the first 5 rows.

Note:<br>
If you are running this notebook in Colab, please upload *sales.csv* from the data folder.


```python
if RunningInCOLAB:
  from google.colab import files

  uploaded = files.upload()

  for fn in uploaded.keys():
    print('User uploaded file "{name}" with length {length} bytes'.format(
        name=fn, length=len(uploaded[fn])))
```



<input type="file" id="files-6e69800b-3319-4b46-b730-e352058e6e99" name="files[]" multiple disabled
   style="border:none" />
<output id="result-6e69800b-3319-4b46-b730-e352058e6e99">
 Upload widget is only available when the cell has been executed in the
 current browser session. Please rerun this cell to enable.
 </output>
 <script src="/nbextensions/google.colab/files.js"></script> 


    Saving sales.csv to sales.csv
    User uploaded file "sales.csv" with length 1922818 bytes



```python
# load data and view first 5 rows
if RunningInCOLAB:
  sales_df = pd.read_csv('./sales.csv')
else:
  sales_df = pd.read_csv('../data/sales.csv')  # use this if running notebook in local 
sales_df.info()
sales_df.head()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 58230 entries, 0 to 58229
    Data columns (total 4 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   date      58230 non-null  object 
     1   store_id  58230 non-null  object 
     2   cat_id    58230 non-null  object 
     3   sales     58230 non-null  float64
    dtypes: float64(1), object(3)
    memory usage: 1.8+ MB





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
      <th>date</th>
      <th>store_id</th>
      <th>cat_id</th>
      <th>sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-29</td>
      <td>TX_1</td>
      <td>FOODS</td>
      <td>3950.35</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-30</td>
      <td>TX_1</td>
      <td>FOODS</td>
      <td>3844.97</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-31</td>
      <td>TX_1</td>
      <td>FOODS</td>
      <td>2888.03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-02-01</td>
      <td>TX_1</td>
      <td>FOODS</td>
      <td>3631.28</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-02-02</td>
      <td>TX_1</td>
      <td>FOODS</td>
      <td>3072.18</td>
    </tr>
  </tbody>
</table>
</div>



As seen from the data preview above, there are 4 columns with ~58k rows. The daily sales (column: `sales`) for each store (column: `store_id`) and category (column: `cat_id`) are listed in each row.

If we generate the unique value count of the `store_id` and `cat_id`, we will see that there are 10 stores and 3 categories.

Instruction:<br>
Verify the number of unique values of `store_id` and `cat_id` each.


```python
# unique count of store_id and cat_id
print(f"There are {sales_df.store_id.nunique()} stores and {sales_df.cat_id.nunique()} categories.")
```

    There are 10 stores and 3 categories.


And in terms of the time period, our sales data start from 2011 to 2016. We also set the `date` column to datetime format.

Instructions:<br>
Verify the date range of the `sales` data


```python
# max and min date
sales_df['date'] = pd.to_datetime(sales_df.date)
print(f"Sales start from {sales_df.date.min()} to {sales_df.date.max()}")
```

    Sales start from 2011-01-29 00:00:00 to 2016-05-22 00:00:00


## Processing of Data

There are 3 common data quality issues to check for time series data:
1. Irregular time series
2. Outliers
3. Missing data

### Data quality issue - Irregular time series

In our case, our sales data is of daily frequency. However, the data may possibly miss out some dates (see point 1 in image below) or added more rows for the same date (see point 2 in image below). Hence, we need to ensure that every date is accounted by exactly one row. 

For the scenario of point 2 in image below, we will also need to check with the data owner on the proper treatment, i.e., do we add up the sales of both rows to get sales for 2011-02-19 or do we just keep the row with time 00:00:00 because the data is a snapshot of sales? 



![Irregular Time Series](https://s3.ap-southeast-1.amazonaws.com/www.jiahao.io/manning/irregular_time_period.PNG)

Instruction:<br>
Check for duplicated dates within a store and category group


```python
# if our date column is datetime, then we need to extract the date component and 
# check for duplicate dates within a store and category group
sales_df['date_part'] = sales_df['date'].dt.date
sales_df[sales_df.duplicated(['date_part', 'store_id', 'cat_id'])]
sales_df.drop(columns='date_part', inplace=True)
```

Good, there is no duplicated date in a store and category group.

Next let's try to see if there is missing date in between the start and end date of sales.

Instructions:<br>
Generate dataframe with the daily dates in between the start and end dates of sales. Then check if there is any missing date(s) in between.


```python
# generate daily dates between the start and end of sales
total_days = (sales_df.date.max() - sales_df.date.min()).days + 1
dates_df = pd.DataFrame(pd.date_range(sales_df.date.min(), periods=total_days, freq="D"), columns=['date'])
display(dates_df.head(3))
display(dates_df.tail(3))
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
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-29</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-31</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1938</th>
      <td>2016-05-20</td>
    </tr>
    <tr>
      <th>1939</th>
      <td>2016-05-21</td>
    </tr>
    <tr>
      <th>1940</th>
      <td>2016-05-22</td>
    </tr>
  </tbody>
</table>
</div>



```python
# merge the dates dataframe with sales dataframe to check for missing dates
# in our case, lucky for us that there is no irregular time series data
sales_daily_df = dates_df.merge(sales_df, on='date', how='left')
sales_daily_df.info()
sales_daily_df.head()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 58230 entries, 0 to 58229
    Data columns (total 4 columns):
     #   Column    Non-Null Count  Dtype         
    ---  ------    --------------  -----         
     0   date      58230 non-null  datetime64[ns]
     1   store_id  58230 non-null  object        
     2   cat_id    58230 non-null  object        
     3   sales     58230 non-null  float64       
    dtypes: datetime64[ns](1), float64(1), object(2)
    memory usage: 2.2+ MB





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
      <th>date</th>
      <th>store_id</th>
      <th>cat_id</th>
      <th>sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-29</td>
      <td>TX_1</td>
      <td>FOODS</td>
      <td>3950.35</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-29</td>
      <td>TX_1</td>
      <td>HOBBIES</td>
      <td>759.99</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-29</td>
      <td>TX_1</td>
      <td>HOUSEHOLD</td>
      <td>1876.34</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-29</td>
      <td>TX_2</td>
      <td>FOODS</td>
      <td>5877.90</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-29</td>
      <td>TX_2</td>
      <td>HOBBIES</td>
      <td>1294.04</td>
    </tr>
  </tbody>
</table>
</div>



Luckily for us, there is no missing date in between. If there is any, the `sales` value will be null and we can impute them later.

### Data quality issue - Outliers

There are various ways to check for outliers in time series. Here, we use the Interquartile Range (IQR) method mentioned in the book [Forecasting: Principles and Practice](https://otexts.com/fpp3/missing-outliers.html) by Rob J Hyndman and George Athanasopoulos. However, we modified the method a little by detrending the time series first before applying the IQR method.

Instructions:<br>
- Code the IQR method and implement it on the sales data. Take care to check for outliers by each store and category group. The sales dataframe should have a column named `anomaly` that will indicate each outlier as True and non-outliers as False.
- Detrend the sales by each time series before applying the IQR method.


```python
def outlier_detection(sales_series_df):
    """
    Add column 'anomaly' to dataframe to mark outliers as True, non-outliers as False. 
    """
    # calculate interquartile range
    Q1 = sales_series_df['sales_detrend'].quantile(0.25)
    Q3 = sales_series_df['sales_detrend'].quantile(0.75)
    IQR = Q3 - Q1

    # identify outliers
    # Filtering Values between Q1-3IQR and Q3+3IQR
    sales_series_df['anomaly'] = (sales_series_df.sales_detrend >= Q3 + 3*IQR) | (sales_series_df.sales_detrend <= Q1 - 3*IQR)
    
    return sales_series_df

outlier_marked_list = []

for store in sales_daily_df.store_id.unique():
    for category in sales_daily_df.cat_id.unique():
        sales_series_df = sales_daily_df.loc[(sales_daily_df.store_id==store) & (sales_daily_df.cat_id==category), :]
        # detrend sales for better detection of outliers
        sales_series_df['sales_detrend'] = signal.detrend(sales_series_df['sales'].values)
        sales_outlier_marked_df = outlier_detection(sales_series_df)
        outlier_marked_list.append(sales_outlier_marked_df)

outlier_marked_df = pd.concat(outlier_marked_list)
```

Let's take a look at how our outlier detection method performs.

In general, the method is able to capture most of the outliers (as seen from the diagrams below) that our human judgement would also determine to be outlying points. There are some outlying points that were missed out but we shall see in a while what these points may be and we can manually mark them as outlier if necessary.

We also noticed that some of the points towards the tail end of the time series for the store *CA_2* and category *FOODS* seem to be wrongly marked as outliers. We can unmark these points as well.

Instructions:<br>
Visualize the outliers identified by overlaying them on a sales line plot.


```python
def visualize_outliers(store, category):
    outlier_series_df = outlier_marked_df.loc[(outlier_marked_df.store_id==store) & (outlier_marked_df.cat_id==category), :]
    # visualization of outliers detected
    fig, ax = plt.subplots(figsize=(15,5))
    a = outlier_series_df.loc[outlier_series_df['anomaly'] == 1, ['date', 'sales']] #anomaly
    ax.plot(outlier_series_df.date, outlier_series_df['sales'], color='black', label = 'Normal')
    ax.scatter(a.date, a['sales'], color='red', label = 'Anomaly')
    plt.title(f'Store: {outlier_series_df.store_id.unique()} Category: {outlier_series_df.cat_id.unique()}')
    plt.legend()
    plt.show()

# sample one of the store category to see how our outlier detection performs
visualize_outliers('TX_1', 'FOODS')
```


    
![png](Part-1---Data-Preparation-and-Processing_files/Part-1---Data-Preparation-and-Processing_35_0.png)
    



```python
# sample one of the store category to see how our outlier detection performs
visualize_outliers('CA_2', 'FOODS')
```


    
![png](Part-1---Data-Preparation-and-Processing_files/Part-1---Data-Preparation-and-Processing_36_0.png)
    


We can unmark some of the points as outliers.

Instructions:<br>
If any of the "outliers" identified are incorrect, unmark them in your `anomaly` column.


```python
# unmark points as outliers
outlier_marked_df['anomaly'] = outlier_marked_df.apply(lambda x: False if x.store_id=='CA_2' and x.cat_id=='FOODS' and x.sales>10000 else x.anomaly, 1)

visualize_outliers('CA_2', 'FOODS')
```


    
![png](Part-1---Data-Preparation-and-Processing_files/Part-1---Data-Preparation-and-Processing_39_0.png)
    


It turns out that when we filter and zoom into those outlying points with sales very close to 0, they seem to generally occur on 25th Dec. Seems likely due to Christmas... Turns out when you spoke to the data owner, it was revealed that the stores close on Christmas hence the close to zero sales.

Let's mark these as outliers for the time being although we will need to take care to set predicted sales on Christmas to zero later.

Instructions:<br>
Mark any additional points as outliers if necessary in your `anomaly` column


```python
outlier_marked_df.loc[outlier_marked_df.sales < 100]
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
      <th>date</th>
      <th>store_id</th>
      <th>cat_id</th>
      <th>sales</th>
      <th>sales_detrend</th>
      <th>anomaly</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9900</th>
      <td>2011-12-25</td>
      <td>TX_1</td>
      <td>FOODS</td>
      <td>0.00</td>
      <td>-4325.732392</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20880</th>
      <td>2012-12-25</td>
      <td>TX_1</td>
      <td>FOODS</td>
      <td>0.00</td>
      <td>-4564.055431</td>
      <td>False</td>
    </tr>
    <tr>
      <th>31830</th>
      <td>2013-12-25</td>
      <td>TX_1</td>
      <td>FOODS</td>
      <td>1.58</td>
      <td>-4800.147315</td>
      <td>True</td>
    </tr>
    <tr>
      <th>42780</th>
      <td>2014-12-25</td>
      <td>TX_1</td>
      <td>FOODS</td>
      <td>0.00</td>
      <td>-5039.399199</td>
      <td>True</td>
    </tr>
    <tr>
      <th>53730</th>
      <td>2015-12-25</td>
      <td>TX_1</td>
      <td>FOODS</td>
      <td>1.58</td>
      <td>-5275.491082</td>
      <td>True</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9929</th>
      <td>2011-12-25</td>
      <td>WI_3</td>
      <td>HOUSEHOLD</td>
      <td>0.00</td>
      <td>-2244.693915</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20909</th>
      <td>2012-12-25</td>
      <td>WI_3</td>
      <td>HOUSEHOLD</td>
      <td>0.00</td>
      <td>-2328.663615</td>
      <td>False</td>
    </tr>
    <tr>
      <th>31859</th>
      <td>2013-12-25</td>
      <td>WI_3</td>
      <td>HOUSEHOLD</td>
      <td>0.00</td>
      <td>-2412.403890</td>
      <td>False</td>
    </tr>
    <tr>
      <th>42809</th>
      <td>2014-12-25</td>
      <td>WI_3</td>
      <td>HOUSEHOLD</td>
      <td>0.00</td>
      <td>-2496.144164</td>
      <td>False</td>
    </tr>
    <tr>
      <th>53759</th>
      <td>2015-12-25</td>
      <td>WI_3</td>
      <td>HOUSEHOLD</td>
      <td>0.00</td>
      <td>-2579.884439</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>155 rows × 6 columns</p>
</div>




```python
# mark points as outliers
outlier_marked_df['anomaly'] = outlier_marked_df.apply(lambda x: True if x.date.month==12 and x.date.day==25 else x.anomaly, 1)

visualize_outliers('CA_2', 'FOODS')
```


    
![png](Part-1---Data-Preparation-and-Processing_files/Part-1---Data-Preparation-and-Processing_43_0.png)
    


Finally, we will need to set our outliers to null sales value. In reality, we should talk to the data owner to find out the reasons for these outliers and decide if they are legitimate data points (e.g. the Christmas points) or erroneous points.

For the Christmas data points, they reflect the calendar effect and in some models, we can leave these data points as they are and add in covariates indicating these calendar effects into the model. However for us, we will be using univariate model and it would better for us to remove these calendar effects.

Before we set the outliers to null sales value, we can check for other null sales values if any and perform imputation in our next data quality check.


```python
print(f"In total, we detected {outlier_marked_df.anomaly.sum()} outliers.")
```

    In total, we detected 175 outliers.


### Data quality issue - Missing data

First let's check for missing sales value. Then we proceed to set outlier to null sales value as mentioned in previous data quality check. And finally perform imputation of sales.

Instructions:<br>
Check how many rows have missing sales value


```python
# check number of rows with missing sales
sum(outlier_marked_df.sales.isna())
```




    0



Good, seems like no missing sales. Well in fact, we knew there's no missing sales when we were previewing the data. But this is just for procedure.

Next we can set our outliers to null sales value.

Instructions:<br>
Set outliers to null sales value


```python
outlier_marked_df.loc[outlier_marked_df['anomaly']==1, 'sales'] = np.nan
```


```python
# check our missing sales data is equal to the number of outliers
sum(outlier_marked_df.sales.isna())
```




    175



We can now proceed to impute the missing sales. In our case, we have no extended missing sales period and the number of data points with missing sales is little compared to the total number of data points. Hence, we shall adopt linear interpolation method.

First, we need to set the `date`, `store_id` and `cat_id` columns as index.

Instructions:<br>
Impute the missing sales using linear interpolation method.

Hint: [https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html)


```python
sales_imputed_df = outlier_marked_df.set_index(['store_id', 'cat_id', 'date'])
sales_imputed_df.interpolate(inplace=True)
sales_imputed_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    MultiIndex: 58230 entries, ('TX_1', 'FOODS', Timestamp('2011-01-29 00:00:00')) to ('WI_3', 'HOUSEHOLD', Timestamp('2016-05-22 00:00:00'))
    Data columns (total 3 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   sales          58230 non-null  float64
     1   sales_detrend  58230 non-null  float64
     2   anomaly        58230 non-null  bool   
    dtypes: bool(1), float64(2)
    memory usage: 1.3+ MB



```python
def visualize_sales(store, category):
    sales_series_df = sales_imputed_df.loc[(sales_imputed_df.index.get_level_values('store_id')==store) & (sales_imputed_df.index.get_level_values('cat_id')==category), :]
    sales_series_df.reset_index(inplace=True)
    # visualization of sales
    fig, ax = plt.subplots(figsize=(15,5))
    a = sales_series_df.loc[sales_series_df['anomaly'] == 1, ['date', 'sales']] #anomaly
    ax.plot(sales_series_df.date, sales_series_df['sales'], color='black', label = 'Normal', alpha=0.5)
    ax.scatter(a.date, a['sales'], color='red', label = 'Outlier Imputed')
    plt.title(f'Store: {sales_series_df.store_id.unique()} Category: {sales_series_df.cat_id.unique()}')
    plt.legend()
    plt.show()

# sample one of the store category to see how our outlier detection performs
visualize_sales('CA_2', 'FOODS')
```


    
![png](Part-1---Data-Preparation-and-Processing_files/Part-1---Data-Preparation-and-Processing_57_0.png)
    


With these data quality checks done, save the preprocessed data as `sales_processed.csv` in the data folder.

Instructions:<br>
Save the preprocessed data as `sales_processed.csv` in the data folder


```python
if RunningInCOLAB:
  sales_imputed_df.drop(columns=['sales_detrend', 'anomaly']).to_csv('./sales_processed.csv')
else:
  # run this if executing this notebook in local
  sales_imputed_df.drop(columns=['sales_detrend', 'anomaly']).to_csv('../data/sales_processed.csv')
```
