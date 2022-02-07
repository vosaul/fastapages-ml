## Importing Necessary Libraries and Functions

The first thing we need to do is import the necessary functions and libraries that we will be working with throughout the topic. We should also go ahead and upload all the of the necessary data sets here instead of loading them as we go. We will be using energy production data from PJM Interconnection. They are a regional transmission organization that coordinates the movement of wholesale electricity in parts of the United States. Specifically, we will be focused on a region of Pennsylvania. We will also be using temperature data collected from the National Oceanic and Atmospheric Assocation (NOAA).


```python
!conda update -n base -c defaults conda

!conda install pandas -y
!conda install numpy -y
!conda install matplotlib -y
!conda install statsmodels -y
!pip install scipy 
```


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics import tsaplots
from statsmodels.graphics import tsaplots
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
```

Notice how we added an additional pieces above from the ```statsmodels``` module. We need to build time series models in this milestone and so we will need the above pieces to do so. We will be building exponential smoothing models as well as ARIMA models.

This milestone builds off the previous ones so we should complete the following steps to the first milestone again to have our data prepped and ready to go. We should also rebuild our last model from milestone 3 since that is our foundational model!

## Preparing the Energy and Temperature Data##

First we need to load our weather and energy data sets for cleaning. Let's use the pandas library and the ```read.csv``` function to do this.


```python
# Loading the Needed Data Sets 
weather = pd.read_csv('.../hr_temp_20170201-20200131_subset.csv')
energy = pd.read_csv('.../hrl_load_metered - 20170201-20200131.csv')

```

It is always good practice to take a look at the first few observations of the data set to make sure that everything looks like how we expected it to when we read in our CSV file. Let's use the ```head``` function for this.


```python
weather.head()
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
      <th>STATION</th>
      <th>DATE</th>
      <th>REPORT_TYPE</th>
      <th>SOURCE</th>
      <th>HourlyDryBulbTemperature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>72520514762</td>
      <td>2017-02-01T00:53:00</td>
      <td>FM-15</td>
      <td>7</td>
      <td>37.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>72520514762</td>
      <td>2017-02-01T01:53:00</td>
      <td>FM-15</td>
      <td>7</td>
      <td>37.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>72520514762</td>
      <td>2017-02-01T02:53:00</td>
      <td>FM-15</td>
      <td>7</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>72520514762</td>
      <td>2017-02-01T03:53:00</td>
      <td>FM-15</td>
      <td>7</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>72520514762</td>
      <td>2017-02-01T04:53:00</td>
      <td>FM-15</td>
      <td>7</td>
      <td>36.0</td>
    </tr>
  </tbody>
</table>
</div>



Perfect! We have temperature as well as time. There are some other pieces of information like the station number, source of the reading and reading type, but we don't need those.

Let's take a look at the first few observations of the energy data as well!


```python
energy.head()
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
      <th>datetime_beginning_utc</th>
      <th>datetime_beginning_ept</th>
      <th>nerc_region</th>
      <th>mkt_region</th>
      <th>zone</th>
      <th>load_area</th>
      <th>mw</th>
      <th>is_verified</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2/1/2017 5:00</td>
      <td>2/1/2017 0:00</td>
      <td>RFC</td>
      <td>WEST</td>
      <td>DUQ</td>
      <td>DUQ</td>
      <td>1419.881</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2/1/2017 6:00</td>
      <td>2/1/2017 1:00</td>
      <td>RFC</td>
      <td>WEST</td>
      <td>DUQ</td>
      <td>DUQ</td>
      <td>1379.505</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2/1/2017 7:00</td>
      <td>2/1/2017 2:00</td>
      <td>RFC</td>
      <td>WEST</td>
      <td>DUQ</td>
      <td>DUQ</td>
      <td>1366.106</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2/1/2017 8:00</td>
      <td>2/1/2017 3:00</td>
      <td>RFC</td>
      <td>WEST</td>
      <td>DUQ</td>
      <td>DUQ</td>
      <td>1364.453</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2/1/2017 9:00</td>
      <td>2/1/2017 4:00</td>
      <td>RFC</td>
      <td>WEST</td>
      <td>DUQ</td>
      <td>DUQ</td>
      <td>1391.265</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



Great! Again, we have the important information of time as well as megawatt (MW) readings per hour. Again, there are some other varibales that we won't end up using in this data set as well.

Let's get rid of the variables we don't need and combine the variables that we do need into one pandas data frame. Dictionaries are an easy way of doing this. Here, we are pulling the MW column from the energy data set as well as the temperature and date columns from the weather data set. These data sets already line up on time which makes this much easier.


```python
d = {'MW': energy['mw'], 'Temp': weather['HourlyDryBulbTemperature'], 'Date': weather['DATE']}
```

Now let's create our pandas data frame.


```python
df = pd.DataFrame(d)
```

One of the problems when loading a data set you want to run time series analysis on is the type of object Python sees for the "date" variable. Let's look at the pandas data frame data types for each of our variables.


```python
print(df.dtypes)
```

    MW      float64
    Temp    float64
    Date     object
    dtype: object


Here we can see that the Date variable is a general object and not a "date" according to Python. We can change that with the pandas function ```to_datetime``` as we have below.


```python
df['Date'] = pd.to_datetime(df['Date'])
print(df.dtypes)
```

    MW             float64
    Temp           float64
    Date    datetime64[ns]
    dtype: object


Good! Now that we have a ```datetime64``` object in our data set we can easily create other forms of date variables. The hour of day, day of week, month of year, and possibly even the year itself might all impact the energy usage. Let's extract these variables from our date object so that we can use them in our analysis. Pandas has some wonderful functionality to do this with the ```hour```, ```day```, ```dayofweek```, ```month```, and ```year``` functions. Then let's inspect the first few observations to make sure things look correct.


```python
df['hour'] = pd.DatetimeIndex(pd.to_datetime(df['Date'])).hour
df['day'] = pd.DatetimeIndex(pd.to_datetime(df['Date'])).day
df['weekday'] = df['Date'].dt.dayofweek
df['month'] = pd.DatetimeIndex(pd.to_datetime(df['Date'])).month
df['year'] = pd.DatetimeIndex(pd.to_datetime(df['Date'])).year

df.head()
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
      <th>MW</th>
      <th>Temp</th>
      <th>Date</th>
      <th>hour</th>
      <th>day</th>
      <th>weekday</th>
      <th>month</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1419.881</td>
      <td>37.0</td>
      <td>2017-02-01 00:53:00</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1379.505</td>
      <td>37.0</td>
      <td>2017-02-01 01:53:00</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1366.106</td>
      <td>36.0</td>
      <td>2017-02-01 02:53:00</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1364.453</td>
      <td>36.0</td>
      <td>2017-02-01 03:53:00</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1391.265</td>
      <td>36.0</td>
      <td>2017-02-01 04:53:00</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2017</td>
    </tr>
  </tbody>
</table>
</div>



Everything looks good in the first few observations above. If you still aren't convinced you could pull different pieces of the data frame to make sure that other observations are structured correctly.

Now we should set this Python date object as the index of our data set. This will make it easier for plotting as well as forecasting later. We can use the ```set_index``` function for this.


```python
df = df.set_index('Date')
```

Good! Now that we have our data structured as we would like, we can start the cleaning of the data. First, let's check if there are any missing values in the temperature column. The ```is.null``` function will help us here.


```python
sum(df['Temp'].isnull())
```




    37



Looks like there are 37 missing values in our temperature data. We shoudl impute those. However, we don't just want to put the average temperature in these spots as the overall average across three years probably isn't a good guess for any one hour. The temperature of the hours on either side of the missing observation would be more helpful. Let's do a linear interpolation across missing values to help with this. This will essentially draw a straight line between the two known points to fill in the missing values. We can use the ```interpolate(method='linear')``` function for this.


```python
df['Temp'] = df['Temp'].interpolate(method='linear')
```

Now let's see if we have any more missing temperature values.


```python
sum(df['Temp'].isnull())
```




    0



No more! Time to check if the energy data has any missing values.


```python
sum(df['MW'].isnull())
```




    0



No missing values there either! Perfect.

Now it is time to split the data into two pieces - training and testing. The training data set is the data set we will be building our model on, while the testing data set is what we will be reporting results on since the model wouldn't have seen it ahead of time. Using the date index we can easily do this in our data frame.


```python
#Training and Validation Split #
train = pd.DataFrame(df['2017-01-01':'2019-12-31'])
test = pd.DataFrame(df['2020-01-01':'2020-01-31'])
```

Now let's look at the first few observations for our training data set.


```python
train.head()
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
      <th>MW</th>
      <th>Temp</th>
      <th>hour</th>
      <th>day</th>
      <th>weekday</th>
      <th>month</th>
      <th>year</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-02-01 00:53:00</th>
      <td>1419.881</td>
      <td>37.0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>2017-02-01 01:53:00</th>
      <td>1379.505</td>
      <td>37.0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>2017-02-01 02:53:00</th>
      <td>1366.106</td>
      <td>36.0</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>2017-02-01 03:53:00</th>
      <td>1364.453</td>
      <td>36.0</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>2017-02-01 04:53:00</th>
      <td>1391.265</td>
      <td>36.0</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2017</td>
    </tr>
  </tbody>
</table>
</div>



Everything looks good there!

Now let's do the same for our testing data set.


```python
test.head()
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
      <th>MW</th>
      <th>Temp</th>
      <th>hour</th>
      <th>day</th>
      <th>weekday</th>
      <th>month</th>
      <th>year</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-01 00:53:00</th>
      <td>1363.428</td>
      <td>31.0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2020</td>
    </tr>
    <tr>
      <th>2020-01-01 01:53:00</th>
      <td>1335.975</td>
      <td>29.0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2020</td>
    </tr>
    <tr>
      <th>2020-01-01 02:53:00</th>
      <td>1296.817</td>
      <td>30.0</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2020</td>
    </tr>
    <tr>
      <th>2020-01-01 03:53:00</th>
      <td>1288.403</td>
      <td>30.0</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2020</td>
    </tr>
    <tr>
      <th>2020-01-01 04:53:00</th>
      <td>1292.263</td>
      <td>31.0</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2020</td>
    </tr>
  </tbody>
</table>
</div>



Excellent! We now have our data cleaned and split. By combining and cleaning the data sets, we will make the exploration of these data sets as well as the modeling of these data sets much easier for the upcoming sections!

## Building Naive Energy Model

Now that we have recreated the pieces of milestone 1 that clean and split our data we can start the modeling phase of milestone 3.

First, let's review some of the findings we have from the first two milestones:
- Energy usage changes depending on month / season
- Energy usage changes depending on day of week
- Energy usage changes depending on hour of day
- Energy usage changes depending on outside temperature
- The relationship between temperature and energy usage appears quadratic in nature

Looking at this last bullet point, we need to create a quadratic variable on temperature as temperature in the model by itself won't be enough to model energy usage. It is always good practice to standardize (mean of 0 and standard deviation of 1) any variable you are going to raise to a higher power in a regression to help prevent multicollinearity problems. We can standardize the variable *Temp* by using the ```mean``` and ```std``` functions.


```python
train['Temp_Norm'] = (train['Temp']-train['Temp'].mean())/train['Temp'].std()
```

Now that temperature is standardized (or normalized) we can just multiply it by itself to get our quadratic term.


```python
train['Temp_Norm2'] = train['Temp_Norm']**2
```

Let's do a brief look at the first few observations in our training data set to make sure that things worked as expected.


```python
train.head()
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
      <th>MW</th>
      <th>Temp</th>
      <th>hour</th>
      <th>day</th>
      <th>weekday</th>
      <th>month</th>
      <th>year</th>
      <th>Temp_Norm</th>
      <th>Temp_Norm2</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-02-01 00:53:00</th>
      <td>1419.881</td>
      <td>37.0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2017</td>
      <td>-0.871499</td>
      <td>0.759511</td>
    </tr>
    <tr>
      <th>2017-02-01 01:53:00</th>
      <td>1379.505</td>
      <td>37.0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2017</td>
      <td>-0.871499</td>
      <td>0.759511</td>
    </tr>
    <tr>
      <th>2017-02-01 02:53:00</th>
      <td>1366.106</td>
      <td>36.0</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2017</td>
      <td>-0.924494</td>
      <td>0.854690</td>
    </tr>
    <tr>
      <th>2017-02-01 03:53:00</th>
      <td>1364.453</td>
      <td>36.0</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2017</td>
      <td>-0.924494</td>
      <td>0.854690</td>
    </tr>
    <tr>
      <th>2017-02-01 04:53:00</th>
      <td>1391.265</td>
      <td>36.0</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2017</td>
      <td>-0.924494</td>
      <td>0.854690</td>
    </tr>
  </tbody>
</table>
</div>




```python
results = sm.OLS.from_formula('MW ~ Temp_Norm*C(hour) + Temp_Norm2*C(hour) + Temp_Norm*C(month) + Temp_Norm2*C(month) + C(weekday)*C(hour)', 
                              data=train).fit()
print(results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                     MW   R-squared:                       0.924
    Model:                            OLS   Adj. R-squared:                  0.924
    Method:                 Least Squares   F-statistic:                     1248.
    Date:                Fri, 09 Oct 2020   Prob (F-statistic):               0.00
    Time:                        12:43:23   Log-Likelihood:            -1.4774e+05
    No. Observations:               25536   AIC:                         2.960e+05
    Df Residuals:                   25287   BIC:                         2.980e+05
    Df Model:                         248                                         
    Covariance Type:            nonrobust                                         
    =================================================================================================
                                        coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------------------------
    Intercept                      1229.6515      9.041    136.011      0.000    1211.931    1247.372
    C(hour)[T.1]                    -21.5265      9.497     -2.267      0.023     -40.141      -2.912
    C(hour)[T.2]                    -25.0718      9.485     -2.643      0.008     -43.663      -6.480
    C(hour)[T.3]                     -6.9363      9.475     -0.732      0.464     -25.507      11.635
    C(hour)[T.4]                     48.0591      9.474      5.073      0.000      29.489      66.629
    C(hour)[T.5]                    136.7171      9.474     14.431      0.000     118.147     155.287
    C(hour)[T.6]                    211.4750      9.493     22.276      0.000     192.867     230.083
    C(hour)[T.7]                    259.9536      9.525     27.291      0.000     241.283     278.624
    C(hour)[T.8]                    291.9217      9.566     30.516      0.000     273.171     310.672
    C(hour)[T.9]                    312.8325      9.618     32.525      0.000     293.980     331.685
    C(hour)[T.10]                   324.4094      9.647     33.629      0.000     305.501     343.317
    C(hour)[T.11]                   326.6089      9.663     33.799      0.000     307.668     345.550
    C(hour)[T.12]                   333.2134      9.653     34.520      0.000     314.294     352.133
    C(hour)[T.13]                   320.6632      9.675     33.145      0.000     301.700     339.626
    C(hour)[T.14]                   309.1144      9.674     31.952      0.000     290.152     328.077
    C(hour)[T.15]                   302.4094      9.675     31.257      0.000     283.446     321.373
    C(hour)[T.16]                   308.6771      9.664     31.942      0.000     289.736     327.618
    C(hour)[T.17]                   328.0391      9.641     34.027      0.000     309.143     346.935
    C(hour)[T.18]                   341.0574      9.618     35.462      0.000     322.206     359.908
    C(hour)[T.19]                   336.0446      9.594     35.028      0.000     317.241     354.849
    C(hour)[T.20]                   297.8209      9.576     31.099      0.000     279.051     316.591
    C(hour)[T.21]                   219.9381      9.564     22.997      0.000     201.192     238.684
    C(hour)[T.22]                   126.9058      9.548     13.292      0.000     108.192     145.620
    C(hour)[T.23]                    50.0603      9.534      5.251      0.000      31.373      68.748
    C(month)[T.2]                   -10.9536      6.587     -1.663      0.096     -23.864       1.957
    C(month)[T.3]                   -58.4207      6.602     -8.848      0.000     -71.362     -45.480
    C(month)[T.4]                  -110.3894      6.439    -17.143      0.000    -123.011     -97.768
    C(month)[T.5]                  -122.2577      6.548    -18.671      0.000    -135.092    -109.423
    C(month)[T.6]                  -105.6638      8.055    -13.117      0.000    -121.453     -89.875
    C(month)[T.7]                   -87.2652     14.169     -6.159      0.000    -115.037     -59.494
    C(month)[T.8]                   -80.4514     11.193     -7.187      0.000    -102.391     -58.512
    C(month)[T.9]                   -91.9013      7.370    -12.470      0.000    -106.347     -77.456
    C(month)[T.10]                 -111.9445      6.423    -17.428      0.000    -124.535     -99.354
    C(month)[T.11]                  -45.0605      6.751     -6.675      0.000     -58.293     -31.828
    C(month)[T.12]                  -18.2699      7.454     -2.451      0.014     -32.881      -3.659
    C(weekday)[T.1]                   6.1527      9.085      0.677      0.498     -11.654      23.960
    C(weekday)[T.2]                  32.7819      9.087      3.608      0.000      14.971      50.593
    C(weekday)[T.3]                  37.2304      9.092      4.095      0.000      19.409      55.052
    C(weekday)[T.4]                  37.0628      9.087      4.078      0.000      19.251      54.875
    C(weekday)[T.5]                  17.1413      9.087      1.886      0.059      -0.670      34.953
    C(weekday)[T.6]                  -9.1433      9.083     -1.007      0.314     -26.947       8.660
    C(weekday)[T.1]:C(hour)[T.1]      1.2708     12.844      0.099      0.921     -23.905      26.446
    C(weekday)[T.2]:C(hour)[T.1]     -7.0511     12.848     -0.549      0.583     -32.233      18.131
    C(weekday)[T.3]:C(hour)[T.1]     -1.0763     12.852     -0.084      0.933     -26.268      24.115
    C(weekday)[T.4]:C(hour)[T.1]     -0.0966     12.845     -0.008      0.994     -25.273      25.080
    C(weekday)[T.5]:C(hour)[T.1]     -5.4920     12.846     -0.428      0.669     -30.671      19.687
    C(weekday)[T.6]:C(hour)[T.1]     -2.4908     12.843     -0.194      0.846     -27.664      22.682
    C(weekday)[T.1]:C(hour)[T.2]     -4.6528     12.844     -0.362      0.717     -29.828      20.523
    C(weekday)[T.2]:C(hour)[T.2]     -6.1394     12.847     -0.478      0.633     -31.320      19.041
    C(weekday)[T.3]:C(hour)[T.2]    -11.4852     12.854     -0.894      0.372     -36.679      13.709
    C(weekday)[T.4]:C(hour)[T.2]     -9.3409     12.845     -0.727      0.467     -34.518      15.836
    C(weekday)[T.5]:C(hour)[T.2]    -20.8512     12.846     -1.623      0.105     -46.031       4.328
    C(weekday)[T.6]:C(hour)[T.2]    -16.2096     12.843     -1.262      0.207     -41.382       8.963
    C(weekday)[T.1]:C(hour)[T.3]     -4.5071     12.844     -0.351      0.726     -29.682      20.668
    C(weekday)[T.2]:C(hour)[T.3]     -8.5981     12.846     -0.669      0.503     -33.778      16.581
    C(weekday)[T.3]:C(hour)[T.3]    -13.8075     12.853     -1.074      0.283     -39.000      11.385
    C(weekday)[T.4]:C(hour)[T.3]    -12.3027     12.845     -0.958      0.338     -37.480      12.874
    C(weekday)[T.5]:C(hour)[T.3]    -34.6792     12.846     -2.700      0.007     -59.859      -9.499
    C(weekday)[T.6]:C(hour)[T.3]    -37.6174     12.843     -2.929      0.003     -62.791     -12.444
    C(weekday)[T.1]:C(hour)[T.4]     -5.2068     12.844     -0.405      0.685     -30.383      19.969
    C(weekday)[T.2]:C(hour)[T.4]     -6.0556     12.846     -0.471      0.637     -31.234      19.123
    C(weekday)[T.3]:C(hour)[T.4]     -9.5735     12.852     -0.745      0.456     -34.765      15.618
    C(weekday)[T.4]:C(hour)[T.4]    -12.6740     12.845     -0.987      0.324     -37.851      12.503
    C(weekday)[T.5]:C(hour)[T.4]    -67.7274     12.846     -5.272      0.000     -92.907     -42.548
    C(weekday)[T.6]:C(hour)[T.4]    -79.0581     12.844     -6.155      0.000    -104.232     -53.884
    C(weekday)[T.1]:C(hour)[T.5]     -3.4440     12.845     -0.268      0.789     -28.620      21.732
    C(weekday)[T.2]:C(hour)[T.5]     -1.9474     12.846     -0.152      0.880     -27.125      23.231
    C(weekday)[T.3]:C(hour)[T.5]     -8.4281     12.852     -0.656      0.512     -33.619      16.763
    C(weekday)[T.4]:C(hour)[T.5]    -12.5802     12.845     -0.979      0.327     -37.757      12.596
    C(weekday)[T.5]:C(hour)[T.5]   -128.9868     12.846    -10.041      0.000    -154.166    -103.807
    C(weekday)[T.6]:C(hour)[T.5]   -149.5590     12.843    -11.645      0.000    -174.733    -124.385
    C(weekday)[T.1]:C(hour)[T.6]      5.4641     12.844      0.425      0.671     -19.711      30.640
    C(weekday)[T.2]:C(hour)[T.6]      3.7926     12.846      0.295      0.768     -21.386      28.972
    C(weekday)[T.3]:C(hour)[T.6]     -3.2823     12.852     -0.255      0.798     -28.473      21.908
    C(weekday)[T.4]:C(hour)[T.6]     -5.9154     12.845     -0.461      0.645     -31.092      19.261
    C(weekday)[T.5]:C(hour)[T.6]   -173.4048     12.847    -13.498      0.000    -198.585    -148.225
    C(weekday)[T.6]:C(hour)[T.6]   -203.5208     12.843    -15.847      0.000    -228.694    -178.348
    C(weekday)[T.1]:C(hour)[T.7]      8.3028     12.844      0.646      0.518     -16.873      33.479
    C(weekday)[T.2]:C(hour)[T.7]      4.2140     12.847      0.328      0.743     -20.967      29.395
    C(weekday)[T.3]:C(hour)[T.7]      1.5437     12.851      0.120      0.904     -23.646      26.733
    C(weekday)[T.4]:C(hour)[T.7]     -3.0864     12.845     -0.240      0.810     -28.263      22.090
    C(weekday)[T.5]:C(hour)[T.7]   -174.5214     12.846    -13.585      0.000    -199.701    -149.342
    C(weekday)[T.6]:C(hour)[T.7]   -213.6351     12.843    -16.635      0.000    -238.808    -188.462
    C(weekday)[T.1]:C(hour)[T.8]      6.2398     12.845      0.486      0.627     -18.938      31.418
    C(weekday)[T.2]:C(hour)[T.8]      0.2350     12.848      0.018      0.985     -24.947      25.417
    C(weekday)[T.3]:C(hour)[T.8]     -0.9942     12.852     -0.077      0.938     -26.184      24.196
    C(weekday)[T.4]:C(hour)[T.8]     -4.2566     12.845     -0.331      0.740     -29.434      20.921
    C(weekday)[T.5]:C(hour)[T.8]   -160.6811     12.847    -12.508      0.000    -185.861    -135.501
    C(weekday)[T.6]:C(hour)[T.8]   -202.4275     12.843    -15.762      0.000    -227.601    -177.254
    C(weekday)[T.1]:C(hour)[T.9]      7.3545     12.846      0.573      0.567     -17.824      32.533
    C(weekday)[T.2]:C(hour)[T.9]     -8.6074     12.849     -0.670      0.503     -33.792      16.577
    C(weekday)[T.3]:C(hour)[T.9]     -4.3096     12.852     -0.335      0.737     -29.500      20.880
    C(weekday)[T.4]:C(hour)[T.9]     -5.2384     12.846     -0.408      0.683     -30.417      19.940
    C(weekday)[T.5]:C(hour)[T.9]   -151.5667     12.848    -11.797      0.000    -176.749    -126.385
    C(weekday)[T.6]:C(hour)[T.9]   -187.9846     12.843    -14.637      0.000    -213.158    -162.811
    C(weekday)[T.1]:C(hour)[T.10]     4.3423     12.847      0.338      0.735     -20.839      29.523
    C(weekday)[T.2]:C(hour)[T.10]   -12.2992     12.851     -0.957      0.339     -37.487      12.889
    C(weekday)[T.3]:C(hour)[T.10]    -5.4245     12.852     -0.422      0.673     -30.615      19.766
    C(weekday)[T.4]:C(hour)[T.10]   -11.1744     12.847     -0.870      0.384     -36.355      14.006
    C(weekday)[T.5]:C(hour)[T.10]  -153.6771     12.848    -11.961      0.000    -178.860    -128.495
    C(weekday)[T.6]:C(hour)[T.10]  -177.2951     12.844    -13.804      0.000    -202.469    -152.121
    C(weekday)[T.1]:C(hour)[T.11]     3.8983     12.848      0.303      0.762     -21.284      29.081
    C(weekday)[T.2]:C(hour)[T.11]   -12.1382     12.852     -0.944      0.345     -37.328      13.052
    C(weekday)[T.3]:C(hour)[T.11]    -7.2745     12.853     -0.566      0.571     -32.466      17.917
    C(weekday)[T.4]:C(hour)[T.11]   -19.2054     12.848     -1.495      0.135     -44.388       5.977
    C(weekday)[T.5]:C(hour)[T.11]  -155.4069     12.849    -12.095      0.000    -180.591    -130.223
    C(weekday)[T.6]:C(hour)[T.11]  -174.2920     12.844    -13.570      0.000    -199.467    -149.117
    C(weekday)[T.1]:C(hour)[T.12]    -6.0155     12.850     -0.468      0.640     -31.203      19.172
    C(weekday)[T.2]:C(hour)[T.12]   -19.4370     12.855     -1.512      0.131     -44.633       5.759
    C(weekday)[T.3]:C(hour)[T.12]   -15.7628     12.855     -1.226      0.220     -40.959       9.433
    C(weekday)[T.4]:C(hour)[T.12]   -26.9476     12.849     -2.097      0.036     -52.133      -1.762
    C(weekday)[T.5]:C(hour)[T.12]  -173.4917     12.849    -13.502      0.000    -198.677    -148.306
    C(weekday)[T.6]:C(hour)[T.12]  -174.9069     12.845    -13.617      0.000    -200.083    -149.731
    C(weekday)[T.1]:C(hour)[T.13]     4.2963     12.850      0.334      0.738     -20.890      29.483
    C(weekday)[T.2]:C(hour)[T.13]    -8.5240     12.853     -0.663      0.507     -33.717      16.669
    C(weekday)[T.3]:C(hour)[T.13]    -8.0834     12.854     -0.629      0.529     -33.278      17.111
    C(weekday)[T.4]:C(hour)[T.13]   -19.2660     12.849     -1.499      0.134     -44.451       5.919
    C(weekday)[T.5]:C(hour)[T.13]  -168.8350     12.850    -13.139      0.000    -194.021    -143.649
    C(weekday)[T.6]:C(hour)[T.13]  -156.0927     12.846    -12.151      0.000    -181.271    -130.914
    C(weekday)[T.1]:C(hour)[T.14]     3.5832     12.853      0.279      0.780     -21.609      28.775
    C(weekday)[T.2]:C(hour)[T.14]   -10.7601     12.854     -0.837      0.403     -35.956      14.435
    C(weekday)[T.3]:C(hour)[T.14]    -9.4850     12.855     -0.738      0.461     -34.682      15.712
    C(weekday)[T.4]:C(hour)[T.14]   -29.6413     12.851     -2.307      0.021     -54.830      -4.453
    C(weekday)[T.5]:C(hour)[T.14]  -169.1355     12.850    -13.162      0.000    -194.323    -143.948
    C(weekday)[T.6]:C(hour)[T.14]  -146.4668     12.846    -11.402      0.000    -171.646    -121.288
    C(weekday)[T.1]:C(hour)[T.15]     7.6746     12.852      0.597      0.550     -17.515      32.864
    C(weekday)[T.2]:C(hour)[T.15]    -7.8948     12.853     -0.614      0.539     -33.087      17.298
    C(weekday)[T.3]:C(hour)[T.15]    -4.4390     12.854     -0.345      0.730     -29.633      20.755
    C(weekday)[T.4]:C(hour)[T.15]   -36.2094     12.851     -2.818      0.005     -61.398     -11.021
    C(weekday)[T.5]:C(hour)[T.15]  -151.9363     12.851    -11.823      0.000    -177.124    -126.748
    C(weekday)[T.6]:C(hour)[T.15]  -131.3300     12.846    -10.224      0.000    -156.509    -106.151
    C(weekday)[T.1]:C(hour)[T.16]    10.1146     12.852      0.787      0.431     -15.075      35.304
    C(weekday)[T.2]:C(hour)[T.16]   -13.4999     12.855     -1.050      0.294     -38.696      11.696
    C(weekday)[T.3]:C(hour)[T.16]    -3.8428     12.853     -0.299      0.765     -29.036      21.351
    C(weekday)[T.4]:C(hour)[T.16]   -41.0287     12.850     -3.193      0.001     -66.216     -15.841
    C(weekday)[T.5]:C(hour)[T.16]  -150.7744     12.850    -11.733      0.000    -175.961    -125.588
    C(weekday)[T.6]:C(hour)[T.16]  -107.2603     12.846     -8.350      0.000    -132.438     -82.082
    C(weekday)[T.1]:C(hour)[T.17]     6.1321     12.851      0.477      0.633     -19.057      31.321
    C(weekday)[T.2]:C(hour)[T.17]   -15.8141     12.854     -1.230      0.219     -41.009       9.381
    C(weekday)[T.3]:C(hour)[T.17]   -13.7108     12.853     -1.067      0.286     -38.904      11.483
    C(weekday)[T.4]:C(hour)[T.17]   -60.7848     12.851     -4.730      0.000     -85.973     -35.597
    C(weekday)[T.5]:C(hour)[T.17]  -141.3721     12.850    -11.002      0.000    -166.558    -116.186
    C(weekday)[T.6]:C(hour)[T.17]   -95.5304     12.845     -7.437      0.000    -120.707     -70.354
    C(weekday)[T.1]:C(hour)[T.18]     2.1231     12.851      0.165      0.869     -23.065      27.311
    C(weekday)[T.2]:C(hour)[T.18]   -14.7173     12.853     -1.145      0.252     -39.910      10.475
    C(weekday)[T.3]:C(hour)[T.18]   -18.6454     12.853     -1.451      0.147     -43.838       6.547
    C(weekday)[T.4]:C(hour)[T.18]   -74.4545     12.851     -5.794      0.000     -99.643     -49.266
    C(weekday)[T.5]:C(hour)[T.18]  -134.3285     12.850    -10.454      0.000    -159.515    -109.142
    C(weekday)[T.6]:C(hour)[T.18]   -89.5318     12.845     -6.970      0.000    -114.708     -64.355
    C(weekday)[T.1]:C(hour)[T.19]     2.9413     12.851      0.229      0.819     -22.247      28.130
    C(weekday)[T.2]:C(hour)[T.19]    -6.7866     12.852     -0.528      0.597     -31.977      18.404
    C(weekday)[T.3]:C(hour)[T.19]   -16.1142     12.852     -1.254      0.210     -41.305       9.077
    C(weekday)[T.4]:C(hour)[T.19]   -72.3864     12.851     -5.633      0.000     -97.574     -47.198
    C(weekday)[T.5]:C(hour)[T.19]  -126.0005     12.850     -9.806      0.000    -151.187    -100.814
    C(weekday)[T.6]:C(hour)[T.19]   -74.0456     12.845     -5.765      0.000     -99.222     -48.870
    C(weekday)[T.1]:C(hour)[T.20]     4.2810     12.851      0.333      0.739     -20.908      29.470
    C(weekday)[T.2]:C(hour)[T.20]   -10.0548     12.852     -0.782      0.434     -35.245      15.136
    C(weekday)[T.3]:C(hour)[T.20]   -15.0396     12.852     -1.170      0.242     -40.231      10.151
    C(weekday)[T.4]:C(hour)[T.20]   -65.5416     12.850     -5.101      0.000     -90.728     -40.355
    C(weekday)[T.5]:C(hour)[T.20]  -106.0449     12.849     -8.253      0.000    -131.231     -80.859
    C(weekday)[T.6]:C(hour)[T.20]   -52.1952     12.845     -4.064      0.000     -77.372     -27.019
    C(weekday)[T.1]:C(hour)[T.21]    10.6535     12.851      0.829      0.407     -14.534      35.841
    C(weekday)[T.2]:C(hour)[T.21]   -14.8287     12.852     -1.154      0.249     -40.020      10.362
    C(weekday)[T.3]:C(hour)[T.21]   -18.6243     12.852     -1.449      0.147     -43.816       6.567
    C(weekday)[T.4]:C(hour)[T.21]   -50.5523     12.850     -3.934      0.000     -75.740     -25.365
    C(weekday)[T.5]:C(hour)[T.21]   -80.5735     12.850     -6.270      0.000    -105.760     -55.387
    C(weekday)[T.6]:C(hour)[T.21]   -35.5806     12.845     -2.770      0.006     -60.757     -10.404
    C(weekday)[T.1]:C(hour)[T.22]    13.5161     12.851      1.052      0.293     -11.672      38.705
    C(weekday)[T.2]:C(hour)[T.22]   -11.4437     12.852     -0.890      0.373     -36.635      13.748
    C(weekday)[T.3]:C(hour)[T.22]   -12.9284     12.852     -1.006      0.314     -38.120      12.263
    C(weekday)[T.4]:C(hour)[T.22]   -35.8049     12.851     -2.786      0.005     -60.993     -10.617
    C(weekday)[T.5]:C(hour)[T.22]   -48.8802     12.850     -3.804      0.000     -74.067     -23.693
    C(weekday)[T.6]:C(hour)[T.22]   -16.6274     12.845     -1.294      0.196     -41.804       8.549
    C(weekday)[T.1]:C(hour)[T.23]    15.8422     12.852      1.233      0.218      -9.348      41.032
    C(weekday)[T.2]:C(hour)[T.23]    -5.2718     12.853     -0.410      0.682     -30.463      19.920
    C(weekday)[T.3]:C(hour)[T.23]    -6.3897     12.853     -0.497      0.619     -31.581      18.802
    C(weekday)[T.4]:C(hour)[T.23]   -23.8773     12.852     -1.858      0.063     -49.067       1.313
    C(weekday)[T.5]:C(hour)[T.23]   -31.3646     12.851     -2.441      0.015     -56.553      -6.176
    C(weekday)[T.6]:C(hour)[T.23]    -0.4384     12.846     -0.034      0.973     -25.617      24.740
    Temp_Norm                       -68.6609      9.453     -7.263      0.000     -87.190     -50.132
    Temp_Norm:C(hour)[T.1]           -5.3790      5.038     -1.068      0.286     -15.254       4.496
    Temp_Norm:C(hour)[T.2]           -2.1789      5.163     -0.422      0.673     -12.298       7.940
    Temp_Norm:C(hour)[T.3]            3.6797      5.259      0.700      0.484      -6.628      13.988
    Temp_Norm:C(hour)[T.4]           15.8257      5.331      2.969      0.003       5.377      26.275
    Temp_Norm:C(hour)[T.5]           10.0501      5.315      1.891      0.059      -0.367      20.467
    Temp_Norm:C(hour)[T.6]          -18.3154      5.090     -3.598      0.000     -28.292      -8.339
    Temp_Norm:C(hour)[T.7]          -37.3330      4.765     -7.834      0.000     -46.673     -27.993
    Temp_Norm:C(hour)[T.8]          -47.7697      4.505    -10.603      0.000     -56.601     -38.939
    Temp_Norm:C(hour)[T.9]          -57.0840      4.347    -13.131      0.000     -65.605     -48.563
    Temp_Norm:C(hour)[T.10]         -61.0501      4.286    -14.245      0.000     -69.450     -52.650
    Temp_Norm:C(hour)[T.11]         -58.5721      4.277    -13.695      0.000     -66.955     -50.189
    Temp_Norm:C(hour)[T.12]         -54.8722      4.292    -12.785      0.000     -63.285     -46.460
    Temp_Norm:C(hour)[T.13]         -48.5805      4.312    -11.268      0.000     -57.031     -40.130
    Temp_Norm:C(hour)[T.14]         -37.7095      4.320     -8.729      0.000     -46.177     -29.242
    Temp_Norm:C(hour)[T.15]         -23.0896      4.320     -5.345      0.000     -31.556     -14.623
    Temp_Norm:C(hour)[T.16]         -16.1147      4.301     -3.747      0.000     -24.545      -7.684
    Temp_Norm:C(hour)[T.17]         -27.4735      4.298     -6.392      0.000     -35.899     -19.048
    Temp_Norm:C(hour)[T.18]         -23.0480      4.335     -5.317      0.000     -31.544     -14.552
    Temp_Norm:C(hour)[T.19]          -1.9596      4.419     -0.443      0.657     -10.620       6.701
    Temp_Norm:C(hour)[T.20]          12.0138      4.500      2.670      0.008       3.194      20.834
    Temp_Norm:C(hour)[T.21]           8.5639      4.606      1.859      0.063      -0.465      17.592
    Temp_Norm:C(hour)[T.22]           4.2954      4.721      0.910      0.363      -4.957      13.548
    Temp_Norm:C(hour)[T.23]           2.4786      4.840      0.512      0.609      -7.007      11.965
    Temp_Norm:C(month)[T.2]          16.2930     10.020      1.626      0.104      -3.347      35.933
    Temp_Norm:C(month)[T.3]          23.3681     10.105      2.312      0.021       3.561      43.175
    Temp_Norm:C(month)[T.4]          65.7583      9.199      7.148      0.000      47.728      83.789
    Temp_Norm:C(month)[T.5]         159.1896     10.874     14.640      0.000     137.876     180.503
    Temp_Norm:C(month)[T.6]         166.9858     15.377     10.860      0.000     136.847     197.125
    Temp_Norm:C(month)[T.7]         264.6119     25.623     10.327      0.000     214.389     314.835
    Temp_Norm:C(month)[T.8]         178.5430     22.036      8.102      0.000     135.352     221.734
    Temp_Norm:C(month)[T.9]         133.1027     13.847      9.612      0.000     105.962     160.244
    Temp_Norm:C(month)[T.10]        127.0785      9.337     13.611      0.000     108.778     145.379
    Temp_Norm:C(month)[T.11]         23.0011     11.289      2.038      0.042       0.875      45.128
    Temp_Norm:C(month)[T.12]         -6.1431     11.769     -0.522      0.602     -29.212      16.926
    Temp_Norm2                       46.8979      4.051     11.578      0.000      38.958      54.838
    Temp_Norm2:C(hour)[T.1]          -6.7922      4.045     -1.679      0.093     -14.721       1.137
    Temp_Norm2:C(hour)[T.2]          -7.2007      4.068     -1.770      0.077     -15.173       0.772
    Temp_Norm2:C(hour)[T.3]          -8.7551      4.072     -2.150      0.032     -16.737      -0.773
    Temp_Norm2:C(hour)[T.4]          -8.3277      4.085     -2.038      0.042     -16.335      -0.320
    Temp_Norm2:C(hour)[T.5]         -14.1854      4.062     -3.492      0.000     -22.147      -6.224
    Temp_Norm2:C(hour)[T.6]         -23.7351      3.968     -5.982      0.000     -31.513     -15.957
    Temp_Norm2:C(hour)[T.7]         -30.2120      3.837     -7.874      0.000     -37.733     -22.691
    Temp_Norm2:C(hour)[T.8]         -35.5661      3.770     -9.434      0.000     -42.955     -28.177
    Temp_Norm2:C(hour)[T.9]         -37.6081      3.753    -10.020      0.000     -44.964     -30.252
    Temp_Norm2:C(hour)[T.10]        -36.8893      3.747     -9.844      0.000     -44.235     -29.544
    Temp_Norm2:C(hour)[T.11]        -32.2166      3.764     -8.559      0.000     -39.595     -24.839
    Temp_Norm2:C(hour)[T.12]        -29.6368      3.774     -7.853      0.000     -37.033     -22.240
    Temp_Norm2:C(hour)[T.13]        -26.8816      3.788     -7.097      0.000     -34.306     -19.457
    Temp_Norm2:C(hour)[T.14]        -20.2397      3.789     -5.342      0.000     -27.666     -12.813
    Temp_Norm2:C(hour)[T.15]        -14.8185      3.790     -3.909      0.000     -22.248      -7.389
    Temp_Norm2:C(hour)[T.16]         -9.0722      3.774     -2.404      0.016     -16.469      -1.676
    Temp_Norm2:C(hour)[T.17]         -0.6812      3.770     -0.181      0.857      -8.070       6.708
    Temp_Norm2:C(hour)[T.18]          3.1939      3.811      0.838      0.402      -4.277      10.664
    Temp_Norm2:C(hour)[T.19]          7.9381      3.870      2.051      0.040       0.352      15.524
    Temp_Norm2:C(hour)[T.20]         13.7865      3.905      3.530      0.000       6.132      21.441
    Temp_Norm2:C(hour)[T.21]         16.3843      3.962      4.136      0.000       8.619      24.149
    Temp_Norm2:C(hour)[T.22]         13.2291      4.004      3.304      0.001       5.381      21.077
    Temp_Norm2:C(hour)[T.23]          6.8167      4.039      1.688      0.091      -1.099      14.733
    Temp_Norm2:C(month)[T.2]          9.9211      4.109      2.414      0.016       1.866      17.976
    Temp_Norm2:C(month)[T.3]         22.8510      4.289      5.327      0.000      14.443      31.258
    Temp_Norm2:C(month)[T.4]         90.0379      4.505     19.985      0.000      81.207      98.869
    Temp_Norm2:C(month)[T.5]        187.5059      6.130     30.591      0.000     175.492     199.520
    Temp_Norm2:C(month)[T.6]        269.1734      8.172     32.937      0.000     253.155     285.192
    Temp_Norm2:C(month)[T.7]        224.0601     11.515     19.458      0.000     201.490     246.631
    Temp_Norm2:C(month)[T.8]        286.2384     11.027     25.958      0.000     264.625     307.852
    Temp_Norm2:C(month)[T.9]        270.5962      7.293     37.105      0.000     256.302     284.890
    Temp_Norm2:C(month)[T.10]       229.8348      4.986     46.094      0.000     220.061     239.608
    Temp_Norm2:C(month)[T.11]         1.0268      5.737      0.179      0.858     -10.218      12.272
    Temp_Norm2:C(month)[T.12]        -7.6658      4.506     -1.701      0.089     -16.498       1.166
    ==============================================================================
    Omnibus:                     3046.293   Durbin-Watson:                   0.224
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):            21208.345
    Skew:                           0.357   Prob(JB):                         0.00
    Kurtosis:                       7.407   Cond. No.                         331.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


All of those terms appeared significant too! Excellent. Now we have our naive energy model. It takes into account the hour of day, day of week, month of year, and the complicated relationship with temperature. 

Time to see how good our predictions are. One evaluation of model performance is the mean absolute percentage error (MAPE). This evaluates on average how far off are our predictions in terms of percentages. We need to get our predictions from our training data set. The ```fittedvalues``` function will do that for us. Then we can calculate the MAPE ourselves.


```python
train['fitted'] = results.fittedvalues

train['APE'] = abs((train['MW']-train['fitted'])/train['MW'])*100
print("Training Naive Model MAPE is: ", train['APE'].mean())
```

    Training Naive Model MAPE is:  3.5119541032055452


On average, our model incorrectly predicted energy usage by a little over 3.5%! That gives us a good baseline to compare our future models with.




```python
test['Temp_Norm'] = (test['Temp']-test['Temp'].mean())/test['Temp'].std()
test['Temp_Norm2'] = test['Temp_Norm']**2
```

Let's forecast out our model by scoring the test data set with the linear regression we built. Remember, we don't want to build a model on the test data set, just run the observations through the equation we got from the training model. These are our January 2020 predictions! The ```predict``` function will help us with this. We need to specify which data set we are predicting as you see with the ```predict(test)``` below. Let's look at the first few observations from this prediction!


```python
test['pred'] = results.predict(test)

test.head()
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
      <th>MW</th>
      <th>Temp</th>
      <th>hour</th>
      <th>day</th>
      <th>weekday</th>
      <th>month</th>
      <th>year</th>
      <th>Temp_Norm</th>
      <th>Temp_Norm2</th>
      <th>pred</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-01 00:53:00</th>
      <td>1363.428</td>
      <td>31.0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2020</td>
      <td>-0.435454</td>
      <td>0.189621</td>
      <td>1301.224887</td>
    </tr>
    <tr>
      <th>2020-01-01 01:53:00</th>
      <td>1335.975</td>
      <td>29.0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2020</td>
      <td>-0.627840</td>
      <td>0.394184</td>
      <td>1296.150033</td>
    </tr>
    <tr>
      <th>2020-01-01 02:53:00</th>
      <td>1296.817</td>
      <td>30.0</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2020</td>
      <td>-0.531647</td>
      <td>0.282649</td>
      <td>1280.104337</td>
    </tr>
    <tr>
      <th>2020-01-01 03:53:00</th>
      <td>1288.403</td>
      <td>30.0</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2020</td>
      <td>-0.531647</td>
      <td>0.282649</td>
      <td>1292.227132</td>
    </tr>
    <tr>
      <th>2020-01-01 04:53:00</th>
      <td>1292.263</td>
      <td>31.0</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2020</td>
      <td>-0.435454</td>
      <td>0.189621</td>
      <td>1334.757899</td>
    </tr>
  </tbody>
</table>
</div>



Good! Now let's plot our predictions for the test data set against the actual values.


```python
test['MW'].plot(color = 'blue', figsize=(9,7))

plt.ylabel('MW Hours')
plt.xlabel('Date')

test['pred'].plot(color = 'green', linestyle = 'dashed', figsize=(9,7))

plt.legend(loc="best");

plt.show()
```


    
![png](Part-4---Dynamic-Time-Series-Model_files/Part-4---Dynamic-Time-Series-Model_54_0.png)
    


Those look like rather good predictions! Let's see what the MAPE is on these.


```python
test['APE'] = abs((test['MW']-test['pred'])/test['MW'])*100
print("Naive Model MAPE is: ", test['APE'].mean())
```

    Naive Model MAPE is:  4.3947190107463365


Great! Remember, the MAPE is probably going to be higher because our model hasn't seen this data before. This is a great way to truly evaluate how well your model will do when deployed in a real world setting since you won't know energy data before you predict it. Looks like our model is only off by 4.4% on average.

The foundation is laid in this step. Model building can be complicated and sometimes it is hard to know when to stop. The best plan is to build a foundational model that you can try to build upon and/or outperform with later editions of your model. Without a good baseline, you won't know how good your final model is. These seasonal effects of hours of day, days of week, months of year as well as the temperature effects build a great first attempt at forecasting future energy usage.

This is a great initial model if your boss needs a check-in to see your progress. This model gets you a long way there since you have incorporated temperature's complicated relationship. In the next milestones you get to build on this great foundation to really show your boss what you can do!

## Dynamic Time Series Model

Now that we have recreated the important pieces of milestones 1 and 3, we can move on to milestone 4's objectives. 

We have a great foundational, naive energy model. This model accounts for the energy's relationship with hour of day, day of week, month of year, and the complicated relationship with temperature. However, previous values of energy usage probably play some impact on the prediction of current energy usage. This is the basis for time series modeling!

First, we need to get the residuals from the naive energy model. We will use these residuals as inputs to our dynamic time series model. We can use the ```resid``` function to do this.


```python
train['resid'] = results.resid
```

Just like with our original energy data, let's plot the residuals from our model to see what we have.


```python
ax1 = train['resid'].plot(color = 'blue', figsize=(9,7))

ax1.set_ylabel('Residuals')
ax1.set_xlabel('Date')

plt.show()
```


    
![png](Part-4---Dynamic-Time-Series-Model_files/Part-4---Dynamic-Time-Series-Model_62_0.png)
    


Looks like we still see the seasonal effects that we had in our original data. Summer months seem to have bigger residuals (model errors) than the rest of the year. 

Let's zoom in on a specific week from December to see what our residuals look like.


```python
ax1 = train['2019-12-01':'2019-12-07']['resid'].plot(color = 'blue', figsize=(9,7))

ax1.set_ylabel('Residuals')
ax1.set_xlabel('Date')

plt.show()
```


    
![png](Part-4---Dynamic-Time-Series-Model_files/Part-4---Dynamic-Time-Series-Model_64_0.png)
    


It appears that we still have some daily effects as well. Different hours of the day we do worse at predicting energy than other hours. Let's see if time series models can help us correct this!

### Exponential Smoothing Models 

#### Winters Seasonal Exponential Smoothing Model 

Exponential smoothing models can be used to predict a variety of different types of data. There are different models depending on whether our data is trending and/or contains a seasonal effect as well. The Winters exponential smoothing model accounts for seasonal effects while the Holt exponential smoothing model accounts for trend. Since our residuals don't trend, but still have a seasonal effect we should use the Winter's Seasonal Exponential Smoothing Model. Let's try to forecast our energy residuals with this model!

The ```ExponentialSmoothing``` function will help us with this. Remember that we don't want a trend. Also, since our data is hourly and appears we have a daily effect, we set the seasonal periods to 24. You can play around with either an additive (```seasonal='add'```) or multiplicative (```seasonal='mult'```) effect. Use the resources provided with the milestone if you are interested in learning the difference between those!


```python
mod_tes = ExponentialSmoothing(train['resid'], trend=None, seasonal='add', seasonal_periods=24)

res_tes = mod_tes.fit()
print(res_tes.summary())
```

    C:\Users\adlabarr\Anaconda3\lib\site-packages\statsmodels\tsa\base\tsa_model.py:218: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      ' ignored when e.g. forecasting.', ValueWarning)


                           ExponentialSmoothing Model Results                       
    ================================================================================
    Dep. Variable:                    endog   No. Observations:                25536
    Model:             ExponentialSmoothing   SSE                       35226454.604
    Optimized:                         True   AIC                         184663.578
    Trend:                             None   BIC                         184875.422
    Seasonal:                      Additive   AICC                        184663.641
    Seasonal Periods:                    24   Date:                 Fri, 09 Oct 2020
    Box-Cox:                          False   Time:                         12:43:36
    Box-Cox Coeff.:                    None                                         
    =================================================================================
                              coeff                 code              optimized      
    ---------------------------------------------------------------------------------
    smoothing_level               0.8841008                alpha                 True
    smoothing_seasonal            0.0015706                gamma                 True
    initial_level                 1.9693647                  l.0                 True
    initial_seasons.0             42.625468                  s.0                 True
    initial_seasons.1             40.492795                  s.1                 True
    initial_seasons.2             38.905314                  s.2                 True
    initial_seasons.3             37.254289                  s.3                 True
    initial_seasons.4             37.787142                  s.4                 True
    initial_seasons.5             38.090428                  s.5                 True
    initial_seasons.6             41.466053                  s.6                 True
    initial_seasons.7             44.329264                  s.7                 True
    initial_seasons.8             46.693702                  s.8                 True
    initial_seasons.9             49.819394                  s.9                 True
    initial_seasons.10            51.133331                 s.10                 True
    initial_seasons.11            51.513382                 s.11                 True
    initial_seasons.12            52.262008                 s.12                 True
    initial_seasons.13            52.104013                 s.13                 True
    initial_seasons.14            51.733222                 s.14                 True
    initial_seasons.15            52.170736                 s.15                 True
    initial_seasons.16            52.468410                 s.16                 True
    initial_seasons.17            39.051232                 s.17                 True
    initial_seasons.18            41.584666                 s.18                 True
    initial_seasons.19            44.644031                 s.19                 True
    initial_seasons.20            47.282884                 s.20                 True
    initial_seasons.21            47.372397                 s.21                 True
    initial_seasons.22            46.417112                 s.22                 True
    initial_seasons.23            43.476889                 s.23                 True
    ---------------------------------------------------------------------------------


We can then use the ```forecast``` functions to forecast out the month of January which is 744 observations. Careful though. These forecasts are the **residuals**.


```python
forecast = pd.DataFrame(res_tes.forecast(744))
forecast.index = test.index.copy()

ax1 = forecast.plot(color = 'blue', figsize=(9,7))

ax1.set_ylabel('Forecast')
ax1.set_xlabel('Date')

plt.show()
```

    C:\Users\adlabarr\Anaconda3\lib\site-packages\statsmodels\tsa\base\tsa_model.py:583: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
      ValueWarning)



    
![png](Part-4---Dynamic-Time-Series-Model_files/Part-4---Dynamic-Time-Series-Model_71_1.png)
    


Let's go ahead and save these model fitted values (from the training data) and forecasts (the test data) to our respective data frames. That way we can evaluate them best.


```python
train['fitted_resid'] = res_tes.fittedvalues
test['pred_resid'] = forecast
```

Our energy forecast isn't the residual forecast. It is the combination the forecast from the naive model **and** the new exponential smoothing model on the residuals. Add these two forecasts together to get your new dynamic energy model forecasts for each the training and test data sets. 


```python
train['fitted_ESM'] = train['fitted'] + train['fitted_resid']
test['pred_ESM'] = test['pred'] + test['pred_resid']
```

Now let's view our forecast just like we did with the naive model!


```python
test['MW'].plot(color = 'blue', figsize=(9,7))

plt.ylabel('MW Hours')
plt.xlabel('Date')

test['pred_ESM'].plot(color = 'green', linestyle = 'dashed', figsize=(9,7))

plt.legend(loc="best");

plt.show()
```


    
![png](Part-4---Dynamic-Time-Series-Model_files/Part-4---Dynamic-Time-Series-Model_77_0.png)
    


Just like with the naive model, let's calculate the MAPE for our new dynamic energy model using exponential smoothing. First let's do this on the training data.


```python
train['APE_ESM'] = abs((train['MW']-train['fitted_ESM'])/train['MW'])*100
print("Training Naive + ESM Model MAPE is: ", train['APE_ESM'].mean())
```

    Training Naive + ESM Model MAPE is:  1.4789918216232285


Wow! Our naive model had a training data set of about 3.5%, but this is down to nearly 1.5%! Our model seems to have improved. Let's check the test data set though and calculate a MAPE there.


```python
test['APE_ESM'] = abs((test['MW']-test['pred_ESM'])/test['MW'])*100
print("Naive + ESM Model MAPE is: ", test['APE_ESM'].mean())
```

    Naive + ESM Model MAPE is:  5.458113823008118


So we didn't see as much improvement in the test data set, but we still have some promise here based on the training data set improvement. 

Exponential smoothing models aren't the only time series models we could use. Instead of using ESM's we could try another class of time series model - the ARIMA model.

### ARIMA Model

#### Model Selection

There are many techniques to building ARIMA models. Classical approaches involve looking at correlation functions. More modern approaches use computer algorithms to build grids of models and compare. The nuances of these approaches are discussed in the resources provided. A brief outline is given here.

Looking at the correlation patterns of the data across time can reveal the best underlying model for the data. There are two correlation functions that we need to look at to get the full picture:
 1. Autocorrelation Function (ACF)
 2. Partial Autocorrelation Function (PACF)

Let's look at the ACF of our data with the ```plot_acf``` function.


```python
fig = tsaplots.plot_acf(train['resid'].diff(24)[25:], lags = 72)

plt.show()
```


    
![png](Part-4---Dynamic-Time-Series-Model_files/Part-4---Dynamic-Time-Series-Model_86_0.png)
    


From this plot we can see an exponentially decreasing pattern. This signals some potential for autoregressive (AR) terms in our model. We also see a random spike at 24. This signals a potential moving average (MA) term as well.

Now let's look at the PACF of the residuals with the ```plot_pacf``` function.


```python
fig = tsaplots.plot_pacf(train['resid'].diff(24)[25:], lags = 72)

plt.show()
```


    
![png](Part-4---Dynamic-Time-Series-Model_files/Part-4---Dynamic-Time-Series-Model_88_0.png)
    


We have a couple of spikes early on in this plot followed by a lot of nothing. Definitely an AR patterns with 2 as its order (p = 2 in ARIMA terminology). We also see an exponentially decreasing set of spikes every 24 hours. This coincides with the single spike at 24 from the ACF plot. Definitely a moving average (MA) term at that seasonal period (in ARIMA terminology this is Q = 1).

We also know that our data still has some seasonal effects every 24 hours so we should take a seasonal difference to account for this. 


```python
mod = SARIMAX(train['resid'], order=(2,0,0), seasonal_order=(0,1,1,24))
res = mod.fit()

print(res.summary())
```

    C:\Users\adlabarr\Anaconda3\lib\site-packages\statsmodels\tsa\base\tsa_model.py:218: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      ' ignored when e.g. forecasting.', ValueWarning)
    C:\Users\adlabarr\Anaconda3\lib\site-packages\statsmodels\tsa\base\tsa_model.py:218: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      ' ignored when e.g. forecasting.', ValueWarning)


                                          SARIMAX Results                                       
    ============================================================================================
    Dep. Variable:                                resid   No. Observations:                25536
    Model:             SARIMAX(2, 0, 0)x(0, 1, [1], 24)   Log Likelihood             -127287.455
    Date:                              Fri, 09 Oct 2020   AIC                         254582.910
    Time:                                      12:46:11   BIC                         254615.498
    Sample:                                           0   HQIC                        254593.447
                                                - 25536                                         
    Covariance Type:                                opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    ar.L1          0.8113      0.003    270.427      0.000       0.805       0.817
    ar.L2          0.0771      0.003     23.489      0.000       0.071       0.084
    ma.S.L24      -0.9379      0.002   -565.286      0.000      -0.941      -0.935
    sigma2      1259.8291      2.930    430.044      0.000    1254.087    1265.571
    ===================================================================================
    Ljung-Box (Q):                      435.61   Jarque-Bera (JB):            807459.76
    Prob(Q):                              0.00   Prob(JB):                         0.00
    Heteroskedasticity (H):               0.98   Skew:                             2.26
    Prob(H) (two-sided):                  0.44   Kurtosis:                        30.19
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).


Let's take a look at the results that we just got. It appears based on the p-values above that all of our terms are significant which is great. 

Let's forecast out the next 744 hours (our test data set) to see what it looks like. Again, we can use the ```forecast``` function to do this. Remember though, this is only a forecast of our residuals!


```python
forecast = pd.DataFrame(res.forecast(744))
forecast.index = test.index.copy()

ax1 = forecast.plot(color = 'blue', figsize=(9,7))

ax1.set_ylabel('Forecast')
ax1.set_xlabel('Date')

plt.show()
```

    C:\Users\adlabarr\Anaconda3\lib\site-packages\statsmodels\tsa\base\tsa_model.py:583: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
      ValueWarning)



    
![png](Part-4---Dynamic-Time-Series-Model_files/Part-4---Dynamic-Time-Series-Model_92_1.png)
    


Just with the ESM model, let's go ahead and save the predicted values and forecasts to our respective data frames. This will make it easier to see how well we did.


```python
train['fitted_resid2'] = res.predict()
test['pred_resid2'] = forecast
```

Now, let's add these ARIMA forecasts of our residuals to the previous forecasts we developed from our naive energy model to form our dynamic energy model using ARIMA techniques. 


```python
train['fitted_ARIMA'] = train['fitted'] + train['fitted_resid2']
test['pred_ARIMA'] = test['pred'] + test['pred_resid2']
```

Let's plot this forecast to see how well we did in the test data set.


```python
test['MW'].plot(color = 'blue', figsize=(9,7))

plt.ylabel('MW Hours')
plt.xlabel('Date')

test['pred_ARIMA'].plot(color = 'green', linestyle = 'dashed', figsize=(9,7))

plt.legend(loc="best");

plt.show()
```


    
![png](Part-4---Dynamic-Time-Series-Model_files/Part-4---Dynamic-Time-Series-Model_98_0.png)
    


It is visually a little hard to determine how well we did in comparison to the other models that we have developed. Let's calculate the MAPE for both our training and testing data sets.


```python
train['APE_ARIMA'] = abs((train['MW']-train['fitted_ARIMA'])/train['MW'])*100
print("Training Naive + ARIMA Model MAPE is: ", train['APE_ARIMA'].mean())
```

    Training Naive + ARIMA Model MAPE is:  1.3663869635441892


Wow! Our naive model had a training data set of about 3.5%, and ESM dynamic model had a MAPE of 1.5%, but this is down to nearly 1.4%! Our model seems to have improved. Let's check the test data set though and calculate a MAPE there.


```python
test['APE_ARIMA'] = abs((test['MW']-test['pred_ARIMA'])/test['MW'])*100
print("Naive + ARIMA Model MAPE is: ", test['APE_ARIMA'].mean())
```

    Naive + ARIMA Model MAPE is:  5.5553216961887655


Again, we didn't see as much improvement in the test data set, but we still have some promise here based on the training data set improvement. 

Feel free to play around with other seasonal ARIMA models to see if you can improve the forecasts! These techniques are memory intensive and time consuming however. Just be prepared for this as you build models. If you are running this in a colab environment, you might need to restart the kernel at each model build because of the memory and time consumption. Local installations might not have this problem. 

One potential improvement to modeling in time series is to ensemble (or average) multiple models' forecasts to make a better forecast. It doesn't always work, but always worth trying since it is rather easy. First, let's take the average of our two residual forecasts and add that to our naive model instead of just picking either the ESM or the ARIMA.


```python
train['fitted_Ensemble'] = train['fitted'] + 0.5*train['fitted_resid'] + 0.5*train['fitted_resid2']
test['pred_Ensemble'] = test['pred'] + 0.5*test['pred_resid'] + 0.5*test['pred_resid2']
```

Now let's check the MAPE of both the training and testing data sets.


```python
train['APE_Ensemble'] = abs((train['MW']-train['fitted_Ensemble'])/train['MW'])*100
print("Training Naive + Ensemble Model MAPE is: ", train['APE_Ensemble'].mean())
```

    Training Naive + Ensemble Model MAPE is:  1.3776474681630708



```python
test['APE_Ensemble'] = abs((test['MW']-test['pred_Ensemble'])/test['MW'])*100
print("Naive + Ensemble Model MAPE is: ", test['APE_Ensemble'].mean())
```

    Naive + Ensemble Model MAPE is:  5.4762931305278135


Looks like the ensemble didn't do too much to improve our forecasts. If that is the case, it might not be the analytical techniques as much as the variables that go into them. That is what we will be covering in the next milestone!

So many times forecasters will stop at simple regression techniques or only use time series approaches in isolation. The benefit can really be felt by merging the two together as you will do in this milestone. Gaining the benefit of the external variable relationships as well as the correlations across time can greater improve your forecasts and reduce your prediction errors. Now you can really display your analytical talent for your boss. If they were impressed with your last model, then this one should really help drive home the impact you are making in helping them getting more accurate forecasts to improve their business decisions!

#### OPTIONAL Additional Code in ARIMA

Python has some built in functions to try and select ARIMA models automatically. Unfortunately, they use grid search techniques to build many different ARIMA models which as mentioned earlier can be both time and memory intensive. For this reason, we are not going over this function in this course. However, feel free to play around with the code below and investigate more on your own!


```python
#!pip install scipy 
#!pip install pmdarima 

#from pmdarima import auto_arima

#mod_auto = auto_arima(train['resid'], start_p=0, start_q=0, max_p=3, max_q=3, 
                      #start_P=2, start_Q=0, max_P=2, max_Q=0, m=24, 
                      #seaonal=True, trace=True, d=0, D=1, error_action='warn', 
                      #stepwise=True)
```
