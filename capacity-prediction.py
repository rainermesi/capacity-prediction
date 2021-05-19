# libraries

import pandas as pd
#from io import BytesIO
from prophet import Prophet

raw_df = pd.read_csv('output.csv')

def strip_col(x):
    x['Capacity:'] = x['Capacity:'].str.rstrip('%')
    return x

def set_dtypes(x):
  dtypesDict = {'Venue:': 'category', 'Capacity:': 'int64', 'Timestamp:': 'datetime64'}
  return x.astype(dtypesDict)

def set_timezone(x):
    x['Timestamp:'] = x['Timestamp:'].dt.tz_localize('UTC').dt.tz_convert('Europe/Helsinki').dt.tz_localize(None)
    x['Timestamp:'] = x['Timestamp:'].dt.floor('h')
    return x

def run_prophet(x):
    z = x.rename(columns={'Venue:':'venue','Capacity:':'y','Timestamp:':'ds'})
    listOfVenues = z.venue.unique()
    df = z[z.venue == listOfVenues[1]]
    df = df[['ds','y']]
    m = Prophet().fit(df)
    future = m.make_future_dataframe(periods=4,freq='H')
    forecast = m.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

test_df = run_prophet(set_timezone(set_dtypes(strip_col(raw_df))))
test_df
