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
    z = x[x['Timestamp:'] > '2021-05-01']
    return z

def run_prophet(x):
    m = Prophet().fit(x)
    future = m.make_future_dataframe(periods=24,freq='H')
    forecast = m.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(24)

def loop_df(x):
    z = x.rename(columns={'Venue:':'venue','Capacity:':'y','Timestamp:':'ds'})
    listOfVenues = z.venue.unique()
    forecast_df = pd.DataFrame(columns=['ds', 'yhat', 'yhat_lower', 'yhat_upper','origin_ds','venue'])
    for i in listOfVenues:
        df = z[z['venue'] == i]
        df = df[['ds','y']]
        temp_forecast_df = run_prophet(df)
        temp_forecast_df['venue'] = i
        temp_forecast_df['origin_ds'] = df['ds'].iloc[-1]
        forecast_df = forecast_df.append(temp_forecast_df,ignore_index=True)
    return forecast_df

test_df = loop_df(set_timezone(set_dtypes(strip_col(raw_df))))

test_df


