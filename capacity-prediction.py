# libraries

import pandas as pd
#from io import BytesIO
from fbprophet import Prophet

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

def prep_prophet(x):
    z = x.rename(columns={'Venue:':'venue','Capacity:':'y','Timestamp:':'ds'})
    listOfVenues = z.venue.unique()
    dictOfDataframes = {elem : pd.DataFrame for elem in listOfVenues}
    for key in dictOfDataframes.keys():
        dictOfDataframes[key] = z[:][z.venue == key]
    #z.set_index(keys=['venue'], drop=True,inplace=True)
    return dictOfDataframes

test_df = prep_prophet(set_timezone(set_dtypes(strip_col(raw_df))))
test_df
