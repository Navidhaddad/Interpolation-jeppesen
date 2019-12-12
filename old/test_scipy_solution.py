import pandas as pd
import numpy as np

import json 

from sklearn.model_selection import train_test_split
from scipy_interpolation_linear import scipy_interpolation
from scipy_interpolation_linear import closest_points
import scipy



#get data
with open('data_rich_ac.bsad') as json_file:
    json_data = json.load(json_file)
    
frames = []
for j in range(len(json_data['tables'])):
    df = pd.DataFrame(np.array(json_data['tables'][j]['table'])[:,:], 
                           columns = json_data['tables'][j]['header']['variables'][:])
    df['state'] = json_data['tables'][j]['header']['flightphase']
    if df['state'][0] == 'cruise':
        frames.append(df)
    
df = pd.concat(frames,ignore_index=True)
df = df[['DISA','ALTITUDE','MASS','MACH','FUELFLOW']]

temp = df.drop(['FUELFLOW'], axis=1)
val =[]
dif= []

for i in range(100) :
    print(i)
    #choose point to interpolate
    interpol_point = temp.iloc[i]
    
    deleted_df = df.drop([i],axis=0)
    
    #call function
    value = scipy_interpolation(deleted_df,interpol_point)
    difference = abs(value - df['FUELFLOW'].iloc[i])
    val.append(value)
    dif.append(difference)
    
    
print(np.mean(dif))