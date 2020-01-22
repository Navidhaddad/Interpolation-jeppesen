import pandas as pd
import numpy as np

import codecs, json
import scipy
from sklearn.preprocessing import MinMaxScaler

import scipy_interpolation_functions as scipy_int

def read_poor_data(filenames, objective):
  df = pd.DataFrame([])
  for filename in filenames:
      with open(filename) as json_file:
          json_data = json.load(json_file)

      frames = []
      for j in range(len(json_data['tables'])):
          temp_df = pd.DataFrame(np.array(json_data['tables'][j]['table'])[:,:],
                                 columns = json_data['tables'][j]['header']['variables'])
          temp_df['MACH_MODE'] = json_data['tables'][j]['header']['mode']
          temp_df['STATE'] = json_data['tables'][j]['header']['flightphase']
          if temp_df['STATE'][0] == 'cruise':
              frames.append(temp_df)

      temp_df = pd.concat(frames,ignore_index=True)

      temp_df = temp_df[temp_df['MACH_MODE'].str.contains("Mach")]
      temp_df['MACH_MODE'] = temp_df['MACH_MODE'].map(lambda x: x.lstrip('Mach ')).astype(float)

      temp_df = temp_df[['DISA','ALTITUDE','MASS','MACH_MODE', objective]]

  return pd.concat([df,temp_df])

def speed_converter_poor_data(df):
  TROPOPAUSE_ALT = 11000.0
  STRATOSPHERE_ALT = 20000

  TROPOPAUSE_TEMP = 216.65
  STD_TEMP = 288.15

  STD_LAPSE_RATE = -0.0065

  SPEED_OF_SOUND = 340.294

  df['ISA_CONDITION'] = False
  df.loc[(0 <= df['ALTITUDE']) & (df['ALTITUDE'] <= TROPOPAUSE_ALT), 'ISA_CONDITION'] = True

  df['ISA'] = 0

  df.loc[df.ISA_CONDITION == False, 'ISA'] = TROPOPAUSE_TEMP
  df.loc[df.ISA_CONDITION == True, 'ISA'] = round(STD_TEMP + STD_LAPSE_RATE * df[df.ISA_CONDITION == True].ALTITUDE,2)
  df['SPEED_SOUND'] = SPEED_OF_SOUND * np.sqrt((df['ISA'] + df['DISA'])/ STD_TEMP)

  #df['MACH'] = round(df['MACH_'] / df['SPEED_SOUND'],4)
  df['SPEED'] = round((df['MACH_MODE'] * df['SPEED_SOUND']), 1) # SPEED here is TAS
  df = df.drop(['ISA_CONDITION', 'ISA','SPEED_SOUND', 'MACH_MODE'],axis=1)

  return df

def read_rich_data(filename, objective):
  with open(filename) as json_file:
      json_data = json.load(json_file)

  frames = []
  for j in range(len(json_data['tables'])):
      df = pd.DataFrame(np.array(json_data['tables'][j]['table'])[:,:],
                             columns = json_data['tables'][j]['header']['variables'][:])
      df['state'] = json_data['tables'][j]['header']['flightphase']
      if df['state'][0] == 'cruise':
          frames.append(df)

  df = pd.concat(frames,ignore_index=True)
  df = df[['DISA','ALTITUDE','MASS','MACH',objective]]
  df = df.rename(columns ={'MACH':'SPEED'})

  scaler = MinMaxScaler()
  df = scaler.fit_transform(df)

  df = pd.DataFrame(df, columns = ['DISA','ALTITUDE','MASS','SPEED', objective])
  
  return df

def pandas_interpolation(df, objective, temperature, altitude, mass, speed):
  point = {'DISA': temperature, 'ALTITUDE': altitude, 'MASS': mass, 'SPEED': speed}
  
  def closest_neighbours(df, point, nbr_neighbours):
    temp_df = df.copy()
    temp_df['distance'] = X_train_1.sub(point).pow(2).sum(1).pow(0.5) # euclidean distance
    
    df_sort_by_dist = temp_df.sort_values('distance').iloc[0:nbr_neighbours]
    df_sort_by_dist = df_sort_by_dist.drop(['distance'],axis = 1)
    df_sort_by_dist = df_sort_by_dist.reset_index()
    df_sort_by_dist = df_sort_by_dist.drop(['index'], axis=1)
    
    return df_sort_by_dist
  
  df = closest_neighbours(df, point, 16)
  
  df_interpolated = df[0:1].append(point)
  df_interpolated = df_interpolated.append(df[2:16])
  
  df_interpolated = df_interpolated.interpolate(method = 'linear')
  point_interpolated = df_interpolated[1:2]
  
  return point_interpolated  

def scipy_linear_interpolation(objective_name, df, temperature, altitude, mass, speed): 
  point = {'DISA': temperature, 'ALTITUDE': altitude, 'MASS': mass, 'SPEED': speed}

  objective_interpolated = scipy_int.scipy_interpolation_linear(df, point)

  # if point is not interpolated, it is null
  # counter by copying closest value
  if np.isnan(objective_interpolated)== True : 
      y = df_interpolated.objective_name
      x = df_interpolated.drop([objective_name], axis=1)
      p, v = scipy_int.closest_points(x,y,point,1) # returns closest point p and its value v
      objective_interpolated = v
  
  point[objective_name] = objective_interpolated
  
  return point

def scipy_nearest_interpolation(df, temperature, altitude, mass, speed):
  point = {'DISA': temperature, 'ALTITUDE': altitude, 'MASS': mass, 'SPEED': speed}
  
  objective_interpolated = scipy_int.scipy_interpolation_nearest(df, point)
  
  point[objective] = objective_interpolated
  
  return point


  
  
