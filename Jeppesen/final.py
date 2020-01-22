import pandas as pd
import numpy as np

import codecs, json
import scipy
from sklearn.preprocessing import MinMaxScaler

import scipy_interpolation_functions as scipy_int

# read in "poor" data based on list of filenames and objective of interest (for interpolation)
def read_poor_data(filenames, objective):
  df = pd.DataFrame([]) # create data frame
  # read in as json objects
  for filename in filenames: 
      with open(filename) as json_file:
          json_data = json.load(json_file)
      
      # create data frame with columns consisting of temperature, altitude,
      # mass, speed (mach modes), and objective (fuel flow, for instance)
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
      
      df = pd.concat([df,temp_df])
      
      # normalize data
      scaler = MinMaxScaler()
      df = scaler.fit_transform(df)
      
  return df

# takes data frame df and returns same data frame with SPEED column (MACH modes converted to TAS)
def speed_converter(df):
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

  df['SPEED'] = round((df['MACH_MODE'] * df['SPEED_SOUND']), 1) # SPEED here is TAS
  df = df.drop(['ISA_CONDITION', 'ISA','SPEED_SOUND', 'MACH_MODE'],axis=1)

  return df

# reads in "rich" data using a filename and objective as input 
# (objective being drag or fuel flow, for instance)
def read_rich_data(filename, objective):
  # create json object
  with open(filename) as json_file:
      json_data = json.load(json_file)
  
  # create data frame with columns consisting of temperature, altitude,
  # mass, speed (mach), and objective
  frames = []
  for j in range(len(json_data['tables'])):
      df = pd.DataFrame(np.array(json_data['tables'][j]['table'])[:,:],
                             columns = json_data['tables'][j]['header']['variables'][:])
      df['state'] = json_data['tables'][j]['header']['flightphase']
      if df['state'][0] == 'cruise':
          frames.append(df)

  df = pd.concat(frames,ignore_index=True)
  df = df[['DISA','ALTITUDE','MASS','MACH',objective]]
  df = df.rename(columns ={'MACH':'SPEED'}) # we rename MACH to SPEED to generalize 
                                            # the variable for interpolation methods
  
  # normalize data
  scaler = MinMaxScaler()
  df = scaler.fit_transform(df)
  
  # keep necessary columns
  df = pd.DataFrame(df, columns = ['DISA','ALTITUDE','MASS','SPEED', objective])
  
  return df

# pandas interpolation method:
# inputs are a given data frame, objective name,
# values for temperature, altitude, mass and speed (MACH, TAS, for instance)
def pandas_interpolation(objective_name, df, temperature, altitude, mass, speed):
  # create point consisting of the temperature, altitude, mass and speed values
  point_data = {'DISA': temperature, 'ALTITUDE': altitude, 'MASS': mass, 'SPEED': speed}
  point = pd.DataFrame(data=point_data)
  
  # function for calculating closest neighbours of a data frame, df,
  # to a given point. the number of neighbours returned is set by nbr_neighbours parameter
  def closest_neighbours(df, point, nbr_neighbours):
    temp_df = df.copy()
    temp_df['distance'] = temp_df.sub(point).pow(2).sum(1).pow(0.5) # euclidean distance
    
    df_sort_by_dist = temp_df.sort_values('distance').iloc[0:nbr_neighbours]
    df_sort_by_dist = df_sort_by_dist.drop(['distance'],axis = 1)
    df_sort_by_dist = df_sort_by_dist.reset_index()
    df_sort_by_dist = df_sort_by_dist.drop(['index'], axis=1)
    
    return df_sort_by_dist
  
  
  df = closest_neighbours(df, point, 16)
  
  # add point in question between closest and second closest points
  df_interpolated = df[0:1].append(point)
  df_interpolated = df_interpolated.append(df[2:16])
  
  # perform interpolation
  df_interpolated = df_interpolated.interpolate(method = 'linear')
  point_interpolated = df_interpolated[1:2]
  
  return point_interpolated  

# scipy linear interpolation method:
# inputs are a given data frame, objective name,
# values for temperature, altitude, mass and speed (MACH, TAS, for instance)
def scipy_linear_interpolation(objective_name, df, temperature, altitude, mass, speed):
  # create point consisting of the temperature, altitude, mass and speed values
  point_data = {'DISA': temperature, 'ALTITUDE': altitude, 'MASS': mass, 'SPEED': speed}
  point = pd.DataFrame(data=point_data)
  
  # return interpolated objective value
  objective_interpolated = scipy_int.scipy_interpolation_linear(objective_name, df, point)

  # if point is not interpolated, it is null
  # counter by copying closest value
  if np.isnan(objective_interpolated)== True : 
      p = scipy_int.closest_points(objective_name, df_interpolated, point, 1) # returns closest point p
      objective_interpolated = p[objective_name]
  
  # attach objective value to point
  point[objective_name] = objective_interpolated
  
  return point

# scipy nearest neighbour interpolation method:
# inputs are a given data frame, objective name,
# values for temperature, altitude, mass and speed (MACH, TAS, for instance)
def scipy_nearest_interpolation(objective_name, df, temperature, altitude, mass, speed):
  point_data = {'DISA': temperature, 'ALTITUDE': altitude, 'MASS': mass, 'SPEED': speed}
  point = pd.DataFrame(data=point_data)

  # return interpolated objective value  
  objective_interpolated = scipy_int.scipy_interpolation_nearest(objective_name, df, point)
  
  # attach objective value to point  
  point[objective] = objective_interpolated
  
  return point

# interpolate intervals for speed limits given data frame and
# a preferred scipy interpolation method ('nearest' or 'linear')
def scipy_speed_limit_interpolation(df, interpolation_method):
  # include only necessary variables
  df = df[['DISA', 'ALTITUDE', 'MASS', 'SPEED']]
  
  # use groupby to obtain min and max speed values
  group_min = df.groupby(['DISA', 'ALTITUDE', 'MASS']).min()
  group_max = df.groupby(['DISA', 'ALTITUDE', 'MASS']).max()
  df_group = pd.DataFrame(index=range(len(group_min)),columns=range(5))
  
  # set minimal and maximum speeds
  nbr_points = len(df)
  for i in range(nbr_points):
      df_group.iloc[i] = [group_min.index[i][0],
                          group_min.index[i][1],
                          group_min.index[i][2],
                          group_min.iloc[i][0],
                          group_max.iloc[i][0]]

  df_group.columns = ['DISA', 'ALTITUDE', 'MASS', 'SPEED_MIN', 'SPEED_MAX']
  
  df_group_min = df_group.drop(['SPEED_MIN'], axis=1)
  df_group_max = df_group.drop(['SPEED_MAX'], axis=1)

  y_min = df_group.SPEED_MIN
  y_max = df_group.SPEED_MAX
  
  # perform interpolation using scipy's functions with given interpolation method
  interpolated_val_min = scipy.interpolate.griddata(df_group_min, y_min, "X-values for new point", method=interpolation_method)
  interpolated_val_max = scipy.interpolate.griddata(df_group_max, y_max, "X-values for new point", method=interpolation_method)
