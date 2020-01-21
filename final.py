import pandas as pd
import numpy as np

import codecs, json
import scipy

import scipy_interpolation_functions as scipy_int

def read_poor_data(filename):
  df = pd.DataFrame([])
  for i in range(7):
      with open('/Users/calmaleh/Desktop/school/project_course/jeppesen/ac_poor_' + str(i+1) +'.bsad') as json_file:
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

      temp_df = temp_df[['DISA','ALTITUDE','MASS','MACH_MODE','FUELFLOW']]

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
    df['TAS'] = round((df['MACH_MODE'] * df['SPEED_SOUND']), 1)

    return df.drop(['ISA_CONDITION', 'ISA','SPEED_SOUND', 'MACH_MODE'],axis=1)

def read_rich_data(filename):
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
  df = df[['DISA','ALTITUDE','MASS','MACH','FUELFLOW']]

  scaler = MinMaxScaler()
  df = scaler.fit_transform(df)

  df = pd.DataFrame(df, columns = ['DISA','ALTITUDE','MASS','MACH','FUELFLOW'])

def pandas_interpolation(df):





