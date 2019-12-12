import pandas as pd
import numpy as np

import json 

from sklearn.model_selection import train_test_split
import scipy

def scipy_interpolation (data, interpolation_point):
    
    y = data.FUELFLOW
    X = data.drop(['FUELFLOW'], axis=1)
   
    
    points , values = closest_points(X,y,interpolation_point,4000)  
                                                               
    
    interpolated_val = scipy.interpolate.griddata(points, values, interpolation_point, method='linear', rescale='TRUE')
    

    
    return interpolated_val
    

def closest_points(X,y,interpolation_point, n):
    X['distance'] = X.sub(interpolation_point).pow(2).sum(1).pow(0.5) #calculate euclidean distance
    X = X.sort_values(['distance']).iloc[0:2500] #sort
    X = X.drop(['distance'],axis=1)
    X = X.reset_index() 
    X.columns = ['index_', 'DISA', 'ALTITUDE', 'MASS','MACH'] #rename columns
    
    y.columns = ['index_', 'FUELFLOW']
    X = X.join(y, on = 'index_', how='left',) #join 20 nearest point with 
                                                                #corresonding fuel flow
    val = X['FUELFLOW']
    poi = X.drop(['index_','FUELFLOW'], axis=1)
    
    return poi , val