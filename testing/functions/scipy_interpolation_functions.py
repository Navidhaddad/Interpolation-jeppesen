import pandas as pd
import numpy as np

import json 
import scipy

def scipy_interpolation_linear (data, interpolation_point, parameter):
    
    if (parameter == 'FUELFLOW' ) :
        
        y = data.FUELFLOW
        X = data.drop(['FUELFLOW'], axis=1)
        
    elif (parameter == 'DRAG'):
        
        y = data.DRAG
        X = data.drop(['DRAG'], axis=1)
   
    
    points , values = closest_points(X,y,interpolation_point,2500,parameter)  
                                                               
    
    interpolated_val = scipy.interpolate.griddata(points, values, interpolation_point, method='linear', rescale='TRUE')
    

    
    return interpolated_val

def scipy_interpolation_nearest (data, interpolation_point,parameter):
    
    if (parameter == 'FUELFLOW'):
    
        y = data.FUELFLOW
        X = data.drop(['FUELFLOW'], axis=1)
        
    elif (parameter == 'DRAG'):
        
        y = data.DRAG
        X = data.drop(['DRAG'], axis=1)
   
    
    points , values = closest_points(X,y,interpolation_point,2500,parameter)  
                                                               
    
    interpolated_val = scipy.interpolate.griddata(points, values, interpolation_point, method='nearest', rescale='TRUE')
    

    
    return interpolated_val
    

def closest_points(X,y,interpolation_point, n, parameter):
    X['distance'] = X.sub(interpolation_point).pow(2).sum(1).pow(0.5) #calculate euclidean distance
    X = X.sort_values(['distance']).iloc[0:n] #sort
    X = X.drop(['distance'],axis=1)
    X = X.reset_index()
    
    X.columns = ['index_', 'DISA', 'ALTITUDE', 'MASS','MACH'] #rename columns
    
    if (parameter == 'FUELFLOW'):
        y.columns = ['index_', 'FUELFLOW']
        X = X.join(y, on = 'index_', how='left',) #join 20 nearest point with 
                                                                    #corresonding fuel flow
        val = X['FUELFLOW']
        poi = X.drop(['index_','FUELFLOW'], axis=1)
     
    
    elif (parameter == 'DRAG'):
        
        y.columns = ['index_', 'DRAG']
        X = X.join(y, on = 'index_', how='left',) #join 20 nearest point with 
                                                                    #corresonding fuel flow
        val = X['DRAG']
        poi = X.drop(['index_','DRAG'], axis=1)
     
    
    return poi , val
