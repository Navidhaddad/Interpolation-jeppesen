import pandas as pd
import numpy as np

import json 
import scipy

def scipy_interpolation_linear(objective_name, data, interpolation_point):
    points, values = closest_points(objective_name, data, interpolation_point, 2500) 
                                                                   
    interpolated_val = scipy.interpolate.griddata(points, values, interpolation_point, method='linear', rescale='TRUE') 
    
    return interpolated_val

def scipy_interpolation_nearest(objective_name, data, interpolation_point):   
    points, values = closest_points(objective_name, data, interpolation_point, 16)  
                                                                   
    interpolated_val = scipy.interpolate.griddata(points, values, interpolation_point, method='nearest', rescale='TRUE')
       
    return interpolated_val    

def closest_points(objective_name, df, interpolation_point, n):
    temp_df = df.copy()
    temp_df['distance'] = temp_df.sub(point).pow(2).sum(1).pow(0.5) # euclidean distance
    
    df_sort_by_dist = temp_df.sort_values('distance').iloc[0:nbr_neighbours]
    df_sort_by_dist = df_sort_by_dist.drop(['distance'],axis = 1)
    df_sort_by_dist = df_sort_by_dist.reset_index()
    df_sort_by_dist = df_sort_by_dist.drop(['index'], axis=1)
    
    val = df_sort_by_dist[objective_name]
    poi = df_sort_by_dist.drop([objective_name], axis=1)
    
    return val, poi
