import pandas as pd
import numpy as np

import scipy


# interpolation function using scipy's linear interpolation method:
# parameters: name of objective (for instance 'FUELFLOW'), data frame, point of interest (for interpolation)
def scipy_interpolation_linear(objective_name, data, interpolation_point):
    points, values = closest_points(objective_name, data, interpolation_point, 2500) 
                                                                   
    interpolated_val = scipy.interpolate.griddata(points, values, interpolation_point, method='linear', rescale='TRUE') 
    
    return interpolated_val

# interpolation function using scipy's linear interpolation method:
# parameters: name of objective (for instance 'FUELFLOW'), data frame, point of interest (for interpolation)
def scipy_interpolation_nearest(objective_name, data, interpolation_point):   
    points, values = closest_points(objective_name, data, interpolation_point, 16)  
                                                                   
    interpolated_val = scipy.interpolate.griddata(points, values, interpolation_point, method=interpolation_point, rescale='TRUE')
       
    return interpolated_val    

# function for calculating closest neighbours for a given point using euclidean distance
# parameters: name of objective (for instance 'FUELFLOW'), data frame with neighbours, point of interest, 
#             number of neighbours returned
def closest_points(objective_name, df, interpolation_point, n):
    temp_df = df.copy()
    temp_df['distance'] = temp_df.sub(point).pow(2).sum(1).pow(0.5) # euclidean distance
    
    # sort values from closest to farthest, including nbr_neighbours amount of points
    df_sort_by_dist = temp_df.sort_values('distance').iloc[0:nbr_neighbours] 
    df_sort_by_dist = df_sort_by_dist.drop(['distance'],axis = 1)
    df_sort_by_dist = df_sort_by_dist.reset_index()
    df_sort_by_dist = df_sort_by_dist.drop(['index'], axis=1)
    
    val = df_sort_by_dist[objective_name]
    poi = df_sort_by_dist.drop([objective_name], axis=1)
    
    return val, poi
