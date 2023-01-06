import scipy
import numpy as np
import sklearn
import pandas as pd

def custom_distance(x,y,w):
    cat1_x=x[0:8]
    cat2_x=x[8:29+8]
    cat3_x=x[29+8:]
    cat1_y=y[0:8]
    cat2_y=y[8:29+8]
    cat3_y=y[29+8:]
    cat1_d=w[0]*scipy.spatial.distance.minkowski(cat1_x, cat1_y, p=2, w=None)*(1/np.sqrt(2))
    cat2_d=w[1]*scipy.spatial.distance.minkowski(cat2_x, cat2_y, p=2, w=None)*(1/np.sqrt(2))
    cat3_d=scipy.spatial.distance.minkowski(cat3_x, cat3_y, p=2, w=None)
    return cat1_d+cat2_d+cat3_d+w[2]

def find_bis_year(years):
    bis_year=[]
    for year in np.unique(years):
        cond=False
        for k in range(0,3):
            cond=(pd.to_datetime('{}-12-21'.format(year))+pd.DateOffset(weeks=k)).isocalendar().week==53 or cond
        if cond==True:
            bis_year.append(year)
    return bis_year

def map_category_to_vector(all_cat_vector,category):
    """
    This functions performs One-Hot Encoding, that is to say it takes a category C as input and 
    returns a vector filled with 0 except at the spot where the category C is in the vector all_cat_vector.
    ex : all_cat_vector=[CAT1,CAT2,CAT3,CAT4] and category = CAT 3, the output will be [0,0,1,0].
    Args:
        all_cat_vector(np.array or pd.Series): The vector containing all the category that will be used
        for mapping.
        category: An element of all_cat_vector to be mapped.
    """
    x=np.zeros_like(all_cat_vector)
    i=np.where(all_cat_vector==category)
    x[i]=1
    return x

def norm(x,max,min):
    if max==min:
        return(x-min)
    else:
        return(x-min)/(max-min)

def inversenorm(x,max,min):
    if max==min:
        return(x+min)
    else:
        return(x*(max-min)+min)

def inverse_diff(first,diff):
    return np.concatenate(([first], diff)).cumsum()

def correlation(a,y,y_t): 
    return sklearn.metrics.mean_absolute_error((a*y).reshape(1,-1),y_t.reshape(1,-1))