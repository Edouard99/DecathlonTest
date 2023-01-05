import scipy
import numpy as np
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