import numpy as np 
from random import random
import pandas as pd
def generate_synthetic_dataset(sections=1):
    rotation_covariation_parameter = np.random.normal(loc=0, scale=100, size=sections)
    X_scale_covariation = np.random.normal(loc=0, scale=200, size=sections)
    Y_scale_covariation = np.random.normal(loc=0, scale=150, size=sections)
    columns = ['ox', 'oy', 'oz', 'ux', 'uy', 'uz', 'vx', 'vy', 'vz']
    Filenames = ['Synthetic__'+str(i) for i in np.arange(sections)]
    ##ox is origin x position
    ox = np.random.normal(loc = 454, scale = 25, size=sections)
    ##oy is origin z depth
    oy = np.random.uniform(low=0, high = 550, size=sections)
    ##oz is origin y position
    oz = np.random.normal(loc = 315, scale = 25, size=sections)
    ##ux is the x distance of the top right corner from the origin
    ux = np.random.normal(loc = -500, scale = 25, size=sections)
    ##uy is the z distance of the top right corner from the origin
    uy = np.random.normal(loc = 0, scale = 0.11, size=sections)*ux
    ##This parameter controls the level of "parrallelogramness"
    ##uz is the y distance of the top right corner from the origin
    uz = np.random.normal(loc = 0, scale = 25, size=sections)
    ##This parameter controls the level of "parrallelogramness"
    ##vx is the x distance of the bottom left corner from the origin
    vx = np.random.normal(loc = 0, scale = 25, size=sections)
    ##vz is the y distance of the bottom left corner from the origin
    vz = np.random.normal(loc = -330, scale = 25, size=sections)
    ##vy is the z distance of the bottom left corner from the origin
    vy = np.random.normal(loc = 0, scale = 0.11, size=sections)*vz
    ox+=X_scale_covariation
    ux-=X_scale_covariation
    oz+=Y_scale_covariation
    vz-=Y_scale_covariation
    uz+=rotation_covariation_parameter
    vx-=rotation_covariation_parameter
    df = pd.DataFrame({'Filenames':Filenames, 'ox':ox, 'oy': oy, 'oz': oz, 'ux':ux, 'uy':uy, 'uz':uz, 'vx':vx, 'vy':vy, 'vz':vz})
    return df
   
   
   