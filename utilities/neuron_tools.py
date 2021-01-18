import numpy as np
from utilities import QuickNII_functions 
import glob
import pandas as pd
def points_to_3d(points, plane):
    resolution = points[['width', 'height']].values.astype(np.float64)[0]  
    points = points[['X', 'Y']].values.astype(np.float64)
    ## Scale points by the resolution (numbers will now be between 0 and 1
    points/=resolution
    O_plane = plane[['ox','oy','oz']].values.astype(np.float64)
    X_plane = plane[['ux','uy','uz']].values.astype(np.float64)
    Y_plane = plane[['vx','vy','vz']].values.astype(np.float64)
    ## Since we have the vectors U&V and the fractional cell coordinates
    ## We can convert these onto the points on the plane
    ## Should probably be vectorised
    X_dim   = [X_plane * points[i, 0] for i in range(len(points))]
    Y_dim   = [Y_plane * points[i, 1] for i in range(len(points))]
    points  = np.sum((X_dim,Y_dim), axis=0)
    ##Now we add the origin to these coordinates to transform them into our alignment space
    points += O_plane
    points  = points.reshape((len(points), 3))
    return points

def analyse_slice(point_name, plane):
    image_name = point_name.split('/')[-1][:-4]
    raw_names  = [name[0].split('/')[-1] for name in plane["Filenames"].str.split('.')]
    index      = [i==image_name for i in raw_names]
    plane      = plane[index]
    points     = pd.read_csv(point_name)
    points     = points_to_3d(points, plane)
    return points
    
def analyse_brain(plane_file, points_folder = 'points/', save_csv=False):
    plane =  QuickNII_functions.XML_to_csv(plane_file)
    cells = np.empty((0,3))
    for points in glob.glob(points_folder+'/*.csv'):
        cells = np.vstack((cells, analyse_slice(points, plane)))
    cells = pd.DataFrame(cells, columns = ['X', 'Y', 'Z'])
    cells.to_hdf(plane_file[:-4]+'Whole_Brain_Cell_Count.h5')
    if save_csv:
        cells.to_csv(plane_file[:-4]+'Whole_Brain_Cell_Count.csv')


