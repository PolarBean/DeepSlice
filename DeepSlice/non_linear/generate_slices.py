import pandas as pd
import numpy as np
import os
from .NBB.models import vgg19_model
from .NBB.algorithms import neural_best_buddies as NBBs
from .NBB.util import util, MLS
import nibabel as nib
import math
from ..read_and_write.QuickNII_functions import write_QUINT_JSON



def generate_target_slice(alignment, volume):
    Ox,Oy,Oz,Ux,Uy,Uz,Vx,Vy,Vz = alignment
    ##just for mouse for now
    bounds = [455, 527, 319]
    X_size = np.sqrt(np.sum(np.square((Ux,Uy,Uz))))
    Z_size = np.sqrt(np.sum(np.square((Vx,Vy,Vz))))
    print(X_size, Z_size)
    X_size =  np.round(X_size).astype(int)
    Z_size =  np.round(Z_size).astype(int)

    #arange from 0 to 456 matching CCF U dimension (mediolateral)
    U_increment = np.arange(Ox, Ox+Ux, -1)
    #arange from 0 to 320 matching CCF V dimension (dorsoventral)
    V_increment = np.arange(Oz, Oz+Vz, -1)
    #make this into a grid (0,0) to (320,456)
    Uarange = np.arange(0,1,1/X_size)
    Varange = np.arange(0,1,1/Z_size)
    Ugrid, Vgrid = np.meshgrid(Uarange, Varange)
    Ugrid_x = Ugrid * Ux
    Ugrid_y = Ugrid * Uy
    Ugrid_z = Ugrid * Uz
    Vgrid_x = Vgrid * Vx
    Vgrid_y = Vgrid * Vy
    Vgrid_z = Vgrid * Vz

    X_Coords = (Ugrid_x + Vgrid_x).flatten() + Ox
    Y_Coords = (Ugrid_y + Vgrid_y).flatten() + Oy
    Z_Coords = (Ugrid_z + Vgrid_z).flatten() + Oz

    X_Coords = np.round(X_Coords).astype(int)
    Y_Coords = np.round(Y_Coords).astype(int)
    Z_Coords = np.round(Z_Coords).astype(int)


    out_bounds_Coords = (X_Coords>bounds[0]) | (Y_Coords>bounds[1]) | (Z_Coords>bounds[2])
    X_pad = X_Coords.copy()
    Y_pad = Y_Coords.copy()
    Z_pad = Z_Coords.copy()

    X_pad[out_bounds_Coords] = 0
    Y_pad[out_bounds_Coords] = 0
    Z_pad[out_bounds_Coords] = 0


    regions = volume[X_pad, Y_pad, Z_pad]
    ##this is a quick hack to solve rounding errors
    C = len(regions)
    compare = (C - X_size*Z_size)
    if abs(compare) == X_size:
        if compare>0:
            Z_size+=1
        if compare<0:
            Z_size-=1
    elif abs(C - X_size*Z_size) == Z_size:
        if compare>0:
            X_size+=1
        if compare<0:
            X_size-=1
    regions = regions.reshape((abs(Z_size), abs(X_size)))
    return regions


def neural_best_buddies(imageA, alignment,volume, model):

    nbbs = NBBs.sparse_semantic_correspondence(model, -1, 0.05, 7, '', 15, 10, True)
    A = util.read_image(imageA, 224)
    Barray = generate_target_slice(alignment, volume)
    B3Chan = np.repeat(Barray[:,:,np.newaxis],3, axis=2)
    B = util.numpy_to_image(B3Chan, width = 224)
    points = nbbs.run(A, B)
    return points

def neural_best_buddies_full_brain(df,path, volume_path = r"/home/harry/Github/DeepSlice-nesys/DeepSlice/metadata/volumes/ara_nissl_25.nii"):
    volume = nib.load(volume_path)
    volume = np.array(volume.get_fdata())
    model = vgg19_model.define_Vgg19()
    nbbs = NBBs.sparse_semantic_correspondence(model, -1, 0.05, 7, '', inf, 25, True)
    df["markers"] = [[]] * len(df)
    df['markers'] = df['markers'].astype(object)

    for row in df.iterrows():
            prediction = row[1]
            image_path = path + prediction.Filenames
            A = util.read_image(image_path, 224)
            alignment = prediction[['ox', 'oy', 'oz', 'ux', 'uy', 'uz', 'vx', 'vy', 'vz']]
            imageB = generate_target_slice(alignment, volume)
            imageBLoad = np.repeat(imageB[:,:,np.newaxis],3, axis=2)
            B = util.numpy_to_image(imageBLoad, width = 224)
            points = nbbs.run(A, B)
            height, width = prediction[['height', 'width']].values
            points = convert_points_to_visualign(points,height,width)
            df.loc[row[0],'markers'] = [points]
            write_QUINT_JSON(df, path + '__temp__', 'dev', target = "ABA_Mouse_CCFv3_2017_25um.cutlas")
    return df

    
def convert_points_to_visualign(points, target_height, target_width):
    predsize = 224
    filea_lines,fileb_lines = points
    filea_lines_scaled = [[i/predsize for i in line] for line in filea_lines]
    fileb_lines_scaled = [[i/predsize for i in line] for line in fileb_lines]

    filea_lines_scaled_rotated = [rotate((0.5,0.5),line, np.deg2rad(90))  for line in filea_lines_scaled]
    fileb_lines_scaled_rotated = [rotate((0.5,0.5),line, np.deg2rad(90))  for line in fileb_lines_scaled]

    filea_lines_scaled_mirrored = [[1-line[0], line[1]] for line in filea_lines_scaled_rotated]
    fileb_lines_scaled_mirrored = [[1-line[0], line[1]] for line in fileb_lines_scaled_rotated]

    filea_lines_target = [[line[0] * target_width, line[1] * target_height] for line in filea_lines_scaled_mirrored]
    fileb_lines_target = [[line[0] * target_width, line[1] * target_height] for line in fileb_lines_scaled_mirrored]
    markers = [[*i, *j] for i,j in zip(fileb_lines_target, filea_lines_target)]
    return markers
    
def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

