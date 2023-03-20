import numpy as np
from .plane_alignment_functions import plane_alignment


def calculate_brain_center_depth(section):
    """
    Calculates the depth of the brain center for a given section

    :param section: the section coordinates as an array consisting of Oxyz,Uxyz,Vxyz 
    :type section: np.array
    :return: the depth of the brain center
    :rtype: float
    """
    cross, k = plane_alignment.find_plane_equation(section)
    translated_volume = np.array((456, 0, 320))
    linear_point = (
        ((translated_volume[0] / 2) * cross[0])
        + ((translated_volume[2] / 2) * cross[2])
    ) + k
    depth = -(linear_point / cross[1])
    return depth


def calculate_brain_center_depths(predictions):
    """
    Calculates the depths of the brain center for a series of predictions
    
    :param predictions: dataframe of predictions
    :type predictions: pandas.DataFrame
    :return: a list of depths
    :rtype: list[float]
    """
    depths = []
    for prediction in predictions[
        ["ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz"]
    ].values:
        depths.append(calculate_brain_center_depth(prediction))
    return depths
