import numpy as np
from .plane_alignment_functions import plane_alignment
from .depth_estimation import calculate_brain_center_depths


def calculate_brain_center_coordinate(section, atlas_shape, axis):
    """
    Calculates the coordinate closest to the middle of the section as defined by the two 
    dimensions not orthogonal to the dimension along which the series is being aligned.
    for example, if the series is being aligned coronally then this will return the midpoint
    in X and Z as coronal series are spaced along the Y coordinate in the CCF.

    :param section: The section to calculate the center for
    :param atlas_shape: The shape of the atlas
    :param axis: The axis along which the series is being aligned ('sagittal', 'coronal', 'horizontal')
    :return: The coordinate closest to the middle of the section
    """
    if axis not in ["sagittal", "coronal", "horizontal"]:
        raise ValueError("axis must be one of sagittal, coronal, or horizontal")
    cross, k = plane_alignment.find_plane_equation(section)
    if axis == "sagittal":
        center_point = (
            ((atlas_shape[1] / 2) * cross[1]) + ((atlas_shape[2] / 2) * cross[2]) + k
        )
        center_point_value = -(center_point / cross[0])
    elif axis == "coronal":
        center_point = (
            ((atlas_shape[0] / 2) * cross[0]) + ((atlas_shape[2] / 2) * cross[2]) + k
        )
        center_point_value = -(center_point / cross[1])
    elif axis == "horizontal":
        center_point = (
            ((atlas_shape[0] / 2) * cross[0]) + ((atlas_shape[1] / 2) * cross[1]) + k
        )
        center_point_value = -(center_point / cross[2])
    return center_point_value


def calculate_angles(df):
    """
    Calculates the Mediolateral and Dorsoventral angles for a series of predictions
    
    :param df: The dataframe containing the predictions
    :type df: pandas.DataFrame
    :return: a list of calculated ML and DV angles
    :rtype: list[float], list[float]
    """
    DV_list, ML_list = [], []
    for alignment in df.iterrows():
        # the quickNII coordinate vector
        m = alignment[1][
            ["ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz"]
        ].values.astype(np.float64)
        cross, k = plane_alignment.find_plane_equation(m)
        # calculate the Mediolateral and Dorsoventral angles
        DV = plane_alignment.get_angle(m, cross, k, "DV")
        ML = plane_alignment.get_angle(m, cross, k, "ML")
        # add the angles to the dataframe as new columns
        DV_list.append(DV)
        ML_list.append(ML)
    return DV_list, ML_list


def get_mean_angle(DV_list, ML_list, method, depths=None, species=None):
    """
    Propagates the Mediolateral and Dorsoventral angles for a series of predictions
    :param df: The dataframe containing the predictions
    :param method: The method used to calculate mean angles ('mean', 'weighted_mean')
    :param species: The species of the subject being aligned ('rat', 'mouse')
    :type df: pandas.DataFrame
    :type method: str
    :return: The calculated DV and ML angles
    :rtype: float, float
    """
    if method == "mean":
        DV_angle = np.mean(DV_list)
        ML_angle = np.mean(ML_list)
    elif method == "weighted_mean":
        df_center = depths
        if species == "mouse":
            min, max = 0, 528
        elif species == "rat":
            min, max = 0, 1024
        if len(df_center) > 2:
            weighted_accuracy = plane_alignment.make_gaussian_weights(max)
        else:
            weighted_accuracy = [1.0] * len(df_center)
        df_center = np.array(df_center)
        df_center[df_center < min] = min
        df_center[df_center > max] = max-1
        weighted_accuracy = [weighted_accuracy[int(y)] for y in df_center]
        print(weighted_accuracy)
        DV_angle = np.average(DV_list, weights=weighted_accuracy)
        ML_angle = np.average(ML_list, weights=weighted_accuracy)
    else:
        raise ValueError("method must be one of 'mean' or 'weighted_mean'")
    return DV_angle, ML_angle


def propagate_angles(df, method, species):
    """
    Propagates the Mediolateral and Dorsoventral angles for a series of predictions
    :param df: The dataframe containing the predictions
    :type df: pandas.DataFrame
    :return: an adjusted dataframe with the propagated angles
    :rtype: pandas.DataFrame
    """
    # get the angles for each section in the dataset
    DV_angle_list, ML_angle_list = calculate_angles(df)
    if method == "weighted_mean":
        depths = calculate_brain_center_depths(
            df[["ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz"]]
        )
    DV_angle, ML_angle = get_mean_angle(
        DV_angle_list, ML_angle_list, method, depths, species
    )
    print(f"DV angle: {DV_angle}\nML angle: {ML_angle}")
    # adjust the angles for each section in the dataset
    df = set_angles(df, DV_angle, ML_angle)
    return df


def set_angles(df, DV_angle, ML_angle):
    """
    Sets the Mediolateral and Dorsoventral angles for a series of predictions
    :param df: The dataframe containing the predictions
    :type df: pandas.DataFrame
    :return: an adjusted dataframe with the propagated angles
    :rtype: pandas.DataFrame
    """
    # adjust the angles for each section
    columns = ["ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz"]
    sections = []
    for section in df[columns].iterrows():
        section = np.array(section[1])
        section = plane_alignment.section_adjust(section, mean=DV_angle, direction="DV")
        section = plane_alignment.section_adjust(section, mean=ML_angle, direction="ML")
        sections.append(section)
    df[columns] = sections
    return df

