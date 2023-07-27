from typing import Union, List, Optional
import numpy as np
import pandas as pd
import re
from .depth_estimation import calculate_brain_center_depths
from .plane_alignment_functions import plane_alignment

def trim_mean(arr: np.array, percent: int) -> float:
    """"
    Calculates the trimmed mean of an array, sourced from:
    https://gist.github.com/StuffbyYuki/6f25f9f2f302cb5c1e82e4481016ccde

    :param arr: the array to calculate the trimmed mean of
    :type arr: np.array
    :param percent: the percentage of values to trim
    :type percent: int
    :return: the trimmed mean
    :rtype: float
    """
    n = len(arr)
    k = int(round(n * (float(percent) / 100) / 2))
    return np.mean(arr[k + 1 : n - k])


def calculate_average_section_thickness(
    section_numbers: List[Union[int, float]], section_depth: List[Union[int, float]], bad_sections, method="weighted",
    species="mouse"
) -> float:
    """
    Calculates the average section thickness for a series of predictions

    :param section_numbers: List of section numbers
    :param section_depth: List of section depths
    :type section_numbers: List[int, float]
    :type section_depth: List[int, float]
    :return: the average section thickness
    :rtype: float
    """
    # inter section number differences
    if bad_sections is not None:
        section_numbers = section_numbers[bad_sections == False].reset_index(drop=True)
        section_depth = section_depth[bad_sections == False]
    number_spacing = section_numbers[:-1].values - section_numbers[1:].values
    # inter section depth differences
    depth_spacing = section_depth[:-1] - section_depth[1:]
    # dividing depth spacing by number spacing allows us to control for missing sections
    weighted_accuracy = calculate_weighted_accuracy(section_numbers, section_depth, species, None, method)
    section_thicknesses = depth_spacing / number_spacing
    average_thickness = np.average(section_thicknesses, weights = weighted_accuracy[1:])
    return average_thickness






def ideal_spacing(
    section_numbers: List[Union[int, float]],
    section_depth: List[Union[int, float]],
    average_thickness: Union[int, float],
    bad_sections: List[bool] = None,
    method = "weighted",
    species = "mouse"
) -> float:
    """
    Calculates the ideal spacing for a series of predictions

    :param section_numbers: List of section numbers
    :param section_depth: List of section depths
    :param average_thickness: The average section thickness
    :type section_numbers: List[int, float]
    :type section_depth: List[int, float]
    :type average_thickness: int, float
    :return: the ideal spacing
    :rtype: float
    """
    # unaligned voxel position of section numbers (evenly spaced depths)
    index_spaced_depth = section_numbers * average_thickness
    # average distance between the depths and the evenly spaced depths
    
    weighted_accuracy = calculate_weighted_accuracy(section_numbers, section_depth, species, bad_sections, method)
    distance_to_ideal = np.average(section_depth - index_spaced_depth, weights = weighted_accuracy)
    # adjust the evenly spaced depths to minimise their distance to the predicted depths
    ideal_index_spaced_depth = index_spaced_depth + distance_to_ideal
    return ideal_index_spaced_depth


def determine_direction_of_indexing(depth: List[Union[int, float]]) -> str:
    """
    Determines the direction of indexing for a series of predictions

    :param depth: List of depths sorted by section index
    :type depth: List[int, float]
    :return: the direction of indexing
    :rtype: str
    """

    if trim_mean(depth[1:] - depth[:-1], 10) > 0:
        direction = "rostro-caudal"
    else:
        direction = "caudal-rostro"
    return direction


def enforce_section_ordering(predictions):
    """
    Ensures that the predictions are ordered by section number

    :param predictions: dataframe of predictions
    :type predictions: pandas.DataFrame
    :return: the input dataframe ordered by section number
    :rtype: pandas.DataFrame
    """
    predictions = predictions.sort_values(by=["nr"], ascending=True).reset_index(drop=True)
    if len(predictions) == 1:
        raise ValueError("Only one section found, cannot space according to index")
    if "nr" not in predictions:
        raise ValueError(
            "No section indexes found, cannot enforce index order. You likely did not run predict() with section_numbers=True"
        )
    else:
        predictions = predictions.reset_index(drop=True)
        depths = calculate_brain_center_depths(predictions)
        depths = np.array(depths)
        direction = determine_direction_of_indexing(depths)
        predictions["depths"] = depths
        
        temp = predictions.copy()
        if direction == "caudal-rostro":
            ascending = False
        if direction == "rostro-caudal":
            ascending = True
        if "bad_section" in temp:
            temp_good = temp[temp["bad_section"] == False].copy().reset_index(drop=True)
            temp_good_copy = temp_good.copy()
            temp_good_copy = temp_good_copy.sort_values(by=["depths"], ascending=ascending).reset_index(
                drop=True
            )
            temp_good["oy"] = temp_good_copy["oy"]
            
            predictions.loc[predictions["bad_section"] == False, "oy"] = temp_good["oy"].values
        else:
            temp = temp.sort_values(by=["depths"], ascending=ascending).reset_index(drop=True)

        
            predictions["oy"] = temp["oy"].values
    return predictions


def space_according_to_index(predictions, section_thickness = None, voxel_size = None, suppress = False, species = "mouse"):
    """
    Space evenly according to the section indexes, if these indexes do not represent the precise order in which the sections were
    cut, this will lead to less accurate predictions. Section indexes must account for missing sections (ie, if section 3 is missing
    indexes must be 1, 2, 4).

    :param predictions: dataframe of predictions
    :type predictions: pandas.DataFrame
    :return: the input dataframe with evenly spaced sections
    :rtype: pandas.DataFrame
    """
    if voxel_size == None:
        raise ValueError("voxel_size must be specified")
    if section_thickness is not None:
        section_thickness/=voxel_size
    predictions["oy"] = predictions["oy"].astype(float)
    if len(predictions) == 1:
        raise ValueError("Only one section found, cannot space according to index")
    if "nr" not in predictions:
        raise ValueError(
            "No section indexes found, cannot space according to a missing index. You likely did not run predict() with section_numbers=True"
        )
    else:
        if 'bad_section' in predictions:
            bad_sections = predictions['bad_section'].values
        else:
            bad_sections = None
        predictions = enforce_section_ordering(predictions)
        depths = calculate_brain_center_depths(predictions)
        depths = np.array(depths)
        if not section_thickness:
            section_thickness = calculate_average_section_thickness(
                predictions["nr"], section_depth = depths, bad_sections=bad_sections, species=species
            )
            if not suppress:
                print(f'predicted thickness is {section_thickness * voxel_size}µm')
        else:
            if not suppress:
                print(f'specified thickness is {section_thickness * voxel_size}µm')

        calculated_spacing = ideal_spacing(predictions["nr"], depths, section_thickness, bad_sections, species=species)
        distance_to_ideal = calculated_spacing - depths
        predictions["oy"] = predictions["oy"] + distance_to_ideal
    return predictions


def number_sections(filenames: List[str], legacy=False) -> List[int]:
    """
    returns the section numbers of filenames

    :param filenames: list of filenames
    :type filenames: list[str]
    :return: list of section numbers
    :rtype: list[int]
    """
    filenames = [filename.split("\\")[-1] for filename in filenames]
    section_numbers = []
    for filename in filenames:
        if not legacy:
            match = re.findall(r"\_s\d+", filename)
            if len(match) == 0:
                raise ValueError(f"No section number found in filename: {filename}")
            if len(match) > 1:
                raise ValueError(
                    "Multiple section numbers found in filename, ensure only one instance of _s### is present, where ### is the section number"
                )
            section_numbers.append(int(match[-1][2:]))
        else:
            match = re.sub("[^0-9]", "", filename)
            ###this gets the three numbers closest to the end
            section_numbers.append(match[-3:])
    if len(section_numbers) == 0:
        raise ValueError("No section numbers found in filenames")
    return section_numbers


def set_bad_sections_util(df: pd.DataFrame, bad_sections: List[str], auto = False) -> pd.DataFrame:
    """
    Sets the damaged sections and sections which deepslice may not perform well on for a series of predictions
    
    :param bad_sections: List of bad sections
    :param df: dataframe of predictions
    :param auto: automatically set bad sections based on if theyre badly positioned relative to their section index
    :type bad_sections: List[int]
    :type df: pandas.DataFrame
    :type auto: bool
    :return: the input dataframe with bad sections labeled as such
    :rtype: pandas.DataFrame
    """

    bad_section_indexes = [
        df.Filenames.str.contains(bad_section) for bad_section in bad_sections
    ]
    if np.any([np.sum(x)>1 for x in bad_section_indexes]):
        raise ValueError("Multiple sections match the same bad section string, make sure each bad section string is unique")
    bad_section_indexes = [np.where(x)[0] for x in bad_section_indexes]
    bad_section_indexes = np.concatenate(bad_section_indexes)
    df.loc[~df.index.isin(bad_section_indexes), "bad_section"] = False
    if auto:
        df['depths'] = calculate_brain_center_depths(df)
        x = df['nr'].values
        y = df['depths'].values
        m,b = np.polyfit(x,y,1)
        residuals = y - (m*x + b)
        outliers = np.abs(residuals) > 1.5*np.std(residuals)
        df.loc[outliers, 'bad_section'] = True
        
    df.loc[bad_section_indexes, "bad_section"] = True
    # make the other sections are False
    
    bad_sections_found = np.sum(bad_section_indexes)
    # Tell the user which sections were identified as bad
    if bad_sections_found > 0:
        print(
            f"{bad_sections_found} sections out of {len(bad_sections)} were marked as bad, \n\
        They are:\n {df.Filenames[bad_section_indexes]}"
        )
    return df


def calculate_weighted_accuracy(section_numbers: List[int], depths: List[float], species: str, bad_sections: List[Optional[bool]]  = None, method: str = "weighted") -> List[float]:
    """
    Calculates the weighted accuracy of a list of section numbers for a given species
    
    :param section_numbers: List of section numbers
    :param species: Species to calculate accuracy for
    :param bad_sections: List of bad sections
    :param method: Method to use for weighting, defaults to "weighted"
    :type section_numbers: List[int]
    :type species: str
    :type bad_sections: List[Optional[bool]]
    :type method: str
    :return: List of weighted accuracies
    :rtype: List[float]
    """
    if species == "mouse":
        min, max = 0, 528
    elif species == "rat":
        min, max = 0, 1024
    if method == "weighted":
        weighted_accuracy = plane_alignment.make_gaussian_weights(max + 1)
        depths = np.array(depths)
        depths[depths < min] = min
        depths[depths > max] = max
        weighted_accuracy = [weighted_accuracy[int(y)] for y in depths]
    elif method is None:
        weighted_accuracy = [1 for y in section_numbers]
    if len(section_numbers) <= 2:
        weighted_accuracy = [0.5, 0.5]
        
    if bad_sections is not None:
        weighted_accuracy = [x if y == False else 0 for x,y in zip(weighted_accuracy,bad_sections)]
        
    return weighted_accuracy