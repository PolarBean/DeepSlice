from statistics import mean
import numpy as np
from scipy.stats import norm

import math


def find_plane_equation(plane):
    """
    Finds the plane equation of a plane
    :param plane: the plane to find the equation of
    :type plane: :any:`numpy.ndarray`
    :returns: the normal vector of the plane and the constant k
    :rtype: :any:`numpy.ndarray`, float
    """
    a, b, c = (
        np.array(plane[0:3], dtype=np.float64),
        np.array(plane[3:6], dtype=np.float64),
        np.array(plane[6:9], dtype=np.float64),
    )
    cross = np.cross(b, c)
    cross /= 9
    k = -((a[0] * cross[0]) + (a[1] * cross[1]) + (a[2] * cross[2]))
    return (cross, k)


def get_angle(inp, cross, k, direction):
    # inp is the input plane, represented by 3 xzy sets
    # cross and k is the normal vector of the plane
    # Direction defines whether we want the mediolateral or dorsaventral angle
    section = inp.copy()
    # transform vector into absolute coordinates
    for i in range(3):
        section[i + 3] += section[i]
        section[i + 6] += section[i]
    if direction == "ML":
        # original xzy point
        a = section[0:2]
        # calculate a point which differs from this point only in the x dimension
        # to do this we use the plane equation which tells us the position of every point on the plane
        linear_point = (((section[0] - 100) * cross[0]) + ((section[2]) * cross[2])) + k
        # this tells us the depth of that point which differs in x dimension but lies on the same plane
        depth = -(linear_point / cross[1])
        b = np.array((section[0] - 100, depth))
        c = b + [100, 0]

    if direction == "DV":
        a = section[1:3]
        linear_point = (((section[0]) * cross[0]) + ((section[2] - 100) * cross[2])) + k
        depth = -(linear_point / cross[1])
        b = np.array((depth, section[2] - 100))
        c = b + [0, 100]
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    # This looks redundant, needs to be tested
    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle)
    if direction == "ML":
        if b[1] > a[1]:
            angle *= -1
    if direction == "DV":
        if b[0] < a[0]:
            angle *= -1
    return angle


def rotation_around_axis(axis, angle):
    """
    Generates a 3x3 rotation matrix using the Euler-Rodrigues formula
    following the definition here:
    https://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_formula.
    :param axis: the axis around which to rotate as a vector of length 3
                 (no normalisation required)
    :type axis: array like
    :param angle: the angle in radians to rotate
    :type angle: float
    :returns: the rotation matrix
    :rtype: a 3x3 :any:`numpy.ndarray`
    """
    angle = np.radians(angle)
    axis = axis / np.linalg.norm(axis)

    a = math.cos(angle / 2.0)
    b, c, d = axis * math.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d

    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc - ad), 2 * (bd + ac)],
            [2 * (bc + ad), aa + cc - bb - dd, 2 * (cd - ab)],
            [2 * (bd - ac), 2 * (cd + ab), aa + dd - bb - cc],
        ]
    )



def make_gaussian_weights(size):
    x = np.linspace(-np.pi, np.pi, size)
    weights = np.exp(-(x ** 2) / 2) / np.sqrt(2 * np.pi)
    weights[weights>size-1] = size-1
    weights[weights<0] = 0
    return weights


def get_axis(m, translation_vector, direction, plane_of_section=None, atlas="AMBA"):
    """
    :param m: the matrix to rotate
    :type m: 3x3 :any:`numpy.ndarray`
    :param translation_vector: the translation vector to apply
    :type translation_vector: 3x1 :any:`numpy.ndarray`
    :param direction: the direction of the rotation
    :type direction: string
    :param plane: the plane to rotate around
    :type plane: string
    :returns: the axis of rotation
    :rtype: 3x1 :any:`numpy.ndarray`
    """
    # find the plane equation for a set of QuickNII coordinates
    cross, k = find_plane_equation(m)
    if atlas == "AMBA":
        volume = np.array((528, 320, 456))
        posx, posy, posz = volume / 2

    if atlas == "WHS":
        volume = np.array((512, 1024, 512))
        posx, posy, posz = 256, 512, 256

    translated_volume = volume - translation_vector

    cor_linear_point = (
        (((translated_volume[0] / 2)) * cross[0])
        + ((translated_volume[2] / 2) * cross[2])
    ) + k
    cor_Y = -(cor_linear_point / cross[1])
    #     cor_axis = ((translated_volume[0] / 2, depth, translated_volume[2] / 2))

    sag_linear_point = (
        ((translated_volume[1] / 2) * cross[1])
        + ((translated_volume[2] / 2) * cross[2])
    ) + k
    sag_X = -(sag_linear_point / cross[0])
    #     sag_axis = ((translated_volume[1] / 2, depth, translated_volume[2] / 2))

    horz_linear_point = (
        ((translated_volume[0] / 2) * cross[0])
        + ((translated_volume[1] / 2) * cross[1])
    ) + k
    horz_Z = -(horz_linear_point / cross[2])
    if plane_of_section is None:
        plane_of_section = np.argmin(
            np.abs((cor_Y - posy, sag_X - posx, horz_Z - posz))
        )
    choices = {"x": sag_X, " y": cor_Y, " z": horz_Z}

    if plane_of_section == 0:
        axis = (translated_volume[0] / 2, cor_Y, translated_volume[2] / 2)
        if direction == "DV":
            linear_point = (
                ((translated_volume[0]) * cross[0])
                + ((translated_volume[2] / 2) * cross[2])
            ) + k
            Ypred = -(linear_point / cross[1])
            ##the way QNII rotates is but i prefer my way
            #             axis2 = ((translated_volume[0], cor_Y, translated_volume[2] / 2))
            axis2 = (translated_volume[0], Ypred, translated_volume[2] / 2)

        ##this gives me the depth of a point directly beside the coronal center point
        if direction == "ML":
            linear_point = (
                ((translated_volume[0] / 2) * cross[0])
                + ((translated_volume[2]) * cross[2])
            ) + k
            Ypred = -(linear_point / cross[1])
            axis2 = (translated_volume[0] / 2, Ypred, translated_volume[2])

    if plane_of_section == 1:
        axis = (sag_X, translated_volume[1] / 2, translated_volume[2] / 2)
        if direction == "DV":
            linear_point = (
                ((translated_volume[1]) * cross[1])
                + ((translated_volume[2] / 2) * cross[2])
            ) + k
            Xpred = -(linear_point / cross[0])
            axis2 = (Xpred, translated_volume[1], translated_volume[2] / 2)
        if direction == "ML":
            linear_point = (
                ((translated_volume[1] / 2) * cross[1])
                + ((translated_volume[2]) * cross[2])
            ) + k
            Xpred = -(linear_point / cross[0])
            axis2 = (Xpred, translated_volume[1] / 2, translated_volume[2])

    if plane_of_section == 2:
        axis = (translated_volume[0] / 2, translated_volume[1] / 2, horz_Z)
        if direction == "DV":
            linear_point = (
                ((translated_volume[0]) * cross[0])
                + ((translated_volume[1] / 2) * cross[1])
            ) + k
            Zpred = -(linear_point / cross[2])
            axis2 = (translated_volume[0], translated_volume[1] / 2, Zpred)

        if direction == "ML":
            linear_point = (
                ((translated_volume[0] / 2) * cross[0])
                + ((translated_volume[1]) * cross[1])
            ) + k
            Zpred = -(linear_point / cross[2])
            axis2 = (translated_volume[0] / 2, translated_volume[1], Zpred)
    axis_vector = np.array(axis) - np.array(axis2)
    return axis_vector


def rotate_section(section, degrees, direction, plane_of_section=None, atlas="AMBA"):
    """
    Rotates a section
    :param section: the section to rotate
    :type section: :any:`numpy.ndarray`
    :param degrees: the degrees to rotate the section
    :type degrees: float
    :param direction: the direction of the rotation
    :type direction: string
    :param plane_of_section: the plane to rotate around
    :type plane_of_section: string
    :returns: the rotated section
    :rtype: :any:`numpy.ndarray`
    """

    cross, k = find_plane_equation(section)

    # this looks redundant
    # if direction==ML:
    #   ML=get_angle(section.reshape(9),cross,k,direction=direction)
    section_points = section.copy()
    for i in range(3):
        section_points[i + 3] += section_points[i]
        section_points[i + 6] += section_points[i]

    points = section_points.reshape(3, 3)
    if atlas == "WHS":
        translated_volume = np.array((489, 1024, 590))
        posx, posy, posz = 256, 512, 256
    if atlas == "AMBA":
        translated_volume = np.array((528, 320, 456))
        posx, posy, posz = translated_volume / 2

    cor_linear_point = (
        (((translated_volume[0] / 2)) * cross[0])
        + ((translated_volume[2] / 2) * cross[2])
    ) + k
    cor_Y = -(cor_linear_point / cross[1])
    #     cor_axis = ((translated_volume[0] / 2, depth, translated_volume[2] / 2))

    sag_linear_point = (
        ((translated_volume[1] / 2) * cross[1])
        + ((translated_volume[2] / 2) * cross[2])
    ) + k
    sag_X = -(sag_linear_point / cross[0])
    #     sag_axis = ((translated_volume[1] / 2, depth, translated_volume[2] / 2))

    horz_linear_point = (
        ((translated_volume[0] / 2) * cross[0])
        + ((translated_volume[1] / 2) * cross[1])
    ) + k
    horz_Z = -(horz_linear_point / cross[2])
    if plane_of_section is None:
        plane_of_section = np.argmin(
            np.abs((cor_Y - posy, sag_X - posx, horz_Z - posz))
        )
    #     midpoint = translated_volume/2
    #     x = symbols('x')
    #     expr = sum((x * cross + midpoint) * cross) - k
    #     m = solve(expr)
    #     translation_vector = np.array((m * cross + midpoint), dtype=np.float)
    if plane_of_section == 0:

        translation_vector = (translated_volume[0] / 2, cor_Y, translated_volume[2] / 2)

    if plane_of_section == 1:
        translation_vector = (sag_X, translated_volume[1] / 2, translated_volume[2] / 2)

    if plane_of_section == 2:
        translation_vector = (
            translated_volume[0] / 2,
            translated_volume[1] / 2,
            horz_Z,
        )

    translated_points = points - translation_vector
    axis = get_axis(
        section,
        translation_vector,
        direction=direction,
        plane_of_section=plane_of_section,
    )
    rot_matrix = rotation_around_axis(axis, degrees)
    # perform rotation, centred on (0,0,0)
    rotated_translated_points = np.dot(translated_points, rot_matrix)
    # translate back to original geometric centre.
    rotated_points = rotated_translated_points + translation_vector
    rotated_points = rotated_points.reshape(9)
    for i in range(3):
        rotated_points[i + 3] -= rotated_points[i]
        rotated_points[i + 6] -= rotated_points[i]
    return rotated_points


def section_adjust(section, direction, mean):
    cross, k = find_plane_equation(section)
    angle = get_angle(section, cross, k, direction)
    dif = angle - mean
    rot = rotate_section(section, -dif, direction)
    return rot

