import numpy as np
import math
from PIL import Image
import glob
image_list = []

## a 2-dimensional plane can be fully describable if you know any three points that lie on its surface
def find_plane_equation(plane):
    a, b, c = np.array(plane[0:3], dtype=np.float64), np.array(plane[3:6], dtype=np.float64), np.array(plane[6:9],
                                                                                                       dtype=np.float64)
    cross = np.cross(b, c)
    cross /= 9
    k = -((a[0] * cross[0]) + (a[1] * cross[1]) + (a[2] * cross[2]))
    return (cross, k)


def get_angle(inp, cross, k, direction):
    section = inp.copy()
    for i in range(3):
        section[i + 3] += section[i]
        section[i + 6] += section[i]
    if direction == 'ML':
        a = section[0:2]
        linear_point = (((section[0] - 100) * cross[0]) + ((section[2]) * cross[2])) + k
        depth = -(linear_point / cross[1])
        b = np.array((section[0] - 100, depth))
        c = b + [100, 0]
    if direction == 'DV':
        a = section[1:3]
        linear_point = (((section[0]) * cross[0]) + ((section[2] - 100) * cross[2])) + k
        depth = -(linear_point / cross[1])
        b = np.array((depth, section[2] - 100))
        c = b + [0, 100]
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    ##This looks redundant, needs to be tested
    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle)
    if b[1] > a[1]:
        angle *= -1
    return (angle)


def rotation_around_axis(axis, angle):
    '''
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
    '''
    angle = np.radians(angle)
    axis = axis / np.linalg.norm(axis)

    a = math.cos(angle / 2.0)
    b, c, d = axis * math.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d

    return np.array([[aa + bb - cc - dd, 2 * (bc - ad), 2 * (bd + ac)],
                     [2 * (bc + ad), aa + cc - bb - dd, 2 * (cd - ab)],
                     [2 * (bd - ac), 2 * (cd + ab), aa + dd - bb - cc]])


def get_axis(m, translation_vector, direction):
    ##find the plane equation for a set of QuickNII coordinates
    cross, k = find_plane_equation(m)
    translated_volume = np.array((528, 0, 456)) - translation_vector
    linear_point = (((translated_volume[0] / 2) * cross[0]) + ((translated_volume[2] / 2) * cross[2])) + k
    depth = -(linear_point / cross[1])
    axis = ((translated_volume[0] / 2, depth, translated_volume[2] / 2))
    if direction == 'DV':
        linear_point = (((translated_volume[0]) * cross[0]) + ((translated_volume[2] / 2) * cross[2])) + k
    if direction == 'ML':
        linear_point = (((translated_volume[0] / 2) * cross[0]) + ((translated_volume[2]) * cross[2])) + k
    depth = -(linear_point / cross[1])
    if direction == 'DV':
        axis2 = ((translated_volume[0], depth, translated_volume[2] / 2))
    if direction == 'ML':
        axis2 = ((translated_volume[0] / 2, depth, translated_volume[2]))
    axis_vector = np.array(axis) - np.array(axis2)
    return (axis_vector)


def rotate_section(section, degrees, direction):
    cross, k = find_plane_equation(section)
    ##this looks redundant
    # if direction==ML:
    #   ML=get_angle(section.reshape(9),cross,k,direction=direction)
    section_points = section.copy()
    for i in range(3):
        section_points[i + 3] += section_points[i]
        section_points[i + 6] += section_points[i]

    points = section_points.reshape(3, 3)
    translated_volume = np.array((528, 0, 456))
    linear_point = (((translated_volume[0] / 2) * cross[0]) + ((translated_volume[2] / 2) * cross[2])) + k
    depth = -(linear_point / cross[1])
    translation_vector = ((translated_volume[0] / 2, depth, translated_volume[2] / 2))
    translated_points = points - translation_vector
    axis = get_axis(section, translation_vector, direction=direction)
    rot_matrix = rotation_around_axis(axis, degrees)
    rotated_translated_points = np.dot(translated_points, rot_matrix)  # perform rotation, centred on (0,0,0)
    rotated_points = rotated_translated_points + translation_vector  # translate back to original geometric centre.
    rotated_points = rotated_points.reshape(9)
    for i in range(3):
        rotated_points[i + 3] -= rotated_points[i]
        rotated_points[i + 6] -= rotated_points[i]
    return (rotated_points)

def Section_adjust(section,direction,mean):
    cross,k=find_plane_equation(section)
    angle=get_angle(section,cross,k, direction)
    dif=angle-mean
    rot=(rotate_section(section,-dif,direction))
    return(rot)