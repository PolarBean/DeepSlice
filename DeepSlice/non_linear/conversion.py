
    
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

