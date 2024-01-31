def min_max_to_loc_scale(minimum, maximum):
    loc = (maximum + minimum) / 2
    scale = (maximum - minimum) / 2
    return loc, scale
