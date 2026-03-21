import numpy as np
import matplotlib as mpl
import matplotlib.colors as mcolors


def generate_cmap(colors, cmap_name):
    values = range(len(colors))
    vmax = np.ceil(np.max(values))
    color_list = []
    for vi, ci in zip(values, colors):
        color_list.append( ( vi/ vmax, ci) )

    return mcolors.LinearSegmentedColormap.from_list(cmap_name, color_list, 256)


def norm(out, min, max):
    return ( out - min ) / ( max - min )