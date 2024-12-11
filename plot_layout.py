#%%

# Import necessary libraries
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
#%%
def plotcolors(Type,n):
    if Type=='quadratic':
        endcolors = ['blue','limegreen']
        custom_cmap = LinearSegmentedColormap.from_list("BlueToGreen",endcolors)
        colors = custom_cmap(np.linspace(0,1,n))
        return colors
    elif Type=='kipping':
        endcolors = ['red','blueviolet']
        custom_cmap = LinearSegmentedColormap.from_list("RedToViolet",endcolors)
        colors = custom_cmap(np.linspace(0,1,n))
        return colors
    elif Type=='truth':
        endcolors = ['darkgreen','mediumaquamarine']
        custom_cmap = LinearSegmentedColormap.from_list("Green",endcolors)
        colors = custom_cmap(np.linspace(0,1,n))
        return colors
    else:
        print("Not valid Type")
        return
#%%
def plotsizes():
    fontsize = {
        'title': 16,
        'legend': 14,
        'axis': 14,
        'tick_label': 12
    }
    figsize = (14,8)
    return figsize, fontsize
#%%