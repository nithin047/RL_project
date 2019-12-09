import gym 
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
import sys
from tqdm import trange
from myNetworkEnvironment import myNetworkEnvironment
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from scipy.spatial import Voronoi, voronoi_plot_2d

if __name__ == "__main__":

    flipped_BSLocations = np.loadtxt("./visualization_data/phase_2/flipped_BSLocations1.txt")
    data = np.loadtxt("./visualization_data/phase_2/data1.txt")
    X = np.loadtxt("./visualization_data/phase_2/X1.txt")
    Y = np.loadtxt("./visualization_data/phase_2/Y1.txt")

    numColors = len(X)

    cmap = cm.get_cmap('PiYG', numColors)
    bounds = range(numColors+1)
    norm = colors.BoundaryNorm(bounds, cmap.N)


    #data = np.flipud(data)
    #data = np.fliplr(data)


    # flipped_BSLocations = np.flipud(flipped_BSLocations)
    # flipped_BSLocations = np.fliplr(flipped_BSLocations)

    vor = Voronoi(flipped_BSLocations)

    fig, ax = plt.subplots()
    ax.imshow(data, cmap=cmap, norm=norm)
    #plt.scatter(Y, 3161 - X, c = 'black')
    #plt.savefig("visualization_withBS.eps")


    #fig1, ax1 = plt.subplots()
    ax.set_xlim((0, 3000))
    ax.set_ylim((0, 3000))
    voronoi_plot_2d(vor, ax, show_vertices = False)

    plt.savefig("ground_truth_combined.eps")