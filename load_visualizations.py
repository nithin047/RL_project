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

    flipped_BSLocations = np.loadtxt("./visualization_data/flipped_BSLocations.txt")
    data = np.loadtxt("./visualization_data/data.txt")
    X = np.loadtxt("./visualization_data/X.txt")
    Y = np.loadtxt("./visualization_data/Y.txt")

    numColors = len(X)

    data = assignBSIDs(env, networkLength, model_name)

    cmap = cm.get_cmap('PiYG', numColors)
    bounds = range(numColors+1)
    norm = colors.BoundaryNorm(bounds, cmap.N)



    fig, ax = plt.subplots()
    ax.imshow(data, cmap=cmap, norm=norm)
    plt.scatter(X, Y, c = 'black')
    plt.savefig("visualization_withBS.eps")
    plt.close()

    voronoi_plot_2d(vor, show_vertices = False)
    plt.savefig("ground_truth.eps")