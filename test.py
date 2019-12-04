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

lambdaBS = 3e-6;
lambdaUE = 0;
networkArea = 1e7;
k = 10;
episodeLength = 3;
handoffDuration = 0;
velocity = 0; # 20 meters per second
deltaT = 2;

#create the environment
env = myNetworkEnvironment(lambdaBS, lambdaUE, networkArea, k, handoffDuration, velocity, deltaT, episodeLength)
env.reset()
BSLocations = env.myNetwork.BSLocation
flipped_BSLocations = []
for ii in range(len(BSLocations)):
    flipped_BSLocations.append(BSLocations[ii][::-1])
np.savetxt("./visualization_data/flipped_BSLocations.txt", flipped_BSLocations)

vor = Voronoi(flipped_BSLocations)
X = []
Y = []
for ii in range(len(BSLocations)):
    X.append(BSLocations[ii][0])
    Y.append(BSLocations[ii][1])
voronoi_plot_2d(vor, show_vertices = False)
plt.savefig("test.eps")
plt.close()
plt.scatter(X, Y, c = 'black')
plt.savefig("Help.eps")