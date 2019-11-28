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

def assignBSIDs(env, networkLength):

    #we will treat every element of data as a square of unit area in R^2
    #we determine associations for a user placed at every such unit square

    data = np.zeros((networkLength, networkLength))

    model = torch.load("policy_network.pt")
    model.eval()

    for ii in trange(networkLength):
        for jj in trange(networkLength):

            UECoordinates = [ii, jj]

            KClosestBS_withID = env.getKClosestBS(UECoordinates)
            k = env.k

            obs_torch = torch.from_numpy(KClosestBS_withID[0, :])
            value = model(obs_torch.float())
            probability_array = value.data.numpy()

            sampledSINRPosition = np.random.choice(range(k), p=probability_array)

            data[ii, jj] = KClosestBS_withID[1, sampledSINRPosition]

    return data


if __name__ == "__main__":

    model = torch.load("policy_network.pt")
    model.eval()

    lambdaBS = 3e-6;
    lambdaUE = 3e-5;
    networkArea = 1e7;
    k = 10;
    episodeLength = 20;

    #create the environment
    env = myNetworkEnvironment(lambdaBS, lambdaUE, networkArea, k, episodeLength)
    env.reset()
    #networkLength = 5
    networkLength = int(np.sqrt(networkArea))
    #assume that one square meter is an entry in the data matrix
    #data(i, j) = the id of the BS that a user at coordinates (i, j) will associate to in the network

    numColors = env.myNetwork.numberOfBS

    #data = np.random.rand(networkLength, networkLength) * numColors
    data = assignBSIDs(env, networkLength)

    cmap = cm.get_cmap('viridis', numColors)
    bounds = range(numColors+1)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots()
    ax.imshow(data, cmap=cmap, norm=norm)


    plt.savefig("visualization.eps")

