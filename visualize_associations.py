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

def assignBSIDs(env, networkLength, model_name):

    #we will treat every element of data as a square of unit area in R^2
    #we determine associations for a user placed at every such unit square

    data = np.zeros((networkLength, networkLength))

    model = torch.load(model_name)
    model.eval()

    k = env.k
    lambdaUE = env.lambdaUE

    if lambdaUE == 0:

        print("Visualizing Phase 1")

        for ii in trange(networkLength):
            for jj in trange(networkLength):

                UECoordinates = [ii, jj]

                KClosestBS_withID = env.getKClosestBSSINR(UECoordinates)
                obs_torch = torch.from_numpy(KClosestBS_withID[0, :])
                value = model(obs_torch.float())
                probability_array = value.data.numpy()

                sampledSINRPosition = np.random.choice(range(k), p=probability_array)

                data[ii, jj] = KClosestBS_withID[1, sampledSINRPosition]

        # for ii in trange(networkLength):
        #     for jj in trange(networkLength):

        #         UECoordinates = [ii, jj]

        #         KClosestBS_withIDAndLoad = env.getKClosestBSSINRShared(UECoordinates)
        #         obs_torch = torch.from_numpy(np.concatenate(KClosestBS_withIDAndLoad[0, :], KClosestBS_withIDAndLoad[1, :]))
        #         value = model(obs_torch.float())
        #         probability_array = value.data.numpy()

        #         sampledSINRPosition = np.random.choice(range(k), p=probability_array)

        #         data[ii, jj] = KClosestBS_withIDAndLoad[2, sampledSINRPosition]

    else:

        print("Visualizing Phase 2")

        for ii in trange(networkLength):
            for jj in trange(networkLength):

                UECoordinates = [ii, jj]

                KClosestBS_withIDAndLoad = env.getKClosestBSSINRShared(UECoordinates)
                obs_torch = torch.from_numpy(np.concatenate(KClosestBS_withIDAndLoad[0, :], KClosestBS_withIDAndLoad[1, :]))
                value = model(obs_torch.float())
                probability_array = value.data.numpy()

                sampledSINRPosition = np.random.choice(range(k), p=probability_array)

                data[ii, jj] = KClosestBS_withIDAndLoad[2, sampledSINRPosition]


    return data


if __name__ == "__main__":

    model_name = str(sys.argv[1])

    lambdaBS = 3e-6;
    lambdaUE = 0;
    networkArea = 1e7;
    k = 10;
    episodeLength = 3;
    handoffDuration = 0;

    #create the environment
    env = myNetworkEnvironment(lambdaBS, lambdaUE, networkArea, k, handoffDuration, episodeLength)
    env.reset()

    BSLocations = env.myNetwork.BSLocation
    vor = Voronoi(BSLocations)
    X = []
    Y = []
    for ii in range(len(BSLocations)):
        X.append(BSLocations[ii][0])
        Y.append(BSLocations[ii][1])
    

    #networkLength = 1000
    networkLength = int(np.sqrt(networkArea))
    #assume that one square meter is an entry in the data matrix
    #data(i, j) = the id of the BS that a user at coordinates (i, j) will associate to in the network

    numColors = env.myNetwork.numberOfBS

    data = assignBSIDs(env, networkLength, model_name)

    cmap = cm.get_cmap('PiYG', numColors)
    bounds = range(numColors+1)
    norm = colors.BoundaryNorm(bounds, cmap.N)



    fig, ax = plt.subplots()
    ax.imshow(data, cmap=cmap, norm=norm)
    plt.plot(BSLocations)
    plt.scatter(X, Y, c = 'black')
    plt.savefig("visualization_withBS.eps")
    plt.close()

    voronoi_plot_2d(vor, show_vertices = False)
    plt.savefig("ground_truth.eps")


