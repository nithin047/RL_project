import gym 
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
import sys
from tqdm import trange
from myNetworkEnvironment import myNetworkEnvironment
import matplotlib.pyplot as plt
from typing import Iterable


if __name__ == "__main__":

    model_name = str(sys.argv[1])

    #train_and_save()
    model = torch.load(model_name)
    #model.eval()

    accuracy_array = []

    lambdaBS = 3e-6;
    lambdaUE = 0.8e-5;
    networkArea = 1e7;
    k = 5;
    episodeLength = 3;
    handoffDuration = 0;
    velocity = 0; # 20 meters per second
    deltaT = 2;

    #create the environment
    env = myNetworkEnvironment(lambdaBS, lambdaUE, networkArea, k, handoffDuration, velocity, deltaT, episodeLength)

    for _ in trange(10000):


        obs = env.reset()
        k = env.k
        maxSINRPostionShared = np.argmax(np.divide(obs[0:k], obs[k:2*k]+1))

        obs_torch = torch.from_numpy(obs)
        value = model(obs_torch.float())
        probability_array = value.data.numpy()

        sampledSINRPosition = np.random.choice(range(k), p=probability_array)

        if maxSINRPostionShared == sampledSINRPosition:
            accuracy_array.append(1)
        else:
            accuracy_array.append(0)


    print("accuracy:")
    print(np.mean(accuracy_array))

