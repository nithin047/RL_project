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
from reinforce import train_and_save


if __name__ == "__main__":



    #train_and_save()
    model = torch.load("policy_network_phase3.pt")
    model.eval()

    accuracy_array = []

    lambdaBS = 3e-6;
    lambdaUE = 3e-5;
    networkArea = 1e7;
    k = 10;
    episodeLength = 20;
    handoffDuration = 0; # 2 steps
    velocity = 0; # 20 meters per second
    deltaT = 2;

    #create the environment
    env = myNetworkEnvironment(lambdaBS, lambdaUE, networkArea, k, handoffDuration, velocity, deltaT, episodeLength)

    for _ in trange(10000):


        obs = env.reset()
        k = len(obs)
        maxPerUserSINRPostion = np.argmax(np.divide(obs[0:k], obs[k+1:]))

        obs_torch = torch.from_numpy(obs)
        value = model(obs_torch.float())
        probability_array = value.data.numpy()

        sampledPerUserSINRPosition = np.random.choice(range(k), p=probability_array)

        if maxPerUserSINRPostion == sampledPerUserSINRPosition:
            accuracy_array.append(1)
        else:
            accuracy_array.append(0)


    print("accuracy:")
    print(np.mean(accuracy_array))

