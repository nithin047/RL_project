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
    model = torch.load("policy_network.pt")
    model.eval()

    accuracy_array = []

    lambdaBS = 3e-6;
    lambdaUE = 3e-5;
    networkArea = 1e7;
    k = 10;
    episodeLength = 20;

    #create the environment
    env = myNetworkEnvironment(lambdaBS, lambdaUE, networkArea, k, episodeLength)

    for _ in trange(10000):


        obs = env.reset()
        k = len(obs)
        maxSINRPostion = np.argmax(obs)

        obs_torch = torch.from_numpy(obs)
        value = model(obs_torch.float())
        probability_array = value.data.numpy()

        sampledSINRPosition = np.random.choice(range(k), p=probability_array)

        if maxSINRPostion == sampledSINRPosition:
            accuracy_array.append(1)
        else:
            accuracy_array.append(0)


    print("accuracy:")
    print(np.mean(accuracy_array))

