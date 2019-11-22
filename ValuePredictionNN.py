#setting up the basic structure of a neural network to approximate state values

import gym
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable

class ValueFunctionWithNN():
    def __init__(self, k):
        """
        k: the number of nearest BSs that we care about
        """
        self.n_in = 2*k
        self.n_h = 32
        self.n_out = 1

        self.model = nn.Sequential(nn.Linear(self.n_in, self.n_h), nn.ReLU(), nn.Linear(self.n_h, self.n_h), nn.ReLU(), nn.Linear(self.n_h, self.n_out))
        self.model = self.model.float()
        self.optimizer = optim.Adam(self.model.parameters(), lr = 1e-3)


    def __call__(self, s):
        # TODO: implement this method
        s = Variable(torch.from_numpy(s))
        value = self.model(s.float())
        return float(value.data.numpy()[0])
        

    def update(self, alpha, G, s_tau):

        self.optimizer.zero_grad()
        s_tau = Variable(torch.from_numpy(s_tau))
        value = self.model(s_tau.float())
        #v_hat = float(value.data.numpy()[0])
        loss = alpha * 0.5 * (G - value).pow(2)
        loss.backward()
        self.optimizer.step()

        return None