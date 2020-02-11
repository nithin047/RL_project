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

class PiApproximationWithNN():
    #this class defines the neural network that approximates the policy
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha = 3e-4):
        
        #initializing the neural network
        
        # input layer size equals state space size
        self.n_in = state_dims 
        # each hidden layer has 32 neurons
        self.n_h = 32 
        # output layer has a neuron for each possible action
        self.n_out = num_actions 
        # NN learning rate
        self.alpha = alpha 

        # construct the NN
        self.model = nn.Sequential(nn.Linear(self.n_in, self.n_h), nn.ReLU(),
                                   nn.Linear(self.n_h, self.n_h), nn.ReLU(), 
                                   nn.Linear(self.n_h, self.n_out), 
                                   nn.Softmax(dim=0))
        # convert pytorch NN parameters to float
        self.model = self.model.float()
        
        # set the solver
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.alpha)



    def __call__(self, s, k) -> int:
        # this function returns an action sampled according to the probability 
        # distribution that the policy network generates
        
        # convert input to pytorch object
        s = torch.from_numpy(s)
        # convert pytorch object parameters to float
        value = self.model(s.float())
        # retrieve the probability vector, output of the NN softmax layer
        prob_array = value.data.numpy()
        
        # choose action wrt probability vector
        action = np.random.choice(range(k), p=prob_array)

        return action


    def update(self, s, a, gamma_t, delta):
        #carries out the optimization update to the policy network
        self.optimizer.zero_grad()
        s = (torch.from_numpy(s))
        pi = self.model(s.float())
        pi_A_t = pi[a]
        loss = - self.alpha * gamma_t * delta * torch.log(pi_A_t)
        loss.backward()
        self.optimizer.step()

class Baseline(object):
    # Naive implementation of a fixed baseline. 
    # Using the state values as baseline resulted in very unstable policies, 
    # so I think we should keep a zero baseline.
    def __init__(self,b):
        self.b = b

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        pass

def REINFORCE(
env, #open-ai environment
gamma:float,
num_episodes:int,
pi:PiApproximationWithNN,
V:Baseline) -> Iterable[float]:

    G_list = []

    for _ in trange(num_episodes):
        S = []
        R = []
        A = []
        state, r, done = env.reset(), 0., False
        S.append(state)

        while(True):
            a = pi(state, env.k)
            A.append(a)
            next_state, r, done, info = env.step(a)
            R.append(r)
            if done == True:
                break
            else:
                S.append(next_state)
                state = next_state

        for ii in range(len(S)):
            G_t = 0
            for jj in range(ii, len(S)):
                G_t = G_t + gamma**(jj - ii) * R[jj]
            if ii == 0:
                    G_list.append(G_t)

            pi.update(S[ii], A[ii], gamma**ii, G_t - V(S[ii]))

    return G_list

def train_and_save():
    lambdaBS = 3e-6;
    lambdaUE = 3e-5;
    networkArea = 1e8;
    k = 6;
    episodeLength = 5;

    #create the environment
    env = myNetworkEnvironment(lambdaBS, lambdaUE, networkArea, k, episodeLength)
    gamma = 1.
    alpha = 3e-4

    pi = PiApproximationWithNN(
    env.k,
    env.action_space.n,
    alpha)

    # set baseline to be 0
    B = Baseline(0.)

    G = REINFORCE(env, gamma, 100000, pi,B)
    
    #saving the trained policy network
    torch.save(pi.model, "policy_network.pt")

if __name__ == "__main__":

    train_and_save()


