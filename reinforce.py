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
        self.n_in = state_dims
        self.n_h = 32
        self.n_out = num_actions
        self.alpha = alpha

        self.model = nn.Sequential(nn.Linear(self.n_in, self.n_h), nn.ReLU(), nn.Linear(self.n_h, self.n_h), nn.ReLU(), nn.Linear(self.n_h, self.n_out), nn.Softmax(dim=0))
        self.model = self.model.float()
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.alpha)



    def __call__(self, s, k) -> int:
        #returns an action sampled according to the probability distribution that the policy network generates
        s = torch.from_numpy(s)
        value = self.model(s.float())
        prob_array = value.data.numpy()
        action_array = range(k)

        action = np.random.choice(action_array, p=prob_array)

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
    #naive implementation of a zero baseline. Using the state values as baseline resulted in very unstable policies, so I think
    #we should keep a zero baseline.
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
    V_list = []

    for _ in trange(num_episodes):
        S = []
        R = []
        A = []
        state, r, done = env.reset(), 0., False
        S.append(state)
        #V_list.append(V(state))
        #pi(state)
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

            #V.update(S[ii], G_t)
            pi.update(S[ii], A[ii], gamma**ii, G_t - V(S[ii]))

    return G_list

if __name__ == "__main__":

    lambdaBS = 3e-6;
    lambdaUE = 3e-5;
    networkArea = 1e7;
    k = 10;
    episodeLength = 20;

    #create the environment
    env = myNetworkEnvironment(lambdaBS, lambdaUE, networkArea, k, episodeLength)
    gamma = 1.
    alpha = 3e-4

    pi = PiApproximationWithNN(
    env.k,
    env.action_space.n,
    alpha)

    B = Baseline(0.)

    G = REINFORCE(env, gamma, 50000, pi,B)
    obs = env.reset()
    print(obs)
    print("Position of max SINR in SNR array")
    print(np.argmax(obs))
    obs = torch.from_numpy(obs)
    value = pi.model(obs.float())
    print("probability vector")
    print(value.data.numpy())
    # plt.plot(G)
    # plt.savefig("help.eps")

