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
        self.n_h = 16
        self.n_out = num_actions
        self.alpha = alpha

        self.model = nn.Sequential(nn.Linear(self.n_in, self.n_h), nn.ReLU(), nn.Linear(self.n_h, self.n_h), nn.ReLU(), nn.Linear(self.n_h, self.n_h), nn.ReLU(), nn.Linear(self.n_h, self.n_out), nn.Softmax(dim=0))
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
        loss = - gamma_t * delta * torch.log(pi_A_t)
        loss.backward()
        self.optimizer.step()
        
class maxSINRPolicy():
    #this class defines the greedy max SINR policy
    def __init__(self, k):
        self.k = k;

    def __call__(self, state) -> int:
        # assume state space is only the SINR from k closest BS
        action = np.argmax(state[0:self.k]);
        return action;
    
class maxSharedRatePolicy():
    #this class defines the greedy max SINR policy
    def __init__(self, k):
        self.k = k

    def __call__(self, state) -> int:
        # assume state space is only the SINR from k closest BS
        action = np.argmax(state[0:self.k]);
        return action

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

def evaluatePolicyPerformance(env, RLPolicy, nEpisodes):
    
    myMaxSINRPolicy = maxSINRPolicy(env.k);
    myMaxSharedRatePolicy = maxSharedRatePolicy(env.k);
    
    meanRateListMaxSINR = list(); # list storing mean rate per episode for max sinr policy
    meanRateListMaxSharedRate = list(); # list storing mean rate per episode for max sinr policy
    meanRateListRLPolicy = list(); # list storing mean rate per episode for learnt policy
    
    # start with max sinr policy
    for epi in range(nEpisodes):
                
        # start with max SINR policy
        
        done = False;
        sumRate = 0;
        episodeLength = 0;
        state = env.reset();
        
        while (not done):
            a = myMaxSINRPolicy(state);
            state, reward, done, info = env.step(a);
            sumRate += reward;
            episodeLength += 1;
            
        meanRateListMaxSINR.append(sumRate/episodeLength);
        #print('a')
        # then max shared rate policy
        
        done = False;
        sumRate = 0;
        episodeLength = 0;
        state = env.repeat();
        
        while (not done):
            a = myMaxSharedRatePolicy(state);
            state, reward, done, info = env.step(a);
            sumRate += reward;
            episodeLength += 1;
            
        meanRateListMaxSharedRate.append(sumRate/episodeLength);
        #print('b')
        # then RL policy
        
        done = False;
        sumRate = 0;
        episodeLength = 0;
        state= env.repeat();
        
        while (not done):
            a = RLPolicy(state, env.k);
            state, reward, done, info = env.step(a);
            sumRate += reward;
            episodeLength += 1;
            
        meanRateListRLPolicy.append(sumRate/episodeLength);
        #print('c')
    
    # plot results    
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.hist(np.log(meanRateListMaxSINR), density=True, histtype='step', bins = 'auto', cumulative=True, linewidth = 4)
    ax.hist(np.log(meanRateListMaxSharedRate), density=True, histtype='step', bins = 'auto', cumulative=True, linewidth = 4)
    ax.hist(np.log(meanRateListRLPolicy), density=True, histtype='step', bins = 'auto', cumulative=True, linewidth = 4)
    
    # tidy up the figure
    ax.grid(True)
    ax.legend(('max SINR', 'max shared rate', 'RL'), loc='right')
    ax.set_title('Average Rate Per Episode Histogram')
    ax.set_xlabel('Average Rate (bps)')
    ax.set_ylabel('Likelihood of occurrence')
    
    plt.show()
    
    # plot results    
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.hist(meanRateListMaxSINR, density=True, histtype='step', bins = 'auto', cumulative=True, linewidth = 4)
    ax.hist(meanRateListMaxSharedRate, density=True, histtype='step', bins = 'auto', cumulative=True, linewidth = 4)
    ax.hist(meanRateListRLPolicy, density=True, histtype='step', bins = 'auto', cumulative=True, linewidth = 4)
    
    # tidy up the figure
    ax.grid(True)
    ax.legend(('max SINR', 'max shared rate', 'RL'), loc='right')
    ax.set_title('Average Rate Per Episode Histogram')
    ax.set_xlabel('Average Rate (bps)')
    ax.set_ylabel('Likelihood of occurrence')
    
    plt.show()
    
    print("mean rate max sinr = ", np.mean(meanRateListMaxSINR))
    print("mean rate max shared rate = ", np.mean(meanRateListMaxSharedRate))
    print("mean rate RL = ", np.mean(meanRateListRLPolicy))
    
        

if __name__ == "__main__":

    lambdaBS = 3e-6;
    lambdaUE = 0.8e-5; # 1e-5
    networkArea = 2e7;
    k = 5;
    
    # be careful to choose the parameters below carefully
    # e.g. I want the UE to experience roughly 4 handoffs per episode
    # the velocity is 20meters/s = 76 km/h. If the BS density is 3BS/km^2, 
    # then the cell diameter is roughly sqrt(1/3e-6) ~= 600m
    # the episode should then cover around 2400m, or 120s given the velocity
    # if the episode length is of 60 steps, then deltaT should be 2s.
    episodeLength = 10; # 60 steps
    handoffDuration = 0; # 2 steps
    velocity = 0; # 20 meters per second
    deltaT = 2;

    #create the environment
    env = myNetworkEnvironment(lambdaBS, lambdaUE, networkArea, k, handoffDuration, velocity, deltaT, episodeLength)
    gamma = 1.
    stepSize = 3e-4

    pi = PiApproximationWithNN(
    env.k,
    env.action_space.n,
    stepSize)

    B = Baseline(30.)

    G = REINFORCE(env, gamma, 5000, pi,B)
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
    #aa = obs.data.numpy()[0:k];
    #bb = obs.data.numpy()[k:2*k];
    #print("Probability 1 should be at index: ", np.argmax(aa/(bb+1)));
    
    evaluatePolicyPerformance(env, pi, 800);

