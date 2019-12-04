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
        action = np.argmax(state[0:self.k]/(state[self.k:2*self.k]+1));
        return action

def evaluate_and_plot_rate(env, model, nEpisodes):

    myMaxSINRPolicy = maxSINRPolicy(env.k);
    myMaxSharedRatePolicy = maxSharedRatePolicy(env.k);
    
    meanRateListMaxSINR = list(); # list storing mean rate per episode for max sinr policy
    meanRateListMaxSharedRate = list(); # list storing mean rate per episode for max sinr policy
    meanRateListRLPolicy = list(); # list storing mean rate per episode for learnt policy
    
    # start with max sinr policy
    for epi in trange(nEpisodes):
                
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
        
        # then max shared rate policy
        
        done = False;
        sumRate = 0;
        episodeLength = 0;
        state = env.reset();
        
        while (not done):
            a = myMaxSharedRatePolicy(state);
            state, reward, done, info = env.step(a);
            sumRate += reward;
            episodeLength += 1;
            
        meanRateListMaxSharedRate.append(sumRate/episodeLength);
        
        # then RL policy
        
        done = False;
        sumRate = 0;
        episodeLength = 0;
        state = env.reset();
        
        while (not done):

            state_torch = torch.from_numpy(state)
            torch_output = model(state_torch.float())
            action_probability_array = torch_output.data.numpy()

            a = np.random.choice(range(k), p=action_probability_array)
            state, reward, done, info = env.step(a);
            sumRate += reward;
            episodeLength += 1;
            
        meanRateListRLPolicy.append(sumRate/episodeLength);
    
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
    
    #plt.show()
    plt.savefig("histogram_log.eps")
    plt.close()
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
    
    #plt.show()
    plt.savefig("histogram.eps")
    plt.close()
    
    print("mean rate max sinr = ", np.mean(meanRateListMaxSINR))
    print("mean rate max shared rate = ", np.mean(meanRateListMaxSharedRate))
    print("mean rate RL = ", np.mean(meanRateListRLPolicy))

if __name__ == "__main__":

    model_name = str(sys.argv[1])

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
    
    model = torch.load(model_name)

    evaluate_and_plot_rate(env, model, 10000)