# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 22:36:06 2019

@author: Saadallah
"""

import gym
from myNetworkEnvironment import myNetworkEnvironment

if __name__ == "__main__":
    
    # define input parameters
    lambdaBS = 3e-6;
    lambdaUE = 3e-5;
    networkArea = 1e7;
    k = 10;
    episodeLength = 20;
    
    # create the environment
    env = myNetworkEnvironment(lambdaBS, lambdaUE, networkArea, k, episodeLength);
    
    for j in range(1000):
        obs = env.reset();
        for i in range(20):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            env.render()
            if(done):
                continue;
        
    env.close()
    