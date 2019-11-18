# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 18:20:32 2019

@author: Saadallah
"""

import gym
from gym import spaces
from network import Network
import matplotlib.pyplot as plt

class myNetworkEnvironment(gym.Env):
    # Custom environment for our network simulator
        
    def __init__(self, lambdaBS, lambdaUE, networkArea, k, episodeLength):
        # Constructor function
        # k: number of closest BSs to consider in the action space
        
        super(myNetworkEnvironment, self).__init__();
        
        
        # Define a discrete action space
        # k possible actions, where action 1 corresponds to closest BS, action
        # 2 corresponds to the second closest BS, ..., action k --> k closest
        # BS
        
        self.action_space = spaces.Discrete(k);
        
        # Define State Space, or observation space
        # There are k features in state space, where feature i corresponds to
        # the capacity received by BS i
        self.observation_space = spaces.Discrete(k);
        
        # Create an empty network object
        self.myNetwork = Network(lambdaBS, lambdaUE, networkArea);
        
        # save input variables
        self.lambdaBS = lambdaBS;
        self.lambdaUE = lambdaUE;
        self.networkArea = networkArea;
        self.k = k;
        self.episodeLength = episodeLength;
        
        
    def step(self, action):
        # Execute one time step within the environment
        self.__take_action__(action)
        self.current_step += 1
        
        reward = self.currentRate;
        done = self.current_step == self.episodeLength-1
        obs = self.taggedUERates;
        
    def __take_action__(self, action):
        # this function updates the currentRate attribute
        self.currentRate = self.taggedUERates[action];
                
    
    def reset(self):
        # Reset the state of the environment to an initial state
        # Instantiate the network object
        
        # Generate network instance
        self.myNetwork.generateNetwork(lambdaBS, lambdaUE, networkArea);
        # Train KNN model for BSs
        self.myNetwork.trainKNearestBSModel(self.k);
        
        # tagged user id is 1 w.l.o.g.
        self.taggedUEId = 1;
        
        # tagged user coordinates
        self.taggedCoord = self.myNetwork.UELocation[self.taggedUEId, :];
        
        # get list of k closest BSs from tagged user
        taggedUEKClosestBS = self.myNetwork.kClosestBS(self.taggedCoord[0], 
                                                       self.taggedCoord[1]);
        
        # compute capacities received from k closest BSs
        self.taggedUERates = np.zeros((self.k, 1));
        for i in range(self.k):
            currentBSId = taggedUEKClosestBS[i];
            self.taggedUERates[i] = self.myNetwork.getRate(currentBSId, self.taggedCoord, 10, 3, 1e-17, 1e8);
        
        # set current step to 0
        self.currentStep = 0;
        
        # return initial state
        return self.taggedUERates;
    
    
    def render(self, mode='human', close=False):
    # Render the environment to the screen
        myFig = self.myNetwork.showNetwork();
        plt.figure(myFig.number);
        