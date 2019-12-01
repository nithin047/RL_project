# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 18:20:32 2019

@author: Saadallah
"""

import gym
from gym import spaces
from network import Network
import matplotlib.pyplot as plt
import numpy as np

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
        #defining limits for the observation space
        self.high = np.ones(k) * np.finfo(np.float32).max
        self.low = np.zeros(k)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        
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
        self.currentStep += 1
        
        reward = self.currentRate;
        done = self.currentStep == self.episodeLength
        obs = self.taggedUERates;
        
        return obs, reward, done, {}
        
    def __take_action__(self, action):
        # this function updates the currentRate attribute
        self.currentRate = self.taggedUERates[action];
        self.currentAction = action;
                
    
    def reset(self):
        # Reset the state of the environment to an initial state
        # Instantiate the network object
        
        # Generate network instance
        self.myNetwork.generateNetwork();
        # Train KNN model for BSs
        self.myNetwork.trainKNearestBSModel(self.k);
        
        # tagged user id is 1 w.l.o.g.
        self.taggedUEId = 0;
        
        # tagged user coordinates
        self.taggedCoord = self.myNetwork.UELocation[self.taggedUEId, :];
        
        # get list of k closest BSs from tagged user
        taggedUEKClosestBS = self.myNetwork.kClosestBS(self.taggedCoord[0], 
                                                       self.taggedCoord[1])[0];
        
        # compute capacities received from k closest BSs
        self.taggedUERates = np.zeros(self.k);
        for i in range(self.k):
            currentBSId = taggedUEKClosestBS[i];
            self.taggedUERates[i] = self.myNetwork.getRate(currentBSId, self.taggedCoord, 10, 3, 1e-17, 1);

        self.taggedUERates = np.random.permutation(self.taggedUERates)
        
        # set current step to 0
        self.currentStep = 0;
        
        # return initial state
        return self.taggedUERates;

    def getKClosestBS(self, UECoordinates):
        #returns a 2-D array, of size [k, 2]. The first row has rate values, and the second consists of unique BS IDs corresponding to those values. 

        taggedUEKClosestBS = self.myNetwork.kClosestBS(UECoordinates[0], UECoordinates[1])[0]
        taggedUERates = np.zeros(self.k);
        for i in range(self.k):
            currentBSId = taggedUEKClosestBS[i];
            taggedUERates[i] = self.myNetwork.getRate(currentBSId, UECoordinates, 10, 3, 1e-17, 1)

        return_array = []
        return_array.append(taggedUERates)
        return_array.append(taggedUEKClosestBS)

        return np.asarray(return_array)



    
    def render(self): #MODIFY THIS!
    # Render the environment to the screen
        myFig = self.myNetwork.showNetwork();
        plt.figure(myFig.number);
        plt.scatter(self.myNetwork.BSLocation[self.currentAction,0], self.myNetwork.BSLocation[self.currentAction,1], c='g', marker = '^', s=400);
        plt.scatter(self.myNetwork.UELocation[self.taggedUEId, 0], self.myNetwork.UELocation[self.taggedUEId,1], c='g', s=200);
