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
        
    def __init__(self, lambdaBS, lambdaUE, networkArea, k, handoffDuration, velocity, deltaT, episodeLength):
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
        self.observation_space = spaces.Discrete(k); #DOUBLE CHECK THIS
        
        # Create an empty network object
        self.myNetwork = Network(lambdaBS, lambdaUE, networkArea, handoffDuration, velocity, deltaT);
        
        # save input variables
        self.lambdaBS = lambdaBS;
        self.lambdaUE = lambdaUE;
        self.networkArea = networkArea;
        self.k = k;
        self.episodeLength = episodeLength;
        
        # tagged user id is 1 w.l.o.g.
        self.taggedUEId = 0;
        
        
    def step(self, action):
        # Execute one time step within the environment
        self.__take_action__(action)
        self.currentStep += 1;
        
        reward = self.currentRate;
        done = self.currentStep == self.episodeLength
        
        loadVector = self.myNetwork.BSLoads[self.taggedUEKClosestBS];
        loadVector = loadVector[self.randomPermutation];
        
        obs = np.concatenate((self.taggedUERates[self.randomPermutation], np.transpose(loadVector)));
        
        return obs, reward, done, {}
        
    def __take_action__(self, action):
        # this function updates the currentRate attribute
        
        # set current action as the BS serving the tagged UE
        self.myNetwork.setCurrentBS(self.taggedUEKClosestBS[self.randomPermutation[action]]); 
        
        if (self.myNetwork.isRateZero()):
            self.currentRate = 0;
        else:
            self.currentRate = self.taggedUERates[self.randomPermutation[action]]/(self.myNetwork.BSLoads[self.taggedUEKClosestBS[self.randomPermutation[action]]]+1);

        self.currentAction = self.randomPermutation[action];
        self.myNetwork.stepForward(self.taggedUEId);
        self.taggedCoord = self.myNetwork.UELocation[self.taggedUEId, :];
        
        # get list of k closest BSs from tagged user
        self.taggedUEKClosestBS = self.myNetwork.kClosestBS(self.taggedCoord[0], 
                                                       self.taggedCoord[1])[0];
                                                            
        # compute capacities received from k closest BSs
        self.taggedUERates = np.zeros(self.k);
        for i in range(self.k):
            currentBSId = self.taggedUEKClosestBS[i];
            self.taggedUERates[i] = self.myNetwork.getRate(currentBSId, self.taggedCoord, 10, 3, 1e-17, 100);
            
    
    def reset(self):
        # Reset the state of the environment to an initial state
        # Instantiate the network object
        
        # Generate network instance
        self.myNetwork.generateNetwork();
        # Train KNN model for BSs
        self.myNetwork.trainKNearestBSModel(self.k);
        
        # tagged user coordinates
        self.taggedCoord = self.myNetwork.UELocation[self.taggedUEId, :];
        self.taggedCoordInit = self.myNetwork.UELocation[self.taggedUEId, :];
        
        # get list of k closest BSs from tagged user
        self.taggedUEKClosestBS = self.myNetwork.kClosestBS(self.taggedCoord[0], 
                                                       self.taggedCoord[1])[0];
        
        # compute capacities received from k closest BSs
        self.taggedUERates = np.zeros(self.k);
        for i in range(self.k):
            currentBSId = self.taggedUEKClosestBS[i];
            self.taggedUERates[i] = self.myNetwork.getRate(currentBSId, self.taggedCoord, 10, 3, 1e-17, 100);
        
        #rng_state = np.random.get_state()
        #np.random.shuffle(self.taggedUERates)
        #np.random.set_state(rng_state)
        #np.random.shuffle(self.myNetwork.BSLoads)

        self.randomPermutation = np.random.permutation(self.k); #np.array(range(self.k));
        self.randomPermutation = np.array(range(self.k));
        # set current step to 0
        self.currentStep = 0;
        
        loadVector = self.myNetwork.BSLoads[self.taggedUEKClosestBS];
        loadVector = loadVector[self.randomPermutation];
        
        # return initial state
        return np.concatenate((self.taggedUERates[self.randomPermutation], np.transpose(loadVector)));
    
    def repeat(self):
        # Reset the state of the environment to last initial state
        # Uses last network object
        
        self.myNetwork.timeSinceLastHandoff = self.myNetwork.handoffDuration + 1; # time the last handoff started 
        self.myNetwork.currentBS = -1;
        
        # tagged user coordinates
        self.taggedCoord = self.taggedCoordInit
        
        # get list of k closest BSs from tagged user
        self.taggedUEKClosestBS = self.myNetwork.kClosestBS(self.taggedCoord[0], 
                                                       self.taggedCoord[1])[0];
        
        # compute capacities received from k closest BSs
        self.taggedUERates = np.zeros(self.k);
        for i in range(self.k):
            currentBSId = self.taggedUEKClosestBS[i];
            self.taggedUERates[i] = self.myNetwork.getRate(currentBSId, self.taggedCoord, 10, 3, 1e-17, 100);
                
        # set current step to 0
        self.currentStep = 0;
        
        loadVector = self.myNetwork.BSLoads[self.taggedUEKClosestBS];
        loadVector = loadVector[self.randomPermutation];
        
        # return initial state
        return np.concatenate((self.taggedUERates[self.randomPermutation], np.transpose(loadVector)));
    
    def render(self): #MODIFY THIS!
    # Render the environment to the screen
        myFig = self.myNetwork.showNetwork();
        plt.figure(myFig.number);
        plt.scatter(self.myNetwork.BSLocation[self.currentAction,0], self.myNetwork.BSLocation[self.currentAction,1], c='g', marker = '^', s=400);
        plt.scatter(self.myNetwork.UELocation[self.taggedUEId, 0], self.myNetwork.UELocation[self.taggedUEId,1], c='g', s=200);
