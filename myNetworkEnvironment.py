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
from sklearn.neighbors import NearestNeighbors

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
        self.high = np.ones(3*k+1) * np.finfo(np.float32).max
        self.low = np.zeros(3*k+1)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        
        # Create an empty network object
        self.myNetwork = Network(lambdaBS, lambdaUE, networkArea, handoffDuration, velocity, deltaT);
        
        # save input variables
        self.lambdaBS = lambdaBS;
        self.lambdaUE = lambdaUE;
        self.networkArea = networkArea;
        self.k = k;
        self.episodeLength = episodeLength;
        
        # tagged user id is 0 w.l.o.g.
        self.taggedUEId = 0;
        
        self.velocity = velocity;
        self.deltaT = deltaT;
        
    def step(self, action):
        # Execute one time step within the environment
        self.__take_action__(action)
        self.currentStep += 1;
        
        reward = self.currentRate;
        done = self.currentStep == self.episodeLength
        obs = np.concatenate((self.taggedUERates, np.transpose(self.loadVector), self.mobilityFeatures, self.associatedBSFeature));
        
        return obs, reward, done, {}
        
    def __take_action__(self, action):
        # this function updates the currentRate attribute
        
        # Step 1: figure out the reward
        # set current action as the BS serving the tagged UE
        self.myNetwork.setCurrentBS(self.taggedUEKClosestBS[action]); 
        
        if (self.myNetwork.isRateZero()):
            self.currentRate = 0;
        else:
            self.currentRate = self.taggedUERates[action]/(self.loadVector[action]+1);

        self.currentAction = action;
        
        # Step 2: figure out the next observation
        
        self.myNetwork.stepForward(self.taggedUEId);
        self.taggedCoord = self.myNetwork.UELocation[self.taggedUEId, :];
        
        # get list of k closest BSs from tagged user
        self.taggedUEKClosestBS = self.myNetwork.kClosestBS(self.taggedCoord[0], 
                                                       self.taggedCoord[1])[0];
                                                            
        # compute capacities received from k closest BSs
        self.taggedUERates = np.zeros(self.k);
        for i in range(self.k):
            self.taggedUERates[i] = self.myNetwork.getRate(self.taggedUEKClosestBS[i], self.taggedCoord, 10, 3, 1e-17, 1);
            
        # get the loads of the k closest BSs
        self.loadVector = self.myNetwork.BSLoads[self.taggedUEKClosestBS];
        
        # get mobility features
        self.mobilityFeatures = np.zeros(self.k);
        self.stepsLookahead = 10;
        
        localVoronoiModel = NearestNeighbors(n_neighbors=1);
        localVoronoiModel.fit(self.myNetwork.BSLocation[self.taggedUEKClosestBS, :]); 
        discount = 0.95;
        
        for i in range(1, self.stepsLookahead+1):
            futureLocation = self.myNetwork.getMobilityTrace(self.taggedUEId, i);
            dist, ind = localVoronoiModel.kneighbors(futureLocation);
            self.mobilityFeatures[ind] += 1*discount**(i-1);
                
        # associated BS feature
        self.associatedBSFeature = np.where(self.taggedUEKClosestBS == self.myNetwork.getCurrentBS())[0];
        
        # just in case, but should (almost) never happen
        if (self.associatedBSFeature.size == 0):
            self.associatedBSFeature = np.array([self.k]); 
        else:
            self.associatedBSFeature = np.array([self.associatedBSFeature[0]]);
                
        

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
        self.taggedCoordInit = self.myNetwork.UELocation[self.taggedUEId, :];
        
        # get list of k closest BSs from tagged user
        self.taggedUEKClosestBS = self.myNetwork.kClosestBS(self.taggedCoord[0], 
                                                       self.taggedCoord[1])[0];
        
        # compute capacities received from k closest BSs
        self.taggedUERates = np.zeros(self.k);
        for i in range(self.k):
            self.taggedUERates[i] = self.myNetwork.getRate(self.taggedUEKClosestBS[i], self.taggedCoord, 10, 3, 1e-17, 1);

        self.taggedUERatesNonPermuted = np.copy(self.taggedUERates)
        
        # get loads of k closest BSs
        self.loadVector = self.myNetwork.BSLoads[self.taggedUEKClosestBS];
        
        # get mobility features
        self.mobilityFeatures = np.zeros(self.k);
        self.stepsLookahead = 10;
        
        localVoronoiModel = NearestNeighbors(n_neighbors=1);
        localVoronoiModel.fit(self.myNetwork.BSLocation[self.taggedUEKClosestBS, :]); 
        discount = 0.95;
        
        for i in range(1, self.stepsLookahead+1):
            futureLocation = self.myNetwork.getMobilityTrace(self.taggedUEId, i);
            dist, ind = localVoronoiModel.kneighbors(futureLocation);
            self.mobilityFeatures[ind] += 1*discount**(i-1);

        # set current step to 0
        self.currentStep = 0;
        
        # set initial BS
        self.myNetwork.setCurrentBS(self.taggedUEKClosestBS[0]);
        self.myNetwork.timeSinceLastHandoff = self.myNetwork.handoffDuration; # to avoid getting 0 rate at first
        
        # associated BS feature
        self.associatedBSFeature = np.array([0]);
        
        # return initial state
        return np.concatenate((self.taggedUERates, np.transpose(self.loadVector), self.mobilityFeatures, self.associatedBSFeature));
    

    def getKClosestBSSINR(self, UECoordinates):
        #returns a 2-D array, of size [2, k]. The first row has rate values, and the second consists of unique BS IDs corresponding to those values. 

        taggedUEKClosestBS = self.myNetwork.kClosestBS(UECoordinates[0], UECoordinates[1])[0]
        taggedUERates = np.zeros(self.k);
        for i in range(self.k):
            taggedUERates[i] = self.myNetwork.getRate(self.taggedUEKClosestBS[i], UECoordinates, 10, 3, 1e-17, 1)

        return_array = []
        return_array.append(taggedUERates)
        return_array.append(taggedUEKClosestBS)

        return np.asarray(return_array)

    
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
            self.taggedUERates[i] = self.myNetwork.getRate(self.taggedUEKClosestBS[i], self.taggedCoord, 10, 3, 1e-17, 1);
                
        # set current step to 0
        self.currentStep = 0;
        
        # get loads of k closest BSs
        self.loadVector = self.myNetwork.BSLoads[self.taggedUEKClosestBS];
        
        # get mobility features
        self.mobilityFeatures = np.zeros(self.k);
        self.stepsLookahead = 10;
        
        localVoronoiModel = NearestNeighbors(n_neighbors=1);
        localVoronoiModel.fit(self.myNetwork.BSLocation[self.taggedUEKClosestBS, :]); 
        discount = 0.95;
        
        for i in range(1, self.stepsLookahead+1):
            futureLocation = self.myNetwork.getMobilityTrace(self.taggedUEId, i);
            dist, ind = localVoronoiModel.kneighbors(futureLocation);
            self.mobilityFeatures[ind] += 1*discount**(i-1);

        # set current step to 0
        self.currentStep = 0;
        
        # set initial BS
        self.myNetwork.setCurrentBS(self.taggedUEKClosestBS[0]);
        self.myNetwork.timeSinceLastHandoff = self.myNetwork.handoffDuration; # to avoid getting 0 rate at first
        
        # associated BS feature
        self.associatedBSFeature = np.array([0]);
        
        # return initial state
        return np.concatenate((self.taggedUERates, np.transpose(self.loadVector), self.mobilityFeatures, self.associatedBSFeature));
    
    
    def render(self): #MODIFY THIS!
    # Render the environment to the screen
        myFig = self.myNetwork.showNetwork();
        plt.figure(myFig.number);
        plt.scatter(self.myNetwork.BSLocation[self.currentAction,0], self.myNetwork.BSLocation[self.currentAction,1], c='g', marker = '^', s=400);
        plt.scatter(self.myNetwork.UELocation[self.taggedUEId, 0], self.myNetwork.UELocation[self.taggedUEId,1], c='g', s=200);
