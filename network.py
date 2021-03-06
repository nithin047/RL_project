# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 11:38:42 2019

@author: Saadallah
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import math

from scipy.spatial import Voronoi, voronoi_plot_2d


class Network(object):
    def __init__(self, lambdaBS, lambdaUE, networkArea, handoffDuration, velocity, deltaT):
        self.lambdaBS = lambdaBS; # intensity of base stations in BS/m^2
        self.lambdaUE = lambdaUE;
        self.networkArea = networkArea; # area of the network to be simulated
        
        self.k = 0;  # number of nearest BSs of interest
        
        self.numberOfBS = 0; # number of base stations in the network
        self.BSLocation = np.zeros((1,2)); # X-Y coordinates of BSs in network
        
        self.numberOfUE = 0; # number of UEs in the network
        self.UELocation = np.zeros((1,2)); # X-Y coordinates of UEs in network
        self.UEMotionDirection = np.zeros((1,1)); # angles from 0-2pi of motion directions 
        
        self.handoffDuration = handoffDuration; # how many time slots service is lost
        
        self.timeSinceLastHandoff = handoffDuration + 1; # time the last handoff started 
        self.currentBS = -1;
        self.velocity = velocity;
        self.deltaT = deltaT;

    def generateNetwork(self): 
        # this functions places the BSs in the network + places the UEs in the 
        # network and determines their motion direction
        
        # Determine number of BSs to be placed in network
        self.numberOfBS = np.random.poisson(self.lambdaBS * self.networkArea); 
        
        # Determine their locations
        self.BSLocation = np.random.rand(self.numberOfBS, 2)*np.sqrt(self.networkArea);
        
        # Determine number of UEs to be placed in network
        # There is at least 1 UE in the entwork, namely the tagged UE
        # tagged UE is UE with index 1
        self.numberOfUE = np.random.poisson(self.lambdaUE * self.networkArea) + 1;
        
        # Determine their locations
        self.UELocation = np.random.rand(self.numberOfUE, 2)*np.sqrt(self.networkArea);
        
        uelocs1 = np.random.normal(np.sqrt(self.networkArea)*0.6, np.sqrt(self.networkArea)/4, (math.ceil(self.numberOfUE/2), 2))
        uelocs2 = np.random.normal(np.sqrt(self.networkArea)*0.25, np.sqrt(self.networkArea)/10, (math.floor(self.numberOfUE/2), 2))
        self.UELocation = np.concatenate((uelocs1, uelocs2), axis=0)
        
        # Determine their direction of motion
        # angles from 0-2pi of motion directions 
        self.UEMotionDirection = np.random.rand(self.numberOfUE, 1)*2*np.pi; 
        
        # Determine load of each BS
        self.BSLoads = np.zeros(self.numberOfBS) # load of each BS
        self.getVoronoiLoads();
        
    def trainKNearestBSModel(self, k): 
        # this function trains the KNN model on BS locations data
        # call this function (once) before calling kClosestBS function
        self.k = k;
        self.neighborsModel = NearestNeighbors(n_neighbors=self.k);
        self.neighborsModel.fit(self.BSLocation); 
                

    def kClosestBS(self, x, y):
        # returns BS id of k nearest BSs of point X-Y
        
        dist, ind = self.neighborsModel.kneighbors(np.array([x, y]).reshape(1, -1))
        
        return ind
    
    def getVoronoiLoads(self):
        # returns loads of each BS assuming Voronoi association
        
        voronoiModel = NearestNeighbors(n_neighbors=1);
        voronoiModel.fit(self.BSLocation); 
        
        # the loads no NOT include the tagged UE!!
        for i in range(1, self.numberOfUE):
            dist, ind = voronoiModel.kneighbors(self.UELocation[i, :].reshape(1, -1));
            self.BSLoads[ind] += 1;
        
        # set current base station attribute, i.e. BS tagged UE is connected to
        dist, ind = voronoiModel.kneighbors(self.UELocation[0, :].reshape(1, -1));
        self.currentBS = ind;

    
    def showNetwork(self):
        myFig = plt.figure(figsize=(15,15))
        plt.scatter(self.BSLocation[:,0], self.BSLocation[:,1], c='b', marker = '^', s=200)
        plt.scatter(self.UELocation[:,0], self.UELocation[:,1], c='r', s=50)
        plt.xlim((0, np.sqrt(self.networkArea)))
        plt.ylim((0, np.sqrt(self.networkArea)))
        plt.legend(('Base Stations', 'UEs'), loc = 1, fontsize = 'xx-large')
        plt.show()
        
        return myFig;
        
    def getRate(self, BSid, UECoordinates, PTx, alpha, N0, B):
        # this function returns rate seen by a UE located at location
        # UECoordinates if it were to connect to BS BSid
        
        # distances from selected UE to all others BSs in network
        dists = np.linalg.norm(self.BSLocation-UECoordinates, 2, axis = 1);
        
        # total received power at selected UE, Signal + Interference
        totalRxPower = np.sum(PTx*(dists**(-alpha)));
        
        # Signal power from BS BSid, capped at 1
        RxSignalPower = PTx*np.min([(dists[BSid]**(-alpha)), 1]);
        
        # Interference Power
        RxInterferencePower = totalRxPower - RxSignalPower;
        
        # Compute SINR
        SINR = RxSignalPower/(N0*B +  RxInterferencePower);
        
        # Capacity: Shannon Ergodic Rate
        capacity = B*np.log2(1+SINR);

        return capacity
    
    def getMobilityTrace(self, UEid, steps = 1):
        # This function returns the location of the UE UEid after deltaT time
        # assuming constant velocity v, and fixed environment, i.e., other 
        # UEs are NOT moving; single step by default
        
        # get initial location of UE UEid
        initUELoc = self.UELocation[UEid, :];
        distanceTravelled = self.velocity*steps*self.deltaT;
        theta = self.UEMotionDirection[UEid];
        
        displacementVector = np.transpose(np.array([np.cos(theta), np.sin(theta)]))*distanceTravelled;
        finalLocation = (initUELoc + displacementVector)%(np.sqrt(self.networkArea));
        
        return finalLocation;
    
    def stepForward(self, UEid):
        # updates location of UE UEid
        self.UELocation[UEid, :] = self.getMobilityTrace(UEid, 1);
        
    def setCurrentBS(self, BSid):
        # This function sets the current BS the tagged UE is connected to
        
        if (self.currentBS == BSid):
            # do nothing, but update time since last handoff
            self.timeSinceLastHandoff +=1;
        else:
            # reset time since last handoff
            self.timeSinceLastHandoff = 0;
            self.currentBS = BSid;
            
    def isRateZero(self):
        # Returns true if the tagged UE currently receives 0 rate
        
        if (self.timeSinceLastHandoff < self.handoffDuration):
            return True;
        else:
            return False;
        
    def getCurrentBS(self):
        return self.currentBS;

def generateFigurePresentation():

    numberOfBS = 5; 
    numberOfUE = 30;
    networkArea = 1e6
    BSLocation = np.random.rand(numberOfBS, 2)*np.sqrt(networkArea);
    UELocation = np.random.rand(numberOfUE, 2)*np.sqrt(networkArea);
    
    
    plt.figure(figsize=(20,20));
    plt.scatter(BSLocation[:,0], BSLocation[:,1], c='b', marker = '^', s=500)
    plt.xlim((0, np.sqrt(networkArea)))
    plt.ylim((0, np.sqrt(networkArea)))
    plt.legend(("Base Stations",), fontsize = 'xx-large')
    plt.show()
    
    fig = plt.figure(figsize=(20,20));
    plt.scatter(BSLocation[:,0], BSLocation[:,1], c='b', marker = '^', s=500)
    plt.scatter(UELocation[0,0], UELocation[0,1], c='r', s=100)
    plt.xlim((0, np.sqrt(networkArea)))
    plt.ylim((0, np.sqrt(networkArea)))
    plt.legend(('Base Stations', 'UEs'), loc = 1, fontsize = 'xx-large')
    plt.show()
    
    vor = Voronoi(BSLocation)
    fig = plt.figure(figsize=(20,20));
    plt.scatter(BSLocation[:,0], BSLocation[:,1], c='b', marker = '^', s=500)
    plt.scatter(UELocation[0,0], UELocation[0,1], c='r', s=100)
    voronoi_plot_2d(vor, fig.gca(), show_points=False, show_vertices = False)
    plt.legend(('Base Stations', 'UEs'), loc = 1, fontsize = 'xx-large')
    plt.xlim((0, np.sqrt(networkArea)))
    plt.ylim((0, np.sqrt(networkArea)))
    plt.show()
    
    fig = plt.figure(figsize=(20,20));
    plt.scatter(BSLocation[:,0], BSLocation[:,1], c='b', marker = '^', s=500)
    plt.scatter(UELocation[:,0], UELocation[:,1], c='r', s=100)
    plt.xlim((0, np.sqrt(networkArea)))
    plt.ylim((0, np.sqrt(networkArea)))
    plt.legend(('Base Stations', 'UEs'), loc = 1, fontsize = 'xx-large')
    plt.show()
    
    fig = plt.figure(figsize=(20,20));
    plt.scatter(BSLocation[:,0], BSLocation[:,1], c='b', marker = '^', s=500)
    plt.scatter(UELocation[:,0], UELocation[:,1], c='r', s=100)
    voronoi_plot_2d(vor, fig.gca(), show_points=False, show_vertices = False)
    plt.legend(('Base Stations', 'UEs'), loc = 1, fontsize = 'xx-large')
    plt.xlim((0, np.sqrt(networkArea)))
    plt.ylim((0, np.sqrt(networkArea)))
    plt.show()
    
if __name__ == "__main__":
    myNetwork = Network(3e-6, 3e-5, 1e7, 2, 0,0);
    myNetwork.generateNetwork();
    myNetwork.showNetwork();
    myNetwork.trainKNearestBSModel(3);
    print(myNetwork.kClosestBS(1000, 1000))
    
    print("Current tagged UE location = ", myNetwork.UELocation[1, :]);
    print("Current tagged UE rate = ", myNetwork.getRate(3, myNetwork.UELocation[1, :], 10, 3, 1e-17, 1e8));
    
    print("Future tagged UE location = ", myNetwork.getMobilityTrace(1, 1, 10));
    print("Future tagged UE rate = ", myNetwork.getRate(3, myNetwork.getMobilityTrace(1, 1, 10), 10, 3, 1e-17, 1e8));
    
    #generateFigurePresentation();
