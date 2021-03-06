import gym
from gym import error, spaces, utils
from gym.utils import seeding
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import numpy.random as rn
import math


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
        self.UELocation = np.random.normal(np.sqrt(self.networkArea)/2, np.sqrt(self.networkArea)/10, (self.numberOfUE, 2))
        
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

        self.myNetwork.BSLoads = rn.randint(low = 1, high = 6, size = self.myNetwork.numberOfBS)
        
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

        self.randomPermutation = np.random.permutation(self.k);  
        #self.randomPermutation = np.array(range(self.k));  

        self.loadVector = self.myNetwork.BSLoads[self.taggedUEKClosestBS];
        
        self.taggedUERatesNonPermuted = np.copy(self.taggedUERates)

        self.taggedUERates = self.taggedUERates[self.randomPermutation];
        #self.loadVector = self.loadVector[self.randomPermutation];
        
        # set current step to 0
        self.currentStep = 0;
        
        # return initial state
        return np.concatenate((self.taggedUERates, np.transpose(self.loadVector)));

    def getKClosestBSSINRShared(self, UECoordinates):
        #returns a 2-D array, of size [3, k]. The first row has rate values, the second has the user loads for those BSs,
        #and the third consists of unique BS IDs corresponding to those values. 

        taggedUEKClosestBS = self.myNetwork.kClosestBS(UECoordinates[0], UECoordinates[1])[0]
        taggedUERates = np.zeros(self.k);
        taggedUEBSLoads = np.zeros(self.k);

        for i in range(self.k):
            currentBSId = taggedUEKClosestBS[i];
            taggedUERates[i] = self.myNetwork.getRate(currentBSId, UECoordinates, 10, 3, 1e-17, 1)
            taggedUEBSLoads[i] = self.myNetwork.BSLoads[taggedUEKClosestBS]

        return_array = []
        return_array.append(taggedUERates)
        return_array.append(taggedUEBSLoads)
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
