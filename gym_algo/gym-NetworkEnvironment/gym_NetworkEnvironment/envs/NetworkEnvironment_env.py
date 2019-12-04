import gym
from gym import error, spaces, utils
from gym.utils import seeding
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

class Network(object):
    def __init__(self, lambdaBS, lambdaUE, networkArea):
        self.lambdaBS = lambdaBS; # intensity of base stations in BS/m^2
        self.lambdaUE = lambdaUE;
        self.networkArea = networkArea; # area of the network to be simulated
        
        self.k = 0;  # number of nearest BSs of interest
        
        self.numberOfBS = 0; # number of base stations in the network
        self.BSLocation = np.zeros((1,2)); # X-Y coordinates of BSs in network
        
        self.numberOfUE = 0; # number of UEs in the network
        self.UELocation = np.zeros((1,2)); # X-Y coordinates of UEs in network
        self.UEMotionDirection = np.zeros((1,1)); # angles from 0-2pi of motion directions 

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
    
    def getMobilityTrace(self, UEid, deltaT, velocity):
        # This function returns the location of the UE UEid after deltaT time
        # assuming constant velocity v, and fixed environment, i.e., other 
        # UEs are NOT moving
        
        # get initial location of UE UEid
        initUELoc = self.UELocation[UEid, :];
        distanceTravelled = velocity*deltaT;
        theta = self.UEMotionDirection[UEid];
        
        displacementVector = np.transpose(np.array([np.cos(theta), np.sin(theta)]))*distanceTravelled;
        finalLocation = initUELoc + displacementVector;
        
        return finalLocation;

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
        self.high = np.ones(2*k) * np.finfo(np.float32).max
        self.low = np.zeros(2*k)
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
        obs = np.concatenate((self.taggedUERates, np.transpose(self.myNetwork.BSLoads[self.taggedUEKClosestBS])));
        
        return obs, reward, done, {}
        
    def __take_action__(self, action):
        # this function updates the currentRate attribute
        self.currentRate = self.taggedUERates[action]/(self.myNetwork.BSLoads[self.taggedUEKClosestBS[action]]+1);
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
        self.taggedUEKClosestBS = self.myNetwork.kClosestBS(self.taggedCoord[0], 
                                                       self.taggedCoord[1])[0];
        
        # compute capacities received from k closest BSs
        self.taggedUERates = np.zeros(self.k);
        for i in range(self.k):
            currentBSId = self.taggedUEKClosestBS[i];
            self.taggedUERates[i] = self.myNetwork.getRate(currentBSId, self.taggedCoord, 10, 3, 1e-17, 1);


        # rng_state = np.random.get_state()
        np.random.shuffle(self.taggedUERates)
        # np.random.set_state(rng_state)
        # np.random.shuffle(self.myNetwork.BSLoads)
        
        # set current step to 0
        self.currentStep = 0;
        
        # return initial state
        return np.concatenate((self.taggedUERates, np.transpose(self.myNetwork.BSLoads[self.taggedUEKClosestBS])));
    
    def render(self): #MODIFY THIS!
    # Render the environment to the screen
        myFig = self.myNetwork.showNetwork();
        plt.figure(myFig.number);
        plt.scatter(self.myNetwork.BSLocation[self.currentAction,0], self.myNetwork.BSLocation[self.currentAction,1], c='g', marker = '^', s=400);
        plt.scatter(self.myNetwork.UELocation[self.taggedUEId, 0], self.myNetwork.UELocation[self.taggedUEId,1], c='g', s=200);
