import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import numpy.random as rn
import scipy.integrate
import gym
from gym import spaces
from gym.utils import seeding

class VoronoiEnv(gym.Env):

	"""
	We initialize a PPP of BSs, each transmitting at constant power. 


	"""

	def __init__(self, lambdaBS, NumSelectableBS):

		self.L = 100
		self.lambdaBS = lambdaBS #intensity of base stations
		self.NumSelectableBS = NumSelectableBS #value of k (k-closest BSs can be chosen)

		#area of simulation
		self.area = self.L**2
		self.NumBS = rn.poisson(self.lambdaBS * self.area)

		#generating the PPP
		self.BS_X_position = rn.uniform(0, L, self.NumBS)
		self.BS_Y_position = rn.uniform(0, L, self.NumBS)

		self.EpisodeLength = 100 #number of time steps each episode lasts for
		self.ElapsedTimeSteps = 0


		self.action_space = spaces.Discrete(self.NumBS) #define the action space as a discrete space

		low = np.zeros(self.NumSelectableBS) # defining high and low values for observable states
		high = np.ones(self.NumSelectableBS) * np.inf

		self.observation_space = spaces.Box(low, high, dtype = np.float32) #defining the observation space as vector of length k that can take values in [0, Inf]

		self.seed()
		self.state = None #state here will be a random position in space?

	def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

    	"""
    	Input: an action
    	Output: observation/state, reward, done (end of episode or not), info (information useful for debugging)

    	TODO: implement reward calculation, output observations/states
    	"""


    	self.ElapsedTimeSteps = self.ElapsedTimeSteps + 1 #increments counter of timesteps

    	if self.ElapsedTimeSteps == self.EpisodeLength: #checks for end of episode
    		done = True
    	else:
    		done = False


    	return np.array(self.state), reward, done, {}

    def reset(self):

    	self.state = np.array[rn.uniform(0, L), rn.uniform(0, L)] #pick a new random location in space
    	self.ElapsedTimeSteps = 0

    	return np.array(self.state)

    def render(self):
    	"""
    	function that displays the learned Voronoi tessellation
    	"""

