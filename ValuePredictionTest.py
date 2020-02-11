#playing around with the environment
#this script initializes a network configuration, and attempts to estimate
#the value (under a fixed policy-here either deterministic or uniformly random) 
#of a state using a neural network.

import numpy as np
import gym
import sys
from tqdm import trange
from ValuePredictionNN import ValueFunctionWithNN
from myNetworkEnvironment import myNetworkEnvironment
import matplotlib.pyplot as plt


def test_nn():

	lambdaBS = 3e-6;
	lambdaUE = 3e-5;
	networkArea = 1e7;
	k = 10;
	episodeLength = 1;

	# create the environment
	env = myNetworkEnvironment(lambdaBS, lambdaUE, networkArea, k, episodeLength);
	V = ValueFunctionWithNN(k)
	

	NUM_ITERATIONS = 10000
	gamma = 1
	n = 1
	alpha = 0.1

	state_value_list = []
	state, r, done = env.reset(), 0., False
	initial_state = state

	for _ in trange(NUM_ITERATIONS):

		#n-step-semigradient-td
		R = []
		S = []
		state, r, done = initial_state, 0., False
		env.currentStep = 0
		state_value_list.append(V(initial_state))
		S.append(state)
		T = np.Inf
		t = 0

		while t <= T+n-2:
			
			if t<T:

				state = S[t]
				a = env.action_space.sample()
				#a = 3
				next_state, r, done, info = env.step(a)
				R.append(r)
				S.append(next_state)
				if done:
					T = t + 1

			tau = t - n + 1
			if tau >= 0:
				G = 0
				for ii in range(tau+1, min(tau+n, T) + 1):
					G = G + gamma**(ii - tau - 1) * R[ii-1]
				if tau+n<T:
					G = G + gamma**n * V(S[tau+n])
				V.update(alpha, G, S[tau])

			t = t+1

	print(np.mean(state_value_list))
	print(np.mean(env.taggedUERates))

	plt.plot(state_value_list)
	plt.savefig("prediction.eps")


if __name__ == "__main__":
	test_nn()
