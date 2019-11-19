import numpy as np
from policy import Policy
from tqdm import trange

class ValueFunctionWithApproximation(object):
    def __call__(self,s) -> float:
        #calling an instance of the class with argument will call this function
        """
        return the value of given state; hat{v}(s)

        input:
            state
        output:
            value of the given state
        """
        raise NotImplementedError()

    def update(self,alpha,G,s_tau):
        """
        Implement the update rule;
        w <- w + alpha[G- hat{v}(s_tau;w)] nabla hat{v}(s_tau;w)

        input:
            alpha: learning rate
            G: TD-target
            s_tau: target state for updating (yet, update will affect the other states)
        ouptut:
            None
        """
        raise NotImplementedError()

def semi_gradient_n_step_td(
    env, #open-ai environment
    gamma:float,
    pi:Policy,
    n:int,
    alpha:float,
    V:ValueFunctionWithApproximation,
    num_episode:int,
):
    """
    implement n-step semi gradient TD for estimating v

    input:
        env: target environment
        gamma: discounting factor
        pi: target evaluation policy
        n: n-step
        alpha: learning rate
        V: value function
        num_episode: #episodes to iterate
    output:
        None
    """
    for _ in trange(num_episode):

        R = []
        S = []
        state, r, done = env.reset(), 0., False
        S.append(state)
        T = np.Inf
        t = 0

        while t <= T+n-2:

            if t<T:

                state = S[t]
                a = pi.action(state)
                next_state, r, done, info = env.step(a)
                R.append(r)
                S.append(next_state)
                if done:
                    T = t+1
            tau = t - n + 1
            if tau >= 0:
                G = 0
                for ii in range(tau+1, min(tau+n, T) + 1):
                    G = G + gamma**(ii - tau - 1) * R[ii-1]
                if tau+n<T:
                    G = G + gamma**n * V(S[tau+n])
                V.update(alpha, G, S[tau])

            t = t+1

