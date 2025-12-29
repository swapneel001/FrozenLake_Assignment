import numpy as np
from RLalgs.utils import epsilon_greedy
import random

def SARSA(env, num_episodes, gamma, lr, e):
    """
    Implement the SARSA algorithm following epsilon-greedy exploration.

    Inputs:
    env: OpenAI Gym environment 
            env.P: dictionary
                    P[state][action] are tuples of tuples tuples with (probability, nextstate, reward, terminal)
                    probability: float
                    nextstate: int
                    reward: float
                    terminal: boolean
            env.observation_space.n: int
                    number of states
            env.action_space.n: int
                    number of actions
    num_episodes: int
            Number of episodes of training
    gamma: float
            Discount factor.
    lr: float
            Learning rate.
    e: float
            Epsilon value used in the epsilon-greedy method.

    Outputs:
    Q: numpy.ndarray
            State-action values
    """
    
    Q = np.zeros((env.observation_space.n, env.action_space.n))
   
    #TIPS: Call function epsilon_greedy without setting the seed
    #      Choose the first state of each episode randomly for exploration.
    ############################
    # YOUR CODE STARTS HERE
    nS = env.observation_space.n
    nA = env.action_space.n

    for state in range(nS):
        if not env.P[state][0][0][3]:
            Q[state] = np.random.rand(nA)

    for _ in range(num_episodes):
        s = env.reset()
        done = False

        while not done:

            # epsilon-greedy
            a = int(epsilon_greedy(Q[s], e))
            s_next, reward, done, _ = env.step(a)

            # TD target
            if done:
                target = reward
            else:
                a_next = int(epsilon_greedy(Q[s_next], e))
                target = reward + gamma * Q[s_next,a_next]

            # update
            Q[s, a] += lr * (target - Q[s, a])

            s = s_next

   
    # YOUR CODE ENDS HERE
    ############################

    return Q