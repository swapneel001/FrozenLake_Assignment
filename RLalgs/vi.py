import numpy as np
from RLalgs.utils import action_evaluation # type: ignore

def value_iteration(env, gamma, max_iteration, theta):
    """
    Implement value iteration algorithm. 

    Inputs:
    env: OpenAI Gym environment.
            env.P: dictionary
                    the transition probabilities of the environment
                    P[state][action] is list of tuples. Each tuple contains probability, nextstate, reward, terminal
            env.observation_space.n: int
                    number of states
            env.action_space.n: int
                    number of actions
    gamma: float
            Discount factor.
    max_iteration: int
            The maximum number of iterations to run before stopping.
    theta: float
            The threshold of convergence.
    
    Outputs:
    V: numpy.ndarray
    policy: numpy.ndarray
    numIterations: int
            Number of iterations
    """

    V = np.zeros(env.observation_space.n)
    numIterations = 0

    #Implement the loop part here
    ############################
    # YOUR CODE STARTS HERE
    nS = env.observation_space.n
    nA = env.action_space.n

    while numIterations < max_iteration:
        delta = 0.0
        V_new = np.copy(V)
        for s in range(nS):
            # Bellman optimality backup
            best = -np.inf
            for a in range(nA):
                val = 0.0
                for prob, next_s, reward, done in env.P[s][a]:
                    if done:
                        val += prob * reward
                    else:
                        val += prob * (reward + gamma * V[next_s])
                if val > best:
                    best = val
            V_new[s] = best
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        numIterations += 1
        if delta < theta:
            break
    # YOUR CODE ENDS HERE
    ############################
    
    #Extract the "optimal" policy from the value function
    policy = extract_policy(env, V, gamma)
    
    return V, policy, numIterations

def extract_policy(env, v, gamma):

    """ 
    Extract the optimal policy given the optimal value-function.

    Inputs:
    env: OpenAI Gym environment.
            env.P: dictionary
                    P[state][action] is tuples with (probability, nextstate, reward, terminal)
                    probability: float
                    nextstate: int
                    reward: float
                    terminal: boolean
            env.observation_space.n: int
                    number of states
            env.action_space.n: int
                    number of actions
    v: numpy.ndarray
        value function
    gamma: float
        Discount factor. Number in range [0, 1)
    
    Outputs:
    policy: numpy.ndarray
    """

    policy = np.zeros(env.observation_space.n)
    ############################
    # YOUR CODE STARTS HERE
    q = action_evaluation(env, gamma, v)
    policy = np.argmax(q, axis=1).astype(int)
    # YOUR CODE ENDS HERE
    ############################

    return policy