import numpy as np
from RLalgs.utils import action_evaluation
from tqdm import tqdm
import random 
def policy_iteration(env, gamma, max_iteration, theta):
    """
    Implement Policy iteration algorithm.

    Inputs:
    env: OpenAI Gym environment.
            env.P: dictionary
                    P[state][action] is list of tuples. Each tuple contains probability, nextstate, reward, terminal
                    probability: float
                    nextstate: int
                    reward: float
                    terminal: boolean
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
    """

    V = np.zeros(env.observation_space.n)
    policy = np.zeros(env.observation_space.n)
    policy_stable = False
    numIterations = 0
    
    
    while not policy_stable and numIterations < max_iteration:
                # Implement it with function policy_evaluation and policy_improvement
                ############################
                # YOUR CODE STARTS HERE
                V = policy_evaluation(env, policy, gamma, theta)
                policy, policy_stable = policy_improvement(env, V, policy, gamma)                
                # YOUR CODE ENDS HERE
                ############################
                numIterations += 1  
                    
    return V, policy, numIterations


def policy_evaluation(env, policy, gamma, theta):
    """
    Evaluate the value function from a given policy.

    Inputs:
    env: OpenAI Gym environment.
            env.P: dictionary
                    
            env.observation_space.n: int
                    number of states
            env.action_space.n: int
                    number of actions

    gamma: float
            Discount factor.
    policy: numpy.ndarray
            The policy to evaluate. Maps states to actions.
    theta: float
            The threshold of convergence.
    
    Outputs:
    V: numpy.ndarray
            The value function from the given policy.
    """
    ############################ 
    # YOUR CODE STARTS HERE
    nS = env.observation_space.n
    V = np.zeros(nS)
    while True:
        delta = 0.0
        for s in range(nS):
            v_old = V[s]
            a = policy[s]
            v_new = 0.0
            for prob, next_s, reward, done in env.P[s][a]:
                # if done:
                #     v_new += prob * reward
                # else:
                    v_new += prob * (reward + gamma * V[next_s])
            V[s] = v_new
            delta = max(delta, abs(v_old - v_new))
        if delta < theta:
            break
        
    # YOUR CODE ENDS HERE
    ############################

    return V


def policy_improvement(env, value_from_policy, policy, gamma):
    """
    Given the value function from policy, improve the policy.

    Inputs:
    env: OpenAI Gym environment
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

    value_from_policy: numpy.ndarray
            The value calculated from the policy
    policy: numpy.ndarray
            The previous policy.
    gamma: float
            Discount factor.

    Outputs:
    new policy: numpy.ndarray
            An array of integers. Each integer is the optimal action to take
            in that state according to the environment dynamics and the
            given value function.
    policy_stable: boolean
            True if the "optimal" policy is found, otherwise false
    """
    ############################
    # YOUR CODE STARTS HERE
    q = action_evaluation(env, gamma, value_from_policy)
    nS = env.observation_space.n
    policy_stable = True
    new_policy = np.copy(policy)
    for s in range(nS):
        old_action = policy[s]
        best_action = np.argmax(q[s])
        new_policy[s] = best_action
        if best_action != old_action:
            policy_stable = False
    policy = new_policy
    # YOUR CODE ENDS HERE
    ############################   
    return policy, policy_stable