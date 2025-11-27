import random
import time
from typing import Dict
import numpy as np
import pygame
from utility import play_q_table
from cat_env import make_env
#############################################################################
# TODO: YOU MAY ADD ADDITIONAL IMPORTS OR FUNCTIONS HERE.                   #
#############################################################################

# Helper function to compute Manhattan distance between bot and cat
def compute_dist(state):
    # State encoding: agent_row * 1000 + agent_col * 100 + cat_row * 10 + cat_col
    bot_r = state // 1000
    bot_c = (state % 1000) // 100
    cat_r = (state % 100) // 10
    cat_c = state % 10
    return abs(bot_r - cat_r) + abs(bot_c - cat_c)

#############################################################################
# END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
#############################################################################

def train_bot(cat_name, render: int = -1):
    env = make_env(cat_type=cat_name)
    
    # Initialize Q-table with all possible states (0-9999)
    # Initially, all action values are zero.
    q_table: Dict[int, np.ndarray] = {
        state: np.zeros(env.action_space.n) for state in range(10000)
    }

    # Training hyperparameters
    episodes = 5000 # Training is capped at 5000 episodes for this project
    
    #############################################################################
    # TODO: YOU MAY DECLARE OTHER VARIABLES AND PERFORM INITIALIZATIONS HERE.   #
    #############################################################################
    # Hint: You may want to declare variables for the hyperparameters of the    #
    # training process such as learning rate, exploration rate, etc.            #
    #############################################################################
    
    alpha = 0.1 # How fast to learn (higher = faster but less stable)
    gamma = 0.9 # Discount factor
    epsilon = 1.0 # Start with 100% random actions
    epsilon_min = 0.05 # Smallest allowed epsilon
    epsilon_decay = 0.999 # Reduce exploration over time

    #############################################################################
    # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
    #############################################################################
    
    for ep in range(1, episodes + 1):
        # 1. Reset environment to start a new episode
        state, _ = env.reset()
        episode_reward = 0
        done = False

        # Train until bot has not caught cat
        while not done: 
            # 2. Explore or exploit
            # Generate a random number between 0 and 1
            rnd = np.random.random()
            # If there's no best action, take a random one
            if rnd < epsilon:
                action = env.action_space.sample() # explore
            # Choose the action with the highest value in the current state
            else:
                action = np.argmax(q_table[state]) # exploit

            # 3. Take the action and observe the next state
            new_state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 4. Compute reward manually
            old_dist = compute_dist(state)
            new_dist = compute_dist(new_state)

            if new_dist == 0:
                reward = 100 # caught cat
            elif new_dist < old_dist: # getting closer to cat
                reward = 1
            elif new_dist > old_dist: # getting further from cat
                reward = -1
            else: 
                reward = -0.1 

            episode_reward += reward

            # 5. Update Update Q(s,a)
            q_table[state][action] += alpha * (
                reward + gamma * np.max(q_table[new_state]) - q_table[state][action])
            
            # Update the current state
            state = new_state

        # Update epsilon
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        #############################################################################
        # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
        #############################################################################

        # If rendering is enabled, play an episode every 'render' episodes
        if render != -1 and (ep == 1 or ep % render == 0):
            viz_env = make_env(cat_type=cat_name)
            play_q_table(viz_env, q_table, max_steps=100, move_delay=0.02, window_title=f"{cat_name}: Training Episode {ep}/{episodes}")
            print('episode', ep)

    return q_table