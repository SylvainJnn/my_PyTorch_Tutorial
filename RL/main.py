import os
import numpy as np
import torch
from PPO import *


## Get the number of action the agent can do in the given environment
# @param environment: agent's environment
# @return action_dimension: number of possible agent's actions
def get_action_dimension(environment):
     # Identifier le type d'espace d'action
    if isinstance(environment.action_space, gym.spaces.Discrete):
        action_dimension = environment.action_space.n
        print(f"Espace d'action Discrete avec {action_dimension} actions possibles.")
    elif isinstance(environment.action_space, gym.spaces.Box):
        action_dimension = environment.action_space.shape[0]
        print(f"Espace d'action Box avec une dimension de {action_dimension}.")
    else:
        raise TypeError("Espace d'action inconnu.")
    return(action_dimension)


## Test the agent  
# @param trained_agent: the trained agent that is going to be tested
# @param test_episodes: maximum number of episodes to test the agent
def test(trained_agent, test_episodes): # put in ppo ? 
    
    trained_agent.load_models()

    trained_agent.policy_network.eval()
    trained_agent.value_network.eval()

    for _ in range(test_episodes):
        current_state, info = test_environment.reset()
        terminated = False,
        truncated = False
        done = False # over or note
        score = 0
        cmp = 0
        while not done:
            action, probabilities, value = agent.choose_action(current_state) # act()

            # take action and observe reward
            next_state, reward, terminated, truncated, info = test_environment.step(action)
            
            if(terminated or truncated):
                done = True

            current_state = next_state  



if __name__ == '__main__':

    env_name = 'CartPole-v1'
    file_n = "modeltrained/"+ env_name +"/mine"

    # Hyperparameters
    episodes = 300
    batch_size = 5
    epochs = 4
    learning_rate = 0.0003
    n_steps = 20 

    train_environment = gym.make(env_name)
    test_environment = gym.make(env_name, render_mode="human") 

    state_dimension = test_environment.observation_space.shape[0] 
    action_dimension = get_action_dimension(test_environment)

    # create file if does not exist
    if not os.path.exists(file_n):
        os.makedirs(file_n)  # Crée le dossier et ses parents s'ils n'existent pas
        print(f"Dossier créé : {file_n}")

    figure_folder = "figure_folder"
    figure_file = figure_folder+'/my_version_' + env_name + '.png'

    agent = Agent(gamma= 0.99,
                  lambda_value= 0.95, # already exist in python # what is it in ppo ? 
                  epsilon=0.2, # = 0.2 e
                  learning_rate= learning_rate,
                  epochs= epochs,
                  batch_size= epochs, 
                  state_dimension = state_dimension,
                  action_dimension = action_dimension, 
                  file_name= file_n,
                  device= "cpu")

    training(train_environment, agent, episodes, n_steps, figure_file)
    test(agent, 100)
