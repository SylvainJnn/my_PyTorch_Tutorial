import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt

class PPO_memory():
    def  __init__(self, batch_size):
        # rename en old ?
        self.states = []
        self.actions= []
        self.action_probs = [] # rename ?
        self.rewards = []
        self.vals = [] # store values from value network
        self.dones = []

        self.batch_size = batch_size

    # take among (among or all?) previous experiences and create random batches from it, returns thoses batches. 
    def generate_batches(self): # get memory ? 
        states_size = len(self.states) 
        
        batch_start = np.arange(0, states_size, self.batch_size) # create an array [0; nb_state-1] with a step of self.batch_size
        
        indices = np.arange(states_size, dtype=np.int64) # create array of indices [0; nb_state-1]
        np.random.shuffle(indices)

        batches_indices = [] # 2D array: each array is a batch, each element is the indices of the data (eg states, actions)
        for x in batch_start: # shuffle batches. 
            # indices are shuffled, we take the size of a batch // x -> batch_start; x + batch_sizr -> batch end
            batch_indices = indices[x: x + self.batch_size] #batch_indices -> 1D array containing the indices of the element
            batches_indices.append(batch_indices)  # append the array of indices (=1 batch) in the array of indices

        return(np.array(self.states),
               np.array(self.actions),
               np.array(self.action_probs),
               np.array(self.vals),
               np.array(self.rewards),
               np.array(self.dones),
               batches_indices)
        
    def store_memory(self, state, action, action_prob, val, reward,done):
        self.states.append(state)
        self.actions.append(action)
        self.action_probs.append(action_prob)
        self.rewards.append(reward)
        self.vals.append(val)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions= []
        self.action_probs = []
        self.rewards = []
        self.vals = []
        self.dones = []


class policy_network(torch.nn.Module):
    def __init__(self,
                 state_dimension: int,   # input of the NN
                 action_dimensions: int, # output of the NN
                 learning_rate: float,
                 file_name: str = "",
                 device: str = "cpu"):
        
        super().__init__()

        self.model = torch.nn.Sequential(
                      torch.nn.Linear(state_dimension, 128),
                      torch.nn.ReLU(),
                      torch.nn.Linear(128, 64),
                      torch.nn.ReLU(),
                      torch.nn.Linear(64, action_dimensions),
                      torch.nn.Softmax())
        
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(params=self.parameters(), 
                                          lr=self.learning_rate) # arams=self.model.parameters(),

        self.file_name = file_name
        self.device = device
        self.to(self.device)

    def forward(self, input): # input is state vector in the case of PPO deep RL
        output = self.model(input)
        distribution = torch.distributions.categorical.Categorical(output) # Creates a probability distribution for each action from the network's outputs. --> instead of having a simple number, it transforms it to probabily using specific rules 
        return(distribution)

    def save_model(self):
        torch.save(self.state_dict(), self.file_name)

    def load_model(self):
        self.load_state_dict(torch.load(self.file_name))


class value_network(torch.nn.Module):
    def __init__(self, 
                 state_dimension: int, # input of the NN
                 learning_rate: float,
                 file_name: str = "",
                 device: str = "cpu"):
        
        super().__init__()

        self.model = torch.nn.Sequential(
                     torch.nn.Linear(state_dimension, 128),
                     torch.nn.ReLU(),
                     torch.nn.Linear(128, 64),
                     torch.nn.ReLU(),
                     torch.nn.Linear(64, 1))
        
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(params=self.parameters(), 
                                          lr=self.learning_rate) # arams=self.model.parameters(),

        self.file_name = file_name
        self.device = device
        self.to(self.device)
    
    def forward(self, input):
        output = self.model(input)
        return(output)
    
    def save_model(self):
        torch.save(self.state_dict(), self.file_name)

    def load_model(self):
        self.load_state_dict(torch.load(self.file_name))


class Agent():
    def __init__(self,
                 gamma: float,
                 lambda_value: float, # already exist in python 
                 epsilon: float, # = 0.2 e
                 learning_rate: int,
                 epochs: int,
                 batch_size: int, 
                 state_dimension: int,
                 action_dimension: int,
                 file_name: str,
                 device: str = "cpu"):
    
        # Hyper parameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.lambda_value = lambda_value
        self.learning_rate = learning_rate
        self.epochs = epochs # epoch refers to a complete pass through the collected batch of experience data to update the agent's policy and value networks. DOUBT
        

        self.batch_size = batch_size
        self.state_dimension = state_dimension
        self.action_dimension = action_dimension

        self.file_name = file_name
        self.device = device

        self.policy_network = policy_network(self.state_dimension,
                                             self.action_dimension,
                                             self.learning_rate,
                                             self.file_name + "/policy",
                                             self.device)

        self.value_network = value_network(self.state_dimension, 
                                           self.learning_rate,
                                           self.file_name+"/value", 
                                           self.device)
        
        self.memory = PPO_memory(self.batch_size)
        

    def choose_action(self, state): # do a forward to chose the action
        """
        Do a forward propagation to chose the action

        Input:
        - state: state of the environement 

        Output:
        - action: output of the policy network -> action choosed randomly based on the probabilities
        - probabilities: output of the policy network -> array that contains the probability of all the possible action
        - value: output of the value network -> (guessed action ??? TODO) 
        """
        state = torch.tensor([state], dtype=torch.float).to(self.policy_network.device) # set data acording to device
        
        # POLICY Network
        distribution = self.policy_network.forward(state)

        # Sample an action from the probability distribution generated by the policy_network network.
        action = distribution.sample()
        
        # Compute the log-probability of the sampled action. This will be used later for policy_network updates.
        probabilities = torch.squeeze(distribution.log_prob(action)).item()
        
        # Convert the sampled action from a PyTorch tensor to a plain Python scalar for interaction with the environment.
        action = torch.squeeze(action).item()

        # VALUE Network
        value = self.value_network.forward(state)
        value = torch.squeeze(value).item()

        return(action, probabilities, value)

    def calculate_running_At_v1(self, t, T_changer_nom, dones_arr, reward_arr, value_arr):
        """
        Compute At[t], the advantages for t 
        
        Input:
        - t: indices between [0, T-1]
        - T_changer_nom (int): limit of the number of data to look before (same number as batch size) 
        - dones_arr: array containing dones from memory
        - reward_arr: array containing reward from memory 
        - value_arr: array containing value from memory 
        
        Output:
        - running_At: At[t] = running_At

        # RE FAIRE AVEC LE PAPIER TODO
        # ce ne serait pass rewar ddu tout mais RATIO
        # -> faudrait faire une version (branch) Où plutpot que d'avoir At pour le batch entier, il faut le faire que pour un seul petit batch et ça prendrait le ratio en plutôt qu reward
        """
        discount = 1
        running_At = 0
        for k in range(t, T_changer_nom-1): # equation 11 from paper
            if(dones_arr[k]): # check if at this moment episode is over (=done) or not. If yes -> directly subtract the vlaue to the reward (ignore future episodes in this case)
                running_At += reward_arr[k] - value_arr[k]
            else:
                running_At += reward_arr[k] + (self.gamma*value_arr[k+1]) - value_arr[k] # δk​=rk​ + γ⋅V(sk+1​) − V(sk​) (?) -> rk reward or ration ?
        
            running_At = discount * running_At
            # discount *= self.gamma * self.lambda_value # was not in the code when it works # je pense que le mec était sous jack car c'est pas reward mais ration si je comprend le papier
        return(running_At)

    def calculate_running_At_v2(self, t, T_changer_nom, dones_arr, reward_arr, value_arr): # check https://kr.mathworks.com/help/reinforcement-learning/ug/proximal-policy-optimization-agents.html#mw_06e7348f-8170-408c-a080-ce2d579252d1  et https://ai.stackexchange.com/questions/34347/ppo-advantage-estimate-why-does-advantage-estimate-have-r-t-gamma-vs-t1
        running_At = 0
        for k in range(t, T_changer_nom-1):
            delta = reward_arr[k] + self.gamma*value_arr[k+1] - value_arr[k]
            running_At += (self.lambda_value * self.gamma)**(k-t) * delta # not sure // we can create a discount factor variable discountf and do: dicount = 1; dicout += lambda * gamma

        return(running_At)

    # maybe we can try other At computation method 
    def calculate_At(self,reward_arr, value_arr, dones_arr):# fucntion that caculates the advantage At
        """
        Function that calculates the Advantages At used to calculate the loss
        "This style requires an advantage estimator that does not look beyond timestep T."
        
        Input: 
        - reward_arr: array that contains rewards from PPO memory
        - value_arr: array that contains values from PPO memory
        - dones_arr: array that contains dones from PPO memory

        Output:
        - At: array that contains the advantage of the sample from memory

        """
        T_changer_nom = len(reward_arr) # on garde R je coirs // it looks for T data, the same size as bach sample memory (vlaue choose in PPO memory) 
        At = np.zeros(len(reward_arr), dtype=np.float32) # array contaning the advantages

        for t in range(0, T_changer_nom-1): # for each elelment from the (whole?) batch, it calcultes the adcantge A(t)
            
            # A COMPRENDRE
            # foutre dans une fonciton et on choisit ???
            running_At = self.calculate_running_At_v1(t, T_changer_nom, dones_arr, reward_arr, value_arr)      

            # store it in the Advantages array
            At[t] = running_At

        At = torch.tensor(At).to(self.policy_network.device) # converse to pytorch tensor
        
        return(At)

    def learn(self): # Backward
        """
        learn from previous examples. the agent takes samples from its memory (previous actions) and learn from it.
        It divides this memory to batches to process it.         
        """
        for _ in range(self.epochs):
            # get memory -> take sample of previous experiences sotred from PPO_memory
            # all arrays 
            # all empty at first ? 
            state_arr,  \
            action_arr, \
            memory_probs_arr, \
            value_arr,  \
            reward_arr, \
            dones_arr, \
            batches_indices = self.memory.generate_batches()

            At_arr = self.calculate_At(reward_arr, value_arr, dones_arr) # Array that contains advantage At for the whole batch 
            values_old = torch.tensor(value_arr).to(self.value_network.device) # array that contains  values

            for batch_indices in batches_indices:
                # get one batch from the whole
                states = torch.tensor(state_arr[batch_indices], dtype=torch.float).to(self.policy_network.device)
                actions = torch.tensor(action_arr[batch_indices], dtype=torch.float).to(self.policy_network.device)
                memory_probs = torch.tensor(memory_probs_arr[batch_indices], dtype=torch.float).to(self.policy_network.device)


                # HERE the code is very similar to choose_action -> updateing fuction to create a new one that be called by both choose_action and learn ? 
    
                # s or not ? 
                # policy
                # REFAIRE AVEC LES MATHS ET UNE FEUILLES
                distribution = self.policy_network.forward(states) # forward prpagation to get the array contaning the possible action 
                new_probs = distribution.log_prob(actions) #policy pi(a|t)! # TODO pourquoi ? 
                prob_ratio = new_probs.exp() / memory_probs.exp() # pk exp() ?  -> ek famoso (pi)/(pi_old) -> if yes rename memory en old ? 

                # policy
                prob_clip = torch.clamp(prob_ratio, 1-self.epsilon, 1+self.epsilon) * At_arr[batch_indices]
                loss_clip_policy = -torch.min(prob_ratio * At_arr[batch_indices], prob_clip).mean() # if not .mean() -> unstable / in torch the goal is to reduce so it needs a - sign


                # Value
                # s or not ? 
                value = self.value_network.forward(states) # critic # si crach enelever le .forward
                value = torch.squeeze(value) # get the value from value network after forward

                # Rt = At + V_old(?)(st) // Value target or return target, At advantages and V(st) the values from value network from memory
                value_target = At_arr[batch_indices] + values_old[batch_indices] 
                loss_value = ((value - value_target)**(2)).mean() # (V - Rt)**2
                
                c1 = 0.5 
                # POURQUOI + et pas - comme dans dans forumle ? (car pytorch cherche le plus petit donc il faut faire *(-1) ?)
                Loss_total = loss_clip_policy + c1*loss_value # Lclip_policy + value = E[... - c1 L ... + c2 S] 

                self.policy_network.optimizer.zero_grad()
                self.value_network.optimizer.zero_grad()

                Loss_total.backward()
                
                self.policy_network.optimizer.step()
                self.value_network.optimizer.step()

        self.memory.clear_memory()

    def store_data(self, state, action, action_prob, val, reward, done):
        self.memory.store_memory(state, action, action_prob, val, reward, done)
       
    def save_models(self):
        print('... Saving Models ......')
        self.value_network.save_model()
        self.policy_network.save_model()
    
    def load_models(self): # change this to a fvar, same for load_model functions !!
        print('... Loading models ...')
        self.value_network.load_model()
        self.policy_network.load_model()


def training(environment, agent, episodes, N, figure_file):
    """
    Train the agent

    Input:
    - environment: environment of the game (gymanisum)
    - agent: the tranined agent
    - episodes (int): total number of games the environment is going to play to train the agent
    - N (int) : variable that determines the number of time steps (n or t_steps) between two agent updates
    - figure_file (str): 
    """
    n_steps = 0 # n_step (discrete) or t_steps (continue) -> represent the total time (or nb of iteration) done in this envronment
    nb_learned = 0 # how many time did the agent learned

    score_history = [] # array that stores the score for each episode
    average_score = 0  # average array of the last 100 episodes  

    best_score = environment.reward_range[0] # inistliase mediuam score
    
    for i in range(episodes):
        current_state, info = environment.reset() # reset the game 
        terminated = False # True if the agent reached the goal
        truncated = False  # Episode ended due to a time or step limit
        done = False # (terminated + truncated=
        score = 0
        while not done:
            action, probabilities, value = agent.choose_action(current_state) # based on the current state, get the action with the probabily of all the other possible action (policy) and value (value)

            # take action and observe reward
            next_state, reward, terminated, truncated, info = environment.step(action) # make the agent do the action
            
            if(terminated or truncated):
                done = True

            score += reward # add the reward tot he score, the higher the better

            # Store this data to learn based on previous action in the futur // OR agent.memory.store_memory(...)
            # NB: the stored state is the state that makes the agent chose the current action (so current action, proba, value, etc..). it is NOT the state that outcomes from the last choosed action 
            agent.store_data(current_state, action, probabilities, value, reward, done) 
            n_steps += 1
            if(n_steps % N == 0): # every N times, the agent learns based on its previous action
                agent.learn()
                nb_learned +=1

            current_state = next_state # keep in memory for the name n_steps the state after the current action
            
        score_history.append(score)
        average_score = np.mean(score_history[-100:]) 

        if(average_score > best_score):
            best_score = average_score
            agent.save_models()
            
        print('episode', i, 
              'score %.1f' % score, 
              'avg score %.1f' % average_score,
              'time_steps', n_steps, 
              'learning_steps', nb_learned)
        
    print("training is over")
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)



"""

TODO:
- réécire sur une feuilles l'algo
- double checks varaibles names
- - une fois commenté, nommer les fonction en précisant les méthodes des calcul des var: 
- - - Rt = Monte carlo OU Approche apr estimation
- apparament ça marche pas avec env continue (on verra plus tard)

Next version:
- le faire marcher sur:
    - un truc pour marcher 
    - un jeux en utilisant l'image du jeux (cf CNN -> comme pokémon RL)
- rajouter un temps limite si y'en a pas déjà un


"""

    