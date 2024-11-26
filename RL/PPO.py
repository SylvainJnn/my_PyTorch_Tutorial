# https://github.com/hermesdt/reinforcement-learning/blob/master/ppo/cartpole_ppo_online.ipynb
import os
import gymnasium as gym
import torch
# from torch.utils import tensorboard
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
                 lambda_value: float, # already exist in python # what is it in ppo ? 
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
        state = torch.tensor([state], dtype=torch.float).to(self.policy_network.device) # set data acording to device
        
        # Policy Network
        distribution = self.policy_network.forward(state)
        # Sample an action from the probability distribution generated by the policy_network network.
        action = distribution.sample()
        # Compute the log-probability of the sampled action. This will be used later for policy_network updates.
        probabilities = torch.squeeze(distribution.log_prob(action)).item()
        # Convert the sampled action from a PyTorch tensor to a plain Python scalar for interaction with the environment.
        action = torch.squeeze(action).item()

        # Value Network
        value = self.value_network.forward(state)
        value = torch.squeeze(value).item()

        return(action, probabilities, value)

    # maybe we can try other At computation method 
    def calculate_At(self,reward_arr, value_arr, dones_arr):# fucntion that caculates the advantage At
        T_changer_nom = len(reward_arr)
        At = np.zeros(len(reward_arr), dtype=np.float32) # array contaning the adcantages

        for t in range(0, T_changer_nom-1):
            discount = 1
            running_At= 0
            # A COMPRENDRE
            for k in range(t, T_changer_nom-1): 
                if(dones_arr[k]): # check if at this moment episode is over (=done) or not. If yes -> directly subtract the vlaue to the reward (ignore future episodes in this case)
                    running_At += reward_arr[k] - value_arr[k]
                else:
                    running_At += reward_arr[k] + (self.gamma*value_arr[k+1]) - value_arr[k] # δk​=rk​ + γ⋅V(sk+1​) − V(sk​) (?) -> rk reward or ration ??

            running_At = discount * running_At
            At[t] = running_At

        At = torch.tensor(At).to(self.policy_network.device) # converse to pytorch tensor
        
        return(At)

    def learn(self): # Backward
        for _ in range(self.epochs):
            # get memory -> take sample of previous experiences sotred in PPO_memory
            # all arrays 
            # all empty at first ? 
            state_arr,  \
            action_arr, \
            memory_probs_arr, \
            value_arr,  \
            reward_arr, \
            dones_arr, \
            batches_indices = self.memory.generate_batches()

            At_arr = self.calculate_At(reward_arr, value_arr, dones_arr) # ? 
            values = torch.tensor(value_arr).to(self.value_network.device)

            for batch_indices in batches_indices:
                # get
                states = torch.tensor(state_arr[batch_indices], dtype=torch.float).to(self.policy_network.device)
                actions = torch.tensor(action_arr[batch_indices], dtype=torch.float).to(self.policy_network.device)
                memory_probs = torch.tensor(memory_probs_arr[batch_indices], dtype=torch.float).to(self.policy_network.device)


                # HERE the code is very similar to choose_action -> updateing fuction to create a new one that be called by both choose_action and learn ? 
    
                # s or not ? 
                # policy
                # REFAIRE AVEC LES MATHS ET UNE FEUILLES
                distribution = self.policy_network.forward(states)
                new_probs = distribution.log_prob(actions) #policy pi(a|t)!
                prob_ratio = new_probs.exp() / memory_probs.exp() # pk exp() ?  -> ek famoso (pi)/(pi_old) -> if yes rename memory en old ? 

                # policy
                prob_clip = torch.clamp(prob_ratio, 1-self.epsilon, 1+self.epsilon) * At_arr[batch_indices]
                loss_clip_policy = -torch.min(prob_ratio * At_arr[batch_indices], prob_clip).mean() # if not .mean() -> unstable / in torch the goal is to reduce so it needs a - sign



                # Value
                # s or not ? 
                value = self.value_network(states) # critic
                value = torch.squeeze(value)

                # Rt = At + V(st)
                Rt = At_arr[batch_indices] + values[batch_indices] # from code # poruqoi y'a deux fois values (pas le même input, y'en a un c'est state[k] et l'autre c'est quoi ?)
                # Rt = prob_ratio + self.gamma * self.values € states + 1 ? ????
                loss_value = ((value - Rt)**(2)).mean() # (V - Rt)**2
                
                c1 = 0.5 # why multiply ?
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



def training(environment, agent, episodes, n_steps, figure_file):
    n = 0 # time steps ? WEIRD. it is related to n_steps
    nb_learned = 0

    average_score = 0
    best_score = environment.reward_range[0]

    score_history = []

    for i in range(episodes):
        current_state, info = environment.reset()
        terminated = False,
        truncated = False
        done = False # over or note
        score = 0
        while not done:
            action, probabilities, value = agent.choose_action(current_state) # act()

            # take action and observe reward
            next_state, reward, terminated, truncated, info = environment.step(action)
            
            if(terminated or truncated):
                done = True

            score += reward
            
            agent.store_data(current_state, action, probabilities, value, reward, done) # OR agent.memory.store_memory(....)

            n += 1
            if(n % n_steps == 0): # every n_steps
                agent.learn()
                nb_learned +=1

            current_state = next_state   
            
        score_history.append(score)
        average_score = np.mean(score_history[-100:]) # ??

        if(average_score > best_score):
            best_score = average_score
            agent.save_models()
            
        print('episode', i, 
              'score %.1f' % score, 
              'avg score %.1f' % average_score,
              'time_steps', n, 
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

def train(env_name, file_n):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # file_n = os.getcwd() + 
    # env_name = 'Acrobot-v1'
    # file_n = "modeltrained/"+ env_name +"/mine"
    figure_file = 'my_version_' + env_name + '.png'

    if not os.path.exists(file_n):
        os.makedirs(file_n)  # Crée le dossier et ses parents s'ils n'existent pas
        print(f"Dossier créé : {file_n}")
    # else:
    #     print(f"Dossier existe déjà : {file_n}")

    # render_mode="human"
    environment = gym.make(env_name, render_mode="nohuman") # mettre ça en variable le nom 
    # environment = gym.make(env_name, render_mode="humanNON", continuous=False, gravity=-10.0,
    #            enable_wind=False, wind_power=15.0, turbulence_power=1.5)
    
    # recoder ça
    state_dimension = environment.observation_space.shape[0] # ????
    # action_dimension = environment.action_space.n

    # Identifier le type d'espace d'action
    if isinstance(environment.action_space, gym.spaces.Discrete):
        action_dimension = environment.action_space.n
        print(f"Espace d'action Discrete avec {action_dimension} actions possibles.")
    elif isinstance(environment.action_space, gym.spaces.Box):
        action_dimension = environment.action_space.shape[0]
        print(f"Espace d'action Box avec une dimension de {action_dimension}.")
    else:
        raise TypeError("Espace d'action inconnu.")

    batch_size = 5
    epochs = 4
    learning_rate = 0.0003

    episodes = 400
    
    n_steps = 20 # number of simulation steps (= episode) done by the agent before being updated. # small n_steps -> very sensitive to immediate change (only care about recent event) BUT easy to intriduce noise // bug n_step -> better to estimate futur feeback BUT takes more time and needs bigger memory 
    # n_updates = total_timesteps // (n_steps * n_envs)




    print(state_dimension, action_dimension)

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

    training(environment, agent, episodes, n_steps, figure_file)

def test(env_name, file_n):

    test_environment = gym.make(env_name, render_mode="human") # mettre ça en variable le nom 
    

    state_dimension = test_environment.observation_space.shape[0] # ????
    # action_dimension = test_environment.action_space.n

    # Identifier le type d'espace d'action
    if isinstance(test_environment.action_space, gym.spaces.Discrete):
        action_dimension = test_environment.action_space.n
        print(f"Espace d'action Discrete avec {action_dimension} actions possibles.")
    elif isinstance(test_environment.action_space, gym.spaces.Box):
        action_dimension = test_environment.action_space.shape[0]
        print(f"Espace d'action Box avec une dimension de {action_dimension}.")
    else:
        raise TypeError("Espace d'action inconnu.")

    batch_size = 5
    epochs = 4
    learning_rate = 0.0003

    episodes = 400
    
    n_steps = 20 # number of simulation steps (= episode) done by the agent before being updated. # small n_steps -> very sensitive to immediate change (only care about recent event) BUT easy to intriduce noise // bug n_step -> better to estimate futur feeback BUT takes more time and needs bigger memory 
    # n_updates = total_timesteps // (n_steps * n_envs)

    test_episodes = 100

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
    
    agent.load_models()

    agent.policy_network.eval()
    agent.value_network.eval()


    for i in range(test_episodes):
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

            score += reward
            
            # FOR ELAGNING
            # agent.store_data(current_state, action, probabilities, value, reward, done) # OR agent.memory.store_memory(....)

            # n += 1
            # if(n % n_steps == 0): # every n_steps
            #     agent.learn()
            #     nb_learned +=1

            current_state = next_state  



if __name__ == '__main__':
    env_name = "LunarLander-v2"
    file_n = "modeltrained/"+ env_name +"/mine"
    train(env_name, file_n)
    test(env_name, file_n)

"""

TODO:
- réécire sur une feuilles l'algo
- double checks varaibles names
- rajouter du commentaire
- - une fois commenté, nommer les fonction en précisant les méthodes des calcul des var: 
- - - Rt = Monte carlo OU Approche apr estimation
- apparament ça marche pas avec env continue (on verra plus tard)

Next version:
- le faire marcher sur:
    - un truc pour marcher 
    - un jeux en utilisant l'image du jeux (cf CNN -> comme pokémon RL)
- rajouter un temps limite si y'en a pas déjà un

mettre sur guthub !!!


"""

    