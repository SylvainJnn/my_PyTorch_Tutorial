def train(env_name, file_n):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # file_n = os.getcwd() + 
    # env_name = 'Acrobot-v1'
    # file_n = "modeltrained/"+ env_name +"/mine"
    
    figure_folder = "figure_folder"
    #if not os.path.exists(figure_folder):
	#os.makedirs(file_n)  # Crée le dossier et ses parents s'ils n'existent pas
        #print(f"Dossier créé : {file_n}")

    figure_file = figure_folder+'/my_version_' + env_name + '.png'


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
	# changer npm de n_steps
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
