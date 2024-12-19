# FROM https://www.gocoder.one/blog/rl-tutorial-with-openai-gym/
# Also check https://blog.paperspace.com/getting-started-with-openai-gym/
import numpy as np
import gymnasium as gym
import random


class Q_table():
    def __init__(self,
                 new_environment_training,
                 new_environment_testing,
                 new_learning_rate=0.9,
                 new_discount_rate=0.8,
                 new_epsilon=1.0,
                 new_decay_rate=0.005,
                 new_epochs=1000,
                 new_max_steps=99 # per epochs
                 ):
        
        self.environment_training = new_environment_training
        self.environment_testing = new_environment_testing

        # initialize q-table
        self.state_size = self.environment_training.observation_space.n
        self.action_size = self.environment_training.action_space.n
        self.qtable = np.zeros((self.state_size, 
                                self.action_size))

        # hyperparameters
        self.learning_rate = new_learning_rate
        self.discount_rate = new_discount_rate
        self.epsilon = new_epsilon
        self.decay_rate= new_decay_rate

        # training variables
        self.epochs = new_epochs
        self.max_steps = new_max_steps # per episode


    def training(self):
        for epoch in range(self.epochs):
            # reset the environment
            state, info = self.environment_training.reset()

            for s in range(self.max_steps):
                print(f"step: {s} out of {self.max_steps} | episode: {epoch} out of {self.epochs}")

                # exploration-exploitation tradeoff
                if random.uniform(0,1) < self.epsilon:
                    # explore
                    action = self.environment_training.action_space.sample()
                else:
                    # exploit
                    print("state", state)
                    print("shape", self.qtable.shape)
                    action = np.argmax(self.qtable[state,:])

                # si je dois résumer : https://www.gymlibrary.dev/environments/toy_text/taxi/
                # sqtate rerpésente tout les états et choix:
                # - la première dimension c'est toutes les cases possible (en vrai plus j'ai pas tout capté, mais tout les états possible (voiture + pasager) -> There are 500 discrete states since there are 25 taxi positions, 5 possible locations of the passenger (including the case when the passenger is in the taxi), and 4 destination locations.
                # - la deuxième sont tout les choix (nord sud est ouest)
                # au début choix random pour donner un score à la q table
                # au bout d'un moment, il s'entraine en regardant le meilleur score de sa table. 

                # take action and observe reward
                new_state, reward, terminated, truncated, info = self.environment_training.step(action)

                # Q-learning algorithm # Q learning information http://www.incompleteideas.net/book/RLbook2018trimmed.pdf // formule de Belleman
                self.qtable[state,action] = self.qtable[state,action] + self.learning_rate * (reward + self.discount_rate * np.max(self.qtable[new_state,:]) - self.qtable[state,action])

                # Update to our new state
                state = new_state

                if terminated or truncated:
                    break

            # Decrease epsilon
            self.epsilon = np.exp(-self.decay_rate * epoch)
            print(f"Training completed over {self.epochs} episodes")
            # input("Press Enter to watch trained agent...")
    
            self.environment_training.close()


    def testing(self):
        state, info = self.environment_testing.reset()
        # done = False # truncated intead
        rewards = 0

        for s in range(self.epochs): # not epochs here

            print(f"TRAINED AGENT")
            print("Step {}".format(s+1))

            action = np.argmax(self.qtable[state,:])
            new_state, reward, terminated, truncated, info = self.environment_testing.step(action)
            rewards += reward
            # env.render("human")
            print(f"score: {rewards}")
            state = new_state

            if terminated or truncated:
                break

        self.environment_testing.close()

if __name__ == "__main__":
    env_train = gym.make('Taxi-v3')
    env_test = gym.make('Taxi-v3', render_mode="human")
    myQ = Q_table(env_train,
                  env_test)
    myQ.training()
    myQ.testing()

