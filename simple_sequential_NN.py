import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class my_first_pytorch_model(nn.Module):
    def __init__(self, new_nb_input, new_nb_output, new_hidden_layers, new_epochs, new_learning_rate):
        super().__init__() # instancies our nn module

        self.nb_input = new_nb_input
        self.nb_output = new_nb_output
        self.hidden_layers = new_hidden_layers # take off this one ? 
        self.layers = [self.nb_input] + new_hidden_layers + [self.nb_output] # ot new_hidden_layers.append(self.nb_output) # change name as number of layers
        self.neuronal_network = self.create_flexible_sequential_NN(self.layers) # call function # marchera pas comme comme avec avec al fonction 1 car on lui rend un type, cherche ce qui est le mieux entre et et 2


        self.losses_training = [] # array that contains the result of all loss function for each epoch // keep it ? 

        # Hyper parameters
        self.epochs = new_epochs
        self.learning_rate = new_learning_rate

        # Set the criterion of model to measure the error, how far  off the predictions are from the data
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def create_sequential_NN(self): # create hardcoded nn 4 9 8 3
        neuronal_network = nn.Sequential(nn.Linear(self.nb_input, 8),
                                            nn.ReLU(),
                                            nn.Linear(8, 9),
                                            nn.ReLU(),
                                            nn.Linear(9, self.nb_output),
                                            nn.ReLU)
        return(neuronal_network)

    def create_flexible_sequential_NN(self, layers):
        neuronal_network = []
        for i in range(len(layers) - 1):
            fc = nn.Linear(layers[i], 
                           layers[i+1]) # create fc_i. input, the current number of layer and output the next number of layers
            activation = nn.ReLU() # it can work if I don't put any activation function !! / I guess there is one preset in nn.Sequential OR there is apreset activation function in the forward method of Pytorch// we can put it as hyperparameter
                                    # Sigmoids for binary, softmaw for multiclass
            neuronal_network.append(fc)  
            neuronal_network.append(activation)
        
        new_neuronal_network = nn.Sequential(*neuronal_network)
        return(new_neuronal_network)

    def forward_sequential(self, input):
        output = self.neuronal_network(input)
        return(output)

    def my_custom_training(self, X_train, y_train):
        print("Training start")
        self.neuronal_network.train() # put model in training mode (mode by default)
        losses = []
        loss_testing = []
        for epoch in range(self.epochs): # with torch inference mode ? 
            # 1 - Forawr propagation
            y_pred = self.forward_sequential(X_train) 

            # 2 - calculate loss: how much different is model ouput comapre to result
            loss = self.criterion(y_pred, y_train) # train car on a split en deux: train et test, on fait les predictions sur _pred, on check avec _train. Quand NN finit entrainer, il s'entrainera avec y_test
            # Keep track of our losses
            losses.append(loss.detach().numpy())
            
            # 3 - zero the gradient of the optimizer (they accumulate by default)
            self.optimizer.zero_grad()

            # 4 - Perfom the back propagation on the loss (compute the gradient of every aprameters - requieres_grad = True)
            loss.backward()

            #5 - Gradient descent -> Progress/stop the optimizer on the loss
            self.optimizer.step()

            # print every 10 epochs
            if(epoch % 10 == 0):
                #print(f'Epoch: {epoch} and loss {loss}')
                print(epoch, loss.detach().numpy())
                # call testing, return loss_testing
            #FONFCTION
            with torch.inference_mode():
                # self.neuronal_network.eval() # pass to testing mode 
                y_eval = self.neuronal_network(X_test)
                loss_testing.append(self.criterion(y_eval, y_test).detach().numpy())
            # loss_testing.append(self.my_custom_testing().detach().numpy())

        plot_data(self.epochs, [losses, loss_testing])
        print("Training over")   
        # add confustion matrix --> ajouter une autre fonction qui appelle, trainign, testing et confusion matrix ? 
        return(None)

    def my_custom_testing(self): # MISS X_TEST AND Y TEST
        self.neuronal_network.eval() # pass to testing mode 
        with torch.no_grad(): # basically turn off backward progapation # with torch.inference_mode() (?????)
            y_eval = self.neuronal_network(X_test) # testing using test dataset # self.forward_sequential(X_test) 
            loss = self.criterion(y_eval, y_test) # find the loss or errors
            correct = 0 
            for i, data in enumerate(X_test):
                y_val = self.forward_sequential(data) 
                # print(f'{i+1}.) {str(y_val)} \t {y_test[i]}')

                if(y_val.argmax().item() == y_test[i]):
                    correct +=1
            # print("correct answer: ", correct, " on ", len(X_test))
            # print("Loss function result: ", loss)
        
        # https://www.youtube.com/watch?v=Xp0LtPBcos0

def plot_data(X_data, Y_data):
    # for data in Y_data:
    #     plt.plot(range(X_data), data)
    plt.plot(range(X_data), Y_data[0], label = "Loss training")
    plt.plot(range(X_data), Y_data[1], label = "Loss testing")
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def init_data():
    # import data
    url = "../datasets/iris.csv"
    df = pd.read_csv(url)

    # turn species column from string to float
    my_df = df
    my_df['species'] = my_df['species'].replace("setosa", 0.0)
    my_df['species'] = my_df['species'].replace("versicolor", 1.0)
    my_df['species'] = my_df['species'].replace("virginica", 2.0)

    print(my_df)

    # Split train test
    X = my_df.drop('species', axis=1)
    y = my_df['species']

    # convert to numpy arrays
    X = X.values
    y = y.values
    return(X, y)

if(__name__=='__main__'):
    # pick a manual seed for randomization(41)
    torch.manual_seed(42)

    X, y = init_data()

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) # it is random, we can add to be like in tutorial: random_state=41

    # convert X features to float tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)

    # convert y labels to tensorlongs
    y_train = torch.LongTensor(y_train) # Long tensor are 64 bits integers (int cause we don't care if they are 2.0, only care if they are 0 1 or 2 I think)
    y_test = torch.LongTensor(y_test)


    nb_input = X.shape[1]
    nb_outpout = 3              # the output of the NN is a list of tensor representing each perceptron output (logique hein mais j'avais pas capté, surtout que ça fait tout à ta palce derrière onc bon)
    hidden_layers = [8,9]

    epochs = 100
    learning_rate = 1e-2

    my_model = my_first_pytorch_model(nb_input, 
                                      nb_outpout, 
                                      hidden_layers, 
                                      epochs,
                                      learning_rate)

    print(my_model.parameters)

    # Train # put train in metods in NN class ? also make as argument the parameters like peochs
    my_model.my_custom_training(X_train, y_train)

    # my_model.my_custom_testing()

    """
    en gros, on peut activer et descativier le training, go tester de mettre le testing dans le training pour afficher  son évolution
    
    rajouter fonction load and ssave NN ( faire ça tout les 10 epochs par exemple)
    """

    # CHECK https://www.youtube.com/watch?v=V_xro1bcAuA

    # understand shape or layers, what is good, look for activation funtion and dropout things

    # petit exo pour voir si j'ai bien tout capté, on regait un NN sequential, qui doit train si c'est dans le cercle ou non