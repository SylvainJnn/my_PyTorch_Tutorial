import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class my_first_pytorch_model(nn.Module):
    def __init__(self, new_nb_input, new_nb_output, new_epochs, new_learning_rate):
        super().__init__() # instancies our nn module

        self.nb_input = new_nb_input
        self.nb_output = new_nb_output

        # create a 2 hidden layer 1 output layer NN : input, 4, 8, output # create a fucntion that create the number of hiddel layer based on an array ? 
        self.fc1 = nn.Linear(self.nb_input, 8) # fc for fully connected
        self.fc2 = nn.Linear(8, 9)
        self.out = nn.Linear(9, self.nb_output) 

        self.losses = [] # array that contains the result of all loss function for each epoch // keep it ? 

        # Hyper parameters
        self.epochs = new_epochs
        self.learning_rate = new_learning_rate


    def forward(self, input):
        x = torch.nn.functional.relu(self.fc1(input))
        x = torch.nn.functional.relu(self.fc2(x))
        output = torch.nn.functional.relu(self.out(x)) # test sans torch
        return(output)


    def my_custom_training(self, X_train, y_train):
        print("TRAINING")
        losses = []
        optimizer = torch.optim.Adam(my_model.parameters(), lr=self.learning_rate)
        for epoch in range(self.epochs):
            # go forward and get a predictions
            y_pred = self.forward(X_train) # my_model becomes seld (??)

            # where is criterion ??? from outside of class
            loss = criterion(y_pred, y_train) # train car on a split en deux: train et test, on fait les predictions sur _pred, on check avec _train. Quand NN finit entrainer, il s'entrainera avec y_test

            # Keep track of our losses
            losses.append(loss.detach().numpy())

            # print every 10 epochs
            if(epoch % 10 == 0):
                print(epoch, loss.detach().numpy())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        plt.plot(range(self.epochs), losses)
        plt.show()
        print("DONE")   
        return(None)
    
    # TESTING - https://www.youtube.com/watch?v=rgBu8CbH9XY
    # evaluate model on train dataset
    def my_custom_testing(self, X_test, y_test):
        with torch.no_grad(): # basically turn off backward progapation
            y_eval = my_model.forward(X_test) # testing using test dataset
            loss = criterion(y_eval, y_test) # find the loss or errors

            correct = 0 
            for i, data in enumerate(X_test):
                y_val = my_model.forward(data)
                print(f'{i+1}.) {str(y_val)} \t {y_test[i]}')

                if(y_val.argmax().item() == y_test[i]):
                    correct +=1
            print("correct answer: ", correct, " on ", len(X_test))
        # https://www.youtube.com/watch?v=Xp0LtPBcos0

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


X, y = init_data()

# pick a manual seed for randomization(41)
torch.manual_seed(41)

my_model = my_first_pytorch_model(4,3, 100, 1e-2)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) # it is random, we can add to be like in tutorial: random_state=41

# convert X features to float tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# convert y labels to tensorlongs
y_train = torch.LongTensor(y_train) # Long tensor are 64 bits integers (int cause we don't care if they are 2.0, only care if they are 0 1 or 2 I think)
y_test = torch.LongTensor(y_test)

# Set the criterion of model to measure the error, how far  off the predictions are from the data
criterion = nn.CrossEntropyLoss()


print(my_model.parameters)

# Train # put train in metods in NN class ? also make as argument the parameters like peochs
my_model.my_custom_training(X_train, y_train)
print("ouiii\n")

my_model.my_custom_testing(X_test, y_test)






"""
take off useless comment
make sure it works

add confusion matrix

put everything in functions 

put training in the NN class

in another code, make the neuronal network "flexible" (on donne un array: [3,4,6,2], l'input layer c'est les données d'entré (a automatisé aussi) et tu fais forward et création du NN en mode auto)


"""
