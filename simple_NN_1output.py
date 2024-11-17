import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class my_first_pytorch_model(nn.Module):
    def __init__(self, new_nb_input, new_nb_output):
        super().__init__() # instancies our nn module

        self.nb_input = new_nb_input
        self.nb_output = new_nb_output

        # create a 2 hidden layer 1 output layer NN : input, 4, 8, output # create a fucntion that create the number of hiddel layer based on an array ? 
        self.fc1 = nn.Linear(self.nb_input, 4) # fc for fully connected
        self.fc2 = nn.Linear(4, 8)
        self.out = nn.Linear(8, self.nb_output)
        # check what is nn.Linear et linear stack ? https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

    def forward(self, input):
        x = nn.functional.sigmoid(self.fc1(input))
        x = nn.functional.sigmoid(self.fc2(x))
        output = nn.functional.sigmoid(self.out(x)) 

        # output = x.T[0] # je dois faire ça pour que ce soit dans le même format. en gros X c'est un 2d array [120,1] : 120 cases contenant chacun 1 tableau de 1 élément. Sauf qu'parès on le compare avec un tableau contenant 120 élement e non pas 120 dans 1 élement (j'iamgine que soit on fait comme j'ai fait, soit on met y_pred dans des tableau de 1 120 fois)
        # the 
        if(output.shape != torch.Size([1])): #  find a better way
            # pass from shape (N,1) to (N) // froom array in array to simple array
            # output = output.T[0] # the other way to do it to pass fropm 
            output = output.squeeze(1)  # Squeeze, take off one dimension, unsqueeze, add a dimension

        return(output)

def init_data():
    # import data
    url = "../datasets/iris.csv"
    df = pd.read_csv(url)

    # turn species column from string to float
    my_df = df

    my_df['species'] = my_df['species'].replace("setosa", 0.0)
    my_df['species'] = my_df['species'].replace("versicolor", 1.0)
    my_df['species'] = my_df['species'].replace("virginica", 1.0)

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

my_model = my_first_pytorch_model(4,1)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=41) # it is random, we can add to be like in tutorial: random_state=41

# convert X features to float tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# convert y labels to tensorlongs
y_train = torch.FloatTensor(y_train) #  // IN THE VIDEO he used LongTensor, me I sued Float, why it does not work another way // Long tensor are 64 bits integers (int cause we don't care if they are 2.0, only care if they are 0 1 or 2 I think)
y_test = torch.FloatTensor(y_test)


# Set the criterion of model to measure the error, how far  off the predictions are from the data
# In case of binary classification, take a the binary cross entropy function
criterion = nn.BCELoss()
# chose Adam optimize // c'est quoi ?? popular one askip
optimizer = torch.optim.Adam(my_model.parameters(), lr=0.01)

print(my_model.parameters)

# Train # put train in metods in NN class ? also make as argument the parameters like peochs

epochs = 100
losses = []

for epoch in range(epochs):
    # go forward and get a predictions
    y_pred = my_model.forward(X_train)

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

plt.plot(range(epochs), losses)
plt.show()
print("DONE")

with torch.no_grad(): # basically turn off backward progapation
    y_eval = my_model.forward(X_test) # testing using test dataset
    loss = criterion(y_eval, y_test) # find the loss or errors
    # print(loss)

    correct = 0 
    y_result = []
    for i, data in enumerate(X_test):
        y_val = my_model.forward(data)
        # print(f'{i+1}.) {str(y_val)} \t {y_test[i]}')

        # # faire mieux
        
        # for val in y_val.argmax().item():
        # y_result.append(float(np.round(y_val.argmax().item())))
        if(y_val.item() <0.5):
            y_result.append((float(0)))
        else:
            y_result.append((float(1)))
        # y_val[0] # en gros récupérer la valeur et on peut faire des trucs avec
        # print(y_result)
        # faire tout ça en mieux, en gros ça marche, mais la loss function part en couille j'ai l'impression ? 
        # print("\n TYPE")
        # print(y_val, type(y_val))
        # print(y_test[i], type(y_test[i]))
        if(y_result[i]== float(y_test[i])):
            correct +=1
    print("correct answer: ", correct, " on ", len(X_test))
        # https://www.youtube.com/watch?v=Xp0LtPBcos0

# print(y_pred)


# https://www.youtube.com/watch?v=Xp0LtPBcos0

