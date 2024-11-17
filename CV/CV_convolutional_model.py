# Import PyTorch
import torch
from torch import nn
from torch.utils.data import DataLoader

# Import torchvision
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

# Import matplotlib for visualization
import matplotlib.pyplot as plt

class CV_convolutional_model(nn.Module):
    def __init__(self,
                 new_hidden_units: int = 10,
                 new_epochs: int = 3,
                 new_learning_rate: float = 0.1,
                 new_batche_size: int = 32,
                 new_dataset: datasets = datasets.FashionMNIST, # dunno the type
                 new_device: str = "cpu"):#: torch.device):

        super().__init__() # instancies our nn module


        #######################
        ### Hyperparameters ###
        #######################
        self.epochs = new_epochs
        self.learning_rate = new_learning_rate
        self.batche_size = new_batche_size

        #################
        ### init data ###   (in a function ?)
        #################
        # get dataset from datasets from Pytorch, split train/test
        self.train_data = new_dataset( # we can change and make this as parameter in function
            root="data", # where to download data to?
            train=True, # get training data
            download=True, # download data if it doesn't exist on disk
            transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
            target_transform=None # you can transform labels as well
        )
        self.test_data = new_dataset(
            root="data",
            train=False, # get testing data
            download=True, # download data if it doesn't exist on disk
            transform=ToTensor() # images come as PIL format, we want to turn into Torch tensors
            # target_transform=None # you can transform labels as well // why no ?
        )

        # Turn datasets into iterables (batches)
        self.train_dataloader = DataLoader(self.train_data, # dataset to turn into iterable
            batch_size=self.batche_size, # how many samples per batch?
            shuffle=True # shuffle data every epoch?
        )
        self.test_dataloader = DataLoader(self.test_data,
            batch_size=self.batche_size,
            shuffle=False # don't necessarily have to shuffle the testing data
        )


        #####################
        ### attribute (?) ###
        #####################
        #we want the number of element in one image: we take one image (the first one -> it is a tuple -> we get the number of element in it )
        # image shape is [1, 28, 28] (colour channels, height, width)
        image, label = self.train_data[0]
        self.input_shape =  image.shape[0] #  the number of input is the number of channel in our image, for MNIST it is 1
        self.hidden_units = new_hidden_units
        self.output_shape = len(self.train_data.classes) # the output is the number of existing class // Why to device ???
        # self.neuronal_network = self.create_linear_nn()
        self.device = new_device
        print(self.device)

        self.create_CNN()

        # Set the criterion of model to measure the error, how far  off the predictions are from the data
        self.criterion = nn.CrossEntropyLoss()
        print(self.parameters)
        self.optimizer = torch.optim.SGD(params = self.parameters(),
                                         lr = self.learning_rate)

        # print("TEST IMAGE")
        # randimg = torch.randn(size=(1,28,28))
        # print(image.shape)
        # print(randimg.shape)
        # print("start")
        # self.forward(randimg)




    def create_CNN(self):# why 2d and not 3d or 1d ?
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels = self.input_shape,
                      out_channels = self.hidden_units,
                      kernel_size = 3, # ...
                      stride = 1, #  .... // default
                      padding = 1),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
            nn.ReLU(),
            nn.Conv2d(in_channels = self.hidden_units,
                      out_channels = self.hidden_units,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, # why
                         stride = 2)# default stride value is same as kernel_size
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(self.hidden_units, self.hidden_units, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_units, self.hidden_units, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from?
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features = self.hidden_units*7*7, # FIND A WAY TO AUTOMATISE IT : https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
                      out_features = self.output_shape)
        )

    def forward(self, x):
        # print(x.shape)
        x = self.block1(x)
        # print(x.shape)
        x = self.block2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x)
        return(x)

    def train_step(self):
        self.train() # put model in training mode (mode by default) # ?
        # Add a loop to loop through training batches
        train_loss, train_acc = 0, 0
        self.to(self.device)
        for batch, (X, y) in enumerate(self.train_dataloader):
            # Send data to proper device
            X = X.to(self.device)
            y = y.to(self.device)

            # 1 Forward Propagation
            y_pred = self.forward(X)

            #2 calculate loss function
            loss = self.criterion(y_pred, y)
            train_loss += loss
            train_acc += accuracy_fn(y_true=y,
                                     y_pred=y_pred.argmax(dim=1)) # Go from logits -> pred labels

            # 3 - Zero Gradient of the optimizer
            self.optimizer.zero_grad()

            # 4 Backpropagation
            loss.backward()

            # 5 Gradient Descent
            self.optimizer.step()

            if(batch % 400 == 0):
                print(f"Looked at {batch * len(X)}/{len(self.train_dataloader.dataset)} samples")
            
        # Calculate loss and accuracy per epoch and print out what's happening
        train_loss /= len(self.train_dataloader)
        train_acc /= len(self.train_dataloader)
        print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

    def test_step(self):
        test_loss, test_acc = 0, 0
        self.to(self.device)
        self.eval() # put model in eval mode
        # Turn on inference context manager
        with torch.inference_mode():
            for X, y in self.test_dataloader:
                # Send data to proper device
                X = X.to(self.device)
                y = y.to(self.device)

                # 1 forward pass
                test_pred = self.forward(X)

                # 2. Calculate loss and accuracy
                test_loss += self.criterion(test_pred, y)
                test_acc += accuracy_fn(y_true=y,
                    y_pred=test_pred.argmax(dim=1) # Go from logits -> pred labels
                )
            # Adjust metrics and print out
            test_loss /= len(self.test_dataloader)
            test_acc /= len(self.test_dataloader)
            print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")
            return(test_loss.item(), test_acc)
    def custom_training2(self):
        self.train() # put model in training mode (mode by default)
        for epoch in range(self.epochs):
            print(f"Epoch: {epoch}\n---------")
            self.train_step()
            # self.test_step()

    def eval_model(self):
        test_loss, test_acc = self.test_step()
        info = {"model_type": self.__class__.__name__, # only works when model was created with a class
                "model_loss": test_loss,
                "model_acc": test_acc}
        print(info)


def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

if(__name__ == '__main__'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model_conv = CV_convolutional_model(new_hidden_units = 10,
                                        new_epochs = 3,
                                        new_learning_rate = 0.1,
                                        new_batche_size = 32,
                                        new_dataset = datasets.FashionMNIST,
                                        new_device = device).to(device)#: torch.device):


    model_conv.custom_training2()
    model_conv.eval_model()


# 18:24
# regarder vidéo et lire le machin site pour comprendre les hyper parametères