from torch import nn, optim, save
from torch.utils.data import DataLoader
# torchvision is domain specific library for computer vision
from torchvision import datasets
from torchvision import transforms
import def_model, def_train_process

# hello_pytorch is used to show and introduce the procedure for training 
# https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
# Quickstart tutorial: we will take a vision dataset as it is easier to understand and visualize.
# FashionMNIST Dataset: FashionMNIST are black and white images of size 28x28

# ----------- Hyperparameters --------------------------------
batch_size = 64                 # number of data samples propogated through network before parameters are updates
epochs = 20                      # number of times to iterate over the dataset
learning_rate = 1e-3            # how much to update models parameters at each epoch or batch

# ----------- 0. Data Preprocessing. -----------------------------------
# But for any other data, you'd need to preprocess and bring it to tensor form.
# FashionMNIST or MNIST or some others are already preprocessed and can be directly transformed to tensor.

# --------- 1. Load Data. This will download dataset and prepare iterables ---------
# Here we use GTSRB dataset (German Traffic Sign Recognition Benchmark)
train_data = datasets.FashionMNIST(
    root="data",
    train= True,
    download=True,
    transform=transforms.ToTensor())

test_data = datasets.FashionMNIST(
    root="data",
    train= False,
    download=True,
    transform=transforms.ToTensor())

# ------------------- 2. Pass the loaded dataset to DataLoader -------------------
# Default DataLoader stacks tensors into a batch but works only if images are of same size.
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])
# Shape of y: torch.Size([64]) torch.int64
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# ------------- 3. Create Models by defining layers ---------------------
# create an object of neural network. An init constructor is run which will create the necessary model.
model = def_model.NeuralNetwork()
print(model)

# --------------- 4. Train a Model -------------------------------------
# requires creation of loss function and optimizer
loss_fn = nn.CrossEntropyLoss()                         # since this is a classification problem
optimizer = optim.SGD(model.parameters(), lr=learning_rate)      # returns an iterator over model parameters

# during each epoch better parameter selection is done to make better predictions.
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    def_train_process.train(train_dataloader, model, loss_fn, optimizer)
    def_train_process.test(test_dataloader, model, loss_fn)
print("Training the network is completed!")

# --------------- 5. Saving the model ------------------------------------
save(model.state_dict(), "hello.pth")
print("Saved PyTorch Model State to hello.pth")