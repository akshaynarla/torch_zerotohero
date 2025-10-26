import def_model
import torch

from torchvision import datasets, transforms

test_data = datasets.FashionMNIST(
    root="data",
    train= False,
    download=True,
    transform=transforms.ToTensor())

trained_model = def_model.NeuralNetwork()
trained_model.load_state_dict(torch.load("hello.pth", weights_only=True))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

trained_model.eval()

x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = trained_model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')