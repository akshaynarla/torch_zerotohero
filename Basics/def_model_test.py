import def_model
import torch

# https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

# create a random tensor with 3 images, similar to a FashionMNIST data point.
input_image = torch.rand(3,28,28)
print(input_image.size())

# flatten the image
flatten = torch.nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# we take 20 output features, so that it can be visualized or observed in the output.
layer1 = torch.nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# appying non-linear activations
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = torch.nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

# putting all modules together
seq_modules = torch.nn.Sequential(
    flatten,
    layer1,
    torch.nn.ReLU(),
    torch.nn.Linear(20, 10)
)

logits = seq_modules(input_image)

softmax = torch.nn.Softmax(dim=1)
pred_probab = softmax(logits)
# print(pred_probab, pred_probab.shape)

# Looking into the actual model parameters for FashionMNIST now.
model = def_model.NeuralNetwork()
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")