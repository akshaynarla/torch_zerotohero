import torch

# training process. Do forward pass and then backpropogation
# forward pass makes a prediction with the current weights and bias
# backpropogation optimizes it by minimizing loss, calculated by difference in forward pass prediction and actual label.
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    print("Currently training num. samples:", size)
    model.train()
    
    # go over each dataloader element. X data. y true label.
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error for a single data point of the batch by forward pass using the loss function.
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()                 # gradients of loss w.r.t each parameter is given
        optimizer.step()                # gradient descent and adjust parameters
        optimizer.zero_grad()           # reset to prevent double-counting

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# test dataset to test the model in the end of each epoch
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")