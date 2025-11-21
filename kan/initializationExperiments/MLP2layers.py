import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size=2, hidden_sizes=3, num_classes=1):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_sizes),
            nn.ReLU(),
            nn.Linear(hidden_sizes, num_classes)
        )

    def forward(self, x):
        y_pred = self.linear_relu_stack(x)
        return y_pred

def train_loop(dataloader, model, loss_fn, optimizer):
        device = "cpu"
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # if batch % 100 == 0:
            #     loss, current = loss.item(), batch * 64 + len(X)
            #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
        device = "cpu"
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test loss: {test_loss:>7f}, Accuracy: {correct:>7f}")