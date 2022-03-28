import numpy as np
import torch
from torch import nn
import torch.optim as optim
from scipy.spatial.distance import cosine, euclidean

# Local modules
from WordVectors import WordVectors, intersection
from s4 import s4
from alignment import align


# Initialize random seeds
np.random.seed(1)
torch.manual_seed(1)


class S4Network(nn.Module):
    def __init__(self, dim=100):
        super(S4Network, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        # x = self.flatten(x)
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).type(torch.FloatTensor)
        logits = self.linear_relu_stack(x)
        return logits

    def fit(self, x, y, epochs=2):
        running_loss = 0.0
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

        if type(x) == np.ndarray:
            x = torch.from_numpy(x).type(torch.FloatTensor)
        if type(y) == np.ndarray:
            y = torch.from_numpy(y).type(torch.FloatTensor)

        for epoch in range(epochs):
            inputs, labels = x, y.reshape(-1, 1)
            running_loss = 0.0
            optimizer.zero_grad()
            outputs = self(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # print("epoch %d loss: %.3f" % (epoch+1, running_loss))

        return running_loss

    def predict(self, x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).type(torch.FloatTensor)
        logits = self(x)
        # proba = nn.Softmax(dim=1)(logits)
        proba = nn.Sigmoid()(logits)
        return proba


if __name__ == "__main__":
    pass
