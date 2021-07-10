
import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 2000)
        self.fc4 = nn.Linear(2000, 10)

        self.fc5 = nn.Linear(10, 2000)
        self.fc6 = nn.Linear(2000, 500)
        self.fc7 = nn.Linear(500, 500)
        self.fc8 = nn.Linear(500, 784)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        u = self.fc4(x)

        x = self.fc5(u)
        x = torch.relu(x)
        x = self.fc6(x)
        x = torch.relu(x)
        x = self.fc7(x)
        x = torch.relu(x)
        x = self.fc8(x)

        return x, u

# FCM-Net structure
class FCMNet(nn.Module):
    def __init__(self):
        super(FCMNet, self).__init__()

        self.fc1 = nn.Linear(10, 100)
        self.fc2 = nn.Linear(100, 200)
        self.fc3 = nn.Linear(200, 10)
        self.last = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = self.last(x)
        return x