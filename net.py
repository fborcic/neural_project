import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_layer1 = nn.Conv2d(in_channels=1, out_channels=32,
                kernel_size=3, stride=1, padding=1)
        self.max_pool12 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.c_layer2 = nn.Conv2d(in_channels=32, out_channels=64,
                kernel_size=3, stride=1, padding=1)
        self.max_pool23 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout23 = nn.Dropout(p=0.1)
        self.layer3 = nn.Linear(7*7*64, 128)
        self.dropout34 = nn.Dropout(p=0.5)
        self.layer4 = nn.Linear(128, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.c_layer1(x)
        x = self.relu(x)
        x = self.max_pool12(x)
        x = self.c_layer2(x)
        x = self.relu(x)
        x = self.max_pool23(x)
        x = torch.flatten(x, 1)
        x = self.dropout23(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.dropout34(x)
        x = self.layer4(x)
        prediction = nn.functional.log_softmax(x, dim=1)
        return prediction
