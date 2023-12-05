import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, rgb=True):
        super(SimpleCNN, self).__init__()
        if rgb:
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0)
        else:
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.relu = nn.ReLU()
        
        self.fc1 = nn.Linear(12 * 12 * 128, 64)

        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))

        #print("Shape of x after convolutions: ", x.shape)

        # Flatten for fc layer
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
