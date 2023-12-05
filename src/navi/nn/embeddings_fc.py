
from torch import nn


class ResNet50FC(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # X: (B, T, F_in) -> (B, T, H)
        x = self.relu(self.fc1(x))
        # X: (B, T, H) -> (B, T, 1)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    import torch
    example_x = torch.randn(8, 100, 2048)
    model = ResNet50FC(input_size=2048, hidden_size=256)
    logits = model(example_x)
    print(logits.shape)

