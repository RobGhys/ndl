from torch import nn


class ResNet50GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_layers: int, dropout=0.3):
        super().__init__()
        self.rnn = nn.GRU(
            input_size,
            hidden_size,
            batch_first=True,
            num_layers=n_layers,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # X: (B, T, F_in) -> (B, T, H)
        outs, _ = self.rnn(x)
        # X: (B, T, H) -> (B, T, 1)
        outs = self.fc(outs[:, -1])
        return outs


if __name__ == '__main__':
    import torch
    example_x = torch.randn(8, 100, 2048)
    model = ResNet50GRU(input_size=2048, hidden_size=256, n_layers=1)
    logits = model(example_x)
    print(logits.shape)
