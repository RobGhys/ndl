import torch


class ToTensor:
    def __init__(self, dtype=torch.float):
        self.dtype = dtype

    def __call__(self, x):
        return torch.from_numpy(x).to(self.dtype)
