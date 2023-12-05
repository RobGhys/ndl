import os

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class SingleVideoWithContext(Dataset):
    def __init__(
            self,
            root: str,
            videos: pd.DataFrame,
            label_map: pd.DataFrame,
            window_size: int = 150,
            window_stride: int = 1,
    ):
        super().__init__()
        self.root = root
        self.label_map = label_map
        self.videos = videos
        self.current_video = None
        self.window_size = window_size
        self.window_stride = window_stride

        self.targets = None
        self.embedding = None

    def _load_embedding(self, index):
        del self.embedding
        # print(f'index: {index}')
        # print(f'video: {self.videos}')
        self.current_video = self.videos.iloc[index]
        # print(self.current_video)
        # print(f"id: {self.current_video['id']}")
        embedding = np.load(os.path.join(self.root, f"{self.current_video['id']}.npy"))
        # print(embedding)

        return embedding

    def _load_targets(self):
        del self.targets
        current_video_id = self.current_video['id']
        video_targets = self.label_map[current_video_id]

        video_targets = np.array([f['label'] for f in video_targets])

        return video_targets

    def __len__(self):
        return self.videos.shape[0]
        # if self.embedding is not None:
        #     return len(self.embedding)
        # else:
        #     return 0

    def __getitem__(self, index):
        self.embedding = self._load_embedding(index)
        x = self.embedding
        # print(f'Shape of x: {x.shape}')
        self.targets = self._load_targets()
        targets = self.targets
        # print(f'Shape of targets: {targets.shape}')

        # Padding
        # noinspection PyTypeChecker
        x = np.pad(x, ((self.window_size - 1, 0), (0, 0)), mode='edge')
        # print(f'x after padding: {x.shape}')
        x = torch.from_numpy(x).float()

        # Sliding windows
        x = x.unfold(dimension=0, size=self.window_size, step=self.window_stride)
        # print(f'x after sliding windows: {x.shape}')

        x = x.permute(0, 2, 1)
        # print(f'x after permute: {x.shape}')

        return x, targets


if __name__ == '__main__':
    videos = pd.read_csv("../../../data/maps/mapping_test.csv")
    import json

    with open("../../../data/labels/ground_truth_testset.json", 'rb') as file:
        label_map = json.load(file)

    root = "../../../data/embeddings_test"

    dataset = SingleVideoWithContext(root, videos, label_map, window_size=150)

    example_windows, example_targets = dataset[0]

    print(f'example_windows shape: {example_windows.shape}')
    print(f'example_targets shape: {example_targets.shape}')
