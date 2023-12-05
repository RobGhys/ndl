import os

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class VideoWithContext(Dataset):
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
        self.videos = videos
        self.window_size = window_size
        self.window_stride = window_stride

        self.targets = []
        self._load_targets(label_map)

        self.embeddings = []
        self._load_embeddings()

    def _load_embeddings(self):
        for _, video in self.videos.iterrows():
            embedding = np.load(os.path.join(self.root, f"{video['id']}.npy"))
            self.embeddings.append(embedding)

    def _load_targets(self, label_map: pd.DataFrame):
        for _, video in self.videos.iterrows():
            vid_id = video['id']
            #print(f'video id: {vid_id}')
            video_targets = label_map[video['id']]
            video_targets = np.array([f['label'] for f in video_targets])
            self.targets.append(video_targets)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, index):
        x = self.embeddings[index]
        print(f'Shape of x: {x.shape}')
        targets = self.targets[index]
        print(f'Shape of targets: {targets.shape}')

        # Padding
        # noinspection PyTypeChecker
        x = np.pad(x, ((self.window_size - 1, 0), (0, 0)), mode='edge')
        print(f'x after padding: {x.shape}')

        x = torch.from_numpy(x).float()

        # Sliding windows
        x = x.unfold(dimension=0, size=self.window_size, step=self.window_stride)
        x = x.permute(0, 2, 1)

        return x, targets


if __name__ == '__main__':
    videos = pd.read_csv("../../../data/maps/mapping_test.csv")
    import json
    with open("../../../data/labels/ground_truth_testset.json", 'rb') as file:
        label_map = json.load(file)

    root = "../../../data/embeddings_test"

    dataset = VideoWithContext(root, videos, label_map, window_size=150)
    example_windows, example_targets = dataset[0]

    print(example_windows.shape)
    print(example_targets.shape)

