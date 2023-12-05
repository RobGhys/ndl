import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm


class SequenceToSequenceEmbeddings(Dataset):
    def __init__(
            self,
            root: str,
            videos: pd.DataFrame,
            label_map: pd.DataFrame,
            window_size: int = 100,
            window_stride: int = 80,
            skip_frames: int = 0,
            drop_empty_windows: bool = False,
            transforms=None,
            target_transforms=None,
    ):
        super().__init__()
        self.root = root
        self.videos = videos

        self.embeddings = []
        self._load_embeddings()

        self.targets = []
        self._load_targets(label_map)

        self.windows = []
        self.window_size = window_size * (1 + skip_frames)
        self.window_stride = window_stride * (1 + skip_frames)
        self.skip_frames = skip_frames

        self.drop_empty_windows = drop_empty_windows
        self._create_windows()

        self.transforms = transforms
        self.target_transforms = target_transforms

    def _load_embeddings(self):
        for _, video in self.videos.iterrows():
            embedding = np.load(os.path.join(self.root, f"{video['id']}.npy"))
            self.embeddings.append(embedding)

    def _load_targets(self, label_map: pd.DataFrame):
        for _, video in self.videos.iterrows():
            video_targets = label_map[video['id']]
            video_targets = np.array([f['label'] for f in video_targets])
            self.targets.append(video_targets)

    def _create_windows(self):
        for video_index, video_embedding in enumerate(self.embeddings):
            video_len = video_embedding.shape[0]
            for window_end in range(video_len, 0, -self.window_stride):
                window_start = max(window_end - self.window_size, 0)
                window = (video_index, window_start, window_end)
                is_empty = self.targets[video_index][window_start:window_end].sum() == 0
                if (not self.drop_empty_windows) or (not is_empty):
                    self.windows.append(window)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index: int):
        video_index, window_start, window_end = self.windows[index]
        embedding = self.embeddings[video_index]

        step = self.skip_frames + 1
        window = embedding[window_start:window_end:step]
        targets = self.targets[video_index][window_start:window_end:step]

        if window.shape[0] * step < self.window_size:
            padding = self.window_size - window.shape[0]
            # noinspection PyTypeChecker
            window = np.pad(window, ((padding, 0), (0, 0)), "edge")
            # noinspection PyTypeChecker
            targets = np.pad(targets, ((padding, 0),), "edge")

        if self.transforms:
            window = self.transforms(window)
        if self.target_transforms:
            targets = self.target_transforms(targets)

        return window, targets


if __name__ == '__main__':
    videos = pd.read_csv("../../../data/maps/mapping.csv")
    videos = videos.loc[videos['directory'].isin(
        ['video_20211108-170844990 - 20211108-170844990', 'video_20220107-161940816 - 20220107-161940816'])]
    root = "/run/media/ppoitier/ppoitier/datasets/navi/embeddings"

    import json
    with open("../../../prep/predictions_old.json", 'rb') as file:
        label_map = json.load(file)

    import torch
    from src.navi.transforms import ToTensor

    from torch.utils.data import DataLoader

    transform = ToTensor(dtype=torch.float)
    target_transform = ToTensor(dtype=torch.long)

    dataset = SequenceToSequenceEmbeddings(
        root,
        videos,
        label_map,
        window_size=10,
        window_stride=8,
        transforms=transform,
        target_transforms=target_transform,
    )

    # dataloader = DataLoader(dataset, batch_size=8, num_workers=4)
    #
    # example_window, example_targets = next(iter(dataloader))
    # print(example_window.shape)
    # print(example_targets.shape)
    #
    # print(example_targets)

    for index, (window, label) in tqdm(enumerate(dataset)):
        if label.shape[0] != 10:
            print(index, label.shape)
