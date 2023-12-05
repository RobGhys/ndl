import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class FullVideoEmbeddings(Dataset):
    def __init__(
            self,
            root: str,
            videos: pd.DataFrame,
            label_map: pd.DataFrame,
            skip_frames: int = 0,
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
        self.skip_frames = skip_frames

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

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, index: int):
        features = self.embeddings[index]
        targets = self.targets[index]

        if self.skip_frames > 0:
            features = features[::self.skip_frames+1, :]
            targets = targets[::self.skip_frames+1, :]

        if self.transforms:
            features = self.transforms(features)
        if self.target_transforms:
            targets = self.target_transforms(targets)

        return features, targets


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

    transform = ToTensor(dtype=torch.float)
    target_transform = ToTensor(dtype=torch.long)

    dataset = FullVideoEmbeddings(
        root,
        videos,
        label_map,
        transforms=transform,
        target_transforms=target_transform,
    )

    example_features, example_target = dataset[0]

    print(example_features.shape, example_target.shape)
