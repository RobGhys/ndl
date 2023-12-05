import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class FramesDataset(Dataset):
    def __init__(
            self,
            root: str,
            videos: pd.DataFrame,
            label_map: pd.DataFrame,
            transform=None,
    ):
        super().__init__()
        self.root = root
        self.videos = videos
        self.transform = transform

        print("Loading embeddings...")
        self.embeddings = []
        self._load_embeddings()

        print("Loading targets...")
        self.targets = []
        self._load_targets(label_map)

        print("Indexing frames...")
        self.frame_index = self._create_frame_index()

        print("Dataset loaded.")

    def _load_embeddings(self):
        for _, video in self.videos.iterrows():
            embedding = np.load(os.path.join(self.root, f"{video['id']}.npy"))
            self.embeddings.append(embedding)

    def _load_targets(self, label_map: pd.DataFrame):
        for _, video in self.videos.iterrows():
            video_targets = label_map[video['id']]
            video_targets = np.array([f['label'] for f in video_targets])
            self.targets.append(video_targets)

    def _create_frame_index(self):
        index = []
        for embed_idx, embed in enumerate(self.embeddings):
            for frame_nb in range(embed.shape[0]):
                index.append((embed_idx, frame_nb))
        return index

    def __len__(self):
        return len(self.frame_index)

    def __getitem__(self, index: int):
        embed_idx, frame_idx = self.frame_index[index]
        frame = self.embeddings[embed_idx][frame_idx]
        target = self.targets[embed_idx][frame_idx]

        if self.transform is not None:
            frame = self.transform(frame)

        return frame, target


class FramesWithContextDataset(FramesDataset):
    def __init__(self, *args, context_size: int, padding: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_size = context_size
        self.padding = padding

    def __getitem__(self, index: int):
        embed_idx, frame_idx = self.frame_index[index]

        window_start = max(0, frame_idx - self.context_size)
        frame_with_context = self.embeddings[embed_idx][window_start:frame_idx+1]

        n_frames = frame_with_context.shape[0]
        if self.padding and n_frames < self.context_size + 1:
            padding = self.context_size - n_frames + 1
            # noinspection PyTypeChecker
            frame_with_context = np.pad(frame_with_context, ((padding, 0), (0, 0)), mode='edge')

        if self.transform is not None:
            frame_with_context = self.transform(frame_with_context)

        target = self.targets[embed_idx][frame_idx]
        return frame_with_context, target
