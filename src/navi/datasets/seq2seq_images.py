import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os


def load_image(path: str) -> np.ndarray:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f).convert('RGB')
        return np.array(img)


class SequenceToSequenceVideoImages(Dataset):
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
    ):
        super().__init__()
        self.root = root
        self.videos = videos

        self.video_index = []
        self._create_video_index()

        self.targets = []
        self._load_targets(label_map)

        self.windows = []
        self.window_size = window_size * (1 + skip_frames)
        self.window_stride = window_stride * (1 + skip_frames)
        self.skip_frames = skip_frames

        self.drop_empty_windows = drop_empty_windows
        self._create_windows()

        self.transforms = transforms

    def _create_video_index(self):
        for row_nb, video in self.videos.iterrows():
            video_dir = os.path.join(self.root, video['directory'])
            video_images = os.listdir(video_dir)
            self.video_index.append((video_dir, video_images))

    def _load_targets(self, label_map: pd.DataFrame):
        for _, video in self.videos.iterrows():
            video_targets = label_map[video['id']]
            video_targets = np.array([f['label'] for f in video_targets])
            self.targets.append(video_targets)

    def _create_windows(self):
        for video_index, (_, video_images) in enumerate(self.video_index):
            video_len = len(video_images)
            for window_end in range(video_len, 0, -self.window_stride):
                window_start = max(window_end - self.window_size, 0)
                window = (video_index, window_start, window_end)

                is_empty = self.targets[video_index][window_start:window_end].sum() == 0
                if (not self.drop_empty_windows) or (not is_empty):
                    self.windows.append(window)

    def _load_window(self, video_path, video_images, window_start, window_end):
        step = self.skip_frames + 1
        images = video_images[window_start:window_end:step]
        images = list(map(lambda img: load_image(os.path.join(video_path, img)), images))
        # images = Parallel(n_jobs=self.num_jobs)(delayed(load_image)(img_path) for img_path in images)
        images = np.stack(images)

        if images.shape[0] * step < self.window_size:
            padding = self.window_size - images.shape[0]
            # noinspection PyTypeChecker
            images = np.pad(images, ((padding, 0), (0, 0), (0, 0), (0, 0)), "edge")

        return images

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index: int):
        video_index, window_start, window_end = self.windows[index]
        video_path, video_images = self.video_index[video_index]
        window = self._load_window(video_path, video_images, window_start, window_end)
        label = self.targets[video_index][window_start:window_end:self.skip_frames+1]

        if self.transforms:
            window = self.transforms(window)

        return window, label


if __name__ == '__main__':
    videos = pd.read_csv('../../data/maps/mapping.csv')
    videos = videos.loc[videos['directory'].isin(
        ['video_20211108-170844990 - 20211108-170844990', 'video_20220107-161940816 - 20220107-161940816'])]

    import json
    with open("../../prep/predictions_old.json", 'rb') as file:
        label_map = json.load(file)

    dataset = SequenceToSequenceVideoImages(
        root="/run/media/ppoitier/ppoitier/datasets/navi/videos",
        videos=videos,
        label_map=label_map,
        window_size=100,
        window_stride=100,
        skip_frames=1,
        drop_empty_windows=True,
    )

    from tqdm import tqdm
    for window in tqdm(dataset):
        ...
