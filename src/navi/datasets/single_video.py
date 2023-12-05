import os

from PIL import Image
from torch.utils.data import Dataset


def load_image(path: str) -> Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert('RGB')


class SingleVideoDataset(Dataset):
    def __init__(self, video_dir_path: str, transforms=None):
        super().__init__()
        self.root = video_dir_path
        self.frames = sorted(os.listdir(video_dir_path))
        self.transforms = transforms

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index: int):
        img_path = os.path.join(self.root, self.frames[index])
        img = load_image(img_path)
        if self.transforms is not None:
            img = self.transforms(img)
        return img
