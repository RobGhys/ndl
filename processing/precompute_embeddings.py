import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as T
from tqdm import tqdm

from src.navi.datasets.single_video import SingleVideoDataset


def compute_video_embeddings(video_id: str, video_path: str, model, device, with_progress_bar=False):
    transform = T.Compose([
        T.Resize(256, antialias=True),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    embeddings = []
    dataset = SingleVideoDataset(
        os.path.join(video_path),
        transforms=transform,
    )
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4)

    for frames in tqdm(dataloader, total=len(dataloader), desc=video_id, disable=(not with_progress_bar)):
        frames = frames.to(device)
        embedding = model(frames)
        embeddings.append(embedding.cpu())
    return torch.concat(embeddings)


def compute_embeddings(root: str, videos: pd.DataFrame, dest_dir_path: str):
    device = 'cuda'

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    # Remove the tail and keep the embeddings
    model.fc = nn.Identity()
    # Freeze th ResNet50 layers
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for video_idx, video in videos.iterrows():
            video_id = video['id']
            video_path = os.path.join(root, video['directory'])
            video_embeddings = compute_video_embeddings(video_id, video_path, model, device, with_progress_bar=True)
            np.save(os.path.join(dest_dir_path, f"{video_id}.npy"), video_embeddings.numpy())


if __name__ == '__main__':
    videos = pd.read_csv("../../data/maps/mapping.csv")
    local_videos = ['video_20211108-170844990 - 20211108-170844990', 'video_20220107-161940816 - 20220107-161940816']
    videos = videos.loc[videos['directory'].isin(local_videos)]

    compute_embeddings(
        root="/run/media/ppoitier/ppoitier/datasets/navi/videos",
        videos=videos,
        dest_dir_path="/run/media/ppoitier/ppoitier/datasets/navi/embeddings",
    )
