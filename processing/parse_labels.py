import json
import numpy as np
import pandas as pd


def video_labels_to_nparray(video_id: str, label_map: dict):
    video_labels = label_map.get(video_id)
    if video_labels is None:
        raise ValueError(f'Missing video in label map: {video_id}')
    video_labels = [frame_label['label'] for frame_label in video_labels]
    return np.array(video_labels)


def load_label_map(json_path: str):
    with open(json_path, 'rb') as file:
        return json.load(file)


if __name__ == '__main__':
    videos = pd.read_csv("../../data/maps/mapping.csv")
    local_videos = ['video_20211108-170844990 - 20211108-170844990', 'video_20220107-161940816 - 20220107-161940816']

    # Keep only local videos
    videos = videos.loc[videos['directory'].isin(local_videos)]
    video_ids = videos['id'].to_list()

    label_map = load_label_map("../../prep/predictions.json")
    label_map = {k: v for (k, v) in label_map.items() if k in video_ids}
    label_map = {k: video_labels_to_nparray(k, label_map) for (k, v) in label_map.items()}

    np.save("/run/media/ppoitier/ppoitier/datasets/navi/targets.npy", label_map, allow_pickle=True)
