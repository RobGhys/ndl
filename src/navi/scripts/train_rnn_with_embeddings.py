import argparse
import json
import os
import time
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import torch

from navi.datasets.seq2seq_embeddings import SequenceToSequenceEmbeddings
from navi.nn.models_using_embeddings import ResNet50GRU
from navi.nn.embeddings_fc import ResNet50FC
from navi.trainers.seq2seq import SequenceToSequenceTrainer
from navi.transforms import ToTensor
from navi.utils import Utils

from torch import nn, optim


def compute_positive_weight(dataset):
    counts = [0, 0]
    for _, targets in dataset:
        counts[0] += (targets == 0).sum().item()
        counts[1] += (targets == 1).sum().item()
    total = sum(counts)
    frequencies = [v/total for v in counts]
    print(f'Frequencies of labels in train dataset: pos={frequencies[1]} ; neg={frequencies[0]}')
    weight = 1 / frequencies[1]
    print(f'Positive weight: {weight}')
    return weight


def launch_experiment(
        topmodel_name,
        embeddings_dir,
        video_csv,
        label_map,
        snapshot_dir,
        snapshot_interval,
        device,
        seed,
        skip_frames,
        drop_empty_windows,
        wandb_key,
        model_type: str,
        target_metric: str,
):
    print("Loading metadata...")
    videos = pd.read_csv(video_csv)
    videos = videos.sample(frac=1, random_state=seed)
    with open(label_map, 'rb') as file:
        label_map = json.load(file)

    logging.basicConfig(filename=f"{snapshot_dir}/log.txt",
                        level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    utils = Utils(snapshot_dir)
    k_fold = KFold(n_splits=5, shuffle=True, random_state=seed)
    params = utils.get_all_grid_params()
    utils.log_params()  # add csv file with all combinations and their key

    for idx, (train_indices, test_indices) in enumerate(k_fold.split(videos, groups=videos['id'])):
        train_videos = videos.iloc[train_indices]
        test_videos = videos.iloc[test_indices]

        transforms = ToTensor()
        target_transforms = ToTensor()

        for i, grid_param in enumerate(params):
            split_path = os.path.join(Path(snapshot_dir), f'grid_{i}')
            os.makedirs(split_path, exist_ok=True)

            model_name = topmodel_name + '_grid_' + str(i) + '_fold_' + str(idx)
            print(f'Run # {i} | k = {idx} | Current model parameters: {grid_param} | Model name: {model_name}')
            logger.info(f'Run # {i} | k = {idx} | Current model parameters: {grid_param} | Model name: {model_name}')

            print("Loading dataset...")

            datasets = {
                x: SequenceToSequenceEmbeddings(
                    embeddings_dir,
                    train_videos if x == 'train' else test_videos,
                    label_map,
                    window_size=grid_param['window_size'],
                    window_stride=int(grid_param['window_size'] * (1.0 - grid_param['window_overlap'])),
                    skip_frames=skip_frames,
                    drop_empty_windows=drop_empty_windows,
                    transforms=transforms,
                    target_transforms=target_transforms,
                )
                for x in ['train', 'test']
            }

            data_loaders = {
                x: DataLoader(datasets[x], batch_size=grid_param['batch_size'], shuffle=True, drop_last=True)
                for x in ['train', 'test']
            }

            pos_weight = compute_positive_weight(datasets['train'])
            pos_weight = torch.tensor(pos_weight).to(device)

            if model_type == 'gru':
                model = ResNet50GRU(
                    input_size=2048,
                    hidden_size=grid_param['hidden_size'],
                    n_layers=grid_param['n_layers'],
                )
            elif model_type == 'gru_nnunet':
                model = ResNet50GRU(
                    input_size=2016,
                    hidden_size=grid_param['hidden_size'],
                    n_layers=grid_param['n_layers'],
                )
            elif model_type == 'fc':
                model = ResNet50FC(
                    input_size=2048,
                    hidden_size=grid_param['hidden_size'],
                )
            elif model_type == 'fc_nnunet_embeddings':
                model = ResNet50FC(
                    input_size=2016,
                    hidden_size=grid_param['hidden_size'],
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}.")

            if target_metric == 'balanced_accuracy':
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.AdamW(model.parameters(), lr=grid_param['learning_rate'])
            lr_scheduler = utils.get_scheduler(scheduler_name=grid_param['scheduler'], optimizer=optimizer)

            trainer = SequenceToSequenceTrainer(
                model_name, model, criterion, optimizer,
                device=device,
                gradient_clipping=True,
                snapshot_dir=split_path,
                fold_nb=idx,
                snapshot_interval=snapshot_interval,
                lr_scheduler=lr_scheduler,
                tracked_metric=target_metric
            )

            print("Start training...")

            start_time = time.time()

            trainer.start_wandb_logging({
                'rnn': model_type.upper(),
                'learning_rate': grid_param['learning_rate'],
                'epochs': grid_param['epochs'],
                'hidden_size': grid_param['hidden_size'],
                'n_rnn_layers': grid_param['n_layers'],
                'criterion': 'BCE',
                'optimizer': 'AdamW',
                'input_type': 'precomputed_embeddings',
            }, key=wandb_key)
            best_result = trainer.launch_training(data_loaders['train'], data_loaders['test'], grid_param['epochs'])
            print(f'Best balanced accuracy: {best_result}')
            logger.info(f'Best balanced accuracy: {best_result}')
            trainer.stop_wandb_logging()

            end_time = time.time() - start_time
            print("Done.")  # end of one model
            print(f"Took {end_time // 3600} hours, {(end_time % 3600) // 60} minutes and {end_time % 60} seconds.")

        print(f'Finished all training for fold # {idx}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="Navi training with GRU and embeddings.",
        description="/",
    )
    parser.add_argument('--topmodel-name', required=True)
    parser.add_argument('--embeddings-dir', required=True)
    parser.add_argument('--video-csv', required=True)
    parser.add_argument('--label-map', required=True)
    parser.add_argument('--snapshot-dir', required=True)
    parser.add_argument('--snapshot-interval', default=5, type=int)
    parser.add_argument('--device', default="cuda")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--skip-frames', default=0, type=int)
    parser.add_argument('--drop_empty_windows', default=False, type=bool)
    parser.add_argument('--wandb-key', required=False)
    parser.add_argument('--model-type', default="gru")
    parser.add_argument('--target-metric', default="balanced_accuracy")

    args = vars(parser.parse_args())

    launch_experiment(**args)
