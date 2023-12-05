import argparse
import json
import os
import time
import logging
from pathlib import Path

import pandas as pd
import wandb
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import torch

from navi.datasets.frames_embeddings import FramesWithContextDataset
from navi.nn.models_using_embeddings import ResNet50GRU
from navi.nn.embeddings_fc import ResNet50FC
from navi.trainers.seq2seq import SequenceToSequenceTrainer
from navi.transforms import ToTensor
from navi.utils import Utils
from navi.sampler.imbalance_sampler import ImbalancedDatasetSampler

from torch import nn, optim


def load_metadata(video_csv_path: str, label_map_path: str):
    videos = pd.read_csv(video_csv_path)
    with open(label_map_path, 'rb') as file:
        label_map = json.load(file)
    return videos, label_map


def launch_experiment(
        topmodel_name,
        embeddings_dir,
        video_csv,
        label_map,
        snapshot_dir,
        snapshot_interval,
        device,
        seed,
        wandb_key,
        model_type: str,
        n_splits: int,
):
    print("Loading metadata...")
    videos, label_map = load_metadata(video_csv, label_map)

    logging.basicConfig(filename=f"{snapshot_dir}/log.txt",
                        level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    utils = Utils(snapshot_dir)
    k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    params = utils.get_all_grid_params()
    utils.log_params()  # add csv file with all combinations and their key

    wandb.init(
        project='navi_rnn',
        config={
        }
    )

    for idx, (train_indices, test_indices) in enumerate(k_fold.split(videos, groups=videos['id'])):
        train_videos = videos.iloc[train_indices]
        test_videos = videos.iloc[test_indices]

        transforms = ToTensor()

        for i, grid_param in enumerate(params):
            split_path = os.path.join(Path(snapshot_dir), f'grid_{i}')
            os.makedirs(split_path, exist_ok=True)

            model_name = topmodel_name + '_grid_' + str(i) + '_fold_' + str(idx)
            print(f'Run # {i} | k = {idx} | Current model parameters: {grid_param} | Model name: {model_name}')
            logger.info(f'Run # {i} | k = {idx} | Current model parameters: {grid_param} | Model name: {model_name}')

            print("Loading dataset...")

            datasets = {
                x: FramesWithContextDataset(
                    embeddings_dir,
                    train_videos if x == 'train' else test_videos,
                    label_map,
                    transform=transforms,
                    context_size=grid_param['context_size'],
                )
                for x in ['train', 'test']
            }

            sampler = ImbalancedDatasetSampler(datasets['train'], num_samples=200000)
            data_loaders = {
                'train': DataLoader(
                    datasets['train'], batch_size=grid_param['batch_size'], sampler=sampler, drop_last=True),
                'test': DataLoader(
                    datasets['test'], batch_size=grid_param['batch_size'], shuffle=False, drop_last=True)
            }

            # pos_weight = compute_positive_weight(datasets['train'])
            # pos_weight = torch.tensor(pos_weight).to(device)

            if model_type == 'gru':
                model = ResNet50GRU(
                    input_size=2048,
                    hidden_size=grid_param['hidden_size'],
                    n_layers=grid_param['n_layers'],
                )
            elif model_type == 'fc':
                model = ResNet50FC(
                    input_size=2048,
                    hidden_size=grid_param['hidden_size'],
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}.")

            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.AdamW(model.parameters(), lr=grid_param['learning_rate'])

            trainer = SequenceToSequenceTrainer(
                model_name, model, criterion, optimizer,
                device=device,
                gradient_clipping=True,
                snapshot_dir=split_path,
                fold_nb=idx,
                snapshot_interval=snapshot_interval,
            )
            trainer.use_wandb = True

            print("Start training...")

            start_time = time.time()

            # trainer.start_wandb_logging({
            #     'rnn': model_type.upper(),
            #     'learning_rate': grid_param['learning_rate'],
            #     'epochs': grid_param['epochs'],
            #     'hidden_size': grid_param['hidden_size'],
            #     'n_rnn_layers': grid_param['n_layers'],
            #     'criterion': 'BCE',
            #     'optimizer': 'AdamW',
            #     'input_type': 'precomputed_embeddings',
            # }, key=wandb_key)
            trainer.launch_training(data_loaders['train'], data_loaders['test'], grid_param['epochs'])
            # trainer.stop_wandb_logging()

            end_time = time.time() - start_time
            print("Done.")  # end of one model
            print(f"Took {end_time // 3600} hours, {(end_time % 3600) // 60} minutes and {end_time % 60} seconds.")

        print(f'Finished all training for fold # {idx}')


    wandb.finish()


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
    parser.add_argument('--wandb-key', required=False)
    parser.add_argument('--model-type', default="gru")
    parser.add_argument('--n-splits', default=5, type=int)

    args = vars(parser.parse_args())

    launch_experiment(**args)
