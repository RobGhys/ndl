from typing import Literal, Optional
import os
from copy import deepcopy

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score
import torch.optim.lr_scheduler as lr_scheduler

import wandb


class SequenceToSequenceTrainer:

    def __init__(
            self,
            name: str,
            model: nn.Module,
            criterion: nn.Module,
            optimizer: optim.Optimizer,
            fold_nb: int,
            lr_scheduler: Optional[lr_scheduler] = None,
            device: str = 'cuda',
            gradient_clipping: bool = False,
            snapshot_dir: str | None = None,
            snapshot_interval: int | None = None,
            tracked_metric: str = 'balanced_accuracy'
    ):
        self.name = name
        self.fold_nb = fold_nb
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.gradient_clipping = gradient_clipping

        self.MAX_EPOCH_NO_IMPROVEMENT = 10
        self.nb_epochs_no_improvement = 0
        self.use_wandb = False
        self.tracked_metric = tracked_metric

        self.best_metric = 0
        self.best_weights = None

        self.snapshot_dir = snapshot_dir
        self.snapshot_interval = snapshot_interval

    def perform_epoch(self, data_loader, mode: Literal['train', 'eval'], epoch_nb: int):
        if mode == 'train':
            self.model.train()
        elif mode == 'eval':
            self.model.eval()

        n_correct = 0
        n_total = 0
        losses = []

        true_labels = []
        predicted_labels = []

        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
        progress_bar.set_description(f"epoch {epoch_nb} - {mode}")
        grad_context = torch.inference_mode if mode == 'eval' else torch.enable_grad
        with grad_context():
            for b_index, (features, targets) in progress_bar:
                features = features.to(self.device)
                targets = targets.to(self.device)
                logits = self.model(features).squeeze(dim=-1)
                loss = self.criterion(logits, targets)
                losses.append(loss.item())

                if mode == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.gradient_clipping:
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                predictions = (logits.sigmoid() > 0.5).float()
                true_labels.append(targets.cpu().numpy())
                predicted_labels.append(predictions.cpu().numpy())

                n_total += predictions.numel()
                n_correct += (predictions == targets).sum().item()

        true_labels = np.stack(true_labels).reshape(-1)
        predicted_labels = np.stack(predicted_labels).reshape(-1)
        bal_acc = balanced_accuracy_score(true_labels, predicted_labels)
        #print(f'\nbalanced accuracy: {bal_acc}')
        recall = recall_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels)

        accuracy = n_correct / n_total
        #print(f'\nacc: {accuracy}')
        mean_loss = np.mean(losses)

        target_metric = self.get_target_metric(accuracy, bal_acc)  # metric to track

        if (mode == 'eval') & (self.nb_epochs_no_improvement < self.MAX_EPOCH_NO_IMPROVEMENT) & (target_metric > self.best_metric):
            self.best_weights = deepcopy(self.model.state_dict())
            self.nb_epochs_no_improvement = 0
            self.best_metric = target_metric
        else:
            self.nb_epochs_no_improvement += 1

        if self.use_wandb:
            wandb.log({
                f'accuracy/{mode}': accuracy,
                f'balanced_accuracy/{mode}': bal_acc,
                f'recall/{mode}': recall,
                f'precision/{mode}': precision,
                f'loss/{mode}': mean_loss,
            }, step=epoch_nb)

        if mode == 'train' and self.lr_scheduler:
            self.lr_scheduler.step(target_metric)
            after_lr = self.optimizer.param_groups[0]["lr"]

            if self.use_wandb:
                wandb.log({
                    'learning_rate': after_lr
                }, step=epoch_nb)

        if self.snapshot_dir and self.snapshot_interval and ((epoch_nb % self.snapshot_interval) == 0):
            self.save_weights(f'ep{epoch_nb:03d}')

    def get_target_metric(self, accuracy, bal_acc):
        target_metric = None
        if self.tracked_metric == 'accuracy':
            target_metric = accuracy
        elif self.tracked_metric == 'balanced_accuracy':
            target_metric = bal_acc
        return target_metric

    def launch_training(self, train_data_loader, eval_data_loader, n_epochs):
        for epoch in range(1, n_epochs + 1):
            self.perform_epoch(train_data_loader, mode='train', epoch_nb=epoch)
            self.perform_epoch(eval_data_loader, mode='eval', epoch_nb=epoch)
        if self.best_weights is not None and self.snapshot_dir:
            self.save_weights('best')

        return  self.best_metric
        #return self.best_weights

    def save_weights(self, suffix: str):
        dest_path = f"{self.snapshot_dir}/fold_{self.fold_nb}"
        os.makedirs(dest_path, exist_ok=True)
        torch.save(self.model.state_dict(), f"{dest_path}/{suffix}.pt")

    def start_wandb_logging(self, config: dict = None, key=None):
        if config is None:
            config = {}
        if key is not None:
            wandb.login(key=key)
        wandb.init(
            project=self.name,
            config=config,
        )
        self.use_wandb = True

    def stop_wandb_logging(self):
        if self.use_wandb:
            wandb.finish()
