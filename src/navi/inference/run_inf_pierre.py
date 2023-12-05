import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import torch
import json

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

import seaborn as sns
from tqdm import tqdm
import argparse

from navi.datasets.seq2seq_embeddings_full import FullVideoEmbeddings
from navi.datasets.single_video_with_context import SingleVideoWithContext
from navi.datasets.video_with_context import VideoWithContext
from navi.nn.models_using_embeddings import ResetNet50GRU


def create_dfs(video_csv: str, label_map_path: str) -> dict:
    # Dict to store all dataframes
    result: dict = {}
    mapping_df = pd.read_csv(video_csv)  # df that maps a video id to its folder

    # Open json file
    with open(label_map_path, 'r') as f:
        label_map: pd.DataFrame = json.load(f)

        for index, row in mapping_df.iterrows():
            video_id = row['id']
            if video_id in label_map:
                video_data = label_map[video_id]
                if isinstance(video_data, list) and all(isinstance(item, dict) for item in video_data):
                    video_data_df = pd.DataFrame(video_data)
                else:
                    raise Exception(f'Invalid data format for video ID: {video_id}')
                result[video_id] = video_data_df
            else:
                raise Exception(f'Could not find the following id in json file: {video_id}')

    return result


def verify_matching_keys(dataframe_dict, df_dict_one_model):
    keys_dataframe_dict = set(dataframe_dict.keys())
    keys_df_dict_one_model = set(df_dict_one_model.keys())

    assert keys_dataframe_dict == keys_df_dict_one_model, "Keys from dataframe_dict and df_dict do not match!"


def concat_dataframe(predictions_df: pd.DataFrame, labels_df: pd.DataFrame):
    if len(predictions_df) != len(labels_df):
        raise ValueError('DataFrames do not have the same length')
    result: pd.DataFrame = pd.merge(predictions_df, labels_df, on='frame')

    return result


def main_inference(weights_dir: str, k: int, embeddings_dir: str, video_csv: str,
                   label_map_path: str, window_size: int,
                   batch_size: int, hidden_size: int, n_layers: int):
    output_dir = '/home/rob/Documents/Github/navi_lstm/output/inference'
    output_sub_dir: dict = {}
    dataframe_dict = create_dfs(video_csv, label_map_path)
    test_videos = pd.read_csv(video_csv)

    print('Pre Processing Dataframes')
    for key, dataframe in dataframe_dict.items():
        try:
            # pre_process_df(dataframe, k)
            # dataframe_dict[key] = dataframe
            # stores the path to save output
            output_sub_dir[key] = get_output_sub_dir(output_dir, key)
        except Exception as e:
            raise Exception(f'Error during Pre Processing of Dataframes for key: {key} | \nException: {e}')

    video_ids = sorted(dataframe_dict.keys())

    print('Collecting Models')
    model_paths = get_models(weights_dir)
    with open(label_map_path, 'rb') as file:
        label_map = json.load(file)

    df_dict_one_model = infer(model_paths=model_paths,
                              embeddings_dir=embeddings_dir, test_videos=test_videos, label_map=label_map,
                              window_size=window_size, batch_size=batch_size, hidden_size=hidden_size,
                              n_layers=n_layers, video_ids=video_ids)

    verify_matching_keys(dataframe_dict, df_dict_one_model)
    
    # For each df (1 per video) in df_dict_one_model
    for key, dataframe in df_dict_one_model.items():
        # Concat predictions with labels and replace dataframe
        df_dict_one_model[key] = concat_dataframe(dataframe, dataframe_dict[key])
        del dataframe
        dataframe = df_dict_one_model[key]

        # average_prediction(dataframe, range(k))
        average_prediction(dataframe, range(1))
        # if not os.path.exists(output_sub_dir[key]):
        #     os.makedirs(output_sub_dir[key])

        output_file_path = os.path.join(output_sub_dir[key], 'probas_labels.csv')
        dataframe.to_csv(output_file_path, index=False)
        get_confusion_matrix(true_labels=dataframe['label'].tolist(), avg_model_labels=dataframe['pred_avg'].tolist(),
                             file_name=f'confusion_matrix.png', output_dir=output_sub_dir[key], key=key)

        plot_labels(dataframe, f'Average Predictions {key}', output_dir=output_sub_dir[key])

        results_frame_df = evaluate_models(dataframe)
        output_file_path = os.path.join(output_sub_dir[key], 'eval_output_post.csv')
        results_frame_df.to_csv(output_file_path, index=False)


def get_output_sub_dir(output_dir: str, key: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_sub_dir = os.path.join(output_dir, key)
    if not os.path.exists(output_sub_dir):
        os.makedirs(output_sub_dir)

    return output_sub_dir


def infer(model_paths: list, embeddings_dir: str, test_videos: pd.DataFrame, label_map: pd.DataFrame,
          window_size: int, batch_size: int, hidden_size: int, n_layers: int, video_ids: list) -> dict:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    model_nb = 0

    model = get_model(device, model_paths[model_nb], hidden_size, n_layers)

    # Get Dataset
    inference_dataset = FullVideoEmbeddings(
        embeddings_dir,
        test_videos,
        label_map,
        #window_size=window_size
    )
    # Get DataLoader
    inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False)
    print(next(iter(inference_loader))[0].shape)

    # Run inference
    df_dict_one_model: dict = run_batches(device, model, inference_loader, model_nb=model_nb, video_ids=video_ids)

    return df_dict_one_model


def run_batches(device, model, inference_loader, model_nb, video_ids: list) -> dict:
    results: dict = {}

    for batch, (sequence, labels) in tqdm(enumerate(inference_loader), desc='inference'):
        print(f'Batch: {batch}')
        with torch.inference_mode():
            # sequence: (1, T, 2048)
            sequence = sequence.to(device)
            labels = labels.squeeze().numpy()

            # (1, T, 2048) -> (1, T, 1) -> (T)
            logits = model(sequence).squeeze()
            # (T)
            probabilities = logits.sigmoid().cpu().numpy()
            predicted_labels = (probabilities > 0.5).astype('int32')
            print(predicted_labels.shape)

            n_frames = logits.shape[0]
            print(labels)
            print(predicted_labels)
            print(labels == predicted_labels)
            n_correct = (labels == predicted_labels).sum()
            accuracy = n_correct / n_frames
            print(f'accuracy: {accuracy}')

            batch_df = pd.DataFrame({
                'frame': range(n_frames),
                f'proba_{model_nb}': probabilities
            })

            key = video_ids[batch]
            results[key] = batch_df
    return results


def get_model(device, model_path, hidden_size, n_layers):
    model = ResetNet50GRU(
        input_size=2048,
        hidden_size=hidden_size,
        n_layers=n_layers,
    )
    checkpoint = torch.load(model_path)

    model.load_state_dict(checkpoint)
    model.eval()
    model = model.to(device)
    return model


def average_prediction(frame_df, k_range):
    prob_cols = [f'proba_{k}' for k in k_range]
    for col in prob_cols:
        if col not in frame_df.columns:
            raise ValueError(f"Missing column {col} in DataFrame")

    # Average probability for each row (axis=1 -> column)
    frame_df['proba_avg'] = frame_df[prob_cols].mean(axis=1)
    # Average prediction
    frame_df['pred_avg'] = np.where(frame_df['proba_avg'] > 0.5, 1, 0)


def get_confusion_matrix(true_labels: list, avg_model_labels: list, file_name: str, output_dir: str, key: str):
    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(true_labels, avg_model_labels, labels=[0, 1])

    # Calculate percentages
    total = np.sum(conf_matrix)
    percentage_matrix = (conf_matrix / total) * 100

    # Create an annotated matrix that includes percentages
    annot_matrix = np.array([[f"{val}\n({percentage:.2f}%)" for val, percentage in zip(row, percentage_row)]
                             for row, percentage_row in zip(conf_matrix, percentage_matrix)])

    # Visualize the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=annot_matrix, fmt='',
                xticklabels=[0, 1],
                yticklabels=[0, 1],
                cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {key}', fontsize=14, fontweight='bold')

    conf_matrix_path = os.path.join(output_dir, file_name)
    plt.savefig(conf_matrix_path)

    plt.show()


def calculate_metrics(true_labels, pred_labels, label_name):
    # Calculate metrics
    accuracy = round(accuracy_score(true_labels, pred_labels) * 100, 2)  # TP + TN / n
    precision = round(precision_score(true_labels, pred_labels, pos_label=1, average='binary') * 100,
                      2)  # TP / TP + FP
    recall = round(recall_score(true_labels, pred_labels, pos_label=1, average='binary') * 100, 2)  # TP / TP + FN
    f1 = round(f1_score(true_labels, pred_labels, pos_label=1, average='binary') * 100,
               2)  # 2 * [(recall * precision) / (recall + precision)]

    # Calculate percentages
    correct_no = round(
        sum([1 for true, pred in zip(true_labels, pred_labels) if true == pred and true == 0]) / true_labels.count(
            0) * 100, 2)
    incorrect_no = round(
        sum([1 for true, pred in zip(true_labels, pred_labels) if true != pred and true == 0]) / true_labels.count(
            0) * 100, 2)
    correct_spur = round(
        sum([1 for true, pred in zip(true_labels, pred_labels) if true == pred and true == 1]) / true_labels.count(
            1) * 100, 2)
    incorrect_spur = round(
        sum([1 for true, pred in zip(true_labels, pred_labels) if true != pred and true == 1]) / true_labels.count(
            1) * 100, 2)

    return [label_name, accuracy, precision, recall, f1, correct_no, incorrect_no, correct_spur, incorrect_spur]


def plot_labels(df, plot_title, output_dir):
    df = df.sort_values(by='frame')

    true_labels = df['label'].tolist()
    model_labels = df['pred_avg']

    fig, axs = plt.subplots(2, 1, figsize=(15, 4))
    tick_interval = 250
    frame_numbers = df['frame'].tolist()
    xticks = np.arange(min(frame_numbers), max(frame_numbers) + tick_interval, tick_interval)
    xticklabels = np.arange(min(frame_numbers), max(frame_numbers) + tick_interval, tick_interval).astype(int)

    for i, label in enumerate(true_labels):
        color = 'white' if label == 0 else 'dimgray'
        axs[0].axhspan(0, 1, xmin=i / len(frame_numbers), xmax=(i + 1) / len(frame_numbers), facecolor=color)

    axs[0].set_yticks([])
    axs[0].set_title('True Labels', fontsize=14, fontweight='bold')
    axs[0].set_xticks(xticks / max(frame_numbers))
    axs[0].set_xticklabels(xticklabels)

    for i, label in enumerate(model_labels):
        color = 'white' if label == 0 else 'darkgray'
        axs[1].axhspan(0, 1, xmin=i / len(frame_numbers), xmax=(i + 1) / len(frame_numbers), facecolor=color)

    axs[1].set_yticks([])
    axs[1].set_title(plot_title, fontsize=14, fontweight='bold')
    axs[1].set_xticks(xticks / max(frame_numbers))
    axs[1].set_xticklabels(xticklabels)

    plt.xlabel('Frame Number')
    plt.tight_layout()

    label_rectangle_path = os.path.join(output_dir, 'label_vis.png')
    plt.savefig(label_rectangle_path)

    plt.show()


def evaluate_models(df):
    # Initialize DataFrame
    results_df = pd.DataFrame(
        columns=['k', 'accuracy', 'precision', 'recall', 'f1_score', 'correct_no', 'incorrect_no', 'correct_spur',
                 'incorrect_spur'])

    # Extract true labels from df and filter None values
    true_labels_k = [label for label in df['label'].tolist() if label is not None]

    for col in df.columns:
        if col.startswith('pred_'):
            # Filter None values
            model_labels_k = [label for label in df[col].tolist() if label is not None]

            if len(true_labels_k) != len(model_labels_k):
                print(f"Skipping evaluation for {col} due to inconsistent label lengths after filtering None values.")
                continue

            model_name = col.split('_')[1]
            results_df.loc[len(results_df)] = calculate_metrics(true_labels_k, model_labels_k, model_name)

    return results_df


def get_models(weights_dir: str, verbose=False):
    weights = []
    nb_models = 5  # k-fold with k=5

    for i in range(nb_models):
        fold_dir = os.path.join(weights_dir, f"fold_{i}")

        best_pt_path = os.path.join(fold_dir, 'best.pt')

        if os.path.exists(best_pt_path):
            weights.append(best_pt_path)
        else:
            raise Exception(f'Missing weights for fold at: {fold_dir}')

    if verbose:
        for weight in weights:
            print(weight)

    return weights


def pre_process_df(frame_df: pd.DataFrame, k: int):
    # Create columns for each prediction
    for i in range(k):
        proba_col = f'proba_{i}'
        frame_df[proba_col] = None

    # Create a column for the average prediction
    frame_df['proba_avg'] = None


def get_df(csv_loc: str) -> pd.DataFrame:
    df = pd.read_csv(csv_loc, index_col=False)

    # Initialize frame_objects list
    frame_objects = []

    for index, row in df.iterrows():
        # Get frame ranges & labels
        frame_start = row['frame_start']
        frame_end = row['frame_end']
        label = row['label']

        # Create one object for each frame
        for frame_number in range(frame_start, frame_end + 1):
            frame_object = {'frame_number': frame_number, 'label': label}
            frame_objects.append(frame_object)

    frame_df = pd.DataFrame(frame_objects)

    return frame_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse data folder paths.')

    parser.add_argument('--weights_dir', help='Directory that contains model weights.',
                        default='/home/rob/Documents/Github/navi_lstm/results/snapshot/grid_0', type=str)
    parser.add_argument('--k', help='Number of folds.', default=5, type=int)
    parser.add_argument('--embeddings_dir', default='/home/rob/Documents/Github/navi_lstm/data/embeddings_test')
    parser.add_argument('--video_csv', default='/home/rob/Documents/Github/navi_lstm/data/maps/mapping_test.csv')
    parser.add_argument('--label_map_path',
                        default='/home/rob/Documents/Github/navi_lstm/data/labels/ground_truth_testset.json')
    parser.add_argument('--window_size', default=150, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--hidden_size', default=64, type=int)
    parser.add_argument('--n_layers', default=1, type=int)

    args = vars(parser.parse_args())

    # get_models(args['weights_dir'], verbose=True)

    print('Running Main Program...')
    main_inference(**args)
