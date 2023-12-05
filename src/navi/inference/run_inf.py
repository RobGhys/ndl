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

from navi.datasets.seq2seq_embeddings import SequenceToSequenceEmbeddings
from navi.nn.models_using_embeddings import ResetNet50GRU
from navi.transforms import ToTensor


def main_inference(weights_dir: str, k: int, embeddings_dir: str, video_csv: str,
                   label_map_path: str, window_size: int, window_stride: int,
                   skip_frames: int, drop_empty_windows: bool,
                   batch_size: int, hidden_size: int, n_layers: int):
    output_dir = '/home/rob/Documents/Github/navi_lstm/output/inference'
    data_dict = get_data()

    for key, (csv_loc, image_folder_path) in data_dict.items():
        print(f'Video with key: {key}')
        output_sub_dir = get_output_sub_dir(output_dir, key)
        # Prepare raw dataframe
        frame_df = get_df(csv_loc=csv_loc)
        # Get dataframe with true labels
        pre_process_df(frame_df=frame_df, k=k)
        # Get dir with all weights
        model_paths = get_models(weights_dir)

        with open(label_map_path, 'rb') as file:
            label_map = json.load(file)
        test_videos = pd.read_csv(video_csv)

        transforms = ToTensor()
        target_transforms = ToTensor()

        # Run inference
        infer(frame_df=frame_df, model_paths=model_paths,
              embeddings_dir=embeddings_dir, test_videos=test_videos, label_map=label_map,
              window_size=window_size, window_stride=window_stride, skip_frames=skip_frames,
              drop_empty_windows=drop_empty_windows,
              transforms=transforms, target_transforms=target_transforms,
              batch_size=batch_size, hidden_size=hidden_size, n_layers=n_layers
              )
        # Get the average
        average_prediction(frame_df, range(k))

        output_file_path = os.path.join(output_sub_dir, 'data_frame_output.csv')
        frame_df.to_csv(output_file_path, index=False)
        get_confusion_matrix(true_labels=frame_df['label'].tolist(), avg_model_labels=frame_df['pred_avg'].tolist(),
                             file_name='confusion_matrix_model.png', output_dir=output_sub_dir)

        plot_labels(frame_df, 'Average Model Predictions', output_dir=output_sub_dir)

        results_frame_df = evaluate_models(frame_df)
        output_file_path = os.path.join(output_sub_dir, 'eval_output_post.csv')
        results_frame_df.to_csv(output_file_path, index=False)


def get_output_sub_dir(output_dir: str, key: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_sub_dir = os.path.join(output_dir, key)
    if not os.path.exists(output_sub_dir):
        os.makedirs(output_sub_dir)
    return output_sub_dir


def infer(frame_df: pd.DataFrame, model_paths: list,
          embeddings_dir: str, test_videos: pd.DataFrame, label_map: pd.DataFrame,
          window_size: int, window_stride: int, skip_frames, drop_empty_windows,
          transforms, target_transforms, batch_size: int, hidden_size: int, n_layers: int):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    for i, model_path in enumerate(model_paths):
        print(f'In model fold k={i}')
        model = get_model(device, model_path, hidden_size, n_layers)

        # Get Dataset
        inference_dataset = SequenceToSequenceEmbeddings(
            embeddings_dir,
            test_videos,
            label_map,
            window_size=window_size,
            window_stride=window_stride,
            skip_frames=skip_frames,
            drop_empty_windows=drop_empty_windows,
            transforms=transforms,
            target_transforms=target_transforms,
        )
        # Get DataLoader
        inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)
        # Run inference
        print(f'Start inference for model with fold k={i}')
        run_batches(frame_df, device, i, model, inference_loader, window_size, window_stride)

    print('...Exit infer...')


# def infer(frame_df: pd.DataFrame, model_paths: list,
#           embeddings_dir: str, test_videos: pd.DataFrame, label_map: pd.DataFrame,
#           window_size: int, window_stride: int, skip_frames, drop_empty_windows,
#           transforms, target_transforms, batch_size: int, hidden_size: int, n_layers: int):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f'Device: {device}')
#
#     for i, model_path in enumerate(model_paths):
#         print(f'In model fold k={i}')
#         model = get_model(device, model_path, hidden_size, n_layers)
#
#         # Get Dataset
#         inference_dataset = SequenceToSequenceEmbeddings(
#             embeddings_dir,
#             test_videos,
#             label_map,
#             window_size=window_size,
#             window_stride=window_stride,
#             skip_frames=skip_frames,
#             drop_empty_windows=drop_empty_windows,
#             transforms=transforms,
#             target_transforms=target_transforms,
#         )
#         # Get DataLoader
#         inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)
#         # Run inference
#         print(f'Start inference for model with fold k={i}')
#         run_batches(frame_df, device, i, model, inference_loader)
#
#     print('...Exit infer...')


def run_batches(frame_df, device, i, model, inference_loader, window_size, window_stride):
    progress_bar = tqdm(enumerate(inference_loader), total=len(inference_loader))
    progress_bar.set_description(f"Running Inference...")

    with torch.inference_mode():
        for b_index, (sequence, label) in progress_bar:
            print(f'Batch index: {b_index}')
            #print('shape', sequence.shape)
            # (1, 150, 2048)
            sequence = sequence.squeeze(0).to(device)
            # (150, 2048)
            logits = model(sequence).squeeze(-1)
            # (150,)
            #logits = logits[-1]
            probabilities = logits.sigmoid().cpu().numpy()
            predicted_labels = (probabilities > 0.5).astype(int)

            n_frames = logits.shape[0]
            assert n_frames == window_size

            start_frame = b_index * window_size
            end_frame = start_frame + window_size
            print(f'start: {start_frame}, end: {end_frame}')

            for frame_number in range(start_frame, end_frame):
                relative_frame_number = frame_number - start_frame
                frame_idx = frame_df[frame_df['frame_number'] == frame_number].index[0]
                frame_df.at[frame_idx, f'pred_{i}'] = 'no' if predicted_labels[relative_frame_number] == 0 else 'spur'
                #frame_df.at[frame_idx, f'pred_{i}'] = 'no' if predicted_labels[frame_number] == 0 else 'spur'
                #frame_df.at[frame_idx, f'prob_{i}'] = probabilities[frame_number]
                frame_df.at[frame_idx, f'prob_{i}'] = probabilities[relative_frame_number]

            print(frame_df)
    print('...Exit run_batches...')


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
    # Create columns for average prediction and probability if they don't exist
    if 'pred_avg' not in frame_df.columns:
        frame_df['pred_avg'] = None
    if 'prob_avg' not in frame_df.columns:
        frame_df['prob_avg'] = None

    # Calculate the average prediction and probability for each row
    for index, row in frame_df.iterrows():
        prob_cols = [f'prob_{k}' for k in k_range]

        prob_values = row[prob_cols]
        avg_prob = np.mean(prob_values)
        avg_pred = 'spur' if avg_prob > 0.5 else 'no'

        frame_df.at[index, 'prob_avg'] = avg_prob
        frame_df.at[index, 'pred_avg'] = avg_pred

        # print(frame_df.head(10))


def get_confusion_matrix(true_labels, avg_model_labels, file_name, output_dir):
    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(true_labels, avg_model_labels, labels=['no', 'spur'])

    # Calculate percentages
    total = np.sum(conf_matrix)
    percentage_matrix = (conf_matrix / total) * 100

    # Create an annotated matrix that includes percentages
    annot_matrix = np.array([[f"{val}\n({percentage:.2f}%)" for val, percentage in zip(row, percentage_row)]
                             for row, percentage_row in zip(conf_matrix, percentage_matrix)])

    # Visualize the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=annot_matrix, fmt='',
                xticklabels=['no', 'spur'],
                yticklabels=['no', 'spur'],
                cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')

    conf_matrix_path = os.path.join(output_dir, file_name)
    plt.savefig(conf_matrix_path)

    plt.show()


def calculate_metrics(true_labels, pred_labels, label_name):
    # Calculate metrics
    accuracy = round(accuracy_score(true_labels, pred_labels) * 100, 2)  # TP + TN / n
    precision = round(precision_score(true_labels, pred_labels, pos_label='spur', average='binary') * 100,
                      2)  # TP / TP + FP
    recall = round(recall_score(true_labels, pred_labels, pos_label='spur', average='binary') * 100, 2)  # TP / TP + FN
    f1 = round(f1_score(true_labels, pred_labels, pos_label='spur', average='binary') * 100,
               2)  # 2 * [(recall * precision) / (recall + precision)]

    # Calculate percentages
    correct_no = round(
        sum([1 for true, pred in zip(true_labels, pred_labels) if true == pred and true == 'no']) / true_labels.count(
            'no') * 100, 2)
    incorrect_no = round(
        sum([1 for true, pred in zip(true_labels, pred_labels) if true != pred and true == 'no']) / true_labels.count(
            'no') * 100, 2)
    correct_spur = round(
        sum([1 for true, pred in zip(true_labels, pred_labels) if true == pred and true == 'spur']) / true_labels.count(
            'spur') * 100, 2)
    incorrect_spur = round(
        sum([1 for true, pred in zip(true_labels, pred_labels) if true != pred and true == 'spur']) / true_labels.count(
            'spur') * 100, 2)

    return [label_name, accuracy, precision, recall, f1, correct_no, incorrect_no, correct_spur, incorrect_spur]


def plot_labels(df, second_plot_title, output_dir):
    df = df.sort_values(by='frame_number')

    true_labels = df['label'].tolist()
    model_labels = df['pred_avg'] if second_plot_title == 'Average Model Predictions' else df['post_pred'].tolist()

    fig, axs = plt.subplots(2, 1, figsize=(15, 4))  # 2 lignes, 1 colonne
    tick_interval = 250
    frame_numbers = df['frame_number'].tolist()
    xticks = np.arange(min(frame_numbers), max(frame_numbers) + tick_interval, tick_interval)
    xticklabels = np.arange(min(frame_numbers), max(frame_numbers) + tick_interval, tick_interval).astype(int)

    for i, label in enumerate(true_labels):
        color = 'white' if label == 'no' else 'dimgray'
        axs[0].axhspan(0, 1, xmin=i / len(frame_numbers), xmax=(i + 1) / len(frame_numbers), facecolor=color)

    axs[0].set_yticks([])
    axs[0].set_title('True Labels', fontsize=14, fontweight='bold')
    axs[0].set_xticks(xticks / max(frame_numbers))
    axs[0].set_xticklabels(xticklabels)

    for i, label in enumerate(model_labels):
        color = 'white' if label == 'no' else 'darkgray'
        axs[1].axhspan(0, 1, xmin=i / len(frame_numbers), xmax=(i + 1) / len(frame_numbers), facecolor=color)

    axs[1].set_yticks([])
    axs[1].set_title(second_plot_title, fontsize=14, fontweight='bold')
    axs[1].set_xticks(xticks / max(frame_numbers))
    axs[1].set_xticklabels(xticklabels)

    plt.xlabel('Frame Number')
    plt.tight_layout()

    label_rectangle_path = os.path.join(output_dir,
                                        'label_vis.png') if second_plot_title == 'Average Model Predictions' else os.path.join(
        output_dir
        , 'label_vis_post.png')
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
    for i in range(0, k):
        pred_col = f'pred_{i}'
        frame_df[pred_col] = None
        proba_col = f'prob_{i}'
        frame_df[proba_col] = None

    # Create a column for the average prediction
    frame_df['pred_avg'] = None
    frame_df['prob_avg'] = None


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


def get_data():
    result: dict = {}

    name = 'id_x1'
    csv_loc = '/home/rob/Documents/Github/navi_lstm/data/labels/full_video_annotation - id_x1.csv'
    image_folder_path = os.path.expanduser('~/Documents/data/img_from_vid_endo/Londero 19042022 - 20220107-160703531')
    result[name] = (csv_loc, image_folder_path)

    name = 'id_x2'
    csv_loc = '/home/rob/Documents/Github/navi_lstm/data/labels/full_video_annotation - id_x2.csv'
    image_folder_path = os.path.expanduser('~/Documents/data/img_from_vid_endo/Loppe 15122022 - 20221215-113843662')
    result[name] = (csv_loc, image_folder_path)

    name = 'id_x3'
    csv_loc = '/home/rob/Documents/Github/navi_lstm/data/labels/full_video_annotation - id_x3.csv'
    image_folder_path = os.path.expanduser("~/Documents/data/img_from_vid_endo/Prata 13102022 - Rien d'anormal")
    result[name] = (csv_loc, image_folder_path)

    name = 'id_x4'
    csv_loc = '/home/rob/Documents/Github/navi_lstm/data/labels/full_video_annotation - id_x4.csv'
    image_folder_path = os.path.expanduser(
        "~/Documents/data/img_from_vid_endo/20230525-090005917_MIN.PROBE18-020-080004 - 20230525-093556392")
    result[name] = (csv_loc, image_folder_path)

    name = 'id_x5'
    csv_loc = '/home/rob/Documents/Github/navi_lstm/data/labels/full_video_annotation - id_x5.csv'
    image_folder_path = os.path.expanduser(
        '~/Documents/data/img_from_vid_endo/20230907-111125510_MIN.PROBE18-020-080004 - 20230907-113354200')
    result[name] = (csv_loc, image_folder_path)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse data folder paths.')

    parser.add_argument('--weights_dir', help='Directory that contains model weights.',
                        default='/home/rob/Documents/Github/navi_lstm/output/snapshot/grid_0', type=str)
    parser.add_argument('--k', help='Number of folds.', default=5, type=int)
    parser.add_argument('--embeddings_dir', default='/home/rob/Documents/Github/navi_lstm/data/embeddings_test')
    parser.add_argument('--video_csv', default='/home/rob/Documents/Github/navi_lstm/data/maps/mapping_test.csv')
    parser.add_argument('--label_map_path',
                        default='/home/rob/Documents/Github/navi_lstm/data/labels/ground_truth_testset.json')
    parser.add_argument('--window_size', default=150, type=int)
    parser.add_argument('--window_stride', default=120, type=int)
    parser.add_argument('--skip-frames', default=0, type=int)
    parser.add_argument('--drop_empty_windows', default=False, type=bool)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--hidden_size', default=64, type=int)
    parser.add_argument('--n_layers', default=1, type=int)

    args = vars(parser.parse_args())

    # get_models(args['weights_dir'], verbose=True)
    #get_output_sub_dir('output_inf', 'k_1')
    print('Running Main Program...')
    main_inference(**args)
