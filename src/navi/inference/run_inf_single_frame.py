import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, RocCurveDisplay

import os
import torch
import json

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

import seaborn as sns
from tqdm import tqdm
import argparse

from navi.datasets.single_video_with_context import SingleVideoWithContext
from navi.datasets.video_with_context import VideoWithContext
from navi.nn.embeddings_fc import ResNet50FC
from navi.nn.models_using_embeddings import ResNet50GRU


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
                   batch_size: int, hidden_size: int, n_layers: int, model_name: str):
    output_dir = '/home/rob/Documents/Github/navi_lstm/output/inference'
    output_sub_dir: dict = {}
    dataframe_dict = create_dfs(video_csv, label_map_path)
    test_videos = pd.read_csv(video_csv)

    print('Pre Processing Dataframes')
    for key, dataframe in dataframe_dict.items():
        try:
            output_sub_dir[key] = get_output_sub_dir(output_dir, key)
        except Exception as e:
            raise Exception(f'Error during Pre Processing of Dataframes for key: {key} | \nException: {e}')

    video_ids = sorted(dataframe_dict.keys())

    print('Collecting Models')
    model_paths = get_models(weights_dir, nb_models=k, verbose=False)

    with open(label_map_path, 'rb') as file:
        label_map = json.load(file)

    df_dict_one_model = infer(model_paths=model_paths,
                              embeddings_dir=embeddings_dir, test_videos=test_videos, label_map=label_map,
                              window_size=window_size, batch_size=batch_size, hidden_size=hidden_size,
                              n_layers=n_layers, video_ids=video_ids, model_name=model_name)

    verify_matching_keys(dataframe_dict, df_dict_one_model)
    
    # For each df (1 per video) in df_dict_one_model
    for key, dataframe in df_dict_one_model.items():
        # Concat predictions with labels and replace dataframe
        df_dict_one_model[key] = concat_dataframe(dataframe, dataframe_dict[key])
        del dataframe
        dataframe = df_dict_one_model[key]

        average_prediction(dataframe, range(k))

        df_file_path = os.path.join(output_sub_dir[key], 'probas_labels.csv')
        dataframe.to_csv(df_file_path, index=False)

        get_confusion_matrix(true_labels=dataframe['label'].tolist(), avg_model_labels=dataframe['pred_avg'].tolist(),
                             file_name=f'confusion_matrix.png', output_dir=output_sub_dir[key], key=key)

        plot_labels(dataframe, f'Average Predictions {key}', output_dir=output_sub_dir[key])

        results_frame_df = evaluate_models(dataframe)
        output_file_path = os.path.join(output_sub_dir[key], 'eval_output.csv')
        results_frame_df.to_csv(output_file_path, index=False)

        compute_auc_roc(output_sub_dir[key], dataframe, 'roc_curve.pdf')

        # Post-processing
        post_processed_df = run_post_processing(dataframe)
        post_output_file_path = os.path.join(output_sub_dir[key], 'post_probas_labels.csv')
        post_processed_df.to_csv(post_output_file_path, index=False)

        # get_confusion_matrix(true_labels=post_processed_df['label'].tolist(),
        #                      avg_model_labels=post_processed_df['post_pred'].tolist(),
        #                      file_name=f'post_confusion_matrix.png', output_dir=output_sub_dir[key], key=key)
        #
        # plot_labels(post_processed_df, f'Post_Average Predictions {key}', output_dir=output_sub_dir[key],
        #             filename='post_label_vis.png')
        # post_results_frame_df = evaluate_models(post_processed_df)
        # output_file_path = os.path.join(output_sub_dir[key], 'post_eval_output.csv')
        # post_results_frame_df.to_csv(output_file_path, index=False)
        #
        # compute_auc_roc(output_sub_dir[key], post_processed_df, 'post_roc_curve.pdf')


def compute_auc_roc(output_file_path, results_frame_df, roc_curve_file_name):
    auc_roc = compute_roc_auc(results_frame_df)
    fpr, tpr, thresholds = compute_roc_curve(results_frame_df)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc_roc).plot()
    roc_display.figure_.savefig(os.path.join(output_file_path, roc_curve_file_name))


def run_post_processing(df):
    # Only keep 2 columns from df
    frame_df = df[['frame', 'proba_avg', 'label']].copy()
    frame_df = frame_df.sort_values(by='frame')

    # Setup post-processing params
    nb_periods = 5
    target_rate = 0.5
    b = compute_rate(target_rate, nb_periods)

    # Add post-processed probas
    post_process_w_exp_decay(frame_df, a=target_rate, b=b, t=nb_periods)

    return frame_df


def get_output_sub_dir(output_dir: str, key: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_sub_dir = os.path.join(output_dir, key)
    if not os.path.exists(output_sub_dir):
        os.makedirs(output_sub_dir)

    return output_sub_dir


def infer(model_paths: list, embeddings_dir: str, test_videos: pd.DataFrame, label_map: pd.DataFrame,
          window_size: int, batch_size: int, hidden_size: int, n_layers: int, video_ids: list, model_name: str) -> dict:
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print(f'Device: {device}')

    # Get Dataset
    inference_dataset = SingleVideoWithContext(
        embeddings_dir,
        test_videos,
        label_map,
        window_size=window_size
    )
    # Get DataLoader
    inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)
    # Run inference
    df_dict_one_model: dict = {}
    for model_nb in range(len(model_paths)):
        print(f'model nb: {model_nb}')
        model = get_model(device, model_paths[model_nb], hidden_size, n_layers, model_name)
        df_dict_one_model = run_batches(device, model, inference_loader, model_nb=model_nb,
                                        video_ids=video_ids, dict_df=df_dict_one_model)

    return df_dict_one_model


def run_batches(device, model, inference_loader, model_nb, video_ids: list, dict_df: dict) -> dict:
    for batch, (sequence, label) in tqdm(enumerate(inference_loader), desc='inference'):
        print(f'Batch: {batch}')
        with torch.inference_mode():
            # (1, T, 150, 2048)
            sequence = sequence.squeeze(0).to(device)
            # (T, 150, 2048)
            logits = model(sequence).squeeze(-1)
            # (T, 150)
            logits = logits[:, -1]
            probabilities = logits.sigmoid().cpu().numpy()

            n_frames = logits.shape[0]
            batch_df = pd.DataFrame({
                'frame': range(n_frames),
                f'proba_{model_nb}': probabilities
            })

            key = video_ids[batch]
            if model_nb == 0:
                dict_df[key] = batch_df
            else:
                dict_df[key] = dict_df[key].merge(batch_df, on='frame', how='left')

    return dict_df


def get_model(device, model_path, hidden_size, n_layers, model_name):
    if model_name == 'gru':
        model = ResNet50GRU(
            input_size=2048,
            hidden_size=hidden_size,
            n_layers=n_layers,
        )
    elif model_name == 'gru_nnunet':
        model = ResNet50GRU(
            input_size=2016,
            hidden_size=hidden_size,
            n_layers=n_layers,
        )
    elif model_name == 'fc':
        model = ResNet50FC(
            input_size=2048,
            hidden_size=hidden_size,
        )
    elif model_name == 'fc_nnunet_embeddings':
        model = ResNet50FC(
            input_size=2016,
            hidden_size=hidden_size,
        )
    checkpoint = torch.load(model_path)

    model.load_state_dict(checkpoint)
    model.eval()
    model = model.to(device)

    return model


def post_process_w_exp_decay(df: pd.DataFrame, a: float, b: np.array, t: int) -> None:
    # Create new columns in df
    df['post_proba'] = np.nan
    df['post_pred'] = np.nan

    for i, row in df.iterrows():
        # Get frame at time t
        t_frame = row['proba_avg']
        # Get frames [i-t, i] if there are at least 't' frames, else get '0' frames
        t_minus_frames = df['proba_avg'].iloc[max(0, i - t):i].to_numpy()
        adjusted_b = b[:len(t_minus_frames)]

        # Arima
        post_proba = t_frame * a + np.dot(t_minus_frames, adjusted_b) if len(t_minus_frames) == t else t_frame
        df.at[i, 'post_proba'] = post_proba

        # Pred
        df.at[i, 'pred_avg'] = 1 if df.at[i, 'post_proba'] > 0.5 else 0


def compute_rate(target_rate: float, t: int, verbose: bool = False) -> np.array:
    result = np.zeros(t)
    current_rate = target_rate
    for i in range(t - 1):
        current_rate = current_rate / 2
        if verbose: print(f'Current rate: {current_rate}')
        result[i] = current_rate
    # last rate is the difference between target, and what we currently have
    result[-1] = target_rate - np.sum(result)
    if verbose: print(f'Last rate: {result[-1]}')

    # Reverse the array so that higher rates come near the end
    result = np.flip(result)
    if verbose:
        print(f'Result: \n{result}')
        print(f'Total = {np.sum(result)}')
    return result


def compute_roc_auc(df, proba='proba_avg'):
    y = df['label']
    ŷ = df[proba]
    result = roc_auc_score(y, ŷ)

    return result

def compute_roc_curve(df, proba='proba_avg'):
    #y = df['label'].map({'no': 0, 'spur': 1})
    y = df['label']
    ŷ = df[proba].astype(float)

    result = roc_curve(y, ŷ, pos_label=1)

    return result


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


def plot_labels(df, plot_title, output_dir, filename='label_vis.png'):
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

    label_rectangle_path = os.path.join(output_dir, filename)
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


def get_models(weights_dir: str, nb_models, verbose=False):
    weights = []

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse data folder paths.')

    parser.add_argument('--weights_dir', help='Directory that contains model weights.',
                        default='/home/rob/Documents/Github/navi_lstm/output/snapshot/v15_resnet_rnn/grid_0', type=str)
    parser.add_argument('--k', help='Number of folds.', default=5, type=int)
    parser.add_argument('--embeddings_dir', default='/home/rob/Documents/Github/navi_lstm/data/embeddings_nnunet_test')
    parser.add_argument('--video_csv', default='/home/rob/Documents/Github/navi_lstm/data/maps/mapping_test.csv')
    parser.add_argument('--label_map_path',
                        default='/home/rob/Documents/Github/navi_lstm/data/labels/ground_truth_testset.json')
    parser.add_argument('--window_size', default=150, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--hidden_size', default=64, type=int)
    parser.add_argument('--n_layers', default=1, type=int)
    parser.add_argument('--model_name', default='gru_nnunet', type=str)

    args = vars(parser.parse_args())

    # model_paths = get_models(args['weights_dir'], verbose=True, nb_models=5)
    # for path in model_paths:
    #     checkpoint = torch.load(path)
    #     print(checkpoint.keys())

    print('Running Main Program...')
    main_inference(**args)
