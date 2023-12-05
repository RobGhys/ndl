import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, RocCurveDisplay
import os
import json

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import seaborn as sns
import argparse
from pathlib import Path


def main_inference(predictions_dir: str):
    output_dir = Path('/home/rob/Documents/Github/navi_lstm/output/inference/nnunet')
    os.makedirs(output_dir, exist_ok=True)

    folder_path = Path(predictions_dir)

    # Vérifier si le dossier existe
    if not folder_path.is_dir():
        print(f"Le dossier {folder_path} n'existe pas.")
        return

    # Parcourir les fichiers CSV dans le dossier
    for file in folder_path.glob('*.csv'):
        # Créer un DataFrame à partir du fichier CSV
        df = pd.read_csv(file)
        df['label'] = df['label'].replace({'no': 0, 'spur': 1})
        df['pred'] = df['pred'].replace({'no': 0, 'spur': 1})

        # Appeler les fonctions nécessaires
        key = file.stem  # Utilise le nom du fichier sans l'extension comme clé
        output_sub_dir = output_dir / key  # Crée un sous-dossier pour les sorties
        output_sub_dir.mkdir(exist_ok=True)  # Crée le sous-dossier si nécessaire

        get_confusion_matrix(true_labels=df['label'].tolist(), avg_model_labels=df['pred'].tolist(),
                             file_name=f'confusion_matrix.png', output_dir=output_sub_dir, key=key)

        plot_labels(df, f'Average Predictions {key}', output_dir=output_sub_dir)
        results_frame_df = evaluate_models(df)
        output_file_path = os.path.join(output_sub_dir, 'eval_output.csv')

        compute_auc_roc(output_sub_dir, df, 'roc_curve.pdf', results_frame_df)
        results_frame_df.to_csv(output_file_path, index=False)


def post_process_w_exp_decay(df: pd.DataFrame, a: float, b: np.array, t: int) -> None:
    # Create new columns in df
    df['post_proba'] = np.nan
    df['post_pred'] = np.nan

    for i, row in df.iterrows():
        # Get frame at time t
        t_frame = row['proba']
        # Get frames [i-t, i] if there are at least 't' frames, else get '0' frames
        t_minus_frames = df['proba'].iloc[max(0, i - t):i].to_numpy()
        adjusted_b = b[:len(t_minus_frames)]

        # Arima
        post_proba = t_frame * a + np.dot(t_minus_frames, adjusted_b) if len(t_minus_frames) == t else t_frame
        df.at[i, 'post_proba'] = post_proba

        # Pred
        df.at[i, 'pred'] = 1 if df.at[i, 'post_proba'] > 0.5 else 0


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


def compute_roc_auc(df, csv_file, proba='proba'):
    y = df['label']
    ŷ = df[proba]
    result = roc_auc_score(y, ŷ)
    csv_file['roc_auc'] = result

    return result


def compute_auc_roc(output_file_path, results_frame_df, roc_curve_file_name, csv_file):
    auc_roc = compute_roc_auc(results_frame_df, csv_file)
    fpr, tpr, thresholds = compute_roc_curve(results_frame_df)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc_roc).plot()
    roc_display.figure_.savefig(os.path.join(output_file_path, roc_curve_file_name))


def compute_roc_curve(df, proba='proba'):
    #y = df['label'].map({'no': 0, 'spur': 1})
    y = df['label']
    ŷ = df[proba].astype(float)

    result = roc_curve(y, ŷ, pos_label=1)

    return result


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


def calculate_metrics(true_labels, pred_labels):
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

    return [accuracy, precision, recall, f1, correct_no, incorrect_no, correct_spur, incorrect_spur]


def plot_labels(df, plot_title, output_dir, filename='label_vis.png'):
    df = df.sort_values(by='frame')

    true_labels = df['label'].tolist()
    model_labels = df['pred']

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
        columns=['accuracy', 'precision', 'recall', 'f1_score', 'correct_no', 'incorrect_no', 'correct_spur',
                 'incorrect_spur'])

    # Extract true labels from df and filter None values
    true_labels_k = [label for label in df['label'].tolist() if label is not None]

    for col in df.columns:
        if col.startswith('pred'):
            # Filter None values
            model_labels_k = [label for label in df[col].tolist() if label is not None]

            if len(true_labels_k) != len(model_labels_k):
                print(f"Skipping evaluation for {col} due to inconsistent label lengths after filtering None values.")
                continue

            results_df.loc[len(results_df)] = calculate_metrics(true_labels_k, model_labels_k)

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse data folder paths.')

    parser.add_argument('--predictions_dir', default='/home/rob/Documents/Github/navi_lstm/output/nnunet/nnunet_grid_0')

    args = vars(parser.parse_args())

    print('Running Main Program...')
    main_inference(**args)