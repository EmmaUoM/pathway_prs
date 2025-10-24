import numpy as np
import pandas as pd
import copy
import os
import yaml
import random
from joblib import dump
import torch
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def load_config(config_path="/data/gpfs/projects/punim1484/emma/pathway_prs_ml/code/configs/config.yaml", experiment_name=None):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    globals_config = config.get('globals', {})
    paths_config = config.get('paths', {})

    if experiment_name:
        experiment_config = config.get(experiment_name, {})
        config = {**globals_config, **experiment_config, **paths_config}
    else:
        config = {**globals_config, **paths_config}

    return config

# Set random seeds for reproducibility
def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_errors(y_test, y_pred, false_positives, false_negatives, true_positives, true_negatives, error_save_dir, fold_idx):
    # 计算误差并排序
    errors = pd.DataFrame({
        'ID': y_test.index,
        'y_pred': y_pred,
        'y_test': y_test
    })
    errors['abs_error'] = (errors['y_pred'] - errors['y_test']).abs()

    false_positives_df = errors.loc[false_positives]
    false_negatives_df = errors.loc[false_negatives]
    true_positives_df = errors.loc[true_positives]
    true_negatives_df = errors.loc[true_negatives]

    false_positives_sorted = false_positives_df.sort_values(by='abs_error', ascending=False)
    false_negatives_sorted = false_negatives_df.sort_values(by='abs_error', ascending=False)
    true_positives_sorted = true_positives_df.sort_values(by='abs_error', ascending=False)
    true_negatives_sorted = true_negatives_df.sort_values(by='abs_error', ascending=False)

    # 确保所有 DataFrame 的索引都是唯一的
    false_positives_sorted = false_positives_sorted.reset_index(drop=True)
    false_negatives_sorted = false_negatives_sorted.reset_index(drop=True)
    true_positives_sorted = true_positives_sorted.reset_index(drop=True)
    true_negatives_sorted = true_negatives_sorted.reset_index(drop=True)

    # 创建 DataFrame 保存所有错误类型
    max_len = max(len(false_positives_sorted), len(false_negatives_sorted), len(true_positives_sorted), len(true_negatives_sorted))

    def pad_df(df, max_len):
        current_len = len(df)
        if current_len < max_len:
            pad_length = max_len - current_len
            padding = pd.DataFrame([{"ID": None, "y_pred": None, "y_test": None, "abs_error": None}] * pad_length)
            df = pd.concat([df, padding], ignore_index=True)
        return df

    false_positives_sorted = pad_df(false_positives_sorted, max_len)
    false_negatives_sorted = pad_df(false_negatives_sorted, max_len)
    true_positives_sorted = pad_df(true_positives_sorted, max_len)
    true_negatives_sorted = pad_df(true_negatives_sorted, max_len)

    combined_df = pd.DataFrame({
        'False Positives ID': false_positives_sorted['ID'],
        'False Positives y_pred': false_positives_sorted['y_pred'],
        'False Positives y_test': false_positives_sorted['y_test'],
        'False Negatives ID': false_negatives_sorted['ID'],
        'False Negatives y_pred': false_negatives_sorted['y_pred'],
        'False Negatives y_test': false_negatives_sorted['y_test'],
        'True Positives ID': true_positives_sorted['ID'],
        'True Positives y_pred': true_positives_sorted['y_pred'],
        'True Positives y_test': true_positives_sorted['y_test'],
        'True Negatives ID': true_negatives_sorted['ID'],
        'True Negatives y_pred': true_negatives_sorted['y_pred'],
        'True Negatives y_test': true_negatives_sorted['y_test']
    })

    combined_df.to_csv(f'{error_save_dir}/{fold_idx}.csv', index=False)

def load_profiles_from_directory(directory_path):
    dataframes = []
    feature_names = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.profile'):
            file_path = os.path.join(directory_path, filename)

            # 提取 feature 名字
            feature_name = os.path.splitext(filename)[0]
            feature_names.append(feature_name)

            df = pd.read_csv(file_path, sep=r'\s+')

            if 'SCORESUM' in df.columns:
                # 只选择 IID 和 SCORESUM 列，并重命名 SCORESUM 列
                df = df[['IID', 'SCORESUM']].rename(columns={'SCORESUM': feature_name})
                # 处理 IID 列，去掉 _ 之前的部分
                if 'adni' in directory_path:
                    df['IID'] = df['IID'].apply(lambda x: '_'.join(x.split('_')[1:]))
            else:
                # 选择其他列，例如 e2, e4
                df = df[['IID', feature_name]]

            # 将读取的 DataFrame 添加到列表中
            dataframes.append(df)

    # 合并所有 DataFrame，以 IID 为键
    merged_df = dataframes[0]
    for df in dataframes[1:]:
        merged_df = pd.merge(merged_df, df, on='IID', how='outer')

    return merged_df, feature_names

def load_and_merge_profiles(directories):
    merged_df = pd.DataFrame()
    all_feature_names = []

    for directory in directories:
        # 加载单个目录下的特征数据
        df, feature_names = load_profiles_from_directory(directory)
        all_feature_names.extend(feature_names)

        if merged_df.empty:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on='IID', how='outer')

    return merged_df, all_feature_names

def merge_with_csv(merged_df, csv_path):
    csv_df = pd.read_csv(csv_path)
    csv_df.rename(columns={'PTID': 'IID'}, inplace=True)
    csv_df.set_index('IID', inplace=True)
    final_df = pd.merge(csv_df, merged_df, left_index=True, right_on='IID', how='left')
    final_df.set_index('IID', inplace=True)
    # final_df.dropna(inplace=True)

    return final_df

def load_prs(experiment, config, model_cov=None):
    if experiment == "APOE" or experiment == "APOE_aibl":
        directories = [config['data_path']]
    elif 'aibl' in experiment:
        directories = [config['data_path'], config['apoe_prs_aibl_path']]
    else:
        directories = [config['data_path'], config['apoe_prs_adni_path']]

    merged_df, all_feature_names = load_and_merge_profiles(directories)

    cov_label_path = '/data/gpfs/projects/punim1484/emma/prs/sex_age_edu_Ab_adni.csv'
    cov_label_aibl_path = '/data/gpfs/projects/punim1484/emma/prs/sex_age_edu_Ab_aibl.csv'
    label_column = 'Abeta'

    # 合并包含 PTID 信息的 CSV 数据
    if 'aibl' in experiment:
        final_df = merge_with_csv(merged_df, cov_label_aibl_path)
    else:
        final_df = merge_with_csv(merged_df, cov_label_path)

    # 保留 CSV 文件中存在的 PTID
    final_df = final_df[final_df.index.notnull()]
    final_df = final_df[final_df['Abeta'].notnull()]

    # 提取特征和标签
    features = final_df.drop(columns=[label_column])
    labels = final_df[label_column]

    selected_feature_names = all_feature_names.copy()
    if model_cov == "sex":
        # 只选择 sex 相关特征
        features = features[['Sex'] + all_feature_names]
        selected_feature_names = ['Sex'] + all_feature_names
    elif model_cov == "sex_age":
        # 选择 sex 和 age 相关特征
        features = features[['Sex', 'Age'] + all_feature_names]
        selected_feature_names = ['Sex', 'Age'] + all_feature_names
    elif model_cov == "sex_age_edu":
        # 选择 sex, age 和 edu 相关特征
        features = features[['Sex', 'Age', 'Education'] + all_feature_names]
        selected_feature_names = ['Sex', 'Age', 'Education'] + all_feature_names
    else:
        # 不包含 sex, age 和 education 特征
        features = features[all_feature_names]

    return features, labels, selected_feature_names

def save_model(model, save_dir, experiment_name, model_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"{experiment_name}_{model_name}_best.joblib")
    dump(model, save_path)
    print(f"Model saved to {save_path}")

def save_torch_model(model, save_dir, experiment_name, fold, model_selection):
    save_dir = os.path.join(save_dir, f"{experiment_name}", model_selection)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"fold{fold}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Torch model saved to {save_path}")

def find_best_threshold(y_true, y_scores, metric='f1'):
    thresholds = np.linspace(0, 1, 101)
    best_threshold = 0.5
    best_score = 0

    for threshold in thresholds:
        if isinstance(y_scores, torch.Tensor):
            y_scores = y_scores.cpu().numpy()
        y_pred = (y_scores >= threshold).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred)
        elif metric == 'accuracy':
            score = accuracy_score(y_true, y_pred)
        else:
            raise ValueError("Unsupported metric. Choose from 'f1', 'precision', 'recall', 'accuracy'.")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score
