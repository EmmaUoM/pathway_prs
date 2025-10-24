import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, roc_auc_score, r2_score
from sklearn.model_selection import ParameterGrid
from torch.utils.tensorboard import SummaryWriter
from models.dnn import build_nn
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from utils import save_torch_model

# 定义一个激活钩子函数以捕获激活值
activation = {}

def get_activation(name):
    """创建钩子函数来捕获激活值"""
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def log_histograms(writer, model, epoch):
    """记录模型参数和梯度到TensorBoard"""
    for name, param in model.named_parameters():
        if param.requires_grad:
            writer.add_histogram(f'{name}/weights', param.data.cpu().numpy(), epoch)
            if param.grad is not None:
                writer.add_histogram(f'{name}/grads', param.grad.cpu().numpy(), epoch)

def log_activations(writer, epoch):
    """记录激活值"""
    for name, act in activation.items():
        writer.add_histogram(f'{name}/activation', act.cpu().numpy(), epoch)

def train_nn(X_train, y_train, X_test, y_test, cv_inner, config, device, fold_idx, experiment, feature_names,
             model_selection, cov, early_stopping_patience=10):
    '''
    在这个inner cv中模型的表现如何。
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param cv_inner:
    :param config:
    :param device:
    :param fold_idx:
    :param experiment:
    :param feature_names:
    :param model_selection:
    :param cov:
    :param early_stopping_patience:
    :return:
    '''

    writer = SummaryWriter(log_dir=f'runs/{experiment}_fold_{fold_idx}')

    # 使用SelectKBest进行特征选择
    k = 25  # 选择K个最佳特征
    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    param_grid = config['hyperparameters_NN']
    best_model = None
    best_score = None
    best_params = None

    def train_model(X_train_tensor, y_train_tensor, lr, weight_decay):
        model = build_nn(input_dim=X_train_selected.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        model.train()

        best_val_loss = np.inf
        epochs_no_improve = 0

        for epoch in range(100):
            model.train()
            for X_batch, y_batch in DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True):
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

            # 验证阶段
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for X_val_batch, y_val_batch in DataLoader(TensorDataset(X_val_inner_tensor, y_val_inner_tensor), batch_size=32, shuffle=False):
                    y_val_pred = model(X_val_batch)
                    val_loss += criterion(y_val_pred, y_val_batch).item()

            # Early Stopping 判断
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict()  # 保存最佳模型
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stopping_patience:
                # print(f"Early stopping triggered at epoch {epoch+1}")
                break

        model.load_state_dict(best_model_state)  # 恢复最佳模型权重
        return model

    # 内层交叉验证调参并选择最佳模型
    for params in ParameterGrid(param_grid):
        scores = []
        for train_inner_idx, val_inner_idx in cv_inner.split(X_train_selected, y_train):
            X_train_inner, X_val_inner = X_train_selected[train_inner_idx], X_train_selected[val_inner_idx]
            y_train_inner, y_val_inner = y_train.iloc[train_inner_idx], y_train.iloc[val_inner_idx]

            X_train_inner_tensor = torch.tensor(X_train_inner, dtype=torch.float32).to(device)
            y_train_inner_tensor = torch.tensor(y_train_inner.values, dtype=torch.float32).view(-1, 1).to(device)
            X_val_inner_tensor = torch.tensor(X_val_inner, dtype=torch.float32).to(device)
            y_val_inner_tensor = torch.tensor(y_val_inner.values, dtype=torch.float32).view(-1, 1).to(device)

            model = train_model(X_train_inner_tensor, y_train_inner_tensor, params['lr'], params['weight_decay'])
            model.eval()
            with torch.no_grad():
                y_val_pred = model(X_val_inner_tensor)
                val_score = mean_squared_error(y_val_inner_tensor.cpu().numpy(), y_val_pred.cpu().numpy())
                scores.append(val_score)

        avg_score = np.mean(scores)
        if best_score is None or avg_score < best_score:
            best_score = avg_score
            best_params = params  # 保存最佳参数

    # 打印最佳参数
    print(f"Best parameters for fold {fold_idx}: {best_params}")

    # 用最佳参数在整个训练集上训练模型
    X_train_tensor = torch.tensor(X_train_selected, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)

    # 使用最佳参数重新训练模型
    best_model = train_model(X_train_tensor, y_train_tensor, best_params['lr'], best_params['weight_decay'])

    # 在外层测试集上评估最佳模型
    X_test_tensor = torch.tensor(X_test_selected, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)

    best_model.eval()
    test_preds = []
    test_targets = []
    with torch.no_grad():
        y_pred = best_model(X_test_tensor)
        test_preds.extend(y_pred.cpu().numpy().flatten())
        test_targets.extend(y_test_tensor.cpu().numpy().flatten())

    mse_error = mean_squared_error(test_targets, test_preds)
    test_score = roc_auc_score(test_targets, test_preds)
    r2 = r2_score(test_targets, test_preds)
    writer.add_scalar('Test/MSE', mse_error, 0)
    writer.add_scalar('Test/AUC', test_score, 0)

    print(f"R^2 Value: {r2:.2f}")
    print(f"Test MSE error for this fold: {mse_error}")
    print(f"Test AUC score for this fold: {test_score}")

    # 保存最佳模型
    save_torch_model(best_model, config['model_save_path'], experiment, fold_idx, f'{model_selection}_{cov}')

    writer.close()
    return test_score, best_model, test_preds

