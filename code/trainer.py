import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, roc_auc_score, confusion_matrix, precision_score, recall_score, \
    f1_score, accuracy_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
import joblib

from train_nn import train_nn
from utils import *

def model(data, label, model_selection, config, experiment, feature_names, cov):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    set_seed(0)

    if 'Age' in data.columns:
        scaler = StandardScaler()
        data['Age'] = scaler.fit_transform(data[['Age']])

    # 标准预处理步骤
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    if not isinstance(label, pd.Series):
        label = pd.Series(label)

    def get_sklearn_model(model_selection, param_dict=None):

        # 设置模型和参数
        if model_selection == 'LA':
            model = Lasso(random_state=0, max_iter=10000, **param_dict)
        elif model_selection == 'EN':
            model = ElasticNet(random_state=0, max_iter=10000, **param_dict)
        elif model_selection == 'RF':
            model = RandomForestRegressor(random_state=0, **param_dict)
        elif model_selection == 'SVM_linear':
            model = LinearSVC(random_state=0, max_iter=10000, **param_dict)
        elif model_selection == 'SVM_rbf':
            model = SVC(kernel='rbf', random_state=0, **param_dict)
        return model

    cv_outer = RepeatedStratifiedKFold(n_splits=config['cv_out'], n_repeats=config['cv_repeats'], random_state=0)
    cv_inner = RepeatedStratifiedKFold(n_splits=config['cv_in'], n_repeats=1, random_state=0)

    best_models = []
    test_scores = []
    all_confusion_matrices = []
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1s = []

    fold_idx = 0
    for train_idx, test_idx in cv_outer.split(data, label):
        fold_idx += 1
        X_train, X_test = data.iloc[train_idx], data.iloc[test_idx]
        y_train, y_test = label.iloc[train_idx], label.iloc[test_idx]

        print(f"Fold {fold_idx} data shapes: X_train: {X_train.shape}, X_test: {X_test.shape}")

        if model_selection in ['LA', 'EN', 'RF', 'SVM_linear', 'SVM_rbf']:
            param_grid = config['hyperparameters_' + model_selection]
            model = get_sklearn_model(model_selection, param_grid)
            scorer = make_scorer(mean_squared_error, greater_is_better=False)

            inner_clf = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv_inner, scoring=scorer, return_train_score=True, n_jobs=-1)
            inner_clf.fit(X_train, y_train)

            best_params = inner_clf.best_params_
            print(f"Best parameters from inner CV for fold {fold_idx}: {best_params}")

            # 2. 重新实例化一个新的模型对象，并设置最佳参数
            new_model = get_sklearn_model(model_selection, best_params)
            
            # 3. 在 outer 训练集上进行训练
            new_model.fit(X_train, y_train)

            # 4. 在 outer 测试集上预测并打分
            y_pred_test = new_model.predict(X_test)
            test_score = roc_auc_score(y_test, y_pred_test)
            test_scores.append(test_score)
            best_models.append(new_model)

            print(f"Fold {fold_idx} Test AUC: {test_score}")

            # 5. 保存模型
            if model_selection == 'EN':
                print(f"Fold {fold_idx} model coefficients: {len(new_model.coef_)}")
                print(new_model.coef_)
                model_filename = os.path.join(
                    config['model_save_path'], "pathway32", f"{model_selection}_{cov}", 
                    f"fold{fold_idx}.pt"
                )
                print(f"Saving ElasticNet model for fold {fold_idx} to: {model_filename}")
                joblib.dump(new_model, model_filename)

        elif model_selection == 'LR':
            model = LinearRegression()
            scorer = make_scorer(mean_squared_error, greater_is_better=False)
            final_model = model.fit(X_train, y_train)
            y_pred_test = final_model.predict(X_test)
            test_score = roc_auc_score(y_test, y_pred_test)

            best_models.append(final_model)
            test_scores.append(test_score)

            print(f"Test score for this fold: {test_score}")

        elif model_selection == 'NN':

            test_score, best_model, y_pred_test = train_nn(X_train, y_train, X_test, y_test, cv_inner, config, device, \
                                                           fold_idx, experiment, feature_names, model_selection, cov)
            best_models.append(best_model)
            test_scores.append(test_score)

        # 找到最佳阈值并计算混淆矩阵及各项指标
        best_threshold, best_metric_score = find_best_threshold(y_test, y_pred_test, metric='f1')
        y_pred_binary = (y_pred_test >= best_threshold).astype(int)

        cm = confusion_matrix(y_test, y_pred_binary)
        all_confusion_matrices.append(cm)
        print(f"Confusion Matrix for fold {fold_idx}:")
        print(cm)

        accuracy = accuracy_score(y_test, y_pred_binary)
        precision = precision_score(y_test, y_pred_binary)
        recall = recall_score(y_test, y_pred_binary)
        f1 = f1_score(y_test, y_pred_binary)

        all_accuracies.append(accuracy)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)

        print(f"Accuracy for fold {fold_idx}: {accuracy:.2f}")
        print(f"Precision for fold {fold_idx}: {precision:.2f}")
        print(f"Recall for fold {fold_idx}: {recall:.2f}")
        print(f"F1 Score for fold {fold_idx}: {f1:.2f}")

        # 保存错误样本的ID
        false_positives = y_test.index[(y_test == 0) & (y_pred_binary == 1)]
        false_negatives = y_test.index[(y_test == 1) & (y_pred_binary == 0)]
        true_positives = y_test.index[(y_test == 1) & (y_pred_binary == 1)]
        true_negatives = y_test.index[(y_test == 0) & (y_pred_binary == 0)]

        error_save_dir = f'{config["results_save_path"]}/{experiment}_{model_selection}_{cov}_fold_errors'
        if not os.path.exists(error_save_dir):
            os.makedirs(error_save_dir)

        # 保存错误样本并按照误差排序
        save_errors(y_test, y_pred_test, false_positives, false_negatives, true_positives, true_negatives, error_save_dir, fold_idx)

    if not test_scores:
        print("No models were successfully trained and evaluated. Check your data and model configuration.")
        raise ValueError("No models were successfully trained and evaluated. Check your data and model configuration.")

    best_model_idx = np.argmax(test_scores)
    final_best_model = best_models[best_model_idx]

    print(f"Average test AUC across all folds: {np.mean(test_scores)}")

    # 计算平均混淆矩阵和指标
    average_confusion_matrix = np.mean(all_confusion_matrices, axis=0)
    print("Average Confusion Matrix:")
    print(average_confusion_matrix)

    avg_accuracy = np.mean(all_accuracies)
    avg_precision = np.mean(all_precisions)
    avg_recall = np.mean(all_recalls)
    avg_f1 = np.mean(all_f1s)

    print(f"Average Accuracy across all folds: {avg_accuracy:.2f}")
    print(f"Average Precision across all folds: {avg_precision:.2f}")
    print(f"Average Recall across all folds: {avg_recall:.2f}")
    print(f"Average F1 Score across all folds: {avg_f1:.2f}")

    results_df = pd.DataFrame(test_scores, columns=['AUC'])
    results_df.to_csv(f'{config["results_save_path"]}/{experiment}_{model_selection}_{cov}.csv', index=False)

    save_model(final_best_model, config['model_save_path'], experiment, f'{model_selection}_{cov}')

    print(f"Average test AUC across all folds: {np.mean(test_scores)}")

    return test_scores
