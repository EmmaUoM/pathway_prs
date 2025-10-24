import time
import argparse

from trainer import model
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Run machine learning models with nested cross-validation.")
    parser.add_argument("-e", "--experiment", required=True, help="Experiment")
    parser.add_argument("-m", "--model_selection", required=True, choices=['RF', 'SVM_linear', 'SVM_rbf', 'LR', 'LA', \
                                                                           'EN', 'NN'], help="Model selection")
    parser.add_argument("-s", "--save_model", help="Path to save the trained model")
    parser.add_argument("-d", "--dataset", required=True, help="Dataset name")
    parser.add_argument("-c", "--covariate", required=True, choices=['None', 'sex', 'sex_age'], help="covariates")
    return parser.parse_args()


def main():
    start = time.time()

    args = parse_args()

    experiment = args.experiment
    config = load_config(experiment_name=experiment)
    model_selection = args.model_selection
    cov = args.covariate
    print("Experiment is", experiment)


    data, label, feature_names = load_prs(experiment, config, model_cov=cov)

    print("Feature names:", feature_names)
    print("Data shape:", data.shape)
    print("Label shape:", label.shape)

    model(data, label, model_selection, config, experiment, feature_names, cov)

    end = time.time()

    elapsed_time = end - start
    days, rem = divmod(elapsed_time, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    print('Finish, time taken: {:0>2} days {:0>2}:{:0>2}:{:05.2f}'.format(int(days), int(hours), int(minutes), seconds))

if __name__ == '__main__':
    main()
