# %%
# this examlpe shows how to build a ML mode to predict disruption and
# evaluate its performance using jddb
# this depands on the output FileRepo of basic_data_processing.py

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import lightgbm as lgb
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from jddb.performance import Result
from jddb.performance import Report
from jddb.file_repo import FileRepo


# %% define function to build model specific data


def matrix_build(shot_list, file_repo, tags):
    """
    get x and y from file_repo by shots and tags
    Args:
        shot_list: shots for data matrix
        file_repo:
        tags: tags from file_repo

    Returns: matrix of x and y

    """
    x_set = np.empty([0, len(tags) - 1])
    y_set = np.empty([0])
    for shot in shot_list:
        shot = int(shot)
        x_data = file_repo.read_data(shot, tags)
        y_data = file_repo.read_data(shot, ['alarm_tag'])
        x_data.pop('alarm_tag', None)
        res = np.array(list(x_data.values())).T
        res_y = np.array(list(y_data.values())).T.flatten()
        x_set = np.append(x_set, res, axis=0)
        y_set = np.append(y_set, res_y, axis=0)
    return x_set, y_set


# inference on shot


def get_shot_result(y_pred, threshold_sample):
    """
    get shot result by a threshold and compare to start time
    Args:
        y_pred: sample result from model
        threshold_sample: disruptive predict level

    Returns:
        shot predict result:The prediction result for the shot

    """
    binary_result = 1 * (y_pred >= threshold_sample)
    for k in range(len(binary_result) - 2):
        if np.sum(binary_result[k:k + 3]) == 3:
            predicted_dis = 1
            break
        else:
            predicted_dis = 0
    return predicted_dis


if __name__ == '__main__':

    test_file_repo = FileRepo(
        "..//FileRepo//train_file//$shot_2$00//")
    test_shot_list = test_file_repo.get_all_shots()
    print(len(test_shot_list))
    tag_list = test_file_repo.get_tag_list(test_shot_list[0])
    # disruption tag for dataset split
    is_disrupt = []
    for shot in test_shot_list:
        dis_label = test_file_repo.read_labels(shot, ['IsDisrupt'])
        is_disrupt.append(dis_label['IsDisrupt'])

    # %% build model specific data
    # train test split on shot not sample according to whether shots are disruption
    # set test_size=0.5 to get 50% shots as test set
    train_shots, test_shots, _, _ = \
        train_test_split(test_shot_list, is_disrupt, test_size=0.2,
                         random_state=1, shuffle=True, stratify=is_disrupt)

    # # create x and y matrix for ML models
    # # %%
    X_train, y_train = matrix_build(train_shots, test_file_repo, tag_list)
    X_test, y_test = matrix_build(test_shots, test_file_repo, tag_list)
    lgb_train = lgb.Dataset(X_train, y_train)  # create dataset for LightGBM
    lgb_val = lgb.Dataset(X_test, y_test)  # create dataset for LightGBM

    # %% use LightGBM to train a model.
    # hyper-parameters
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},

        'is_unbalance': True

    }
    evals_result = {}  # to record eval results for plotting
    print('Starting training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=300,
                    valid_sets={lgb_train, lgb_val},
                    evals_result=evals_result,
                    early_stopping_rounds=30)

    # %% generate result and evaluate
    # generate predictions for each shot
    shot_nos = test_shots  # shot list
    shots_pred_disrurption = []  # shot predict result
    shots_true_disruption = []  # shot true disruption label
    shots_pred_disruption_time = []  # shot predict time
    for shot in test_shots:
        true_disruption = 0 if test_file_repo.read_labels(
            shot)["IsDisrupt"] == False else 1
        shots_true_disruption.append(true_disruption)

        X, _ = matrix_build([shot], test_file_repo, tag_list)
        # get sample result from LightGBM
        y_pred = gbm.predict(X, num_iteration=gbm.best_iteration)

        # using the sample reulst to predict disruption on shot
        predicted_disruption = get_shot_result(
            y_pred, .5)  # get shot result by a threshold
        shots_pred_disrurption.append(predicted_disruption)

    # add predictions for each shot to the result dataframe
    pred_result = pd.DataFrame({'Shot': shot_nos,
                                'shot_pred': shots_pred_disrurption})
    pred_result.to_csv(r'..\_temp_test\test_result.csv')

    # %% plot some of the result: confusion matrix
    matrix = confusion_matrix(shots_true_disruption, shots_pred_disrurption)
    sns.heatmap(matrix, annot=True, cmap="Blues", fmt='.0f')
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    # plt.savefig(os.path.join('.//_temp_test//', 'Confusion Matrix.png'), dpi=300)
    plt.show()
