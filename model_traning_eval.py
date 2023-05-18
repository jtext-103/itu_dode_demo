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


def get_shot_result(y_red, threshold_sample,start_time):
    """
    get shot result by a threshold
    Args:
        y_red: sample result from model
        threshold_sample: disruptive predict level

    Returns: shot predict result and predict time

    """
    binary_result = 1 * (y_pred >= threshold_sample)
    for k in range(len(binary_result) - 2):
        if np.sum(binary_result[k:k + 3]) == 3:
            predicted_dis_time = (k + 2) / 1000 + start_time
            predicted_dis = 1
            break
        else:
            predicted_dis_time = -1
            predicted_dis = 0
    return predicted_dis, predicted_dis_time


# %% init FileRepo
if __name__ == '__main__':

    test_file_repo = FileRepo(
        "..//FileRepo//tag_file//$shot_2$00//")
    test_shot_list = test_file_repo.get_all_shots()
    print(len(test_shot_list))
    tag_list = test_file_repo.get_tag_list(test_shot_list[0])
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
    # save sample result to a dict, so when predicting shot with differnet trgging logic,
    # you don't have to re-infor the testshot

    # create an empty result object
    test_result = Result(r'..\_temp_test\test_result.csv')
    sample_result = dict()

    # generate predictions for each shot
    shot_nos = test_shots  # shot list
    shots_pred_disrurption = []  # shot predict result
    shots_pred_disruption_time = []  # shot predict time
    for shot in test_shots:
        X, _ = matrix_build([shot], test_file_repo, tag_list)
        # get sample result from LightGBM
        y_pred = gbm.predict(X, num_iteration=gbm.best_iteration)
        sample_result.setdefault(shot, []).append(
            y_pred)  # save sample results to a dict

        # using the sample reulst to predict disruption on shot, and save result to result file using result module.
        start_time = test_file_repo.read_labels(shot, ['StartTime'])
        predicted_disruption, predicted_disruption_time = get_shot_result(
            y_pred, .5,start_time)  # get shot result by a threshold
        shots_pred_disrurption.append(predicted_disruption)
        shots_pred_disruption_time.append(predicted_disruption_time)

    # add predictions for each shot to the result object
    test_result.add(shot_nos, shots_pred_disrurption,
                    shots_pred_disruption_time)
    # get true disruption label and time
    test_result.get_all_truth_from_file_repo(
        test_file_repo)

    test_result.lucky_guess_threshold = .8
    test_result.tardy_alarm_threshold = .005
    test_result.calc_metrics()
    test_result.save()
    print("precision = " + str(test_result.precision))
    print("tpr = " + str(test_result.tpr))

    # %% plot some of the result
    sns.heatmap(test_result.confusion_matrix, annot=True, cmap="Blues", fmt='.0f')
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    # plt.savefig(os.path.join('..//_temp_test//', 'Confusion Matrix.png'), dpi=300)
    plt.show()

    test_result.plot_warning_time_histogram(
        [-1, .002, .01, .05, .1, .3], '..//_temp_test//')
    test_result.plot_accumulate_warning_time('..//_temp_test//')

    # %% scan the threshold for shot prediction to get
    # many results, and add them to a report
    # simply change different disruptivity triggering level and logic, get many result.
    test_report = Report('..//_temp_test//report.csv')
    thresholds = np.linspace(0, 1, 50)
    for threshold in thresholds:
        shot_nos = test_shots
        shots_pred_disrurption = []
        shots_pred_disruption_time = []
        for shot in test_shots:
            y_pred = sample_result[shot][0]
            predicted_disruption, predicted_disruption_time = get_shot_result(
                y_pred, threshold)
            shots_pred_disrurption.append(predicted_disruption)
            shots_pred_disruption_time.append(predicted_disruption_time)
        # i dont save so the file never get created
        temp_test_result = Result('../_temp_test/temp_result.csv')
        temp_test_result.lucky_guess_threshold = .8
        temp_test_result.tardy_alarm_threshold = .001
        temp_test_result.add(shot_nos, shots_pred_disrurption,
                             shots_pred_disruption_time)
        temp_test_result.get_all_truth_from_file_repo(test_file_repo)

        # add result to the report
        test_report.add(temp_test_result, "thr="+str(threshold))
        test_report.save()
    # plot all metrics with roc
    test_report.plot_roc('../_temp_test/')
