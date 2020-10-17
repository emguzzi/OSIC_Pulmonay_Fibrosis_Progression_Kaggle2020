#  Copyright (c) 2020.
#  Author: Silvio
from PIL import Image
import numpy as np
import pandas as pd
import pydicom
from skimage import transform
import torch.utils.data
from torchvision import transforms
import os
from sklearn import preprocessing
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm, trange
import random
import torch
import torch.utils.data
import csv
import time as time
import matplotlib.pyplot as plt
from sklearn import random_projection
import random


# =============================================================================
# Def Regression NN
# =============================================================================
class RegressionNetwork(nn.Module):
    def __init__(self, input_size=1051, activation=nn.Softplus()):
        super(RegressionNetwork1, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 1)
        self.drop = nn.Dropout(p=0.3)
        self.activation = activation

    def forward(self, x):
        m = self.activation
        out = m(self.fc1(x))
        out = self.drop(out)
        out = m(self.fc2(out))
        out = self.drop(out)
        out = m(self.fc3(out))
        out = self.drop(out)
        out = m(self.fc4(out))
        out = self.drop(out)
        out = m(self.fc5(out))
        out = self.drop(out)
        out = -m(self.fc6(out))  # use - as we expect a decay
        return out


def metric(true_fvc, predicted_fvc, confidence):
    # =============================================================================
    # Returns metric as given by the competition
    # =============================================================================
    sigma_cliped = max(confidence, 70)
    delta = min(abs(true_fvc - predicted_fvc), 1000)
    met = -np.sqrt(2) * delta / sigma_cliped - np.log(np.sqrt(2) * sigma_cliped)
    return met


def compute_score_cv(submission, test):
    # =============================================================================
    # submission is in the format given by predict as pandas while test is the test.csv as list
    # Returns score as given by the competition
    # =============================================================================
    score = []
    for row in test:
        true_fvc = row[1]
        week = row[0]
        patient = row[4]
        # on euler we have to keep the last index [:2] while local [:1] as in compute score
        predicted_fvc = submission.loc[submission['Patient_Week'] == patient + '_' + str(int(float(week)))][
            ['FVC', 'Confidence']].values[0][0]
        confidence = submission.loc[submission['Patient_Week'] == patient + '_' + str(int(float(week)))][
            ['FVC', 'Confidence']].values[0][1]
        score.append(metric(float(true_fvc), float(predicted_fvc), float(confidence)))
    return sum(score) / len(score)


def compute_score(submission, test):
    # =============================================================================
    # submission is in the format given by predict1 as panda while test is the test.csv as list
    # Returns score as given by the competition
    # =============================================================================
    score = []
    for row in test:
        true_fvc = row[2]
        week = row[1]
        patient = row[0]
        predicted_fvc = submission.loc[submission['Patient_Week'] == patient + '_' + str(int(float(week)))][
            ['FVC', 'Confidence']].values[0][0]
        confidence = submission.loc[submission['Patient_Week'] == patient + '_' + str(int(float(week)))][
            ['FVC', 'Confidence']].values[0][1]
        score.append(metric(float(true_fvc), float(predicted_fvc), float(confidence)))

    return sum(score) / len(score)


def prepare_val(data_path, working_path, n_val):
    # =============================================================================
    #     choose n_val random patient to validate the model, and create
    #     3 files val_train.csv, val_test.csv, val_true.csv st they can be
    #     plugged in the pipeline to get a validation score
    # =============================================================================
    pd_train = pd.read_csv(data_path + 'train.csv')
    ids = pd_train['Patient'].unique()

    # select n_val id at random
    val_ids = random.sample(list(ids), n_val)

    # avoid problematic id for validation
    prob1 = 'ID00011637202177653955184'
    prob2 = 'ID00052637202186188008618'
    while (prob1 in val_ids or prob2 in val_ids):
        val_ids = random.sample(list(ids), n_val)

    # remove the corresponding row from pd_train and save as val_train
    pd_train[pd_train['Patient'].isin(val_ids) == False].to_csv(working_path + 'val_train.csv', index=False)

    # save the validation values
    pd_train[pd_train['Patient'].isin(val_ids)].to_csv(working_path + 'val_true.csv', index=False)

    # prepare the test file
    val_test = pd.DataFrame()
    for patient in val_ids:
        patient_line = pd_train[pd_train['Patient'] == patient].iloc[0, :]
        val_test = val_test.append(patient_line, ignore_index=True)
    cols = ['Patient', 'Weeks', 'FVC', 'Percent', 'Age', 'Sex', 'SmokingStatus']
    val_test = val_test[cols]
    val_test.to_csv(working_path + 'val_test.csv', index=False)


def prepare_train_data(data_path, working_path, validation):
    # =============================================================================
    #     prepare the data for the prediction in the same format
    #     as for prepare_train_data
    # =============================================================================

    if validation:
        train = pd.read_csv(working_path + 'val_train.csv')
        folder = 'train/'
    else:
        train = pd.read_csv(data_path + 'train.csv')
        folder = 'train/'

    train_ids = train['Patient'].unique()
    n_dicom_dict = {"Patient": [], "n_dicom": [], "list_dicom": []}

    for Patient_id in train_ids:
        n_dicom_dict["n_dicom"].append(len(os.listdir(data_path + folder + Patient_id)))
        n_dicom_dict["Patient"].append(Patient_id)
        list_dicom_id = sorted(list(np.random.choice(
            np.array([int(i.split("/")[-1][:-4]) for i in os.listdir(data_path + folder + Patient_id)]), 10)))
        n_dicom_dict["list_dicom"].append(list_dicom_id)
    dicom_pd = pd.DataFrame(n_dicom_dict)
    temp_pd = pd.DataFrame(columns=train.columns)

    for i in range(len(dicom_pd)):
        patient_pd = train[train.Patient == dicom_pd.iloc[i].Patient]
        zeroweek = patient_pd['Weeks'].min()
        temp_pd = temp_pd.append(patient_pd[patient_pd.Weeks == zeroweek].iloc[0])
    dicom_pd = pd.merge(dicom_pd, temp_pd, on=['Patient'])
    dicom_pd.rename(columns={'FVC': 'BaseFVC', 'Weeks': 'BaseWeek', 'Percent': 'BasePercent'}, inplace=True)
    df = pd.DataFrame(columns=dicom_pd.columns)
    df['Weeks'] = None
    df['FVC'] = None
    for i in range(len(dicom_pd)):
        dicom_pd_patient = dicom_pd[dicom_pd.Patient == dicom_pd.iloc[i].Patient]
        patient_pd = train[train.Patient == dicom_pd.iloc[i].Patient]

        dicom_pd_patient = pd.concat([dicom_pd_patient] * len(patient_pd), axis=0, ignore_index=True)

        dicom_pd_patient = pd.concat(
            [dicom_pd_patient.reset_index(drop=True), patient_pd[['Weeks', 'FVC']].reset_index(drop=True)], axis=1)
        df = df.append(dicom_pd_patient)
    dicom_pd = df
    # one hot encoding for Sex & SmokingStatus
    cols = ['Sex', 'SmokingStatus']
    values = {'Sex': ['Male', 'Female'], 'SmokingStatus': ['Ex-smoker', 'Never smoked', 'Currently smokes']}
    for col in cols:
        for val in values[col]:
            dicom_pd[val] = (dicom_pd[col] == val).astype(int)
        dicom_pd.drop(col, axis=1, inplace=True)
    Y_train = dicom_pd[['Weeks', 'FVC', 'BaseWeek', 'BaseFVC']]
    # set the columns of X_test in the same order as the one in X_train.csv
    cols = ['Patient', 'Age', 'Male', 'Female', 'Ex-smoker', 'Never smoked', 'Currently smokes', 'BaseWeek',
            'BaseFVC', 'BasePercent', 'list_dicom']

    dicom_pd = dicom_pd[cols]
    dicom_pd.to_csv(working_path + 'X_train_new.csv', index=False)
    Y_train.to_csv(working_path + 'Y_train_new.csv', index=False)
    return dicom_pd, Y_train


def my_loss(output, target_fvc, week_from_0, base_week, base_fvc):
    # =============================================================================
    # torch loss function as which computes the MSE of the linear function with slope
    # output and which goes trough (base_week,base_fvc) at week_from_0 (true val is target_fvc)
    # =============================================================================
    loss = torch.mean((output * (week_from_0 - base_week) + base_fvc - target_fvc) ** 2)
    return loss


def train_linear_model(data_path, working_path, net, pd_X0, pd_Y0,
                       num_epochs, batch_size, learning_rate):
    # =============================================================================
    # Trains the 'net' model 'num_epochs' of times wiht standard paramenters if not stated differentely
    #     num_epochs = 1
    #     batch_size = 50
    #     learning_rate = 0.005
    #     feature_size = 25088
    #     net = RegressionNetwork()
    # =============================================================================

    # This fixes the random seed in order to have reproducible date
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    pd_X = pd_X0
    pd_Y = pd_Y0
    index_feat = torch.tensor(np.load(working_path + 'feat_with_var.npy'))
    optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_values = []
    pd_list_scans = pd_X[['Patient', 'list_dicom']]
    pd_X = pd_X.drop(['list_dicom'], axis=1)
    np_X = pd_X.to_numpy()
    np_Y = pd_Y.to_numpy()
    # encode the ids using sklearn labelencoders
    lab_enc_train = preprocessing.LabelEncoder()
    lab_enc_train.fit(np_X[:, 0])
    np_X[:, 0] = lab_enc_train.transform(np_X[:, 0])

    # prepare the data as pt
    pt_train = torch.from_numpy(np_X.astype(float)).float()
    pt_Y = torch.from_numpy(np_Y.astype(float)).float()

    for epoch in range(num_epochs):
        print('epoch' + str(epoch + 1))
        total_computations = 0
        train_loss = 0.0
        net.train()
        train = torch.utils.data.TensorDataset(pt_train, pt_Y)
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size)
        for (X, Y) in train_loader:
            optimiser.zero_grad()  # start form 0
            # get the ids (as string) of the patient i
            ids = lab_enc_train.inverse_transform(X[:, 0].numpy().astype(int))
            pt_X_to_feed = torch.empty((0, X.shape[1] + index_feat.shape[0] - 1)).float()
            for patient in ids:
                list_scans = pd_list_scans[pd_list_scans['Patient'] == patient]['list_dicom'].values[0]
                pd_patient = pd_X[(pd_X['Patient'] == patient)]
                pd_patient.drop('Patient', axis=1, inplace=True)
                len_pd_pat0 = len(pd_patient)
                pd_patient = pd.concat([pd_patient] * len(list_scans), axis=0, ignore_index=True)
                pt_patient = torch.from_numpy(pd_patient.values.astype(float)).float()
                # read the feature for the scan
                features_batch_patient = torch.empty((0, index_feat.shape[0])).float()
                for scan in list_scans:
                    features = torch.load(working_path + 'feat_vect_vgg16/' + str(patient) + '/' + str(scan) + '.pt',
                                          map_location=torch.device('cpu')).float()
                    features = features.index_select(1, index_feat)
                    features_batch_patient = torch.cat((features_batch_patient, features), 0)
                features_batch_patient0 = features_batch_patient
                features_batch_patient = torch.cat([features_batch_patient0 for k in range(len_pd_pat0)], 0)
                pt_patient = torch.cat((pt_patient, features_batch_patient), 1)
                pt_X_to_feed = torch.cat((pt_X_to_feed, pt_patient), 0)
                # predict the values
            outputs = net(pt_X_to_feed)
            loss = my_loss(output=outputs, target_fvc=Y[:, 1],
                           week_from_0=Y[:, 0], base_week=Y[:, 2], base_fvc=Y[:, 3])
            # backward and optimise
            loss.backward()  # computes gradient
            optimiser.step()  # updates the weights
            # update loss
            train_loss += loss.item()
            total_computations += batch_size
        loss_values.append(train_loss / total_computations)
    return net, loss_values


def prepare_test_data(data_path, working_path, validation):
    # =============================================================================
    #     prepare the data for the prediction in the same format
    #     as for prepare_train_data
    # =============================================================================

    if validation:
        test = pd.read_csv(working_path + 'val_test.csv')
        folder = 'train/'
    else:
        test = pd.read_csv(data_path + 'test.csv')
        folder = 'test/'
    tot_test = len(test)

    test_ids = test['Patient'].unique()
    n_dicom_dict = {"Patient": [], "n_dicom": [], "list_dicom": []}
    for Patient_id in test_ids:
        n_dicom_dict["n_dicom"].append(len(os.listdir(data_path + folder + Patient_id)))
        n_dicom_dict["Patient"].append(Patient_id)
        # list_dicom_id = sorted(list(np.random.choice(np.array([int(i.split("/")[-1][:-4]) for i in os.listdir(data_path + folder + Patient_id)]),10)))
        scan_names = np.sort(
            np.array([int(i.split("/")[-1][:-4]) for i in os.listdir(data_path + folder + Patient_id)]))
        delta = int(len(scan_names) / 10)
        list_dicom_id = [scan_names[i * delta] for i in range(10)]
        n_dicom_dict["list_dicom"].append(list_dicom_id)
    dicom_pd = pd.DataFrame(n_dicom_dict)
    temp_pd = pd.DataFrame(columns=test.columns)
    for i in range(len(dicom_pd)):
        patient_pd = test[test.Patient == dicom_pd.iloc[i].Patient]
        zeroweek = patient_pd['Weeks'].min()
        temp_pd = temp_pd.append(patient_pd[patient_pd.Weeks == zeroweek].iloc[0])
    dicom_pd = pd.merge(dicom_pd, temp_pd, on=['Patient'])

    # one hot encoding for Sex & SmokingStatus
    cols = ['Sex', 'SmokingStatus']
    values = {'Sex': ['Male', 'Female'], 'SmokingStatus': ['Ex-smoker', 'Never smoked', 'Currently smokes']}
    for col in cols:
        for val in values[col]:
            dicom_pd[val] = (dicom_pd[col] == val).astype(int)
        dicom_pd.drop(col, axis=1, inplace=True)
    dicom_pd.rename(columns={'FVC': 'BaseFVC', 'Weeks': 'BaseWeek', 'Percent': 'BasePercent'}, inplace=True)
    weeks = []
    diff = []
    for week in range(-12, 134):
        weeks.extend([week for i in range(0, tot_test)])
        diff.extend([week - dicom_pd.loc[i, 'BaseWeek'] for i in range(0, tot_test)])

    cols = ['Patient', 'Age', 'Male', 'Female', 'Ex-smoker', 'Never smoked', 'Currently smokes', 'BaseWeek',
            'BaseFVC', 'BasePercent', 'list_dicom']

    dicom_pd = dicom_pd[cols]
    dicom_pd.to_csv(working_path + 'X_test.csv', index=False)
    return dicom_pd


def predict_new_linear_model(pd_test0, net, working_path, coef_confi):
    # =============================================================================
    # Generates a 'submission.csv' file as requested by the competition by leting go pd_test0 trough the net
    # =============================================================================
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    pd_test = pd_test0
    weeks = range(-12, 134)
    ids = pd_test['Patient'].unique()
    net.eval()

    index_feat = torch.tensor(np.load(working_path + 'feat_with_var.npy'))

    pd_list_scans = pd_test[['Patient', 'list_dicom']]
    pd_base = pd_test[['Patient', 'BaseFVC', 'BaseWeek']]
    np_base = pd_base.to_numpy()
    pd_test = pd_test.drop(['list_dicom'], axis=1)
    np_X = pd_test.to_numpy()
    # encode the ids using sklearn labelencoders
    lab_enc_train = preprocessing.LabelEncoder()
    lab_enc_train.fit(np_X[:, 0])
    np_X[:, 0] = lab_enc_train.transform(np_X[:, 0])

    # prepare the data as pt
    pt_test = torch.from_numpy(np_X.astype(float)).float()

    predictions_list = []
    predictions_slope_list = []
    pt_X_to_feed = torch.empty((0, pt_test.shape[1] + index_feat.shape[0] - 1)).float()
    for patient in ids:
        base_week = pd_base[pd_base['Patient'] == patient]['BaseWeek'].values[0]
        base_fvc = pd_base[pd_base['Patient'] == patient]['BaseFVC'].values[0]
        list_scans = pd_list_scans[pd_list_scans['Patient'] == patient]['list_dicom'].values[0]
        pd_patient = pd_test[(pd_test['Patient'] == patient)]
        pd_patient.drop('Patient', axis=1, inplace=True)
        len_pd_pat0 = len(pd_patient)
        pd_patient = pd.concat([pd_patient] * len(list_scans), axis=0, ignore_index=True)
        pt_patient = torch.from_numpy(pd_patient.values.astype(float)).float()
        # read the feature for the scan
        features_batch_patient = torch.empty((0, index_feat.shape[0])).float()
        for scan in list_scans:
            features = torch.load(working_path + 'feat_vect_vgg16/' + str(patient) + '/' + str(scan) + '.pt',
                                  map_location=torch.device('cpu')).float()
            features = features.index_select(1, index_feat)
            features_batch_patient = torch.cat((features_batch_patient, features), 0)
        features_batch_patient0 = features_batch_patient
        features_batch_patient = torch.cat([features_batch_patient0 for k in range(len_pd_pat0)], 0)
        pt_patient = torch.cat((pt_patient, features_batch_patient), 1)
        pt_X_to_feed = torch.cat((pt_X_to_feed, pt_patient), 0)
        # predict the values
        Y_predicted = net(pt_X_to_feed).detach().numpy()

        slope = np.mean(Y_predicted[0])
        for week in weeks:
            fvc = slope * (week - base_week) + base_fvc
            pred = {'Patient_Week': patient + '_' + str(week), 'FVC': fvc,
                    'Confidence': 70 + coef_confi[0] + abs(week - base_week) * coef_confi[1]}
            predictions_list.append(pred)
    predictions = pd.DataFrame(predictions_list)
    predictions.to_csv(working_path + 'submission.csv', index=False)


def analyse_performance(data_path, working_path, net, pd_X, pd_Y,
                        num_epochs, batch_size, learning_rate, feature_size,
                        pd_test, coef_confi=3, validation=True):
    # =============================================================================
    # Outputs a plot with loss values and validation score
    # =============================================================================
    with open(working_path + 'val_true.csv') as f:
        reader = csv.reader(f)
        test = list(reader)

    test = test[1:]

    losses = []
    score = []
    for i in range(num_epochs):
        net, losses_one = train_new_linear_model(data_path, working_path, net, pd_X, pd_Y,
                                                 1, batch_size, learning_rate, feature_size)
        losses_one = losses_one[0]
        losses.append(losses_one)

        predict_new_linear_model(pd_test, net, working_path, coef_confi=3)
        predictions = pd.read_csv(working_path + 'submission.csv')

        score.append(compute_score(predictions, test))
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss', color=color)
        ax1.plot(losses, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('Validation Score', color=color)  # we already handled the x-label with ax1
        ax2.plot(score, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig('Loss_vs_ValScore.png')


def prepare_fold(data_path, number_fold):
    # =============================================================================
    # return number_fold dataframe with disjoint ids to perform cv
    # =============================================================================

    pd_train = pd.read_csv(data_path + 'train.csv')
    ids = pd_train['Patient'].unique()

    patient_per_fold = int(len(ids) / number_fold)
    # shuffle the ids before dividing them into folds
    random.shuffle(ids)

    # create the df corresponding to the folds
    fold_data = {}
    for fold in range(number_fold):
        if fold == number_fold - 1:
            # include in the last fold all the remaining ids
            X = pd_train[pd_train['Patient'].isin(ids[fold * patient_per_fold:]) == True]

        else:
            X = pd_train[pd_train['Patient'].isin(ids[fold * patient_per_fold:(fold + 1) * patient_per_fold]) == True]

        fold_data[fold] = X

    return fold_data


def prepare_train_data_cv(data_path, working_path, train, validation):
    # =============================================================================
    # same as predict_train_data but train is given as input so that this works with the validation
    # =============================================================================
    folder = 'train/'

    train_ids = train['Patient'].unique()
    n_dicom_dict = {"Patient": [], "n_dicom": [], "list_dicom": []}

    for Patient_id in train_ids:
        n_dicom_dict["n_dicom"].append(len(os.listdir(data_path + folder + Patient_id)))
        n_dicom_dict["Patient"].append(Patient_id)
        # list_dicom_id = sorted(list(np.random.choice(
        #    np.array([int(i.split("/")[-1][:-4]) for i in os.listdir(data_path + folder + Patient_id)]), 10)))
        # instead of taking randomly selected scan take linearly spaced scan
        scan_names = np.sort(
            np.array([int(i.split("/")[-1][:-4]) for i in os.listdir(data_path + folder + Patient_id)]))
        delta = int(len(scan_names) / 10)
        list_dicom_id = [scan_names[i * delta] for i in range(10)]
        n_dicom_dict["list_dicom"].append(list_dicom_id)
    dicom_pd = pd.DataFrame(n_dicom_dict)
    temp_pd = pd.DataFrame(columns=train.columns)

    for i in range(len(dicom_pd)):
        patient_pd = train[train.Patient == dicom_pd.iloc[i].Patient]
        zeroweek = patient_pd['Weeks'].min()
        temp_pd = temp_pd.append(patient_pd[patient_pd.Weeks == zeroweek].iloc[0])
    dicom_pd = pd.merge(dicom_pd, temp_pd, on=['Patient'])
    dicom_pd.rename(columns={'FVC': 'BaseFVC', 'Weeks': 'BaseWeek', 'Percent': 'BasePercent'}, inplace=True)
    df = pd.DataFrame(columns=dicom_pd.columns)
    df['Weeks'] = None
    df['FVC'] = None
    for i in range(len(dicom_pd)):
        dicom_pd_patient = dicom_pd[dicom_pd.Patient == dicom_pd.iloc[i].Patient]
        patient_pd = train[train.Patient == dicom_pd.iloc[i].Patient]

        dicom_pd_patient = pd.concat([dicom_pd_patient] * len(patient_pd), axis=0, ignore_index=True)

        dicom_pd_patient = pd.concat(
            [dicom_pd_patient.reset_index(drop=True), patient_pd[['Weeks', 'FVC']].reset_index(drop=True)], axis=1)
        df = df.append(dicom_pd_patient)
    dicom_pd = df
    # one hot encoding for Sex & SmokingStatus
    cols = ['Sex', 'SmokingStatus']
    values = {'Sex': ['Male', 'Female'], 'SmokingStatus': ['Ex-smoker', 'Never smoked', 'Currently smokes']}
    for col in cols:
        for val in values[col]:
            dicom_pd[val] = (dicom_pd[col] == val).astype(int)
        dicom_pd.drop(col, axis=1, inplace=True)
    Y_train = dicom_pd[['Weeks', 'FVC', 'BaseWeek', 'BaseFVC']]
    # set the columns of X_test in the same order as the one in X_train.csv
    cols = ['Patient', 'Weeks', 'Age', 'Male', 'Female', 'Ex-smoker', 'Never smoked', 'Currently smokes', 'BaseWeek',
            'BaseFVC', 'BasePercent', 'list_dicom']
    X_train = dicom_pd[cols]
    # drop the line corresponding to the base week
    Y_train = Y_train[~(Y_train.Weeks == Y_train.BaseWeek)]
    X_train = X_train[~(X_train.Weeks == X_train.BaseWeek)]

    no_weeks = ['Patient', 'Age', 'Male', 'Female', 'Ex-smoker', 'Never smoked', 'Currently smokes', 'BaseWeek',
                'BaseFVC', 'BasePercent', 'list_dicom']
    X_train = X_train[no_weeks]
    X_train.to_csv(working_path + 'X_train_new.csv', index=False)
    Y_train.to_csv(working_path + 'Y_train_new.csv', index=False)

    return X_train, Y_train


def prepare_train_data_cv_scaler(data_path, working_path, train, validation, scaler):
    # =============================================================================
    #  same as prepare_train_data_cv but now some columns are scaled
    # =============================================================================
    folder = 'train/'

    train_ids = train['Patient'].unique()
    n_dicom_dict = {"Patient": [], "n_dicom": [], "list_dicom": []}

    for Patient_id in train_ids:
        n_dicom_dict["n_dicom"].append(len(os.listdir(data_path + folder + Patient_id)))
        n_dicom_dict["Patient"].append(Patient_id)
        scan_names = np.sort(
            np.array([int(i.split("/")[-1][:-4]) for i in os.listdir(data_path + folder + Patient_id)]))
        delta = int(len(scan_names) / 10)
        list_dicom_id = [scan_names[i * delta] for i in range(10)]
        n_dicom_dict["list_dicom"].append(list_dicom_id)
    dicom_pd = pd.DataFrame(n_dicom_dict)
    temp_pd = pd.DataFrame(columns=train.columns)

    for i in range(len(dicom_pd)):
        patient_pd = train[train.Patient == dicom_pd.iloc[i].Patient]
        zeroweek = patient_pd['Weeks'].min()
        temp_pd = temp_pd.append(patient_pd[patient_pd.Weeks == zeroweek].iloc[0])
    dicom_pd = pd.merge(dicom_pd, temp_pd, on=['Patient'])
    dicom_pd.rename(columns={'FVC': 'BaseFVC', 'Weeks': 'BaseWeek', 'Percent': 'BasePercent'}, inplace=True)
    df = pd.DataFrame(columns=dicom_pd.columns)
    df['Weeks'] = None
    df['FVC'] = None
    for i in range(len(dicom_pd)):
        dicom_pd_patient = dicom_pd[dicom_pd.Patient == dicom_pd.iloc[i].Patient]
        patient_pd = train[train.Patient == dicom_pd.iloc[i].Patient]

        dicom_pd_patient = pd.concat([dicom_pd_patient] * len(patient_pd), axis=0, ignore_index=True)

        dicom_pd_patient = pd.concat(
            [dicom_pd_patient.reset_index(drop=True), patient_pd[['Weeks', 'FVC']].reset_index(drop=True)], axis=1)
        df = df.append(dicom_pd_patient)
    dicom_pd = df
    # one hot encoding for Sex & SmokingStatus
    cols = ['Sex', 'SmokingStatus']
    values = {'Sex': ['Male', 'Female'], 'SmokingStatus': ['Ex-smoker', 'Never smoked', 'Currently smokes']}
    for col in cols:
        for val in values[col]:
            dicom_pd[val] = (dicom_pd[col] == val).astype(int)
        dicom_pd.drop(col, axis=1, inplace=True)
    Y_train = dicom_pd[['Weeks', 'FVC', 'BaseWeek', 'BaseFVC']]
    # set the columns of X_test in the same order as the one in X_train.csv
    cols = ['Patient', 'Weeks', 'Age', 'Male', 'Female', 'Ex-smoker', 'Never smoked', 'Currently smokes', 'BaseWeek',
            'BaseFVC', 'BasePercent', 'list_dicom']

    X_train = dicom_pd[cols]
    # drop the line corresponding to the base week
    Y_train = Y_train[~(Y_train.Weeks == Y_train.BaseWeek)]
    X_train = X_train[~(X_train.Weeks == X_train.BaseWeek)]

    no_weeks = ['Patient', 'Age', 'Male', 'Female', 'Ex-smoker', 'Never smoked', 'Currently smokes', 'BaseWeek',
                'BaseFVC', 'BasePercent', 'list_dicom']
    X_train = X_train[no_weeks]

    # scaler
    cols_to_std = ['Age', 'BaseWeek', 'BaseFVC', 'BasePercent']
    X_train[cols_to_std] = scaler.fit_transform(X_train[cols_to_std])

    X_train.to_csv(working_path + 'X_train_new.csv', index=False)
    Y_train.to_csv(working_path + 'Y_train_new.csv', index=False)

    return X_train, Y_train
