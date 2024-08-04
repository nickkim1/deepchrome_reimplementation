import csv
import torch 
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import Dataset, DataLoader, BatchSampler
from matplotlib import pyplot as plt
import seaborn as sn
import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import time as time
import os
import argparse as ap

def run_all(tr_ds, v_ds, te_ds, o_dir, c_type): 

    """
    ** This is a script for running the baseline ML models: SVM, Random Forest, *XGBoost (not used in original paper) 
    for comparison against DeepChrome.

    This is a wrapper function around the core of model training and evaluation. This includes: 
    - Dataloading: see first function
    - Data preprocessing: see HistoneDataset and BatchSampler
    - Model definition
    - Model training
    - Model evaluation
    - Model figure plotting: namely loss, auc/roc, confusion matrix metrics
    """
    
    ##########################################################################################
    # DATA LOADING
    ##########################################################################################

    training_ds = tr_ds
    valid_ds = v_ds
    testing_ds = te_ds
    output_dir = o_dir
    cell_type = c_type

    num_rows = 0
    num_cols = 0
    num_windows = 100 
    num_zeros = 0
    num_ones = 0 
    marker = 0
    csv_file = None
    pos_weights = 0

    with open(training_ds,'r') as file: 
        csv_file = csv.reader(file)
        for row in csv_file: 
            if int(row[len(row)-1]) == 0:
                num_zeros += 1
            if int(row[len(row)-1]) == 1:
                num_ones += 1
            if marker == 0: 
                num_cols = len(row)
                marker += 1
            num_rows += 1
    num_genes = num_rows / num_windows
    pos_weights = num_zeros / num_ones

    ##########################################################################################
    # DATA PREPROCESSING AND DATASET DEFINITION
    ##########################################################################################

    class HistoneDataset(Dataset):
        def __init__(self, file_path): 
            xy = np.loadtxt(file_path, delimiter=",",dtype=np.float32)
            self.features = torch.from_numpy(xy[:, 2:num_cols-1])
            self.f = {}
            data = {} 
            counter = 0
            for idx, line in enumerate(torch.from_numpy(xy)):
                data[counter] = {}
                for col, val in enumerate(line):
                    data[counter][col] = val 
                if counter % num_windows == 0:
                    # at that gene_id -> get the first element of the line 
                    self.f[int(line[0].item())] = data[counter]
                counter+=1
            self.labels = torch.tensor(xy[:, num_cols-1]) # 1000 x 1
            self.gene_ids = torch.from_numpy(xy[:, 0]) # should be 1000 x 1
            self.indices = {}
            classes = []
            for idx, gene_id in enumerate(self.gene_ids):
                if idx % num_windows == 0:
                    # print('gene id: ', gene_id.item(), 'other: ', xy[idx, 0]) -- qc
                    classes.append(int(gene_id.item()))
                    # self.indices[int(gene_id.item())] = self.f[int(gene_id.item())]
                    self.indices[int(gene_id.item())] = np.arange(idx-100, idx, 1)
            self.classes = classes
            self.n_samples = num_rows 
        def __len__(self):
            return self.n_samples
        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]
        def gene_ids_and_indices(self):
            return self.classes, self.indices
        def return_features_labels(self): 
            return self.features, self.labels

    class BSampler(BatchSampler): 
        def __init__(self, gene_ids, indices, batch_size):
            super(BSampler, self).__init__(training_dataset, batch_size, drop_last=False)
            self.gene_ids = gene_ids 
            self.indices = indices 
            self.n_batches = int(num_rows / batch_size) 
            self.batch_size = batch_size 
        def __iter__(self):
            batches = []
            for _ in range(self.n_batches):
                batch = []
                batch_class = random.choice(self.gene_ids) 
                vals = torch.from_numpy(self.indices[batch_class])
                for ignore, idx in enumerate(vals):
                    batch.append(idx)
                batches.append(batch) 
            return iter(batches)
        def __len__(self):
            return self.n_batches

    # -- create the datasets -- histone dataset 
    training_dataset = HistoneDataset(training_ds)
    training_dataloader = DataLoader(dataset=training_dataset, batch_sampler=BSampler(gene_ids=training_dataset.gene_ids_and_indices()[0], indices=training_dataset.gene_ids_and_indices()[1], batch_size=100))
    if not v_ds: 
        test_dataset = HistoneDataset(testing_ds)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False, drop_last=False)
    else:
        valid_dataset = HistoneDataset(valid_ds)
        valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=100, shuffle=False, drop_last=False) 
    n_total_steps = len(training_dataloader)

    ##########################################################################################
    # MODEL DEFINITIONS
    ##########################################################################################

    # 10, 20, 30 ... etc, with a max size of 200 estimators.
    random_forest_parameters = {'n_estimators': [i for i in range(10, 201, 10)]}
    svm_parameters = {'kernel_type': 'rbf', 'C': [0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.01, 0.1, 1, 2]}

    # note - used default parameters for everything else 
    random_forest = RandomForestClassifier(n_estimators=random_forest_parameters['n_estimators'][0], criterion='log_loss')
    svm = SVC(kernel=svm_parameters['kernel_type'], C=svm_parameters['C'][0], gamma=svm_parameters['gamma'][0])
    xgboost = GradientBoostingClassifier(loss='log_loss')

    ##########################################################################################
    # MODEL TRAINING AND TESTING
    ##########################################################################################

    def train_test_svm(X_train, Y_train, X_test, Y_test): 
        clf = make_pipeline(StandardScaler(), svm(gamma='auto'))
        clf.fit(X_train, Y_train)
        clf.predict(X_test, Y_test)

    def train_test_random_forest(X_train, Y_train, X_test, Y_test): 
        clf = make_pipeline(StandardScaler(), random_forest(gamma='auto'))
        clf.fit(X_train, Y_train)
        clf.predict(X_test, Y_test)

    def train_test_xgboost(X_train, Y_train, X_test, Y_test): 
        clf = make_pipeline(StandardScaler(), xgboost(gamma='auto'))
        clf.fit(X_train, Y_train)
        clf.predict(X_test, Y_test)

    train_test_svm(training_dataset.return_features_labels[0], training_dataset.return_features_labels[1], test_dataset.return_features_labels[0], test_dataset.return_features_labels[1])
    train_test_random_forest(training_dataset.return_features_labels[0], training_dataset.return_features_labels[1], test_dataset.return_features_labels[0], test_dataset.return_features_labels[1])
    train_test_xgboost(training_dataset.return_features_labels[0], training_dataset.return_features_labels[1], test_dataset.return_features_labels[0], test_dataset.return_features_labels[1])

def main():

    parser = ap.ArgumentParser(prog='Reimplementation of Deepchrome!', description='[1] Written in Pytorch. [2] All files must be titled train.csv, test.csv, or valid.csv. [3] All cell types should have respective directory named after them.')
    parser.add_argument('-d', type=str, help='-> input name of dataset directory to use', nargs=1)
    parser.add_argument('-tv', type=str, help='-> input test or valid', nargs=1)
    parser.add_argument('-s', type=str, help='-> input Y or N', nargs=1)
    parser.add_argument('-o', type=str, help='-> input name of directory to save results to', nargs=1)
    parser.add_argument('-e', type=int, help='-> input number of epochs you want to run the model for', nargs=1)
    parser.add_argument('-sm', type=str, help='-> input name of directory to save model to')

    # -- parse all the args -> list format 
    args = vars(parser.parse_args())

    # -- parse where the data directory is 
    data_directory = args['d'][0]
    output_directory = ''

    # -- parse where the model should be saved
    model_path = args['sm']

    # -- check if the results should be sent to another directory or not 
    if args['s'][0] == 'Y':  
        output_directory = args['o'][0] # -- the directory it should be saved to 
    else: 
        output_directory = 'DELETE'

    # -- recurses down to the lowest level subdirectory to retrieve filepaths for train.csv, test.csv, and/or valid.csv
    lowest_dirs = []
    for root, dirs, files_ignore in os.walk(data_directory): 
        if not dirs: 
            lowest_dirs.append(root)
    all_cell_types = next(os.walk(data_directory))[1] # -- this obtains the cell types

    # -- iterate now through the directories and run the model on their respective data 
    marker=0
    for dirs in lowest_dirs: 
        if args['tv'][0] == 'valid': # -- check if valid over test dataset 
            run_all(tr_ds=f'{dirs}/train.csv', v_ds=f'{dirs}/valid.csv', te_ds=None, c_type=all_cell_types[marker], o_dir=output_directory)
        elif args['tv'][0] == 'test': # -- check if test over valid dataset 
            run_all(tr_ds=f'{dirs}/train.csv', te_ds=f'{dirs}/valid.csv',  v_ds=None, c_type=all_cell_types[marker], o_dir=output_directory)

if __name__ == '__main__':
    main()
