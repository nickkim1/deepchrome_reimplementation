import csv
import torch 
import torch.nn as nn
import torch.nn.functional as F
import random
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader, BatchSampler
from matplotlib import pyplot as plt
import seaborn as sn
import pandas as pd 
import numpy as np
import math
import time as time
import os
import argparse as ap

def run_all(tr_ds, v_ds, te_ds, o_dir, c_type, n_epochs, model_path): 

    """
    ** This is a script for running the core Deepchrome model and generating associated statistics. 

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

    # -- first set the device used
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("===================================")
    print(f"0. Using {device} device!")

    # -- initially read in the data just to check dimensions
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

    print("===================================")
    print("INITIAL DIMENSIONS CHECK")
    print("1. The number of features/histone modification types is: ", num_cols-3)
    print("2. The number of genes is: ", num_genes)
    print("3. The number of windows and number of samples is: ", num_windows)
    print(f"4. The number of ones is {num_ones} and the number of zeros is {num_zeros}")
    print(f"5. The ratio of negative : positive examples is {pos_weights}")
    print("===================================")

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
        
    # -- custom BatchSampler class
    # -- creates (mini-batches of) indices and passes to the dataset to fetch corresponding samples
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
    # MODEL DEFINITION
    ##########################################################################################

    # - hyperparameters
    # -- entire model
    learning_rate = 0.001
    num_epochs = n_epochs
    batch_size = 100
    n_batches = int(num_rows / batch_size) # technically don't need here 
    num_features = 5

    # -- mlp 
    # num_hidden_layers = 2
    hidden_layer_units = {"first": 625, "second": 125}

    # -- cnn 
    num_filters = 50
    filter_size = 10
    pool_size = 5

    # -- unused parameters
    num_inputs = num_features * num_windows # should just be the size of each gene sample
    num_outputs = 2 
    stride_length = 1

    # - model class 
    # -- 1d cnn -> mlp 
    class ConvNet(nn.Module):
        def __init__(self): 
            super(ConvNet, self).__init__()
            self.conv = nn.Conv1d(in_channels=5, out_channels=50, kernel_size=10)
            self.pool = nn.MaxPool1d(pool_size)
            self.dropout = nn.Dropout(p=0.5)
            self.fc1 = nn.Linear(math.ceil((num_windows-filter_size)/pool_size)*num_filters, hidden_layer_units["first"])
            self.fc2 = nn.Linear(hidden_layer_units["first"], hidden_layer_units["second"])
            self.fc3 = nn.Linear(hidden_layer_units["second"], 2)
        def forward(self, x): 
            x = self.conv(x)
            x = F.relu(x)
            x = self.pool(x)
            x = x.view(math.ceil((num_windows-filter_size)/pool_size)*num_filters)
            x = self.dropout(x)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = F.relu(x)
            x = self.fc3(x) 
            x = x.unsqueeze(0)
            x = F.log_softmax(x, dim=1)
            return x

    # -- create instance of model, create loss and optimization functions 
    model = ConvNet().to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer=optimizer, factor=1e-7)

    ##########################################################################################
    # MODEL PLOTTING DEFINITIONS
    ##########################################################################################

    # -- functions for plotting the loss, accuracy, CM, and ROC curve
    f, a = plt.subplots(2, 2, figsize=(10,7.5), layout='constrained')

    # -- set internal plots
    def plot_all(num_epochs, training_loss, vt_loss, training_accuracy, vt_accuracy, tpr, fpr, auc, df_cm, o_dir):
 
        # -- set title for cell type  
        f.suptitle(cell_type)

        # -- set title for testing or validation loss 
        val_or_test = ''

        if not v_ds: 
            val_or_test = 'Testing'
        else:
            val_or_test = 'Validation'
        
        # -- plot loss 
        a[0,0].plot(num_epochs, training_loss, label='Training Loss')
        a[0,0].plot(num_epochs, vt_loss, label=f'{val_or_test} Loss')
        a[0,0].set_xlabel('Number of Epochs')
        a[0,0].set_ylabel('Average Loss')
        a[0,0].set_title(f'Training and {val_or_test} Loss')
        a[0,0].legend()

        # -- plot accuracy
        a[0,1].plot(num_epochs, training_accuracy, label='Training Accuracy')
        a[0,1].plot(num_epochs, vt_accuracy, label=f'{val_or_test} Accuracy')
        a[0,1].set_xlabel('Number of Epochs')
        a[0,1].set_ylabel('Accuracy')
        a[0,1].set_title(f'Training and {val_or_test} Accuracy')
        a[0,1].legend()

        # -- plot roc
        a[1,0].plot(fpr, tpr, label=f'ROC (AUC = {round(auc, 3)})')
        a[1,0].axline((0, 0), slope=1, label='Random (AUC = 0.5)', color='orange')
        a[1,0].set_xlabel('FPR')
        a[1,0].set_ylabel('TPR')
        a[1,0].set_title('ROC Curve')
        a[1,0].legend()

        # -- plot confusion matrix
        a[1,1] = sn.heatmap(df_cm, annot=True)
        a[1,1].set_title('Confusion Matrix')

        # -- save the consolidated file into a sub directory if specified as such 
        if o_dir != 'DELETE':
            plt.savefig(f'{output_dir}/{cell_type}.png')

    ##########################################################################################
    # MODEL TEST LOOP
    ##########################################################################################

    def test_model(log):

        # define testing params: ROC, confusion matrix, evaluation time 
        target_scores = []
        cm_scores = []
        predicted_scores = []
        evaluation_time_list = []
        n_samples = num_genes

        model.load_state_dict(torch.load(f'{model_path}/{c_type}_params.pth'))

        for epoch in range(num_epochs):
            
            # switch the model to evaluation mode for specific layers -- dropout, etc 
            model.eval() 

            # for accuracy measurements
            vt_n_correct = 0

            # set the proper dataloader 
            if not v_ds: 
                vt_dataloader = test_dataloader
            else:
                vt_dataloader = valid_dataloader

            # run the validation or testing process
            with torch.no_grad():
                # log evaluation loss
                vt_loss_per_batch = []

                # log start time for evaluation
                e_start = time.time()
                for i, (samples, labels) in enumerate(vt_dataloader):
                    
                    # -- set samples
                    samples = samples.permute(1, 0)
                    samples = samples.to(device) 
                
                    # -- set labels 
                    labels = torch.tensor([labels[0]]).long()

                    labels = labels.to(device)
                    
                    # -- send model to the device
                    model.to(device)
                    predicted = model(samples)

                    # -- calculate rough approx accuracy 
                    if torch.argmax(predicted) == 0 and labels.item()==0:
                        vt_n_correct+=1
                    if torch.argmax(predicted) == 1 and labels.item()==1:
                        vt_n_correct+=1
                    
                    # # -- y is for ROC and confusion matrix
                    target_scores.append(labels.item())
                    # max_index = torch.argmax(predicted).item()
                    predicted_scores.append(torch.exp(predicted[0][1]).item())
                    
                    # -- cm_scores is for confusion matrix
                    if (torch.argmax(predicted)==0 and labels.item()==0) or (torch.argmax(predicted)==1 and labels.item()==1):
                        cm_scores.append(1)
                    else:
                        cm_scores.append(0)
                    
                    # -- calculate the loss
                    loss = criterion(predicted, labels) 
                    vt_loss_per_batch.append(loss.item()) # append the loss for all (10) batches

                    if (i+1) % n_batches == 0: # -- this is where the i term above is used in for loop
                        print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
                
                # -- calculate average validation loss for each epoch 
                vt_loss = sum(vt_loss_per_batch) / len(vt_loss_per_batch)
                log['vt_loss_per_epoch'].append(vt_loss)
                # print('Validation loss', vt_loss)
                
                # -- calculate validation accuracy -- rough approx -- for each epoch 
                vt_accuracy = vt_n_correct / n_samples
                log['vt_accuracy_per_epoch'].append(vt_accuracy)
                # print('Validation accuracy: ', vt_accuracy)
                
                # -- calculate end time and elapsed time for evaluation, appending to list 
                e_end = time.time()
                e_elapsed = e_end - e_start
                evaluation_time_list.append(e_elapsed)

        # -- create necessary parameters for plotting 
        target_scores = np.asarray(target_scores, dtype=np.int32)
        predicted_scores = np.asarray(predicted_scores, dtype=np.float32)
        # print(predicted_scores)
        fpr, tpr, thresholds_ignore = metrics.roc_curve(target_scores, predicted_scores, pos_label=1)
        auc = metrics.roc_auc_score(target_scores, predicted_scores)
        cf_matrix = np.array(metrics.confusion_matrix(target_scores, cm_scores))

        # -- have to swap around the cells -- was wrong! 
        temp = cf_matrix[0][1]
        cf_matrix[0][1] = cf_matrix[0][0]
        cf_matrix[0][0] = temp

        # -- set classes 
        classes = {0, 1}

        # -- convert the numpy array for the confusion matrix -> pandas dataframe 
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
        
        return fpr, tpr, auc, df_cm, evaluation_time_list

    ##########################################################################################
    # MODEL TRAIN LOOP
    ##########################################################################################

    def train_model():

        # -- create training time list
        training_time_list = []
        
        # -- set log for necessary values
        log = {
            'training_loss_per_epoch':[], 
            'vt_loss_per_epoch':[], 
            'training_accuracy_per_epoch':[], 
            'vt_accuracy_per_epoch':[]
        }

        # -- run the model over several (specified number of) epochs 
        for epoch in range(num_epochs):

            # -- set the model to training mode -- resets it after switching to eval mode later (for validation)
            model.train()

            # -- vars for accuracy measurements 
            t_n_correct = 0
            n_samples = num_genes
            
            # -- create list for keeping track of training loss per batch
            t_loss_per_batch = []

            # -- initialize time for training
            t_start = time.time()
            for i, (samples, labels) in enumerate(training_dataloader):
                
                # set the samples, send to the device
                samples = samples.permute(1, 0)
                samples = samples.to(device) 

                # set the labels, send to the device 
                labels = torch.tensor([labels[0]]).long()
                labels = labels.to(device)

                # -- send the model to the device
                model.to(device)
                predicted = model(samples)
                
                loss = criterion(predicted, labels) 

                if torch.argmax(predicted)==0 and labels.item()==0:
                    t_n_correct+=1
                if torch.argmax(predicted)==1 and labels.item()==1:
                    t_n_correct+=1

                # -- update and append the loss
                t_loss_per_batch.append(loss.item()) 

                # --empty out the gradients, update the loss, step 
                optimizer.zero_grad()
                loss.backward() 
                optimizer.step() 

                if (i+1) % n_batches == 0: # -- this is where the i term above is used in for loop
                    print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            
            # -- step with the scheduler, update the lr 
            scheduler.step()

            # -- calculate average training loss for every epoch
            t_loss = sum(t_loss_per_batch) / len(t_loss_per_batch)
            log['training_loss_per_epoch'].append(t_loss)

            # -- calculate training accuracy -- rough approx -- with just givens 
            t_accuracy = t_n_correct / n_samples
            log['training_accuracy_per_epoch'].append(t_accuracy)

            # -- calculate end time and elapsed time for training, appending to list
            t_end = time.time()
            t_elapsed = t_end - t_start
            training_time_list.append(t_elapsed)


        # -- save the model if specified to do so (i was stupid, was saving at every epoch)\
        if model_path != '': 
            PATH = f'{model_path}/{cell_type}_params.pth'
        torch.save(model.state_dict(), PATH) # -- save the model's params 

        # -- return log for the testing procedure 
        return log, training_time_list
    
    ##########################################################################################
    # MODEL EVALUATION WRAPPER: runs train, test functions then plots
    ##########################################################################################
    
    def evaluate():
        log, training_time_list = train_model()
        fpr, tpr, auc, df_cm, evaluation_time_list = test_model(log)

        # --plot loss, accuracy, ROC curve, and create confusion matrix
        plot_all(num_epochs=np.linspace(1, num_epochs, num=num_epochs).astype(int), training_loss=log['training_loss_per_epoch'], 
                vt_loss=log['vt_loss_per_epoch'], training_accuracy=log['training_accuracy_per_epoch'], 
                vt_accuracy=log['vt_accuracy_per_epoch'], tpr=tpr, fpr=fpr, auc=auc, df_cm=df_cm, o_dir=output_dir)

        # -- print the average time elapsed to train the model
        total_training_time = round(sum(training_time_list) / 60, 2)
        print(f'total time for training is: {total_training_time} seconds')
        total_evaluation_time = round(sum(evaluation_time_list) / 60, 2)
        print(f'total time for evaluation is: {total_evaluation_time} seconds')
        print(f'total time for training and evaluating the model is: {round(total_training_time + total_evaluation_time, 2)} seconds')

        return auc 
    
    # -- run the full evaluate function
    final_auc = evaluate()

    # -- return the final_auc value as the value that it was set equal to from the evaluate function 
    return final_auc

def plot_final_aucs(auc_df, o_dir):
    fig, ax = plt.subplots(figsize=(8,5))
    fig.suptitle('AUC Scores For All Cell Types')
    ax.bar('cells', 'auc', data=auc_df)
    ax.set_xlabel('Cell Types')
    ax.set_ylabel('AUC Score')
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(np.arange(0,1.1,0.1), fontsize=9)

    # -- save the consolidated file into a sub directory if specified as such 
    if o_dir != 'DELETE':
        plt.savefig(f'{o_dir}/final_aucs.png')

def main():

   # -- argparser: what args to look for? 
        # -- ** note that -h can be used to reference what these flags are / their meaning
        # -- 1. whether or not to save results -> where to save results (directory folder) 
        # -- 2. which datasets to use -> training, then testing or validation 
        # -- 3. hyperparameters to use 
        # -- 4. ** to add later -- which model / n units to use 

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
    auc_list = []
    for root, dirs, files_ignore in os.walk(data_directory): 
        if not dirs: 
            lowest_dirs.append(root)
    all_cell_types = next(os.walk(data_directory))[1] # -- this obtains the cell types

    # -- iterate now through the directories and run the model on their respective data 
    marker=0
    for dirs in lowest_dirs: 
        if args['tv'][0] == 'valid': # -- check if valid over test dataset 
            auc = run_all(tr_ds=f'{dirs}/train.csv', v_ds=f'{dirs}/valid.csv', te_ds=None, c_type=all_cell_types[marker], o_dir=output_directory, n_epochs=args['e'][0], model_path=model_path)
        elif args['tv'][0] == 'test': # -- check if test over valid dataset 
            auc = run_all(tr_ds=f'{dirs}/train.csv', te_ds=f'{dirs}/valid.csv',  v_ds=None, c_type=all_cell_types[marker], o_dir=output_directory, n_epochs=args['e'][0], model_path=model_path)
        auc_list.append(round(auc, 3))
        marker+=1
    print('max of auc list is: ', max(auc_list))
    print('min of auc list is: ', min(auc_list))
    print('mean of auc list is: ', round(sum(auc_list) / len(auc_list), 2))

    # -- add aucs to a df that is sorted 
    auc_df = pd.DataFrame(dict(cells = all_cell_types, auc = auc_list))
    auc_sorted_df = auc_df.sort_values('auc', ascending=False)
    plot_final_aucs(auc_df=auc_sorted_df, o_dir=output_directory)

if __name__ == '__main__':
    main()


# old code: 
# -- randomly initialize the parameters? actually works better if i don't for other models 
# def init_weights(m):
#     if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
#         torch.nn.init.xavier_uniform_(m.weight)
#         m.bias.data.fill_(0.01)
# model.apply(init_weights)