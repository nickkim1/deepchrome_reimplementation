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

# -- parse in args as variables  
def parse_in(tr_ds, vt_ds, test_or_valid, o_dir, c_type, n_epochs, hp, mp): 
    # -- set the right parameters
    # global training_ds, valid_training_ds, output_dir, cell_type, hp_specs, model_path, n_e 
    training_ds, valid_testing_ds, test_or_valid, output_dir, cell_type, hp_specs, model_path, n_e = tr_ds, vt_ds, test_or_valid, o_dir, c_type, hp, mp, n_epochs
    return training_ds, valid_testing_ds, test_or_valid, output_dir, cell_type, hp_specs, model_path, n_e

def set_device():
    # global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("===================================")
    print(f"0. Using {device} device!")

def check_dimensions(training_ds):
    # global num_rows, num_cols, num_zeros, num_ones, marker, pos_weights, num_windows, num_genes
    num_rows = 0 
    num_cols = 0 
    num_zeros = 0 
    num_ones = 0 
    marker = 0 
    pos_weights = 0
    num_windows = 100 

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
    num_genes = int(num_rows / num_windows)
    pos_weights = num_zeros / num_ones
    
    print("===================================")
    print("INITIAL DIMENSIONS CHECK")
    print("1. The number of features/histone modification types is: ", num_cols-3)
    print("2. The number of genes is: ", num_genes)
    print("3. The number of windows and number of samples is: ", num_windows)
    print(f"4. The number of ones is {num_ones} and the number of zeros is {num_zeros}")
    print(f"5. The ratio of negative : positive examples is {pos_weights}")
    print("===================================")

    return num_rows, num_cols, num_zeros, num_ones, marker, pos_weights, num_windows, num_genes

# -- custom dataset class
class HistoneDataset(Dataset):
    def __init__(self, file_path, num_cols, num_windows): 
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
                self.f[int(line[0].item())] = data[counter]
            counter+=1
        self.labels = torch.tensor(xy[:, num_cols-1]) # 1000 x 1
        self.gene_ids = torch.from_numpy(xy[:, 0]) # should be 1000 x 1
        self.indices = {}
        classes = []
        for idx, gene_id in enumerate(self.gene_ids):
            if idx % num_windows == 0:
                classes.append(int(gene_id.item()))
                self.indices[int(gene_id.item())] = np.arange(idx-100, idx, 1)
        self.classes = classes
        self.n_samples = len(xy) 
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    def gene_ids_and_indices(self):
        return self.classes, self.indices
    
# # -- custom batchsampler class -- this creates (mini-batches of) indices and passes to the dataset to fetch corresponding samples 
class BSampler(BatchSampler):
    def __init__(self, training_dataset, gene_ids, indices, batch_size):
        super(BSampler, self).__init__(training_dataset, batch_size, drop_last=False)
        self.gene_ids = gene_ids 
        self.indices = indices 
        self.n_batches = int(len(training_dataset) / batch_size) 
        self.batch_size = batch_size
    def __iter__(self):
        batches = []
        for _ in range(self.n_batches):
            batch = []
            batch_class = random.choice(self.gene_ids) 
            vals = torch.from_numpy(self.indices[batch_class])
            for ignore, idx in enumerate(vals):
                batch.append(idx)
            batches.append(batch) # should be selecting from a dictionary
        return iter(batches)
    def __len__(self):
        return self.n_batches

def create_datasets_dataloaders(training_ds, vt_ds, num_cols, num_windows): 
    # -- create the datasets -- histone dataset
    # global training_dataset, training_dataloader, test_dataset, test_dataloader, valid_dataset, valid_dataloader, n_total_steps
    training_dataset = HistoneDataset(file_path=training_ds, num_cols=num_cols, num_windows=num_windows)
    training_dataloader = DataLoader(dataset=training_dataset, batch_sampler=BSampler(training_dataset=training_dataset, gene_ids=training_dataset.gene_ids_and_indices()[0], indices=training_dataset.gene_ids_and_indices()[1], batch_size=100))
    vt_dataset = HistoneDataset(file_path=vt_ds, num_cols=num_cols, num_windows=num_windows)
    vt_dataloader = DataLoader(dataset=vt_dataset, batch_size=100, shuffle=False, drop_last=False)
    n_total_steps = len(training_dataloader)
    return training_dataset, training_dataloader, vt_dataset, vt_dataloader, n_total_steps

def set_hyperparams(num_windows, n_e, hp_specs):
    # -- set as global variables 
    # global width, hidden_layer_units, num_filters, filter_size, pool_size, num_epochs, learning_rate, batch_size
    
    # -- spec values 
    width = num_windows
    learning_rate = 0.001
    num_epochs = n_e
    batch_size = 100
    num_features = 5
    num_filters = 50
    hidden_layer_units = {"first": 625, "second": 125}
    num_inputs = num_features * width # should just be the size of each gene sample
    num_outputs = 2 
    stride_length = 1

    if hp_specs == 4: 
        filter_size = 10 
        pool_size = 5 
    elif hp_specs == 1: 
        filter_size = 5
        pool_size = 2
    elif hp_specs == 2: 
        filter_size = 5
        pool_size = 5
    elif hp_specs == 3:
        filter_size = 10 
        pool_size = 2

    return width, hidden_layer_units, num_filters, filter_size, pool_size, num_epochs, learning_rate, batch_size

# - model class 
# -- 1d cnn -> mlp 
class ConvNet(nn.Module):
    def __init__(self, width, num_filters, filter_size, pool_size, hidden_layer_units:dict): 
        super(ConvNet, self).__init__()
        self.conv = nn.Conv1d(in_channels=5, out_channels=num_filters, kernel_size=filter_size)
        self.pool = nn.MaxPool1d(kernel_size=pool_size)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(math.ceil((width-filter_size)/pool_size)*num_filters, hidden_layer_units["first"])
        self.fc2 = nn.Linear(hidden_layer_units["first"], hidden_layer_units["second"])
        self.fc3 = nn.Linear(hidden_layer_units["second"], 1)
        self.width = width
        self.filter_size = filter_size
        self.pool_size = pool_size 
        self.num_filters = num_filters
    def forward(self, x): 
        # print('very first input: ', x.unsqueeze(0).shape)
        x = self.conv(x)
        # print('first conv layer: ', x.shape)
        x = F.relu(x)
        # print('first relu: ', x.shape)
        x = self.pool(x)
        # print('max pooling: ', x.shape)
        x = x.view(math.ceil((self.width-self.filter_size)/self.pool_size)*self.num_filters)
        # print('flattened: ', x.unsqueeze(0).shape)
        x = self.dropout(x)
        # print('dropout: ', x.unsqueeze(0).shape)
        x = self.fc1(x)
        x = F.relu(x)
        # print('second relu + first linear: ', x.unsqueeze(0).shape)
        x = self.fc2(x)
        x = F.relu(x)
        # print('third relu + second linear: ', x.unsqueeze(0).shape)
        x = self.fc3(x) 
        # print('last linear: ', x.unsqueeze(0).shape)
        return x

def model_loss_optim(width, num_filters, filter_size, pool_size, hidden_layer_units:dict, device, pos_weights, learning_rate):
    # global model, criterion, optimizer, scheduler
    model = ConvNet(width=width, num_filters=num_filters, filter_size=filter_size, pool_size=pool_size, hidden_layer_units=hidden_layer_units).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weights))
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer=optimizer, factor=1e-7)
    return model, criterion, optimizer, scheduler

# -- plot literally everything
def plot_all(num_epochs, cell_type, training_loss, 
             training_accuracy, test_or_valid, vt_loss, 
             vt_accuracy, tpr, fpr, auc, df_cm, o_dir):

    f, a = plt.subplots(2, 2, figsize=(10,7.5), layout='constrained')       
    f.suptitle(cell_type)

    if not test_or_valid == 'test': # -- need to fix this section 
        val_or_test = 'Testing'
    else:
        val_or_test = 'Validation'

    a[0,0].plot(num_epochs, training_loss, label='Training Loss')
    a[0,0].plot(num_epochs, vt_loss, label=f'{val_or_test} Loss')
    a[0,0].set_xlabel('Number of Epochs')
    a[0,0].set_ylabel('Average Loss')
    a[0,0].set_title(f'Training and {val_or_test} Loss')
    a[0,0].legend()

    a[0,1].plot(num_epochs, training_accuracy, label='Training Accuracy')
    a[0,1].plot(num_epochs, vt_accuracy, label=f'{val_or_test} Loss')
    a[0,1].set_xlabel('Number of Epochs')
    a[0,1].set_ylabel('Accuracy')
    a[0,1].set_title(f'Training and {val_or_test} Accuracy')
    a[0,1].legend()

    a[1,0].plot(fpr, tpr, label=f'ROC (AUC = {round(auc, 3)})')
    a[1,0].axline((0, 0), slope=1, label='Random (AUC = 0.5)', color='orange')
    a[1,0].set_xlabel('FPR')
    a[1,0].set_ylabel('TPR')
    a[1,0].set_title('ROC Curve')
    a[1,0].legend()

    a[1,1] = sn.heatmap(df_cm, annot=True)
    a[1,1].set_title('Confusion Matrix')
    # -- save the consolidated file into a sub directory if specified as such 
    if o_dir != 'DELETE':
        plt.savefig(f'{o_dir}/{cell_type}.png')

def train_model(model, cell_type, num_epochs, num_genes, training_dataloader, device, criterion, optimizer, n_total_steps, scheduler, model_path):

    training_time_list = []
    log = {'training_loss_per_epoch':[], 'vt_loss_per_epoch':[], 'training_accuracy_per_epoch':[], 'vt_accuracy_per_epoch':[]}

    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    model.apply(init_weights)

    for epoch in range(num_epochs):
        model.train()
        t_n_correct = 0
        n_samples = num_genes
        t_loss_per_batch = []
        t_start = time.time()
        for i, (samples, labels) in enumerate(training_dataloader):
             
            samples = samples.permute(1, 0)
            samples = samples.to(device)

            labels = torch.tensor([labels[0]]).float()
            labels = labels.to(device)
 
            model.to(device)
            predicted = model(samples)
           
            if predicted[0] < 0 and labels.item()==0:
                t_n_correct+=1
            if predicted[0] > 0 and labels.item()==1:
                t_n_correct+=1

            loss = criterion(predicted, labels) 
            t_loss_per_batch.append(loss.item())

            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 
            if (i+1) % num_genes == 0: 
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

        scheduler.step()

        t_loss = sum(t_loss_per_batch) / len(t_loss_per_batch)
        log['training_loss_per_epoch'].append(t_loss)
        t_accuracy = t_n_correct / n_samples
        log['training_accuracy_per_epoch'].append(t_accuracy)
        t_end = time.time()
        t_elapsed = t_end - t_start
        training_time_list.append(t_elapsed)
    
    # -- save the model 
    if model_path != '': 
        PATH = f'{model_path}/{cell_type}_params.pth'
        torch.save(model.state_dict(), PATH) # -- save the model's params 
    return training_time_list, log, PATH


def test_model(m, saved_model_path, vt_dataloader, device, criterion, num_genes, num_epochs, log): 
    target_scores = []
    cm_scores = []
    predicted_scores = []
    evaluation_time_list = []

    model = m # -- load the structure of the model 
    model.load_state_dict(torch.load(saved_model_path))
   
    with torch.no_grad():

        for epoch in range(num_epochs):   
            model.eval() # -- switch modes for layers that act diff

            vt_n_correct = 0
            vt_loss_per_batch = []
            e_start = time.time()

            for i, (samples, labels) in enumerate(vt_dataloader):
                samples = samples.permute(1, 0)
                samples = samples.to(device) 
                labels = torch.tensor([labels[0]]).float()
                labels = labels.to(device)

                model.to(device)
                predicted = model(samples)

                if predicted[0] < 0 and labels.item()==0:
                    vt_n_correct+=1
                if predicted[0] > 0 and labels.item()==1:
                    vt_n_correct+=1

                target_scores.append(labels.item())
                predicted_scores.append(torch.sigmoid(predicted).item())
                
                if (predicted[0] > 0 and labels.item()==1) or (predicted[0] < 0 and labels.item()==0): 
                    cm_scores.append(1)
                else:
                    cm_scores.append(0)

                loss = criterion(predicted, labels) 
                vt_loss_per_batch.append(loss.item()) 
       
            vt_loss = sum(vt_loss_per_batch) / len(vt_loss_per_batch)
            log['vt_loss_per_epoch'].append(vt_loss)

            vt_accuracy = vt_n_correct / num_genes
            log['vt_accuracy_per_epoch'].append(vt_accuracy)

            # print(vt_accuracy)
       
    e_end = time.time()
    e_elapsed = e_end - e_start
    evaluation_time_list.append(e_elapsed)

    return target_scores, predicted_scores, cm_scores, evaluation_time_list, log


def final_out(target_s, predicted_s, cm_scores, num_epochs, log, test_or_valid, cell_type, o_dir): 

    # -- set necessary globals for downstream plotting 
    # global fpr, tpr, auc, df_cm
    target_scores = np.asarray(target_s, dtype=np.int32)
    predicted_scores = np.asarray(predicted_s, dtype=np.float32)
    fpr, tpr, thresholds = metrics.roc_curve(target_scores, predicted_scores, pos_label=1)
    auc = metrics.roc_auc_score(target_scores, predicted_scores)
    cf_matrix = np.array(metrics.confusion_matrix(target_scores, cm_scores))

    temp = cf_matrix[0][1]
    cf_matrix[0][1] = cf_matrix[0][0]
    cf_matrix[0][0] = temp
    classes = {0, 1}
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes], columns = [i for i in classes])

    plot_all(num_epochs=np.linspace(1, num_epochs, num=num_epochs).astype(int),
            test_or_valid=test_or_valid, 
            cell_type=cell_type,
            training_loss=log['training_loss_per_epoch'], 
            vt_loss=log['vt_loss_per_epoch'], 
            training_accuracy=log['training_accuracy_per_epoch'], 
            vt_accuracy=log['vt_accuracy_per_epoch'], 
            tpr=tpr, fpr=fpr, auc=auc, df_cm=df_cm, o_dir=o_dir)

    # -- plot everything then just return the auc 
    return auc

def time_out(training_time_list, evaluation_time_list):
    total_training_time = round(sum(training_time_list) / 60, 2)
    print(f'total time for training is: {total_training_time} minutes')
    total_evaluation_time = round(sum(evaluation_time_list) / 60, 2)
    print(f'total time for evaluation is: {total_evaluation_time} minutes')
    print(f'total time for training and evaluating the model is: {round(total_training_time + total_evaluation_time, 2)} minutes')


# -- function for plotting the final list of aucs for each cell type 
def plot_final_aucs(auc_df, o_dir):
    fig, ax = plt.subplots(figsize=(8,5))
    fig.suptitle('AUC Scores For All Cell Types')
    ax.bar('cells', 'auc', data=auc_df)
    ax.set_xlabel('Cell Types')
    ax.set_ylabel('AUC Score')
    plt.xticks(rotation=90, fontsize=7)
    # -- save the consolidated file into a sub directory if specified as such 
    if o_dir != 'DELETE':
        plt.savefig(f'{o_dir}/final_aucs.png')

def main():
    parser = ap.ArgumentParser(prog='Reimplementation of Deepchrome (Pytorch)!', 
                               description='[1] All data files must be titled train.csv, test.csv, or valid.csv. [2] All cell types should have respective directory named after them. [3] Folders for saving model, results, and storing data should be in same folder as where this script is ')
    parser.add_argument('-d', type=str, help='-> input name of folder to use for processing in data', nargs=1)
    parser.add_argument('-tv', type=str, help='-> input test or valid', nargs=1)
    parser.add_argument('-s', type=str, help='-> input Y or N for saving results or not', nargs=1)
    parser.add_argument('-o', type=str, help='-> input name of folder to save results to', nargs=1)
    parser.add_argument('-e', type=int, help='-> input number of epochs you want to run the model for', nargs=1)
    parser.add_argument('-hp', type=int, help='-> input 1 for (5,2) 2 for (5,5) 3 for (10,2) 4 for (10,5) to specify hp',default=4)
    parser.add_argument('-sm', type=str, help='-> input name of folder to save model to when finished',default='')
    args = vars(parser.parse_args()) # -- .parse_args() must be called for help message to be shown

    # -- set up workflow proper: 
    data_directory = args['d'][0]
    if args['s'][0] == 'Y': 
        output_directory = args['o'][0]
    else:
         output_directory ='DELETE'
    test_or_valid = args['tv'][0]
    num_epochs = args['e'][0]
    hyperparams = args['hp']
    model_path = args['sm']

    # -- recur down to the lowest level subdirectory, get the right filepath
    lowest_dirs = []
    auc_list = []
    for root, dirs, files_ignore in os.walk(data_directory): 
        if not dirs: 
            lowest_dirs.append(root)
    all_cell_types = next(os.walk(data_directory))[1]

    # -- iterate through appended filepaths and run model => workflow 
    for index, dirs in enumerate(lowest_dirs): 
        training_ds, valid_training_ds, test_or_valid, output_dir, cell_type, hp_specs, model_path, n_e = parse_in(tr_ds=f'{dirs}/train.csv', vt_ds=f'{dirs}/{test_or_valid}.csv', test_or_valid=test_or_valid, c_type=all_cell_types[index], o_dir=output_directory,n_epochs=num_epochs,hp=hyperparams,mp=model_path)
        device = set_device()
        # -- 1. check dimensions 
        num_rows, num_cols, num_zeros, num_ones, i, pos_weights, num_windows, num_genes = check_dimensions(training_ds=training_ds)
        # -- 2. create datasets and dataloaders 
        training_dataset, training_dataloader, vt_dataset, vt_dataloader, n_total_steps = create_datasets_dataloaders(training_ds=training_ds, vt_ds=valid_training_ds, num_cols=num_cols, num_windows=num_windows)
        # -- 3. set the hyperparameters
        width, hidden_layer_units, num_filters, filter_size, pool_size, num_epochs, learning_rate, batch_size = set_hyperparams(num_windows=num_windows, n_e=num_epochs, hp_specs=hp_specs)
        # -- 4. create model, loss criterion, optimizer, and lr scheduler 
        model, criterion, optimizer, scheduler = model_loss_optim(width=width, num_filters=num_filters, filter_size=filter_size, pool_size=pool_size, hidden_layer_units=hidden_layer_units, device=device, pos_weights=pos_weights, learning_rate=learning_rate)
        # -- 5. train and save the model 
        training_time_list, log, saved_model_path = train_model(model=model, cell_type=all_cell_types[index], num_epochs=num_epochs, num_genes=num_genes, training_dataloader=training_dataloader, device=device, criterion=criterion, optimizer=optimizer, n_total_steps=n_total_steps, scheduler=scheduler, model_path=model_path)
        # -- 6. load and test the model 
        target_scores, predicted_scores, cm_scores, evaluation_time_list, log = test_model(m=model, saved_model_path=saved_model_path,vt_dataloader=vt_dataloader, device=device, criterion=criterion, num_genes=num_genes, num_epochs=num_epochs, log=log)
        auc = final_out(target_s=target_scores, predicted_s=predicted_scores, cell_type=all_cell_types[index], cm_scores=cm_scores, num_epochs=num_epochs, log=log, test_or_valid=test_or_valid, o_dir=output_directory)
        # -- 7. print final time elapsed model 
        time_out(training_time_list=training_time_list, evaluation_time_list=evaluation_time_list)
        # -- 8. add to final auc graph, update the marker for next iteration 
        auc_list.append(auc) 
    # -- 9. print out max, min, mean auc and plot the final auc graph 
    print('max of auc list is: ', max(auc_list))
    print('min of auc list is: ', min(auc_list))
    print('mean of auc list is: ', round(sum(auc_list) / len(auc_list), 2))
    auc_df = pd.DataFrame(dict(cells = all_cell_types, auc = auc_list))
    auc_sorted_df = auc_df.sort_values('auc', ascending=False)
    plot_final_aucs(auc_df=auc_sorted_df, o_dir=output_directory)

if __name__ == '__main__':
    main()