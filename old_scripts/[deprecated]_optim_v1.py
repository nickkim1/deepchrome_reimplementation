import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import argparse as ap
import time as time
import random
import math
import scipy 
import os

def load_data(ct, ds):
    xy = torch.from_numpy(np.loadtxt(f'{ds}/{ct}/train.csv', delimiter=",",dtype=np.float32))
    num_cols = len(xy[0])
    num_zeros = 0
    num_ones = 0
    
    for i in range(len(xy)):
        if xy[i][num_cols-1]==0:
            num_zeros += 1
        if xy[i][num_cols-1]==1:
            num_ones += 1

    pos_weight = torch.tensor([num_zeros / num_ones])
    return pos_weight


def set_device():
     # -- set the device used
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"0. Using {device} device!")
    return device

# -- define convnet class
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
        x = self.conv(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(math.ceil((self.width-self.filter_size)/self.pool_size)*self.num_filters)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x) 
        return x
    

def opt(X, model_folder, ct, exp_val):
    
    # params
    lambda_val = 0.009
    criterion = nn.BCEWithLogitsLoss()
    model = ConvNet(width=100, 
                    num_filters=50, 
                    filter_size=10, 
                    pool_size=5, 
                    hidden_layer_units={"first": 625, "second": 125})
    
    model.load_state_dict(torch.load(f'{model_folder}/{ct}_params.pth'))
    model.eval()

    # manually removed the dropout layer
    
    predicted = model(X)
    loss = criterion(predicted, torch.FloatTensor([exp_val]))
    input_grads = torch.autograd.grad(loss, X, retain_graph=True)[0]

    loss += lambda_val * (X.norm() ** 2)
    input_grads = input_grads + (X * 2 * lambda_val)

    return loss, input_grads
    
# do the SGD loop 
def loop(model_folder, ct, exp_val, n_rep_interactions, n_prom_interactions): 
    torch.manual_seed(5000)
    X = torch.rand(5, 100)

    # print('initial X: ', X)
    loss_list = []
    for i in range(1000):
        X.requires_grad_(True)
        loss, _ = opt(X, model_folder, ct, exp_val)
        loss_list.append(loss.item())
        optimizer = torch.optim.SGD({X}, lr=0.1, momentum=0.9)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0: 
            print(i, ': ', loss.item())

    X = torch.clamp(X, 0, 1)

    X = X.permute(1,0) # transpose
    max_val = X.max()
    for i in range(100):
        sum_of_row = X[i].sum()
        if sum_of_row == 0:
            X[i] = torch.zeros(5)
        else:
            X[i] = X[i] / max_val

    mods_to_frequencies = {}
    mods_to_frequencies['H3K27me3 (R)'] = 0
    mods_to_frequencies['H3K36me3 (P)'] = 0
    mods_to_frequencies['H3K4me1 (DP)'] = 0
    mods_to_frequencies['H3K4me3 (P)'] = 0
    mods_to_frequencies['H3K9me3 (R)'] = 0

    X = X.detach().numpy()
    color_list = []
    for i in range(100): # it is now 100 x 5
        for mod, bin_val in enumerate(X[i]):
            if mod == 0 and bin_val > 0.25:
                mods_to_frequencies['H3K27me3 (R)'] += 1
            elif mod == 1 and bin_val > 0.25: 
                mods_to_frequencies['H3K36me3 (P)'] += 1
            elif mod == 2 and bin_val > 0.25:
                mods_to_frequencies['H3K4me1 (DP)'] += 1
            elif mod == 3 and bin_val > 0.25: 
                mods_to_frequencies['H3K4me3 (P)'] += 1
            elif mod == 4 and bin_val > 0.25: 
                mods_to_frequencies['H3K9me3 (R)'] += 1

    print('vals: ', mods_to_frequencies.values())
    mean_frequency = sum(mods_to_frequencies.values()) / len(mods_to_frequencies.values())

    print('mean frequency:', mean_frequency)
    print('frequencies', mods_to_frequencies)
    
    histone_mods = ['H3K27me3 (R)', 'H3K36me3 (P)','H3K4me1 (DP)', 'H3K4me3 (P)', 'H3K9me3 (R)']

    for mods in histone_mods:
        if mods_to_frequencies[mods] > mean_frequency:
            color_list.append("black")
        if mods_to_frequencies[mods] <= mean_frequency:
            color_list.append("gray")
        if (mods == 'H3K9me3 (R)' or mods == 'H3K27me3 (R)') and (mods_to_frequencies[mods] > mean_frequency) and exp_val == 0:
            n_rep_interactions.append(1)
        if (mods != 'H3K9me3 (R)' and mods != 'H3K27me3 (R)') and (mods_to_frequencies[mods] > mean_frequency) and exp_val == 1:
            n_prom_interactions.append(1)

    return X, color_list, mods_to_frequencies, loss_list


def plot_heatmap(cell_type, all_normalized_arrs, 
                 mods_to_frequencies, color_list, 
                 histone_mods, bins, expression_val, o_dir):

    # -- plot the graph for heatmap + barplot 
    f, a = plt.subplots(2, 1, figsize=(15,6) ,layout='constrained')

    # -- set custom titles based on expression value for the given sample 
    a[0].set_title(f'Bar Graph of Active Bins per Modification for {cell_type}', fontsize=10)
    a[0].bar(histone_mods, mods_to_frequencies, color=color_list, width=0.2)

    # -- set custom titles based on expression value for the given sample 
    if expression_val == 1:
        a[1].set_title(f'Heatmap of Bins for {cell_type}. Gene Expression = High', fontsize=10)
    elif expression_val == 0:
        a[1].set_title(f'Heatmap of Bins for {cell_type}. Gene Expression = Low', fontsize=10)
    
    # -- create custom colorbar (not discretized)
    img = a[1].imshow(all_normalized_arrs)
    cbar = a[1].figure.colorbar(img, ax=a[1], shrink=0.2, pad=0.01)
    cbar.ax.set_ylabel('', rotation=-90, va="bottom")

    # -- show all the ticks and label them with respective entries (if specified)
    a[1].set_yticks(np.arange(0, 5, 1)-0.5, labels=histone_mods)
    a[1].set_xticks(np.arange(-1.5, 99, 1)+1)
    a[1].tick_params(axis='y', which='major', labelsize=6)
    a[1].tick_params(axis='x', which='major', labelsize=2)

    # -- create grid to distinguish between cells
    a[1].grid(color='black', visible=True, which='both',linestyle='-',linewidth=0.3)

    # -- save the file to specified output folder
    if o_dir != 'DELETE':
        plt.savefig(f'{o_dir}/{cell_type}_{int(expression_val)}_heatmap.png')
    # plt.show()

def plot_loss(cell_type, num_epochs, loss, expression_val, o_dir):
    f, a = plt.subplots(layout='constrained') 
    f.suptitle(f'Evaluation Loss for {cell_type}') 
    a.plot(num_epochs, loss, label='Loss')
    a.set_xlabel('Number of Epochs')
    a.set_ylabel('Loss')
    a.legend()  
    # plt.show()
    if o_dir != 'DELETE':
        plt.savefig(f'{o_dir}/{cell_type}_{int(expression_val)}_loss.png')

# run all 
parser = ap.ArgumentParser(prog='Visualization of Deepchrome!', description='Please specify the cell-types to visualize below.')
parser.add_argument('-d', type=str, help='-> input name of dataset to eval', nargs=1)
parser.add_argument('-ct', type=str, help='-> input name of ct to eval', nargs=1)
parser.add_argument('-sm', type=str, help= '-> input name of folder where saved model params are located')
parser.add_argument('-e', type=int, help= '-> input expression value to eval on')
parser.add_argument('-o', type=str, help= '-> output dir', nargs=1)

args = vars(parser.parse_args())

# -- set necessary parameters from parsed in information
model_folder = args['sm']
ct = args['ct'][0]
data_directory = args['d'][0]
exp_val = args['e']
output_dir = args['o'][0]

lowest_dirs = []
for root, dirs, files_ignore in os.walk(data_directory): # -- get lowest directories 
    if not dirs: 
        lowest_dirs.append(root)
all_cell_types = next(os.walk(data_directory))[1] # -- this obtains the cell types

n_rep_interactions = []
n_prom_interactions = []

for idx, dirs in enumerate(lowest_dirs):
    X, color_list, mods_to_frequencies, loss_list = loop(model_folder=model_folder, 
                                  ct=all_cell_types[idx], exp_val=exp_val, 
                                  n_rep_interactions=n_rep_interactions, 
                                  n_prom_interactions=n_prom_interactions)
    
    histone_mods = ['H3K27me3 (R)', 'H3K36me3 (P)','H3K4me1 (DP)', 'H3K4me3 (P)', 'H3K9me3 (R)']
    bins = np.linspace(0, 99, 100).astype(int)
    plot_heatmap(all_cell_types[idx], np.swapaxes(X, 1, 0), mods_to_frequencies.values(),
                  color_list, histone_mods, bins, exp_val, output_dir)
    plot_loss(all_cell_types[idx], np.linspace(1,1000,1000), loss_list, 
                  exp_val, output_dir)

print(sum(n_rep_interactions))
print(sum(n_prom_interactions))