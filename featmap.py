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

def set_device():
     # -- set the device used
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"0. Using {device} device!")
    return device

# -- function for loading in the parsed data into tensor form
def load_data(datasets:list):
    num_windows = 100
    inputs = []
    for ct, ds in datasets: 
        xy = torch.from_numpy(np.loadtxt(f'{ds}/train.csv', delimiter=",",dtype=np.float32))
        num_rows = len(xy)
        num_cols = len(xy[0])
        num_genes = num_rows / num_windows
        marker = 0 
        data = {}
        num_zeros = 0
        num_ones = 0
        for i in range(len(xy)):
            if xy[i][num_cols-1]==0:
                num_zeros += 1
            if xy[i][num_cols-1]==1:
                num_ones += 1
            if (i+1) % num_windows==0:
                data[marker]=(xy[i-99:i+1,2:num_cols-1], xy[i-99:i+1,[num_cols-1]]) # (features, labels
                # print(data[marker][0][0], data[marker][1][0])
                marker+=1 
        pos_weights = torch.tensor([num_zeros / num_ones])
        num = random.randint(0,num_genes-1) # -- generate a random number from 0-(num_genes-1) in ds inclusive, -1 bc of indexing
        inputs.append((ct, data[num])) # -- use cell type to mark each input 
    # print(inputs[0][1][0][0])
    return inputs, pos_weights

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
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x) 
        return x


def set_hyperparams(num_windows, n_e, hp_specs):
    # -- spec values 
    width = num_windows
    learning_rate = 0.001
    num_epochs = n_e
    num_filters = 50
    hidden_layer_units = {"first": 625, "second": 125}

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

    return width, hidden_layer_units, num_filters, filter_size, pool_size, num_epochs, learning_rate

# -- function for extracting bin specific scores
def extract_features(device, inputs:list, ct, model_path, width, num_filters, filter_size, pool_size, hidden_layer_units):

    all_features = []

    for (ct, X) in inputs: 
        # -- get the sample input from tensor data
        sample_input = X[0].permute(1,0)

        # -- load in the model for that cell type
        model = ConvNet(width=width, num_filters=num_filters, filter_size=filter_size, pool_size=pool_size, hidden_layer_units=hidden_layer_units)
        if torch.cuda.is_available():
            model.cuda()
        model.load_state_dict(torch.load(f'{model_path}/{ct}_params.pth'))
        model.eval()
        # -- enter evaluation loop
        with torch.no_grad():
            conv_output = None
            # -- get output of first convolutional layer with hook function
            def my_hook(module, input, output):
                nonlocal conv_output
                conv_output = output.detach() # -- detach from computational graph
            # -- run the hook function on the convolution layer 
            model.conv.register_forward_hook(my_hook)
            sample_input = sample_input.to(device)
            model.to(device)
            model(sample_input)
            all_features.append(conv_output)
    return all_features


def plot_feat_map(mean_of_bin_list, bin_number_list, o_dir):
    f, a = plt.subplots(layout='constrained') 
    f.suptitle(f'Deepchrome Feature Output Distribution across Cell Types') 
    a.set_xlabel('Bins in Feature Map after Convolution (100-k+1)')
    a.set_ylabel('Avg Feature Map Output across 56 Cell Types')
    plt.scatter(x=bin_number_list, y=mean_of_bin_list)

    if o_dir != 'DELETE':
        plt.savefig(f'{o_dir}/new_feature_map_distribution.png')


# load in data into tensor format 
def main():

   # -- argparser: what args to look for? 
        # -- ** note that -h can be used to reference what these flags are / their meaning
        # -- input list of cell types to parse and visualize
    parser = ap.ArgumentParser(prog='Visualization of Deepchrome!', description='Please specify the cell-types to visualize below.')
    parser.add_argument('-d', type=str, help='-> input name of dataset directory to use', nargs=1)
    parser.add_argument('-sm', type=str, help= '-> input name of folder where saved model params are located')
    parser.add_argument('-e', type=int, help='-> input number of epochs that the model should optimize through', nargs=1)
    parser.add_argument('-hp', type=int, help='-> input 1 for (5,2) 2 for (5,5) 3 for (10,2) 4 for (10,5) to specify hp',default=4)
    parser.add_argument('-o',type=str, help='-> input name of folder where results should be saved')
    args = vars(parser.parse_args())

    # -- set necessary parameters from parsed in information
    data_directory = args['d'][0]
    num_epochs = args['e'][0]
    model_path = args['sm']
    hp_specs = args['hp']
    output_dir = args['o']
    lowest_dirs = [] # -- list of the lowest possible directory (without reaching into files)
    for root, dirs, files_ignore in os.walk(data_directory):
        if not dirs: 
            lowest_dirs.append(root)
    all_cell_types = next(os.walk(data_directory))[1] # -- this obtains the cell types
    eval_datasets = [] # -- for input into "dataloader"
    for ct in all_cell_types:
        for dirs in lowest_dirs:
            if f'/{ct}' in root: 
                eval_datasets.append((ct, dirs))

    # -- run the functions proper with parsed in data from above
    device = set_device()
    data_outputs, pos_weights = load_data(datasets=eval_datasets)
    width, hidden_layer_units, num_filters, filter_size, pool_size, num_epochs, learning_rate = set_hyperparams(num_windows=100, n_e=num_epochs, hp_specs=hp_specs)

    # -- iterate over cell types and get feature map measurements
    for ct in all_cell_types:
        # -- plot the feature map 
        all_features = extract_features(device=device, inputs=data_outputs, ct=ct, model_path=model_path, width=width, num_filters=num_filters, pool_size=pool_size, hidden_layer_units=hidden_layer_units, filter_size=filter_size)
        for ct_feat in all_features:
            # -- each output of convolution is (50, 91). create graph of each bin (91) post convolution
            flipped_ct_feat = ct_feat.permute(1,0)
            max_of_bin_list = []
            for ignore, bin_tensor in enumerate(flipped_ct_feat): # -- i want to get tensors along the 1, not 0, axis from original
                max_of_bin = max(bin_tensor)
                max_of_bin_list.append(max_of_bin)

    # -- pass in both lists to the plot function to plot 
    mean_of_bin_list = np.array(max_of_bin_list) / len(all_cell_types)
    bin_number_list = np.linspace(0,90,91)
    plot_feat_map(mean_of_bin_list=mean_of_bin_list, bin_number_list=bin_number_list, o_dir=output_dir)

if __name__ == '__main__':
    main()