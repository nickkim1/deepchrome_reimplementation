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


# -- function for optimization 
def opt(inputs:list, pos_weights, num_epochs, model_path, device, width, num_filters, filter_size, pool_size, hidden_layer_units):

    # -- dictionaries for all cell types
    outputs_dict = {}
    loss_dict = {}
    expressions_dict = {}
    model = ConvNet(width=width, num_filters=num_filters, filter_size=filter_size, pool_size=pool_size, hidden_layer_units=hidden_layer_units)

    for (ct, X) in inputs: # -- enter the training loop 
        # -- create loss list for each cell type 
        losses = []

        # -- load in model w/ parameters for EACH cell type 
        model.load_state_dict(torch.load(f'{model_path}/{ct}_params.pth'))
        # print('before for model', next(model.parameters()).device)
        # print('after for model', next(model.parameters()).device)
        model.eval()
        # -- set up custom parameter group for optimizer
        s = X[0]
        s.requires_grad_(True)
        bin_list = {s}

        # -- set universal hyperparams for all cell types 
        learning_rate=0.1
        momentum=0.9
        lda=0.009
        
        optimizer = torch.optim.SGD(bin_list, lr=learning_rate, momentum=momentum, weight_decay=lda)
        pos_weights = pos_weights.to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

        for epoch in range(num_epochs):
            samples = X[0].permute(1,0)
            samples = samples.to(device)
            labels = X[1][0]
            labels = labels.to(device)
            model.to(device)
            predicted = model(samples)
            loss = criterion(predicted, labels) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # -- print the loss
            # print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            losses.append(loss.item())
        
        #-- get the output
        pg = optimizer.param_groups
        output = pg[0]['params'][0].detach().numpy() 
        outputs_dict[ct] = output
        loss_dict[ct] = losses
        expressions_dict[ct] = X[1][0].item()

    return outputs_dict, loss_dict, expressions_dict

def norm(raw_outputs_dict):
    all_normalized_arrs = {}

    # -- get the maximum of all the features 
    curr_max = 0
    for ignore, raw_output_array in raw_outputs_dict.items():
        for sub_array in raw_output_array:
            if np.amax(sub_array, axis=0) > curr_max:
                curr_max = np.amax(sub_array, axis=0) 

    for ct, raw_output_array in raw_outputs_dict.items():
        norm_array = []
        for sub_array in raw_output_array:
            normalized = np.clip(sub_array / curr_max, 0, 1) # -- clamp to range of [0,1] and normalize with max
            norm_array.append(normalized)
        norm_array = np.array(norm_array)
        all_normalized_arrs[ct] = norm_array

    return all_normalized_arrs

def plot_heatmap(cell_type, all_normalized_arrs, mods_to_frequencies, color_list, histone_mods, bins, expression_val, o_dir):

    # -- plot the graph for heatmap + barplot 
    f, a = plt.subplots(2, 1, figsize=(21,9) ,layout='constrained')

    if expression_val == 1:
        a[0].set_title(f'Bar Graph of Heatmap Values for {cell_type}. Gene Expression = High')
    elif expression_val == 0:
         a[0].set_title(f'Bar Graph of Heatmap Values for {cell_type}. Gene Expression = Low') 
    # color_list = ['black', 'green', 'red', 'blue', 'yellow']
    a[0].bar(histone_mods, mods_to_frequencies, color=color_list)

    if expression_val == 1:
        a[1].set_title(f'Cell Type: {cell_type}. Gene Expression = High')
    elif expression_val == 0:
        a[1].set_title(f'Cell Type: {cell_type}. Gene Expression = Low')
    
    # -- create custom discretized colorbar 
    # colors = ['blue','green','orange','red']
    # n_bins = 4
    # cmap_name = 'test'
    # cm = mpl.colors.LinearSegmentedColormap.from_list(cmap_name,colors,N=n_bins)
    # img = a[1].imshow(all_normalized_arrs, interpolation='nearest', cmap=cm)
    img = a[1].imshow(all_normalized_arrs)
    cbar = a[1].figure.colorbar(img, ax=a[1], shrink=0.2)
    cbar.ax.set_ylabel('Bin Activation', rotation=-90, va="bottom")
    # a[1].figure.colorbar(img, cmap=cm, ticks=[0.25,0.5,0.75,1], orientation='vertical',shrink=0.2)

    # -- show all the ticks and label them with respective entries (if specified)
    a[1].set_yticks(np.arange(0, 5, 1)-0.5, labels=histone_mods)
    a[1].set_xticks(np.arange(-1.5, 99, 1)+1)
    a[1].tick_params(axis='y', which='major', labelsize=6)
    a[1].tick_params(axis='x', which='major', labelsize=2)

    # -- create grid to distinguish between cells
    a[1].grid(color='black', visible=True, which='both',linestyle='-',linewidth=0.3)

    # -- save the file to specified output folder
    if o_dir != 'DELETE':
        plt.savefig(f'{o_dir}/{cell_type}_heatmap.png')
    


# == optional function for plotting loss (atm evidently all over the place)
def plot_loss(cell_type, num_epochs, loss):
    # -- plot the loss
    f, a = plt.subplots(layout='constrained') 
    f.suptitle(f'Evaluation Loss for {cell_type}') 
    a.plot(num_epochs, loss, label='Loss')
    a.set_xlabel('Number of Epochs')
    a.set_ylabel('Loss')
    a.legend()  

# def plot_feat_map(mean_of_bin_list, bin_number_list, o_dir):
#     f, a = plt.subplots(layout='constrained') 
#     f.suptitle(f'Deepchrome Feature Output Distribution across Cell Types') 
#     a.set_xlabel('Bins in Feature Map after Convolution (100-k+1)')
#     a.set_ylabel('Avg Feature Map Output across 56 Cell Types')
#     plt.scatter(x=bin_number_list, y=mean_of_bin_list)

#     if o_dir != 'DELETE':
#         plt.savefig(f'{o_dir}/feature_map_distribution.png')

# load in data into tensor format 
def main():

   # -- argparser: what args to look for? 
        # -- ** note that -h can be used to reference what these flags are / their meaning
        # -- input list of cell types to parse and visualize
    parser = ap.ArgumentParser(prog='Visualization of Deepchrome!', description='Please specify the cell-types to visualize below.')
    parser.add_argument('-d', type=str, help='-> input name of dataset directory to use', nargs=1)
    parser.add_argument('-sm', type=str, help= '-> input name of folder where saved model params are located')
    parser.add_argument('-c', help='-> input list of cell types to parse through', nargs='*')
    parser.add_argument('-e', type=int, help='-> input number of epochs that the model should optimize through', nargs=1)
    parser.add_argument('-hp', type=int, help='-> input 1 for (5,2) 2 for (5,5) 3 for (10,2) 4 for (10,5) to specify hp',default=4)
    parser.add_argument('-o',type=str, help='-> input name of folder where results should be saved')
    args = vars(parser.parse_args())

    # -- set necessary parameters from parsed in information
    cell_types = list(args['c']) # -- list of all the cell types to evaluate 
    data_directory = args['d'][0]
    num_epochs = args['e'][0]
    model_path = args['sm']
    hp_specs = args['hp']
    output_dir = args['o']
    eval_datasets = [] # -- list of the lowest possible directory (without reaching into files)
    for ct in cell_types: 
        for root, dirs, files_ignore in os.walk(data_directory):
            if not dirs and f'/{ct}'in root: 
                eval_datasets.append((ct, root))

    # -- run the functions proper with parsed in data from above
    device = set_device()
    data_outputs, pos_weights = load_data(datasets=eval_datasets)
    width, hidden_layer_units, num_filters, filter_size, pool_size, num_epochs, learning_rate = set_hyperparams(num_windows=100, n_e=num_epochs, hp_specs=hp_specs)
    outputs_dict, loss_dict, expressions_dict = opt(inputs=data_outputs, pos_weights=pos_weights, num_epochs=num_epochs, model_path=model_path, device=device, width=width, num_filters=num_filters, filter_size=filter_size, pool_size=pool_size, hidden_layer_units=hidden_layer_units)
    all_normalized_dict = norm(raw_outputs_dict=outputs_dict)

    # -- run plotting functions 
    # histone_mods = ['H3K4me3 (P)','H3K4me3 (P)','H3K4me1 (DP)','H3K36me3 (P)','H3K27me3 (R)']
    histone_mods = ['H3K27me3 (R)', 'H3K36me3 (P)','H3K4me1 (DP)', 'H3K4me3 (P)', 'H3K9me3 (R)']
    bins = np.linspace(0, 99, 100).astype(int)

    # -- iterate over cell types and plot heatmap (and) loss if specified 
    for ct in cell_types:
        d = np.swapaxes(all_normalized_dict[ct], 1, 0)
        mods_to_frequencies = []
        for row in d:
            frequency = 0
            for bins in row:
                if bins >= 0.25:   
                    frequency+=1
            mods_to_frequencies.append(frequency) 
        avg = np.mean(mods_to_frequencies)
        color_list = []
        for frequencies in mods_to_frequencies:
            if frequencies > avg:
                color_list.append("black")
            elif frequencies < avg:
                color_list.append("gray")
        plot_heatmap(cell_type=ct, all_normalized_arrs=d, mods_to_frequencies=mods_to_frequencies, color_list=color_list, histone_mods=histone_mods, bins=bins, expression_val=expressions_dict[ct], o_dir=output_dir)
    
    #     # -- plot the feature map 
    #     all_features = extract_features(inputs=data_outputs, ct=ct, model_path=model_path, width=width, num_filters=num_filters, pool_size=pool_size, hidden_layer_units=hidden_layer_units, filter_size=filter_size)
    #     for ct_feat in all_features:
    #         # -- each output of convolution is (50, 91). create graph of each bin (91) post convolution
    #         flipped_ct_feat = ct_feat.permute(1,0)
    #         max_of_bin_list = []
    #         for idx, bin_tensor in enumerate(flipped_ct_feat): # -- i want to get tensors along the 1, not 0, axis from original
    #             max_of_bin = max(bin_tensor)
    #             max_of_bin_list.append(max_of_bin)

    #         # -- pass in both lists to the plot function to plot 
    # mean_of_bin_list = np.array(max_of_bin_list) / len(cell_types)
    # bin_number_list = np.linspace(0,90,91)
    # plot_feat_map(mean_of_bin_list=mean_of_bin_list, bin_number_list=bin_number_list, o_dir=output_dir)

if __name__ == '__main__':
    main()