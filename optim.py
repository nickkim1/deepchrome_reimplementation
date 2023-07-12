import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib as mpl
from matplotlib import pyplot as plt
from doall import ConvNet
import numpy as np
import argparse as ap
import time as time
import random
import os


def set_device():
     # -- set the device used
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
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

# -- function for optimization 
def opt(inputs:list, pos_weights, num_epochs, model_path, device, width, num_filters, filter_size, pool_size, hidden_layer_units):

    # -- set universal hyperparams for all cell types 
    learning_rate=0.1
    momentum=0.9
    lda=0.009

    # -- dictionaries for all cell types
    outputs_dict = {}
    loss_dict = {}
    expressions_dict = {}

    for (ct, X) in inputs: # -- enter the training loop 
        # -- create loss list for each cell type 
        losses = []
        # -- load in model w/ parameters for EACH cell type 
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        model = ConvNet(width=width, num_filters=num_filters, filter_size=filter_size, pool_size=pool_size, hidden_layer_units=hidden_layer_units)
        model.load_state_dict(torch.load(f'{model_path}/{ct}_params.pth'))
        model.eval()
        # -- set up custom parameter group for optimizer
        s = X[0]
        s.requires_grad_(True)
        bin_list = {s}
        optimizer = torch.optim.SGD(bin_list, lr=learning_rate, momentum=momentum, weight_decay=lda)

        for epoch in range(num_epochs):
            samples = X[0].permute(1,0)
            samples = samples.to(device)
            labels = X[1][0]
            labels = labels.to(device)
            predicted = model(samples)
            loss = criterion(predicted, labels) 
            # zero the gradients 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # -- print the loss
            print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
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

def plot_heatmap(cell_type, all_normalized_arrs, histone_mods, bins, expression_val):

    # -- plot the heatmap
    f, a = plt.subplots(figsize=(11,8) ,layout='constrained') 
    # im = a.imshow(all_normalized_arrs)
    if expression_val == 1:
        a.set_title(f'Cell Type: {cell_type}. Gene Expression = High')
    else:
        a.set_title(f'Cell Type: {cell_type}. Gene Expression = Low')

    # -- show all the ticks and label them with respective entries (if specified)
    a.set_yticks(np.arange(0, 5, 1)-0.5, labels=histone_mods)
    a.set_xticks(np.arange(-1.5, 99, 1)+1)
    # a.set_xlim(left=1, right=99)
    a.tick_params(axis='y', which='major', labelsize=6)
    a.tick_params(axis='x', which='major', labelsize=2)
    
    # -- create custom discretized colorbar 
    colors = ['blue','green','orange','red']
    n_bins = 5
    cmap_name = 'test'
    cm = mpl.colors.LinearSegmentedColormap.from_list(cmap_name,colors,N=n_bins)
    img = plt.imshow(all_normalized_arrs, interpolation='nearest', cmap=cm)
    plt.colorbar(img, cmap=cm,ticks=[0,0.25,0.5,0.75,1], orientation='vertical',shrink=0.2)

    # -- create grid to distinguish between cells
    plt.grid(color='black',visible=True,which='both',linestyle='-',linewidth=0.3)
    plt.show()

def plot_loss(cell_type, num_epochs, loss):
    # -- plot the loss
    f, a = plt.subplots(layout='constrained') 
    f.suptitle(f'Evaluation Loss for {cell_type}') 
    a.plot(num_epochs, loss, label='Loss')
    a.set_xlabel('Number of Epochs')
    a.set_ylabel('Loss')
    a.legend()  

# load in data into tensor format 
def main():

   # -- argparser: what args to look for? 
        # -- ** note that -h can be used to reference what these flags are / their meaning
        # -- input list of cell types to parse and visualize
    parser = ap.ArgumentParser(prog='Visualization of Deepchrome!', description='Please specify the cell-types to visualize below.')
    parser.add_argument('-d', type=str, help='-> input name of dataset directory to use', nargs=1)
    parser.add_argument('-sm', type=str, help= '-> input name of folder where saved model is located')
    parser.add_argument('-c', help='-> input list of cell types to parse through', nargs='*')
    parser.add_argument('-e', type=int, help='-> input number of epochs that the model should optimize through', nargs=1)
    parser.add_argument('-hp', type=int, help='-> input 1 for (5,2) 2 for (5,5) 3 for (10,2) 4 for (10,5) to specify hp',default=4)
    parser.add_argument('-ol',type=int,help='-> decide to display loss or optimization or both')
    parser.add_argument('-o',type=int,help='-> input name of folder where results should be saved')
    args = vars(parser.parse_args())

    # -- set necessary parameters from parsed in information
    cell_types = list(args['c']) # -- list of all the cell types to evaluate 
    data_directory = args['d'][0]
    num_epochs = args['e'][0]
    model_path = args['sm']
    hp_specs = args['hp']
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
    histone_mods = ['H3K9me3 (R)','H3K4me3 (P)','H3K4me1 (DP)','H3K36me3 (P)','H3K27me3 (R)']
    bins = np.linspace(0, 99, 100).astype(int)

    # -- iterate over cell types and plot heatmap (and) loss if specified 
    for ct in cell_types:
        d = np.swapaxes(all_normalized_dict[ct], 1, 0)
        plot_heatmap(cell_type=ct, all_normalized_arrs=d, histone_mods=histone_mods, bins=bins, expression_val=expressions_dict[ct])
        # plot_loss(cell_type=ct,num_epochs=np.linspace(1, num_epochs, num=num_epochs).astype(int),loss=loss_dict[ct]) 

if __name__ == '__main__':
    main()