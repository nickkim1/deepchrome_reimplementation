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
import os

def set_device():
     # -- set the device used
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"0. Using {device} device!")
    return device

# -- function for loading in the parsed data into tensor form
def load_data(ct, ds):
    num_windows = 100
    inputs = []

    xy = torch.from_numpy(np.loadtxt(f'{ds}/train.csv', delimiter=",",dtype=np.float32))
    num_rows = len(xy)
    num_cols = len(xy[0])
    num_genes = num_rows / num_windows
    marker = 0 
    data = {}
    num_zeros = 0
    num_ones = 0

    # -- set entries for data
    data[0] = []
    data[1] = []

    for i in range(len(xy)):
        if xy[i][num_cols-1]==0:
            num_zeros += 1
        if xy[i][num_cols-1]==1:
            num_ones += 1
        if (i+1) % num_windows==0:
            if xy[i][num_cols-1]==0:
                data[0].append((xy[i-99:i+1,2:num_cols-1], xy[i-99:i+1,[num_cols-1]]))
            if xy[i][num_cols-1]==1:
                data[1].append((xy[i-99:i+1,2:num_cols-1], xy[i-99:i+1,[num_cols-1]]))

    pos_weights = torch.tensor([num_zeros / num_ones])
    random_zero_idx = random.randint(0, len(data[0])-1)
    random_one_idx = random.randint(0, len(data[1])-1)
    inputs.append(ct) # -- use cell type to mark each input 
    # inputs.append(data[0][random_zero_idx])
    # print(data[0][random_zero_idx].shape)
    #print(torch.rand(100,5))
    inputs.append((torch.rand(100,5), torch.zeros(100,1)))
    inputs.append((torch.rand(100,5), torch.ones(100,1)))
    # inputs.append(data[1][random_one_idx])
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


# -- function for optimization 
def opt(inputs:list, pos_weights, num_epochs, model_path, device, width, num_filters, filter_size, pool_size, hidden_layer_units):

    # -- dictionaries for all cell types
    outputs_dict = {}
    loss_dict = {}
    expressions_dict = {}

    # -- unpack inputs
    ct = inputs[0]

    # -- add in an empty list for the ct
    outputs_dict[ct] = []
    loss_dict[ct] = []
    expressions_dict[ct] = []

    # -- create the model 
    model = ConvNet(width=width, num_filters=num_filters, filter_size=filter_size, pool_size=pool_size, hidden_layer_units=hidden_layer_units)

    for i in range(1, len(inputs)): # -- enter the training loop 

        # -- create loss list for each cell type 
        losses = []

        # -- load in model's parameters for EACH cell type 
        model.load_state_dict(torch.load(f'{model_path}/{ct}_params.pth'))
        model.eval()

        # -- set up custom parameter group for optimizer
        s = inputs[i][0]
        s.requires_grad_(True)
        bin_list = {s}

        # -- set universal hyperparams for all cell types 
        
        # -- set optimizer and loss function 
        optimizer = torch.optim.RMSprop(bin_list, lr=0.1, momentum=0.9)
        # print(optimizer.param_groups)
        pos_weights = pos_weights.to(device)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(num_epochs):
            samples = inputs[i][0].permute(1,0)
            samples = samples.to(device)
            labels = inputs[i][1][0]
            labels = labels.to(device)
            model.to(device)
            predicted = model(samples)
            loss = criterion(predicted, labels) 

            # l2_norm = torch.tensor(0, dtype=float)

            # for parameter in model.parameters():
                # print(parameter.shape)
                # l2_norm += (torch.norm(parameter))
                # print(l2_norm)
                # print('initial', parameter.grad)
            # loss += 0.009 * (l2_norm)

            grads = torch.autograd.grad(loss, samples, create_graph=True)
            grad_norm = 0 
            for grad in grads:
                grad_norm += grad.pow(2).sum()
                grad=grad + (samples*2*0.009)
            optimizer.zero_grad()
            loss=loss + (0.009*(grad_norm).pow(2))
            loss.backward()
            optimizer.step()

            # for parameter in model.parameters():
                # print('parameter', parameter)
                # if (epoch+1) % 100 == 0:
                    # print('initial', parameter.grad)
                # parameter.grad = parameter.grad + (parameter * 2 * 0.009)
                # if (epoch+1) % 100 == 0:
                    # print('final', parameter.grad)

            # -- print the loss
            if (epoch + 1) % 100 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            losses.append(loss.item())
        
        #-- get the output
        pg = optimizer.param_groups
        # print(pg)
        output = pg[0]['params'][0].detach().numpy() 
        # print(s)
        outputs_dict[ct].append(output)
        loss_dict[ct].append(losses) # append losses for each expression type 
        expressions_dict[ct].append(inputs[i][1][0].item())

    return outputs_dict, loss_dict, expressions_dict

def norm(raw_outputs_tuple, ct):
    all_normalized_arrs = {}

    # -- get the maximum of all the features 
    curr_max = np.max(raw_outputs_tuple)
    # for sub_array in raw_outputs_tuple:
    #     if np.amax(sub_array, axis=0) > curr_max:
    #         curr_max = np.amax(sub_array, axis=0) 

    norm_array = []
    for sub_array in raw_outputs_tuple:
        normalized = np.clip(sub_array / curr_max, 0, 1) # -- clamp to range of [0,1] and normalize with max
        norm_array.append(normalized)
    norm_array = np.array(norm_array)

    all_normalized_arrs[ct] = norm_array

    # # -- get the maximum of all the features 
    # curr_max = np.max(raw_outputs_tuple)
    # print(curr_max)
    # # for sub_array in raw_outputs_tuple:
    # #     if np.amax(sub_array, axis=0) > curr_max:
    # #         curr_max = np.amax(sub_array, axis=0) 

    # norm_array = []
    # for i in range(100):
    #     summed_array = np.sum(raw_outputs_tuple[i])
    #     if summed_array == 0: 
    #         raw_outputs_tuple[i] = torch.zeros(5)
    #     else:
    #         for j in range(5):
    #             if raw_outputs_tuple[i][j] < 0:
    #                 raw_outputs_tuple[i][j] = 0
    #             raw_outputs_tuple[i][j] = raw_outputs_tuple[i][j] / curr_max
    #     normalized = raw_outputs_tuple[i] # -- clamp to range of [0,1] and normalize with max
    #     print(normalized)
    #     norm_array.append(normalized)
    # norm_array = np.array(norm_array)

    # all_normalized_arrs[ct] = norm_array

    return all_normalized_arrs

def plot_heatmap(cell_type, all_normalized_arrs, mods_to_frequencies, color_list, histone_mods, bins, expression_val, o_dir):

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
    
# == optional function for plotting loss (atm evidently all over the place)
def plot_loss(cell_type, num_epochs, loss, o_dir, expression_val):
    f, a = plt.subplots(layout='constrained') 
    f.suptitle(f'Evaluation Loss for {cell_type}') 
    a.plot(num_epochs, loss, label='Loss')
    a.set_xlabel('Number of Epochs')
    a.set_ylabel('Loss')
    a.legend()  

    if o_dir != 'DELETE':
        plt.savefig(f'{o_dir}/{cell_type}_{int(expression_val)}_loss.png')

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

    lowest_dirs = []
    for root, dirs, files_ignore in os.walk(data_directory): # -- get lowest directories 
        if not dirs: 
            lowest_dirs.append(root)
            
    all_cell_types = next(os.walk(data_directory))[1] # -- this obtains the cell types
    eval_datasets = [] # -- list of the (ct, corres. to lowest possible directory) -- without reaching into files

    marker = 0
    for dirs in lowest_dirs:

        # -- run the functions proper with parsed in data from above
        device = set_device()
        data_outputs, pos_weights = load_data(ds=dirs, ct=all_cell_types[marker])
        width, hidden_layer_units, num_filters, filter_size, pool_size, num_epochs, learning_rate = set_hyperparams(num_windows=100, n_e=num_epochs, hp_specs=hp_specs)
        outputs_dict, loss_dict, expressions_dict = opt(inputs=data_outputs, pos_weights=pos_weights, num_epochs=num_epochs, model_path=model_path, device=device, width=width, num_filters=num_filters, filter_size=filter_size, pool_size=pool_size, hidden_layer_units=hidden_layer_units)
        
        # -- define parameters for plotting functions
        histone_mods = ['H3K27me3 (R)', 'H3K36me3 (P)','H3K4me1 (DP)', 'H3K4me3 (P)', 'H3K9me3 (R)']
        bins = np.linspace(0, 99, 100).astype(int)

        total_num_repressors_correct=0
        total_num_promoters_correct=0
        n_rep = []
        n_prom = []

        for ct, raw_output_list in outputs_dict.items():
            for idx, item in enumerate(raw_output_list): 
                # -- normalize the outputs 
                all_normalized_dict = norm(raw_outputs_tuple=item, ct=ct)
                
                # -- plot everything 
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
                for mod, frequencies in enumerate(mods_to_frequencies):
                    if frequencies > avg:
                        color_list.append("black")
                    if frequencies <= avg:
                        color_list.append("gray")
                    if frequencies > avg and (mod == 1 or mod == 2 or mod == 3) and expressions_dict[ct][idx] == 0:
                        n_rep.append(mod)
                    if frequencies > avg and (mod == 0 or mod == 4) and expressions_dict[ct][idx] == 1: 
                        n_prom.append(mod)

                plot_heatmap(cell_type=ct, all_normalized_arrs=d, mods_to_frequencies=mods_to_frequencies, color_list=color_list, histone_mods=histone_mods, bins=bins, expression_val=expressions_dict[ct][idx], o_dir=output_dir)
                plot_loss(cell_type=ct, num_epochs=np.linspace(1, num_epochs, num_epochs), loss=loss_dict[ct][idx], o_dir=output_dir, expression_val=expressions_dict[ct][idx])

        if 1 in n_rep or 2 in n_rep or 3 in n_rep: 
            total_num_repressors_correct+=0
        else:
            total_num_repressors_correct+=1

        if 0 in n_prom or 4 in n_prom:
            total_num_promoters_correct+=0
        else:
            total_num_promoters_correct+=1

        # -- update the marker 
        marker+=1

    print('(Strict) The total percentage of cell types with a correlation with repressor-associated marks is: ', total_num_repressors_correct / len(all_cell_types))
    print('(Strict) The total percentage of cell types with a correlation with promoter-associated marks is: ', total_num_promoters_correct / len(all_cell_types))


if __name__ == '__main__':
    main()
    