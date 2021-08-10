# -----------------------------------------------------------------------------
#                Copyright Huan Tran (huantd@gmail.com), 2021                 |
# -----------------------------------------------------------------------------
# MATSML script to get data, fingerprint data, train models, and make         |
# predictions as reported in                                                  |
#                                                                             |
#      V. N. Tuoc and T. D. Huan, "Information-fusion based approach for      |
#          small molecule properties", submitted 2021                         |
#                                                                             |
# -----------------------------------------------------------------------------

import pandas as pd
from matsml.fingerprint import Fingerprint
from matsml.models import FCNeuralNet, ProbabilityNeuralNet


##### (1) OBTAIN A RAW DATASET
##### (2) FINGERPRINT THE RAW DATA




##### (3) MODEL DEVELOPMENT #####
# Data parameters
sel = 2
if sel == 1:
    data_file = 'at_energy.csv'
    n_trains = 0.7                    # In unit of the dataset size
    sampling = 'random'               # Options: "random" and "stratified"
    x_scaling = 'minmax'              # Options: "minnmax", "log", ...
    y_scaling = 'normalize'           # Options: "minmax", 
    id_col = ['id']                   # ID column, given as a list
    y_cols = ['DFT_Eatomization']     # Y columns, given as a list
    comment_cols = []                 # Comment columns, not X nor Y nor ID

    data_params = {'data_file':data_file,'id_col':id_col,'y_cols':y_cols,
            'comment_cols':comment_cols,'y_scaling':y_scaling,
            'x_scaling':x_scaling,'sampling':sampling,'n_trains': n_trains}
else:
    data_file = 'comb_data.csv'
    id_col = ['id_cmb']         
    y_cols = ['prop']    
    comment_cols = ['id_std', 'name', 'smiles', 'ref','homo-lumo']                
    n_trains = 0.2
    sampling = 'random'
    x_scaling = 'minmax'
    y_scaling = 'normalize' 

    data_params = {'data_file':data_file, 'id_col':id_col,'y_cols':y_cols, 
            'comment_cols':comment_cols,'y_scaling':y_scaling,
            'x_scaling':x_scaling,'sampling':sampling, 'n_trains':n_trains}

# Model parameters
layers = [5]                     # list of nodes in hidden layers
epochs = 20                      # Epochs
nfold_cv = 5                     # Number of folds for cross validation
use_bias = True                  # Use bias term or not
file_model = 'model.pkl'         # Name of the model file to be created
loss = 'mse'                     #
metric = 'mse'                   #
verbosity = 0
batch_size = 32                  #
activ_funct = 'selu'             # Options: "tanh","relu","sigmoid","softmax", 
                                 # "softplus","softsign","selu","elu",
                                 # "exponential"
optimizer = 'nadam'              # options: "SGD","RMSprop","Adam","Adadelta", 
                                 # "Adagrad","Adamax","Nadam","Ftrl"

model_params = {'layers':layers,'activation_function':activ_funct,
        'epochs':epochs,'nfold_cv':nfold_cv,'optimizer':optimizer,
        'use_bias':use_bias,'file_model':file_model,'loss':loss,
        'metric':metric,'batch_size':batch_size,'verbosity':verbosity}

# Compile a fully connected neural net based model
model = FCNeuralNet(data_params=data_params,model_params=model_params)

# Train the model
model.train()



















