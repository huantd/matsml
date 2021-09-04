# Copyright Huan Tran (huantd@gmail.com), 2021                 
#
# MATSML script to get data, fingerprint data, train models, and make         
# predictions as reported in                                                  
#                                                                             
# V. N. Tuoc and T. D. Huan, "Information-fusion based approach for 
# small molecule properties", 2021                         
#                                                                             

import pandas as pd
from matsml.fingerprint import Fingerprint
from matsml.models import KRR

##### (1) OBTAIN A RAW DATASET

##### (2) FINGERPRINT THE RAW DATA

##### (3) TRAIN A MODEL #####
# data parameters
data_file = '/home/huan/work/papers/ml_mol/data/Jul24/Tm_Jul24/df_2D_num_final_filtered_Tm_num_5K_molecules_RFFE-200.csv'
id_col = ['id_std']         
y_cols = ['New_Tm']    
comment_cols = ['name', 'smiles', 'Tm', 'ref', 'homo-lumo']
n_trains = 1.0
sampling = 'random'
x_scaling = 'minmax'      
y_scaling = 'minmax' 
data_params = {'data_file':data_file, 'id_col':id_col,'y_cols':y_cols, 
        'comment_cols':comment_cols,'y_scaling':y_scaling,
        'x_scaling':x_scaling,'sampling':sampling, 'n_trains':n_trains}


# Model parameters
nfold_cv = 5                     # Number of folds for cross validation
file_model = 'model-04.pkl'      # Name of the model file to be created
metric = 'mse'                   #
verbosity = 0
rmse_cv=True
kernel='gaussian'

model_params = {'metric':metric,'nfold_cv':nfold_cv,'kernel':kernel,'file_model':file_model,
        'verbosity':verbosity,'rmse_cv':rmse_cv}

# Compile a model
model = KRR(data_params=data_params,model_params=model_params)

# Train the model
model.train()
