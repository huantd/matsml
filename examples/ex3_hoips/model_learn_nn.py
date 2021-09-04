#  Copyright Huan Tran (huantd@gmail.com), 2021                 
#
#  MATSML script to get data, fingerprint data, train models, and make         
#  predictions as reported in                                                  
#                                                                             
#  Vu Ngoc Tuoc, Nga T.T. Nguyen, Vinit Sharma, and Tran Doan Huan, 
# "Probabilistic deep learning approach for targeted hybrid 
#  organic-inorganic perovskites" 
#                                                                             

import pandas as pd
from matsml.fingerprint import Fingerprint
from matsml.models import FCNeuralNet

##### (1) OBTAIN A RAW DATASET

##### (2) FINGERPRINT THE RAW DATA

##### (3) TRAIN A MODEL #####
# data parameters
data_file = 'hoips_2dest.csv'
id_col = ['ID']         
y_cols = ['Ymean','Ystd']    
comment_cols = []
n_trains = 0.85
sampling = 'random'
x_scaling = 'minmax'      
y_scaling = 'minmax' 
data_params = {'data_file':data_file, 'id_col':id_col,'y_cols':y_cols,
    'comment_cols':comment_cols,'y_scaling':y_scaling,'x_scaling':x_scaling,
    'sampling':sampling, 'n_trains':n_trains}


# Model parameters
layers = [6]                     # list of nodes in hidden layers
epochs = 50                     # Epochs
nfold_cv = 5                     # Number of folds for cross validation
use_bias = True                  # Use bias term or not
model_file = 'model.pkl'         # Name of the model file to be created
loss = 'mse'                     #
metric = 'mse'                   #
verbosity = 0
batch_size = 32                  #
activ_funct = 'selu'             # Options: "tanh","relu","sigmoid","softmax", 
                                 # "softplus","softsign","selu","elu",
                                 # "exponential"
optimizer = 'nadam'              # options: "SGD","RMSprop","Adam","Adadelta", 
                                 # "Adagrad","Adamax","Nadam","Ftrl"

model_params={'layers':layers,'activ_funct':activ_funct,'epochs':epochs,
    'nfold_cv':nfold_cv,'optimizer':optimizer,'use_bias':use_bias,
    'model_file':model_file,'loss':loss,'metric':metric,
    'batch_size':batch_size,'verbosity':verbosity,'rmse_cv':True}

# Compile a model
model=FCNeuralNet(data_params=data_params,model_params=model_params)

# Train the model
model.train()

# Load the saved model (in model_file)
#model.predict()
