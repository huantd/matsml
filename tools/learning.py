import pandas as pd
import glob, os
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

cdir = os.getcwd()

data = pd.read_csv('training.csv')
sel_cols = [col for col in list(data.columns) if "selector" in col]
md_cols = [col for col in list(data.columns) if "md_" in col]
ref_cols = [col[3:] for col in md_cols]
#sel_cols = ['selector1','selector2']

xlist = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

if len(sel_cols) > 1:
    for sel_col in sel_cols:
        error_df = pd.DataFrame(columns = ['X','rmse_train_mean', 'rmse_train_std', 'rmse_test_mean', 'rmse_test_std'])
        for x in xlist:
            flist = glob.glob(cdir + '/training-' + str(x) + '-*.csv')
            rmse_train_list = []
            for fname in flist:
                data = pd.read_csv(fname)
                sel_data = data[data[sel_col] == 1]
                rmse = mean_squared_error(sel_data[ref_cols[0]], sel_data[md_cols[0]], squared=False)
                rmse_train_list.append(rmse)
            rmse_train_mean = np.mean(rmse_train_list)
            rmse_train_std = np.std(rmse_train_list)
        
            flist = glob.glob(cdir + '/test-' + str(x) + '-*.csv')
            rmse_test_list = []
            for fname in flist:
                data = pd.read_csv(fname)
                sel_data = data[data[sel_col] == 1]
                rmse = mean_squared_error(sel_data[ref_cols[0]], sel_data[md_cols[0]], squared=False)
                rmse_test_list.append(rmse)
            rmse_test_mean = np.mean(rmse_test_list)
            rmse_test_std = np.std(rmse_test_list)
        
            new_row = {'X':x, 'rmse_train_mean': rmse_train_mean, 'rmse_train_std': rmse_train_std, 
                    'rmse_test_mean': rmse_test_mean, 'rmse_test_std': rmse_test_std}

            error_df = pd.concat([error_df, pd.DataFrame([new_row])], axis = 0)
        
        error_df.to_csv('learning_'+str(sel_col)+'.csv',index = False)
elif len(sel_cols) == 0:
    error_df = pd.DataFrame(columns = ['X','rmse_train_mean', 'rmse_train_std', 'rmse_test_mean', 'rmse_test_std'])
    for x in xlist:
        flist = glob.glob(cdir + '/training-' + str(x) + '-*.csv')
        rmse_train_list = []
        for fname in flist:
            data = pd.read_csv(fname)
            rmse = mean_squared_error(data[ref_cols[0]], data[md_cols[0]], squared=False)
            rmse_train_list.append(rmse)
        rmse_train_mean = np.mean(rmse_train_list)
        rmse_train_std = np.std(rmse_train_list)
    
        flist = glob.glob(cdir + '/test-' + str(x) + '-*.csv')
        rmse_test_list = []
        for fname in flist:
            data = pd.read_csv(fname)
            rmse = mean_squared_error(data[ref_cols[0]], data[md_cols[0]], squared=False)
            rmse_test_list.append(rmse)
        rmse_test_mean = np.mean(rmse_test_list)
        rmse_test_std = np.std(rmse_test_list)
    
        new_row = {'X':x, 'rmse_train_mean': rmse_train_mean, 'rmse_train_std': rmse_train_std, 
                'rmse_test_mean': rmse_test_mean, 'rmse_test_std': rmse_test_std}
        error_df = pd.concat([error_df, pd.DataFrame([new_row])], axis = 0)
    
    error_df.to_csv('learning.csv',index = False)
