# Copyright Huan Tran (huantd@gmail.com), 2021
# 
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from scipy.stats import binned_statistic_2d
from sklearn import preprocessing
from matsml.data import ProcessData

from keras.layers import Dense, InputLayer
from keras.models import Sequential
from sklearn.model_selection import KFold
#from sklearn.metrics import mean_squared_error, r2_score
from keras.optimizers import Nadam

def prepare_data(data_params):
    """ Load data, scale data, and split data => train/test sets """

    # Work with data first
    data = ProcessData(data_params=data_params)

    # Extract x and y data from input file
    data.get_xy()

    # Scale x and y data
    data.scale_xy()

    # Prepare train and test sets
    data.prepare_train_test_data()

    data_dict={'id_col':data.id_col,'x_cols':data.x_cols,'y_cols':data.y_cols,
            'onehot_cols':data.onehot_cols,'n_trains':data.n_trains,
            'n_tests':data.n_tests,'train_set':data.train_set,
            'test_set':data.test_set,'y_mean':data.y_mean,'y_std':data.y_std,
            'y_max':data.y_max,'y_min':data.y_min,'y_scaling':data.y_scaling,
            'y_org':data.y}

    return data_dict


class FCNeuralNet:
    """ Generic Fully Connected NeuralNet with Tensorflow

    model_params:    Dictionary, direct input, parameters of the net
    data_params:     Dictionary, obtained from ProcessData, all needed for data

    """

    def __init__(self,data_params,model_params):
        self.data_params = data_params
        self.model_params = model_params

        self.sampling = self.data_params['sampling']
        self.nfold_cv = self.model_params['nfold_cv']
        self.file_model = self.model_params['file_model']
        self.layers = self.model_params['layers']
        self.activ_funct = self.model_params['activation_function']
        self.epochs = self.model_params['epochs']
        self.optimizer = self.model_params['optimizer']
        self.use_bias = self.model_params['use_bias']
        self.loss = self.model_params['loss']
        self.verbosity = self.model_params['verbosity']
        self.batch_size = self.model_params['batch_size']

        print (' ')
        print ('  A fully connected NeuralNet selected.')

    def get_data(self):
        """ Import train/test data and parameters to FCNeuralNet """
        
        # Data preprocessing
        data_dict = prepare_data(data_params=self.data_params)
        
        self.id_col = data_dict['id_col']
        self.x_cols = data_dict['x_cols']
        self.y_cols = data_dict['y_cols']
        self.y_mean = data_dict['y_mean']
        self.y_scaling = data_dict['y_scaling']
        self.y_std = data_dict['y_std']
        self.y_max = data_dict['y_max']
        self.y_min = data_dict['y_min']
        self.y_org = data_dict['y_org']
        self.onehot_cols = data_dict['onehot_cols']
        self.y_md_cols = ['md_'+col for col in self.y_cols]
        self.n_trains = data_dict['n_trains']
        self.n_tests = data_dict['n_tests']
        self.train_set = data_dict['train_set'].reset_index()
        self.test_set = data_dict['test_set'].reset_index()
        self.x_dim = len(self.x_cols)
        self.y_dim = len(self.y_cols)
        self.data_size = self.n_trains+self.n_tests

    def get_cv_data(self,train_cv,test_cv):
        """ Cross-validation datasets 
        
        Given the training set in self.train_set, and two sets of indices, 
        train_inds and test_inds from KFold, the cross-validation datasets
        are returned for training the model
        """

        x_cv_train = np.array(self.train_set.iloc[train_cv][self.x_cols])\
                .astype(np.float32)
        x_cv_test = np.array(self.train_set.iloc[test_cv][self.x_cols])\
                .astype(np.float32)
        y_cv_train = np.array(self.train_set.iloc[train_cv][self.y_cols])\
                .astype(np.float32)
        y_cv_test = np.array(self.train_set.iloc[test_cv][self.y_cols])\
                .astype(np.float32)

        return x_cv_train,x_cv_test,y_cv_train,y_cv_test
            
    def train(self):
        """ Build, train, test, and save model based in FCNeuralNet"""

        self.get_data()
        x_train_fp=np.array(self.train_set[self.x_cols]).astype(np.float32)
        x_test_fp=np.array(self.test_set[self.x_cols]).astype(np.float32)

        TEMPLATE = \
            "      cv,rmse_train,rmse_test,opt_rmse: {0:d} {1:.6f} {2:.6f} {3:.6f}"

        print ('  - Building model: FCNeuralNet')
        nn_model = Sequential()

        # Input layer
        nn_model.add(InputLayer(input_shape=(self.x_dim,)))

        # Append hidden layers
        for layer in self.layers:
            nn_model.add(Dense(layer,kernel_initializer='normal', 
                activation=self.activ_funct, use_bias=self.use_bias))

        # Output layer    
        nn_model.add(Dense(self.y_dim,kernel_initializer='normal',
            use_bias=self.use_bias))

        # Compile
        nn_model.compile(loss=self.loss,optimizer=self.optimizer)

        # Model summary
        if int(self.verbosity) > 0:
            print(nn_model.summary())

        #kfold splitting
        kf_=KFold(n_splits=self.nfold_cv,shuffle=True)
        kf = kf_.split(self.train_set)

        opt_rmse = 1.0E20
        ncv = 0
        ncv_opt = ncv

        print ('  - Training model with cross validation')
        for train_cv,test_cv in kf:

            x_cv_train, x_cv_test, y_cv_train, y_cv_test = \
                    self.get_cv_data(train_cv,test_cv)

            nn_model.fit(x_cv_train,y_cv_train,epochs=self.epochs,batch_size=\
                    int(self.batch_size),verbose=int(self.verbosity))

            y_cv_train_md = nn_model.predict(x_cv_train).flatten()
            y_cv_test_md = nn_model.predict(x_cv_test).flatten()

            rmse_cv_train=np.sqrt(np.mean((np.array(y_cv_train)\
                    .reshape(len(train_cv)*self.y_dim)-y_cv_train_md)**2))
            rmse_cv_test=np.sqrt(np.mean((np.array(y_cv_test)\
                    .reshape(len(test_cv)*self.y_dim)-y_cv_test_md)**2))

            if rmse_cv_test < opt_rmse:
                opt_rmse=rmse_cv_test
                nn_model.save_weights(self.file_model)
                ncv_opt=ncv

            print (TEMPLATE.format(ncv,rmse_cv_train,rmse_cv_test,opt_rmse))
            ncv=ncv+1

        print('    Optimal ncv: ',ncv_opt,"; optimal NET saved.")
        
        nn_model.load_weights(self.file_model)

        # A dictionary of scaling parameters
        scaling_dic = {'id_col':self.id_col,'y_cols':self.y_cols,
                'onehot_cols':self.onehot_cols,'y_org':self.y_org,
                'y_scaling':self.y_scaling,'y_mean':self.y_mean,
                'y_std':self.y_std,'y_min':self.y_min,'y_max':self.y_max}

        # Make predictions on the training and test datasets
        y_train_md = nn_model.predict(x_train_fp)
        predicted_train_set = pd.concat([self.train_set[self.id_col+
            self.onehot_cols],self.train_set[self.y_cols],
            pd.DataFrame(y_train_md,columns=self.y_md_cols)],axis=1,
            ignore_index=True)

        predicted_train_set.columns=self.id_col+self.onehot_cols+\
            self.y_cols+self.y_md_cols

        unscaled_train_set=ProcessData.unscale_y(predicted_train_set,
            scaling_dic,'training')

        unscaled_train_set.to_csv('training.csv',index=False)

        if self.n_tests > 0:
            y_test_md = nn_model.predict(x_test_fp)
            predicted_test_set = pd.concat([self.test_set[self.id_col+
                self.onehot_cols],self.test_set[self.y_cols],
                pd.DataFrame(y_test_md,columns=self.y_md_cols)],axis=1,
                ignore_index=True)

            predicted_test_set.columns=self.id_col+self.onehot_cols+\
                self.y_cols+self.y_md_cols

            unscaled_test_set=ProcessData.unscale_y(predicted_test_set,
                scaling_dic,'test')
            
            unscaled_test_set.to_csv('test.csv',index=False)

        print ('  - NN trained and predictions made')


class KRR:
    """ Kernel Ridge Regression  with scikit-learn """

    def __init__(self,data_params,model_params):
        self.data_params = data_params
        self.model_params = model_params

    def get_data(self):

        self.x_cols,self.y_cols,self.onehot_cols,self.n_trains,\
                self.n_tests,self.x_train_set,self.x_test_set,\
                self.y_train_set,self.y_test_set = \
                prepare_data(data_params=self.data_params)

        self.data_size = self.n_trains+self.n_tests

    def train(self):
        self.get_data()

        print (self.x_train_set)
        print ('Come here')


class GPR:
    """ Gaussian Process Regression with scikit-learn """

    def __init__(self,data_params,model_params):
        self.data_params = data_params
        self.model_params = model_params

    def get_data(self):

        self.x_cols,self.y_cols,self.onehot_cols,self.n_trains,\
                self.n_tests,self.x_train_set,self.x_test_set,\
                self.y_train_set,self.y_test_set = \
                prepare_data(data_params=self.data_params)

        self.data_size = self.n_trains+self.n_tests

    def train(self):
        self.get_data()

        print (self.x_train_set)
        print ('Come here')


class ProbabilityNeuralNet:
    """ Probability NeuralNet with Tensorflow-Probability"""

    def __init__(self,data_params,model_params):
        self.data_params = data_params
        self.model_params = model_params
