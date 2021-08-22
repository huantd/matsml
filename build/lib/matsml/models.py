# By Huan Tran (huantd@gmail.com), 2021
# 
import os,joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from scipy.stats import binned_statistic_2d
from sklearn import preprocessing
from matsml.data import ProcessData

from keras.layers import Dense,InputLayer
from keras.models import Sequential
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from keras.optimizers import Nadam


class FCNeuralNet:
    """ FCNeuralNet: Generic Fully Connected NeuralNet with Tensorflow

    Parameters
    ----------
    model_params:    Dictionary, direct input, parameters of the net
    data_params:     Dictionary, obtained from ProcessData, all needed for data

    """

    def __init__(self,data_params,model_params):
        self.data_params=data_params
        self.model_params=model_params

        self.sampling=self.data_params['sampling']
        self.nfold_cv=self.model_params['nfold_cv']
        self.file_model=self.model_params['file_model']
        self.layers=self.model_params['layers']
        self.activ_funct=self.model_params['activ_funct']
        self.epochs=self.model_params['epochs']
        self.optimizer=self.model_params['optimizer']
        self.use_bias=self.model_params['use_bias']
        self.loss=self.model_params['loss']
        self.verbosity=self.model_params['verbosity']
        self.batch_size=self.model_params['batch_size']
        self.rmse_cv=self.model_params['rmse_cv']
        print (' ')
        print ('  Algorithm: fully connected NeuralNet w/ Tensorflow')

    def load_data(self):
        """ Import train/test data and parameters to FCNeuralNet """
        
        # Data preprocessing
        data_processor=ProcessData(data_params=self.data_params)
        data_dict=data_processor.load_data()
        
        self.id_col=data_dict['id_col']
        self.x_cols=data_dict['x_cols']
        self.y_cols=data_dict['y_cols']
        self.y_scaling=data_dict['y_scaling']
        self.y_mean=data_dict['y_mean']
        self.y_std=data_dict['y_std']
        self.y_max=data_dict['y_max']
        self.y_min=data_dict['y_min']
        self.y_org=data_dict['y_org']
        self.sel_cols=data_dict['sel_cols']
        self.y_md_cols=['md_'+col for col in self.y_cols]
        self.n_trains=data_dict['n_trains']
        self.n_tests=data_dict['n_tests']
        self.train_set=data_dict['train_set'].reset_index()
        self.test_set=data_dict['test_set'].reset_index()
        self.x_dim=len(self.x_cols)
        self.y_dim=len(self.y_cols)
        self.data_size=self.n_trains+self.n_tests
        self.x_train=np.array(self.train_set[self.x_cols]).astype(np.float32)
        self.x_test=np.array(self.test_set[self.x_cols]).astype(np.float32)
            
    def train(self):
        """ Build, train, test, and save model based in FCNeuralNet"""

        self.load_data()
        data_processor=ProcessData(data_params=self.data_params)

        # A dictionary of scaling parameters
        scaling_dic={'id_col':self.id_col,'y_cols':self.y_cols,
            'sel_cols':self.sel_cols,'y_org':self.y_org,
            'y_scaling':self.y_scaling,'y_mean':self.y_mean,
            'y_std':self.y_std,'y_min':self.y_min,'y_max':self.y_max}

        tpl1=\
            "    cv,rmse_train,rmse_test,rmse_opt: {0:d} {1:.6f} {2:.6f} {3:.6f}"

        print ('  Building model: FCNeuralNet')

        nn_model=Sequential()

        # Input layer
        nn_model.add(InputLayer(input_shape=(self.x_dim,)))

        # Append hidden layers
        for layer in self.layers:
            nn_model.add(Dense(layer,kernel_initializer='normal', 
                activation=self.activ_funct,use_bias=self.use_bias))

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
        kf=kf_.split(self.train_set)

        opt_rmse=1.0E20
        ncv=0
        ncv_opt=ncv
        
        print ('  Training model w/ cross validation')

        for train_cv,test_cv in kf:

            x_cv_train,x_cv_test,y_cv_train,y_cv_test=data_processor\
                .get_cv_datasets(self.train_set,self.x_cols,self.y_cols,
                train_cv,test_cv)

            nn_model.fit(x_cv_train,y_cv_train,epochs=self.epochs,batch_size=\
                int(self.batch_size),verbose=int(self.verbosity))

            y_cv_train_md=nn_model.predict(x_cv_train).flatten()
            y_cv_test_md=nn_model.predict(x_cv_test).flatten()

            rmse_cv_train=np.sqrt(mean_squared_error(np.array(y_cv_train)\
                .reshape(len(train_cv)*self.y_dim),y_cv_train_md))
            rmse_cv_test=np.sqrt(mean_squared_error(np.array(y_cv_test)\
                .reshape(len(test_cv)*self.y_dim),y_cv_test_md))

            if rmse_cv_test<opt_rmse:
                opt_rmse=rmse_cv_test
                nn_model.save_weights(self.file_model)
                ncv_opt=ncv

            print (tpl1.format(ncv,rmse_cv_train,rmse_cv_test,opt_rmse))
            ncv=ncv+1

            if self.rmse_cv:
                y_cv_test_md=nn_model.predict(x_cv_test)
                pred_cv_test=pd.concat([self.train_set.iloc[test_cv]\
                    [self.id_col+self.sel_cols+self.y_cols]\
                    .reset_index(),pd.DataFrame(y_cv_test_md,
                    columns=self.y_md_cols)],axis=1,ignore_index=True)
                pred_cv_test.columns=['index']+self.id_col+\
                    self.sel_cols+self.y_cols+self.y_md_cols
                unscaled_cv_test=data_processor.invert_scale_y(pred_cv_test,
                    scaling_dic,'cv_test')

        print('    Optimal ncv: ',ncv_opt,"; optimal NET saved")
        
        nn_model.load_weights(self.file_model)

        print ('  FCNeuralNet trained, now make predictions & invert scaling')

        # Make predictions on the training and test datasets
        y_train_md=nn_model.predict(self.x_train)
        pred_train_set=pd.concat([self.train_set[self.id_col+
            self.sel_cols+self.y_cols],pd.DataFrame(y_train_md,
            columns=self.y_md_cols)],axis=1,ignore_index=True)
        pred_train_set.columns=self.id_col+self.sel_cols+self.y_cols+\
            self.y_md_cols
        unscaled_train_set=data_processor.invert_scale_y(pred_train_set,
            scaling_dic,'training')
        unscaled_train_set.to_csv('training.csv',index=False)

        if self.n_tests > 0:
            y_test_md=nn_model.predict(self.x_test)
            pred_test_set=pd.concat([self.test_set[self.id_col+
                self.sel_cols+self.y_cols],pd.DataFrame(y_test_md,
                columns=self.y_md_cols)],axis=1,ignore_index=True)
            pred_test_set.columns=self.id_col+self.sel_cols+\
                self.y_cols+self.y_md_cols
            unscaled_test_set=data_processor.invert_scale_y(pred_test_set,
                scaling_dic,'test')
            unscaled_test_set.to_csv('test.csv',index=False)

        print ('  Predictions made & saved in "training.csv" & "test.csv"')


class KRR:
    """ KRR: Kernel Ridge Regression with scikit-learn

    Parameters
    ----------
    model_params:    Dictionary, direct input, parameters of the net
    data_params:     Dictionary, obtained from ProcessData, all needed for data

    """

    def __init__(self,data_params,model_params):
        self.data_params=data_params
        self.model_params=model_params
        self.nfold_cv=self.model_params['nfold_cv']
        self.file_model=self.model_params['file_model']
        self.kernel=self.model_params['kernel']
        print (' ')
        print ('  Algorithm: kernel ridge regression w/ scikit-learn')

    def load_data(self):
        """ Import train/test data and parameters to FCNeuralNet """
        
        # Data preprocessing
        data_processor=ProcessData(data_params=self.data_params)
        data_dict=data_processor.load_data()

        self.id_col=data_dict['id_col']
        self.x_cols=data_dict['x_cols']
        self.y_cols=data_dict['y_cols']
        self.y_scaling=data_dict['y_scaling']
        self.y_mean=data_dict['y_mean']
        self.y_std=data_dict['y_std']
        self.y_max=data_dict['y_max']
        self.y_min=data_dict['y_min']
        self.y_org=data_dict['y_org']
        self.sel_cols=data_dict['sel_cols']
        self.y_md_cols=['md_'+col for col in self.y_cols]
        self.n_trains=data_dict['n_trains']
        self.n_tests=data_dict['n_tests']
        self.train_set=data_dict['train_set'].reset_index()
        self.test_set=data_dict['test_set'].reset_index()
        self.data_size=self.n_trains+self.n_tests
        self.x_dim=len(self.x_cols)
        self.y_dim=len(self.y_cols)
        self.x_train=np.array(self.train_set[self.x_cols]).astype(np.float32)
        self.y_train=np.array(self.train_set[self.y_cols]).astype(np.float32)
        self.x_test=np.array(self.test_set[self.x_cols]).astype(np.float32)

        if self.y_dim>1:
            raise ValueError \
                  ('      ERROR: KRR dont support multiple target learning')

    def train(self):
        from sklearn.model_selection import GridSearchCV
        from sklearn.kernel_ridge import KernelRidge

        self.load_data()
        data_processor=ProcessData(data_params=self.data_params)

        # A dictionary of scaling parameters
        scaling_dic={'id_col':self.id_col,'y_cols':self.y_cols,
            'sel_cols':self.sel_cols,'y_org':self.y_org,
            'y_scaling':self.y_scaling,'y_mean':self.y_mean,
            'y_std':self.y_std,'y_min':self.y_min,'y_max':self.y_max}

        print ('  Building model: KRR')
        param_grid={"alpha":np.logspace(-2,5,30),"gamma":np.logspace(-2,5,30)}

        kr=GridSearchCV(KernelRidge(kernel=self.kernel,gamma=0.5),cv=self.nfold_cv,
            param_grid=param_grid)

        kr.fit(self.x_train,self.y_train)

        print ('  KRR model trained, now make predictions & invert scaling')

        y_train_md=kr.predict(self.x_train)
        pred_train_set=pd.concat([self.train_set[self.id_col+
            self.sel_cols+self.y_cols],pd.DataFrame(y_train_md,
            columns=self.y_md_cols)],axis=1,ignore_index=True)
        pred_train_set.columns=self.id_col+self.sel_cols+self.y_cols+\
            self.y_md_cols
        unscaled_train_set=data_processor.invert_scale_y(pred_train_set,
            scaling_dic,'training')
        unscaled_train_set.to_csv('training.csv',index=False)
        if self.n_tests > 0:
            y_test_md=kr.predict(self.x_test)
            pred_test_set=pd.concat([self.test_set[self.id_col+
                self.sel_cols+self.y_cols],pd.DataFrame(y_test_md,
                columns=self.y_md_cols)],axis=1,ignore_index=True)
            pred_test_set.columns=self.id_col+self.sel_cols+self.y_cols+\
                self.y_md_cols
            unscaled_test_set=data_processor.invert_scale_y(pred_test_set,
                scaling_dic,'test')
            unscaled_test_set.to_csv('test.csv',index=False)

        print ('  Predictions made & saved in "training.csv" & "test.csv"')


class GPR:
    """ GPR: Gaussian Process Regression with scikit-learn

    Parameters
    ----------
    model_params:    Dictionary, direct input, parameters of the net
    data_params:     Dictionary, obtained from ProcessData, all needed for data

    """

    def __init__(self,data_params,model_params):
        self.data_params=data_params
        self.model_params=model_params
        self.nfold_cv=self.model_params['nfold_cv']
        self.file_model=self.model_params['file_model']
        self.n_restarts_optimizer=self.model_params['n_restarts_optimizer']
        print (' ')
        print ('  Algorithm: gaussian process regression w/ scikit-learn')

    def load_data(self):
        """ Import train/test data and parameters to FCNeuralNet """
        
        # Data preprocessing
        data_processor=ProcessData(data_params=self.data_params)
        data_dict=data_processor.load_data()
        
        self.id_col=data_dict['id_col']
        self.x_cols=data_dict['x_cols']
        self.y_cols=data_dict['y_cols']
        self.y_scaling=data_dict['y_scaling']
        self.file_model=self.model_params['file_model']
        self.y_mean=data_dict['y_mean']
        self.y_std=data_dict['y_std']
        self.y_max=data_dict['y_max']
        self.y_min=data_dict['y_min']
        self.y_org=data_dict['y_org']
        self.x_scaled=data_dict['x_scaled']
        self.y_scaled=data_dict['y_scaled']
        self.sel_cols=data_dict['sel_cols']
        self.y_md_cols=['md_'+col for col in self.y_cols]
        self.n_trains=data_dict['n_trains']
        self.n_tests=data_dict['n_tests']
        self.train_set=data_dict['train_set'].reset_index()
        self.test_set=data_dict['test_set'].reset_index()
        self.data_size=self.n_trains+self.n_tests
        self.x_dim=len(self.x_cols)
        self.y_dim=len(self.y_cols)
        self.x_train=np.array(self.train_set[self.x_cols]).astype(np.float32)
        self.y_train=np.array(self.train_set[self.y_cols]).astype(np.float32)
        self.x_test=np.array(self.test_set[self.x_cols]).astype(np.float32)

        if self.y_dim>1:
            raise ValueError \
                  ('      ERROR: GPR dont support multiple target learning')

    def train(self):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import WhiteKernel,RBF

        self.load_data()
        data_processor=ProcessData(data_params=self.data_params)

        # A dictionary of scaling parameters
        scaling_dic={'id_col':self.id_col,'y_cols':self.y_cols,
            'sel_cols':self.sel_cols,'y_org':self.y_org,
            'y_scaling':self.y_scaling,'y_mean':self.y_mean,
            'y_std':self.y_std,'y_min':self.y_min,'y_max':self.y_max}

        y_avr=np.average(np.array(self.y_scaled[self.y_cols]))
        noise_avr=np.std(np.array(self.y_scaled[self.y_cols]))
        noise_lb=(noise_avr)**2/8
        noise_ub=(noise_avr)**2*8

        kernel=(y_avr)**2*RBF(length_scale=1)+WhiteKernel(noise_level=\
            noise_avr**2,noise_level_bounds=(noise_lb,noise_ub))

        gp=GaussianProcessRegressor(kernel=kernel,alpha=1e-10, 
            n_restarts_optimizer=self.n_restarts_optimizer)

        opt_gp=gp
        opt_rmse=1.0E20
        ncv=0
        ncv_opt=ncv

        #kfold splitting
        kf_=KFold(n_splits=self.nfold_cv,shuffle=True)
        kf=kf_.split(self.train_set)

        tpl1=\
            "    cv,rmse_train,rmse_test,rmse_opt: {0:d} {1:.6f} {2:.6f} {3:.6f}"

        print ('  Training model w/ cross validation')
        for train_cv,test_cv in kf:

            x_cv_train,x_cv_test,y_cv_train,y_cv_test=data_processor\
                .get_cv_datasets(self.train_set,self.x_cols,self.y_cols,
                train_cv,test_cv)

            gp.fit(x_cv_train,y_cv_train)

            y_cv_train_md=gp.predict(x_cv_train,return_std=False)
            y_cv_test_md=gp.predict(x_cv_test,return_std=False)

            rmse_cv_train=np.sqrt(mean_squared_error(y_cv_train,y_cv_train_md))
            rmse_cv_test=np.sqrt(mean_squared_error(y_cv_test,y_cv_test_md))
            
            if rmse_cv_test < opt_rmse:
                opt_rmse=rmse_cv_test
                opt_gp=gp
                ncv_opt=ncv

            print (tpl1.format(ncv,rmse_cv_train,rmse_cv_test,opt_rmse))
            ncv = ncv+1

        # get optimal kernel, fit the whole training set
        opt_gp=opt_gp
        gp_final=GaussianProcessRegressor(kernel=opt_gp.kernel_,alpha=0,
            optimizer=None)
        gp_final.fit(self.x_train,self.y_train)

        # Save model
        file_model=os.path.join(os.getcwd(),self.file_model)
        joblib.dump(gp_final,file_model)

        print ('  GPR model trained, now make predictions & invert scaling')

        y_train_md= gp_final.predict(self.x_train,return_std=False)
        pred_train_set=pd.concat([self.train_set[self.id_col+
            self.sel_cols+self.y_cols],pd.DataFrame(y_train_md,
            columns=self.y_md_cols)],axis=1,ignore_index=True)
        pred_train_set.columns=self.id_col+self.sel_cols+self.y_cols+\
            self.y_md_cols
        unscaled_train_set=data_processor.invert_scale_y(pred_train_set,
            scaling_dic,'training')
        unscaled_train_set.to_csv('training.csv',index=False)

        if self.n_tests > 0:
            y_test_md= gp_final.predict(self.x_test,return_std=False)
            pred_test_set=pd.concat([self.test_set[self.id_col+
                self.sel_cols+self.y_cols],pd.DataFrame(y_test_md,
                columns=self.y_md_cols)],axis=1,ignore_index=True)
            pred_test_set.columns=self.id_col+self.sel_cols+self.y_cols+\
                self.y_md_cols
            unscaled_test_set=data_processor.invert_scale_y(pred_test_set,
                scaling_dic,'test')
            unscaled_test_set.to_csv('test.csv',index=False)

        print ('  Predictions made & saved in "training.csv" & "test.csv"')


class ProbNeuralNet:
    """ ProbNeuralNet: Probability NeuralNet with Tensorflow-Probability

    Parameters
    ----------
    model_params:    Dictionary, direct input, parameters of the net
    data_params:     Dictionary, obtained from ProcessData, all needed for data

    """

    def __init__(self,data_params,model_params):
        self.data_params=data_params
        self.model_params=model_params

        self.sampling=self.data_params['sampling']
        self.nfold_cv=self.model_params['nfold_cv']
        self.file_model=self.model_params['file_model']
        self.layers=self.model_params['layers']
        self.activ_funct=self.model_params['activ_funct']
        self.epochs=self.model_params['epochs']
        self.optimizer=self.model_params['optimizer']
        self.use_bias=self.model_params['use_bias']
        self.loss=self.model_params['loss']
        self.verbosity=self.model_params['verbosity']
        self.batch_size=self.model_params['batch_size']
        self.rmse_cv=self.model_params['rmse_cv']
        print (' ')
        print ('  Algorithm: Probabilistic NeuralNet w/ Tensorflow-Probability')

