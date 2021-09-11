"""
This module contains the reguler ML models supported by matsML. More to come.
"""
import os,joblib,warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from matsml.data import ProcessData
from matsml.io import goodbye
from matsml.io import plot_det_preds, plot_prob_preds

import tensorflow as tf
from keras.layers import Dense,InputLayer
from keras.models import Sequential
from keras.optimizers import Nadam

import tensorflow_probability as tfp

from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel,RBF


class FCNN:
    """ 
    FCNN: A generic fully-connected Neuplot_train_testralNet with TensorFlow

    model_params: Dictionary, direct input, parameters of the net
    data_params:  Dictionary, obtained from ProcessData, all needed for data

    """

    def __init__(self,data_params,model_params):

        self.data_params=data_params
        self.model_params=model_params

        self.nfold_cv=self.model_params['nfold_cv']
        self.model_file=self.model_params['model_file']
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
        print ('  Learning fingerprinted/featured data')
        print ('    algorithm'.ljust(32),'fully connected NeuralNet w/ TensorFlow')
        print ('    layers'.ljust(32),self.layers)
        print ('    activ_funct'.ljust(32),self.activ_funct)
        print ('    epochs'.ljust(32),self.epochs)
        print ('    optimizer'.ljust(32),self.optimizer)
        print ('    nfold_cv'.ljust(32),self.nfold_cv)


    def load_data(self):
        """ Import train/test data and parameters to FCNN """
        
        # Data preprocessing
        data_processor=ProcessData(data_params=self.data_params)
        self.data_dict=data_processor.load_data()
        
        self.id_col=self.data_dict['id_col']
        self.x_cols=self.data_dict['x_cols']
        self.y_cols=self.data_dict['y_cols']
        self.y_org=self.data_dict['y_org']
        self.sel_cols=self.data_dict['sel_cols']
        self.y_md_cols=self.data_dict['y_md_cols']
        self.n_tests=self.data_dict['n_tests']
        self.train_set=self.data_dict['train_set'].reset_index()
        self.test_set=self.data_dict['test_set'].reset_index()
        self.x_dim=len(self.x_cols)
        self.y_dim=len(self.y_cols)
        self.x_train=np.array(self.train_set[self.x_cols]).astype(np.float32)
        self.x_test=np.array(self.test_set[self.x_cols]).astype(np.float32)

        # NN does not have yerr
        self.data_dict['yerr_md_cols']=[] 


    def build_model(self):
        """ Build a FCNN"""

        print ('  Building model'.ljust(32),'FCNN')
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

        return nn_model


    def train(self):
        """ Train, test, and save the FCNN"""

        self.load_data()
        data_processor=ProcessData(data_params=self.data_params)

        tpl1=\
            "    cv,rmse_train,rmse_test,rmse_opt: {0:d} {1:.6f} {2:.6f} {3:.6f}"

        nn_model=self.build_model()

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
                nn_model.save_weights(self.model_file)
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
                    self.data_dict,'cv_test')

        print('    Optimal ncv: ',ncv_opt,"; optimal NET saved")
        
        nn_model.load_weights(self.model_file)

        print ('  FCNN trained, now make predictions & invert scaling')

        # Make predictions on the training and test datasets
        y_train_md=nn_model.predict(self.x_train)
        pred_train_set=pd.concat([self.train_set[self.id_col+
            self.sel_cols+self.y_cols],pd.DataFrame(y_train_md,
            columns=self.y_md_cols)],axis=1,ignore_index=True)
        pred_train_set.columns=self.id_col+self.sel_cols+self.y_cols+\
            self.y_md_cols
        unscaled_train_set=data_processor.invert_scale_y(pred_train_set,
            self.data_dict,'training')
        unscaled_train_set.to_csv('training.csv',index=False)

        if self.n_tests > 0:
            y_test_md=nn_model.predict(self.x_test)
            pred_test_set=pd.concat([self.test_set[self.id_col+
                self.sel_cols+self.y_cols],pd.DataFrame(y_test_md,
                columns=self.y_md_cols)],axis=1,ignore_index=True)
            pred_test_set.columns=self.id_col+self.sel_cols+\
                self.y_cols+self.y_md_cols
            unscaled_test_set=data_processor.invert_scale_y(pred_test_set,
                self.data_dict,'test')
            unscaled_test_set.to_csv('test.csv',index=False)

        print ('  Predictions made & saved in "training.csv" & "test.csv"')


    def plot(self,pdf_output):
        """ Plot the train and test predictions"""

        plot_det_preds(self.y_cols,self.y_md_cols,pdf_output)


class ProbFCNN:
    """ 
    ProbFCNN: Probability NeuralNet with Tensorflow-Probability

    model_params: Dictionary, direct input, parameters of the net
    data_params:  Dictionary, obtained from ProcessData, all needed for data

    """


    def __init__(self,data_params,model_params):

        self.data_params=data_params
        self.model_params=model_params

        self.nfold_cv=self.model_params['nfold_cv']
        self.model_file=self.model_params['model_file']
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
        print ('  Learning fingerprinted/featured data')
        print ('    algorithm'.ljust(32),'Probabilistic NeuralNet w/ TensorFlow-Probability')
        print ('    layers'.ljust(32),self.layers)
        print ('    activ_funct'.ljust(32),self.activ_funct)
        print ('    epochs'.ljust(32),self.epochs)
        print ('    optimizer'.ljust(32),self.optimizer)
        print ('    nfold_cv'.ljust(32),self.nfold_cv)

        # Check parameters
        self.check_params()
     

    def load_data(self):
        """ Import train/test data and parameters to FCNN """
        
        # Data preprocessing
        data_processor=ProcessData(data_params=self.data_params)
        self.data_dict=data_processor.load_data()
        
        self.id_col=self.data_dict['id_col']
        self.x_cols=self.data_dict['x_cols']
        self.y_cols=self.data_dict['y_cols']
        self.y_org=self.data_dict['y_org']
        self.sel_cols=self.data_dict['sel_cols']
        self.y_md_cols=self.data_dict['y_md_cols']
        self.yerr_md_cols=self.data_dict['yerr_md_cols']
        self.n_tests=self.data_dict['n_tests']
        self.train_set=self.data_dict['train_set'].reset_index()
        self.test_set=self.data_dict['test_set'].reset_index()
        self.x_dim=len(self.x_cols)
        self.y_dim=len(self.y_cols)
        self.x_train=np.array(self.train_set[self.x_cols]).astype(np.float32)
        self.x_test=np.array(self.test_set[self.x_cols]).astype(np.float32)

    def build_model(self):
        """ Build a ProbFCNN"""

        # TFP distributions
        tfd=tfp.distributions

        #Negative log likehood
        negloglik=lambda y, rv_y: -rv_y.log_prob(y)

        print ('  Building model'.ljust(32),'ProbFCNN')

        nn_model=Sequential()

        # input layer
        nn_model.add(InputLayer(input_shape=(self.x_dim,)))

        # hidden layers
        for layer in self.layers:
            nn_model.add(Dense(layer,kernel_initializer='normal', 
                activation=self.activ_funct,use_bias=self.use_bias))

        # probabilistic layer
        nn_model.add(Dense(1+1))
        nn_model.add(tfp.layers.DistributionLambda(lambda t: tfd.Normal\
            (loc=t[..., :1],scale=1e-1+tf.math.softplus(0.1*t[...,1:]))))

        # Compile
        nn_model.compile(loss=negloglik,optimizer=tf.optimizers\
            .Adam(learning_rate=0.01))

        # Model summary
        if int(self.verbosity) > 0:
            print(nn_model.summary())

        return nn_model

    def check_params(self):
        """ 
        Make sure the parameters are valid 
        """

        print ('  Checking parameters')
        if self.data_params['y_scaling']!='none': 
            raise ValueError \
                  ('      ERROR: No y scaling with ProbFCNN')
        if len(self.data_params['y_cols'])>1: 
            raise ValueError \
                  ('      ERROR: No more than 1 targets with ProbFCNN')

    def train(self):
        """ Train, test, and save the ProbFCNN"""

        self.load_data()
        data_processor=ProcessData(data_params=self.data_params)

        tpl1=\
            "    cv,rmse_train,rmse_test,rmse_opt: {0:d} {1:.6f} {2:.6f} {3:.6f}"

        nn_model=self.build_model()

        #kfold splitting
        kf_=KFold(n_splits=self.nfold_cv,shuffle=True)
        kf=kf_.split(self.train_set)

        opt_rmse=1.0E20
        ncv=0
        ncv_opt=ncv
        
        print ('  Training ProbFCNN w/ cross validation')

        for train_cv,test_cv in kf:
            x_cv_train,x_cv_test,y_cv_train,y_cv_test=data_processor\
                .get_cv_datasets(self.train_set,self.x_cols,self.y_cols,
                train_cv,test_cv)

            nn_model.fit(x_cv_train,y_cv_train,epochs=self.epochs,
                batch_size=int(self.batch_size),verbose=int(self.verbosity))

            y_cv_train_md=nn_model.predict(x_cv_train).flatten()
            y_cv_test_md=nn_model.predict(x_cv_test).flatten()

            rmse_cv_train=np.sqrt(mean_squared_error(np.array(y_cv_train)\
                .reshape(len(train_cv)*self.y_dim),y_cv_train_md))
            rmse_cv_test=np.sqrt(mean_squared_error(np.array(y_cv_test)\
                .reshape(len(test_cv)*self.y_dim),y_cv_test_md))

            if rmse_cv_test<opt_rmse:
                opt_rmse=rmse_cv_test
                nn_model.save_weights(self.model_file)
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
                    self.data_dict,'cv_test')

        print('    Optimal ncv: ',ncv_opt)
        
        nn_model.load_weights(self.model_file)

        print ('  ProbFCNN trained, now make predictions & invert scaling')

        # Make predictions on the training and test datasets
        y_train_md=np.array(nn_model(self.x_train).mean())
        yerr_train_md=np.array(nn_model(self.x_train).stddev())

        pred_train_set=pd.concat([self.train_set[self.id_col+
            self.sel_cols+self.y_cols],pd.DataFrame(y_train_md,
            columns=self.y_md_cols),pd.DataFrame(yerr_train_md,
            columns=self.yerr_md_cols)],axis=1,ignore_index=True)
        pred_train_set.columns=self.id_col+self.sel_cols+self.y_cols+\
            self.y_md_cols+self.yerr_md_cols
        unscaled_train_set=data_processor.invert_scale_y(pred_train_set,
            self.data_dict,'training')
        unscaled_train_set.to_csv('training.csv',index=False)

        if self.n_tests > 0:
            y_test_md=np.array(nn_model(self.x_test).mean())
            yerr_test_md=np.array(nn_model(self.x_test).stddev())

            pred_test_set=pd.concat([self.test_set[self.id_col+
                self.sel_cols+self.y_cols],pd.DataFrame(y_test_md,
                columns=self.y_md_cols),pd.DataFrame(yerr_test_md,
                columns=self.yerr_md_cols)],axis=1,ignore_index=True)
            pred_test_set.columns=self.id_col+self.sel_cols+\
                self.y_cols+self.y_md_cols+self.yerr_md_cols
            unscaled_test_set=data_processor.invert_scale_y(pred_test_set,
                self.data_dict,'test')
            unscaled_test_set.to_csv('test.csv',index=False)

        print ('  Predictions made & saved in "training.csv" & "test.csv"')


    def predict(self,predict_params):
        """ Load the FCNN that was trained and saved, and use it to make 
            predictions

        """
        import joblib

        self.load_data()
        nn_model=self.build_model()
        nn_model.load_weights(self.model_file)

        data_fp=pd.read_csv(predict_params['data_file'],delimiter=',',header=0,
            low_memory=False)

        x_cols=[col for col in list(data_fp.columns) if col not in
            (predict_params['y_cols']+predict_params['id_col']+\
            predict_params['comment_cols'])]

        xscaler=joblib.load('xscaler.pkl') 
        x=xscaler.transform(data_fp[x_cols])
        y=nn_model.predict(x)

        print (y)

    def plot(self,pdf_output):
        """ 
        Plot the train and test predictions
        """

        plot_prob_preds(self.y_cols,self.y_md_cols,self.yerr_md_cols,\
            pdf_output)



class KRR:
    """ 
    KRR: Kernel Ridge Regression with scikit-learn

    model_params: Dictionary, direct input, parameters of the net
    data_params:  Dictionary, obtained from ProcessData, all needed for data

    """


    def __init__(self,data_params,model_params):
        self.data_params=data_params
        self.model_params=model_params
        self.kernel=self.model_params['kernel']
        self.nfold_cv=self.model_params['nfold_cv']
        self.alpha=self.model_params['alpha']
        self.gamma=self.model_params['gamma']
        self.n_grids=self.model_params['n_grids']
        self.model_file=self.model_params['model_file']
        print (' ')
        print ('  Learning fingerprinted/featured data')
        print ('    algorithm'.ljust(32),'kernel ridge regression w/ scikit-learn')
        print ('    kernel'.ljust(32),self.kernel)
        print ('    nfold_cv'.ljust(32),self.nfold_cv)
        print ('    alpha'.ljust(32),self.alpha)
        print ('    gamma'.ljust(32),self.gamma)
        print ('    number of alpha/gamma grids'.ljust(32),self.n_grids)

        # Check parameters
        self.check_params()

    def check_params(self):
        """ Make sure the parameters are valid """

        print ('  Checking parameters')
        if len(self.data_params['y_cols'])>1: 
            raise ValueError \
                  ('      ERROR: No more than 1 targets with KRR')

    def load_data(self):
        """ Import train/test data and parameters to FCNN """
        
        # Data preprocessing
        data_processor=ProcessData(data_params=self.data_params)
        self.data_dict=data_processor.load_data()

        self.id_col=self.data_dict['id_col']
        self.x_cols=self.data_dict['x_cols']
        self.y_cols=self.data_dict['y_cols']
        self.y_org=self.data_dict['y_org']
        self.sel_cols=self.data_dict['sel_cols']
        self.y_md_cols=self.data_dict['y_md_cols']
        self.n_tests=self.data_dict['n_tests']
        self.train_set=self.data_dict['train_set'].reset_index()
        self.test_set=self.data_dict['test_set'].reset_index()
        self.x_dim=len(self.x_cols)
        self.y_dim=len(self.y_cols)
        self.x_train=np.array(self.train_set[self.x_cols]).astype(np.float32)
        self.y_train=np.array(self.train_set[self.y_cols]).astype(np.float32)
        self.x_test=np.array(self.test_set[self.x_cols]).astype(np.float32)

        # KRR does not have yerr
        self.data_dict['yerr_md_cols']=[] 

    def train(self):
        self.load_data()
        data_processor=ProcessData(data_params=self.data_params)

        print ('  Building model'.ljust(32),'KRR')
        param_grid={"alpha":np.logspace(self.alpha[0],self.alpha[0],self.n_grids),
            "gamma":np.logspace(self.gamma[0],self.gamma[0],self.n_grids)}

        kr=GridSearchCV(KernelRidge(kernel=self.kernel,gamma=0.5),cv=self.nfold_cv,
            param_grid=param_grid)

        print ('  Training model w/ cross validation')
        kr.fit(self.x_train,self.y_train)
        print ('  KRR model trained, now make predictions & invert scaling')

        y_train_md=kr.predict(self.x_train)
        pred_train_set=pd.concat([self.train_set[self.id_col+
            self.sel_cols+self.y_cols],pd.DataFrame(y_train_md,
            columns=self.y_md_cols)],axis=1,ignore_index=True)
        pred_train_set.columns=self.id_col+self.sel_cols+self.y_cols+\
            self.y_md_cols
        unscaled_train_set=data_processor.invert_scale_y(pred_train_set,
            self.data_dict,'training')
        unscaled_train_set.to_csv('training.csv',index=False)
        if self.n_tests > 0:
            y_test_md=kr.predict(self.x_test)
            pred_test_set=pd.concat([self.test_set[self.id_col+
                self.sel_cols+self.y_cols],pd.DataFrame(y_test_md,
                columns=self.y_md_cols)],axis=1,ignore_index=True)
            pred_test_set.columns=self.id_col+self.sel_cols+self.y_cols+\
                self.y_md_cols
            unscaled_test_set=data_processor.invert_scale_y(pred_test_set,
                self.data_dict,'test')
            unscaled_test_set.to_csv('test.csv',index=False)

        print ('  Predictions made & saved in "training.csv" & "test.csv"')

    def plot(self,pdf_output):
        """ Plot the train and test predictions"""

        plot_det_preds(self.y_cols,self.y_md_cols,pdf_output)


class GPR:
    """ 
    GPR: Gaussian Process Regression with scikit-learn

    model_params: Dictionary, direct input, parameters of the net
    data_params:  Dictionary, obtained from ProcessData, all needed for data

    """


    def __init__(self,data_params,model_params):

        self.data_params=data_params
        self.model_params=model_params
        self.nfold_cv=self.model_params['nfold_cv']
        self.model_file=self.model_params['model_file']
        self.n_restarts_optimizer=self.model_params['n_restarts_optimizer']
        self.rmse_cv=self.model_params['rmse_cv']
        self.optimizer='fmin_l_bfgs_b'
        print (' ')
        print ('  Learning fingerprinted/featured data')
        print ('    algorithm'.ljust(32),'gaussian process regression w/ scikit-learn')
        print ('    nfold_cv'.ljust(32),self.nfold_cv)

        # Check parameter
        self.check_params()

    def load_data(self):
        """ Import train/test data and parameters to FCNN """
        
        # Data preprocessing
        data_processor=ProcessData(data_params=self.data_params)
        self.data_dict=data_processor.load_data()
        
        self.id_col=self.data_dict['id_col']
        self.x_cols=self.data_dict['x_cols']
        self.y_cols=self.data_dict['y_cols']
        self.y_org=self.data_dict['y_org']
        self.x_scaled=self.data_dict['x_scaled']
        self.y_scaled=self.data_dict['y_scaled']
        self.sel_cols=self.data_dict['sel_cols']
        self.y_md_cols=self.data_dict['y_md_cols']
        self.n_tests=self.data_dict['n_tests']
        self.train_set=self.data_dict['train_set'].reset_index()
        self.test_set=self.data_dict['test_set'].reset_index()
        self.x_dim=len(self.x_cols)
        self.y_dim=len(self.y_cols)
        self.x_train=np.array(self.train_set[self.x_cols]).astype(np.float32)
        self.y_train=np.array(self.train_set[self.y_cols]).astype(np.float32)
        self.x_test=np.array(self.test_set[self.x_cols]).astype(np.float32)

        # GPR should have yerr, but not now
        self.data_dict['yerr_md_cols']=[] 

    def check_params(self):
        """ 
        Make sure the parameters are valid 
        """

        print ('  Checking parameters')
        if len(self.data_params['y_cols'])>1: 
            raise ValueError \
                  ('      ERROR: No more than 1 targets with GPR')

    def train(self):

        self.load_data()
        data_processor=ProcessData(data_params=self.data_params)

        y_avr=np.average(np.array(self.y_scaled[self.y_cols]))
        noise_avr=np.std(np.array(self.y_scaled[self.y_cols]))
        noise_lb=(noise_avr)**2/200
        noise_ub=(noise_avr)**2*20

        kernel=(y_avr)**2*RBF(length_scale=1)+WhiteKernel(noise_level=\
            noise_avr**2,noise_level_bounds=(noise_lb,noise_ub))

        gp=GaussianProcessRegressor(kernel=kernel,alpha=1e-10,optimizer=\
            self.optimizer,n_restarts_optimizer=self.n_restarts_optimizer)

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

            # run block of code and catch warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
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

            if self.rmse_cv:
                y_cv_test_md=gp.predict(x_cv_test,return_std=False)
                pred_cv_test=pd.concat([self.train_set.iloc[test_cv]\
                    [self.id_col+self.sel_cols+self.y_cols]\
                    .reset_index(),pd.DataFrame(y_cv_test_md,
                    columns=self.y_md_cols)],axis=1,ignore_index=True)
                pred_cv_test.columns=['index']+self.id_col+\
                    self.sel_cols+self.y_cols+self.y_md_cols
                unscaled_cv_test=data_processor.invert_scale_y(pred_cv_test,
                    self.data_dict,'cv_test')

        # get optimal kernel, fit the whole training set
        opt_gp=opt_gp
        gp_final=GaussianProcessRegressor(kernel=opt_gp.kernel_,alpha=0,
            optimizer=None)
        gp_final.fit(self.x_train,self.y_train)

        # Save model
        model_file=os.path.join(os.getcwd(),self.model_file)
        joblib.dump(gp_final,model_file)

        print ('  GPR model trained, now make predictions & invert scaling')

        y_train_md= gp_final.predict(self.x_train,return_std=False)
        pred_train_set=pd.concat([self.train_set[self.id_col+
            self.sel_cols+self.y_cols],pd.DataFrame(y_train_md,
            columns=self.y_md_cols)],axis=1,ignore_index=True)
        pred_train_set.columns=self.id_col+self.sel_cols+self.y_cols+\
            self.y_md_cols
        unscaled_train_set=data_processor.invert_scale_y(pred_train_set,
            self.data_dict,'training')
        unscaled_train_set.to_csv('training.csv',index=False)

        if self.n_tests > 0:
            y_test_md=gp_final.predict(self.x_test,return_std=False)
            pred_test_set=pd.concat([self.test_set[self.id_col+
                self.sel_cols+self.y_cols],pd.DataFrame(y_test_md,
                columns=self.y_md_cols)],axis=1,ignore_index=True)
            pred_test_set.columns=self.id_col+self.sel_cols+self.y_cols+\
                self.y_md_cols
            unscaled_test_set=data_processor.invert_scale_y(pred_test_set,
                self.data_dict,'test')
            unscaled_test_set.to_csv('test.csv',index=False)

        print ('  Predictions made & saved in "training.csv" & "test.csv"')

    def plot(self,pdf_output):
        """ 
        Plot the train and test predictions
        """

        plot_det_preds(self.y_cols,self.y_md_cols,pdf_output)


