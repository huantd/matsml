# Copyright Huan Tran (huantd@gmail.com), 2021
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import binned_statistic_2d
from sklearn import preprocessing
from scipy.special import comb

def scale_x(X, x_scale):
    '''
    Objective:  Scale X
    Inputs:     X, x_scale
    Outputs:    X_scale
    '''
 
    if x_scale == 'log':
        X_scaled = X
        X[:,:] = np.log(X[:,:])
        xmax = np.amax(np.array(X))
        xmin = np.amin(np.array(X))
        X_scaled[:,:] = -1 + 2*(X[:,:] - xmin)/(xmax - xmin) 
    elif x_scale == 'normalize':
        X_scaled = preprocessing.normalize(X, norm='l2')
    elif x_scale == 'minmax':
        min_max_scaler = preprocessing.MinMaxScaler()
        X_scaled = min_max_scaler.fit_transform(X)
    elif x_scale == 'quantile':
        quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
        X_scaled = quantile_transformer.fit_transform(X)
    elif x_scale == 'yeo-johnson':
        pt = preprocessing.PowerTransformer(method='yeo-johnson', standardize=False)
        X_scaled = pt.fit_transform(X)
    elif x_scale == 'none':
        X_scaled = X
 
    return X_scaled


def scale_y(y, y_dim, data_size, y_scale):
    ''' 
        Scale y data. Currently y data can be scalled in 'minmax' and 'normal' 
        modes. The y data scaling needs to be inverted after being learned and
        predicted. See 'inverse_scaling_y' below.
    '''
    import numpy as np

    y_mean = np.array([np.mean(y[:,i]) for i in range(y_dim)])
    y_std = np.array([np.std(y[:,i]) for i in range(y_dim)])
    y_min = np.array([np.amin(np.array(y[:,i])) for i in range(y_dim)])
    y_max = np.array([np.amax(np.array(y[:,i])) for i in range(y_dim)])

    for i, j in ((a,b) for a in range(data_size) for b in range(y_dim)):
        if str(y_scale) == 'normal':
            y[i,j] = (y[i,j]-y_mean[j])/y_std[j]
        elif str(y_scale) == 'minmax':
            y[i,j] = (y[i,j]-y_min[j])/(y_max[j]-y_min[j])

    return y, y_mean, y_std, y_min, y_max


def inv_y_scaling(y,y_dim,data_size,y_scale,y_mean,y_std,y_min,y_max):
    ''' Invert the scaling of the y data   '''
    
    y_inv = np.zeros((data_size, y_dim))
    for i, j in ((a,b) for a in range(data_size) for b in range(y_dim)):
        if str(y_scale) == 'normal':
            y_inv[i,j] = (y[i,j]*y_std [j])+y_mean[j]
        elif str(y_scale) == 'minmax':
            y_inv[i,j] = y[i,j]*(y_max[j]-y_min[j])+y_min[j]
        elif str(y_scale) == 'none':
            y_inv[i,j] = y[i,j]
    return y_inv


def prep_kfold(X, y, nfold, sampling):
  """
  Written by: Huan Tran, huantd@gmail.com
  Objective:  Preparing data for k-fold cross validation
  Inputs:     X, y, nfold, sampling
  Outputs:    
  """

def prep_train_data(X, y, ntrain, data_size, x_dim, y_dim, sampling):
  """
  Objective:  Preparing data for training a ml model
  Inputs:     X, y, ntrain, data_size, x_dim, y_dim, sampling
  Outputs:    X_train, X_test, y_train, y_test
  """

  if sampling == 'random':
      print ('        random sampling for train/test partition')
      idx_all = list(range(data_size))
      np.random.shuffle(idx_all)
      idx_train = idx_all[:ntrain]
      idx_test = idx_all[ntrain:]

      xtrain_list = []
      ytrain_list = []
      for i in range(min(ntrain,data_size)):
          xtrain_list.append(X[idx_train[i],:])
          ytrain_list.append(y[idx_train[i], :])
      X_train = np.array(xtrain_list).reshape(ntrain, x_dim)
      y_train = np.array(ytrain_list).reshape(ntrain, y_dim)

      ntest = data_size - ntrain
      xtest_list = []
      ytest_list = []
      j = 0
      for i in range(ntrain+1, data_size+1, 1):
          xtest_list.append(X[idx_test[j],:])
          ytest_list.append(y[idx_test[j],:])
          j += 1
      X_test = np.array(xtest_list).reshape(ntest, x_dim)
      y_test = np.array(ytest_list).reshape(ntest, y_dim)

      print ('        DONE: random sampling for train/test partition')

  return X_train, X_test, y_train, y_test
