import sys

class params():
    def __init__(self,param_file_name='matsml.in'):
        self.param_file_name = param_file_name


    def read_params(file_input):
      
        """
        Huan Tran, huan.tran@mse.gatech.edu
        Objective: Preparing data for training a ml model
        Inputs: X, y, ntrain, data_size, feature_dim, sampling
        Outputs: X_train, X_test, y_train, y_test
        """
        input_dict = dict()
        
        try:
            file = open(file_input, 'r')
            for line in file:
                if len(line) > 1 and line[0] != '#':
                    line_split = line.split('=')
                    if len(line_split) >= 2:
                        #print line_split[1].split()
                        var_name= line_split[0].strip('\n').strip('\t').strip(' ')
                        var_value = line_split[1].split('#')[0].strip('\n').strip('\t').strip(' ')
                        if var_name != '' and var_value != '':
                            input_dict[var_name] = var_value
            file.close()
        except:
            print ('  ! ERROR: ' + file_input + ' missing')
            sys.exit()
        
        # Required variables
        echo_param ('file_data', input_dict, 1,  '   ', True)
        if 'run_type' in input_dict:
            run_type = input_dict['run_type']
            print (' %s = %s' %(('   run_type').ljust(30),str(run_type).ljust(30)))
        else:
            print ('  ! Error: "run type" not defined or missing [run_type]')
            sys.exit()
      
        if run_type == 'learning_curve':
            echo_param ('ntrains_start', input_dict, 0.8, '   ', False)
            echo_param ('ntrains_stop', input_dict, 0.9, '   ', False)
            echo_param ('ntrains_incre', input_dict, 0.1, '   ', False)
        elif run_type == 'make_model':
            echo_param ('ntrains', input_dict, 200, '   ', False)
      
        echo_param ('number_runs', input_dict, 1, '   ', False)
        echo_param ('sampling', input_dict, 'random', '   ', False)
        echo_param ('nfold_cv', input_dict, 5, '   ', False)
        echo_param ('y_scale', input_dict, 'none', '   ', False)
        echo_param ('x_scale', input_dict, 'none', '   ', False)
        echo_param ('model_perf', input_dict, 'best_test', '   ', False)
        echo_param ('id_col', input_dict, 'ID', '   ', True)
        echo_param ('y_cols', input_dict, 'none', '   ', True)
        echo_param ('comment_cols', input_dict, 'none', '   ', False)
      
        if 'ml_algo' in input_dict:
            lalgo = len(input_dict['ml_algo'].split())
            algo = input_dict['ml_algo'].split()[0]
            print (' %s = %s' %(('   ml_algo').ljust(30),str(algo).ljust(30)))
            # Some algo can be called from different package
            if lalgo == 2 :
                lpackage = input_dict['ml_algo'].split()[1]
            print (' %s = %s' %(('   package used').ljust(30),str(lpackage).ljust(30)))
        else:
            print ('    ml_algo                = gpr   ! Warning: ML algorithm is not defined. Default setting applied.')
            input_dict['ml_algo'] = 'gpr'
        
        if algo == 'nn':
            echo_param ('nlayers', input_dict, 8, '   ', False)
            echo_param ('nneurons', input_dict, 8, '   ', False)
            echo_param ('activ_func', input_dict, 'relu', '   ', False)
            echo_param ('loss', input_dict, 'mse', '   ', False)
            echo_param ('metrics', input_dict, 'mse', '   ', False)
            echo_param ('epochs', input_dict, 1000, '   ', False)
            echo_param ('optimizer', input_dict, 'nadam', '   ', False)
            echo_param ('batch_size', input_dict, 32, '   ', False)
            echo_param ('use_bias', input_dict, 'True', '   ', False)
            echo_param ('verbosity', input_dict, 0, '   ', False)
            echo_param ('func_learn', input_dict, 0, '   ', False)
        elif algo == 'fnn':
            echo_param ('nlayers', input_dict, 8, '    ', False)
            echo_param ('nneurons', input_dict, 8, '    ', False)
            echo_param ('activ_func', input_dict, 'relu', '    ', False)
            echo_param ('epochs', input_dict, 1000, '    ', False)
            echo_param ('optimizer', input_dict, 'nadam', '    ', False)
            echo_param ('batch_size', input_dict, 32, '    ', False)
            echo_param ('use_bias', input_dict, 'True', '    ', False)
            echo_param ('fbasis_dim', input_dict, 0, '    ', False)
            echo_param ('verbosity', input_dict, 0, '    ', False)
            echo_param ('func_learn', input_dict, 0, '    ', False)
            echo_param ('xstart', input_dict, 0, '    ', False)
            echo_param ('xend', input_dict, 4, '    ', False)
            echo_param ('x_in', input_dict, 0, '    ', False)
            echo_param ('decom_train', input_dict, 'True', '    ', False)
        elif algo == 'svr':
            if 'kernel' in input_dict:
                print ('    - kernel               =', input_dict['kernel'])
            else:
                input_dict['kernel']   = 'rbf'
                print ('    - kernel               =', input_dict['kernel'])
        print (' ')
        return input_dict


    def echo_param(key, input_dict, default_value, indent, required):
        if key in input_dict:
            print (' %s = %s' %((indent+key).ljust(30),input_dict[key].ljust(30)))
        else:
            if required:
                print ('     > ERROR: data file not defined or missing [%s]'%(key.ljust(20)))
                sys.exit()
            else:
                input_dict[key] = default_value
                print ('     > WARNING: data file not defined or missing [%s], default value applied'%(key.ljust(20)))

