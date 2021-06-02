from utils_git.utils import *
# from KF_QN_CNN.utils_plot import *

def train_model(home_path = '/home/jupyter/',
                dataset_name = 'CIFAR-10',
#                 model_name = 'All-CNN-C',
                algorithm = 'KF-BFGS-CNN',
                lr = 0.3,
                lambda_damping = 0.3,
                Adam_epsilon = 1e-4,
                max_cpu_time = 2000):

    args = {}
    
    
    
    args['list_lr'] = [lr]
    
    args['weight_decay'] = 0
    print('need to change')
    
    args['momentum_gradient_dampening'] = 0
    print('need to change, especially for adam')
    
    if dataset_name == 'CIFAR-10':
        args['dataset'] = 'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias-no-regularization'
    else:
    
    
        print('dataset_name')
        print(dataset_name)

        sys.exit()
    
    
    
#     if dataset_name == 'CIFAR-10' and model_name == 'All-CNN-C':
        
#         args['dataset'] = 'CIFAR-10-AllCNNC'
#     elif dataset_name == 'CIFAR-10' and model_name == 'VGG16':
        
#         args['dataset'] = 'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout'
#     elif dataset_name == 'CIFAR-100' and model_name == 'All-CNN-C':
        
#         args['dataset'] = 'CIFAR-100-onTheFly-AllCNNC'
#     elif dataset_name == 'CIFAR-100' and model_name == 'VGG16':
        
#         args['dataset'] = 'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout'
#     else:
#         print('dataset_name')
#         print(dataset_name)

#         print('model_name')
#         print(model_name)

#         sys.exit()
    
    
    
    
    
    
    if algorithm == 'SGD-momentum':
        args['algorithm'] = 'SGD-momentum'
    elif algorithm == 'Adam':
        
        args['algorithm'] = 'Adam-noWarmStart-momentum-grad'
        args['RMSprop_epsilon'] = Adam_epsilon
        args['RMSprop_beta_2'] = 0.9
    elif algorithm == 'KFAC':
        
        args['algorithm'] = 'kfac-correctFisher-warmStart-lessInverse-no-max-no-LM-momentum-grad'
        
        args['kfac_if_svd'] = True
        print('need to change for autoencoder')
        
        args['kfac_if_update_BN'] = True
        args['kfac_if_BN_grad_direction'] = True
        
        args['kfac_cov_update_freq'] = 1
        args['kfac_inverse_update_freq'] = 20
        print('need to change for CNN')
        
        
        args['kfac_rho'] = 0.9
        args['kfac_damping_lambda'] = lambda_damping
    elif algorithm == 'KF-BFGS-CNN':
        
        args['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-regularized-grad-momentum-grad'
        args['Kron_LBFGS_Hg_initial'] = 1
        args['Kron_BFGS_A_LM_epsilon'] = np.sqrt(lambda_damping)
        args['Kron_BFGS_H_epsilon']= np.sqrt(lambda_damping)
    elif algorithm == 'KF-BFGS(L)-CNN':
        
        args['algorithm'] = 'Kron-BFGS(L)-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-regularized-grad-momentum-grad'
        args['Kron_LBFGS_Hg_initial'] = 1
        args['Kron_BFGS_number_s_y'] = 100
        args['Kron_BFGS_A_LM_epsilon'] = np.sqrt(lambda_damping)
        args['Kron_BFGS_H_epsilon']= np.sqrt(lambda_damping)
    else:
        print('algorithm')
        print(algorithm)
        sys.exit()
        

   
    
    
    

    # args['if_max_epoch'] = 1 # 0 means max_time
    args['if_max_epoch'] = 0 # 0 means max_time
    
    
    
    args['max_epoch/time'] = max_cpu_time
    args['record_epoch'] = 1

    args['seed_number'] = 9999

    args['num_threads'] = 8

    # args['initialization_pkg'] = 'numpy'
    # args['initialization_pkg'] = 'default'
    args['initialization_pkg'] = 'normal'
    
    
    
    args['home_path'] = home_path

    

    args['if_gpu'] = True
    # args['if_gpu'] = False
    
    

    args['if_test_mode'] = False
    # args['if_test_mode'] = True
    
    

    args['if_auto_tune_lr'] = False
#     args['if_auto_tune_lr'] = True

    args['if_grafting'] = False

    _ = tune_lr(args)





    return


def plot_results(
    home_path = '/home/jupyter/',
    dataset_name = 'CIFAR-10',
    model_name = 'VGG16',
    algorithms_jupyter = [],
):
    
    
    
    args = {}
    
    args['home_path'] = home_path
    
    

    args['if_gpu'] = True
    # args['if_gpu'] = False
    
    if dataset_name == 'CIFAR-10' and model_name == 'All-CNN-C':
        
        name_dataset = 'CIFAR-10-AllCNNC'
    elif dataset_name == 'CIFAR-10' and model_name == 'VGG16':
        
        name_dataset = 'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout'
    elif dataset_name == 'CIFAR-100' and model_name == 'All-CNN-C':
        
        name_dataset = 'CIFAR-100-onTheFly-AllCNNC'
    elif dataset_name == 'CIFAR-100' and model_name == 'VGG16':
        
        name_dataset = 'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout'
    else:
        print('dataset_name')
        print(dataset_name)

        print('model_name')
        print(model_name)

        sys.exit()



    name_dataset_legend = name_dataset
    
    

    algorithms = []
    
    for algorithm_jupyter in algorithms_jupyter:
        
        
        
        if algorithm_jupyter['name'] == 'KF-BFGS-CNN':
            
            lambda_ = algorithm_jupyter['lambda_damping']
            algorithm = {}
            algorithm['name'] = 'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-regularized-grad-momentum-grad'
            algorithm['params'] = {}
            algorithm['params']['Kron_BFGS_A_LM_epsilon'] = np.sqrt(lambda_)
            algorithm['params']['Kron_BFGS_H_epsilon'] = np.sqrt(lambda_)
            algorithm['legend'] = algorithm_jupyter['name']
            algorithms.append(copy.deepcopy(algorithm))
            
        elif algorithm_jupyter['name'] == 'KF-BFGS(L)-CNN':
            
            lambda_ = algorithm_jupyter['lambda_damping']
            algorithm = {}
            algorithm['name'] = 'Kron-BFGS(L)-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-regularized-grad-momentum-grad'
            algorithm['params'] = {}
            algorithm['params']['Kron_BFGS_A_LM_epsilon'] = np.sqrt(lambda_)
            algorithm['params']['Kron_BFGS_H_epsilon'] = np.sqrt(lambda_)
            algorithm['legend'] = algorithm_jupyter['name']
            algorithms.append(copy.deepcopy(algorithm))
            
        elif algorithm_jupyter['name'] == 'KFAC':
            
            algorithm = {}
            algorithm['name'] = 'kfac-warmStart-lessInverse-no-max-no-LM-momentum-grad'
            algorithm['params'] = {}
            algorithm['params']['kfac_damping_lambda'] = algorithm_jupyter['lambda_damping']
            algorithm['legend'] = 'KFAC'
            algorithms.append(copy.deepcopy(algorithm))
            
        elif algorithm_jupyter['name'] == 'Adam':
            
#             print('algorithm_jupyter')
#             print(algorithm_jupyter)
            
#             sys.exit()
            
            
            algorithm = {}
            algorithm['name'] = 'Adam-noWarmStart-momentum-grad'
            algorithm['params'] = {}
            algorithm['params']['RMSprop_epsilon'] = algorithm_jupyter['Adam_epsilon']
            algorithm['legend'] = 'Adam' +\
            r', $\epsilon$=' +\
            str(algorithm['params']['RMSprop_epsilon'])
            algorithms.append(copy.deepcopy(algorithm))
            
            
            
        elif algorithm_jupyter['name'] == 'SGD-momentum':
            
            algorithm = {}
            algorithm['name'] = 'SGD-momentum'
            algorithm['params'] = {}
            algorithm['legend'] = 'SGD-m'
            algorithms.append(copy.deepcopy(algorithm))
            
        else:
            
            print('algorithm_jupyter')
            print(algorithm_jupyter)

            sys.exit()



    










































    
    
    





    # args['tuning_criterion'] = 'test_acc'
    # args['tuning_criterion'] = 'train_loss'
    # args['tuning_criterion'] = 'train_acc'
    # args['tuning_criterion'] = 'train_minibatch_acc'
    args['tuning_criterion'] = 'train_minibatch_loss'


    args['list_x'] = ['epoch', 'cpu time']


    args['list_y'] = ['training unregularized minibatch loss',
                  'testing error']



    args['x_scale'] = 'linear'
    # args['x_scale'] = 'log'

    args['if_lr_in_legend'] = True

    args['if_show_legend'] = True

    args['if_test_mode'] = False
    # args['if_test_mode'] = True

    # args['if_max_epoch'] = 1
    args['if_max_epoch'] = 0

    args['color'] = None
    
    args['if_title'] = False


    get_plot(name_dataset, name_dataset_legend, algorithms, args)




    
    
    return 
