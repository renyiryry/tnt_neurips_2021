from utils_git.utils import *
# from KF_QN_CNN.utils_plot import *

def train_model(home_path = '/home/jupyter/',
                dataset_name = 'CIFAR-10',
                algorithm = 'KF-BFGS-CNN',
                lr = 0.3,
#                 lambda_damping = 0.3,
#                 Adam_epsilon = 1e-4,
                damping_value = 1e-8,
                max_cpu_time = 2000):
    
    print('change default values')

    args = {}
    
    
    
    args['list_lr'] = [lr]
    
#     args['weight_decay'] = 0
    args['weight_decay'] = 1
    print('need to change')
    
    args['momentum_gradient_dampening'] = 0
    print('need to change, especially for adam')
    
    if dataset_name == 'CIFAR-10':
        args['dataset'] = 'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias-no-regularization'
    elif dataset_name == 'CIFAR-100':
        args['dataset'] = 'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-no-regularization'
    elif dataset_name == 'MNIST':
        args['dataset'] = 'MNIST-autoencoder-relu-N1-1000-sum-loss-no-regularization'
    elif dataset_name == 'FACES':
        args['dataset'] = 'FacesMartens-autoencoder-relu-no-regularization'
    else:
        print('dataset_name')
        print(dataset_name)
        sys.exit()
    
    
    
    
    print('need to change by if lr decay')
    
    if algorithm == 'SGD-momentum':
        args['algorithm'] = 'SGD-momentum'
    elif algorithm == 'Adam':
        
        args['algorithm'] = 'Adam-noWarmStart-momentum-grad'
        args['RMSprop_epsilon'] = damping_value
        args['RMSprop_beta_2'] = 0.9
        
    elif algorithm in ['TNT', 'Shampoo']:
        
        args['shampoo_epsilon'] = damping_value
        
        args['if_Hessian_action'] = False
        
        args['shampoo_decay'] = 0.9
        args['shampoo_weight'] = 0.1
        
        if dataset_name in ['CIFAR-10', 'CIFAR-100']:
            
            if algorithm == 'TNT':
                args['algorithm'] = 'matrix-normal-correctFisher-same-trace-allVariables-filterFlattening-warmStart-momentum-grad-LRdecay'
            elif algorithm == 'Shampoo':
                args['algorithm'] = 'shampoo-allVariables-filterFlattening-warmStart-momentum-grad-LRdecay'
            
            args['shampoo_update_freq'] = 10
            args['shampoo_inverse_freq'] = 100
            
            args['num_epoch_to_decay'] = 40
            args['lr_decay_rate'] = 0.1
            
        elif dataset_name in ['MNIST', 'FACES']:
            
            if algorithm == 'TNT':
                args['algorithm'] = 'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-momentum-grad'
            elif algorithm == 'Shampoo':
                args['algorithm'] = 'shampoo-allVariables-filterFlattening-warmStart-lessInverse-momentum-grad'
            
            args['shampoo_update_freq'] = 1
            args['shampoo_inverse_freq'] = 20
            
        else:
            print('dataset_name')
            print(dataset_name)
            
            sys.exit()
            
        
    elif algorithm == 'KFAC':
        
        
        args['kfac_if_update_BN'] = True
        args['kfac_if_BN_grad_direction'] = True
        
        args['kfac_rho'] = 0.9
        args['kfac_damping_lambda'] = damping_value
        
        
        
        if dataset_name in ['FACES', 'MNIST']:
            
            args['algorithm'] = 'kfac-correctFisher-warmStart-no-max-no-LM-momentum-grad'
            
            args['kfac_if_svd'] = False
            
            args['kfac_cov_update_freq'] = 1
            args['kfac_inverse_update_freq'] = 20
            
        elif dataset_name in ['CIFAR-100', 'CIFAR-10']:
            
            args['algorithm'] = 'kfac-correctFisher-warmStart-no-max-no-LM-momentum-grad-LRdecay'

            args['kfac_if_svd'] = False

            args['kfac_cov_update_freq'] = 10
            args['kfac_inverse_update_freq'] = 100
            
            args['num_epoch_to_decay'] = 40
            args['lr_decay_rate'] = 0.1
            print('could change to only compare dataset')
            
        else:
            print('dataset_name')
            print(dataset_name)
        
            sys.exit()
        
            


            
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
