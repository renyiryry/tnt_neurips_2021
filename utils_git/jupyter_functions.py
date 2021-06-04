from utils_git.utils import *
from utils_git.utils_plot import *

def train_model(home_path = '/home/jupyter/',
                dataset_name = 'CIFAR-10',
                algorithm = 'TNT',
                lr = 1e-4,
                damping_value = 0.01,
                weight_decay = 0,
               ):

    args = {}
    
    
    
    args['list_lr'] = [lr]
    
    args['weight_decay'] = weight_decay
    

    
    if dataset_name == 'CIFAR-10':
        args['dataset'] = 'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias-no-regularization'
        args['initialization_pkg'] = 'kaiming_normal'
    elif dataset_name == 'CIFAR-100':
        args['dataset'] = 'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-no-regularization'
        args['initialization_pkg'] = 'normal'
    elif dataset_name == 'MNIST':
        args['dataset'] = 'MNIST-autoencoder-relu-N1-1000-sum-loss-no-regularization'
        args['initialization_pkg'] = 'normal'
    elif dataset_name == 'FACES':
        args['dataset'] = 'FacesMartens-autoencoder-relu-no-regularization'
        args['initialization_pkg'] = 'normal'
    else:
        print('dataset_name')
        print(dataset_name)
        sys.exit()
    
    
    
    
#     print('need to change by if lr decay')



    
    if dataset_name in ['MNIST']:
        
        # args['if_max_epoch'] = 1 # 0 means max_time
        args['if_max_epoch'] = 0 # 0 means max_time
    
    
    
        args['max_epoch/time'] = 500
        
    elif dataset_name in ['FACES']:
        
        # args['if_max_epoch'] = 1 # 0 means max_time
        args['if_max_epoch'] = 0 # 0 means max_time
    
    
    
        args['max_epoch/time'] = 2000
        
    elif dataset_name in ['CIFAR-10', 'CIFAR-100']:
        
        
        
        if algorithm in ['SGD-m', 'Adam']:
            
            args['if_max_epoch'] = 1 # 0 means max_time
#             args['if_max_epoch'] = 0 # 0 means max_time
    
    
    
            args['max_epoch/time'] = 200
            
            args['num_epoch_to_decay'] = 60
            args['lr_decay_rate'] = 0.1
            
        elif algorithm in ['TNT', 'Shampoo', 'KFAC']:
            
            args['if_max_epoch'] = 1 # 0 means max_time
#             args['if_max_epoch'] = 0 # 0 means max_time
    
    
    
            args['max_epoch/time'] = 100
            
            args['num_epoch_to_decay'] = 40
            args['lr_decay_rate'] = 0.1
            
        else:
            print('algorithm')
            print(algorithm)
        
            sys.exit()
        
    else:
        print('dataset_name')
        print(dataset_name)
    
        sys.exit()
    
    if algorithm == 'SGD-m':
        
        args['momentum_gradient_dampening'] = 0
        
        if dataset_name in ['MNIST', 'FACES']:
            args['algorithm'] = 'SGD-momentum'
        elif dataset_name in ['CIFAR-10', 'CIFAR-100']:
            args['algorithm'] = 'SGD-LRdecay-momentum'
        else:
            print('dataset_name')
            print(dataset_name)
            sys.exit()
        
            
        
        
    elif algorithm == 'Adam':
        
        args['RMSprop_epsilon'] = damping_value
        
        args['RMSprop_beta_2'] = 0.999
        
        args['momentum_gradient_dampening'] = 0.9 # i.e. beta_1
        
        if dataset_name in ['CIFAR-10', 'CIFAR-100']:
            
            args['algorithm'] = 'Adam-noWarmStart-momentum-grad-LRdecay'
            
        elif dataset_name in ['MNIST', 'FACES']:
            args['algorithm'] = 'Adam-noWarmStart-momentum-grad'
        else:
            print('dataset_name')
            print(dataset_name)
            sys.exit()
            
            
        
    elif algorithm in ['TNT', 'Shampoo']:
        
        args['shampoo_epsilon'] = damping_value
        
        args['if_Hessian_action'] = False
        
        args['shampoo_decay'] = 0.9
        args['shampoo_weight'] = 0.1
        
        args['momentum_gradient_dampening'] = 0
        
        if dataset_name in ['CIFAR-10', 'CIFAR-100']:
            
            if algorithm == 'TNT':
                args['algorithm'] = 'matrix-normal-correctFisher-same-trace-allVariables-filterFlattening-warmStart-momentum-grad-LRdecay'
            elif algorithm == 'Shampoo':
                args['algorithm'] = 'shampoo-allVariables-filterFlattening-warmStart-momentum-grad-LRdecay'
            
            args['shampoo_update_freq'] = 10
            args['shampoo_inverse_freq'] = 100
            
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
        
        args['momentum_gradient_dampening'] = 0
        
        
        
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
            
#             args['num_epoch_to_decay'] = 40
#             args['lr_decay_rate'] = 0.1
#             print('could change to only compare dataset')
            
        else:
            print('dataset_name')
            print(dataset_name)
        
            sys.exit()
        
            


            
    else:
        print('algorithm')
        print(algorithm)
        sys.exit()
        

   
    
    
    


    
    
    args['record_epoch'] = 1

    args['seed_number'] = 9999

    args['num_threads'] = 8

    # args['initialization_pkg'] = 'numpy'
    # args['initialization_pkg'] = 'default'
#     args['initialization_pkg'] = 'normal'
#     print('need to change pkg')
    
    
    
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
    algorithms_jupyter = [],
):
    
    
    
    args = {}
    
    args['home_path'] = home_path
    
    

    args['if_gpu'] = True
        
    if dataset_name == 'CIFAR-10':
        name_dataset = 'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias-no-regularization'
#         args['initialization_pkg'] = 'kaiming_normal'
    elif dataset_name == 'CIFAR-100':
        name_dataset = 'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-no-regularization'
#         args['initialization_pkg'] = 'normal'
    elif dataset_name == 'MNIST':
        name_dataset = 'MNIST-autoencoder-relu-N1-1000-sum-loss-no-regularization'
#         args['initialization_pkg'] = 'normal'
    elif dataset_name == 'FACES':
        name_dataset = 'FacesMartens-autoencoder-relu-no-regularization'
#         args['initialization_pkg'] = 'normal'
    else:
        print('dataset_name')
        print(dataset_name)
        sys.exit()



    name_dataset_legend = name_dataset
    
    

    algorithms = []
    
    for algorithm_jupyter in algorithms_jupyter:
        
        
        
#         print('need to add other algorithm')
        
        if algorithm_jupyter['name'] == 'TNT':
            
            
            
            algorithm = {}
            algorithm['params'] = {}
            
            if dataset_name in ['CIFAR-10', 'CIFAR-100']:
                algorithm['name'] = 'matrix-normal-correctFisher-same-trace-allVariables-filterFlattening-warmStart-momentum-grad-LRdecay'
                algorithm['params']['weight_decay'] = algorithm_jupyter['weight_decay']
            elif dataset_name in ['MNIST', 'FACES']:
                algorithm['name'] = 'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-momentum-grad'
                algorithm['params']['shampoo_epsilon'] = algorithm_jupyter['damping_value']
            
            
            
            
            
            algorithm['legend'] = 'TNT'
            algorithms.append(copy.deepcopy(algorithm))
            
        elif algorithm_jupyter['name'] == 'KFAC':
            
            algorithm = {}
            algorithm['params'] = {}
            
            if dataset_name in ['CIFAR-10', 'CIFAR-100']:
                algorithm['name'] = 'kfac-correctFisher-warmStart-no-max-no-LM-momentum-grad-LRdecay'
                algorithm['params']['weight_decay'] = algorithm_jupyter['weight_decay']
            elif dataset_name in ['MNIST', 'FACES']:
#                 algorithm['name'] = 'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-momentum-grad'
#                 algorithm['params']['shampoo_epsilon'] = algorithm_jupyter['damping_value']
            
                algorithm['name'] = 'kfac-correctFisher-warmStart-no-max-no-LM-momentum-grad'
                algorithm['params']['kfac_damping_lambda'] = algorithm_jupyter['damping_value']
            
            algorithm['legend'] = 'KFAC'
            algorithms.append(copy.deepcopy(algorithm))
            
            
        elif algorithm_jupyter['name'] == 'Shampoo':
            
            algorithm = {}
            algorithm['params'] = {}
            
            if dataset_name in ['CIFAR-10', 'CIFAR-100']:
                algorithm['name'] = 'shampoo-allVariables-filterFlattening-warmStart-momentum-grad-LRdecay'
                algorithm['params']['weight_decay'] = algorithm_jupyter['weight_decay']
            elif dataset_name in ['MNIST', 'FACES']:
#                 algorithm['name'] = 'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-momentum-grad'
#                 algorithm['params']['shampoo_epsilon'] = algorithm_jupyter['damping_value']
            
                algorithm['name'] = 'shampoo-allVariables-filterFlattening-warmStart-lessInverse-momentum-grad'
            
                algorithm['params']['shampoo_epsilon'] = algorithm_jupyter['damping_value']
                
            algorithm['legend'] = 'Shampoo'
            algorithms.append(copy.deepcopy(algorithm))
            
        elif algorithm_jupyter['name'] == 'Adam':
            
            
            algorithm = {}
            algorithm['params'] = {}
            
            if dataset_name in ['CIFAR-10', 'CIFAR-100']:
                algorithm['name'] = 'Adam-noWarmStart-momentum-grad-LRdecay'
                algorithm['params']['weight_decay'] = algorithm_jupyter['weight_decay']
            elif dataset_name in ['MNIST', 'FACES']:
#                 algorithm['name'] = 'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-momentum-grad'
#                 algorithm['params']['shampoo_epsilon'] = algorithm_jupyter['damping_value']
            
                algorithm['name'] = 'Adam-noWarmStart-momentum-grad'
            
                algorithm['params']['RMSprop_epsilon'] = algorithm_jupyter['damping_value']
                
            algorithm['legend'] = 'Adam'

            algorithms.append(copy.deepcopy(algorithm))
            
            
            
        elif algorithm_jupyter['name'] == 'SGD-m':
            
            algorithm = {}
            algorithm['params'] = {}
            
            if dataset_name in ['CIFAR-10', 'CIFAR-100']:
                algorithm['name'] = 'SGD-LRdecay-momentum'
                algorithm['params']['weight_decay'] = algorithm_jupyter['weight_decay']
            elif dataset_name in ['MNIST', 'FACES']:
                algorithm['name'] = 'SGD-momentum'
                
            
            algorithm['legend'] = 'SGD-m'
            algorithms.append(copy.deepcopy(algorithm))
            
        else:
            
            print('algorithm_jupyter')
            print(algorithm_jupyter)

            sys.exit()




    
    
    if dataset_name in ['CIFAR-10', 'CIFAR-100']:
        
        args['tuning_criterion'] = 'test_acc'
        
        args['list_y'] = ['training unregularized minibatch loss',
                  'testing error']
        
        args['if_max_epoch'] = 1
        
    elif dataset_name in ['MNIST', 'FACES']:
        
        # args['tuning_criterion'] = 'test_acc'
        # args['tuning_criterion'] = 'train_loss'
        # args['tuning_criterion'] = 'train_acc'
        # args['tuning_criterion'] = 'train_minibatch_acc'
        args['tuning_criterion'] = 'train_minibatch_loss'


#     args['list_y'] = ['training unregularized minibatch loss',
#                   'testing error']
        args['list_y'] = ['training unregularized minibatch loss']
    
        # args['if_max_epoch'] = 1
        args['if_max_epoch'] = 0
        
    else:
        print('dataset_name')
        print(dataset_name)
    
        sys.exit()









    args['list_x'] = ['epoch', 'cpu time']


    



    args['x_scale'] = 'linear'
    # args['x_scale'] = 'log'

    args['if_lr_in_legend'] = True

    args['if_show_legend'] = True

    args['if_test_mode'] = False

    

    args['color'] = None
    
    args['if_title'] = False


    get_plot(name_dataset, name_dataset_legend, algorithms, args)




    
    
    return 
