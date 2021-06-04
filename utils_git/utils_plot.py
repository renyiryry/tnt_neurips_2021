import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import datetime
import math



# from utils_git.utils import *
from utils_git.utils import get_name_loss
from utils_git.utils import get_best_params
from utils_git.utils import get_name_algorithm_with_params
from utils_git.utils import from_dataset_to_N1_N2

def get_mean_and_scaled_error(list_data, key, name_loss, if_max_epoch):
    
#     list_curves = []
    list_curves_raw = []
    for data_ in list_data:
        
        list_curves_raw.append(data_[key])
        
#     print('if_max_epoch')
#     print(if_max_epoch)
    
#     sys.exit()
    
    if if_max_epoch:
        
        sys.exit()
        
        len_ = max([len(curve) for curve in list_curves_raw])
        
    else:
        len_ = min([len(curve) for curve in list_curves_raw])
        
 
    
    
    print('len_')
    print(len_)
    
    list_curves = []
#     for i in range(len(list_curves_raw)):
    for curve in list_curves_raw:

        if if_max_epoch:
        
            sys.exit()

            if len(curve) == len_:
                list_curves.append(curve)
                
        else:
            list_curves.append(curve[:len_])
            
#             list_curves[i] = list_curves[i]
        
    
    for curve in list_curves:
        
        assert len(curve) == len(list_curves[0])
        
    list_curves = np.asarray(list_curves)
    
    print('key')
    print(key)
    
    if key in ['train_unregularized_minibatch_losses',
               'epochs',
               'timesCPU']:
        pass
    elif key == 'test_acces':
        

        
        if name_loss in ['logistic-regression-sum-loss',
                         'linear-regression-half-MSE',]:
            pass
        elif name_loss in ['multi-class classification',]:
            list_curves = 1 - list_curves
        else:
            print('name_loss')
            print(name_loss)
            sys.exit()
        
            
    else:
        sys.exit()
        
#     print('list_curves')
#     print(list_curves)
        
#     print('len(list_curves)')
#     print(len(list_curves))
        
#     print('math.sqrt(len(list_curves))')
#     print(math.sqrt(len(list_curves)))
    
#     sys.exit()
    
#     return np.mean(list_curves, axis=0), np.std(list_curves, axis=0)
    return np.mean(list_curves, axis=0), np.std(list_curves, axis=0) / math.sqrt(len(list_curves))

def get_plot_seed_result(z_value, dataset, algorithms, if_max_epoch, list_y, path_to_home, if_title):
    
#     plt.rcParams['xtick.labelsize']=20
#     plt.rcParams['ytick.labelsize']=20

    plt.rcParams.update({'font.size': 22})

    list_y_legend = []
    for key_y in list_y:        
        if key_y == 'train_unregularized_minibatch_losses':
            list_y_legend.append('train loss')
        elif key_y == 'test_acces':
            list_y_legend.append('val error')
        else:
            print('key_y')
            print(key_y)
        
            sys.exit()

    list_x = ['epochs', 'timesCPU']
    list_x_legend = ['epoch', 'process time (second)']

    # plt.figure(figsize=(30,40))

#     fig, axs = plt.subplots(len(list_y), len(list_x), figsize=(15,15))
    fig, axs = plt.subplots(len(list_y), len(list_x), figsize=(15,7.5*len(list_y)))

    fake_args = {'dataset': dataset}
    from_dataset_to_N1_N2(fake_args)

    N1 = fake_args['N1']
    N2 = fake_args['N2']

    for algorithm in algorithms:

    #     algorithm = 'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-\
    #     momentum-s-y-DDV2-regularized-grad-momentum-grad'

        algorithm_name = algorithm['name']
        algorithm_name_legend = algorithm['name_legend']
        lr = algorithm['lr']






#         path_to_home = '/rigel/home/yr2322/gauss_newton/'

        path_to_dir = path_to_home + 'result/' +\
        dataset + '/' +\
        algorithm_name + '/if_gpu_True/alpha_' + str(lr) +\
        '/N1_' + str(N1) + '/N2_' + str(N2) + '/'

        # os.listdir(path_to_dir)

        list_data = []

        for file in os.listdir(path_to_dir):

            with open(path_to_dir + file, 'rb') as fp:
                data_ = pickle.load(fp)

#             if data_['params']['if_max_epoch'] == 0 and\
#             data_['params']['if_test_mode'] == False:
            if data_['params']['if_test_mode'] == False:
                
                flag = True
                
                if 'params' in algorithm:
                    for key in algorithm['params']:
                        if key not in data_['params']:
                            flag = False
                            break
                        if algorithm['params'][key] != data_['params'][key]:
                            flag = False
                            break
                    

                if flag:
                    
                    print('data_[params]')
                    print(data_['params'])
                    
                    list_data.append(data_)

        print('len(list_data)')
        print(len(list_data))





    #     assert len(list_data) > 0
        assert len(list_data) == 5

        for j in range(len(list_x)):

            key_x = list_x[j]
            
            name_loss = get_name_loss(dataset)

            mean_x, _ = get_mean_and_scaled_error(list_data, key_x, name_loss, if_max_epoch)

            for i in range(len(list_y)):

                # compute the mean curve and the error bar
                key_y = list_y[i]
                
                


#                 mean_y, error = get_mean_and_error(list_data, key_y, name_loss)
                mean_y, scaled_error = get_mean_and_scaled_error(list_data, key_y, name_loss, if_max_epoch)
                
                
                if key_y == 'test_acces':
                    
                    print('1 - np.min(mean_y)')
                    print(1 - np.min(mean_y))
            
#                 print('i, j')
#                 print(i, j)
                
#                 print('axs')
#                 print(axs)
                
                if len(list_y) == 1:
                    ax = axs[j]
                else:
                    ax = axs[i, j]

                # see https://www.mathsisfun.com/data/confidence-interval.html
                ax.plot(mean_x, mean_y, label=algorithm_name_legend)
                
#                 ax.fill_between(
#                     mean_x, 
#                     mean_y-error/math.sqrt(len(list_data)), 
#                     mean_y+error/math.sqrt(len(list_data)), 
#                     alpha=0.5
#                 )
                ax.fill_between(
                    mean_x, 
                    mean_y - z_value * scaled_error, 
                    mean_y + z_value * scaled_error, 
                    alpha=0.5
                )
                


                ax.set_yscale('log')

                ax.set_xlabel(list_x_legend[j])
                ax.set_ylabel(list_y_legend[i])

                
    plt.legend()
    
    if if_title:
        fig.suptitle(dataset)
        
    plt.tight_layout()
    
    
        
    path_to_dir = path_to_home + 'logs/plot_seed_result/' + dataset + '/'
    
    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir)
    
    plt.savefig(
        path_to_dir + str(datetime.datetime.now().strftime('%Y-%m-%d-%X')) + '.pdf'
    )
    
    plt.show()

#     return 

def plot_matrices_norm_kfac(args):
    
    path_to_file = args['path_to_file']

    with open(path_to_file, 'rb') as fp:
        data_ = pickle.load(fp)

    print('data_.keys()')
    print(data_.keys())

    kfac_G_inv_norms_per_epoch =\
    np.asarray(data_['kfac_G_inv_norms_per_epoch'])
    kfac_G_norms_per_epoch = np.asarray(data_['kfac_G_norms_per_epoch'])
    kfac_F_twoNorms_per_epoch = np.asarray(data_['kfac_F_twoNorms_per_epoch'])
    
    kfac_A_twoNorms_per_epoch = np.asarray(data_['kfac_A_twoNorms_per_epoch'])
    kfac_G_twoNorms_per_epoch = np.asarray(data_['kfac_G_twoNorms_per_epoch'])
    layerWiseHessian_twoNorms_per_epoch = np.asarray(data_['layerWiseHessian_twoNorms_per_epoch'])



    num_epoch = kfac_A_twoNorms_per_epoch.shape[0]
    
    num_subplots = 6
    
    plt.figure(figsize=(8,8*num_subplots))
    
    plt.subplot(num_subplots, 1, 1)
    for l in range(kfac_A_twoNorms_per_epoch.shape[1]):
        if l != 0 and l != 5:
            plt.plot(np.arange(num_epoch) + 1, kfac_A_twoNorms_per_epoch[:, l], label='l = ' + str(l))
    plt.legend()
    plt.title('A (without LM)')
    
    plt.subplot(num_subplots, 1, 2)
    for l in range(kfac_G_twoNorms_per_epoch.shape[1]):
        if l != 0 and l != 5:
            plt.plot(np.arange(num_epoch) + 1, kfac_G_twoNorms_per_epoch[:, l], label='l = ' + str(l))
    plt.legend()
    plt.title('G (without LM)')
    
    plt.subplot(num_subplots, 1, 3)
    assert kfac_A_twoNorms_per_epoch.shape[1] ==\
    kfac_G_twoNorms_per_epoch.shape[1]
    for l in range(kfac_A_twoNorms_per_epoch.shape[1]):
        if l != 0 and l != 5:
            plt.plot(
                np.arange(num_epoch) + 1, 
                kfac_A_twoNorms_per_epoch[:, l] *\
                kfac_G_twoNorms_per_epoch[:, l], 
                label='l = ' + str(l)
            )
    plt.legend()
    plt.title('kfac_F (i.e. A kron G) (without LM)')
    
    
    
    
    plt.subplot(num_subplots, 1, 4)
    for l in range(layerWiseHessian_twoNorms_per_epoch.shape[1]):
        if l != 0 and l != 5:
            plt.plot(np.arange(num_epoch) + 1, layerWiseHessian_twoNorms_per_epoch[:, l], label='l = ' + str(l))
    plt.legend()
    plt.title('true Hessian (without LM)')
    
    plt.subplot(num_subplots, 1, 5)
    lambda_kfac = data_['params']['kfac_damping_lambda']
#     print('lambda_A')
#     print(lambda_A)
    assert kfac_A_twoNorms_per_epoch.shape[1] ==\
    kfac_G_twoNorms_per_epoch.shape[1]
    for l in range(kfac_A_twoNorms_per_epoch.shape[1]):
        if l != 0 and l != 5:
            plt.plot(
                np.arange(num_epoch) + 1, 
                (kfac_A_twoNorms_per_epoch[:, l] + np.sqrt(lambda_kfac)) *\
                (kfac_G_twoNorms_per_epoch[:, l] + np.sqrt(lambda_kfac)), 
                label='l = ' + str(l)
            )
    plt.legend()
    plt.title('kfac_F (with LM)')
    
    
    plt.subplot(num_subplots, 1, 6)

    for l in range(layerWiseHessian_twoNorms_per_epoch.shape[1]):
        if l != 0 and l != 5:
            plt.plot(
                np.arange(num_epoch) + 1, 
                layerWiseHessian_twoNorms_per_epoch[:, l] + lambda_kfac, 
                label='l = ' + str(l)
            )
    plt.legend()
    plt.title('true Hessian (with LM)')
    
    
    '''
    
    

    
    plt.subplot(num_subplots, 1, 7)

    lambda_G = data_['params']['Kron_BFGS_H_epsilon']

    lambda_ = lambda_A * lambda_G

    for l in range(layerWiseHessian_twoNorms_per_epoch.shape[1]):
        if l != 0 and l != 5:
            plt.plot(
                np.arange(num_epoch) + 1, 
                layerWiseHessian_twoNorms_per_epoch[:, l] + lambda_, 
                label='l = ' + str(l)
            )


    plt.legend()
    plt.title('true Hessian (with LM)')
    
    plt.subplot(num_subplots, 1, 8)
    for l in range(inverseLayerWiseHessian_LM_twoNorms_per_epoch.shape[1]):
        if l != 0 and l != 5:
            plt.plot(
                np.arange(num_epoch) + 1, 
                inverseLayerWiseHessian_LM_twoNorms_per_epoch[:, l], 
                label='l = ' + str(l)
            )
    plt.legend()
    plt.title('inverse of true Hessian (with LM)')
    
    plt.subplot(num_subplots, 1, 9)
    for l in range(inverseLayerWiseHessian_LM_MA_twoNorms_per_epoch.shape[1]):
        if l != 0 and l != 5:
            plt.plot(
                np.arange(num_epoch) + 1, 
                inverseLayerWiseHessian_LM_MA_twoNorms_per_epoch[:, l], 
                label='l = ' + str(l)
            )
    plt.legend()
    plt.title('inverse of true Hessian (MA) (with LM)')
    '''
    
    
    
    
    
    
    
    home_path = args['home_path']
    
#     print('path_to_file.split(result/)')
#     print(path_to_file.split('result/'))
    
    print('home_path + path_to_file.split(result/)[-1]')
    print(home_path + 'logs/plot_matrices_norm_kron_bfgs/' + path_to_file.split('result/')[-1] + '.pdf')
    
    saved_path_to_file = home_path + 'logs/plot_matrices_norm_kron_bfgs/' + path_to_file.split('result/')[-1] + '.pdf'
    
    if not os.path.exists(saved_path_to_file):
        os.makedirs(saved_path_to_file)
    if os.path.isdir(saved_path_to_file): 
        os.rmdir(saved_path_to_file)
    
    plt.savefig(saved_path_to_file)
    
    print('saved_path_to_file')
    print(saved_path_to_file)
    
    print('saved_path_to_file.split(result_)')
    print(saved_path_to_file.split('result_'))
    
    saved_path_to_dir = saved_path_to_file.split('result_')[0]
    
    name_pkl_file = 'result_' + path_to_file.split('result_')[1]
    
#     print('name_pkl_file')
#     print(name_pkl_file)
    
    
    
    shutil.copyfile(path_to_file, saved_path_to_dir + name_pkl_file)
    
    plt.show()


def plot_matrices_norm_kron_bfgs(args):
    
    path_to_file = args['path_to_file']

    with open(path_to_file, 'rb') as fp:
        data_ = pickle.load(fp)

#     print(data_.keys())

    kron_bfgs_A_twoNorms_per_epoch = np.asarray(data_['kron_bfgs_A_twoNorms_per_epoch'])
    kron_bfgs_G_LM_twoNorms_per_epoch = np.asarray(data_['kron_bfgs_G_LM_twoNorms_per_epoch'])
    kron_bfgs_Hg_twoNorms_per_epoch = np.asarray(data_['kron_bfgs_Hg_twoNorms_per_epoch'])
    kron_bfgs_Ha_twoNorms_per_epoch = np.asarray(data_['kron_bfgs_Ha_twoNorms_per_epoch'])
    layerWiseHessian_twoNorms_per_epoch = np.asarray(data_['layerWiseHessian_twoNorms_per_epoch'])
    inverseLayerWiseHessian_LM_twoNorms_per_epoch = np.asarray(data_['inverseLayerWiseHessian_LM_twoNorms_per_epoch'])
    inverseLayerWiseHessian_LM_MA_twoNorms_per_epoch = np.asarray(data_['inverseLayerWiseHessian_LM_MA_twoNorms_per_epoch'])



    num_epoch = kron_bfgs_A_twoNorms_per_epoch.shape[0]
    
    num_subplots = 5
    
    plt.figure(figsize=(8,8*num_subplots))
    
    plt.subplot(num_subplots, 1, 1)

    for l in range(kron_bfgs_A_twoNorms_per_epoch.shape[1]):
        if l != 0 and l != 5:
            plt.plot(np.arange(num_epoch) + 1, kron_bfgs_A_twoNorms_per_epoch[:, l], label='l = ' + str(l))


    plt.legend()
    plt.title('A (without LM)')
    
#     plt.subplot(num_subplots, 1, 2)
#     for l in range(kron_bfgs_Ha_twoNorms_per_epoch.shape[1]):
#         if l != 0 and l != 5:
#             plt.plot(np.arange(num_epoch) + 1, kron_bfgs_Ha_twoNorms_per_epoch[:, l], label='l = ' + str(l))
#     plt.legend()
#     plt.title('H_a (with LM)')
    
    plt.subplot(num_subplots, 1, 2)
    for l in range(kron_bfgs_Hg_twoNorms_per_epoch.shape[1]):
        if l != 0 and l != 5:
            plt.plot(np.arange(num_epoch) + 1, kron_bfgs_Hg_twoNorms_per_epoch[:, l], label='l = ' + str(l))
    plt.legend()
    plt.title('H_g (with LM)')
    
#     plt.subplot(num_subplots, 1, 4)
#     assert kron_bfgs_Ha_twoNorms_per_epoch.shape[1] ==\
#     kron_bfgs_Hg_twoNorms_per_epoch.shape[1]
#     for l in range(kron_bfgs_Ha_twoNorms_per_epoch.shape[1]):
#         if l != 0 and l != 5:
#             plt.plot(
#                 np.arange(num_epoch) + 1, 
#                 kron_bfgs_Ha_twoNorms_per_epoch[:, l] *\
#                 kron_bfgs_Hg_twoNorms_per_epoch[:, l], 
#                 label='l = ' + str(l)
#             )
#     plt.legend()
#     plt.title('H_a kron H_g')
    
    plt.subplot(num_subplots, 1, 3)
    for l in range(kron_bfgs_G_LM_twoNorms_per_epoch.shape[1]):
        if l != 0 and l != 5:
            plt.plot(np.arange(num_epoch) + 1, kron_bfgs_G_LM_twoNorms_per_epoch[:, l], label='l = ' + str(l))
    plt.legend()
    plt.title('G_LM (i.e. inverse of H_g)')
    
    plt.subplot(num_subplots, 1, 4)
    lambda_A = data_['params']['Kron_BFGS_A_LM_epsilon']
    print('lambda_A')
    print(lambda_A)
    assert kron_bfgs_A_twoNorms_per_epoch.shape[1] ==\
    kron_bfgs_G_LM_twoNorms_per_epoch.shape[1]
    for l in range(kron_bfgs_A_twoNorms_per_epoch.shape[1]):
        if l != 0 and l != 5:
            plt.plot(
                np.arange(num_epoch) + 1, 
                (kron_bfgs_A_twoNorms_per_epoch[:, l] + lambda_A) *\
                kron_bfgs_G_LM_twoNorms_per_epoch[:, l], 
                label='l = ' + str(l)
            )
    plt.legend()
    plt.title('KBFGS_Hessian (i.e. A_LM kron G_LM) (with LM)')

    
    plt.subplot(num_subplots, 1, 5)

    lambda_G = data_['params']['Kron_BFGS_H_epsilon']

    lambda_ = lambda_A * lambda_G

    for l in range(layerWiseHessian_twoNorms_per_epoch.shape[1]):
        if l != 0 and l != 5:
            plt.plot(
                np.arange(num_epoch) + 1, 
                layerWiseHessian_twoNorms_per_epoch[:, l] + lambda_, 
                label='l = ' + str(l)
            )


    plt.legend()
    plt.title('true Hessian (with LM)')
    
#     plt.subplot(num_subplots, 1, 8)
#     for l in range(inverseLayerWiseHessian_LM_twoNorms_per_epoch.shape[1]):
#         if l != 0 and l != 5:
#             plt.plot(
#                 np.arange(num_epoch) + 1, 
#                 inverseLayerWiseHessian_LM_twoNorms_per_epoch[:, l], 
#                 label='l = ' + str(l)
#             )
#     plt.legend()
#     plt.title('inverse of true Hessian (with LM)')
    
#     plt.subplot(num_subplots, 1, 9)
#     for l in range(inverseLayerWiseHessian_LM_MA_twoNorms_per_epoch.shape[1]):
#         if l != 0 and l != 5:
#             plt.plot(
#                 np.arange(num_epoch) + 1, 
#                 inverseLayerWiseHessian_LM_MA_twoNorms_per_epoch[:, l], 
#                 label='l = ' + str(l)
#             )
#     plt.legend()
#     plt.title('inverse of true Hessian (MA) (with LM)')
    
    
    
    
    
    
    
    home_path = args['home_path']
    
#     print('path_to_file.split(result/)')
#     print(path_to_file.split('result/'))
    
    print('home_path + path_to_file.split(result/)[-1]')
    print(home_path + 'logs/plot_matrices_norm_kron_bfgs/' + path_to_file.split('result/')[-1] + '.pdf')
    
    saved_path_to_file = home_path + 'logs/plot_matrices_norm_kron_bfgs/' + path_to_file.split('result/')[-1] + '.pdf'
    
    if not os.path.exists(saved_path_to_file):
        os.makedirs(saved_path_to_file)
    if os.path.isdir(saved_path_to_file): 
        os.rmdir(saved_path_to_file)
    
    plt.savefig(saved_path_to_file)
    
    print('saved_path_to_file')
    print(saved_path_to_file)
    
    print('saved_path_to_file.split(result_)')
    print(saved_path_to_file.split('result_'))
    
    saved_path_to_dir = saved_path_to_file.split('result_')[0]
    
    name_pkl_file = 'result_' + path_to_file.split('result_')[1]
    
#     print('name_pkl_file')
#     print(name_pkl_file)
    
    
    
    shutil.copyfile(path_to_file, saved_path_to_dir + name_pkl_file)
    
    plt.show()


    

def plot_hyperparams_vs_loss_v2(args):
    
    from utils_git.utils_plot import plot_hyperparams_vs_loss_multiple_algos
    
    from scipy.interpolate import griddata

    
    hyperparams = plot_hyperparams_vs_loss_multiple_algos(args)

    hyperparams = hyperparams[0]
    
    algorithm = args['algorithms'][0]
    
    if algorithm['algorithm'] in ['Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad',
                                  'Kron-BFGS-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad']:
        hyperparams[:, 1] = hyperparams[:, 1]**2
    else:
        print('error: check if **2 for ' + algorithm['algorithm'])
        sys.exit()
    
#     print('hyperparams[:, 1]')
#     print(hyperparams[:, 1])
    
#     sys.exit()

    # Achraf's code

#     import pandas as pd
#     import pickle
    import matplotlib.pyplot as plt
    from matplotlib import ticker, cm
    
    
    
    import numpy as np
    import os
    import datetime
    
    plt.rcParams.update({'font.size': 18})
    plt.rc('font', family='serif')
    plt.style.use('seaborn-muted')


    hyperparams #: grid of results stored as a list of (lr, damping, value)
    
    x = []
    y = []
    z = []

    for x_t in hyperparams:
        x += [x_t[0]]
        y += [x_t[1]]
        z += [x_t[2]]

    step = 5
    
    if args['dataset'] in ['CURVES-autoencoder-relu-sum-loss',
                           'MNIST-autoencoder-relu-N1-1000-sum-loss']:
        min_level = 50
        max_level = 200
    elif args['dataset'] == 'FacesMartens-autoencoder-relu':
        min_level = 5
        max_level = 100
    else:
        print('error: min_level for ' + args['dataset'])
#         print('z')
#         print(z)
        print('max(z)')
        print(max(z))
        print('min(z)')
        print(min(z))
        sys.exit()
    
    levels = step*np.arange(min_level/step, max_level/step +1)

    

    xs,ys = np.meshgrid(np.logspace(np.log10(min(x)), np.log10(max(x)), num=50), np.logspace(np.log10(min(y)), np.log10(max(y)), num=50))
    
    resampled = griddata((x, y), z, (xs, ys), method = 'linear')
#     resampled = scipy.interpolate.griddata((x, y), z, (xs, ys), method = 'linear')

    
    

    # Name_tilte = 'K-BFGS(L)' + ', ' + 'Curves Autoencoder'
    Name_tilte = algorithm['algo_legend'] + ', ' + args['dataset_legend']

    fig = plt.figure(figsize = (12,10))
    cp = plt.contourf(xs, ys, resampled, levels = levels, cmap = cm.RdYlBu,  extend='max')
    fig.patch.set_facecolor('white')

    cp.cmap.set_over(cm.RdYlBu_r(1))
    cp.cmap.set_under('red')

    plt.xscale('log')
    plt.yscale('log')

    plt.colorbar(cp)

    plt.title(Name_tilte, fontsize = 25, fontweight='normal')
    plt.xlabel('learning rate')
    plt.ylabel('damping')
#     plt.show()
    
    saving_dir = args['home_path'] + 'logs/' + 'hyperparams_vs_loss_v2/' +\
    args['dataset'] + '/'
    
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    plt.savefig(saving_dir +\
                str(datetime.datetime.now().strftime('%Y-%m-%d-%X')) + '.pdf', bbox_inches='tight')
    
    plt.show()


'''
def plot_hyperparams_vs_loss_v2(args):
    
    hyperparams = plot_hyperparams_vs_loss_multiple_algos(args)

    hyperparams = hyperparams[0]

    print('hyperparams.shape')
    print(hyperparams.shape)

    # Achraf's code

#     import pandas as pd
#     import pickle
#     import matplotlib.pyplot as plt
    from matplotlib import ticker, cm
    
    from scipy.interpolate import griddata
    
#     import numpy as np

    
    plt.rcParams.update({'font.size': 18})
    plt.rc('font', family='serif')
    plt.style.use('seaborn-muted')


    hyperparams #: grid of results stored as a list of (lr, damping, value)

    step = 5
    levels = step*np.arange(50/step, 200/step +1)

    x = []
    y = []
    z = []

    for x_t in hyperparams:
        x += [x_t[0]]
        y += [x_t[1]]
        z += [x_t[2]]

    xs,ys = np.meshgrid(np.logspace(np.log10(min(x)), np.log10(max(x)), num=50), np.logspace(np.log10(min(y)), np.log10(max(y)), num=50))
    
    resampled = griddata((x, y), z, (xs, ys), method = 'linear')
#     resampled = scipy.interpolate.griddata((x, y), z, (xs, ys), method = 'linear')
    

    # Name_tilte = 'K-BFGS(L)' + ', ' + 'Curves Autoencoder'
    Name_tilte = algorithm['algo_legend'] + ', ' + args['dataset_legend']

    fig = plt.figure(figsize = (12,10))
    cp = plt.contourf(xs, ys, resampled, levels = levels, cmap = cm.RdYlBu,  extend='max')
    fig.patch.set_facecolor('white')

    cp.cmap.set_over(cm.RdYlBu_r(1))
    cp.cmap.set_under('red')

    plt.xscale('log')
    plt.yscale('log')

    plt.colorbar(cp)

    plt.title(Name_tilte, fontsize = 25, fontweight='normal')
    plt.xlabel('learning rate')
    plt.ylabel('damping')
#     plt.show()
    
    saving_dir = args['home_path'] + 'logs/' + 'hyperparams_vs_loss_v2/' +\
    args['dataset'] + '/'
    
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    plt.savefig(saving_dir +\
                str(datetime.datetime.now().strftime('%Y-%m-%d-%X')) + '.pdf', bbox_inches='tight')
    
    plt.show()
    '''

    

def plot_hyperparams_vs_loss_multiple_algos(args):

    num_algo = len(args['algorithms'])
    
    plt.figure(figsize=(3,1))

    fig, axs = plt.subplots(1, num_algo)
#     fig, axs = plt.subplots(1, num_algo+1)
    
    if num_algo == 1:
        axs = [axs]
    
    hyperparams = []
    
    index_subplot = 0
    for algo in args['algorithms']:
        
#         print('axs')
#         print(axs)
#         sys.exit()
        
        

        
#         ax = plt.subplot(1, len(args['algorithms']), index_subplot)
        ax = axs[index_subplot]
    
        index_subplot += 1
        
#         print('args.keys()')
#         print(args.keys())
#         sys.exit()
        

        
        args_1 = copy.deepcopy(args)
        args_1.pop('algorithms')
        
        args_1.update(algo)
        
#         print('args_1.keys()')
#         print(args_1.keys())
#         sys.exit()
        
        hyperparams_i = plot_hyperparams_vs_loss(index_subplot, axs, args_1)
        hyperparams.append(hyperparams_i)
        
    saving_dir = args['home_path'] + 'logs/' + 'hyperparams_vs_loss/' +\
    args['dataset'] + '/'
    
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    plt.savefig(saving_dir +\
                str(datetime.datetime.now().strftime('%Y-%m-%d-%X')) + '.pdf', bbox_inches='tight')
    
    plt.show()
    
    return hyperparams

def plot_hyperparams_vs_loss(index_subplot, axs, args):
    
    
    
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    
    import matplotlib.ticker as ticker
    
    ax = axs[index_subplot-1]
    
    


    home_path = args['home_path']
    name_dataset = args['dataset']
    algorithm = args['algorithm']
    list_lr = args['list_lr']
    

    args = from_dataset_to_N1_N2(args)
    N1 = args['N1']
    N2 = args['N2']
    
    fetched_data = []

    for lr in list_lr:
        working_dir = home_path + 'result/' + name_dataset + '/' + algorithm + '/' + 'if_gpu_True/' +\
                  'alpha_' + str(lr) + '/' +\
                  'N1_' + str(N1) + '/' +\
                  'N2_' + str(N2) + '/'



        os.chdir(working_dir)

        print(os.listdir())

        list_file = os.listdir()

        for file_ in list_file:
    #         print(file_)

            print('\n')
            print('len(fetched_data)')
            print(len(fetched_data))

            with open(file_, 'rb') as fp:
                data_ = pickle.load(fp)



            if data_['params']['if_test_mode'] == True:
                continue

            print('data_[params]')
            print(data_['params'])
            
            if algorithm in ['Kron-BFGS-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad',
                             'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad']:
                damping_keyword = 'Kron_BFGS_A_LM_epsilon'
            elif algorithm == 'kfac-no-max-no-LM-momentum-grad':
                damping_keyword = 'kfac_damping_lambda'
            else:
                print('error: unknown algorithm for ' + algorithm)
                sys.exit()
             

             # print('args[list_damping]')
#             print(args['list_damping'])
#             print('data_[params][damping_keyword]')
#             print(data_['params'][damping_keyword])
#             sys.exit()

            if data_['params'][damping_keyword] in args['list_damping']:
                fetched_data.append(
                    [lr, 
                     data_['params'][damping_keyword], 
                     np.min(data_['train_losses'])]
                )

    print('fetched_data')
    print(fetched_data)

    fetched_data = np.asarray(fetched_data)

    print('fetched_data.shape')
    print(fetched_data.shape)

    x = fetched_data[:, 0]
    y = fetched_data[:, 1]
    z = fetched_data[:, 2]
    
    
    cm = plt.cm.get_cmap('RdYlBu')

#     fig = plt.figure(figsize=(6,6))
#     ax = fig.add_subplot(111)

    ax.set_xscale('log')
    ax.set_yscale('log')

#     print('y')
#     print(y)
#     sys.exit()

#     plt.xlim(min(x) / 2, max(x) * 2)
#     plt.ylim(min(y) / 2, max(y) * 2)

    ax.set(xlim=(min(x) / 2, max(x) * 2))
    ax.set(ylim=(min(y) / 2, max(y) * 2))
    
#     plt.xlabel('learning rate')
    ax.set(xlabel='learning rate')
    
    if index_subplot == 1:
#     plt.ylabel('damping')
        ax.set(ylabel='damping')
        


#     plt.title(args['algo_legend'])
    ax.set(title=args['algo_legend'])
    

    # sc = plt.scatter(x, y, c=z, vmin=np.min(z), vmax=np.max(z), s=35, cmap=cm)
    
    if args['if_contour']:
        
#         if index_subplot == 1:
        
#             min_z = np.min(z)
#             max_z = np.max(z)
#         else:
#             min_z = np.minimum(min_z, np.min(z))
#             max_z = np.maximum(max_z, np.max(z))
    
#     sc = ax.tricontour(x, y, z, vmin=np.min(z), vmax=np.max(z), cmap=cm, norm=colors.LogNorm())


        levs = np.geomspace(np.min(z), np.max(z), num=100)
#         levs = np.linspace(np.min(z), np.max(z))
        
#         print('levs')
#         print(levs)

#         ax.tricontour(x, y, z, levs, cmap=cm, norm=colors.LogNorm(vmin=np.min(z), vmax=np.max(z)))
        sc = ax.tricontour(x, y, z, levs, cmap=cm, norm=colors.LogNorm())
#         sc = ax.tricontour(x, y, z, levs, cmap=cm)
#         sc = ax.tricontour(x, y, z, levs)

#         print('return_tricontour')
#         print(return_tricontour)
        
#         plt.setp(return_tricontour, norm=colors.LogNorm())
        
#         sys.exit()

#         sc = ax.tricontourf(x, y, z, levs, cmap=cm, norm=colors.LogNorm(vmin=np.min(z), vmax=np.max(z)))
        sc = ax.tricontourf(x, y, z, levs, cmap=cm, norm=colors.LogNorm())
#         sc = ax.tricontourf(x, y, z, levs, cmap=cm)

#         plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
#         if index_subplot == len(axs):
#             plt.colorbar(sc, ax=axs, norm=colors.LogNorm())
#             plt.colorbar(sc, ax=axs, spacing='proportional')
#             plt.colorbar(ax=axs)
            
#             plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs)
            
#             plt.colorbar.ColorbarBase(tick = levs)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if index_subplot == len(axs):
            
            min_z = np.min(z)
            max_z = np.max(z)
            
            unit = 10**np.floor(np.log10(min_z))
            multiplier = np.floor(min_z / unit)

            levs = []
            while 1:
                levs_i = unit * multiplier


                levs.append(levs_i)

                if multiplier == 9:
                    multiplier = 1
                    unit *= 10
                else:
                    multiplier += 1

                if levs_i > max_z:
                    break

            levs_colorbar = levs
            
            
            plt.colorbar(sc, ax=axs, spacing='proportional', ticks=levs_colorbar)
        
    else:
    
#         sc = ax.scatter(x, y, c=z, vmin=np.min(z), vmax=np.max(z), s=35, cmap=cm, norm=colors.LogNorm())
        sc = ax.scatter(
            x, y, c=z, s=35, cmap=cm, 
            norm=colors.LogNorm(vmin=np.min(z), vmax=np.max(z))
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if index_subplot == len(axs):
            plt.colorbar(sc, ax=axs)
            
    return fetched_data
        
    
    
    
    
    



    


    



def get_subplot(ax, name_dataset, name_dataset_legend, algorithms, x_axis_name, y_axis_name, args):
    
    from utils_git.utils import get_name_algorithm
    
#     print('args')
#     print(args)

    plt.rcParams['xtick.labelsize']=20
    plt.rcParams['ytick.labelsize']=20
    
    ax.set_yscale('log')
    
#     ax.set_xscale('log')
#     ax.set_xscale('linear')
    ax.set_xscale(args['x_scale'])
    
    
    
    
    i = -1
    for algorithm_dict_try in algorithms:
        i += 1

        if isinstance(algorithm_dict_try, str):
            algorithm = algorithm_dict_try
            algorithm_dict = {}
            algorithm_dict['name'] = algorithm
            algorithm_dict['params'] = {}
            algorithm_dict['legend'] = algorithm
        else:
            algorithm = algorithm_dict_try['name']
            algorithm_dict = algorithm_dict_try
            if not 'legend' in algorithm_dict:
                algorithm_dict['legend'] = algorithm
                

        
        if 'if_test_mode' not in algorithm_dict['params']:
            algorithm_dict['params']['if_test_mode'] = args['if_test_mode']
        
        if 'if_max_epoch' not in algorithm_dict['params']:
            algorithm_dict['params']['if_max_epoch'] = args['if_max_epoch']
                

        path_to_google_drive_dir = args['home_path'] + 'result/'

        fake_params = {}
        # fake_params['if_Adam'] = args['if_Adam']
        # fake_params['if_momentum_gradient'] = if_momentum_gradient
        fake_params['algorithm'] = algorithm
        fake_params['if_gpu'] = args['if_gpu']

        name_algorithm, no_algorithm = get_name_algorithm(fake_params)
        
        name_result = name_dataset + '/' + name_algorithm + '/'



        

        # get best_alpha
        


        args['algorithm'] = algorithm
        args['dataset'] = name_dataset
        args['name_loss'] = get_name_loss(name_dataset)
        args['algorithm_dict'] = algorithm_dict

        best_alpha, _, best_name_result_pkl = get_best_params(args, if_plot=False)
    
        if best_name_result_pkl == None:
            print('Error: best_name_result_pkl == None for ' + args['algorithm'])
            sys.exit()


        fake_params = {}
        fake_params['alpha'] = best_alpha
        fake_params['N1'] = args['N1']
        fake_params['N2'] = args['N2']
        # fake_params['if_Adam'] = args['if_Adam']
        # fake_params['if_momentum_gradient'] = if_momentum_gradient
        fake_params['algorithm'] = algorithm
#         fake_params['RMSprop_epsilon'] = best_epsilon
        fake_params['if_gpu'] = args['if_gpu']

        name_algorithm_with_params = get_name_algorithm_with_params(fake_params)
        
        
        # name_result = name_dataset + '_' + name_algorithm
        name_result_with_params = name_dataset + '/' + name_algorithm_with_params + '/'

        with open(path_to_google_drive_dir + name_result_with_params +\
                    best_name_result_pkl, 'rb') as handle:
            record_result = pickle.load(handle)
    
        if y_axis_name == 'training loss':
            
            

            if 'train_losses' in record_result:
                y_data = record_result['train_losses']
            else:
                y_data = record_result['losses']
            
#             print('x_axis_name')
#             print(x_axis_name)
            
#             if x_axis_name == 'epoch':
#                 print('algorithm')
#                 print(algorithm)
#                 print('np.min(y_data)')
#                 print(np.min(y_data))
        
        elif y_axis_name == 'training unregularized loss':
            
            y_data = record_result['train_unregularized_losses']
            
        elif y_axis_name == 'training unregularized minibatch loss':
            
            y_data = record_result['train_unregularized_minibatch_losses']
            
        elif y_axis_name == 'training minibatch error':
            
#             print('get_name_loss(name_dataset)')
#             print(get_name_loss(name_dataset))
            
#             sys.exit()
        
            if get_name_loss(name_dataset) == 'multi-class classification':
                y_data = 1 - record_result['train_minibatch_acces']
            else:
                print('error: need to check')
                sys.exit()
                
        elif y_axis_name == 'training error':
            
#             print('get_name_loss(name_dataset)')
#             print(get_name_loss(name_dataset))
            
#             sys.exit()
            
            if get_name_loss(name_dataset) == 'multi-class classification':
                
                print('record_result.keys()')
                print(record_result.keys())
                
                assert 'train_acces' in record_result
                
                y_data = 1 - record_result['train_acces']
                
#                 sys.exit()
                
            else:
                print('error: need to check')
                sys.exit()

        elif y_axis_name == 'testing error':
            

            
            if get_name_loss(name_dataset) == 'multi-class classification':
                if 'test_acces' in record_result:
                    y_data = 1 - record_result['test_acces']
                else:
                    y_data = 1 - record_result['acces']
            elif get_name_loss(name_dataset) in ['logistic-regression-sum-loss',
                                                 'linear-regression-half-MSE']:
                if 'test_acces' in record_result:
                    y_data = record_result['test_acces']
                else:
                    y_data = record_result['acces']
            else:
                print('error: need to check for ' + get_name_loss(name_dataset))
                sys.exit()
            
                if name_dataset in ['MNIST-autoencoder',
                                    'MNIST-autoencoder-no-regularization',
                                    'MNIST-autoencoder-N1-1000',
                                    'MNIST-autoencoder-N1-1000-sum-loss',
                                    'MNIST-autoencoder-N1-1000-no-regularization',
                                    'MNIST-autoencoder-N1-1000-sum-loss-no-regularization',
                                    'MNIST-autoencoder-relu-N1-1000-sum-loss-no-regularization',
                                    'MNIST-autoencoder-relu-N1-1000-sum-loss',
                                    'CURVES-autoencoder',
                                    'CURVES-autoencoder-no-regularization',
                                    'CURVES-autoencoder-sum-loss-no-regularization',
                                    'CURVES-autoencoder-relu-sum-loss-no-regularization',
                                    'CURVES-autoencoder-relu-sum-loss',
                                    'CURVES-autoencoder-sum-loss',
                                    'CURVES-autoencoder-Botev',
                                    'CURVES-autoencoder-Botev-sum-loss-no-regularization',
                                    'CURVES-autoencoder-shallow',
                                    'FACES-autoencoder',
                                    'FACES-autoencoder-no-regularization',
                                    'FACES-autoencoder-sum-loss-no-regularization',
                                    'FACES-autoencoder-relu-sum-loss-no-regularization',
                                    'FACES-autoencoder-relu-sum-loss',
                                    'FACES-autoencoder-sum-loss',
                                    'FacesMartens-autoencoder-relu',
                                    'sythetic-linear-regression']:
                    if 'test_acces' in record_result:
                        y_data = record_result['test_acces']
                    else:
                        y_data = record_result['acces']
                elif name_dataset in ['MNIST',
                                      'MNIST-no-regularization',
                                      'MNIST-N1-1000',
                                      'MNIST-one-layer',
                                      'DownScaledMNIST-no-regularization',
                                      'DownScaledMNIST-N1-1000-no-regularization',
                                      'webspam',
                                      'Fashion-MNIST',
                                      'Fashion-MNIST-N1-60',
                                      'Fashion-MNIST-N1-60-no-regularization',
                                      'CIFAR',
                                      'CIFAR-deep',
                                      'UCI-HAR']:
                    if 'test_acces' in record_result:
                        y_data = 1 - record_result['test_acces']
                    else:
                        y_data = 1 - record_result['acces']
                else:
                    print('Error: need check name_loss')
                    sys.exit()
                

            if name_dataset == 'MNIST':
                ax.set_ylim([0.01, 0.1])
        else:
            print('Error! need to check for ' + y_axis_name)
            sys.exit()

        
#         print('args[if_lr_in_legend]')
#         print(args['if_lr_in_legend'])
#         sys.exit()
        
        if args['if_lr_in_legend']:

            name_legend = algorithm_dict['legend'] +\
        ', lr = ' + str(best_alpha)
        else:
            name_legend = algorithm_dict['legend']

        if x_axis_name == 'cpu time':
#             time = record_result['timesCPU']
            x_data = record_result['timesCPU']
#             ax.plot(time, y_data, label=name_legend)
#             ax.plot(x_data, y_data, label=name_legend)
        elif x_axis_name == 'wall clock time':
#             time = record_result['timesWallClock']
            x_data = record_result['timesWallClock']
#             ax.plot(time, y_data, label=name_legend)
#             ax.plot(x_data, y_data, label=name_legend)
        elif x_axis_name == 'epoch':
#             epochs = record_result['epochs']
            x_data = record_result['epochs']
#             ax.plot(epochs, y_data, label=name_legend)
#             ax.plot(x_data, y_data, label=name_legend)
        else:
            print('Error.')
            sys.exit()
            
#         print('args[color]')
#         print(args['color'])
        
#         sys.exit()
        
        if args['color'] == None:
            ax.plot(x_data, y_data, label=name_legend)
        else:
            
#             print('i')
#             print(i)
            
#             sys.exit()
            
            
            ax.plot(x_data, y_data, args['color'][i], label=name_legend)
            
    if x_axis_name == 'cpu time':
        plt.xlabel('process time (second)', fontsize=20)
#         plt.xlabel('CPU times (second)')
    elif x_axis_name in ['epoch', 'wall clock time']:
        plt.xlabel(x_axis_name, fontsize=20)
    else:
        print('error: need to check x_axis_name for ' + x_axis_name)
        sys.exit()
        

    
        
    
    
    
    
    if y_axis_name == 'training unregularized minibatch loss':
        y_axis_name_legend = 'training loss'
    elif y_axis_name == 'testing error':
        y_axis_name_legend = 'testing error'
    else:
        print('y_axis_name')
        print(y_axis_name)

        sys.exit()
    
        

    plt.ylabel(y_axis_name_legend, fontsize=20)
    
#     print('args')
#     print(args)
    
#     print('args[if_title]')
#     print(args['if_title'])
    
#     sys.exit()
    
    if args['if_title']:
        plt.suptitle(name_dataset_legend)
    
    plt.grid(True)
    
    
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

def get_plot(name_dataset, name_dataset_legend, algorithms, args):
    
    from utils_git.utils import from_dataset_to_N1_N2
    
#     print('args')
#     print(args)
    
    args['dataset'] = name_dataset
    args = from_dataset_to_N1_N2(args)
    
    
    list_x = args['list_x']

    list_y = args['list_y']
    
#     print('len(list_x)')
#     print(len(list_x))
    
#     print('len(list_y)')
#     print(len(list_y))
    
#     sys.exit()
    
    

#     plt.figure(figsize=(2*5,2*5))
    plt.figure(figsize=(len(list_x)*5,len(list_y)*5))

    index_subplot = 0
    for name_y in list_y:
        for name_x in list_x:
            index_subplot += 1
            ax = plt.subplot(len(list_y), len(list_x), index_subplot)
            get_subplot(ax, name_dataset, name_dataset_legend, algorithms, name_x, name_y, args)
            
#     print('args[if_show_legend]')
#     print(args['if_show_legend'])
#     sys.exit()

    if args['if_show_legend']:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    
    path_to_func_dir = args['home_path'] + 'logs/get_subplot_' + args['tuning_criterion'] + '/'

    if not os.path.exists(path_to_func_dir):
            os.makedirs(path_to_func_dir)
    if not os.path.exists(path_to_func_dir + name_dataset + '/'):
            os.makedirs(path_to_func_dir + name_dataset + '/')

    

    plt.savefig(path_to_func_dir + name_dataset + '/' +\
                str(datetime.datetime.now().strftime('%Y-%m-%d-%X')) + '.pdf', bbox_inches='tight')

    plt.show()


def plot_damping_status(args):
    
    path_to_file = args['path_to_file']
    
    print('path_to_file')
    print(path_to_file)
    
    # get some params
    home_path = path_to_file.split('result')[0]
    name_result = path_to_file.split('result')[1]
    name_result = name_result.split('/')
    
    dataset = name_result[1]
    algorithm = name_result[2]
    lr = name_result[4]
    
    if dataset == 'CURVES-autoencoder-relu-sum-loss':
        dataset_legend = 'CURVES'
    elif dataset == 'MNIST-autoencoder-relu-N1-1000-sum-loss':
        dataset_legend = 'MNIST'
    elif dataset == 'FacesMartens-autoencoder-relu':
        dataset_legend = 'FACES'
    else:
        print('error: no dataset_legend for ' + dataset)
        sys.exit()
        
#     print('algorithm')
#     print(algorithm)
#     sys.exit()
    
    if algorithm == 'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad':
        algorithm_legend = 'K-BFGS(L)'
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad':
        algorithm_legend = 'K-BFGS'
    else:
        print('error: algorithm_legend for ' + algorithm)
        sys.exit()

    


    min_index = 0
    max_index = 200000000000000

    with open(path_to_file, 'rb') as filename:
        data_ = pickle.load(filename)

    plt.plot(np.arange(len(data_['losses_per_iter']))[min_index: max_index],
                 data_['losses_per_iter'][min_index: max_index])

    plt.plot(np.linspace(
            0,
            len(data_['losses_per_iter']),
            len(data_['train_losses'])
                          )[min_index: max_index],
             data_['train_losses'][min_index: max_index])

    plt.yscale('log')
    plt.show()
    
    
#     list_damping = data_['kron_bfgs_damping_statuses'].keys()
#     list_damping = list(list_damping)
    list_damping = []
    
    if 'kron_bfgs_check_dampings' in data_:
#         num_subplot = len(list_damping)+1+1
        num_subplot = len(list_damping)+2
    else:
        num_subplot = len(list_damping)
        
#     print('num_subplot')
#     print(num_subplot)
#     sys.exit()
    

    plt.figure(figsize=(8,16))
    
    plt.figure(figsize=(8, 4 * num_subplot))
    
#     index_subplot = 0
    index_subplot = -1
    while 1:
        if index_subplot == num_subplot - 1:
            break
            
        
        
        index_subplot += 1
        
        print('num_subplot')
        print(num_subplot)
        print('index_subplot+1')
        print(index_subplot+1)
        
        ax = plt.subplot(num_subplot, 1, index_subplot+1)
        
#         print('index_subplot-1')
#         print(index_subplot-1)
        
#         sys.exit()
        
        if index_subplot < len(list_damping):
            
            name_damping = list_damping[index_subplot-1]

            np_kron_bfgs_damping_statuses =\
            data_['kron_bfgs_damping_statuses'][name_damping]
            
            np_kron_bfgs_damping_statuses =\
            np.asarray(np_kron_bfgs_damping_statuses)
        elif index_subplot == len(list_damping):
            
#             name_damping = 'ratio of inequality holds'
            name_damping = 'Fraction of Iters. inequality hold'
            
            np_kron_bfgs_damping_statuses =\
            data_['kron_bfgs_check_dampings']

            np_kron_bfgs_damping_statuses =\
            np.asarray(np_kron_bfgs_damping_statuses)

#             alpha = data_['params']['Kron_BFGS_H_epsilon']
            alpha = 0.2
            
            print('alpha')
            print(alpha)
            
            print('data_[params][Kron_BFGS_H_epsilon]')
            print(data_['params']['Kron_BFGS_H_epsilon'])

            np_kron_bfgs_damping_statuses = np_kron_bfgs_damping_statuses >= alpha / 2
            
        elif index_subplot == len(list_damping)+1:
            
#             name_damping = 'yHy/sy'
            name_damping = 'Average value of yHy/sy'
            
            np_kron_bfgs_damping_statuses =\
            data_['kron_bfgs_check_dampings'] # sy/yHy

            np_kron_bfgs_damping_statuses =\
            np.asarray(np_kron_bfgs_damping_statuses)
            
            np_kron_bfgs_damping_statuses = 1 / np_kron_bfgs_damping_statuses

            alpha = data_['params']['Kron_BFGS_H_epsilon']

#             np_kron_bfgs_damping_statuses = np_kron_bfgs_damping_statuses >= alpha / 2
            


    #     np_kron_bfgs_damping_statuses = np_kron_bfgs_damping_statuses['Powell-H-damping']


        L = np_kron_bfgs_damping_statuses.shape[1]

        if dataset in ['FACES-autoencoder-sum-loss-no-regularization',
                       'FacesMartens-autoencoder-relu']:
            iter_per_epoch = 103
        elif dataset in ['MNIST-autoencoder-N1-1000-sum-loss-no-regularization',
                         'MNIST-autoencoder-relu-N1-1000-sum-loss']:
            iter_per_epoch = 60
        elif dataset in ['CURVES-autoencoder-sum-loss-no-regularization',
                         'CURVES-autoencoder-relu-sum-loss-no-regularization',
                         'CURVES-autoencoder-relu-sum-loss']:
            iter_per_epoch = 20
        else:
            print('error: no iter_per_epoch for ' + dataset)
            sys.exit()

        np_kron_bfgs_damping_statuses =\
        np_kron_bfgs_damping_statuses.reshape(-1, iter_per_epoch, L)

        np_kron_bfgs_damping_statuses =\
            np.mean(np_kron_bfgs_damping_statuses, axis=1)

    #     print(np_kron_bfgs_damping_statuses.shape)


        average_np_kron_bfgs_damping_statuses =\
        np.mean(np_kron_bfgs_damping_statuses, axis=1)

    #     sys.exit()



        for l in range(L):
            if l > 1 and l < L-2:
                continue



            np_kron_bfgs_damping_statuses_l =\
            np_kron_bfgs_damping_statuses[:, l]




    #         np_kron_bfgs_damping_statuses_l =\
    #         np_kron_bfgs_damping_statuses_l.reshape(-1, 103)

    #         np_kron_bfgs_damping_statuses_l =\
    #         np.mean(np_kron_bfgs_damping_statuses_l, axis=1)

            ax.plot(
                np.arange(
                len(np_kron_bfgs_damping_statuses_l)
            )[min_index: max_index]+1,
                np_kron_bfgs_damping_statuses_l[min_index: max_index],
                label='l = {}'.format(l)
            )




        ax.plot(
                np.arange(
                len(average_np_kron_bfgs_damping_statuses)
            )[min_index: max_index]+1,
                average_np_kron_bfgs_damping_statuses[min_index: max_index],
                label='average'
            )

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xlabel('epoch')
        
#         ax.set_ylabel('ratio of ' + name_damping)
        ax.set_ylabel(name_damping)




        # add title
#         text_title = dataset_legend + '\n' +\
#         algorithm[:int(len(algorithm)/2)] + '\n' + algorithm[int(len(algorithm)/2):] +\
#         '\n' + lr
        
        text_title = dataset_legend + ', ' + algorithm_legend

        if index_subplot == len(list_damping):
#         plt.suptitle(text_title)
            plt.title(text_title)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
#     plt.subplots_adjust(top=0.85)
    
    
    # save to disk
    saved_path = home_path + 'logs/'
    saved_path = path_to_file.replace('result/', 'logs/plot_damping_status/') + '.pdf'

    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    if os.path.isdir(saved_path): 
        os.rmdir(saved_path)
    
    plt.tight_layout()
    plt.savefig(saved_path)
    
    plt.show()