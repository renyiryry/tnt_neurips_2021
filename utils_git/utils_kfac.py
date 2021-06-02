import torch
import torch.nn.functional as F
import sys
import math
import copy

from utils_git.utils import get_homo_grad
from utils_git.utils import get_opposite
from utils_git.utils import from_homo_to_weight_and_bias
from utils_git.utils import get_A_A_T

list_algorithm = ['ekfac-EF-VA',
                 'ekfac-EF',
                 'kfac-TR',
                 'kfac-momentum-grad-TR',
                 'kfac-CG',
                 'kfac-momentum-grad-CG',
                 'kfac',
                 'kfac-no-max',
                 'kfac-NoMaxNoSqrt',
                 'kfac-NoMaxNoSqrt-no-LM',
                 'kfac-no-max-no-LM',
                 'kfac-warmStart-no-max-no-LM',
                 'kfac-warmStart-lessInverse-no-max-no-LM',
                 'kfac-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                 'kfac-correctFisher-warmStart-no-max-no-LM',
                 'kfac-correctFisher-warmStart-NoMaxNoSqrt-no-LM',
                 'kfac-correctFisher-warmStart-lessInverse-no-max-no-LM',
                 'kfac-correctFisher-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                 'kfac-no-max-epsilon-A-G-no-LM',
                 'kfac-EF']

def get_saved_params_kfac(params):
    
    params['keys_params_saved'].append('kfac_if_svd')
    
    params['keys_params_saved'].append('kfac_if_update_BN')
    params['keys_params_saved'].append('kfac_if_BN_grad_direction')
    
    if params['algorithm'] in ['kfac-TR',
                               'kfac-momentum-grad-TR']:
        params['keys_params_saved'].append('TR_max_iter')
        
    if params['algorithm'] in ['kfac-CG', 
                               'kfac-momentum-grad-CG']:
        params['keys_params_saved'].append('CG_max_iter')
        
    if params['algorithm'] in ['kfac',
                               'kfac-momentum-grad',
                               'kfac-no-max',
                               'kfac-NoMaxNoSqrt',
                               'kfac-NoMaxNoSqrt-no-LM',
                               'kfac-no-max-no-LM',
                               'kfac-warmStart-no-max-no-LM',
                               'kfac-correctFisher-warmStart-no-max-no-LM',
                               'kfac-correctFisher-warmStart-NoMaxNoSqrt-no-LM',
                               'kfac-correctFisher-warmStart-lessInverse-no-max-no-LM',
                               'kfac-correctFisher-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                               'kfac-warmStart-lessInverse-no-max-no-LM',
                               'kfac-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                               'kfac-no-max-epsilon-A-G-no-LM',
                               'kfac-no-max-momentum-grad']:
        params['keys_params_saved'].append('kfac_inverse_update_freq')
        params['keys_params_saved'].append('kfac_cov_update_freq')
        
    if params['algorithm'] in ['kfac-no-max-no-LM',
                               'kfac-warmStart-no-max-no-LM',
                               'kfac-correctFisher-warmStart-no-max-no-LM',
                               'kfac-correctFisher-warmStart-NoMaxNoSqrt-no-LM',
                               'kfac-correctFisher-warmStart-lessInverse-no-max-no-LM',
                               'kfac-correctFisher-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                               'kfac-warmStart-lessInverse-no-max-no-LM',
                               'kfac-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                               'kfac-NoMaxNoSqrt-no-LM']:
        params['keys_params_saved'].append('kfac_damping_lambda')
        
    if params['algorithm'] == 'kfac-no-max-epsilon-A-G-no-LM':
        params['keys_params_saved'].append('kfac_A_epsilon')
        params['keys_params_saved'].append('kfac_G_epsilon')
    
    return params

def get_g_g_T_BN(model, l, batch_size):
    # In kfac code (https://github.com/tensorflow/kfac), they use a "sum the squares estimator"
    # (see the class ScaleAndShiftFullFB in https://github.com/tensorflow/kfac/blob/master/kfac/python/ops/fisher_blocks.py)
    # For more detail, see ScaleAndShiftFactor in https://github.com/tensorflow/kfac/blob/master/kfac/python/ops/fisher_factors.py
    
    # However, it is difficult to cache the intermediate variable of BN layer in pytorch.
    # Hence, we decide to use a "square the sum estimator", for simplicity
    
    g = torch.cat((model.layers_weight[l]['W'].grad.data, model.layers_weight[l]['b'].grad.data))
                    
    # g is averaged over minibatch
    # should first * batch_size, take outer product, then / batch_size 
    # i.e. "square the sum estimator" in kfac code
    # which is equivalent to batch_size * (g g^T)
    G_j = batch_size * torch.outer(g, g)
    
    return G_j

def kfac_if_inverse(params):
    
    i = params['i']
    inverse_update_freq = params['kfac_inverse_update_freq']
    cov_update_freq = params['kfac_cov_update_freq']
    
    if params['algorithm'] in ['kfac-TR', 'kfac-CG']:
        return False
    
    if params['algorithm'] in ['kfac-warmStart-lessInverse-no-max-no-LM',
                               'kfac-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                               'kfac-correctFisher-warmStart-lessInverse-no-max-no-LM',
                               'kfac-correctFisher-warmStart-lessInverse-NoMaxNoSqrt-no-LM',]:
        
        if i % inverse_update_freq == 0:
            return True
        else:
            return False
        
    elif params['algorithm'] in ['kfac-warmStart-no-max-no-LM',
                                 'kfac-correctFisher-warmStart-no-max-no-LM',
                                 'kfac-correctFisher-warmStart-NoMaxNoSqrt-no-LM',]:
        if (i <= inverse_update_freq and i % cov_update_freq == 0) or i % inverse_update_freq == 0:
            return True
        else:
            return False
    else:
        print('error: need to check for ' + params['algorithm'])
        sys.exit()
    
        

def kfac_update(data_, params):
    true_algorithm = params['algorithm']
    if params['algorithm'] == 'kfac-EF':
        params['algorithm'] = 'kfac'
    elif params['algorithm'] == 'kfac-momentum-grad-TR':
        params['algorithm'] = 'kfac-TR'
    elif params['algorithm'] == 'kfac-momentum-grad-CG':
        params['algorithm'] = 'kfac-CG'
        
    i = params['i']
        
    if i == 0:
        params['kfac_svd_failed'] = False

    A = data_['A']
    G = data_['G']
    model = data_['model']
    
#     model_grad = data_['model_regularized_grad_used_torch']
    model_grad = data_['model_grad_used_torch']
    
#     model_grad_N1 = data_['model_regularized_grad_torch']
    model_grad_N1 = data_['model_grad_torch']

    if params['algorithm'] == 'ekfac-EF-VA':
        U_A = data_['U_A']
        U_G = data_['U_G']
        ekfac_s = data_['ekfac_s']
        ekfac_m = data_['ekfac_m']
    elif params['algorithm'] == 'ekfac-EF':
        U_A = data_['U_A']
        U_G = data_['U_G']
        ekfac_s = data_['ekfac_s']
    elif params['algorithm'] in ['kfac',
                                 'kfac-test',
                                 'kfac-no-max',
                                 'kfac-NoMaxNoSqrt',
                                 'kfac-NoMaxNoSqrt-no-LM',
                                 'kfac-no-max-no-LM',
                                 'kfac-warmStart-no-max-no-LM',
                                 'kfac-warmStart-lessInverse-no-max-no-LM',
                                 'kfac-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                                 'kfac-correctFisher-warmStart-no-max-no-LM',
                                 'kfac-correctFisher-warmStart-NoMaxNoSqrt-no-LM',
                                 'kfac-correctFisher-warmStart-lessInverse-no-max-no-LM',
                                 'kfac-correctFisher-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                                 'kfac-no-max-epsilon-A-G-no-LM']:
        
        if params['kfac_if_svd']:
            U_A = data_['U_A']
            U_G = data_['U_G']
            s_A = data_['s_A']
            s_G = data_['s_G']
        else:
        
            A_inv = data_['A_inv']
            G_inv = data_['G_inv']
    elif params['algorithm'] in ['kfac-TR',
                                 'kfac-CG']:
        1
    else:
        print('Error: unknown algo in kfac_update')
        sys.exit()

    
    
    N1 = params['N1']
    N2 = params['N2']
    
    
    
    if params['algorithm'] in ['kfac',
                               'kfac-no-max',
                               'kfac-NoMaxNoSqrt']:
        lambda_ = params['lambda_']
        lambda_A = math.sqrt(lambda_)
        lambda_G = math.sqrt(lambda_)
        
    elif params['algorithm'] in ['kfac-no-max-no-LM',
                                 'kfac-warmStart-no-max-no-LM',
                                 'kfac-correctFisher-warmStart-no-max-no-LM',
                                 'kfac-correctFisher-warmStart-NoMaxNoSqrt-no-LM',
                                 'kfac-correctFisher-warmStart-lessInverse-no-max-no-LM',
                                 'kfac-correctFisher-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                                 'kfac-warmStart-lessInverse-no-max-no-LM',
                                 'kfac-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                                 'kfac-NoMaxNoSqrt-no-LM']:
        lambda_ = params['kfac_damping_lambda']
        lambda_A = math.sqrt(lambda_)
        lambda_G = math.sqrt(lambda_)
    elif params['algorithm'] == 'kfac-no-max-epsilon-A-G-no-LM':
#         lambda_ = params['kfac_damping_lambda']
        lambda_A = params['kfac_A_epsilon']
        lambda_G = params['kfac_G_epsilon']
    else:
        print('error: need to check lambda for ' + params['algorithm'])
        sys.exit()
        
    
    
#     alpha = params['alpha']
    numlayers = params['numlayers']
    kfac_rho = params['kfac_rho']

    device = params['device']
    
    
    
#     a_grad_N2 = data_['a_grad_N2']
    h_N2 = data_['h_N2']
    
    # h denotes the bar_a in kfac paper, a_grad denotes the g
    G_ = []
    A_ = []
    

        
    if params['algorithm'] in ['kfac-warmStart-no-max-no-LM',
                               'kfac-correctFisher-warmStart-no-max-no-LM',
                               'kfac-correctFisher-warmStart-NoMaxNoSqrt-no-LM',
                               'kfac-correctFisher-warmStart-lessInverse-no-max-no-LM',
                               'kfac-correctFisher-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                               'kfac-warmStart-lessInverse-no-max-no-LM',
                               'kfac-warmStart-lessInverse-NoMaxNoSqrt-no-LM',]:
        rho = kfac_rho
    elif params['algorithm'] == 'kfac-no-max-no-LM':
        rho = min(1-1/(i+1), kfac_rho)
    else:
        print('error: not implemented for ' + params['algorithm'])
        sys.exit()


    # used in ekfac
    homo_model_grad_N1 = get_homo_grad(model_grad_N1, params)
    
    if params['if_momentum_gradient']:
        homo_model_grad = get_homo_grad(model_grad, params)
    else:
        homo_model_grad = homo_model_grad_N1

    if params['algorithm'] in ['kfac-TR',
                               'kfac-CG']:
        if i > 0:
            minus_p_prev = get_opposite(data_['p_torch'])
        else:
            minus_p_prev = data_['model_regularized_grad_used_torch']
        homo_minus_p_prev = get_homo_grad(minus_p_prev, params)
        
    cov_update_freq = params['kfac_cov_update_freq']
        
    # Step
    delta = []
    for l in range(numlayers):
        
        if i % cov_update_freq == 0:
        
            if params['layers_params'][l]['name'] in ['fully-connected',
                                                      'conv',
                                                      'conv-no-activation',
                                                      'conv-no-bias-no-activation']:

                G_.append(get_g_g_T(data_['a_N2'], l, params))

                # no need to save A_, can be improved
                A_.append(get_A_A_T(data_['h_N2'], l, data_, params))

                # Update running estimates of KFAC
                A[l].data = rho*A[l].data + (1-rho)*A_[l].data
                G[l].data = rho*G[l].data + (1-rho)*G_[l].data

            elif params['layers_params'][l]['name'] in ['BN']:

                A_.append([])
                
                if params['kfac_if_update_BN'] and not params['kfac_if_BN_grad_direction']:

                    G_.append(get_g_g_T_BN(model, l, N2))

                    G[l].data = rho*G[l].data + (1-rho)*G_[l].data
                    
                else:
                    G_.append([])
            else:
                print('Error: unknown layer when compute A: ' + params['layers_params'][l]['name'])
                sys.exit()

            
        

        
        # Amortize the inverse. Only update inverses every now and then
        if kfac_if_inverse(params):

            # phi_ = np.sqrt( ( np.trace(A[l].cpu().data.numpy()) / A[l].shape[0] )\
                # / np.maximum( np.trace(G[l].cpu().data.numpy()) / G[l].shape[0], 10**(-3) ) )
            if params['algorithm'] in ['kfac']:
        
                phi_ = torch.sqrt(
                           ( torch.sum(torch.diag(A[l])) / A[l].size()[0] )\
                / max( torch.sum(torch.diag(G[l])) / G[l].size()[0], 10**(-3) ) )
            elif params['algorithm'] in ['kfac-no-max',
                                         'kfac-no-max-no-LM',
                                         'kfac-no-max-epsilon-A-G-no-LM',
                                         'kfac-warmStart-no-max-no-LM',
                                         'kfac-warmStart-lessInverse-no-max-no-LM',
                                         'kfac-correctFisher-warmStart-no-max-no-LM',
                                         'kfac-correctFisher-warmStart-lessInverse-no-max-no-LM',]:
                
                phi_ = 1
                
            elif params['algorithm'] in ['kfac-NoMaxNoSqrt',
                                         'kfac-NoMaxNoSqrt-no-LM',
                                         'kfac-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                                         'kfac-correctFisher-warmStart-NoMaxNoSqrt-no-LM',
                                         'kfac-correctFisher-warmStart-lessInverse-NoMaxNoSqrt-no-LM',]:
                if params['layers_params'][l]['name'] != 'BN':
                    phi_ = torch.sqrt(
                           (torch.sum(torch.diag(A[l])) / A[l].size()[0] )\
                            / torch.sum(torch.diag(G[l])) / G[l].size()[0] 
                           )
                
            else:
                print('error: not implemented')
                sys.exit()
                        
            
            if not params['kfac_if_svd']:

                if params['layers_params'][l]['name'] == 'BN':
                    if params['kfac_if_update_BN'] and not params['kfac_if_BN_grad_direction']:
                        G_l_LM = G[l] + (1 / lambda_) * torch.eye(G[l].size()[0], device=device)
                else:

                    A_l_LM = A[l] + (phi_ * lambda_A) * torch.eye(A[l].size()[0], device=device)

                    G_l_LM = G[l] + (1 / phi_ * lambda_G) * torch.eye(G[l].size()[0], device=device)

            if params['algorithm'] in ['kfac',
                                       'kfac-no-max',
                                       'kfac-NoMaxNoSqrt',
                                       'kfac-NoMaxNoSqrt-no-LM',
                                       'kfac-no-max-no-LM',
                                       'kfac-warmStart-no-max-no-LM',
                                       'kfac-correctFisher-warmStart-no-max-no-LM',
                                       'kfac-correctFisher-warmStart-NoMaxNoSqrt-no-LM',
                                       'kfac-correctFisher-warmStart-lessInverse-no-max-no-LM',
                                       'kfac-correctFisher-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                                       'kfac-warmStart-lessInverse-no-max-no-LM',
                                       'kfac-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                                       'kfac-no-max-epsilon-A-G-no-LM']:
                # inverse() is not working properly on gpu, so use cpu
                # A_l_LM_cpu = A_l_LM.cpu()
                # A_inv_l_cpu = A_l_LM_cpu.inverse()
                # A_inv[l] = A_inv_l_cpu.to(device)

                # G_l_LM_cpu = G_l_LM.cpu()
                # G_inv_l_cpu = G_l_LM_cpu.inverse()
                # G_inv[l] = G_inv_l_cpu.to(device)
                
                if params['kfac_if_svd']:
                    
                    if params['layers_params'][l]['name'] == 'BN':
                        
                        if params['kfac_if_update_BN'] and not params['kfac_if_BN_grad_direction']:
                            
                            try:
                                s_G[l], U_G[l] = torch.symeig(G[l].data, eigenvectors=True)
                            except:
        #                         s_G[l], U_G[l] = torch.symeig(G[l].data.cpu(), eigenvectors=True)
        #                         s_G[l], U_G[l] = s_G[l].to(device), U_G[l].to(device)

                                print('svd faild in G')

                                params['kfac_svd_failed'] = True
                        
                    else:
                    
                        # U * diag{s} * U.t() = A (or G)
    #                     s_A[l], U_A[l] = torch.symeig(A[l].data, eigenvectors=True)
    #                     s_G[l], U_G[l] = torch.symeig(G[l].data, eigenvectors=True)

                        try:
                            s_A[l], U_A[l] = torch.symeig(A[l].data, eigenvectors=True)

    #                         s_A[l] = s_A[l] * (s_A[l] > 1e-10).float()

                        except:
    #                         s_A[l], U_A[l] = torch.symeig(A[l].data.cpu(), eigenvectors=True)

    #                         s_A[l], U_A[l] = s_A[l].to(device), U_A[l].to(device)

                            print('l')
                            print(l)

                            print('svd faild in A')


                            params['kfac_svd_failed'] = True

                        try:
                            s_G[l], U_G[l] = torch.symeig(G[l].data, eigenvectors=True)
                        except:
    #                         s_G[l], U_G[l] = torch.symeig(G[l].data.cpu(), eigenvectors=True)
    #                         s_G[l], U_G[l] = s_G[l].to(device), U_G[l].to(device)

                            print('svd faild in G')

                            params['kfac_svd_failed'] = True
                    
                else:
                
                    if params['layers_params'][l]['name'] == 'BN':
                        if params['kfac_if_update_BN'] and not params['kfac_if_BN_grad_direction']:
                            G_inv[l] = G_l_LM.inverse()
                    else:
                        A_inv[l] = A_l_LM.inverse()
                        G_inv[l] = G_l_LM.inverse()
            elif params['algorithm'] == 'ekfac-EF-VA' or\
            params['algorithm'] == 'ekfac-EF':

                # symeig is used in ekfac's code:
                # https://github.com/Thrandis/EKFAC-pytorch/blob/master/ekfac.py
                # A = U_A * s * U_A.t()

                # use cpu to do eigen-decomp to match kfac
                # A_l_LM_cpu = A_l_LM.cpu()
                # _, U_A[l] = torch.symeig(A_l_LM_cpu, eigenvectors=True)

                # G_l_LM_cpu = G_l_LM.cpu()
                # _, U_G[l] = torch.symeig(G_l_LM_cpu, eigenvectors=True)

                # U_A[l] = U_A[l].to(device)
                # U_G[l] = U_G[l].to(device)

                _, U_A[l] = torch.symeig(A_l_LM, eigenvectors=True)
                _, U_G[l] = torch.symeig(G_l_LM, eigenvectors=True)

                
                
                
                
            else:
                print('Error: unknown algo when inverse')
                sys.exit()



        # update scaling
        if params['algorithm'] == 'ekfac-EF-VA' or\
        params['algorithm'] == 'ekfac-EF':
            if params['layers_params'][l]['name'] == 'fully-connected':
                # homo_s_l = torch.mm(torch.mm(U_G[l].t(), homo_model_grad), U_A[l].t())
                homo_s_l_N1 = torch.mm(torch.mm(U_G[l].t(), homo_model_grad_N1[l]), U_A[l].t())

                ekfac_s[l]['W'] = rho * ekfac_s[l]['W'] + (1 - rho) * (homo_s_l_N1[:, :-1]**2).data
                ekfac_s[l]['b'] = rho * ekfac_s[l]['b'] + (1 - rho) * (homo_s_l_N1[:, -1]**2).data

                if params['algorithm'] == 'ekfac-EF-VA':
                    ekfac_m[l]['W'] = rho * ekfac_m[l]['W'] + (1 - rho) * homo_s_l_N1[:, :-1].data
                    ekfac_m[l]['b'] = rho * ekfac_m[l]['b'] + (1 - rho) * homo_s_l_N1[:, -1].data

                # if np.isnan((ekfac_s[l]['W']).max()):
                    # sys.exit()
                    
                print('error: check why there is a to device')
                sys.exit()

                homo_s_l_N1 = homo_s_l_N1.to(device)

                # if torch.isnan(torch.max(homo_s_l)):
                    # sys.exit()
            else:
                print('Error: unsupported layer when update scaling')
                sys.exit()
            
        # compute kfac direction
        if params['algorithm'] in ['kfac',
                                   'kfac-no-max',
                                   'kfac-NoMaxNoSqrt',
                                   'kfac-NoMaxNoSqrt-no-LM',
                                   'kfac-no-max-no-LM',
                                   'kfac-warmStart-no-max-no-LM',
                                   'kfac-correctFisher-warmStart-no-max-no-LM',
                                   'kfac-correctFisher-warmStart-NoMaxNoSqrt-no-LM',
                                   'kfac-correctFisher-warmStart-lessInverse-no-max-no-LM',
                                   'kfac-correctFisher-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                                   'kfac-warmStart-lessInverse-no-max-no-LM',
                                   'kfac-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                                   'kfac-no-max-epsilon-A-G-no-LM']:
            
            if params['kfac_if_svd']:
                
                if params['layers_params'][l]['name'] == 'BN':
                    
                    if params['kfac_if_update_BN']:
                
                        if params['kfac_if_BN_grad_direction']:
                            homo_delta_l = copy.deepcopy(homo_model_grad[l])
                            
#                             if l == 1:
                            
#                                 print('homo_delta_l')
#                                 print(homo_delta_l)
                            
                        else:
                            print('error: not implemented')
                            sys.exit()
                        
                    else:
                        homo_delta_l = torch.zeros(homo_model_grad[l].size(), device=device)
                        
                    
                else:
                    homo_delta_l = torch.mm(
                        torch.mm(U_G[l].t(), homo_model_grad[l]),
                        U_A[l]
                    )

                    homo_delta_l = homo_delta_l / (torch.outer(s_G[l], s_A[l]) + params['kfac_damping_lambda'])

                    homo_delta_l = torch.mm(
                        torch.mm(U_G[l], homo_delta_l),
                        U_A[l].t()
                    )
                
                # G^{-1} * grad * A^{-1}
                # = (U_G * diag{s_G} * U_G.t())^{-1} * grad * (U_A * diag{s_A} * U_A.t())^{-1}
                # = U_G * diag{s_G}^{-1} * U_G.t() * grad * U_A * diag{s_A}^{-1} * U_A.t()
                
                
                
                
                
                
            else:
            
                if params['layers_params'][l]['name'] == 'BN':

                    if params['kfac_if_update_BN']:
                        
                        if params['kfac_if_BN_grad_direction']:
                            homo_delta_l = copy.deepcopy(homo_model_grad[l])
                        else:

                            homo_delta_l = torch.mv(G_inv[l], homo_model_grad[l])
                    else:
                        homo_delta_l = torch.zeros(homo_model_grad[l].size(), device=device)
                else:

                    homo_delta_l = torch.mm(torch.mm(G_inv[l], homo_model_grad[l]), A_inv[l])

                if params['if_test_mode']:
                    if 'if_record_kfac_G_inv_norm_per_iter' in params and\
                    params['if_record_kfac_G_inv_norm_per_iter']:

                        if l == 0:
                            data_['kfac_G_inv_norms_per_iter'].append([])

                        data_['kfac_G_inv_norms_per_iter'][-1].append(torch.norm(G_inv[l]).item())
            
            
        elif params['algorithm'] == 'kfac-TR':
            # objective function: 1 / 2 * x^T F x - dot(g, x)
            # constaint: ||x||^2 \le 1 / \lambda

            # use beck's method to compute global optimal
            # diferent initializations for easy case and hard case

            # objective function: (x.T).dot(hess).dot(x) / 2 + (grad.T).dot(x)

            # input:
            # hess: n * n
            # grad: n * 1

            # radius = 1. / lambda_
            radius = 1000. / lambda_
            
            # this configuration is for testing 
            # n = np.max(np.shape(hess))
            x_easy = homo_minus_p_prev[l]
            # x_hard = np.random.randn(n, 1)
            

            phi_ = torch.sqrt(\
                           ( torch.sum(torch.diag(A[l])) / A[l].size()[0] )\
                / max( torch.sum(torch.diag(G[l])) / G[l].size()[0], 10**(-3) ) )
            

            A_l_LM = A[l] + (phi_ * math.sqrt(lambda_)) * torch.eye(A[l].size()[0], device=device)
            G_l_LM = G[l] + (1 / phi_ * math.sqrt(lambda_)) * torch.eye(G[l].size()[0], device=device)

            # if torch.sum(G_l_LM != G_l_LM):
                # print('nan in G_l_LM')
                # print('G[l]')
                # print(G[l])
                # print('phi_')
                # print(phi_)
                # print('lambda_')
                # print(lambda_)
            
            iter_ = 0
            while iter_ < params['TR_max_iter']:
                
                iter_ = iter_ + 1

                # easy case
                # compute the gradient
                
                # temp_grad_easy = torch.mm(torch.mm(G[l], x_easy), A[l]) - homo_model_grad[l]
                temp_grad_easy =\
                torch.mm(torch.mm(G_l_LM, x_easy), A_l_LM) - homo_model_grad[l]

                if params['if_test_mode']:
                    if params['if_record_res_grad_norm_per_iter']:
                        if l == 0 and iter_ == 1:
                            data_['res_grad_norms_per_iter'].append([])
                        if iter_ == 1:
                            data_['res_grad_norms_per_iter'][-1].append([])

                        # print('data_[res_grad_norms_per_iter]')
                        # print(data_['res_grad_norms_per_iter'])

                        data_['res_grad_norms_per_iter'][-1][-1].append(
                            np.sqrt(torch.sum(temp_grad_easy**2).item())
                        )

                if params['if_test_mode']:
                    if torch.sum(temp_grad_easy != temp_grad_easy):
                        print('nan in temp_grad_easy')
                        print('G_l_LM')
                        print(G_l_LM)
                        print('x_easy')
                        print(x_easy)
                        print('A_l_LM')
                        print(A_l_LM)
                        print('homo_model_grad[l]')
                        print(homo_model_grad[l])
                    if torch.sum(temp_grad_easy == float('inf')):
                        print('inf in temp_grad_easy')
                        print('G_l_LM')
                        print(G_l_LM)
                        print('x_easy')
                        print(x_easy)
                        print('A_l_LM')
                        print(A_l_LM)
                        print('homo_model_grad[l]')
                        print(homo_model_grad[l])

                # lr_ = 1. / (iter_ + 10)
                lr_ = 1. / (torch.norm(G_l_LM) * torch.norm(A_l_LM))
                x_easy = x_easy - lr_ * temp_grad_easy
                
                
                # if l == 1:
                # print('l')
                # print(l)
                # print('iter_')
                # print(iter_)
                # print('1 / (iter_ + 10)')
                # print(1 / (iter_ + 10))
                # print('1 / (torch.norm(G_l_LM) * torch.norm(A_l_LM))')
                # print(1 / (torch.norm(G_l_LM) * torch.norm(A_l_LM)))
                # print('torch.norm(x_easy)')
                # print(torch.norm(x_easy))
                # print('torch.norm(temp_grad_easy)')
                # print(torch.norm(temp_grad_easy))
                # print('torch.sum(x_easy * torch.mm(torch.mm(G_l_LM, x_easy), A_l_LM))\
                        # - torch.sum(x_easy * homo_model_grad[l])')
                # print(torch.sum(x_easy * torch.mm(torch.mm(G_l_LM, x_easy), A_l_LM))\
                        # - torch.sum(x_easy * homo_model_grad[l]))

                if params['if_test_mode']:
                    if torch.sum(x_easy != x_easy):
                        print('nan in x_easy')
                        print('x_easy')
                        print(x_easy)
                        print('temp_grad_easy')
                        print(temp_grad_easy)

                '''
                temp_norm = torch.norm(x_easy)
                # projection
                if temp_norm > radius:
                    if temp_norm < 0.0001:
                        # restart if the norm is too small
                        x_easy = np.zeros((n, 1))
                    else:

                        x_easy = x_easy / temp_norm * radius
                '''

                

                # hard case
                # compute the gradient
                # temp_grad_hard = hess.dot(x_hard) + grad
                # x_hard = x_hard - temp_grad_hard / (iter + 10)
                # temp_norm = np.linalg.norm(x_hard)
                # if temp_norm > radius:
                    # x_hard = x_hard / temp_norm * radius
                
                # easy_value = (x_easy.T).dot(hess).dot(x_easy) / 2 + (grad.T).dot(x_easy)
                # hard_value = (x_hard.T).dot(hess).dot(x_hard) / 2 + (grad.T).dot(x_hard)
            
            # if easy_value < hard_value:
            homo_delta_l = x_easy
            # else:
                # return x_hard

            if params['if_test_mode']:
                if torch.sum(homo_delta_l!=homo_delta_l):
                    print('nan in homo_delta_l')
                    # sys.exit()

                if params['if_record_kfac_p_norm']:
                    kfac_homo_delta_l = torch.mm(
                        torch.mm(G_l_LM.cpu().inverse().to(device), homo_model_grad[l]),
                        A_l_LM.cpu().inverse().to(device))
                    if l == 0:
                        data_['kfac_p_norms'].append(
                            torch.sqrt(torch.sum(kfac_homo_delta_l**2)).item()
                        )
                    else:
                        data_['kfac_p_norms'][-1] =\
                        np.sqrt(data_['kfac_p_norms'][-1]**2 +\
                                torch.sum(kfac_homo_delta_l**2).item())
                if params['if_record_kfac_p_cosine']:
                    kfac_p_cosine = (torch.sum(kfac_homo_delta_l * homo_delta_l) /\
                            torch.norm(kfac_homo_delta_l) / torch.norm(homo_delta_l)).item()
                    if l == 0:
                        data_['kfac_p_cosines'].append([])
                    data_['kfac_p_cosines'][-1].append(kfac_p_cosine)
                if params['if_record_res_grad_norm']:
                    res_grad_norm = np.sqrt(torch.sum(temp_grad_easy**2).item())
                    if l == 0:
                        data_['res_grad_norms'].append([])
                    data_['res_grad_norms'][-1].append(res_grad_norm)
                if params['if_record_res_grad_random_norm']:
                    temp_grad_random = torch.mm(torch.mm(G_l_LM, torch.randn(x_easy.size(), device=device)), A_l_LM) - homo_model_grad[l]
                    res_grad_random_norm = np.sqrt(torch.sum(temp_grad_random**2).item())
                    if l == 0:
                        data_['res_grad_random_norms'].append([])
                    data_['res_grad_random_norms'][-1].append(res_grad_random_norm)
                if params['if_record_res_grad_grad_norm']:
                    temp_grad_grad = torch.mm(torch.mm(G_l_LM, homo_model_grad[l]), A_l_LM) - homo_model_grad[l]
                    res_grad_grad_norm = np.sqrt(torch.sum(temp_grad_grad**2).item())
                    if l == 0:
                        data_['res_grad_grad_norms'].append([])
                    data_['res_grad_grad_norms'][-1].append(res_grad_grad_norm)
                        

        elif params['algorithm'] == 'kfac-CG':
            phi_ = torch.sqrt(\
                           ( torch.sum(torch.diag(A[l])) / A[l].size()[0] )\
                / max( torch.sum(torch.diag(G[l])) / G[l].size()[0], 10**(-3) ) )
            

            A_l_LM = A[l] + (phi_ * math.sqrt(lambda_)) * torch.eye(A[l].size()[0], device=device)
            G_l_LM = G[l] + (1 / phi_ * math.sqrt(lambda_)) * torch.eye(G[l].size()[0], device=device)
            # A, G are without LM

            func_A = lambda x: torch.mm(torch.mm(G_l_LM, x), A_l_LM)
            # data_['G_l_LM'] = G_l_LM
            # data_['A_l_LM'] = A_l_LM
            homo_delta_l = get_CG(func_A, homo_model_grad[l], homo_minus_p_prev[l],
                                  params['CG_max_iter'], data_)




        elif params['algorithm'] == 'ekfac-EF-VA' or\
        params['algorithm'] == 'ekfac-EF':
            if params['if_momentum_gradient']:
                homo_s_l_used = homo_s_l_N1
            else:
                
                print('error: should U_A be transposed?')
                
                print('error: check why there is a to device')
                sys.exit()
                
                homo_s_l_used = torch.mm(torch.mm(U_G[l].t(), homo_model_grad[l]), U_A[l].t()).to(device)

            homo_delta_l = homo_s_l_used /\
            (torch.cat((ekfac_s[l]['W'], ekfac_s[l]['b'].unsqueeze(1)), dim=1) +\
             10**(-4))
            
            if params['algorithm'] == 'ekfac-EF-VA':
                homo_delta_l = homo_delta_l *\
                (torch.cat((ekfac_m[l]['W'],
                       ekfac_m[l]['b'].unsqueeze(1)), dim=1))**2

                       

            homo_delta_l = torch.mm(torch.mm(U_G[l], homo_delta_l), U_A[l])
        
        else:
            print('Error: unknown algo when compute direction')
            sys.exit()
            
        
        
        

#         # store the delta
#         delta_l = {}
#         if params['layers_params'][l]['name'] == 'fully-connected':
#             delta_l['W'] = homo_delta_l[:, :-1]
#             delta_l['b'] = homo_delta_l[:, -1]
#         elif params['layers_params'][l]['name'] == 'conv':
#             # take Fashion-MNIST as an example
#             # model_grad_N1[l]['W']: 32 * 1 * 5 * 5
#             # model_grad_N1[l]['b']: 32
#             # 32: conv_out_channels
#             # 1: conv_in_channels
#             # 5 * 5: conv_kernel_size
#             delta_l['b'] = homo_delta_l[:, -1]
#             delta_l['W'] = homo_delta_l[:, :-1].reshape(model_grad_N1[l]['W'].size())  
#         else:
#             print('Error: unsupported layer when store the data for ' + params['layers_params'][l]['name'])
#             sys.exit()
            
        delta_l = from_homo_to_weight_and_bias(homo_delta_l, l, params)
        
#         sys.exit()
        

            
            
        delta.append(delta_l)
        



    ##############
    algorithm = params['algorithm']
    if algorithm == 'Fisher-block':
        
        print('error: should not reach here; use SMW-Fisher-BD instead')
        sys.exit()
        
        delta = []
        for l in range(numlayers):
            
#             print('print(model_grad[l])', model_grad[l])

            params['algorithm'] = 'SMW-Fisher'
    
            data_test, _ = SMW_Fisher_update(data_, params)
    
#             print('SMW_Fisher_update(data_, params)', SMW_Fisher_update(data_, params))
            

            p = data_test['p']
        
            delta.append(-p)
        
            params['algorithm'] = 'Fisher-block'
        

    
    

    p = get_opposite(delta)
        

        
    
    
    

    
        
    data_['A'] = A
    data_['G'] = G 
    # A, G are without LM


    
    if params['algorithm'] == 'ekfac-EF-VA':
        data_['U_A'] = U_A
        data_['U_G'] = U_G
        data_['ekfac_s'] = ekfac_s
        data_['ekfac_m'] = ekfac_m
    elif params['algorithm'] == 'ekfac-EF':
        data_['U_A'] = U_A
        data_['U_G'] = U_G
        data_['ekfac_s'] = ekfac_s
    elif params['algorithm'] in ['kfac',
                                 'kfac-no-max',
                                 'kfac-NoMaxNoSqrt',
                                 'kfac-NoMaxNoSqrt-no-LM',
                                 'kfac-no-max-no-LM',
                                 'kfac-warmStart-no-max-no-LM',
                                 'kfac-correctFisher-warmStart-no-max-no-LM',
                                 'kfac-correctFisher-warmStart-NoMaxNoSqrt-no-LM',
                                 'kfac-correctFisher-warmStart-lessInverse-no-max-no-LM',
                                 'kfac-correctFisher-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                                 'kfac-warmStart-lessInverse-no-max-no-LM',
                                 'kfac-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                                 'kfac-no-max-epsilon-A-G-no-LM']:
        
        if params['kfac_if_svd']:
            
            data_['U_A'] = U_A
            data_['U_G'] = U_G
            data_['s_A'] = s_A
            data_['s_G'] = s_G
            
        else:
        
            data_['A_inv'] = A_inv
            data_['G_inv'] = G_inv
    elif params['algorithm'] in ['kfac-TR',
                                 'kfac-CG']:
        1
    else:
        print('Error: unknown algo when save momentum data')
        sys.exit()
    
    data_['p_torch'] = p

    if true_algorithm in ['kfac-EF',
                          'kfac-momentum-grad-CG',
                          'kfac-momentum-grad-TR']:
        params['algorithm'] = true_algorithm
        
        
    data_['a_grad_N2'] = None
    data_['a_N2'] = None
    data_['h_N2'] = None
    
    
        
    return data_, params

def get_g_g_T(a, l, params):
    # returns the AVERAGED g_g_T across a minibatch
    
    layers_params = params['layers_params']
    
    if layers_params[l]['name'] == 'fully-connected':
        
        size_minibatch = a[l].size(0)
        
        
        # we use "size_minibatch *", instead of "1/size_minibatch *"
        # because a[l].grad is actually "1/size_minibatch * a[l].grad"
        G_j = size_minibatch * torch.mm(a[l].grad.t(), a[l].grad).data
        
    elif layers_params[l]['name'] in ['conv',
                                      'conv-no-activation',
                                      'conv-no-bias-no-activation']:
        # take Fashion-MNIST as an example:
        # a[l]: 1000 * 32 * 28 * 28
        # 1000: size of minibatch
        # 32: # out-channel
        # 28 * 28: size of image
        
        # return 1 / |T| / size_minibatch * g_g_T
        

        
        size_minibatch = a[l].size(0)
        
        
        
        
        # a[l].grad is actually "1/size_minibatch * a_l_grad"
        a_l_grad = size_minibatch * a[l].grad
        
        a_l_grad_permuted = a_l_grad.permute(1, 0, 2, 3)
        
#         print('a_l_permuted.size()')
#         print(a_l_permuted.size())
        
        a_l_grad_flattened = torch.flatten(a_l_grad_permuted, start_dim=1)
        
        G_j = torch.mm(a_l_grad_flattened, a_l_grad_flattened.t()) / a_l_grad_flattened.size(1)
    else:
        print('error: not implemented for ' + layers_params[l]['name'])
        sys.exit()
        
    return G_j



