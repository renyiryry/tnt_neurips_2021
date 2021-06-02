import torch
import numpy as np
import sys
import os

from utils_git.utils import get_opposite, get_BFGS_formula_v2

list_algorithm = ['shampoo',
                 'shampoo-allVariables',
                 'shampoo-allVariables-warmStart',
                 'shampoo-allVariables-warmStart-lessInverse',
                 'shampoo-allVariables-filterFlattening-warmStart',
                 'shampoo-allVariables-filterFlattening-warmStart-lessInverse',
                 'shampoo-no-sqrt',
                 'shampoo-no-sqrt-Fisher',
                 'matrix-normal',
                 'matrix-normal-allVariables',
                 'matrix-normal-allVariables-warmStart',
                 'matrix-normal-allVariables-warmStart-MaxEigDamping',
                 'matrix-normal-allVariables-warmStart-noPerDimDamping',
                 'matrix-normal-same-trace',
                 'matrix-normal-same-trace-warmStart',
                 'matrix-normal-same-trace-warmStart-noPerDimDamping',
                 'matrix-normal-same-trace-allVariables',
                 'matrix-normal-same-trace-allVariables-warmStart',
                 'matrix-normal-same-trace-allVariables-warmStart-AvgEigDamping',
                 'matrix-normal-same-trace-allVariables-warmStart-MaxEigDamping',
                 'matrix-normal-same-trace-allVariables-filterFlattening-warmStart',
                 'matrix-normal-same-trace-allVariables-KFACReshaping-warmStart',
                 'matrix-normal-same-trace-allVariables-warmStart-noPerDimDamping'
                 'matrix-normal-correctFisher-allVariables-filterFlattening-warmStart-lessInverse',
                 'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart',
                 'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-lessInverse',
                 'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-MaxEigWithEpsilonDamping',
                 'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-AvgEigWithEpsilonDamping',
                 'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-TraceWithEpsilonDamping',
                 'matrix-normal-correctFisher-same-trace-allVariables-filterFlattening-warmStart',
                 'matrix-normal-correctFisher-same-trace-allVariables-filterFlattening-warmStart-lessInverse',
                 'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping',
                 'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart',
                 'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-lessInverse',]

def get_tensor_reshape_back(delta_l, l, name_variable, params):
    

    if get_tensor_reshape_option(l, name_variable, params) == 'filter-flattening':
        
        kernel_size = params['layers_params'][l]['conv_kernel_size']
        
#         import copy
        
#         delta_l_2 = copy.deepcopy(delta_l.view(delta_l.size(0), delta_l.size(1), kernel_size, kernel_size))
        
#         delta_l_3 = copy.deepcopy(delta_l_2.view(delta_l_2.size(0), delta_l_2.size(1), delta_l_2.size(2)*delta_l_2.size(3)))
        
#         print('torch.norm(delta_l_3 - delta_l)')
#         print(torch.norm(delta_l_3 - delta_l))
        
#         sys.exit()
        
        delta_l = delta_l.view(delta_l.size(0), delta_l.size(1), kernel_size, kernel_size)
        
    elif get_tensor_reshape_option(l, name_variable, params) == 'KFAC-reshaping':
        
        kernel_size = params['layers_params'][l]['conv_kernel_size']
        conv_in_channels = params['layers_params'][l]['conv_in_channels']
        
        delta_l = delta_l.view(delta_l.size(0), conv_in_channels, kernel_size, kernel_size)
        
    elif get_tensor_reshape_option(l, name_variable, params) == 'None':
        pass
        
    else:
        print('get_tensor_reshape_option(l, name_variable, params)')
        print(get_tensor_reshape_option(l, name_variable, params))
        sys.exit()
    
    return delta_l

# def get_if_tensor_reshape(l, name_variable, params):
def get_tensor_reshape_option(l, name_variable, params):
    
    
    
    if params['algorithm'] in ['shampoo-allVariables-filterFlattening-warmStart',
                               'shampoo-allVariables-filterFlattening-warmStart-lessInverse',
                               'matrix-normal-same-trace-allVariables-filterFlattening-warmStart',
                               'matrix-normal-same-trace-allVariables-KFACReshaping-warmStart',
                               'matrix-normal-correctFisher-allVariables-filterFlattening-warmStart-lessInverse',
                               'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart',
                               'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-lessInverse',
                               'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-MaxEigWithEpsilonDamping',
                               'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-AvgEigWithEpsilonDamping',
                               'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-TraceWithEpsilonDamping',
                               'matrix-normal-correctFisher-same-trace-allVariables-filterFlattening-warmStart',
                               'matrix-normal-correctFisher-same-trace-allVariables-filterFlattening-warmStart-lessInverse',
                               'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping',
                               'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart',
                               'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-lessInverse',]:
        
        if params['layers_params'][l]['name'] in ['conv',
                                                  'conv-no-activation',
                                                  'conv-no-bias-no-activation']:
        
            if name_variable == 'W': 
                
                
                
                if params['algorithm'] in ['matrix-normal-same-trace-allVariables-KFACReshaping-warmStart',
                                           'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart',
                                           'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-lessInverse',
                                           'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-MaxEigWithEpsilonDamping',
                                           'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-AvgEigWithEpsilonDamping',
                                           'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-TraceWithEpsilonDamping',
                                           'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping',
                                           'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart',
                                           'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-lessInverse',]:
                    return 'KFAC-reshaping'
                elif params['algorithm'] in ['shampoo-allVariables-filterFlattening-warmStart',
                                             'shampoo-allVariables-filterFlattening-warmStart-lessInverse',
                                             'matrix-normal-same-trace-allVariables-filterFlattening-warmStart',
                                             'matrix-normal-correctFisher-allVariables-filterFlattening-warmStart-lessInverse',
                                             'matrix-normal-correctFisher-same-trace-allVariables-filterFlattening-warmStart',
                                             'matrix-normal-correctFisher-same-trace-allVariables-filterFlattening-warmStart-lessInverse',]:
                    return 'filter-flattening'
                else:
                    print('params[algorithm]')
                    print(params['algorithm'])
                
                    sys.exit()
                
                    

            elif name_variable == 'b':
                return 'None'
                
            else:
                print('name_variable')
                print(name_variable)
                sys.exit()
            
            
        elif params['layers_params'][l]['name'] in ['fully-connected',
                                                    'BN']:
            return 'None'
            
        else:
            print('params[layers_params][l]')
            print(params['layers_params'][l])
            
            sys.exit()
            
    elif params['algorithm'] in ['matrix-normal-allVariables-warmStart-MaxEigDamping',
                                 'matrix-normal-same-trace-allVariables-warmStart',
                                 'matrix-normal-same-trace-allVariables-warmStart-AvgEigDamping',
                                 'matrix-normal-same-trace-allVariables-warmStart-MaxEigDamping',
                                 'matrix-normal-same-trace-allVariables-warmStart-noPerDimDamping',
                                 'shampoo-allVariables-warmStart',
                                 'shampoo-allVariables-warmStart-lessInverse',]:
        return 'None'
    else:
        
        print('params[algorithm]')
        print(params['algorithm'])
        sys.exit()

def get_tensor_reshape(g_W, l, name_variable, params):
    

    

    if get_tensor_reshape_option(l, name_variable, params) == 'filter-flattening':
        g_W = g_W.view(g_W.size(0), g_W.size(1), g_W.size(2) * g_W.size(3))
    elif get_tensor_reshape_option(l, name_variable, params) == 'KFAC-reshaping':
        
#         print('g_W.size()')
#         print(g_W.size())
        
#         sys.exit()
        
        g_W = g_W.view(g_W.size(0), g_W.size(1) * g_W.size(2) * g_W.size(3))
        
    elif get_tensor_reshape_option(l, name_variable, params) == 'None':
        1
    else:
        print('get_tensor_reshape_option(l, name_variable, params)')
        print(get_tensor_reshape_option(l, name_variable, params))
        sys.exit()
        
    return g_W
        
    
    
    '''
    if params['algorithm'] == 'matrix-normal-same-trace-allVariables-filterFlattening-warmStart':
        
        
        
        
        
        if params['layers_params'][l]['name'] == 'conv':
        
            if name_variable == 'W': 
                g_W = g_W.view(g_W.size(0), g_W.size(1), g_W.size(2) * g_W.size(3))

            elif name_variable == 'b':
        
                1
                
            else:
                
                print('name_variable')
                print(name_variable)
        
                sys.exit()
            
            
        elif params['layers_params'][l]['name'] == 'fully-connected':
            1
            
        else:
            print('params[layers_params][l]')
            print(params['layers_params'][l])
            
            sys.exit()
            
        
    else:
        
        print('params[algorithm]')
        print(params['algorithm'])
    
        sys.exit()
    
    
    
    return g_W
    '''

def get_if_shampoo_update(name_variable, params):
    
    if params['algorithm'] in ['matrix-normal',
                            'matrix-normal-same-trace',
                            'matrix-normal-same-trace-warmStart',
                            'matrix-normal-same-trace-warmStart-noPerDimDamping',]:
        print('error: only support allVariables mode now')
        sys.exit()
    
    if name_variable == 'W' or\
    (
        name_variable == 'b'
#         and\
#         params['algorithm'] in ['matrix-normal-allVariables',
#                                 'matrix-normal-allVariables-warmStart',
#                                 'matrix-normal-allVariables-warmStart-noPerDimDamping',
#                                 'matrix-normal-same-trace-allVariables',
#                                 'matrix-normal-same-trace-allVariables-warmStart',
#                                 'matrix-normal-same-trace-allVariables-filterFlattening-warmStart',
#                                 'matrix-normal-same-trace-allVariables-KFACReshaping-warmStart',
#                                 'matrix-normal-same-trace-allVariables-warmStart-noPerDimDamping',
#                                 'shampoo-allVariables',
#                                 'shampoo-allVariables-warmStart']
    ):
        return True
#     elif name_variable == 'b' and\
#     params['algorithm'] in ['matrix-normal',
#                             'matrix-normal-same-trace',
#                             'matrix-normal-same-trace-warmStart',
#                             'matrix-normal-same-trace-warmStart-noPerDimDamping',]:
#         return False
    else:
        print('error: not implemented for ' + name_variable)
#         print('error: not implemented for ' + params['algorithm'])
        sys.exit()
    
def shampoo_kron_matrices_warm_start_per_variable(j, model_grad_N1, l, name_variable, data_, params):
    # model_grad_N1: used for 2nd-order estimate
    
    
    if not get_if_shampoo_update(name_variable, params):
        return
        
        
        





    g_W = model_grad_N1[l][name_variable]

#     print('g_W.size()')
#     print(g_W.size())
    
    g_W = get_tensor_reshape(g_W, l, name_variable, params)
    
#     print('g_W.size()')
#     print(g_W.size())


#         test_H_l = []
#         for ii in range(len(g_W.size())):
#             axes = list(range(len(g_W.size())))
#             axes.remove(ii)
#             test_H_l.append(
#                 torch.tensordot(g_W, g_W, dims=(axes, axes)).data
#             )

    test_H_l = shampoo_get_list_of_contractions(g_W)



#         print('j')
#         print(j)

#         if i == 0:
    if j == 1:

#             print('data_[shampoo_H][l]')
#             print(data_['shampoo_H'][l])






#             if params['algorithm'] in ['shampoo-allVariables',
#                                        'matrix-normal-allVariables']:

#             data_['shampoo_H'].append(test_H_l)
#             data_['shampoo_H'][l] = test_H_l
        data_['shampoo_H'][l][name_variable] = test_H_l

#             else:
#                 print('error: check the impact of warm start for ' + params['algorithm'])
#                 sys.exit()
    else:




        for ii in range(len(data_['shampoo_H'][l][name_variable])):
#                 data_['shampoo_H'][l][name_variable][ii] =\
#     decay_ * data_['shampoo_H'][l][name_variable][ii].data + weight_ * test_H_l[ii]
            data_['shampoo_H'][l][name_variable][ii] *= (j-1)/j
            data_['shampoo_H'][l][name_variable][ii] += 1/j * test_H_l[ii]
        
        
        
#     else:
#         1
    
def shampoo_get_list_of_contractions(g_W):
    
    test_H_l = []
        
    for ii in range(len(g_W.size())):
        axes = list(range(len(g_W.size())))

        axes.remove(ii)

        test_H_l.append(
            torch.tensordot(g_W, g_W, dims=(axes, axes)).data
        )
        
#         print('g_W.size()')
#         print(g_W.size())
        
#         print('test_H_l[-1].size()')
#         print(test_H_l[-1].size())
        
#         sys.exit()
        
    return test_H_l
    

def shampoo_kron_matrices_per_variable(model_grad_N1, l, name_variable, data_, params):
    # model_grad_N1: used for 2nd-order estimate
    
    if not get_if_shampoo_update(name_variable, params):
        return
        
    i = params['i']
    
    if i % params['shampoo_update_freq'] != 0:
        return

    decay_ = params['shampoo_decay']
    weight_ = params['shampoo_weight']

    if not params['if_warm_start']:
        weight_ = max(1/(i+1), weight_)




    g_W = model_grad_N1[l][name_variable]

    g_W = get_tensor_reshape(g_W, l, name_variable, params)






    test_H_l = shampoo_get_list_of_contractions(g_W)



    if i == 0:


        if not params['if_warm_start']:
            data_['shampoo_H'][l][name_variable] = test_H_l
    else:
        # L[l] = decay_ * L[l].data + weight_ * L_[l].data
        # R[l] = decay_ * R[l].data + weight_ * R_[l].data


        for ii in range(len(data_['shampoo_H'][l][name_variable])):
            data_['shampoo_H'][l][name_variable][ii] =\
decay_ * data_['shampoo_H'][l][name_variable][ii].data + weight_ * test_H_l[ii]
    
    


def shampoo_inversion_per_variable(model_grad_N1, l, name_variable, data_, params):
    
    if not get_if_shampoo_update(name_variable, params):
        return
        
    device = params['device']

    i = params['i']
    
#     if params['if_Hessian_action']:
#         inverse_freq = 1
#         inverse_freq = 20
#     else:
    
    inverse_freq = params['shampoo_inverse_freq']

    if params['algorithm'] in ['shampoo-allVariables-warmStart-lessInverse',
                               'shampoo-allVariables-filterFlattening-warmStart-lessInverse',
                               'matrix-normal-correctFisher-allVariables-filterFlattening-warmStart-lessInverse',
                               'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-lessInverse',
                               'matrix-normal-correctFisher-same-trace-allVariables-filterFlattening-warmStart-lessInverse',
                               'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-lessInverse',]:
        pass
    elif params['algorithm'] in ['shampoo-allVariables-filterFlattening-warmStart',
                                 'matrix-normal-same-trace-allVariables-warmStart',
                                 'matrix-normal-same-trace-allVariables-filterFlattening-warmStart',
                                 'matrix-normal-same-trace-allVariables-KFACReshaping-warmStart',
                                 'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart',
                                 'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-TraceWithEpsilonDamping',
                                 'matrix-normal-correctFisher-same-trace-allVariables-filterFlattening-warmStart',
                                 'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping',
                                 'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart',]:
        if i < inverse_freq:
#             inverse_freq = 1
            inverse_freq = params['shampoo_update_freq']
    else:
        print('params[algorithm]')
        print(params['algorithm'])
        sys.exit()

        

    if i % inverse_freq == 0:
        # L[l] = torch.mm(torch.mm(L_l_U, torch.diag(L_l_S)), L_l_V.t())
        # L_l_U, L_l_S, L_l_V = torch.svd(L[l])
        # R_l_U, R_l_S, R_l_V = torch.svd(R[l])

        # L_l_U, L_l_S, L_l_V = torch.svd(L[l])



        if params['if_LM']:
            epsilon = params['lambda_']
        else:





            if params['algorithm'] in ['matrix-normal-allVariables-warmStart-noPerDimDamping',
                                       'matrix-normal-same-trace-warmStart-noPerDimDamping',
                                       'matrix-normal-same-trace-allVariables-warmStart-noPerDimDamping']:

#                     assert params['tau'] == 0



                epsilon = pow(params['shampoo_epsilon'] + params['tau'], 1 / len(data_['shampoo_H'][l][name_variable]))

            elif params['algorithm'] in ['shampoo-allVariables-warmStart',
                                         'shampoo-allVariables-warmStart-lessInverse',
                                         'shampoo-allVariables-filterFlattening-warmStart',
                                         'shampoo-allVariables-filterFlattening-warmStart-lessInverse',
                                         'matrix-normal-allVariables-warmStart',
                                         'matrix-normal-same-trace',
                                         'matrix-normal-same-trace-warmStart',
                                         'matrix-normal-same-trace-allVariables',
                                         'matrix-normal-same-trace-allVariables-warmStart',
                                         'matrix-normal-same-trace-allVariables-filterFlattening-warmStart',
                                         'matrix-normal-same-trace-allVariables-KFACReshaping-warmStart',
                                         'matrix-normal-correctFisher-allVariables-filterFlattening-warmStart-lessInverse',
                                         'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart',
                                         'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-lessInverse',
                                         'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-MaxEigWithEpsilonDamping',
                                         'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-AvgEigWithEpsilonDamping',
                                         'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-TraceWithEpsilonDamping',
                                         'matrix-normal-correctFisher-same-trace-allVariables-filterFlattening-warmStart',
                                         'matrix-normal-correctFisher-same-trace-allVariables-filterFlattening-warmStart-lessInverse',
                                         'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping',
                                         'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart',
                                         'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-lessInverse',]:

                # tau is ignored since we tune shampoo_epsilon

                epsilon = params['shampoo_epsilon']

            elif params['algorithm'] in ['matrix-normal-allVariables-warmStart-MaxEigDamping',
                                         'matrix-normal-same-trace-allVariables-warmStart-AvgEigDamping',
                                         'matrix-normal-same-trace-allVariables-warmStart-MaxEigDamping',]:

                pass

            else:
                print('params[algorithm]')
                print(params['algorithm'])

                sys.exit()



        if params['algorithm'] in ['shampoo-no-sqrt',
                                   'shampoo-no-sqrt-Fisher']:
            power_preconditioner = 1
        elif params['algorithm'] in ['shampoo',
                                     'shampoo-allVariables',
                                     'shampoo-allVariables-warmStart',
                                     'shampoo-allVariables-warmStart-lessInverse',
                                     'shampoo-allVariables-filterFlattening-warmStart',
                                     'shampoo-allVariables-filterFlattening-warmStart-lessInverse',]:
            power_preconditioner = 0.5
        elif params['algorithm'] in ['matrix-normal',
                                     'matrix-normal-allVariables',
                                     'matrix-normal-allVariables-warmStart',
                                     'matrix-normal-allVariables-warmStart-MaxEigDamping',
                                     'matrix-normal-allVariables-warmStart-noPerDimDamping',
                                     'matrix-normal-same-trace',
                                     'matrix-normal-same-trace-warmStart',
                                     'matrix-normal-same-trace-warmStart-noPerDimDamping',
                                     'matrix-normal-same-trace-allVariables',
                                     'matrix-normal-same-trace-allVariables-warmStart',
                                     'matrix-normal-same-trace-allVariables-warmStart-AvgEigDamping',
                                     'matrix-normal-same-trace-allVariables-warmStart-MaxEigDamping',
                                     'matrix-normal-same-trace-allVariables-filterFlattening-warmStart',
                                     'matrix-normal-same-trace-allVariables-KFACReshaping-warmStart',
                                     'matrix-normal-same-trace-allVariables-warmStart-noPerDimDamping',
                                     'matrix-normal-correctFisher-allVariables-filterFlattening-warmStart-lessInverse',
                                     'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart',
                                     'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-lessInverse',
                                     'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-MaxEigWithEpsilonDamping',
                                     'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-AvgEigWithEpsilonDamping',
                                     'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-TraceWithEpsilonDamping',
                                     'matrix-normal-correctFisher-same-trace-allVariables-filterFlattening-warmStart',
                                     'matrix-normal-correctFisher-same-trace-allVariables-filterFlattening-warmStart-lessInverse',
                                     'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping',
                                     'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart',
                                     'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-lessInverse',]:
            1
        else:
            print('Error: unkown algo for power_preconditioner for ' + params['algorithm'])
            sys.exit()



        H_l_LM_minus_2k = []
        H_l_trace = []

        H = data_['shampoo_H']

        if params['algorithm'] in ['matrix-normal',
                                   'matrix-normal-allVariables',
                                   'matrix-normal-allVariables-warmStart',
                                   'matrix-normal-allVariables-warmStart-MaxEigDamping',
                                   'matrix-normal-allVariables-warmStart-noPerDimDamping',
                                   'matrix-normal-correctFisher-allVariables-filterFlattening-warmStart-lessInverse',
                                   'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart',
                                   'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-lessInverse',
                                   'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-MaxEigWithEpsilonDamping',
                                   'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-AvgEigWithEpsilonDamping',
                                   'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-TraceWithEpsilonDamping',]:



            for ii in range(len(H[l][name_variable])):

                H_l_trace.append(torch.trace(H[l][name_variable][ii]).item())



#                     diag_ind = np.diag_indices(H[l][ii].shape[0])

#                     H_l_ii_LM = H[l][ii]

#                     H_l_ii_LM[diag_ind[0], diag_ind[1]] +=\
#                     epsilon * torch.ones(H[l][ii].shape[0], device=device)

                if params['algorithm'] == 'matrix-normal-allVariables-warmStart-MaxEigDamping':
                    epsilon = torch.linalg.norm(H[l][name_variable][ii].data, ord=2).item()
                elif params['algorithm'] == 'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-MaxEigWithEpsilonDamping':
                    epsilon *= torch.linalg.norm(H[l][name_variable][ii].data, ord=2).item()
                elif params['algorithm'] == 'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-AvgEigWithEpsilonDamping':
                    epsilon *= torch.trace(H[l][name_variable][ii].data) / H[l][name_variable][ii].size(0)
                elif params['algorithm'] == 'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-TraceWithEpsilonDamping':
                    
                    epsilon *= torch.trace(H[l][name_variable][ii].data)
                
                if params['if_Hessian_action']:
        
                    g_W = model_grad_N1[l][name_variable]
                
                    g_W = get_tensor_reshape(g_W, l, name_variable, params)
                    
                    axes = list(range(len(g_W.size())))
                    axes.remove(ii)
                    
#                     print('g_W.size()')
#                     print(g_W.size())
                    
#                     print('axes')
#                     print(axes)
                    
                    if len(axes):
                        s = torch.mean(g_W.data, dim=axes)
                    else:
                        s = g_W.data
        
                    y = torch.mv(H[l][name_variable][ii], s) + epsilon * s
            
#                     print('l')
#                     print(l)
                    
#                     print('name_variable')
#                     print(name_variable)

#                     print('data_[shampoo_H_LM_minus_2k][l][name_variable]')
#                     print(data_['shampoo_H_LM_minus_2k'][l][name_variable])
            
                    H_l_LM_minus_2k_ii, _ = get_BFGS_formula_v2(
                        data_['shampoo_H_LM_minus_2k'][l][name_variable][ii],
                        s, y, None, False
                    )
                    H_l_LM_minus_2k.append(H_l_LM_minus_2k_ii)
                
                    
                else:

                    H_l_ii_LM = H[l][name_variable][ii] + epsilon * torch.eye(H[l][name_variable][ii].shape[0], device=device)


                    H_l_LM_minus_2k.append(H_l_ii_LM.inverse())

        elif params['algorithm'] in ['matrix-normal-same-trace',
                                     'matrix-normal-same-trace-warmStart',
                                     'matrix-normal-same-trace-warmStart-noPerDimDamping',
                                     'matrix-normal-same-trace-allVariables',
                                     'matrix-normal-same-trace-allVariables-warmStart',
                                     'matrix-normal-same-trace-allVariables-warmStart-AvgEigDamping',
                                     'matrix-normal-same-trace-allVariables-warmStart-MaxEigDamping',
                                     'matrix-normal-same-trace-allVariables-filterFlattening-warmStart',
                                     'matrix-normal-same-trace-allVariables-KFACReshaping-warmStart',
                                     'matrix-normal-same-trace-allVariables-warmStart-noPerDimDamping',
                                     'matrix-normal-correctFisher-same-trace-allVariables-filterFlattening-warmStart',
                                     'matrix-normal-correctFisher-same-trace-allVariables-filterFlattening-warmStart-lessInverse',
                                     'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping',
                                     'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart',
                                     'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-lessInverse',]:

            c_trace = torch.trace(H[l][name_variable][0])


            list_n = [H_l_ii.size(0) for H_l_ii in H[l][name_variable]]

            c_trace = c_trace / np.prod(list_n)
            c_trace = c_trace**(1/len(list_n))

            for ii in range(len(H[l][name_variable])):
                axes = list(range(len(list_n)))
                axes.remove(ii)

                H_l_same_trace = H[l][name_variable][ii] / c_trace**(len(list_n)-1) /\
                     np.prod(np.asarray(list_n)[axes])

                if params['algorithm'] == 'matrix-normal-same-trace-allVariables-warmStart-AvgEigDamping':
                    epsilon = torch.trace(H_l_same_trace) / H_l_same_trace.size(0)
                elif params['algorithm'] == 'matrix-normal-same-trace-allVariables-warmStart-MaxEigDamping':

#                         if H_l_same_trace.size(0) == 1:
#                             epsilon = H_l_same_trace.data[0][0].item()
#                         else:
#                             epsilon = torch.lobpcg(H_l_same_trace.data)[0].item()

#                         print('torch.linalg.norm(H_l_same_trace.data, ord=2)')
#                         print(torch.linalg.norm(H_l_same_trace.data, ord=2))

#                         sys.exit()

                    epsilon = torch.linalg.norm(H_l_same_trace.data, ord=2).item()

                H_l_LM_same_trace =\
                H_l_same_trace + epsilon * torch.eye(list_n[ii], device=device)

                H_l_LM_minus_2k.append(H_l_LM_same_trace.inverse())

#                     H_l_LM_minus_2k.append(H_l_LM_same_trace.cpu().inverse().cuda())

#                     H_l_LM_same_trace = H_l_LM_same_trace.cpu()
#                     H_l_LM_same_trace = H_l_LM_same_trace.inverse()
#                     H_l_LM_same_trace = H_l_LM_same_trace.cuda()
#                     H_l_LM_minus_2k.append(H_l_LM_same_trace)



        elif params['algorithm'] in ['shampoo',
                                     'shampoo-allVariables',
                                     'shampoo-allVariables-warmStart',
                                     'shampoo-allVariables-warmStart-lessInverse',
                                     'shampoo-allVariables-filterFlattening-warmStart',
                                     'shampoo-allVariables-filterFlattening-warmStart-lessInverse',
                                     'shampoo-no-sqrt',
                                     'shampoo-no-sqrt-Fisher']:

#                 print('error: need to change because data_[shampoo_H] is dict now')
#                 sys.exit()

            for ii in range(len(H[l][name_variable])):
#                     diag_ind = np.diag_indices(H[l][ii].shape[0])
#                     H_l_ii_LM = H[l][ii]
#                     H_l_ii_LM[diag_ind[0], diag_ind[1]] +=\
#                     epsilon * torch.ones(H[l][ii].shape[0], device=device)

                H_l_ii_LM = H[l][name_variable][ii] + epsilon * torch.eye(H[l][name_variable][ii].shape[0], device=device)

#                 if_np_svd = False
#                 if_cpu_svd = False

#                     try:
#                         H_l_U, H_l_S, H_l_V = torch.svd(H_l_ii_LM)
#                     except:
#                         print('H_l_ii_LM')
#                         print(H_l_ii_LM)

#                         print('H_l_ii_LM.detach().cpu().numpy()')
#                         print(H_l_ii_LM.detach().cpu().numpy())

#                         np.save('gpu_svd.npy', H_l_ii_LM.detach().cpu().numpy())

#                         import pickle
#                         with open('gpu_svd.pkl', 'wb') as fp:
#                             pickle.dump(H_l_ii_LM.detach().cpu().numpy(), fp)

#                         sys.exit()


#                     H_l_U, H_l_S, H_l_V = torch.svd(H_l_ii_LM)

#                 H_l_U, H_l_S, H_l_V = get_svd_by_cpu(H_l_ii_LM, params)


                try:
                    H_l_U, H_l_S, H_l_V = torch.svd(H_l_ii_LM)

                    if torch.sum(H_l_S != H_l_S) or\
                    torch.sum(H_l_U != H_l_U) or\
                    torch.sum(H_l_V != H_l_V):
                        H_l_U, H_l_S, H_l_V = get_svd_by_cpu(H_l_ii_LM, params)
                except:
                    H_l_U, H_l_S, H_l_V = get_svd_by_cpu(H_l_ii_LM, params)





                power_H_l_LM_minus_2k = power_preconditioner / len(H[l][name_variable])

                H_l_LM_minus_2k.append(
                    torch.mm(
                        torch.mm(
                            H_l_U, torch.diag(1/(H_l_S**power_H_l_LM_minus_2k))), H_l_V.t())
                )

        else:
            print('Error: unkown algo in svd for ' + params['algorithm'])
            sys.exit()

        data_['shampoo_H_LM_minus_2k'][l][name_variable] = H_l_LM_minus_2k
        data_['shampoo_H_trace'][l][name_variable] = H_l_trace

def shampoo_compute_direction_per_variable(model_grad, l, name_variable, data_, params):
    # model_grad: 1st-order estimate
    
    if get_if_shampoo_update(name_variable, params):
        
        H_l_LM_minus_2k = data_['shampoo_H_LM_minus_2k'][l][name_variable]
        H_l_trace = data_['shampoo_H_trace'][l][name_variable]
        
        delta_l = model_grad[l][name_variable]
        
        delta_l = get_tensor_reshape(delta_l, l, name_variable, params)
        
        
        for ii in range(len(H_l_LM_minus_2k)):
            
            delta_l = torch.tensordot(delta_l, H_l_LM_minus_2k[ii], dims=([0], [0]))
        
        
        delta_l = get_tensor_reshape_back(delta_l, l, name_variable, params)
        
        # re-scaling for matrix-normal
        if params['algorithm'] in ['matrix-normal',
                                   'matrix-normal-allVariables',
                                   'matrix-normal-allVariables-warmStart',
                                   'matrix-normal-allVariables-warmStart-MaxEigDamping',
                                   'matrix-normal-allVariables-warmStart-noPerDimDamping',
                                   'matrix-normal-correctFisher-allVariables-filterFlattening-warmStart-lessInverse',
                                   'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart',
                                   'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-lessInverse',
                                   'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-MaxEigWithEpsilonDamping',
                                   'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-AvgEigWithEpsilonDamping',
                                   'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-TraceWithEpsilonDamping',]:
            delta_l = delta_l * (np.asarray(H_l_trace).prod())**(1 - 1/len(delta_l.size()))


        elif params['algorithm'] in ['shampoo',
                                     'shampoo-allVariables',
                                     'shampoo-allVariables-warmStart',
                                     'shampoo-allVariables-warmStart-lessInverse',
                                     'shampoo-allVariables-filterFlattening-warmStart',
                                     'shampoo-allVariables-filterFlattening-warmStart-lessInverse',
                                     'shampoo-no-sqrt',
                                     'shampoo-no-sqrt-Fisher',
                                     'matrix-normal-same-trace',
                                     'matrix-normal-same-trace-warmStart',
                                     'matrix-normal-same-trace-warmStart-noPerDimDamping',
                                     'matrix-normal-same-trace-allVariables',
                                     'matrix-normal-same-trace-allVariables-warmStart',
                                     'matrix-normal-same-trace-allVariables-warmStart-AvgEigDamping',
                                     'matrix-normal-same-trace-allVariables-warmStart-MaxEigDamping',
                                     'matrix-normal-same-trace-allVariables-filterFlattening-warmStart',
                                     'matrix-normal-same-trace-allVariables-KFACReshaping-warmStart',
                                     'matrix-normal-same-trace-allVariables-warmStart-noPerDimDamping',
                                     'matrix-normal-correctFisher-same-trace-allVariables-filterFlattening-warmStart',
                                     'matrix-normal-correctFisher-same-trace-allVariables-filterFlattening-warmStart-lessInverse',
                                     'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping',
                                     'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart',
                                     'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-lessInverse',]:
            1
        else:
            print('Error: unknown algo in tensor dot for ' + params['algorithm'])
            sys.exit()
            
        return delta_l
    else:
        return model_grad[l][name_variable]

def get_svd_by_cpu(H_l_ii_LM, params):
    device = params['device']

    if_cpu_svd = True
    H_l_cpu = H_l_ii_LM.cpu()

    try:
        H_l_U_cpu, H_l_S_cpu, H_l_V_cpu = torch.svd(H_l_cpu)
    except:
        
#         print('should not reach here')
#         sys.exit()
        
        if_np_svd = True
        np_H_l_U_cpu, np_H_l_S_cpu, np_H_l_V_cpu = np.linalg.svd(H_l_cpu.data.numpy())
        np_H_l_V_cpu = np.transpose(np_H_l_V_cpu)
        H_l_U_cpu, H_l_S_cpu, H_l_V_cpu = \
        torch.from_numpy(np_H_l_U_cpu),\
        torch.from_numpy(np_H_l_S_cpu), torch.from_numpy(np_H_l_V_cpu)

    H_l_U, H_l_S, H_l_V =\
    H_l_U_cpu.to(device), H_l_S_cpu.to(device), H_l_V_cpu.to(device)

    return H_l_U, H_l_S, H_l_V



def shampoo_update(data_, params):
    
    true_algorithm = params['algorithm']


#     model_grad = data_['model_regularized_grad_used_torch']
    model_grad = data_['model_grad_used_torch']
    
    
    
    if params['matrix_name'] in ['Fisher',
                                 'Fisher-correct']:
        model_grad_N1 = data_['model_grad_N2'] # unregularized
    elif params['matrix_name'] == 'None':
        # None is also EF
        model_grad_N1 = data_['model_grad_torch']
    else:
        print('params[matrix_name]')
        print(params['matrix_name'])
        sys.exit()
        

    i = params['i']

    

    # alpha = params['alpha']
    numlayers = params['numlayers']

    device = params['device']
    
#     weight_ = max(1/(i+1), weight_)

    
        
        
    # Step
    delta = []
    for l in range(numlayers):

        for name_variable in data_['model'].layers_weight[l].keys():
            shampoo_kron_matrices_per_variable(model_grad_N1, l, name_variable, data_, params)
        

    




        for name_variable in data_['model'].layers_weight[l].keys():
            shampoo_inversion_per_variable(model_grad_N1, l, name_variable, data_, params)
            


        
        
        
        

        # store the delta
        dict_delta_l = {}
        
        for name_variable in data_['model'].layers_weight[l].keys():
            dict_delta_l[name_variable] = shampoo_compute_direction_per_variable(model_grad, l, name_variable, data_, params)

            
        delta.append(dict_delta_l)



    p = get_opposite(delta)


    
    
    
    data_['p_torch'] = p

    if true_algorithm == 'matrix-normal-LM-momentum-grad':
        params['algorithm'] = true_algorithm

    return data_, params