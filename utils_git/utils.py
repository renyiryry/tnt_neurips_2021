import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import scipy
import copy
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import time
import pickle


import os
import math
import psutil
import itertools
import datetime
import shutil

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import warnings
warnings.filterwarnings('error')
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

# warnings.filterwarnings("ignore", category=UserWarning)

# from utils_git.utils_plot import *
# from utils_git.utils_shampoo import *
# from utils_git.utils_kbfgs import *

from utils_git.utils_get_params import get_params



os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def get_block_Fisher(h, a, l, params):
    
    device = params['device']
    
    size_minibatch = h[l].size(0)
                
    homo_h_l = torch.cat(
        (h[l].data, torch.ones(size_minibatch, 1, device=device)),
        dim=1
    )

    a_l_grad = size_minibatch * a[l].grad.data

    a_a_T = torch.einsum('ij,ik->ijk', homo_h_l, homo_h_l)

    g_g_T = torch.einsum('ij,ik->ijk', a_l_grad, a_l_grad)

#                 print('torch.einsum(eab,ecd->eacbd, a_a_T, g_g_T).view(a_a_T.size(1)*g_g_T.size(1),  a_a_T.size(2)*g_g_T.size(2)).size()')
#                 print(torch.einsum("eab,ecd->eacbd", a_a_T, g_g_T).view(a_a_T.size(1)*g_g_T.size(1),  a_a_T.size(2)*g_g_T.size(2)).size())

    G_j = torch.zeros(homo_h_l.size(1)*a_l_grad.size(1), homo_h_l.size(1)*a_l_grad.size(1), device=device)

    for dp in range(size_minibatch):
        G_j += torch.kron(a_a_T[dp], g_g_T[dp])

    G_j /= size_minibatch
    
    return G_j

def Fisher_BD_update(data_, params):
    
    i = params['i']
    device = params['device']
    
    model_grad = data_['model_grad_used_torch']
    
    homo_model_grad = get_homo_grad(model_grad, params)
    
    delta = []
    for l in range(params['numlayers']):
        
        # compute statistics
        
        if i == 0:
            pass
        else:
            F_l = get_block_Fisher(data_['h_N2'], data_['a_N2'], l, params)
            
#             print('F_l.size()')
#             print(F_l.size())
            
            data_['block_Fisher'][l] *= 0.9
            data_['block_Fisher'][l] += 0.1 * F_l
        
#             sys.exit()
        
        # compute direction
        
        homo_grad_l_vec = torch.reshape(homo_model_grad[l].t(), (-1, 1))
        
#         F_l_LM = data_['block_Fisher'][l] + 0.001 * torch.eye(data_['block_Fisher'][l].size(0), device=device)
        F_l_LM = data_['block_Fisher'][l] + params['Fisher_BD_damping'] * torch.eye(data_['block_Fisher'][l].size(0), device=device)
        
#         F_l_LM = torch.eye(data_['block_Fisher'][l].size(0), device=device)
#         print('need to change back')
        
        homo_delta_l, _ = torch.solve(homo_grad_l_vec, F_l_LM)
        
#         print('homo_delta_l.size()')
#         print(homo_delta_l.size())
        
        homo_delta_l = torch.reshape(homo_delta_l, homo_model_grad[l].t().size()).t()

        delta_l = from_homo_to_weight_and_bias(homo_delta_l, l, params)
    
        delta.append(delta_l)
        
    p = get_opposite(delta)
    
    data_['p_torch'] = p
        
    
#     sys.exit()
    
    return data_, params


def get_BFGS_formula_v2(H, s, y, g_k, if_test_mode):
    
#     print('should move to utils.py')
    
    s = s.data
    y = y.data

    # ger(a, b) = a b^T
    rho_inv = torch.dot(s, y)

    if rho_inv <= 0:
#     if rho_inv <= 10**(-3):
#         print('BFGS not updated (case 1).')
#         print('rho_inv')
#         print(rho_inv)
    
        return H, 1
#     elif rho_inv <= 10**(-4) * torch.dot(s, s) * np.sqrt(torch.dot(g_k, g_k).item()):
        
#         sys.exit()
        
#         return H, 2

    # sHs = torch.dot(s, torch.mv(H, s))
    # if rho_inv < 0.25 * sHs:
        # theta = (0.75 * sHs) / (sHs - rho_inv)



    rho = 1 / rho_inv

    # s = s / np.sqrt(rho_inv.item())
    # y = y / np.sqrt(rho_inv.item())
    # rho = 1

    Hy = torch.mv(H, y)
    
#     H_new = H.data +\
#     (rho**2 * torch.dot(y, torch.mv(H, y)) + rho) * torch.ger(s, s) -\
#     rho * (torch.ger(s, Hy) + torch.ger(Hy, s))
    
#     H_new = H.data +\
#     (rho**2 * torch.dot(y, torch.mv(H, y)) + rho) * torch.ger(s, s) -\
#     (torch.ger(rho*s, Hy) + torch.ger(Hy, rho*s))
    
#     H_new = H.data +\
#     (rho**2 * torch.dot(y, Hy) + rho) * torch.ger(s, s) -\
#     (torch.ger(rho*s, Hy) + torch.ger(Hy, rho*s))
    
    H_new = H.data +\
    torch.ger((rho**2 * torch.dot(y, Hy) + rho)*s, s) -\
    (torch.ger(rho*s, Hy) + torch.ger(Hy, rho*s))
    
    if if_test_mode:
        
        print('torch.norm(s)')
        print(torch.norm(s))
        
        print('torch.norm(y)')
        print(torch.norm(y))
        
        print('torch.norm((rho**2 * torch.dot(y, torch.mv(H, y)) + rho) * torch.ger(s, s))')
        print(torch.norm((rho**2 * torch.dot(y, torch.mv(H, y)) + rho) * torch.ger(s, s)))

        print('torch.norm(rho**2 * torch.dot(y, torch.mv(H, y)) * torch.ger(s, s))')
        print(torch.norm(rho**2 * torch.dot(y, torch.mv(H, y)) * torch.ger(s, s)))
        
        print('torch.norm(rho * torch.dot(y, torch.mv(H, y)))')
        print(torch.norm(rho * torch.dot(y, torch.mv(H, y))))
        
        print('torch.norm(rho * torch.ger(s, s))')
        print(torch.norm(rho * torch.ger(s, s)))

        print('rho')
        print(rho)
        
    

    
#     if torch.norm(H_new) > 2 * torch.norm(H):
#         return H, 3
#     if torch.max(torch.isinf(H_new)):
#         return H, 4
#     else:
    H = H_new

#     if torch.max(torch.isinf(H)):
#         print('inf in H')
#         print('s')
#         print(s)
#         print('y')
#         print(y)
#         sys.exit()

    return H, 0



def from_unregularized_grad_to_regularized_grad(model_grad_torch, data_, params):
    
    if params['if_regularized_grad']:
        # if you want unregularized grad, you should NOT
        # backward the regularized grad and then subtract the 
        # regularization term. Because the accurate value will be
        # overwhelmed by the noise (numerical error?). 

        # however, if you want regularized grad, you can backward 
        # the unregularized grad and then add the regularization,
        # which will gives you the same value as backward
        # the regularized grad

        if params['tau'] == 0:
            1
        else:
            
            model = data_['model']
            
#             print('params[tau]')
#             print(params['tau'])

            model_grad_torch = get_plus_torch(
        model_grad_torch,
        get_multiply_scalar_no_grad(params['tau'], model.layers_weight)
        )
    else:
        1
        
    return model_grad_torch
    

def add_res_block(layers_, in_channels_1, out_channels_1, stride1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias):
    
#     print('need to change name for no bias')
    
#     print('how about 1 * 1 conv?')
    
    layer_ = {}
    
    
    if if_conv_bias == False:
        
        assert if_downsample_only == True
        
#         print('shortcut_type')
#         print(shortcut_type)
        
        
        
        
        
        if shortcut_type == 'padding':
            
#             print('if_BNNoAffine')
#             print(if_BNNoAffine)
            
#             sys.exit()
            
            if stride1 > 1:
                if if_BNNoAffine:
                    layer_['name'] = 'ResBlock-BNNoAffine-PaddingShortcut-NoBias'
                else:
                    layer_['name'] = 'ResBlock-BN-PaddingShortcut-NoBias'
            else:
                if if_BNNoAffine:
                    layer_['name'] = 'ResBlock-BNNoAffine-identityShortcut-NoBias'
                else:
                    layer_['name'] = 'ResBlock-BN-identityShortcut-NoBias'
                
        elif shortcut_type == 'conv':
            
            assert if_BNNoAffine == False
            
            assert if_BN_shortcut == True

            if stride1 > 1:
                layer_['name'] = 'ResBlock-BN-BNshortcut-NoBias'
            else:
                layer_['name'] = 'ResBlock-BN-identityShortcut-NoBias'
        else:
            print('shortcut_type')
            print(shortcut_type)
            sys.exit()
        
            
        
    else:
        
        
    
        if if_downsample_only:

            assert if_BNNoAffine == False

            assert if_BN_shortcut == True

            if stride1 > 1:
                layer_['name'] = 'ResBlock-BN-BNshortcut'
            else:
                layer_['name'] = 'ResBlock-BN-identityShortcut'

        else:

            if if_BNNoAffine:

                assert if_BN_shortcut == False

                layer_['name'] = 'ResBlock-BNNoAffine'
            else:
                if if_BN_shortcut:
                    layer_['name'] = 'ResBlock-BN-BNshortcut'
                else:
                    layer_['name'] = 'ResBlock-BN'

    layer_['conv1'] = {}
    
#     layer_['conv1']['conv_in_channels'] = 16
    layer_['conv1']['conv_in_channels'] = in_channels_1
    
#     layer_['conv1']['conv_out_channels'] = 16
    layer_['conv1']['conv_out_channels'] = out_channels_1
    
    layer_['conv1']['conv_kernel_size'] = 3
    
#     layer_['conv1']['conv_stride'] = 1
    layer_['conv1']['conv_stride'] = stride1
    
    layer_['conv1']['conv_padding'] = 1
    
#     print('if_conv_bias')
#     print(if_conv_bias)
    
    
    
    layer_['conv1']['conv_bias'] = if_conv_bias

    
    
    if if_BNNoAffine:
        layer_['BNNoAffine1'] = {}
        layer_['BNNoAffine1']['num_features'] = out_channels_1
    else:
        layer_['BN1'] = {}
        layer_['BN1']['num_features'] = out_channels_1

    layer_['conv2'] = {}
    

    layer_['conv2']['conv_in_channels'] = out_channels_1
    
    layer_['conv2']['conv_out_channels'] = out_channels_1
    
    layer_['conv2']['conv_kernel_size'] = 3
    layer_['conv2']['conv_stride'] = 1
    layer_['conv2']['conv_padding'] = 1
    layer_['conv2']['conv_bias'] = if_conv_bias

    
    
    if if_BNNoAffine:
        layer_['BNNoAffine2'] = {}
        layer_['BNNoAffine2']['num_features'] = out_channels_1
    else:
        layer_['BN2'] = {}
        layer_['BN2']['num_features'] = out_channels_1
        
        
#     print('layer_[name]')
#     print(layer_['name'])
    
    
    
    if layer_['name'] in ['ResBlock-BN-identityShortcut',
                          'ResBlock-BN-identityShortcut-NoBias',
                          'ResBlock-BN-PaddingShortcut-NoBias']:
        1
    else:
        # 1*1 conv for shortcut
        layer_['conv3'] = {}

        layer_['conv3']['conv_in_channels'] = in_channels_1

        layer_['conv3']['conv_out_channels'] = out_channels_1

        layer_['conv3']['conv_kernel_size'] = 1

        layer_['conv3']['conv_stride'] = stride1

        layer_['conv3']['conv_padding'] = 0
        
        layer_['conv3']['conv_bias'] = if_conv_bias

        if if_BN_shortcut:
            assert if_BNNoAffine == False

            layer_['BN3'] = {}
            layer_['BN3']['num_features'] = out_channels_1



    layers_.append(layer_)
    
    return layers_

def add_conv_block(layers_, in_channels, out_channels, kernel_size, stride, padding, params):
    # conv + (possible) BN + activation
    
    layer_2 = {}
    
    if params['name_dataset'] in ['CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-NoBias',
                                  'CIFAR-10-onTheFly-ResNet32-BNNoAffine-NoBias',
                                  'CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly-NoBias',
                                  'CIFAR-10-onTheFly-N1-128-ResNet32-BNNoAffine-PaddingShortcutDownsampleOnly-NoBias-no-regularization',
                                  'CIFAR-10-onTheFly-N1-128-ResNet32-BN-BNshortcutDownsampleOnly-NoBias',
                                  'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',
                                  'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias-no-regularization',
                                  'CIFAR-100-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',
                                  'CIFAR-100-onTheFly-ResNet34-BN-BNshortcutDownsampleOnly-NoBias',
                                  'CIFAR-100-onTheFly-N1-128-ResNet34-BN-BNshortcutDownsampleOnly-NoBias',
                                  'CIFAR-100-onTheFly-N1-128-ResNet34-BN-PaddingShortcutDownsampleOnly-NoBias',]:
        layer_2['name'] = 'conv-no-bias-no-activation'
    elif params['name_dataset'] in ['CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
                                    'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
                                    'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPool',
                                    'CIFAR-10-onTheFly-N1-512-vgg16-NoAdaptiveAvgPoolNoDropout',
                                    'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
                                    'CIFAR-10-AllCNNC',
                                    'CIFAR-10-N1-128-AllCNNC',
                                    'CIFAR-10-N1-512-AllCNNC',
                                    'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
                                    'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine-no-regularization',
                                    'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
                                    'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-no-regularization',
                                    'CIFAR-100-onTheFly-vgg16-NoLinear-BN-no-regularization',
                                    'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
                                    'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',
                                    'CIFAR-100-onTheFly-AllCNNC',
                                    'CIFAR-10-onTheFly-ResNet32-BNNoAffine',
                                    'CIFAR-10-onTheFly-ResNet32-BN',
                                    'CIFAR-10-onTheFly-ResNet32-BN-BNshortcut',
                                    'CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly',
                                    'CIFAR-100-onTheFly-ResNet34-BNNoAffine',
                                    'CIFAR-100-onTheFly-ResNet34-BN',
                                    'CIFAR-100-onTheFly-ResNet34-BN-BNshortcut',
                                    'CIFAR-100-onTheFly-ResNet34-BN-BNshortcutDownsampleOnly',]:
        layer_2['name'] = 'conv-no-activation'
    else:
        print('params[name_dataset]')
        print(params['name_dataset'])
        sys.exit()
        
    
    layer_2['conv_in_channels'] = in_channels
    layer_2['conv_out_channels'] = out_channels
    
    
    layer_2['conv_kernel_size'] = kernel_size
    
    layer_2['conv_stride'] = stride
    
    layer_2['conv_padding'] = padding
    
    
    layer_2['activation'] = None
    layers_.append(layer_2)
    
    if params['name_dataset'] in ['CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
                                  'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
                                  'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-NoBias',
                                  'CIFAR-10-onTheFly-ResNet32-BN',
                                  'CIFAR-10-onTheFly-ResNet32-BN-BNshortcut',
                                  'CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly',
                                  'CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly-NoBias',
                                  'CIFAR-10-onTheFly-N1-128-ResNet32-BN-BNshortcutDownsampleOnly-NoBias',
                                  'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',
                                  'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias-no-regularization',
                                  'CIFAR-100-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',
                                  'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
                                  'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-no-regularization',
                                  'CIFAR-100-onTheFly-vgg16-NoLinear-BN-no-regularization',
                                  'CIFAR-100-onTheFly-ResNet34-BN',
                                  'CIFAR-100-onTheFly-ResNet34-BN-BNshortcut',
                                  'CIFAR-100-onTheFly-ResNet34-BN-BNshortcutDownsampleOnly',
                                  'CIFAR-100-onTheFly-ResNet34-BN-BNshortcutDownsampleOnly-NoBias',
                                  'CIFAR-100-onTheFly-N1-128-ResNet34-BN-BNshortcutDownsampleOnly-NoBias',
                                  'CIFAR-100-onTheFly-N1-128-ResNet34-BN-PaddingShortcutDownsampleOnly-NoBias',]:
        # references:
        # http://torch.ch/blog/2015/07/30/cifar.html
        # https://github.com/kuangliu/pytorch-cifar
        layer_2 = {}
        layer_2['name'] = 'BN'
        layer_2['num_features'] = out_channels
        layer_2['activation'] = None
        layers_.append(layer_2)
    elif params['name_dataset'] in ['CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',
                                    'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',
                                    'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine-no-regularization',
                                    'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',
                                    'CIFAR-10-onTheFly-ResNet32-BNNoAffine',
                                    'CIFAR-10-onTheFly-ResNet32-BNNoAffine-NoBias',
                                    'CIFAR-10-onTheFly-N1-128-ResNet32-BNNoAffine-PaddingShortcutDownsampleOnly-NoBias-no-regularization',
                                    'CIFAR-100-onTheFly-ResNet34-BNNoAffine',]:
        layer_2 = {}
        layer_2['name'] = 'BNNoAffine'
        layer_2['num_features'] = out_channels
        layer_2['activation'] = None
        layers_.append(layer_2)
    elif params['name_dataset'] in ['CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout',
                                    'CIFAR-10-NoAugmentation-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
                                    'CIFAR-10-vgg16-NoAdaptiveAvgPoolNoDropout',
                                    'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
                                    'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
                                    'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPool',
                                    'CIFAR-10-onTheFly-N1-512-vgg16-NoAdaptiveAvgPoolNoDropout',
                                    'CIFAR-10-AllCNNC',
                                    'CIFAR-10-N1-128-AllCNNC',
                                    'CIFAR-10-N1-512-AllCNNC',
                                    'CIFAR-10-ConvPoolCNNC',
                                    'CIFAR-100-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout',
                                    'CIFAR-100-NoAugmentation-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
                                    'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
                                    'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
                                    'CIFAR-100-onTheFly-AllCNNC']:
        1
    else:
        print('error: need to check for ' + params['name_dataset'])
        sys.exit()

    layer_2 = {}
    layer_2['name'] = 'relu'
    layers_.append(layer_2)
    
    return layers_

def get_next_lr(list_lr_tried, best_lr):
    
    list_lr_complete =\
    [1e-10, 3e-10, 1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]
    
    if best_lr == min(list_lr_tried):
        
        print('list_lr_complete.index(best_lr)')
        print(list_lr_complete.index(best_lr))
        
        if list_lr_complete.index(best_lr) == 0:
            print('error: need to expand list_lr_complete')
            sys.exit()
        else:
            return list_lr_complete[list_lr_complete.index(best_lr) - 1]
    elif best_lr == max(list_lr_tried):
        
        if list_lr_complete.index(best_lr) == len(list_lr_complete) - 1:
            print('error: need to expand list_lr_complete')
            sys.exit()
        else:
            return list_lr_complete[list_lr_complete.index(best_lr) + 1]
        
#         sys.exit()
    elif best_lr > min(list_lr_tried) and best_lr < max(list_lr_tried):
        return -1
    else:
        print('there is an error')
        sys.exit()
    
    return learning_rate

def add_some_if_record_to_args(args):
    
    if not 'if_record_sgd_norm' in args:
        args['if_record_sgd_norm'] = False
    if not 'if_record_sgn_norm' in args:
        args['if_record_sgn_norm'] = False
    if not 'if_record_p_norm' in args:
        args['if_record_p_norm'] = False
    if not 'if_record_kron_bfgs_cosine' in args:
        args['if_record_kron_bfgs_cosine'] = False
    if not 'if_record_kfac_p_norm' in args:
        args['if_record_kfac_p_norm'] = False
    if not 'if_record_kfac_p_cosine' in args:
        args['if_record_kfac_p_cosine'] = False
    if not 'if_record_res_grad_norm' in args:
        args['if_record_res_grad_norm'] = False
    if not 'if_record_res_grad_random_norm' in args:
        args['if_record_res_grad_random_norm'] = False
    if not 'if_record_res_grad_grad_norm' in args:
        args['if_record_res_grad_grad_norm'] = False
    if not 'if_record_res_grad_norm_per_iter' in args:
        args['if_record_res_grad_norm_per_iter'] = False
    
    return args

def add_matrix_name_to_args(args):
    
    if args['algorithm'] == 'SMW-GN':
        args['matrix_name'] = 'GN'
    elif args['algorithm'] in ['SMW-Fisher-signVAsqrt-p',
                               'SMW-Fisher-VA-p',
                               'SMW-Fisher-momentum-p-sign',
                               'SMW-Fisher-momentum-p',
                               'SMW-Fisher-sign',
                               'SMW-Fisher-different-minibatch',
                               'SMW-Fisher',
                               'SMW-Fisher-momentum',
                               'SMW-Fisher-batch-grad-momentum-exponential-decay',
                               'SMW-Fisher-batch-grad-momentum',
                               'SMW-Fisher-batch-grad',
                               'shampoo-no-sqrt-Fisher-momentum-grad',
                               'shampoo-no-sqrt-Fisher-momentum-grad-test',
                               'matrix-normal',
                               'matrix-normal-momentum-grad',
                               'matrix-normal-allVariables-momentum-grad',
                               'matrix-normal-allVariables-warmStart-momentum-grad',
                               'matrix-normal-allVariables-warmStart-MaxEigDamping-momentum-grad',
                               'matrix-normal-allVariables-warmStart-noPerDimDamping-momentum-grad',
                               'matrix-normal-LM-momentum-grad',
                               'matrix-normal-same-trace',
                               'matrix-normal-same-trace-momentum-grad',
                               'matrix-normal-same-trace-warmStart-momentum-grad',
                               'matrix-normal-same-trace-warmStart-noPerDimDamping-momentum-grad',
                               'matrix-normal-same-trace-allVariables-momentum-grad',
                               'matrix-normal-same-trace-allVariables-warmStart-momentum-grad',
                               'matrix-normal-same-trace-allVariables-warmStart-momentum-grad-LRdecay',
                               'matrix-normal-same-trace-allVariables-filterFlattening-warmStart-momentum-grad',
                               'matrix-normal-same-trace-allVariables-KFACReshaping-warmStart-momentum-grad',
                               'matrix-normal-same-trace-allVariables-KFACReshaping-warmStart-momentum-grad-LRdecay',
                               'matrix-normal-same-trace-allVariables-warmStart-noPerDimDamping-momentum-grad',
                               'matrix-normal-same-trace-allVariables-warmStart-AvgEigDamping-momentum-grad',
                               'matrix-normal-same-trace-allVariables-warmStart-MaxEigDamping-momentum-grad',
                               'matrix-normal-same-trace-LM-momentum-grad',
                               'kfac-TR',
                               'kfac-CG',
                               'kfac-momentum-grad-CG',
                               'kfac-momentum-grad-TR',
                               'kfac',
                               'kfac-momentum-grad',
                               'kfac-no-max',
                               'kfac-no-max-no-LM',
                               'kfac-no-max-no-LM-momentum-grad',
                               'kfac-warmStart-no-max-no-LM-momentum-grad',
                               'kfac-warmStart-lessInverse-no-max-no-LM-momentum-grad',
                               'kfac-warmStart-lessInverse-no-max-no-LM-momentum-grad-LRdecay',
                               'kfac-warmStart-lessInverse-NoMaxNoSqrt-no-LM-momentum-grad',
                               'kfac-no-max-epsilon-A-G-no-LM-momentum-grad',
                               'kfac-no-max-momentum-grad',
                               'kfac-NoMaxNoSqrt-momentum-grad',
                               'kfac-NoMaxNoSqrt-no-LM-momentum-grad']:
        args['matrix_name'] = 'Fisher'
    elif args['algorithm'] in ['Fisher-BD',
                               'Fisher-BD-momentum-grad',
                               'kfac-correctFisher-warmStart-no-max-no-LM-momentum-grad',
                               'kfac-correctFisher-warmStart-no-max-no-LM-momentum-grad-LRdecay',
                               'kfac-correctFisher-warmStart-NoMaxNoSqrt-no-LM-momentum-grad',
                               'kfac-correctFisher-warmStart-NoMaxNoSqrt-no-LM-momentum-grad-LRdecay',
                               'kfac-correctFisher-warmStart-lessInverse-no-max-no-LM-momentum-grad',
                               'kfac-correctFisher-warmStart-lessInverse-NoMaxNoSqrt-no-LM-momentum-grad',
                               'matrix-normal-correctFisher-allVariables-filterFlattening-warmStart-lessInverse-momentum-grad',
                               'matrix-normal-correctFisher-allVariables-filterFlattening-warmStart-lessInverse-momentum-grad-LRdecay',
                               'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-momentum-grad',
                               'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-momentum-grad-LRdecay',
                               'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-lessInverse-momentum-grad-LRdecay',
                               'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-MaxEigWithEpsilonDamping-momentum-grad',
                               'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-AvgEigWithEpsilonDamping-momentum-grad',
                               'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-TraceWithEpsilonDamping-momentum-grad',
                               'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-TraceWithEpsilonDamping-momentum-grad-LRdecay',
                               'matrix-normal-correctFisher-same-trace-allVariables-filterFlattening-warmStart-momentum-grad-LRdecay',
                               'matrix-normal-correctFisher-same-trace-allVariables-filterFlattening-warmStart-lessInverse-momentum-grad-LRdecay',
                               'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-momentum-grad',
                               'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-momentum-grad',
                               'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-momentum-grad-LRdecay',
                               'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-lessInverse-momentum-grad',
                               'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-lessInverse-momentum-grad-LRdecay',]:
        args['matrix_name'] = 'Fisher-correct'
    elif args['algorithm'] in ['RMSprop-individual-grad',
                               'RMSprop-individual-grad-no-sqrt',
                               'RMSprop-individual-grad-no-sqrt-LM',
                               'kfac-EF',
                               'ekfac-EF-VA',
                               'ekfac-EF',
                               'Kron-BFGS',
                               'Kron-BFGS-no-norm-gate-regularized-grad',
                               'Kron-BFGS-no-norm-gate-momentum-s-y-regularized-grad',
                               'Kron-BFGS-no-norm-gate-momentum-s-y-damping-regularized-grad',
                               'Kron-BFGS-no-norm-gate-momentum-s-y-damping-regularized-grad-momentum-grad',
                               'Kron-BFGS-no-norm-gate-momentum-s-y-damping-unregularized-grad-momentum-grad',
                               'Kron-BFGS-homo-no-norm-gate-momentum-s-y-damping-unregularized-grad-momentum-grad',
                               'Kron-BFGS-homo-no-norm-gate-momentum-s-y-damping-regularized-grad-momentum-grad',
                               'Kron-BFGS-no-norm-gate-momentum-s-y-regularized-grad-momentum-grad',
                               'Kron-BFGS-no-norm-gate-damping-regularized-grad',
                               'Kron-BFGS-no-norm-gate-Shiqian-damping-regularized-grad',
                               'Kron-BFGS-no-norm-gate-Shiqian-damping-regularized-grad-momentum-grad',
                               'Kron-BFGS-no-norm-gate-Shiqian-damping-unregularized-grad-momentum-grad',
                               'Kron-BFGS-homo-no-norm-gate-Shiqian-damping-unregularized-grad-momentum-grad',
                               'Kron-BFGS-homo-no-norm-gate-Powell-H-damping-unregularized-grad-momentum-grad',
                               'Kron-BFGS-homo-no-norm-gate-PowellBDamping-unregularized-grad-momentum-grad',
                               'Kron-BFGS-homo-no-norm-gate-PowellHDampingV2-unregularized-grad-momentum-grad',
                               'Kron-BFGS-homo-no-norm-gate-Powell-H-damping-regularized-grad-momentum-grad',
                               'Kron-BFGS-homo-no-norm-gate-Powell-double-damping-unregularized-grad-momentum-grad',
                               'Kron-BFGS-homo-no-norm-gate-PowellDoubleDampingV2-unregularized-grad-momentum-grad',
                               'Kron-BFGS-homo-no-norm-gate-momentum-s-y-Powell-double-damping-unregularized-grad-momentum-grad',
                               'Kron-BFGS-homo-no-norm-gate-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad',
                               'Kron-BFGS-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad',
                               'Kron-BFGS-homo-no-norm-gate-HessianActionV2-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad',
                               'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad',
                               'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-DDV2-regularized-grad-momentum-grad',
                               'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2IdentityInitial-momentum-s-y-DDV2-regularized-grad-momentum-grad',
                               'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-unregularized-grad-momentum-grad-LRdecay',
                               'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-regularized-grad-momentum-grad',
                               'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-regularized-grad-momentum-grad-LRdecay',
                               'Kron-BFGS-homo-no-norm-gate-miniBatchANotDamped-HessianActionV2-momentum-s-y-DDV2-regularized-grad-momentum-grad',
                               'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-DDV2-regularized-grad-momentum-grad',
                               'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-SqrtT-regularized-grad-momentum-grad',
                               'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-KFACSplitting-regularized-grad-momentum-grad',
                               'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-DDV2-extraStep-regularized-grad-momentum-grad',
                               'Kron-(L)BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad',
                               'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2IdentityInitial-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad',
                               'Kron-BFGS(L)-homo-no-norm-gate-HessianActionV2-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad',
                               'Kron-BFGS(L)-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad',
                               'Kron-BFGS(L)-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-DDV2-regularized-grad-momentum-grad',
                               'Kron-BFGS(L)-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-regularized-grad-momentum-grad',
                               'Kron-BFGS-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping-doubleGrad-regularized-grad-momentum-grad',
                               'Kron-BFGS-homo-no-norm-gate-scaledHessianAction-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad',
                               'Kron-BFGS-homo-no-norm-gate-HessianActionIdentityInitial-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad',
                               'Kron-LBFGS-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad',
                               'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad',
                               'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-PowellDoubleDampingSkip-regularized-grad-momentum-grad',
                               'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-DoubleDamping-regularized-grad-momentum-grad',
                               'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-H-damping-regularized-grad-momentum-grad',
                               'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-B0-damping-regularized-grad-momentum-grad',
                               'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-Shiqian-damping-regularized-grad-momentum-grad',
                               'Kron-(L)BFGS-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad',
                               'Kron-LBFGS-homo-no-norm-gate-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad',
                               'Kron-BFGS-homo-no-norm-gate-Powell-double-damping-regularized-grad-momentum-grad',
                               'Kron-BFGS-homo-no-norm-gate-Hessian-action-Powell-double-damping-regularized-grad-momentum-grad',
                               'Kron-BFGS-homo-identity-unregularized-grad-momentum-grad',
                               'Kron-BFGS-homo-identity-regularized-grad-momentum-grad',
                               'Kron-BFGS-no-norm-gate-damping-regularized-grad-momentum-grad',
                               'Kron-BFGS-no-norm-gate-damping-unregularized-grad-momentum-grad',
                               'Kron-BFGS-homo-no-norm-gate-damping-unregularized-grad-momentum-grad',
                               'Kron-BFGS-homo-no-norm-gate-damping-regularized-grad-momentum-grad',
                               'Kron-BFGS-no-norm-gate-regularized-grad-momentum-grad',
                               'Kron-BFGS-no-norm-gate-unregularized-grad-momentum-grad',
                               'Kron-BFGS-momentum-grad',
                               'Kron-BFGS-regularized-grad',
                               'Kron-BFGS-homo-regularized-grad',
                               'Kron-BFGS-homo-regularized-grad-momentum-grad',
                               'Kron-BFGS-homo-unregularized-grad-momentum-grad',
                               'Kron-BFGS-unregularized-grad',
                               'Kron-BFGS-wrong-unregularized-grad',
                               'Kron-BFGS-unregularized-grad-momentum-grad',
                               'Kron-BFGS-wrong-unregularized-grad-momentum-grad',
                               'Kron-BFGS-regularized-grad-momentum-grad',
                               'Kron-BFGS-Hessian-action',
                               'Kron-BFGS-Hessian-action-unregularized-grad',
                               'Kron-BFGS-Hessian-action-momentum-grad',
                               'Kron-BFGS-Hessian-action-unregularized-grad-momentum-grad',
                               'Kron-BFGS-wrong-Hessian-action-unregularized-grad-momentum-grad',
                               'Kron-BFGS-LM',
                               'Kron-BFGS-LM-regularized-grad',
                               'Kron-BFGS-LM-sqrt-regularized-grad',
                               'Kron-BFGS-LM-sqrt-regularized-grad-momentum-grad',
                               'Kron-BFGS-LM-unregularized-grad',
                               'Kron-BFGS-LM-unregularized-grad-momentum-grad',
                               'Kron-BFGS-LM-regularized-grad-momentum-grad',
                               'Kron-BFGS-LM-momentum-grad',
                               'Kron-BFGS-1st-layer-only',
                               'Kron-BFGS-block',
                               'Kron-SGD',
                               'Kron-SGD-test']:
        args['matrix_name'] = 'EF'
    elif args['algorithm'] == 'RMSprop-individual-grad-no-sqrt-Fisher':
        args['matrix_name'] = 'Fisher'
    elif args['algorithm'] in ['SGD-yura-BD',
                               'SGD-yura-old',
                               'SGD-yura',
                               'SGD-yura-MA',
                               'SGD-VA',
                               'SGD-sign',
                               'SGD-signVA',
                               'SGD-signVAerf',
                               'SGD-signVAsqrt',
                               'SGD-momentum-yura',
                               'SGD-momentum',
                               'SGD-LRdecay-momentum',
                               'SGD',
                               'RMSprop',
                               'RMSprop-test',
                               'RMSprop-momentum-grad',
                               'RMSprop-warmStart-momentum-grad',
                               'RMSprop-momentum-grad-test',
                               'RMSprop-no-sqrt',
                               'shampoo',
                               'shampoo-momentum-grad',
                               'shampoo-allVariables-momentum-grad',
                               'shampoo-allVariables-warmStart-momentum-grad',
                               'shampoo-allVariables-warmStart-lessInverse-momentum-grad',
                               'shampoo-allVariables-filterFlattening-warmStart-momentum-grad',
                               'shampoo-allVariables-filterFlattening-warmStart-momentum-grad-LRdecay',
                               'shampoo-allVariables-filterFlattening-warmStart-lessInverse-momentum-grad',
                               'shampoo-allVariables-filterFlattening-warmStart-lessInverse-momentum-grad-LRdecay',
                               'shampoo-no-sqrt-momentum-grad',
                               'Adam-momentum-grad',
                               'Adam-noWarmStart-momentum-grad',
                               'Adam-noWarmStart-momentum-grad-LRdecay',
                               'BFGS',
                               'BFGS-homo']:
        args['matrix_name'] = 'None'
    else:
        print('Error: undefined matrix name for ' + args['algorithm'])
        sys.exit()
    
    return args

def get_warm_start(data_, params):
    
    N1 = params['N1']
    
    assert N1 < params['num_train_data']
    # i.e. stochastic setting
    
    device = params['device']
    numlayers = params['numlayers']
    layers_params = params['layers_params']
    
    model = data_['model']
        
    i = 0 # position of training data
    j = 0 # position of mini-batch

    from utils_git.utils_kfac import get_g_g_T, get_g_g_T_BN
    
    from utils_git.utils_shampoo import shampoo_kron_matrices_warm_start_per_variable
    
    print('Begin warm start...')

    while i + N1 <= params['num_train_data']:



        X_mb, t_mb = data_['dataset'].train.next_batch(N1)

#         X_mb = torch.from_numpy(X_mb).to(device)
        
        if not params['if_dataset_onTheFly']:
            X_mb = torch.from_numpy(X_mb)
        X_mb = X_mb.to(device)



        z, a, h = model.forward(X_mb)
        
        if params['matrix_name'] in ['Fisher',
                                     'Fisher-correct']:
            params['N2_index'] = list(range(N1))
            t_mb_pred = sample_from_pred_dist(z, params)
            del params['N2_index']

            t_mb_used = t_mb_pred
        elif params['matrix_name'] == 'None':
            # None is actually EF
            
#             print('t_mb')
#             print(t_mb)
            
#             t_mb = torch.from_numpy(t_mb).to(device)
            
            if not params['if_dataset_onTheFly']:
                t_mb = torch.from_numpy(t_mb)
            t_mb = t_mb.to(device)
            
            t_mb_used = t_mb
        else:
            print('params[matrix_name]')
            print(params['matrix_name'])
            sys.exit()
        
        '''
        if params['algorithm'] == 'shampoo-allVariables-warmStart':
            t_mb = torch.from_numpy(t_mb).to(device)
            t_mb_used = t_mb
        elif params['algorithm'] in ['kfac-warmStart-no-max-no-LM',
                                     'kfac-warmStart-lessInverse-no-max-no-LM',
                                     'kfac-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                                     'kfac-no-max-no-LM',
                                     'matrix-normal-allVariables-warmStart',
                                     'matrix-normal-allVariables-warmStart-noPerDimDamping']:
            params['N2_index'] = list(range(N1))
            t_mb_pred = sample_from_pred_dist(z, params)
            del params['N2_index']

            t_mb_used = t_mb_pred
        else:
            print('error: need to check for ' + params['algorithm'])
            sys.exit()
        '''
        
        

            

        loss = get_loss_from_z(model, z, t_mb_used, reduction='mean') # not regularized

        model.zero_grad()
        loss.backward()
        
        if params['if_model_grad_N2'] or\
        params['algorithm'] in ['shampoo-allVariables-warmStart',
                                'shampoo-allVariables-warmStart-lessInverse',
                                'shampoo-allVariables-filterFlattening-warmStart',
                                'shampoo-allVariables-filterFlattening-warmStart-lessInverse',]:
            
#             assert params['tau'] == 0
            
            model_grad_N2 = get_model_grad(model, params)

        i += N1
        j += 1

        for l in range(numlayers):
            # bar_A_j = 1 / j * (A_1 + ... + A_j)
            # bar_A_j = (j-1) / j * bar_A_{j-1} + 1 / j * A_j

#                     homo_h_l =\
#                     torch.cat((h[l], torch.ones(N1, 1, device=device)), dim=1)
#                     A_j = 1/N1 * torch.mm(homo_h_l.t(), homo_h_l).data


#             print('params[Kron_BFGS_if_homo]')
#             print(params['Kron_BFGS_if_homo'])

            
            if params['algorithm'] in ['matrix-normal-allVariables-warmStart',
                                       'matrix-normal-allVariables-warmStart-MaxEigDamping',
                                       'matrix-normal-allVariables-warmStart-noPerDimDamping',
                                       'matrix-normal-same-trace-warmStart',
                                       'matrix-normal-same-trace-warmStart-noPerDimDamping',
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
                                       'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart',
                                       'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-lessInverse',
                                       'shampoo-allVariables-warmStart',
                                       'shampoo-allVariables-warmStart-lessInverse',
                                       'shampoo-allVariables-filterFlattening-warmStart',
                                       'shampoo-allVariables-filterFlattening-warmStart-lessInverse',]:
                
                for name_variable in data_['model'].layers_weight[l].keys():
                    shampoo_kron_matrices_warm_start_per_variable(j, model_grad_N2, l, name_variable, data_, params)
                    
                    if params['if_Hessian_action'] and not i + N1 <= params['num_train_data']:
                    
                        # to constrain epsilon
                        
                        assert params['algorithm'] == 'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-lessInverse'
                        epsilon = params['shampoo_epsilon']
                
                        H = data_['shampoo_H']
                    
                        H_l_LM_minus_2k = []
                    
                        for ii in range(len(H[l][name_variable])):
                            
                            H_l_ii_LM = H[l][name_variable][ii] + epsilon * torch.eye(H[l][name_variable][ii].shape[0], device=device)
                        
                            H_l_LM_minus_2k.append(H_l_ii_LM.inverse())
                            
#                         sys.exit()
                        
                        data_['shampoo_H_LM_minus_2k'][l][name_variable] = H_l_LM_minus_2k
                
                
                    
            
                
            elif params['algorithm'] in ['kfac-no-max-no-LM',
                                         'kfac-warmStart-no-max-no-LM',
                                         'kfac-warmStart-lessInverse-no-max-no-LM',
                                         'kfac-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                                         'kfac-correctFisher-warmStart-no-max-no-LM',
                                         'kfac-correctFisher-warmStart-NoMaxNoSqrt-no-LM',
                                         'kfac-correctFisher-warmStart-lessInverse-no-max-no-LM',
                                         'kfac-correctFisher-warmStart-lessInverse-NoMaxNoSqrt-no-LM',]:
                
                if layers_params[l]['name'] in ['conv',
                                                'conv-no-activation',
                                                'conv-no-bias-no-activation',
                                                'fully-connected']:
                    
                
                    A_j = get_A_A_T(h, l, data_, params)
                    data_['A'][l] *= (j-1)/j
                    data_['A'][l] += 1/j * A_j

                    G_j = get_g_g_T(a, l, params)
                    data_['G'][l] *= (j-1)/j
                    data_['G'][l] += 1/j * G_j
                    
                elif layers_params[l]['name'] == 'BN':
                    
                    if params['kfac_if_update_BN'] and not params['kfac_if_BN_grad_direction']:
                    
                        G_j = get_g_g_T_BN(model, l, N1)
                    
                        data_['G'][l] *= (j-1)/j
                        data_['G'][l] += 1/j * G_j
                    
                else:
                    print('error: need to check for ' + layers_params[l]['name'])
                    sys.exit()
                    
            elif params['algorithm'] in ['Fisher-BD',]:
                
                print('i')
                print(i)
                
                
        
                G_j = get_block_Fisher(h, a, l, params)
                
                if j == 1:
                    data_['block_Fisher'][l] = G_j
                else:
                    data_['block_Fisher'][l] *= (j-1)/j
                    data_['block_Fisher'][l] += 1/j * G_j
                
#                 sys.exit()
                
            else:
                print('error: need to check for ' + params['algorithm'])
                sys.exit()
                
                
def get_h_l_unfolded_noHomo_noFlatten(h, l, params):
    # for the use of stride, see _extract_patches in
    # https://github.com/gpauloski/kfac_pytorch/blob/master/kfac/utils.py
    
    layers_params = params['layers_params']
    
    assert layers_params[l]['name'] in ['conv', 'conv-no-activation']
    
    padding = layers_params[l]['conv_padding']
    kernel_size = layers_params[l]['conv_kernel_size']
    device = params['device']
    
    stride = layers_params[l]['conv_stride']
    
    assert 2 * padding + 1 == kernel_size
    
    # 2d-conv: a[l]: M * I * |T|, where |T| has two dimensions
        
    # (Take Fashion-MNIST as an example)
    # h[l]: 1000 * 1 * 28 * 28
    # 1000: size of minibatch
    # 1: conv_in_channels
    # 28 * 28: size of input
    # h_l_padded_unfolded: 1000 * 1 * 32 * 32
    # 32 * 32: size of padded input
    h_l_padded = F.pad(
        h[l].data, (padding, padding, padding, padding), "constant", 0
    )

    # h_l_padded_unfolded: 1000 * 1 * 28 * 32 * 5
    # 5: conv_kernel_size
    h_l_padded_unfolded = h_l_padded.unfold(2, kernel_size, stride)

    h_l_padded_unfolded = h_l_padded_unfolded.unfold(3, kernel_size, stride)
    # h_l_padded_unfolded: 1000 * 1 * 28 * 28 * 5 * 5
    
#     return h_l_padded_unfolded

    h_l_padded_unfolded = h_l_padded_unfolded.permute(0, 2, 3, 1, 4, 5)
    # h_l_padded_unfolded: 1000 * 28 * 28 * 1 * 5 * 5

#     h_l_padded_unfolded = h_l_padded_unfolded.flatten(start_dim=3)
    # h_l_padded_unfolded: 1000 * 28 * 28 * 25
    
    
    return h_l_padded_unfolded
                
                
def get_h_l_unfolded_noHomo(h, l, params):
    # for the use of stride, see _extract_patches in
    # https://github.com/gpauloski/kfac_pytorch/blob/master/kfac/utils.py
    
    layers_params = params['layers_params']
    
    assert layers_params[l]['name'] in ['conv', 'conv-no-activation']
    
    padding = layers_params[l]['conv_padding']
    kernel_size = layers_params[l]['conv_kernel_size']
    device = params['device']
    
    stride = layers_params[l]['conv_stride']
    
    assert 2 * padding + 1 == kernel_size
    
    # 2d-conv: a[l]: M * I * |T|, where |T| has two dimensions
        
    # (Take Fashion-MNIST as an example)
    # h[l]: 1000 * 1 * 28 * 28
    # 1000: size of minibatch
    # 1: conv_in_channels
    # 28 * 28: size of input
    # h_l_padded_unfolded: 1000 * 1 * 32 * 32
    # 32 * 32: size of padded input
    h_l_padded = F.pad(
        h[l].data, (padding, padding, padding, padding), "constant", 0
    )

    # h_l_padded_unfolded: 1000 * 1 * 28 * 32 * 5
    # 5: conv_kernel_size
    h_l_padded_unfolded = h_l_padded.unfold(2, kernel_size, stride)

    h_l_padded_unfolded = h_l_padded_unfolded.unfold(3, kernel_size, stride)
    # h_l_padded_unfolded: 1000 * 1 * 28 * 28 * 5 * 5
    
#     return h_l_padded_unfolded

    h_l_padded_unfolded = h_l_padded_unfolded.permute(0, 2, 3, 1, 4, 5)
    # h_l_padded_unfolded: 1000 * 28 * 28 * 1 * 5 * 5

    h_l_padded_unfolded = h_l_padded_unfolded.flatten(start_dim=3)
    # h_l_padded_unfolded: 1000 * 28 * 28 * 25
    
    
    return h_l_padded_unfolded

                

def get_h_l_unfolded(h, l, data_, params):
    # for the use of stride, see _extract_patches in
    # https://github.com/gpauloski/kfac_pytorch/blob/master/kfac/utils.py
    
#     print('should be deprecated')
    
    layers_params = params['layers_params']
    
#     print('layers_params[l][name]')
#     print(layers_params[l]['name'])
    
    assert layers_params[l]['name'] in ['conv',
                                        'conv-no-activation',
                                        'conv-no-bias-no-activation']
    
    padding = layers_params[l]['conv_padding']
    kernel_size = layers_params[l]['conv_kernel_size']
    device = params['device']
    
    stride = layers_params[l]['conv_stride']
    
    assert 2 * padding + 1 == kernel_size
    
    # 2d-conv: a[l]: M * I * |T|, where |T| has two dimensions
        
    # (Take Fashion-MNIST as an example)
    # h[l]: 1000 * 1 * 28 * 28
    # 1000: size of minibatch
    # 1: conv_in_channels
    # 28 * 28: size of input
    # h_l_padded_unfolded: 1000 * 1 * 32 * 32
    # 32 * 32: size of padded input
    h_l_padded = F.pad(
        h[l].data, (padding, padding, padding, padding), "constant", 0
    )

    # h_l_padded_unfolded: 1000 * 1 * 28 * 32 * 5
    # 5: conv_kernel_size
#     h_l_padded_unfolded = h_l_padded.unfold(2, kernel_size, 1)
    h_l_padded_unfolded = h_l_padded.unfold(2, kernel_size, stride)

#     h_l_padded_unfolded = h_l_padded_unfolded.unfold(3, kernel_size, 1)
    h_l_padded_unfolded = h_l_padded_unfolded.unfold(3, kernel_size, stride)
    # h_l_padded_unfolded: 1000 * 1 * 28 * 28 * 5 * 5
    
#     print('need to change back the correct return')
    
#     return h_l_padded_unfolded
    
#     sys.exit()

    h_l_padded_unfolded = h_l_padded_unfolded.permute(0, 2, 3, 1, 4, 5)
    # h_l_padded_unfolded: 1000 * 28 * 28 * 1 * 5 * 5

    h_l_padded_unfolded = h_l_padded_unfolded.flatten(start_dim=3)
    # h_l_padded_unfolded: 1000 * 28 * 28 * 25


    if 'b' not in data_['model'].layers_weight[l].keys():
        
        pass

    elif params['Kron_BFGS_if_homo']:

        h_homo_ones = torch.ones(
        h_l_padded_unfolded.size(0), h_l_padded_unfolded.size(1), h_l_padded_unfolded.size(2), 1, device=device
    )

        h_l_padded_unfolded = torch.cat(
            (h_l_padded_unfolded, h_homo_ones), 
            dim=3
        )
        # h_l_padded_unfolded: 1000 * 28 * 28 * 26

    else:
        print('error: need to check')
        sys.exit()
    
    
    return h_l_padded_unfolded

def get_A_A_T_v_kfac_v2(v, h, l, params, data_):
    
    layers_params = params['layers_params']
    device = params['device']
    
    kernel_size = layers_params[l]['conv_kernel_size']
    padding = layers_params[l]['conv_padding']

    if layers_params[l]['name'] == '1d-conv':
        
        print('error: need to change so that it is averaged on minibatch')
        
        sys.exit()
        
        
        
        # 1d-conv: a[l]: M * I * |T|
        h_l_padded = F.pad(h[l].data, (padding, padding), "constant", 0)
        
        # M * J * |T| * |Delta|
        h_l_padded_unfolded = h_l_padded.unfold(2, kernel_size, 1)
        
        h_l_padded_unfolded = h_l_padded_unfolded.permute(0, 2, 1, 3)
        
        h_l_padded_unfolded = h_l_padded_unfolded.flatten(start_dim=2)
        
        
        print('following wrong for kfac?')
        sys.exit()
        
        if params['Kron_BFGS_if_homo']:
            
            h_homo_ones = torch.ones(
            h_l_padded_unfolded.size(0), h_l_padded_unfolded.size(1), 1, device=device
        )
            
            h_l_padded_unfolded = torch.cat(
                (h_l_padded_unfolded, h_homo_ones), 
                dim=2
            )
            
            
        
        test_2_A_j = torch.einsum('sti,stj->ij', h_l_padded_unfolded, h_l_padded_unfolded)
    elif layers_params[l]['name'] in ['conv',
                                      'conv-no-activation']:
        
#         h_l_padded_unfolded = data_['h_N2_unfolded'][l]
    
#         size_minibatch = h_l_padded_unfolded.size(0)
        
        
        
#         h_l_padded_unfolded_viewed = h_l_padded_unfolded.view(-1, h_l_padded_unfolded.size(-1))    
        
    
#         Av = torch.mv(h_l_padded_unfolded_viewed.t(), torch.mv(h_l_padded_unfolded_viewed, v))



    
        h_l_padded_unfolded_noHomo = data_['h_N2_unfolded_noHomo'][l]
        
        size_minibatch = h_l_padded_unfolded_noHomo.size(0)
    
        h_l_padded_unfolded_noHomo_viewed = h_l_padded_unfolded_noHomo.view(-1, h_l_padded_unfolded_noHomo.size(-1))






#         h_l_padded_unfolded_noHomo_noFlatten = data_['h_N2_unfolded_noHomo_noFlatten'][l]

#         size_minibatch = h_l_padded_unfolded_noHomo_noFlatten.size(0)
    
        if params['Kron_BFGS_if_homo']:
            
            Av = torch.mv(h_l_padded_unfolded_noHomo_viewed, v[:-1]) + v[-1].item()
            
            Av = torch.cat((torch.mv(h_l_padded_unfolded_noHomo_viewed.t(), Av), torch.sum(Av).unsqueeze(dim=0)))

#             v_viewed = v[:-1].view(h_l_padded_unfolded_noHomo_noFlatten.size(-3), h_l_padded_unfolded_noHomo_noFlatten.size(-2), h_l_padded_unfolded_noHomo_noFlatten.size(-1))
            
#             Av = torch.einsum('ijklmn,lmn->ijk', h_l_padded_unfolded_noHomo_noFlatten, v_viewed) + v[-1].item()
            
#             Av = torch.cat(
#                 (
#                     torch.einsum('ijklmn,ijk->lmn', h_l_padded_unfolded_noHomo_noFlatten, Av).view(-1),
#                     torch.sum(Av).unsqueeze(dim=0)
#                 )
#             )
        else:
            print('error: need to check')
            sys.exit()
        
        Av = Av / size_minibatch
        
        
       
    else:
        print('error: not implemented')
        sys.exit()
    
                
#     return test_2_A_j
    return Av


def get_A_A_T_kfac_v2(h, l, data_, params):
    
    layers_params = params['layers_params']
    device = params['device']
    
    kernel_size = layers_params[l]['conv_kernel_size']
    padding = layers_params[l]['conv_padding']


#     if layers_params[l]['name'] == '1d-conv':
#         size_test_2_A_j = kernel_size *\
# layers_params[l]['conv_in_channels']
#     elif layers_params[l]['name'] == 'conv':
        
#         size_test_2_A_j = kernel_size**2 *\
# layers_params[l]['conv_in_channels']


#     test_2_A_j = torch.zeros(
#         size_test_2_A_j, size_test_2_A_j, device=device
#     )

    if layers_params[l]['name'] == '1d-conv':
        
        print('need to make it averaged over minibatch')
        
        
        
        # 1d-conv: a[l]: M * I * |T|
        h_l_padded = F.pad(h[l].data, (padding, padding), "constant", 0)
        
        # M * J * |T| * |Delta|
        h_l_padded_unfolded = h_l_padded.unfold(2, kernel_size, 1)
        
        h_l_padded_unfolded = h_l_padded_unfolded.permute(0, 2, 1, 3)
        
        h_l_padded_unfolded = h_l_padded_unfolded.flatten(start_dim=2)
        
        
        print('following wrong for kfac?')
        sys.exit()
        
        if params['Kron_BFGS_if_homo']:
            
            h_homo_ones = torch.ones(
            h_l_padded_unfolded.size(0), h_l_padded_unfolded.size(1), 1, device=device
        )
            
            h_l_padded_unfolded = torch.cat(
                (h_l_padded_unfolded, h_homo_ones), 
                dim=2
            )
            
            
        
        test_2_A_j = torch.einsum('sti,stj->ij', h_l_padded_unfolded, h_l_padded_unfolded)
        
        
        
#         sys.exit()
        
#         h_homo_ones = torch.ones(h_l_padded.size(0), 1 ,device=device)
        
        

#         for t in range(a[l].size(2)):
            # a[l].size(2) = |T|
            
#             h_l_t = h_l_padded[:, :, t:t+kernel_size].data

            # in the flatten, delta changes the fastest
#             h_l_t_flatten = h_l_t.flatten(start_dim=1)
            
#             if params['Kron_BFGS_if_homo']:
#                 h_l_t_flatten = torch.cat((h_l_t_flatten, h_homo_ones), dim=1)

#             test_2_A_j += torch.mm(h_l_t_flatten.t(), h_l_t_flatten)
    elif layers_params[l]['name'] in ['conv',
                                      'conv-no-activation',
                                      'conv-no-bias-no-activation']:
        
        h_l_padded_unfolded = get_h_l_unfolded(h, l, data_, params)
    
        size_minibatch = h_l_padded_unfolded.size(0)
        
        
            

        
        test_2_A_j = torch.einsum('stli,stlj->ij', h_l_padded_unfolded, h_l_padded_unfolded)
        # test_2_A_j: 26 * 26
        
        test_2_A_j = test_2_A_j / size_minibatch
        
#         h_homo_ones = torch.ones(h_l_padded.size(0), 1 ,device=device)

#         for t1 in range(a[l].size(2)):
#             for t2 in range(a[l].size(3)):
                
#                 h_l_t =\
# h_l_padded[:, :, t1:t1+kernel_size, t2:t2+kernel_size].data

#                 h_l_t_flatten = h_l_t.flatten(start_dim=1)
                
#                 if params['Kron_BFGS_if_homo']:
#                     h_l_t_flatten = torch.cat((h_l_t_flatten, h_homo_ones), dim=1)
    
#                 test_2_A_j += torch.mm(h_l_t_flatten.t(), h_l_t_flatten)
       
    else:
        print('error: not implemented for ' + layers_params[l]['name'])
        sys.exit()
    
                
    return test_2_A_j

def get_A_A_T_v(v, h, l, params, data_):
    
    layers_params = params['layers_params']
    
    device = params['device']
    
    if layers_params[l]['name'] == 'fully-connected':
        
        
        
        size_minibatch = h[l].size(0)
    
        if params['algorithm'] in ['kfac-no-max-no-LM',
                                   'kfac-warmStart-no-max-no-LM',
                                   'kfac-warmStart-lessInverse-no-max-no-LM'] or\
        params['Kron_BFGS_if_homo']:
            
            homo_h_l = torch.cat(
                (h[l], torch.ones(size_minibatch, 1, device=device)),
                dim=1
            )
        elif algorithm in ['Kron-BFGS',
                           'Kron-BFGS-no-norm-gate',
                           'Kron-BFGS-no-norm-gate-momentum-s-y',
                           'Kron-BFGS-no-norm-gate-momentum-s-y-damping',
                           'Kron-BFGS-no-norm-gate-damping',
                           'Kron-BFGS-no-norm-gate-Shiqian-damping',
                           'Kron-BFGS-wrong',
                           'Kron-BFGS-Hessian-action',
                           'Kron-BFGS-wrong-Hessian-action',
                           'Kron-BFGS-LM',
                           'Kron-BFGS-LM-sqrt']:
            homo_h_l = h[l]
        else:
            print('error: not implemented')
            sys.exit()
            
#         print('homo_h_l.size()')
#         print(homo_h_l.size())
        
#         print('v.size()')
#         print(v.size())
            
#         sys.exit()

#         Av = (1/size_minibatch * torch.mm(homo_h_l.t(), homo_h_l).data) * v
#         Av = (1/size_minibatch * homo_h_l.t() * homo_h_l) * v
        Av = torch.mv(homo_h_l.t(), torch.mv(homo_h_l, v)) /size_minibatch

    elif layers_params[l]['name'] in ['1d-conv',
                                      'conv',
                                      'conv-no-activation',
                                      'conv-no-bias-no-activation',]:
        
        
        
        
        




        ########################################

        # a more space consuming but possibly faster way

        Av = get_A_A_T_v_kfac_v2(v, h, l, params, data_)






#         A_j = test_2_A_j
    else:
        print('error in get_A_A_T unknown: ' + layers_params[l]['name'])
        sys.exit()
    
    return Av


def get_A_A_T(h, l, data_, params):
    
    # return the AVERAGED A_A_T over a minibatch
    
    layers_params = params['layers_params']
    
    device = params['device']
    
    if layers_params[l]['name'] == 'fully-connected':
#         N1 = params['N1']
        
        size_minibatch = h[l].size(0)
    
#         if params['algorithm'] in ['kfac-no-max-no-LM',
#                                    'kfac-warmStart-no-max-no-LM',
#                                    'kfac-warmStart-lessInverse-no-max-no-LM'] or\
#         params['Kron_BFGS_if_homo']:
        if params['Kron_BFGS_if_homo']:
            
            homo_h_l = torch.cat(
                (h[l], torch.ones(size_minibatch, 1, device=device)),
                dim=1
            )
        elif algorithm in ['Kron-BFGS',
                           'Kron-BFGS-no-norm-gate',
                           'Kron-BFGS-no-norm-gate-momentum-s-y',
                           'Kron-BFGS-no-norm-gate-momentum-s-y-damping',
                           'Kron-BFGS-no-norm-gate-damping',
                           'Kron-BFGS-no-norm-gate-Shiqian-damping',
                           'Kron-BFGS-wrong',
                           'Kron-BFGS-Hessian-action',
                           'Kron-BFGS-wrong-Hessian-action',
                           'Kron-BFGS-LM',
                           'Kron-BFGS-LM-sqrt']:
            homo_h_l = h[l]
        else:
            print('error: not implemented')
            sys.exit()

        A_j = 1/size_minibatch * torch.mm(homo_h_l.t(), homo_h_l).data

    elif layers_params[l]['name'] in ['1d-conv',
                                      'conv',
                                      'conv-no-activation',
                                      'conv-no-bias-no-activation',]:
    
        



        ####################################

        # benchmark, not use for now

        # get_A_A_T_kfac_v2 is significantly faster than this

    #                         test_2_A_j = get_A_A_T_kfac(a, h, l, params)




        ########################################

        # a more space consuming but possibly faster way
        
#         from utils_git.utils import get_A_A_T_kfac_v2
#         print('need to move the function to this file')

        test_2_A_j = get_A_A_T_kfac_v2(h, l, data_, params)



        #########################################################
        # my original way (worked for in-channel = 1)
        # then improve (worked for 1d-conv)
        # then improve
        # then use F.conv 
        # then revise the part of diff
        # then re-do einsum
        # then re-do slicing



    #                         test_8_A_j = get_A_A_T_kron_bfgs_v5(h, l, params)



        #########################################################
        # my original way (worked for in-channel = 1)
        # then improve (worked for 1d-conv)
        # then improve
        # then use F.conv 
        # then revise the part of diff
        # then re-do einsum

        # clearly slowert than v5



    #                         test_7_A_j = get_A_A_T_kron_bfgs_v4(h, l, params)

        #########################################################
        # my original way (worked for in-channel = 1)
        # then improve (worked for 1d-conv)
        # then improve
        # then use F.conv 
        # then revise the part of diff

        # slower than v4



    #                         test_6_A_j = get_A_A_T_kron_bfgs_v3(h, l, params)

        #########################################################
        # my original way (worked for in-channel = 1)
        # then improve (worked for 1d-conv)
        # then improve
        # then use F.conv 

        # slower than v3 in 2d conv



    #                         test_5_A_j = get_A_A_T_kron_bfgs_v2(h, l, params)



        #########################################################
        # my original way (worked for in-channel = 1)
        # then improve (worked for 1d-conv)
        # then improve

        # this is always slower than get_A_A_T_kron_bfgs_v2

    #                         test_6_A_j = get_A_A_T_kron_bfgs(h, l, params)






        A_j = test_2_A_j
    else:
        print('error in get_A_A_T unknown: ' + layers_params[l]['name'])
        sys.exit()
    
    return A_j

def from_homo_to_weight_and_bias(homo_delta_l, l, params):
    
    layers_params_l = params['layers_params'][l]
    
    delta_l = {}
    if layers_params_l['name'] == 'fully-connected':
        delta_l['W'] = homo_delta_l[:, :-1]
        delta_l['b'] = homo_delta_l[:, -1]
    elif layers_params_l['name'] in ['conv',
                                     'conv-no-activation']:
        # take Fashion-MNIST as an example
        # model_grad_N1[l]['W']: 32 * 1 * 5 * 5
        # model_grad_N1[l]['b']: 32
        # 32: conv_out_channels
        # 1: conv_in_channels
        # 5 * 5: conv_kernel_size



        delta_l['b'] = homo_delta_l[:, -1]
        
#         delta_l['W'] = homo_delta_l[:, :-1].reshape(model_grad_N1[l]['W'].size())
        delta_l['W'] = homo_delta_l[:, :-1].reshape(
            (homo_delta_l.size(0), layers_params_l['conv_in_channels'], layers_params_l['conv_kernel_size'], layers_params_l['conv_kernel_size'])
        )
    
    elif layers_params_l['name'] in ['conv-no-bias-no-activation']:
        
        delta_l['W'] = homo_delta_l.reshape(
            (homo_delta_l.size(0), layers_params_l['conv_in_channels'], layers_params_l['conv_kernel_size'], layers_params_l['conv_kernel_size'])
        )

    elif layers_params_l['name'] in ['BN']:
        
        assert homo_delta_l.size(0) == 2 * layers_params_l['num_features']
        
        delta_l['W'] = homo_delta_l[:layers_params_l['num_features']]
        delta_l['b'] = homo_delta_l[layers_params_l['num_features']:]
        
#         sys.exit()

    else:
        print('Error: unsupported layer when store the data for ' + params['layers_params'][l]['name'])
        sys.exit()
        
    return delta_l

def get_best_params(args, if_plot):
    
    result_path = args['home_path'] + 'result/'
    
    if 'algorithm_dict' in args:
        algorithm_dict = args['algorithm_dict']
    else:
        algorithm_dict = {}
        algorithm_dict['name'] = args['algorithm']
        algorithm_dict['params'] = {}
    
    
    # plot lr vs test accuracy
    if 'list_lr' in args:
        list_lr_try = args['list_lr']
    else:
        '''
        list_lr_try = [\
            0.0005,
            0.001,
            0.005,
            0.01,
            0.05,
            0.1,
            0.2,
            0.4,
            0.5,
            1,
            2,
            4,
            10]
        list_lr_try = [\
            0.0003,
            0.001,
            0.003,
            0.01,
            0.03,
            0.1,
            0.3,
            1,
            3,
            10]
        # list_lr = [\
            # 0.005]
        # print('test lr')
        '''
        
        print('algorithm_dict[name]')
        print(algorithm_dict['name'])
        
        fake_params = {}
        fake_params['algorithm'] = algorithm_dict['name']
        fake_params['if_gpu'] = args['if_gpu']

        test_list_lr_try = os.listdir(result_path + args['dataset'] + '/' + get_name_algorithm(fake_params)[0] + '/')
        
#         print('test_list_lr_try')
#         print(test_list_lr_try)

        test_list_lr_try = [lr_ for lr_ in test_list_lr_try if lr_.startswith('alpha_')]

        test_list_lr_try = [lr_.replace('alpha_','') for lr_ in test_list_lr_try]

#         test_list_lr_try = [float(lr_) for lr_ in test_list_lr_try]

#         print('test_list_lr_try')
#         print(test_list_lr_try)
        
        list_lr_try = test_list_lr_try
        
#         print('list_lr_try')
#         print(list_lr_try)
        
#         print('sorted(list_lr_try, key=float)')
#         print(sorted(list_lr_try, key=float))
        
        list_lr_try = sorted(list_lr_try, key=float)
        
#         list_lr_try = sorted([float(lr) for lr in list_lr_try])
        
#         print('list_lr_try')
#         print(list_lr_try)
        
#         list_lr_try = [str(lr) for lr in list_lr_try]
        
        print('list_lr_try')
        print(list_lr_try)


    

    


    
    
    

    os.chdir(result_path)

    
    list_acc = []
    list_name_result_pkl = []
    list_lr = []
#     for lr, epsilon in itertools.product(list_lr_try, list_epsilon_try):
    for lr in list_lr_try:

        fake_params = {}
        fake_params['alpha'] = lr
        fake_params['N1'] = args['N1']
        fake_params['N2'] = args['N2']

        # fake_params['algorithm'] = args['algorithm']
        fake_params['algorithm'] = algorithm_dict['name']
        fake_params['if_gpu'] = args['if_gpu']

        name_algorithm_with_params = get_name_algorithm_with_params(fake_params)

        path_to_result = args['dataset'] + '/' + name_algorithm_with_params + '/'
        


        if os.path.isdir(path_to_result):
            onlyfiles = [f for f in os.listdir(
            path_to_result) if os.path.isfile(os.path.join(path_to_result, f))]
        else:
            continue
        
#         print('os.listdir(path_to_result)')
#         print(os.listdir(path_to_result))
            
#         print('onlyfiles')
#         print(onlyfiles)

        for f_ in onlyfiles:
            with open(path_to_result + f_, 'rb') as handle:
                
#                 print('path_to_result + f_')
#                 print(path_to_result + f_)
                
#                 print('handle')
#                 print(handle)
                
                record_result = pickle.load(handle)

#             print('params in algorithm_dict')
#             print('params' in algorithm_dict)

            if_candidate_result = True
            if 'params' in algorithm_dict:
                if 'params' in record_result:
                    for key in algorithm_dict['params']:
                        if key in record_result['params']:
                            if algorithm_dict['params'][key] != record_result['params'][key]:
                                if_candidate_result = False
                                break
                        else:
                            if_candidate_result = False
                            break
                else:
                    if algorithm_dict['params'] == {}:
                        # if ('params' in algorithm_dict) and ('params' not in record_result)
                        # and (algorithm_dict['params'] == {})
                        1
                    else:
                        if_candidate_result = False
            else:
                if 'params' in record_result:
                    if_candidate_result = False
                    


            if if_candidate_result == False:
                continue

#             print('record_result.keys()')
#             print(record_result.keys())
            
            if args['tuning_criterion'] in ['test_acc',
                                            'train_acc',
                                            'train_minibatch_acc']:
                
                if args['tuning_criterion'] == 'train_acc':
                    
                    if 'train_acces' in record_result:
                        record_acc = record_result['train_acces']
                    else:
                        print('error: train_acces not in record_result')
                        sys.exit()
                elif args['tuning_criterion'] == 'train_minibatch_acc':
                    
                    assert 'train_minibatch_acces' in record_result
                    
                    record_acc = record_result['train_minibatch_acces']
                    
                elif args['tuning_criterion'] == 'test_acc':
                    if 'test_acces' in record_result:
                        record_acc = record_result['test_acces']
                    else:
                        record_acc = record_result['acces']
                else:
                    print('error: need to check for ' + args['tuning_criterion'])
                    sys.exit()
                
                    

                if args['name_loss'] in ['logistic-regression',
                                         'logistic-regression-sum-loss',
                                         'linear-regression',
                                         'linear-regression-half-MSE']:
                    list_acc.append(np.min(record_acc))
                elif args['name_loss'] in ['multi-class classification',
                                           'binary classification']:
                    list_acc.append(np.max(record_acc))
                else:
                    print('Error: unknown name loss.')
                    sys.exit()
                    
                    
            elif args['tuning_criterion'] == 'train_loss':
                list_acc.append(np.min(record_result['train_losses']))
            elif args['tuning_criterion'] == 'train_minibatch_loss':
                list_acc.append(np.min(record_result['train_unregularized_minibatch_losses']))
            else:
                print('error: unknown tuning criterion for ' + args['tuning_criterion'])
                sys.exit()

            list_name_result_pkl.append(f_)
            list_lr.append(lr)

            
    if list_acc == []:
        return None, None, None
    





    # save the best lr result
    os.chdir(args['home_path'] + 'result/')


    list_acc = np.asarray(list_acc)
    list_name_result_pkl = np.asarray(list_name_result_pkl)
    
    if args['tuning_criterion'] in ['test_acc',
                                    'train_acc',
                                    'train_minibatch_acc']:

        if args['name_loss'] in ['logistic-regression',
                                 'logistic-regression-sum-loss',
                                 'linear-regression',
                                 'linear-regression-half-MSE']:
            # max_indices = np.unravel_index(np.argmin(list_acc, axis=None), list_acc.shape)
            max_indices = np.argmin(list_acc, axis=None)
        elif args['name_loss'] in ['multi-class classification',
                                   'binary classification']:
            # max_indices = np.unravel_index(np.argmax(list_acc, axis=None), list_acc.shape)
            max_indices = np.argmax(list_acc, axis=None)
        else:
            print('Error: unknown name loss when max indices.')
            sys.exit()
            
    elif args['tuning_criterion'] in ['train_loss',
                                      'train_minibatch_loss']:
        
        max_indices = np.argmin(list_acc, axis=None)
            
    else:
        print('error: unknown tuning criterion 2 for ' + args['tuning_criterion'])
        sys.exit()

    print('list_acc[max_indices]')
    print(list_acc[max_indices])

    

    best_lr = list_lr[max_indices]
    best_name_result_pkl = list_name_result_pkl[max_indices]


    # save best params
    fake_params = {}

    # fake_params['algorithm'] = args['algorithm']
    fake_params['algorithm'] = algorithm_dict['name']

    fake_params['if_gpu'] = args['if_gpu']

    name_algorithm, _ = get_name_algorithm(fake_params)
    

    
#     np.savez(
#         args['dataset'] + '/' + name_algorithm + '/' + 'best_params' + '.npz',
#             best_lr=best_lr, best_epsilon=best_epsilon
#     )
    np.savez(
        args['dataset'] + '/' + name_algorithm + '/' + 'best_params' + '.npz',
            best_lr=best_lr
    )

    
    # visualize how to find the best
    if len(list_lr) == len(list_acc) and if_plot:
        

        plt.plot(list_lr, list_acc)
        plt.xlabel('learning rate')
        plt.ylabel('test accuracy')
        plt.xscale('log')
        # plt.title(name_result)
        plt.title(args['dataset'] + '/' + name_algorithm)

        if not os.path.exists(args['home_path'] + 'logs/plot_tune_lr/'):
            os.makedirs(args['home_path'] + 'logs/plot_tune_lr/')
        plt.savefig(args['home_path'] + 'logs/plot_tune_lr/' + str(datetime.datetime.now()) + '.pdf')
        plt.show()
        
#     print('best_lr, best_name_result_pkl')
#     print(best_lr, best_name_result_pkl)


    return best_lr, None, best_name_result_pkl



def get_name_algorithm_with_params(params):
    name_algorithm, _ = get_name_algorithm(params)
    
    if isinstance(params['alpha'], str):
        name_algorithm_with_params = name_algorithm + '/' +\
    'alpha_' + params['alpha'] + '/' +\
    'N1_' + str(params['N1']) + '/' +\
    'N2_' + str(params['N2'])
    else:
        name_algorithm_with_params = name_algorithm + '/' +\
    'alpha_' + str(params['alpha']) + '/' +\
    'N1_' + str(params['N1']) + '/' +\
    'N2_' + str(params['N2'])
    
    return name_algorithm_with_params

def get_name_algorithm(params):
    no_algorithm = 'if_gpu_' + str(params['if_gpu'])
    name_algorithm = params['algorithm'] + '/' + no_algorithm
    return name_algorithm, no_algorithm

def get_model_grad(model, params):
    model_grad_torch = []
    for l in range(model.numlayers):
        # model_grad_l = {}
        model_grad_torch_l = {}
        for key in model.layers_weight[l]:
            
            
            
            model_grad_torch_l[key] = copy.deepcopy(model.layers_weight[l][key].grad)
            
#             print('model.layers_weight[l][key].size()')
#             print(model.layers_weight[l][key].size())
#             print('model.layers_weight[l][key]')
#             print(model.layers_weight[l][key])
#             print('model.layers_weight[l][key].grad.size()')
#             print(model.layers_weight[l][key].grad.size())
#             print('model_grad_torch_l[key].size()')
#             print(model_grad_torch_l[key].size())
            
        # model_grad.append(model_grad_l)
        model_grad_torch.append(model_grad_torch_l)
        
    return model_grad_torch


    

def get_statistics(X_train):
    # X_train: N * m

    print('\n')
    print('max value:')
    print(np.max(X_train))
    print('min value:')
    print(np.min(X_train))

    print('max of per feature mean:')
    print(np.max(np.mean(X_train, axis=0)))
    print('min of per feature mean:')
    print(np.min(np.mean(X_train, axis=0)))

    print('max of per feature std:')
    print(np.max(np.std(X_train, axis=0)))
    print('min of per feature std:')
    print(np.min(np.std(X_train, axis=0)))

    print('\n')
    
    
def sample_from_pred_dist(z, params):
    
    name_loss = params['name_loss']
    N2_index = params['N2_index']

    if name_loss == 'multi-class classification':
        from torch.utils.data import WeightedRandomSampler

#         pred_dist_N2 = F.softmax(a[-1][N2_index], dim=1)
        pred_dist_N2 = F.softmax(z[N2_index], dim=1)

        t_mb_pred_N2 = list(WeightedRandomSampler(pred_dist_N2, 1))
        
        
#         print('torch.tensor(t_mb_pred_N2).to(params[device])')
#         print(torch.tensor(t_mb_pred_N2).to(params['device']))
#         print('torch.tensor(t_mb_pred_N2).to(params[device]).size()')
#         print(torch.tensor(t_mb_pred_N2).to(params['device']).size())
#         print('torch.tensor(t_mb_pred_N2).to(params[device]).dtype')
#         print(torch.tensor(t_mb_pred_N2).to(params['device']).dtype)
        
        t_mb_pred_N2 = torch.tensor(t_mb_pred_N2)
        t_mb_pred_N2 = t_mb_pred_N2.squeeze(dim=1)
        # this will gives a int64 tensor
        
#         print('test_t_mb_pred_N2')
#         print(test_t_mb_pred_N2)
#         print('test_t_mb_pred_N2.size()')
#         print(test_t_mb_pred_N2.size())
#         print('test_t_mb_pred_N2.dtype')
#         print(test_t_mb_pred_N2.dtype)
        

        
#         t_mb_pred_N2 = np.asarray(t_mb_pred_N2)
#         t_mb_pred_N2 = np.squeeze(t_mb_pred_N2, axis=1)
#         t_mb_pred_N2 = torch.from_numpy(t_mb_pred_N2).to(params['device'])


    elif name_loss == 'binary classification':

        pred_dist_N2 = torch.sigmoid(a[-1][N2_index]).cpu().data.numpy()

        t_mb_pred_N2 = np.random.binomial(n=1, p=pred_dist_N2)

        t_mb_pred_N2 = np.squeeze(t_mb_pred_N2, axis=1)

        print('check if need long')
        sys.exit()

        t_mb_pred_N2 = torch.from_numpy(t_mb_pred_N2).long()



    elif name_loss in ['logistic-regression',
                       'logistic-regression-sum-loss']:
        # pred_dist_N2 = torch.sigmoid(a[-1][N2_index]).cpu().data.numpy()
#         pred_dist_N2 = torch.sigmoid(a[-1][N2_index]).data
        pred_dist_N2 = torch.sigmoid(z[N2_index]).data

        if not (torch.max(pred_dist_N2) <= 1):
            print('torch.max(pred_dist_N2)')
            print(torch.max(pred_dist_N2))
            print('a[-1][N2_index]')
            print(a[-1][N2_index])
            print('get_if_nan(model.layers_weight)')
            print(get_if_nan(model.layers_weight))
            for l in range(len(model.layers_weight)):
                for key in model.layers_weight[l]:
                    print('torch.max(model.layers_weight[l][key])')
                    print(torch.max(model.layers_weight[l][key]))
                    print('torch.min(model.layers_weight[l][key])')
                    print(torch.min(model.layers_weight[l][key]))
        if not (torch.min(pred_dist_N2) >= 0):
            print('torch.min(pred_dist_N2)')
            print(torch.min(pred_dist_N2))
            print('a[-1][N2_index]')
            print(a[-1][N2_index])
            print('get_if_nan(model.layers_weight)')
            print(get_if_nan(model.layers_weight))
            for l in range(len(model.layers_weight)):
                for key in model.layers_weight[l]:
                    print('torch.max(model.layers_weight[l][key])')
                    print(torch.max(model.layers_weight[l][key]))
                    print('torch.min(model.layers_weight[l][key])')
                    print(torch.min(model.layers_weight[l][key]))

        # t_mb_pred_N2 = np.random.binomial(n=1, p=pred_dist_N2)
        t_mb_pred_N2 = torch.distributions.Bernoulli(pred_dist_N2).sample()

#                 t_mb_pred_N2 = t_mb_pred_N2.long().to(params['device'])
        t_mb_pred_N2 = t_mb_pred_N2
    
    elif name_loss == 'linear-regression':
        # see register_normal_predictive_distribution in
        # https://github.com/tensorflow/kfac/blob/master/kfac/python/ops/layer_collection.py
        
        # oevrall loss = 2_norm / #minibacth / # feature
        # loss for one data = 2_norm / # feature
        # let coeff = 1, 1 / (2 * var) = 1 / #feature
        # => 2 * var = # feature
        # => var = #feature / 2
        
        # oevrall loss = 2_norm / #minibacth
        # loss for one data = 2_norm
        # let coeff = 1, 1 / (2 * var) = 1
        # => 2 * var = 1
        # => var = 1 / 2
        
        # #feature = z.size(1)
        

#         t_mb_pred_N2 = torch.distributions.Normal(loc=z[N2_index], scale=z.size(1)/2).sample()
        
        t_mb_pred_N2 = torch.distributions.Normal(loc=z[N2_index], scale=1/2).sample()
    
    elif name_loss == 'linear-regression-half-MSE':
        
        # oevrall loss = 2_norm / #minibacth / 2
        # loss for one data = 2_norm / 2
        # let coeff = 1, 1 / (2 * var) = 1 / 2
        # => 2 * var = 2
        # => var = 1
        
        t_mb_pred_N2 = torch.distributions.Normal(loc=z[N2_index], scale=1).sample()
        
#         sys.exit()

    else:
        print('Error: sampling not supported.')
        sys.exit()
        
    t_mb_pred_N2 = t_mb_pred_N2.to(params['device'])
        
    return t_mb_pred_N2


def get_second_order_caches(z, a, h, data_, params):
        
    matrix_name = params['matrix_name']
    model = data_['model']
        
    N1 = params['N1']
    N2 = params['N2']
    
    assert N1 == N2
    
    
    
    if matrix_name in ['Fisher',
                       'EF']:

#     N2_index = np.random.permutation(N1)[:N2]
        N2_index = np.random.permutation(N1)
    elif matrix_name == 'Fisher-correct':
        N2_index = list(range(N1))
    else:
        
        print('matrix_name')
        print(matrix_name)
        
        sys.exit()
    
        
    
    

    X_mb = data_['X_mb']

    
    
    assert params['if_different_minibatch'] == False

    if params['if_different_minibatch']:
        
        print('error: should not reach here')
        sys.exit()
        
        X_mb_N2, _ = data_['dataset'].train.next_batch(N2)
        X_mb_N2 = torch.from_numpy(X_mb_N2).to(params['device'])
        # if name_dataset == 'MNIST-autoencoder':
            # t_mb = X_mb
    else:
        
        if matrix_name in ['Fisher',
                           'EF']:
        
            X_mb_N2 = X_mb[N2_index]
        elif matrix_name == 'Fisher-correct':
            X_mb_N2 = X_mb
        else:
            print('matrix_name')
            print(matrix_name)
            sys.exit()
            
        
    params['N2_index'] = N2_index
    
    data_['X_mb_N1'] = X_mb
    data_['X_mb_N2'] = X_mb_N2

    










    if matrix_name == 'EF':
        if params['if_different_minibatch']:
            print('error: need to check for different minibatch when EF')
            sys.exit()
        else:



            t_mb = data_['t_mb']

            data_['t_mb_pred_N2'] = t_mb[N2_index]

#                 data_['a_grad_N2'] = [N2 * (a_l.grad)[N2_index] for a_l in a]
            data_['mean_a_grad_N2'] = [torch.mean(N2 * (a_l.grad)[N2_index], dim=0).data for a_l in a]
            # use in K-BFGS, for BFGS on G

#                 for h_l in h:
#                     print('h_l.size()')
#                     print(h_l.size())


            data_['h_N2'] = [h_l[N2_index].data if len(h_l) else [] for h_l in h]

#                 data_['a_N2'] = [a_l[N2_index].data for a_l in a]
            data_['mean_a_N2'] = [torch.mean(a_l[N2_index], dim=0).data for a_l in a]

            
    elif matrix_name == 'Fisher-correct':
        
#         assert params['algorithm'] in params['list_algorithm_shampoo']
        
        if params['algorithm'] in params['list_algorithm_shampoo'] and params['i'] % params['shampoo_update_freq'] != 0:
            return data_
        
        if params['algorithm'] in params['list_algorithm_kfac'] and params['i'] % params['kfac_cov_update_freq'] != 0:
            return data_

#         z, a_N2, h_N2 = model.forward(X_mb_N2)

        a_N2 = a
        h_N2 = h

        t_mb_pred_N2 = sample_from_pred_dist(z, params)


        data_['t_mb_pred_N2'] = t_mb_pred_N2
        
        

        

        reduction = 'mean'
        loss = get_loss_from_z(model, z, t_mb_pred_N2, reduction) # this is unregularized loss


        model.zero_grad()

        loss.backward()
#         loss.backward(retain_graph=True)
        
        if params['algorithm'] in ['Fisher-BD',
                                   'kfac-correctFisher-warmStart-no-max-no-LM',
                                   'kfac-correctFisher-warmStart-NoMaxNoSqrt-no-LM',
                                   'kfac-correctFisher-warmStart-lessInverse-no-max-no-LM',
                                   'kfac-correctFisher-warmStart-lessInverse-NoMaxNoSqrt-no-LM',]:


#         data_['a_grad_N2'] = [N2 * (a_l.grad) for a_l in a_N2]
        # not used for the algorithms that we currently care about

            data_['h_N2'] = h_N2

            data_['a_N2'] = a_N2



        if params['if_model_grad_N2']:
            # this is unregularized grad
            data_['model_grad_N2'] = get_model_grad(model, params)

    elif matrix_name == 'GN':
        # print('error: need to check for different minibatch')
        # sys.exit()

        # test_time_wall_clock = time.time()

        data_['z_N2'] = a[-1][N2_index]

        m_L = data_['model'].layersizes[-1]
        params['m_L'] = m_L

        # X_mb_N2 = X_mb[N2_index]

        # numlayers = params['numlayers']
        # m_L = params['m_L']
        N2 = params['N2']

        list_a = [ [] for i in range(m_L) ]

        a_grad_momentum = []
        for l in range(model.numlayers):
            a_grad_momentum.append(
                torch.zeros(m_L, N2, data_['model'].layersizes[l+1], device=params['device']))

        for i in range(m_L):



            # z, list_a[i], h = model.forward(X_mb[N2_index])
            z, a, h = model.forward(X_mb[N2_index])

            # print('time for GN after forward:')
            # print(time.time() - test_time_wall_clock)

            fake_loss = torch.sum(z[:, i])

            test_time_zero = time.time()

            # model = get_model_grad_zerod(model)
            model.zero_grad()

            # print('time for GN zero:')
            # print(time.time() - test_time_zero)

            fake_loss.backward()

            # print('time for GN before second zero:')
            # print(time.time() - test_time_wall_clock)

            for l in range(model.numlayers):
                a_grad_momentum[l][i] = a[l].grad

        # print('time for GN after backward:')
        # print(time.time() - test_time_wall_clock)

        # a_grad_momentum = []
        # for l in range(model.numlayers):
            # a_grad_momentum.append(torch.stack([a[l].grad for a in list_a]))

        # for l in range(model.numlayers):
            # a_grad_momentum[l] = torch.stack([a[l].grad for a in list_a])

            # for i in range(m_L):
                # a_grad_momentum[l][i] = list_a[i][l].grad

        # test_start_time_h = time.process_time()

        h_momentum = [hi.data for hi in h]

        # test_time_1 = time.process_time() - test_start_time_h
        # print('test_time_1')
        # print(test_time_1)

        # print('a_grad_momentum[1].size()')
        # print(a_grad_momentum[1].size())

        for l in range(model.numlayers):
            a_grad_momentum[l] = a_grad_momentum[l].permute(1, 0, 2)

        # print('a_grad_momentum[1].size()')
        # print(a_grad_momentum[1].size())


        data_['a_grad'] = a_grad_momentum
        data_['h'] = h_momentum



        # print('h_momentum[1].size()')
        # print(h_momentum[1].size())

        #====

        # test_start_time_h = time.process_time()



        # test_time_2 = time.process_time() - test_start_time_h
        # print('test_time_2')
        # print(test_time_2)

        # print('test GN')
        # sys.exit()






        # data_['GN_cache'] = GN_cache

        # print('time for GN:')
        # print(time.time() - test_time_wall_clock)
        # print('\n')

    else:
        print('Error: unknown matrix name for ' + matrix_name)
        sys.exit()
            
    return data_

def get_Adam_direction(p, data_, params):
    beta_1 = params['Adam_beta_1']
    beta_2 = params['Adam_beta_2']
    epsilon = params['Adam_epsilon']

    i = params['i'] + 1

    # model_grad = data_['model_grad']
    

    data_['model_grad_Adam_momentum_1'] = get_plus(
    get_multiply_scalar(beta_1, data_['model_grad_Adam_momentum_1']),
    get_multiply_scalar(1 - beta_1, p))

    data_['model_grad_Adam_momentum_2'] = get_plus(
    get_multiply_scalar(beta_2, data_['model_grad_Adam_momentum_2']),
    get_multiply_scalar(1 - beta_2, 
                 get_square(p)))

    hat_m = get_multiply_scalar(1 / (1 - beta_1 ** i), data_['model_grad_Adam_momentum_1'])
    hat_v = get_multiply_scalar(1 / (1 - beta_2 ** i), data_['model_grad_Adam_momentum_2'])


    p_Adam = get_divide(
        hat_m,
        get_plus_scalar(epsilon,
                        get_sqrt(hat_v)))

    return p_Adam, data_

def get_name_loss(dataset):
    if dataset in ['MNIST',
                   'MNIST-no-regularization',
                   'MNIST-N1-1000',
                   'MNIST-one-layer',
                   'DownScaledMNIST-no-regularization',
                   'DownScaledMNIST-N1-1000-no-regularization',
                   'CIFAR',
                   'CIFAR-deep',
                   'CIFAR-10-vgg16',
                   'CIFAR-10-vgg11',
                   'CIFAR-10-NoAugmentation-vgg11',
                   'CIFAR-10-vgg16-NoAdaptiveAvgPoolNoDropout',
                   'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
                   'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
                   'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPool',
                   'CIFAR-10-onTheFly-N1-512-vgg16-NoAdaptiveAvgPoolNoDropout',
                   'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
                   'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-NoBias',
                   'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',
                   'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout',
                   'CIFAR-10-NoAugmentation-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
                   'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
                   'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',
                   'CIFAR-10-vgg16-GAP',
                   'CIFAR-10-AllCNNC',
                   'CIFAR-10-N1-128-AllCNNC',
                   'CIFAR-10-N1-512-AllCNNC',
                   'CIFAR-10-ConvPoolCNNC',
                   'CIFAR-100',
                   'CIFAR-100-NoAugmentation',
                   'CIFAR-100-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout',
                   'CIFAR-100-NoAugmentation-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
                   'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
                   'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine-no-regularization',
                   'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
                   'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-no-regularization',
                   'CIFAR-100-onTheFly-vgg16-NoLinear-BN-no-regularization',
                   'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
                   'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',
                   'CIFAR-100-onTheFly-AllCNNC',
                   'CIFAR-10-onTheFly-ResNet32-BNNoAffine',
                   'CIFAR-10-onTheFly-ResNet32-BN',
                   'CIFAR-10-onTheFly-ResNet32-BN-BNshortcut',
                   'CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly',
                   'CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly-NoBias',
                   'CIFAR-10-onTheFly-N1-128-ResNet32-BNNoAffine-PaddingShortcutDownsampleOnly-NoBias-no-regularization',
                   'CIFAR-10-onTheFly-N1-128-ResNet32-BN-BNshortcutDownsampleOnly-NoBias',
                   'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',
                   'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias-no-regularization',
                   'CIFAR-10-onTheFly-ResNet32-BNNoAffine-NoBias',
                   'CIFAR-100-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',
                   'CIFAR-100-onTheFly-ResNet34-BNNoAffine',
                   'CIFAR-100-onTheFly-ResNet34-BN',
                   'CIFAR-100-onTheFly-ResNet34-BN-BNshortcut',
                   'CIFAR-100-onTheFly-ResNet34-BN-BNshortcutDownsampleOnly',
                   'CIFAR-100-onTheFly-ResNet34-BN-BNshortcutDownsampleOnly-NoBias',
                   'CIFAR-100-onTheFly-N1-128-ResNet34-BN-BNshortcutDownsampleOnly-NoBias',
                   'CIFAR-100-onTheFly-N1-128-ResNet34-BN-PaddingShortcutDownsampleOnly-NoBias',
                   'Fashion-MNIST',
                   'Fashion-MNIST-N1-60',
                   'Fashion-MNIST-N1-60-no-regularization',
                   'Fashion-MNIST-N1-256-no-regularization',
                   'Fashion-MNIST-GAP-N1-60-no-regularization',
                   'STL-10-simple-CNN',
                   'Subsampled-ImageNet-simple-CNN',
                   'Subsampled-ImageNet-vgg16']:
        return 'multi-class classification'
        
    elif dataset == 'webspam':
        return 'binary classification'
    
    elif dataset in ['MNIST-autoencoder',
                     'MNIST-autoencoder-no-regularization',
                     'MNIST-autoencoder-N1-1000',
                     'MNIST-autoencoder-N1-1000-no-regularization',
                     'CURVES-autoencoder-no-regularization',
                     'CURVES-autoencoder',
                     'CURVES-autoencoder-Botev',
                     'CURVES-autoencoder-shallow',
                     'FACES-autoencoder',
                     'FACES-autoencoder-no-regularization']:
        return 'logistic-regression'
    
    elif dataset in ['MNIST-autoencoder-N1-1000-sum-loss',
                     'MNIST-autoencoder-N1-1000-sum-loss-no-regularization',
                     'MNIST-autoencoder-relu-N1-1000-sum-loss-no-regularization',
                     'MNIST-autoencoder-relu-N1-1000-sum-loss',
                     'MNIST-autoencoder-relu-N1-100-sum-loss',
                     'MNIST-autoencoder-relu-N1-500-sum-loss',
                     'MNIST-autoencoder-relu-N1-1-sum-loss',
                     'MNIST-autoencoder-reluAll-N1-1-sum-loss',
                     'FACES-autoencoder-sum-loss-no-regularization',
                     'FACES-autoencoder-relu-sum-loss-no-regularization',
                     'FACES-autoencoder-relu-sum-loss',
                     'FACES-autoencoder-sum-loss',
                     'CURVES-autoencoder-sum-loss-no-regularization',
                     'CURVES-autoencoder-sum-loss',
                     'CURVES-autoencoder-relu-sum-loss-no-regularization',
                     'CURVES-autoencoder-relu-sum-loss',
                     'CURVES-autoencoder-relu-N1-100-sum-loss',
                     'CURVES-autoencoder-relu-N1-500-sum-loss',
                     'CURVES-autoencoder-Botev-sum-loss-no-regularization',]:
        
        return 'logistic-regression-sum-loss'
    
    elif dataset in ['sythetic-linear-regression',
                     'sythetic-linear-regression-N1-1']:
        return 'linear-regression'
    elif dataset in ['FacesMartens-autoencoder-relu',
                     'FacesMartens-autoencoder-relu-no-regularization',
                     'FacesMartens-autoencoder-relu-N1-500',
                     'FacesMartens-autoencoder-relu-N1-100']:
        return 'linear-regression-half-MSE'
    else:
        print('Error: Problem not specified.')
        sys.exit()
        
def from_dataset_to_N1_N2(args):
    
    if not 'tau' in args:
        args['tau'] = 10**(-5) # https://arxiv.org/pdf/1503.05671.pdf
    
    if 'N1' in args or 'N2' in args:
        print('error: N1, N2 not automated')
        sys.exit()
    else:
        if args['dataset'] == 'MNIST-N1-1000':
            args['N1'] = 1000
            args['N2'] = 1000
        elif args['dataset'] == 'Fashion-MNIST':
            args['N1'] = 1000
            args['N2'] = 1000
        elif args['dataset'] == 'Fashion-MNIST-N1-60':
            args['N1'] = 60
            args['N2'] = 60
        elif args['dataset'] == 'Fashion-MNIST-N1-60-no-regularization':
            args['N1'] = 60
            args['N2'] = 60
            args['tau'] = 0
        elif args['dataset'] == 'Fashion-MNIST-N1-256-no-regularization':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 256
            args['N2'] = 256
            args['tau'] = 0
        elif args['dataset'] == 'Fashion-MNIST-GAP-N1-60-no-regularization':
            args['N1'] = 60
            args['N2'] = 60
            args['tau'] = 0
        elif args['dataset'] == 'webspam':
            args['N1'] = 1000
            args['N2'] = 1000
        elif args['dataset'] == 'MNIST':
            args['N1'] = 60
            args['N2'] = 60
        elif args['dataset'] == 'MNIST-no-regularization':
            args['N1'] = 60
            args['N2'] = 60
            args['tau'] = 0
        elif args['dataset'] == 'DownScaledMNIST-no-regularization':
            args['N1'] = 60
            args['N2'] = 60
            args['tau'] = 0
        elif args['dataset'] == 'DownScaledMNIST-N1-1000-no-regularization':
            args['N1'] = 1000
            args['N2'] = 1000
            args['tau'] = 0
        elif args['dataset'] == 'MNIST-autoencoder':
            args['N1'] = 60
            args['N2'] = 60
        elif args['dataset'] == 'MNIST-autoencoder-no-regularization':
            args['N1'] = 60
            args['N2'] = 60
            args['tau'] = 0
        elif args['dataset'] == 'MNIST-autoencoder-N1-1000':
            args['N1'] = 1000
            args['N2'] = 1000
        elif args['dataset'] == 'MNIST-autoencoder-N1-1000-sum-loss':
            args['N1'] = 1000
            args['N2'] = 1000
        elif args['dataset'] == 'MNIST-autoencoder-N1-1000-no-regularization':
            args['N1'] = 1000
            args['N2'] = 1000
            args['tau'] = 0
        elif args['dataset'] == 'MNIST-autoencoder-N1-1000-sum-loss-no-regularization':
            args['N1'] = 1000
            args['N2'] = 1000
            args['tau'] = 0
        elif args['dataset'] == 'MNIST-autoencoder-relu-N1-1000-sum-loss-no-regularization':
            args['N1'] = 1000
            args['N2'] = 1000
            args['tau'] = 0
        elif args['dataset'] == 'MNIST-autoencoder-relu-N1-1000-sum-loss':
            args['N1'] = 1000
            args['N2'] = 1000
        elif args['dataset'] == 'MNIST-autoencoder-relu-N1-100-sum-loss':
            args['N1'] = 100
            args['N2'] = 100
            args['tau'] = 10**(-5)
        elif args['dataset'] == 'MNIST-autoencoder-relu-N1-500-sum-loss':
            args['N1'] = 500
            args['N2'] = 500
        elif args['dataset'] == 'MNIST-autoencoder-relu-N1-1-sum-loss':
            args['N1'] = 1
            args['N2'] = 1
        elif args['dataset'] == 'MNIST-autoencoder-reluAll-N1-1-sum-loss':
            args['N1'] = 1
            args['N2'] = 1
        elif args['dataset'] == 'CURVES-autoencoder':
            args['N1'] = 1000
            args['N2'] = 1000
        elif args['dataset'] == 'CURVES-autoencoder-Botev':
            args['N1'] = 1000
            args['N2'] = 1000
        elif args['dataset'] == 'CURVES-autoencoder-Botev-sum-loss-no-regularization':
            args['N1'] = 1000
            args['N2'] = 1000
            args['tau'] = 0
        elif args['dataset'] == 'CURVES-autoencoder-no-regularization':
            args['N1'] = 1000
            args['N2'] = 1000
            args['tau'] = 0
        elif args['dataset'] == 'CURVES-autoencoder-sum-loss-no-regularization':
            args['N1'] = 1000
            args['N2'] = 1000
            args['tau'] = 0
        elif args['dataset'] == 'CURVES-autoencoder-relu-sum-loss-no-regularization':
            args['N1'] = 1000
            args['N2'] = 1000
            args['tau'] = 0
        elif args['dataset'] == 'CURVES-autoencoder-relu-sum-loss':
            args['N1'] = 1000
            args['N2'] = 1000
        elif args['dataset'] == 'CURVES-autoencoder-relu-N1-500-sum-loss':
            args['N1'] = 500
            args['N2'] = 500
        elif args['dataset'] == 'CURVES-autoencoder-relu-N1-100-sum-loss':
            args['N1'] = 100
            args['N2'] = 100
            args['tau'] = 10**(-5)
        elif args['dataset'] == 'CURVES-autoencoder-sum-loss':
            args['N1'] = 1000
            args['N2'] = 1000
        elif args['dataset'] == 'FACES-autoencoder':
            args['N1'] = 1000
            args['N2'] = 1000
        elif args['dataset'] == 'FACES-autoencoder-no-regularization':
            args['N1'] = 1000
            args['N2'] = 1000
            args['tau'] = 0
        elif args['dataset'] == 'FACES-autoencoder-sum-loss-no-regularization':
            args['N1'] = 1000
            args['N2'] = 1000
            args['tau'] = 0
        elif args['dataset'] == 'FACES-autoencoder-relu-sum-loss-no-regularization':
            args['N1'] = 1000
            args['N2'] = 1000
            args['tau'] = 0
        elif args['dataset'] == 'FACES-autoencoder-relu-sum-loss':
            args['N1'] = 1000
            args['N2'] = 1000
        elif args['dataset'] == 'FacesMartens-autoencoder-relu':
            args['N1'] = 1000
            args['N2'] = 1000
        elif args['dataset'] == 'FacesMartens-autoencoder-relu-no-regularization':
            args['N1'] = 1000
            args['N2'] = 1000
            args['tau'] = 0
        elif args['dataset'] == 'FacesMartens-autoencoder-relu-N1-500':
            args['N1'] = 500
            args['N2'] = 500
        elif args['dataset'] == 'FacesMartens-autoencoder-relu-N1-100':
            args['N1'] = 100
            args['N2'] = 100
            args['tau'] = 10**(-5)
        elif args['dataset'] == 'FACES-autoencoder-sum-loss':
            args['N1'] = 1000
            args['N2'] = 1000
        elif args['dataset'] == 'sythetic-linear-regression':
            args['N1'] = 900
            args['N2'] = 900
        elif args['dataset'] == 'sythetic-linear-regression-N1-1':
            args['N1'] = 1
            args['N2'] = 1
        elif args['dataset'] == 'MNIST-one-layer':
            args['N1'] = 60000
            args['N2'] = 60000
        elif args['dataset'] == 'UCI-HAR':
            args['N1'] = 32
            args['N2'] = 32
        elif args['dataset'] == 'CIFAR-100':
            
            print('error: data is augmented but not on the fly')
            sys.exit()
            
            args['N1'] = 1000
            args['N2'] = 1000
        elif args['dataset'] == 'CIFAR-100-NoAugmentation':
            
            args['N1'] = 1000
            args['N2'] = 1000
        elif args['dataset'] == 'CIFAR-100-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout':
            
            args['N1'] = 1000
            args['N2'] = 1000
            
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
            
        elif args['dataset'] == 'CIFAR-100-NoAugmentation-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout':
            
            args['N1'] = 256
            args['N2'] = 256
            
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
            
        elif args['dataset'] == 'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout':
            
            args['N1'] = 128
            args['N2'] = 128
            
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
            
        elif args['dataset'] == 'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine-no-regularization':
            
            args['N1'] = 128
            args['N2'] = 128
            
            args['tau'] = 0
            
        elif args['dataset'] == 'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN':
            
            args['N1'] = 128
            args['N2'] = 128
            
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
            
        elif args['dataset'] == 'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-no-regularization':
            
            args['N1'] = 128
            args['N2'] = 128
            
            args['tau'] = 0
            
        elif args['dataset'] == 'CIFAR-100-onTheFly-vgg16-NoLinear-BN-no-regularization':
            
            args['N1'] = 128
            args['N2'] = 128
            
            args['tau'] = 0
            
        elif args['dataset'] == 'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout':
            
            args['N1'] = 256
            args['N2'] = 256
            
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
            
        elif args['dataset'] == 'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine':
            
            args['N1'] = 256
            args['N2'] = 256
            
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
            
        elif args['dataset'] == 'CIFAR-100-onTheFly-AllCNNC':
            
            args['N1'] = 256
            args['N2'] = 256
            
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
            
        elif args['dataset'] == 'CIFAR-deep':
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 0.0005
        elif args['dataset'] == 'CIFAR-10-vgg11':
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 0.0005
        elif args['dataset'] == 'CIFAR-10-NoAugmentation-vgg11':
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 0.0005
        elif args['dataset'] == 'CIFAR-10-vgg11-test':
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 0.0005
        elif args['dataset'] == 'CIFAR-10-vgg16':
            # https://arxiv.org/pdf/1910.05446.pdf
            
            print('adaptive avg pool is not neede for CIFAR10 + vgg16')
            sys.exit()
            
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
        elif args['dataset'] == 'CIFAR-10-vgg16-NoAdaptiveAvgPoolNoDropout':
            
            print('error: should use CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout')
            sys.exit()
            
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
            
        elif args['dataset'] == 'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout':
            
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
            
        elif args['dataset'] == 'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 256
            args['N2'] = 256
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
            
        elif args['dataset'] == 'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPool':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 256
            args['N2'] = 256
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
            
        elif args['dataset'] == 'CIFAR-10-onTheFly-N1-512-vgg16-NoAdaptiveAvgPoolNoDropout':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 512
            args['N2'] = 512
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
            
        elif args['dataset'] == 'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN':
            
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
            
        elif args['dataset'] == 'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-NoBias':
            
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
            
        elif args['dataset'] == 'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine':
            
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
            
        elif args['dataset'] == 'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
        elif args['dataset'] == 'CIFAR-10-NoAugmentation-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
        elif args['dataset'] == 'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout-BN':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
        elif args['dataset'] == 'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
        elif args['dataset'] == 'CIFAR-10-vgg16-GAP':
            
            print('GAP is not needed for CIFAR10 + vgg16')
            sys.exit()
            
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 0.0005
            
        elif args['dataset'] == 'CIFAR-10-onTheFly-ResNet32-BNNoAffine':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 256
            args['N2'] = 256
            
            # sec 4.2 of https://arxiv.org/pdf/1512.03385.pdf
            args['tau'] = 0.0001 # inherit from VGG
            
        elif args['dataset'] == 'CIFAR-10-onTheFly-ResNet32-BN':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 256
            args['N2'] = 256
            
            # sec 4.2 of https://arxiv.org/pdf/1512.03385.pdf
            args['tau'] = 0.0001 # inherit from VGG
            
        elif args['dataset'] == 'CIFAR-10-onTheFly-ResNet32-BN-BNshortcut':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 256
            args['N2'] = 256
            
            # sec 4.2 of https://arxiv.org/pdf/1512.03385.pdf
            args['tau'] = 0.0001 # inherit from VGG
            
        elif args['dataset'] == 'CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 256
            args['N2'] = 256
            
            # sec 4.2 of https://arxiv.org/pdf/1512.03385.pdf
            args['tau'] = 0.0001 # inherit from VGG
            
        elif args['dataset'] == 'CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly-NoBias':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 256
            args['N2'] = 256
            
            # sec 4.2 of https://arxiv.org/pdf/1512.03385.pdf
            args['tau'] = 0.0001 # inherit from VGG
            
        elif args['dataset'] == 'CIFAR-10-onTheFly-N1-128-ResNet32-BNNoAffine-PaddingShortcutDownsampleOnly-NoBias-no-regularization':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 128
            args['N2'] = 128
            
            args['tau'] = 0
            
        elif args['dataset'] == 'CIFAR-10-onTheFly-N1-128-ResNet32-BN-BNshortcutDownsampleOnly-NoBias':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 128
            args['N2'] = 128
            
            # sec 4.2 of https://arxiv.org/pdf/1512.03385.pdf
            args['tau'] = 0.0001 # inherit from VGG
            
        elif args['dataset'] == 'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 128
            args['N2'] = 128
            
            # sec 4.2 of https://arxiv.org/pdf/1512.03385.pdf
            args['tau'] = 0.0001 # inherit from VGG
            
        elif args['dataset'] == 'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias-no-regularization':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 128
            args['N2'] = 128
            
            args['tau'] = 0
            
        elif args['dataset'] == 'CIFAR-100-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 128
            args['N2'] = 128
            
            # sec 4.2 of https://arxiv.org/pdf/1512.03385.pdf
            args['tau'] = 0.0001 # inherit from VGG
            
        elif args['dataset'] == 'CIFAR-10-onTheFly-ResNet32-BNNoAffine-NoBias':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 256
            args['N2'] = 256
            
            # sec 4.2 of https://arxiv.org/pdf/1512.03385.pdf
            args['tau'] = 0.0001 # inherit from VGG
            
        elif args['dataset'] == 'CIFAR-100-onTheFly-ResNet34-BNNoAffine':
            # https://zhenye-na.github.io/2018/10/07/pytorch-resnet-cifar100.html
            
            args['N1'] = 256
            args['N2'] = 256
            args['tau'] = 1e-5
            
        elif args['dataset'] == 'CIFAR-100-onTheFly-ResNet34-BN':
            # https://zhenye-na.github.io/2018/10/07/pytorch-resnet-cifar100.html
            
            args['N1'] = 256
            args['N2'] = 256
            args['tau'] = 1e-5
            
        elif args['dataset'] == 'CIFAR-100-onTheFly-ResNet34-BN-BNshortcut':
            # https://zhenye-na.github.io/2018/10/07/pytorch-resnet-cifar100.html
            
            args['N1'] = 256
            args['N2'] = 256
            args['tau'] = 1e-5
            
        elif args['dataset'] == 'CIFAR-100-onTheFly-ResNet34-BN-BNshortcutDownsampleOnly':
            # https://zhenye-na.github.io/2018/10/07/pytorch-resnet-cifar100.html
            
            args['N1'] = 256
            args['N2'] = 256
            args['tau'] = 1e-5
            
        elif args['dataset'] == 'CIFAR-100-onTheFly-ResNet34-BN-BNshortcutDownsampleOnly-NoBias':
            # https://zhenye-na.github.io/2018/10/07/pytorch-resnet-cifar100.html
            
            args['N1'] = 256
            args['N2'] = 256
            args['tau'] = 1e-5
            
        elif args['dataset'] == 'CIFAR-100-onTheFly-N1-128-ResNet34-BN-BNshortcutDownsampleOnly-NoBias':
            # https://github.com/weiaicunzai/pytorch-cifar100/blob/master/train.py
            # https://github.com/bearpaw/pytorch-classification/blob/master/cifar.py
            
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 5e-4
            
        elif args['dataset'] == 'CIFAR-100-onTheFly-N1-128-ResNet34-BN-PaddingShortcutDownsampleOnly-NoBias':
            # https://github.com/weiaicunzai/pytorch-cifar100/blob/master/train.py
            # https://github.com/bearpaw/pytorch-classification/blob/master/cifar.py
            
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 5e-4
            
        elif args['dataset'] == 'CIFAR-10-AllCNNC':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 256
            args['N2'] = 256
            args['tau'] = 0.0005
        elif args['dataset'] == 'CIFAR-10-N1-128-AllCNNC':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 0.0005
        elif args['dataset'] == 'CIFAR-10-N1-512-AllCNNC':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 512
            args['N2'] = 512
            args['tau'] = 0.0005
        elif args['dataset'] == 'CIFAR-10-ConvPoolCNNC':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 256
            args['N2'] = 256
            args['tau'] = 0.0005
        elif args['dataset'] == 'STL-10-simple-CNN':
            args['N1'] = 1000
            args['N2'] = 1000
        elif args['dataset'] == 'Subsampled-ImageNet-simple-CNN':
            args['N1'] = 100
            args['N2'] = 100
        elif args['dataset'] == 'Subsampled-ImageNet-vgg16':
            args['N1'] = 10
            args['N2'] = 10
        else:
            print('error: unknown dataset for ' + args['dataset'])
            sys.exit()
    return args

def tune_lr(args):
    
    assert 'max_epoch/time' in args
    assert 'record_epoch' in args
    assert 'if_test_mode' in args
    
    assert 'if_grafting' in args
    assert 'weight_decay' in args

    

    args['name_loss'] = get_name_loss(args['dataset'])
    args = from_dataset_to_N1_N2(args)
    args = add_matrix_name_to_args(args)
    args = add_some_if_record_to_args(args)
    
#     assert 'max_epoch/time' in args
#     assert 'record_epoch' in args
#     assert 'if_test_mode' in args
    
    args['momentum_gradient_rho'] = 0.9
    args['lambda_'] = 1
    
    if args['if_auto_tune_lr']:
        
        assert len(args['list_lr']) == 2
        
        for learning_rate in args['list_lr']:
            args['alpha'] = learning_rate
            name_result, data_, params_saved = train(args)
            
            print_gpu_usage({'device': 'cuda:0'})
            
            data_ = None
            torch.cuda.empty_cache()
            
            print_gpu_usage({'device': 'cuda:0'})
            
            
            
        list_lr_tried = args['list_lr']
        
        while 1:
            fake_args = {}
            fake_args['home_path'] = args['home_path']
            fake_args['algorithm_dict'] = {}
            fake_args['algorithm_dict']['name'] = args['algorithm']
            fake_args['algorithm_dict']['params'] = params_saved
            fake_args['if_gpu'] = args['if_gpu']
            fake_args['dataset'] = args['dataset']
            fake_args['N1'] = args['N1']
            fake_args['N2'] = args['N2']
            fake_args['name_loss'] = args['name_loss']

            fake_args['tuning_criterion'] = args['tuning_criterion']
            fake_args['list_lr'] = list_lr_tried

            best_lr, _, best_name_result_pkl = get_best_params(fake_args, False)

            print('best_lr')
            print(best_lr)
            

            
            learning_rate = get_next_lr(list_lr_tried, best_lr)
            
            if learning_rate < 0:
                break
            else:
                args['alpha'] = learning_rate
                name_result, data_, params_saved = train(args)
                
                data_ = None
                torch.cuda.empty_cache()
                
                
                
                if learning_rate < min(list_lr_tried):
                    list_lr_tried = [learning_rate] + list_lr_tried
                elif learning_rate > max(list_lr_tried):
                    list_lr_tried = list_lr_tried + [learning_rate]
                else:
                    print('there is an error')
                    sys.exit()
                    
                    
        print('list_lr_tried, best_lr')
        print(list_lr_tried, best_lr)
        
#         sys.exit()
    else:


        for learning_rate in args['list_lr']:
            args['alpha'] = learning_rate
            name_result, data_, _ = train(args)
            
            data_ = None
            torch.cuda.empty_cache()


    return data_

    
    

def get_if_stop(args, i, iter_per_epoch, timesCPU):
    if args['if_max_epoch']:
        # stop by epoch
        
#         print('i')
#         print(i)
        
#         print('int(args[max_epoch/time] * iter_per_epoch)')
#         print(int(args['max_epoch/time'] * iter_per_epoch))
        
#         sys.exit()
        
        if i < int(args['max_epoch/time'] * iter_per_epoch):
            return False
        else:
            return True
    else:
        # stop by time
        
#         print('timesCPU[-1]')
#         print(timesCPU[-1])
        
#         print('args[max_epoch/time]')
#         print(args['max_epoch/time'])
        
        if timesCPU[-1] < args['max_epoch/time']:
            return False
        else:
            return True

def get_erf(p):
    sign_ = []
    for l in range(len(p)):
        sign_l = {}
        for key in p[l]:
            sign_l[key] = torch.erf(p[l][key])
        sign_.append(sign_l)
    return sign_

def get_erf_approx(p):
    sign_ = []
    for l in range(len(p)):
        sign_l = {}
        for key in p[l]:
            sign_l[key] = 1 / (1 + p[l][key]**2)
        sign_.append(sign_l)
    return sign_

def get_reciprocal(p):
    sign_ = []
    for l in range(len(p)):
        sign_l = {}
        for key in p[l]:
            sign_l[key] = 1 / p[l][key]
        sign_.append(sign_l)
    return sign_

def get_sign(p):
    sign_ = []
    for l in range(len(p)):
        sign_l = {}
        for key in p[l]:
            sign_l[key] = np.sign(p[l][key])
        sign_.append(sign_l)
    return sign_


def get_sign_torch(p):
    sign_ = []
    for l in range(len(p)):
        sign_l = {}
        for key in p[l]:
            sign_l[key] = torch.sign(p[l][key])
        sign_.append(sign_l)
    return sign_

'''
def get_zero(params):
    layers_params = params['layers_params']
    
    delta = []
    for l in range(len(layers_params)):
        delta_l = {}
        if layers_params[l]['name'] == 'fully-connected':
            delta_l['W'] = np.zeros((layers_params[l]['output_size'], layers_params[l]['input_size']))
            delta_l['b'] = np.zeros(layers_params[l]['output_size'])
        elif layers_params[l]['name'] == 'conv':
            delta_l['W'] = np.zeros((layers_params[l]['conv_out_channels'],
                                     layers_params[l]['conv_in_channels'],
                                     layers_params[l]['conv_kernel_size'],
                                     layers_params[l]['conv_kernel_size']))
            delta_l['b'] = np.zeros(layers_params[l]['conv_out_channels'])
        elif layers_params[l]['name'] == '1d-conv':
            delta_l['W'] = np.zeros((layers_params[l]['conv_out_channels'],
                                     layers_params[l]['conv_in_channels'],
                                     layers_params[l]['conv_kernel_size']))
            delta_l['b'] = np.zeros(layers_params[l]['conv_out_channels'])
        else:
            print('Error: layers unsupported when get zero for ' + layers_params[l]['name'])
            sys.exit()
        delta.append(delta_l)
        
    return delta
'''

def get_zero_torch(params):
    layers_params = params['layers_params']
    device = params['device']
    
    delta = []
    for l in range(len(layers_params)):
        delta_l = {}
        if layers_params[l]['name'] == 'fully-connected':
            delta_l['W'] = torch.zeros(layers_params[l]['output_size'], layers_params[l]['input_size'], device=device)
            delta_l['b'] = torch.zeros(layers_params[l]['output_size'], device=device)
        elif layers_params[l]['name'] in ['conv',
                                          'conv-no-activation',
                                          'conv-no-bias-no-activation']:
            delta_l['W'] = torch.zeros(layers_params[l]['conv_out_channels'],
                                     layers_params[l]['conv_in_channels'],
                                     layers_params[l]['conv_kernel_size'],
                                     layers_params[l]['conv_kernel_size'], device=device)
            if layers_params[l]['name'] in ['conv',
                                            'conv-no-activation',]:
                delta_l['b'] = torch.zeros(layers_params[l]['conv_out_channels'], device=device)
        elif layers_params[l]['name'] == '1d-conv':
            delta_l['W'] = torch.zeros(layers_params[l]['conv_out_channels'],
                                     layers_params[l]['conv_in_channels'],
                                     layers_params[l]['conv_kernel_size'],
                                       device=device)
            delta_l['b'] = torch.zeros(layers_params[l]['conv_out_channels'], device=device)
        elif layers_params[l]['name'] == 'BN':
            
#             print('layers_params[l][num_features]')
#             print(layers_params[l]['num_features'])
            
#             sys.exit()
            
            delta_l['W'] = torch.zeros(layers_params[l]['num_features'], device=device)
            
            delta_l['b'] = torch.zeros(layers_params[l]['num_features'], device=device)
            
            
        else:
            print('Error: layers unsupported when get zero for ' + layers_params[l]['name'])
            sys.exit()
        delta.append(delta_l)
        
    return delta


def get_full_grad(model, x, t, params):
    N1 = params['N1']
    reduction = 'mean'
    i = 0
    while (i+1) * N1 <= len(x):
        loss = get_regularized_loss_from_x(model, x[i*N1: (i+1)*N1], t[i*N1: (i+1)*N1], reduction)
        model.zero_grad()
        loss.backward()
        grad_i = get_model_grad(model, params)
        if i == 0:
            full_grad = grad_i
        else:
            full_grad = get_plus_torch(full_grad, grad_i)
        i += 1
    full_grad = get_multiply_scalar(1./i, full_grad)

    
    
    return full_grad

def get_regularized_loss_from_x_no_grad(model, x, t, reduction, tau):
    with torch.no_grad():
        z, _, _ = model.forward(x)
#     return get_loss_from_z(model, z, t, reduction)
    return get_regularized_loss_from_z(model, z, t, reduction, tau)



def get_acc_whole_dataset(model, params, x, np_t):
    N1 = params['N1']
    N1 = np.minimum(N1, len(x))
    
    i = 0
    list_acc = []
    model.eval()
    
    while i + N1 <= len(x):
        with torch.no_grad():
            list_acc.append(get_acc_from_x(model, params, x[i: i+N1], np_t[i: i+N1]))
            # z, _, _ = model.forward(x[i: i+N1])

        i += N1
    model.train()
    return sum(list_acc) / len(list_acc)

def get_regularized_loss_from_x_whole_dataset(model, x, t, reduction, params):
    N1 = params['N1']
    device = params['device']
    
    
    i = 0
    list_loss = []
    model.eval()
    
    
    while i + N1 <= len(x):
        with torch.no_grad():
            z, _, _ = model.forward(torch.from_numpy(x[i: i+N1]).to(device))
        
        list_loss.append(
            get_regularized_loss_from_z(model, z, torch.from_numpy(t[i: i+N1]).to(device), reduction).item())
        

        
        i += N1
        

        
        
    model.train()
    return sum(list_loss) / len(list_loss)

def get_regularized_loss_and_acc_from_x_whole_dataset_with_generator(model, generator, reduction, params):
    N1 = params['N1']
#     N1 = np.minimum(N1, len(x))
    
    
    device = params['device']
    
    list_loss = []
    list_unregularized_loss = []
    list_acc = []
    
    model.eval()
    
    for (X_mb, t_mb) in generator:
        
#         print('X_mb.size()')
#         print(X_mb.size())
        
        X_mb, t_mb = X_mb.to(device), t_mb.to(device)
        
        if len(X_mb) != N1:
            break
        
#         with torch.no_grad():
        z, _, _ = model.forward(X_mb)
            
        
        
        loss_i, unregularized_loss_i =\
        get_regularized_loss_from_z(model, z, t_mb, reduction, params['tau'])
        
        list_loss.append(loss_i.item())
        
        list_unregularized_loss.append(unregularized_loss_i.item())
        
        list_acc.append(
            get_acc_from_z(model, params, z, t_mb))
        
    model.train()
    
    return sum(list_loss) / len(list_loss), sum(list_unregularized_loss) / len(list_unregularized_loss), sum(list_acc) / len(list_acc)

def get_regularized_loss_and_acc_from_x_whole_dataset(model, x, t, reduction, params):
    N1 = params['N1']
    N1 = np.minimum(N1, len(x))
    
    
    i = 0
    device = params['device']
    
    list_loss = []
    list_unregularized_loss = []
    list_acc = []
    
    model.eval()
    
    while i + N1 <= len(x):
        
        X_mb = torch.from_numpy(x[i: i+N1]).to(device)
        t_mb = torch.from_numpy(t[i: i+N1]).to(device)
        
#         with torch.no_grad():
        z, _, _ = model.forward(X_mb)
            
        
        
        loss_i, unregularized_loss_i =\
        get_regularized_loss_from_z(model, z, t_mb, reduction, params['tau'])
        
        list_loss.append(loss_i.item())
        
        list_unregularized_loss.append(unregularized_loss_i.item())
        
        list_acc.append(
            get_acc_from_z(model, params, z, t_mb))
        
        i += N1
    model.train()
    
    return sum(list_loss) / len(list_loss), sum(list_unregularized_loss) / len(list_unregularized_loss), sum(list_acc) / len(list_acc)

def get_loss_from_x(model, x, t, reduction):
    # with torch.no_grad():
    z, _, _ = model.forward(x)
    return get_loss_from_z(model, z, t, reduction)

def get_acc_from_x(model, params, x, np_t):
    
    z, _ , _= model.forward(x)

    return get_acc_from_z(model, params, z, np_t)

def get_regularized_loss_from_z(model, z, t, reduction, tau):
    
#     loss = get_loss_from_z(model, z, t, reduction)
    unregularized_loss = get_loss_from_z(model, z, t, reduction)
    
#     loss += 0.5 * tau *\
#     get_dot_product_torch(model.layers_weight, model.layers_weight)
    loss = unregularized_loss + 0.5 * tau *\
    get_dot_product_torch(model.layers_weight, model.layers_weight)
    
    return loss, unregularized_loss


def get_loss_from_z(model, z, t, reduction):
    if model.name_loss == 'multi-class classification':
        # since this is multi-calss cross entropy loss,
        # there is no ambiguity between "average" and "sum"
        
        # common bug: size of z does not match max of t
        

        
#         print('z.size()')
#         print(z.size())
        
#         print('t.size()')
#         print(t.size())

        loss = F.cross_entropy(z, t, reduction = reduction)


    elif model.name_loss == 'binary classification':
        # since this is binary-calss cross entropy loss,
        # there is no ambiguity between "average" and "sum"
        

        loss = torch.nn.BCEWithLogitsLoss(reduction = reduction)(z, t.float().unsqueeze_(1))


        if reduction == 'none':
            loss = loss.squeeze(1)
            
#         sys.exit()
        

    elif model.name_loss == 'logistic-regression':
        # use of cross-entropy endorsed by Hinton and Salakhutdinov 2006 and 
        # Hessian-free code
        
        # if reduction == 'mean', the following gives the sum of loss / #data / #feature
        # i.e. for a single data point, the loss on all pixels are averaged      
        

        if reduction == 'none':
            loss = torch.nn.BCEWithLogitsLoss(reduction = reduction)(z, t.float())
            loss = torch.sum(loss, dim=1)
        elif reduction == 'mean':
            loss = torch.nn.BCEWithLogitsLoss(reduction = 'sum')(z, t.float())
            
            loss = loss / z.size(0) / z.size(1)
#             loss = loss / z.size(0)
            
        elif reduction == 'sum':
            loss = torch.nn.BCEWithLogitsLoss(reduction = reduction)(z, t.float())
            
    elif model.name_loss == 'logistic-regression-sum-loss':
        # use of cross-entropy endorsed by Hinton and Salakhutdinov 2006 and 
        # Hessian-free code
        
        # if reduction == 'mean', the following gives the sum of loss / #data
        # i.e. for a single data point, the loss on all pixels are summed      
        

        if reduction == 'none':
            loss = torch.nn.BCEWithLogitsLoss(reduction = reduction)(z, t.float())
            loss = torch.sum(loss, dim=1)
        elif reduction == 'mean':
            loss = torch.nn.BCEWithLogitsLoss(reduction = 'sum')(z, t.float())
            
#             loss = loss / z.size(0) / z.size(1)
            loss = loss / z.size(0)
            
        elif reduction == 'sum':
            loss = torch.nn.BCEWithLogitsLoss(reduction = reduction)(z, t.float())

    elif model.name_loss == 'linear-regression-half-MSE':
        
        if reduction == 'mean':
            loss = torch.nn.MSELoss(reduction = 'sum')(z, t) / 2
            
            
            # only average by mini-batch, not by #feature
            loss = loss / z.size(0)
            
            # averaged by #mini-bath * #feature
#             loss = loss / z.size(0) / z.size(1)
            
            # averaged by #mini-bath
#             loss = loss / z.size(0) / z.size(1)
        elif reduction == 'none':
            loss = torch.nn.MSELoss(reduction = 'none')(z, t) / 2
            loss = torch.sum(loss, dim=1)
        else:
            print('reduction')
            print(reduction)
            print('error: unknown reduction')
            sys.exit()
            
            sys.exit()
            
    elif model.name_loss == 'linear-regression':
        
        if reduction == 'mean':
            loss = torch.nn.MSELoss(reduction = 'sum')(z, t)
            
            
            # only average by mini-batch, not by #feature
            loss = loss / z.size(0)
            
            # averaged by #mini-bath * #feature
#             loss = loss / z.size(0) / z.size(1)
            
            # averaged by #mini-bath
#             loss = loss / z.size(0) / z.size(1)
        elif reduction == 'none':
            loss = torch.nn.MSELoss(reduction = 'none')(z, t)
            loss = torch.sum(loss, dim=1)
        else:
            print('reduction')
            print(reduction)
            print('error: unknown reduction')
            sys.exit()
            
            sys.exit()
    
    else:
        print('Error: loss function not specified.')
        sys.exit()
    
    return loss



def get_acc_from_z(model, params, z, torch_t):
    
    if model.name_loss == 'multi-class classification':
        
#         np_t = torch_t.cpu().data.numpy()
        
        y = z.argmax(dim=1)
        
#         acc = np.mean(y.cpu().numpy() == np_t)
        
#         print('acc')
#         print(acc)
        
#         print('y == torch_t')
#         print(y == torch_t)
        
#         print('(y == torch_t).float()')
#         print((y == torch_t).float())
        
#         print('torch.mean((y == torch_t).float())')
#         print(torch.mean((y == torch_t).float()))
        
        acc = torch.mean((y == torch_t).float())
        
#         print('np_t should be tensor')
#         sys.exit()
    
    elif model.name_loss == 'binary classification':
        
        print('np_t should be tensor')
        sys.exit()
        
        z_1 = torch.sigmoid(z)
        
        y = (z_1 > 0.5)
        
        y = y[:, 0]
        
        acc = np.mean(y.cpu().data.numpy() == np_t)
    elif model.name_loss in ['logistic-regression',
                             'logistic-regression-sum-loss']:
        # the MSE is the sum of mse / #data / #feature
        
#         print('np_t should be tensor logistic-regression')
#         sys.exit()

        z_sigmoid = torch.sigmoid(z)
    
#         print('z[0][0].item()')
#         print(z[0][0].item())

        criterion = nn.MSELoss(reduction = 'mean')
        acc = criterion(z_sigmoid, torch_t)
        
#         print('z_sigmoid.size()')
#         print(z_sigmoid.size())
#         print('torch_t.size()')
#         print(torch_t.size())
        
#         print('z_sigmoid[0][0].item()')
#         print(z_sigmoid[0][0].item())
        
        
#         sys.exit()
        
#         print('z_sigmoid.size()')
#         print(z_sigmoid.size())
#         print('torch.from_numpy(np_t).to(params[device]).size()')
#         print(torch.from_numpy(np_t).to(params['device']).size())
        
#         test_torch_t = torch.from_numpy(np_t).to(params['device'])
        
#         print('torch.sum((z_sigmoid - test_torch_t)**2) / 256 /625')
#         print(torch.sum((z_sigmoid - test_torch_t)**2) / 256 /625)
        
#         print('acc in logistic-regression')
#         print(acc.item())
#         sys.exit()
        
        
    elif model.name_loss in ['linear-regression',
                             'linear-regression-half-MSE']:
        
#         criterion = nn.MSELoss(reduction = 'mean')

#         criterion = nn.MSELoss(reduction = 'sum')
#         acc = criterion(z, torch_t)
        
#         acc = nn.MSELoss(reduction = 'sum')(z, torch_t)
#         acc = acc / z.size(0)
        
        acc = nn.MSELoss(reduction = 'mean')(z, torch_t)
#         acc = acc / z.size(0)
        
        
#         acc = acc.item()
    else:
        print('Error: unkwoen name_loss')
        sys.exit()
    acc = acc.item()
    
#     print('acc')
#     print(acc)
    
    return acc
    
    


    
    
def compute_sum_J_transpose_V_backp(v, data_, params):
    # use backpropagation
    algorithm = params['algorithm']
    N2 = params['N2']
    numlayers = params['numlayers']
    
    model = data_['model']
    X_mb_N2 = data_['X_mb_N2']
    
    
    
    z, _, _ = model.forward(X_mb_N2)
    
    # if algorithm in ['SMW-Fisher-different-minibatch',
                    #  'SMW-Fisher-batch-grad-momentum-exponential-decay',
                    #  'ekfac-EF',
                    #  'kfac',
                    #  'SMW-Fisher',
                    #  'SMW-Fisher-momentum',
                    #  'SMW-Fisher-D_t-momentum',
                    #  'SMW-Fisher-momentum-D_t-momentum',
                    #  'GI-Fisher',
                    #  'SMW-Fisher-BD',
                    #  'RMSprop-individual-grad-no-sqrt-LM',
                    #  'SMW-Fisher-batch-grad',
                    #  'SMW-Fisher-batch-grad-momentum',
                    #  'matrix-normal-LM-momentum-grad']:
    if params['matrix_name'] == 'Fisher':
                         
                         
        t_mb_N2 = data_['t_mb_pred_N2'] # note that t_mb will be correspond to either EF or Fisher
    
#     model_new = Model()
    
#     model_new.W = model.W 


#     model_new = Model_2()
    
#     print('model_new.W[1]): ', model_new.W[1])
#     print('model.W[1]): ', model.W[1])
    
#     model_new.load_state_dict(model.state_dict())

#     for l in range(numlayers):
#         model_new.W[l].data = model.W[l].data
#     print('test')
    
#     weighted_loss = torch.dot(loss, v)
        
        
#         loss = F.cross_entropy(torch.squeeze(z), t_mb[N2_index], reduction = 'none')
        reduction = 'none'
        loss = get_loss_from_z(model, z, t_mb_N2, reduction)
        
#         print('error: should change to device')
#         sys.exit()
        
#         loss = torch.dot(loss, v.to(params['device']))
        loss = torch.dot(loss, v)
        
    elif algorithm == 'SMW-GN':
        
        m_L = params['m_L']
        
        v = v.view(N2, m_L)
        
#         print('print(z.dtype): ', z.dtype)
        
#         print('print(v.dtype): ', v.dtype)
        
        loss = torch.sum(z * v.data)
        
    else:
        print('Error! 1500')
        sys.exit()


    model.zero_grad()

    # print('loss')
    # print(loss)

    loss.backward()
    
#     print('test 10:28')
    
#     print('model_1.W[1].size():', model_1.W[1].size())

    delta = get_model_grad(model, params)

    

    
    
    


    

    model.zero_grad()
    
    
    return delta

    
def get_D_t(data_, params):
    algorithm = params['algorithm'] 
    N2 = params['N2']
    numlayers = params['numlayers']
    
    if algorithm == 'SMW-Fisher-different-minibatch' or\
    algorithm == 'SMW-Fisher':
        a_grad_momentum = data_['a_grad_N2']
        h_momentum = data_['h_N2']
    
        lambda_ = params['lambda_']
        
        print('error: should change to device')
        sys.exit()

        # compute D_t 
        D_t = lambda_ * torch.eye(N2).to(params['device'])

        for l in range(numlayers):
            # @ == torch.mm in this case, speed also similar
            # D_t += 1 / N2 * (a_grad_momentum[l] @ a_grad_momentum[l].t()) * (h_momentum[l] @ h_momentum[l].t() + 1)
            D_t += 1 / N2 * torch.mm(a_grad_momentum[l], a_grad_momentum[l].t()) *\
            (torch.mm(h_momentum[l], h_momentum[l].t()) + 1)

        # D_t = D_t.cpu().data.numpy()
        torch_D_t = D_t
    elif algorithm == 'SMW-Fisher-momentum' or\
    algorithm == 'SMW-Fisher-D_t-momentum' or\
    algorithm == 'GI-Fisher':
        a_grad_momentum = data_['a_grad_momentum']
        h_momentum = data_['h_momentum']
    
        lambda_ = params['lambda_']
        
        

        # compute D_t 
        D_t = lambda_ * torch.eye(N2, device=params['device'])

        for l in range(numlayers):
            # @ == torch.mm in this case, speed also similar
            # D_t += 1 / N2 * (a_grad_momentum[l] @ a_grad_momentum[l].t()) * (h_momentum[l] @ h_momentum[l].t() + 1)
            D_t += 1 / N2 * torch.mm(a_grad_momentum[l], a_grad_momentum[l].t()) *\
            (torch.mm(h_momentum[l], h_momentum[l].t()) + 1)

        # D_t = D_t.cpu().data.numpy()
        torch_D_t = D_t


    elif algorithm == 'SMW-Fisher-momentum-D_t-momentum':

        a_grad_momentum = data_['a_grad_for_D_t']
        h_momentum = data_['h_for_D_t']
    
        lambda_ = params['lambda_']
        
        

        # compute D_t 
        D_t = lambda_ * torch.eye(N2)
    
        for l in range(numlayers):
        
            D_t += 1 / N2 * torch.mm(a_grad_momentum[l], a_grad_momentum[l].t()) *\
            (torch.mm(h_momentum[l], h_momentum[l].t()) + 1)
        
        D_t = D_t.data.numpy()
    elif algorithm == 'SMW-GN':
        # a_grad[l]: N2, m_L, m_l
        
    
        GN_cache = data_['GN_cache']
        h = GN_cache['h']
        a_grad = GN_cache['a_grad']
        
        m_L = params['m_L']
        lambda_ = params['lambda_']
        
        
        # D_t = np.zeros((m_L * N2, m_L * N2))
        torch_D_t = torch.zeros(m_L * N2, m_L * N2, device=params['device'])
        
        
        
        model = data_['model']
        
#         start_time = time.time()
        
        
        for l in range(numlayers):

            # np_a_grad_l = a_grad[l].cpu().data.numpy()
            # h_l = h[l].cpu().data.numpy()

            torch_a_grad_1 = a_grad[l]
            torch_h_l = h[l]
            
            # h[l]: N2 * m[l]
            
            # a_grad[l]
            
    
            

            
            # permuted_a_grad_l =\
            # np.reshape(
                # np.swapaxes(np_a_grad_l,0,1), (m_L * N2, data_['model'].layersizes[l+1]))
            
            torch_permuted_a_grad_l =\
            torch_a_grad_1.permute(1,0,2).view(m_L * N2, data_['model'].layersizes[l+1])
            
            
            # print('torch.max(\
                # torch_permuted_a_grad_l - torch.from_numpy(permuted_a_grad_l).cuda())')
            # print(torch.max(
                # torch_permuted_a_grad_l - torch.from_numpy(permuted_a_grad_l).cuda()))
            # print('torch.min(\
                # torch_permuted_a_grad_l - torch.from_numpy(permuted_a_grad_l).cuda())')
            # print(torch.min(
                # torch_permuted_a_grad_l - torch.from_numpy(permuted_a_grad_l).cuda()))

            

            

            
            # h_l_h_l_t = np.matmul(h_l, np.transpose(h_l)) + 1 # + 1 for b
            torch_h_l_h_l_t = torch.mm(torch_h_l, torch_h_l.t()) + 1 # + 1 for b
            
            # h_kron = np.kron(h_l_h_l_t, np.ones((m_L, m_L)))
            torch_h_kron = get_kronecker_torch(
                torch_h_l_h_l_t, torch.ones(m_L, m_L, device=params['device']))
            
            # print('torch.max(\
                # torch_h_kron - torch.from_numpy(h_kron).cuda())')
            # print(torch.max(
                # torch_h_kron - torch.from_numpy(h_kron).cuda()))
            # print('torch.min(\
                # torch_h_kron - torch.from_numpy(h_kron).cuda())')
            # print(torch.min(
                # torch_h_kron - torch.from_numpy(h_kron).cuda()))





   
            # D_t += np.multiply(h_kron, np.matmul(permuted_a_grad_l, np.transpose(permuted_a_grad_l)))
            torch_D_t += torch.mul(
                torch_h_kron, torch.mm(torch_permuted_a_grad_l, torch_permuted_a_grad_l.t()))
            
            # print('torch.max(torch_D_t - torch.from_numpy(D_t).cuda())')
            # print(torch.max(torch_D_t - torch.from_numpy(D_t).cuda()))
            # print('torch.min(torch_D_t - torch.from_numpy(D_t).cuda())')
            # print(torch.min(torch_D_t - torch.from_numpy(D_t).cuda()))


            
        
#         print('time for compute J J transpose: ', time.time() - start_time)
        
        # add the H term
        
#         start_time = time.time()
        
        if model.name_loss == 'binary classification':
            # print('need to check')
            # sys.exit()

            torch_D_t = 1 / N2 * torch_D_t
        
#         H = get_H(data_, params)
#         for i in range(N2):
#             D_t[i * m_L: (i+1) * m_L, i * m_L: (i+1) * m_L] += lambda_ * H[i]

            torch_D_t = torch_D_t + lambda_ * torch.eye(m_L * N2, device=params['device'])
        elif model.name_loss == 'multi-class classification':
            # D_t = get_JH(D_t, data_, params)
            torch_D_t = get_JH(torch_D_t, data_, params)

            torch_D_t = 1 / N2 * torch_D_t
        
#         H = get_H(data_, params)
#         for i in range(N2):
#             D_t[i * m_L: (i+1) * m_L, i * m_L: (i+1) * m_L] += lambda_ * H[i]
            torch_D_t = torch_D_t + lambda_ * torch.eye(m_L * N2, device=params['device'])
        else:
            
            print('Error: unknown loss')
            sys.exit()
        
        # H_test = get_JH(np.eye(len(D_t)), data_, params)
        
        # print('H_test')
        # print(H_test)
        
#         print('np.linalg.norm(H_test)')
#         print(np.linalg.norm(H_test))
        
#         print('np.linalg.norm(D_t)')
#         print(np.linalg.norm(D_t))


# ========================================
        
        
        
#         D_t = np.transpose(D_t)
        
#         for i in range(N2 * m_L):
#             D_t[:, i] = get_HV(D_t[:, i], data_, params)
        
#         D_t = np.transpose(D_t)
        
#         print('time for compute H: ', time.time() - start_time)
        
        
        
        # torch_D_t = 1 / N2 * torch_D_t
        
#         H = get_H(data_, params)
#         for i in range(N2):
#             D_t[i * m_L: (i+1) * m_L, i * m_L: (i+1) * m_L] += lambda_ * H[i]
        # torch_D_t = torch_D_t + lambda_ * np.eye(m_L * N2)
        
    else:
        print('Error! 1501')
        sys.exit()
    return torch_D_t

def get_kronecker_torch(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))

def get_JH(torch_D_t, data_, params):
    y = data_['y']
    N2 = params['N2']
    m_L = params['m_L']

    # D_t = torch_D_t.cpu().data.numpy()
    
    # D_t_1
    
    diag_y = y.view(m_L * N2)
    
    diag_y = diag_y.repeat(N2 * m_L, 1)
    
    # D_t_1 = D_t * diag_y.cpu().data.numpy()
    torch_D_t_1 = torch_D_t * diag_y
    
    # D_t_2
    
    # D_t_3 = np.zeros((m_L * N2, m_L * N2))
    torch_D_t_3 = torch.zeros(m_L * N2, m_L * N2, device=params['device'])
    for i in range(N2):
#         D_t_2[:, i] = D_t[:, i * m_L : (i+1) * m_L] @ y[i]

        # y_i = y[i].cpu().data.numpy()[:, np.newaxis]
        torch_y_i = torch.unsqueeze(y[i], -1)

        # print('torch.max(torch_y_i - torch.from_numpy(y_i).cuda())')
        # print(torch.max(torch_y_i - torch.from_numpy(y_i).cuda()))
        # print('torch.min(torch_y_i - torch.from_numpy(y_i).cuda())')
        # print(torch.min(torch_y_i - torch.from_numpy(y_i).cuda()))
    
    

        # D_t_3[:, i * m_L : (i+1) * m_L] = np.matmul(np.matmul(D_t[:, i * m_L : (i+1) * m_L], y_i), np.transpose(y_i))
        torch_D_t_3[:, i * m_L : (i+1) * m_L] =\
        torch.mm(torch.mm(torch_D_t[:, i * m_L : (i+1) * m_L], torch_y_i), torch_y_i.t())
    
    # D_t = D_t_1 - D_t_3
    torch_D_t = torch_D_t_1 - torch_D_t_3

    # print('torch.max(torch_D_t - torch.from_numpy(D_t).cuda())')
    # print(torch.max(torch_D_t - torch.from_numpy(D_t).cuda()))
    # print('torch.min(torch_D_t - torch.from_numpy(D_t).cuda())')
    # print(torch.min(torch_D_t - torch.from_numpy(D_t).cuda()))

    return torch_D_t
    

def get_H(data_, params):
    model = data_['model']
    
    if model.name_loss == 'multi-class classification':
        print('wrong')
        sys.exit()
        
        # N2_index = params['N2_index']
        m_L = params['m_L']
        N2 = params['N2']

        z_N2 = data_['z_N2']
        

        z_data = z_N2.data.numpy()
        
        H = np.zeros((N2, m_L, m_L))
        for i in range(N2):
            H[i] -= np.outer(z_data[i], z_data[i])
            H[np.diag_indices(m_L)] += z_data[i]
    elif model.name_loss == 'binary classification':
        y = data_['y']
        y = y.data.numpy()
        tilde_y = y * (1-y)
        H = np.diag(y)
    else:
        sys.exit()
    
    
    return H

def get_HV(torch_V, data_, params):
    model = data_['model']

    # y = data_['y']
    # y = y.cpu().data.numpy()
    torch_y = data_['y']

    # V = torch_V.cpu().data.numpy()

    if model.name_loss == 'multi-class classification':
    
        N2 = params['N2']
        m_L = params['m_L']
        
        torch_V = torch_V.view(N2, m_L)
        # V = np.reshape(V, (N2, m_L))
        
        
        
        
        
        # HV = np.multiply(y, V)
        torch_HV = torch.mul(torch_y, torch_V)
        
        
        # sum_HV = np.sum(HV, 1) # length N2
        torch_sum_HV = torch.sum(torch_HV, dim=1) # length N2
        
        
        # HV = HV - sum_HV[:, None] * y
        torch_HV = torch_HV - torch_sum_HV[:, None] * torch_y

        # print('torch.max(torch_HV - torch.from_numpy(HV).cuda())')
        # print(torch.max(torch_HV - torch.from_numpy(HV).cuda()))
        # print('torch.min(torch_HV - torch.from_numpy(HV).cuda())')
        # print(torch.min(torch_HV - torch.from_numpy(HV).cuda()))
            
        
        # HV = np.reshape(HV, m_L * N2)
        torch_HV = torch_HV.view(m_L * N2)

    elif model.name_loss == 'binary classification':
        y = np.squeeze(y, axis=1)
        # print('test no squeeze')

        # print('V.shape')
        # print(V.shape)

        # print('np.multiply(y, V).shape')
        # print(np.multiply(y, V).shape)

        HV = np.multiply(y, V)
    else:
        sys.exit()
    
    return torch_HV

def compute_JV(V, data_, params):
    algorithm = params['algorithm']
    
    numlayers = params['numlayers']
    N2 = params['N2']
    
    # if algorithm in ['SMW-Fisher-different-minibatch',
                    #  'SMW-Fisher-batch-grad-momentum-exponential-decay',
                    #  'SMW-Fisher',
                    #  'SMW-Fisher-momentum',
                    #  'ekfac-EF',
                    #  'kfac',
                    #  'SMW-Fisher-D_t-momentum',
                    #  'SMW-Fisher-momentum-D_t-momentum',
                    #  'GI-Fisher',
                    #  'SMW-Fisher-BD',
                    #  'RMSprop-individual-grad-no-sqrt-LM',
                    #  'SMW-Fisher-batch-grad',
                    #  'SMW-Fisher-batch-grad-momentum',
                    #  'matrix-normal-LM-momentum-grad']:
    if params['matrix_name'] == 'Fisher':

        # if algorithm == 'SMW-Fisher' or\
        # algorithm == 'SMW-Fisher-batch-grad-momentum-exponential-decay' or\
        # algorithm == 'RMSprop-individual-grad-no-sqrt-LM' or\
        # algorithm == 'SMW-Fisher-batch-grad' or\
        # algorithm == 'kfac' or\
        # algorithm == 'SMW-Fisher-batch-grad-momentum':
            # 1
        # else:
            # print('Error: need to use current minibatch.')
            # sys.exit()
            # a_grad_momentum = data_['a_grad_momentum']
            # h_momentum = data_['h_momentum']
    
        
        
    
        v = torch.zeros(N2)
        if params['if_gpu']:
            v = v.cuda()

        # a_N2 = data_['a_N2']
        a_grad_N2 = data_['a_grad_N2']
        h_N2 = data_['h_N2']
    
        for l in range(numlayers):
    
            torch_V_W_l = V[l]['W']
            torch_V_b_l = V[l]['b']
            # torch_V_W_l = torch.from_numpy(V[l]['W']).float()
            # torch_V_b_l = torch.from_numpy(V[l]['b']).float()
            # if params['if_gpu']:
                # torch_V_W_l = torch_V_W_l.cuda()
                # torch_V_b_l = torch_V_b_l.cuda()
            
            v += torch.sum(torch.mm(a_grad_N2[l], torch_V_W_l) * h_N2[l], dim = 1)
        
            
            v += torch.sum(torch_V_b_l * a_grad_N2[l], dim=1)


            # test_start_time = time.process_time()

            # test_ans_1 = torch.sum(V['b'][l] * a_grad_momentum[l], dim=1)

            # test_time = time.process_time() - test_start_time

            # print('time for 1')
            # print(test_time)

            # test_start_time = time.process_time()

            # test_ans_2 = torch.mv(a_grad_momentum[l], V['b'][l])

            # test_time = time.process_time() - test_start_time

            # print('time for 2')
            # print(test_time)

            # print('torch.max(test_ans_2 - test_ans_1)')
            # print(torch.max(test_ans_2 - test_ans_1))

            # print('torch.min(test_ans_2 - test_ans_1)')
            # print(torch.min(test_ans_2 - test_ans_1))

            # print('test minibatch')
            # sys.exit()
            
            
#         print('v:', v)
        v = v.data
        
        
#     print('V[1]: ', V[1])
#     print('1/N2 * h_momentum[1].t() @ a_grad_momentum[1]:', 1/N2 * h_momentum[1].t() @ a_grad_momentum[1])
    elif algorithm == 'SMW-GN':
        
        GN_cache = data_['GN_cache']
        
        m_L = params['m_L']
        
        a_grad = GN_cache['a_grad'] # a_grad[l]: N2, m_L, m_l
        h = GN_cache['h']

        # a[l][N2_index] @ model.W[l] # N2 * m[l]
    # (a[l][N2_index] @ model.W[l]) * h[l][N2_index] # N2 * m[l]
    #  torch.sum((a[l][N2_index] @ model.W[l]) * h[l][N2_index], dim = 1)
    
    # a[l].grad: size N1 * m[l+1], it has a coefficient 1 / N1, which should be first compensate
    # h[l]: size N1 * m[l]
    # model.W[l]: size m[l+1] * m[l]
        
        
        
        '''
        test_start_time_cpu = time.process_time()
        
        v = torch.zeros(m_L, N2)
        
        
        
        for l in range(numlayers):
        
            a_grad_l = a_grad[l]
            for i in range(m_L):

    
                
                v[i] += torch.sum(torch.mm(a_grad_l[:, i, :], V['W'][l]) * h[l], dim = 1)
                v[i] += torch.mv(a_grad_l[:, i, :], V['b'][l])
        
        v = v.view(m_L * N2)

        print('time for non-parallel')
        print(time.process_time() - test_start_time_cpu)
        '''

        v = torch.zeros(N2, m_L, device=params['device'])
        for l in range(numlayers):
            a_grad_l = a_grad[l]

            # print('a_grad_l')
            # print(a_grad_l)
            # print('V[l][W]')
            # print(V[l]['W'])
            # print('h[l]')
            # print(h[l])



            # v += torch.sum(torch.matmul(a_grad_l, torch.from_numpy(V['W'][l])) * h[l][:, None, :], dim=2)
            v += torch.sum(
                torch.matmul(a_grad_l, V[l]['W']) * h[l][:, None, :], dim=2)
            
            # v += torch.matmul(a_grad_l, torch.from_numpy(V['b'][l]))
            v += torch.matmul(a_grad_l, V[l]['b'])

        v = v.permute(1, 0)
        # v = np.swapaxes(v,0,1)

        
        v = torch.reshape(v, (m_L * N2,))
        # v = np.reshape(v, (m_L * N2,))

        # print('time for parallel')
        # print(time.process_time() - test_start_time_cpu)

        # print('test parallel')
        
    else:
        print('Error! 1502')
        sys.exit()
    
    return v

def get_cache_momentum(data_, params):
    algorithm = params['algorithm']
    N2 = params['N2']
    
    if algorithm == 'SMW-GN':
        
    
        

        a_grad_momentum = data_['a_grad']
        h_momentum = data_['h']
        
        GN_cache = {}
        GN_cache['a_grad'] = a_grad_momentum
        GN_cache['h'] = h_momentum
        
        
        data_['GN_cache'] = GN_cache

    elif algorithm == 'SMW-Fisher-batch-grad-momentum-exponential-decay' or\
    algorithm == 'SMW-Fisher-batch-grad-momentum':

        # N_iters = 30
        N_iters = params['N_iters']

        batch_grads_i = data_['model_regularized_grad_N2']
        batch_grads_test = data_['batch_grads_test']
        if len(data_['batch_grads']) == 0:

            

            batch_grads_test = {}
            batch_grads_test['W'] = []
            batch_grads_test['b'] = []
            for l in range(data_['model'].numlayers):
                batch_grads_test['W'].append(batch_grads_i['W'][l][np.newaxis,:])
                batch_grads_test['b'].append(batch_grads_i['b'][l][np.newaxis,:])

            
        elif len(data_['batch_grads']) < N_iters:
            for l in range(data_['model'].numlayers):
                # print('batch_grads_test')
                # print(batch_grads_test)
                
                batch_grads_test['W'][l] = np.concatenate(
                    (batch_grads_test['W'][l], batch_grads_i['W'][l][np.newaxis,:]), axis=0)
                batch_grads_test['b'][l] = np.concatenate(
                    (batch_grads_test['b'][l], batch_grads_i['b'][l][np.newaxis,:]), axis=0)
            
        else:
            # print('params[i]')
            # print(params['i'])

            # print('params[i] % N_iters')
            # print(params['i'] % N_iters)

            # replace_index = params['i'] % N_iters
            replace_index = 0
            swap_indices = np.asarray(
                list(range(replace_index+1, N_iters)) + list(range(0, replace_index+1)))

            replace_index = params['i'] % N_iters
            for l in range(data_['model'].numlayers):
                # batch_grads_test['W'][l][replace_index] = batch_grads_i['W'][l]
                # batch_grads_test['b'][l][replace_index] = batch_grads_i['b'][l]
                batch_grads_test['W'][l][0] = batch_grads_i['W'][l]
                batch_grads_test['b'][l][0] = batch_grads_i['b'][l]
                batch_grads_test['W'][l] = batch_grads_test['W'][l][swap_indices]
                batch_grads_test['b'][l] = batch_grads_test['b'][l][swap_indices]

            
        data_['batch_grads_test'] = batch_grads_test

        if len(data_['batch_grads']) == N_iters:
            data_['batch_grads'].popleft()
            data_['batch_grads_a_grad'].popleft()
            data_['batch_grads_h'].popleft()
        elif len(data_['batch_grads']) > N_iters:
            print('Error: len > N_iters')
            sys.exit()

        data_['batch_grads'].append(data_['model_regularized_grad_N2'])
        data_['batch_grads_a_grad'].append(data_['a_grad_N2'])
        data_['batch_grads_h'].append(data_['h_N2'])

        
        
        
        
        
        
    else:
    
        
        a_grad_N2 = data_['a_grad_N2']
        h_N2 = data_['h_N2']
    
    
    
    
        N1 = params['N1']
        
        i = params['i']
        
        numlayers = params['numlayers']
            
    
    
#         a = []
#         h = [X_mb] + h
#         for ii in range(len(cache)):
#             if ii % 2 == 0:
#                 a.append(cache[ii])
#             else:
#                 h.append(cache[ii])        
#         a.append(z)

    
    
    
    
    
    # Update running estimates
        if algorithm == 'SMW-Fisher-momentum':
            
            a_grad_momentum = data_['a_grad_momentum']
            h_momentum = data_['h_momentum']
            
            rho = min(1 - 1/(i+1), 0.95)
        
            for l in range(numlayers):
                a_grad_momentum[l] = rho * a_grad_momentum[l] + (1-rho) * a_grad_N2[l]
                h_momentum[l] = rho * h_momentum[l] + (1-rho) * h_N2[l]
        elif algorithm == 'SMW-Fisher-momentum-D_t-momentum':
            
            a_grad_momentum = data_['a_grad_momentum']
            h_momentum = data_['h_momentum']
            
            rho = min(1 - 1/(i+1), 0.95)
        
            for l in range(numlayers):
                a_grad_momentum[l] = rho * a_grad_momentum[l] + (1-rho) * a_grad_N2[l]
                h_momentum[l] = rho * h_momentum[l] + (1-rho) * h_N2[l]
                
            a_grad_for_D_t = []
            h_for_D_t = []
            for l in range(numlayers):
                a_grad_for_D_t.append(a_grad_N2[l])
                h_for_D_t.append(h_N2[l])
                
            data_['a_grad_for_D_t'] = a_grad_for_D_t
            data_['h_for_D_t'] = h_for_D_t
        
        elif algorithm == 'Fisher-block' or\
        algorithm == 'SMW-Fisher-D_t-momentum' or\
        algorithm == 'GI-Fisher' or\
        algorithm == 'SMW-Fisher-BD':
            print('Error: no need to get momentum.')
            sys.exit()
            # a_grad_momentum = []
            # h_momentum = []
            # for l in range(numlayers):
                # a_grad_momentum.append(a_grad_N2[l])
                # h_momentum.append(h_N2[l])
            
    
        
        
        else:
            print('Error! 1503')
            sys.exit()
        
        data_['a_grad_momentum'] = a_grad_momentum
        data_['h_momentum'] = h_momentum


    return data_



def get_subtract(model_grad, delta, params):
    diff_p = get_zero(params)
    for l in range(params['numlayers']):
        for key in diff_p[l]:
            diff_p[l][key] = np.subtract(model_grad[l][key], delta[l][key])
    return diff_p

def get_subtract_torch(model_grad, delta):
    # diff_p = get_zero(params)
    diff_p = []
    for l in range(len(model_grad)):
        diff_p_l = {}
        for key in model_grad[l]:
            diff_p_l[key] = torch.sub(model_grad[l][key], delta[l][key])
        diff_p.append(diff_p_l)
    return diff_p



def get_plus(model_grad, delta):
    sum_p = []
    for l in range(len(model_grad)):
        sum_p_l = {}
        for key in model_grad[l]:
            sum_p_l[key] = np.add(model_grad[l][key], delta[l][key])
        sum_p.append(sum_p_l)
    return sum_p

def get_plus_torch(model_grad, delta):
    sum_p = []
    for l in range(len(model_grad)):
        sum_p_l = {}
        for key in model_grad[l]:
            sum_p_l[key] = model_grad[l][key] + delta[l][key]
        sum_p.append(sum_p_l)
    return sum_p

def get_if_nan(p):
    for l in range(len(p)):
        for key in p[l]:
            # print('p[l][key] != p[l][key]')
            # print(p[l][key] != p[l][key])
            if torch.sum(p[l][key] != p[l][key]):
                return True
    return False



def get_torch_tensor(p, params):
    p_torch = []
    for l in range(len(p)):
        p_torch_l = {}
        for key in p[l]:
            p_torch_l[key] = torch.from_numpy(p[l][key]).to(params['device'])
        p_torch.append(p_torch_l)
    return p_torch

def get_plus_scalar(alpha, model_grad):
    sum_p = []

    # numlayers = params['numlayers']
    for l in range(len(model_grad)):
        sum_p_l = {}
        for key in model_grad[l]:
            sum_p_l[key] = model_grad[l][key] + alpha
        sum_p.append(sum_p_l)
    return sum_p

def get_multiply_scalar(alpha, delta):
    alpha_p = []
    for l in range(len(delta)):
        alpha_p_l = {}
        for key in delta[l]:
            alpha_p_l[key] = alpha * delta[l][key]
        alpha_p.append(alpha_p_l)
    return alpha_p

def get_multiply_scalar_no_grad(alpha, delta):
    alpha_p = []
    for l in range(len(delta)):
        alpha_p_l = {}
        for key in delta[l]:
            alpha_p_l[key] = alpha * delta[l][key].data
        alpha_p.append(alpha_p_l)
    return alpha_p

def get_multiply_scalar_blockwise(alpha, delta, params):
    alpha_p = []
    for l in range(params['numlayers']):
        alpha_p_l = {}
        for key in delta[l]:
            alpha_p_l[key] = alpha[l] * delta[l][key]
        alpha_p.append(alpha_p_l)
    return alpha_p

def get_multiply_torch(alpha, delta):
    alpha_p = []
    for l in range(len(delta)):
        alpha_p_l = {}
        for key in delta[l]:
            alpha_p_l[key] = torch.mul(alpha[l][key], delta[l][key])
        alpha_p.append(alpha_p_l)
    return alpha_p

def get_multiply(alpha, delta):
    alpha_p = []
    for l in range(len(delta)):
        alpha_p_l = {}
        for key in delta[l]:
            alpha_p_l[key] = np.multiply(alpha[l][key], delta[l][key])
        alpha_p.append(alpha_p_l)
    return alpha_p

def get_weighted_sum_batch(hat_v, batch_grads_test, params):
    alpha_p = get_zero(params)
    for l in range(params['numlayers']):

        # print('hat_v.shape')
        # print(hat_v.shape)
        # print('batch_grads_test[W][l].shape')
        # print(batch_grads_test['W'][l].shape)
        # print('(hat_v * batch_grads_test[W][l]).shape')
        # print((hat_v * batch_grads_test['W'][l]).shape)

        # print('np.sum(hat_v * batch_grads_test[W][l], axis=0).shape')
        # print(np.sum(hat_v * batch_grads_test['W'][l], axis=0).shape)

        alpha_p['W'][l] = np.sum(hat_v[:, None, None] * batch_grads_test['W'][l], axis=0)
        alpha_p['b'][l] = np.sum(hat_v[:, None] * batch_grads_test['b'][l], axis=0)
    return alpha_p

def get_opposite(delta):
    numlayers = len(delta)
    
    

    p = []
    for l in range(numlayers):
        # if params['layers_params'][l]['name'] == 'fully-connected':
        p_l = {}
        for key in delta[l]:
            p_l[key] = -delta[l][key]
        # else:
            # print('Error: layer unsupported')
            # sys.exit()
        p.append(p_l)
        
    return p


def SMW_GN_update(data_, params):
    # a[l].grad: size N1 * m[l+1], it has a coefficient 1 / N1, which should be first compensate
    # h[l]: size N1 * m[l]
    # model.W[l]: size m[l+1] * m[l]
    
    model_grad = data_['model_regularized_grad_used_torch']
    model = data_['model']
    
    
    
    N1 = params['N1']
    N2 = params['N2']
    lambda_ = params['lambda_']
    
    m_L = data_['model'].layersizes[-1]

    params['m_L'] = m_L
    m_L = params['m_L']
    
    
    
    
    
    
    
#     start_time = time.time()
    
    data_ = get_cache_momentum(data_, params)
    # print('test get_cache_momentum')

#     print('time for get cache momentum: ', time.time() - start_time)
    
#     start_time = time.time()
    
    z_N2 = data_['z_N2']
    z_data = z_N2
    y = F.softmax(z_data, dim = 1)
    data_['y'] = y
    # print('test no y')
    
#     print('time for compute y: ', time.time() - start_time)


    
    
    
    
    
        
    # compute the vector after D_t    
    

    
#     start_time = time.time()
    
    v = compute_JV(model_grad, data_, params)
    # v = torch.ones(N2 * m_L)
    # print('test compute_JV')
    
    
#     print('v of compute JV: ', v)
    
#     print('time for compute JV: ', time.time() - start_time)
    
    
    # compute hat_v
    

#     start_time = time.time()
        
    D_t = get_D_t(data_, params)

    # print(D_t)
    
#     print('time for get D_t: ', time.time() - start_time)
    
#     start_time = time.time()
    

    


    # v = torch.unsqueeze(v, -1)
    # D_t_cho = torch.cholesky(D_t.data)
    # hat_v = torch.cholesky_solve(v.data, D_t_cho)
    # hat_v = torch.squeeze(hat_v, dim=1)

    v = torch.unsqueeze(v, -1)
    hat_v, _ = torch.solve(v.data, D_t.data)
    hat_v = torch.squeeze(hat_v, dim=1)

    # theoretically, cholesky should be faster than torch.solve()
    # however, this only happens in practice when the size of matrix is large
    # if matrix is small, torch.solve() is faster
    # since we always want to deal with small matricex, we choose to use torch.solve()

    '''
    #========
    d = D_t.size()[0]
    A = np.random.rand(d, d)
    A = np.matmul(A, np.transpose(A)) + np.eye(d)

    v = np.random.rand(d,1)

    A = torch.from_numpy(A).cuda()
    v = torch.from_numpy(v).cuda()

    start_time_cpu = time.process_time()
    start_time_wall = time.time()


    A_cho = torch.cholesky(A.data)
    torch.cholesky_solve(v.data, A_cho)
    
    print(time.process_time() - start_time_cpu)
    print(time.time() - start_time_wall)
    
    # A = np.random.rand(d, d)
    # A = np.matmul(A, np.transpose(A)) + np.eye(d)

    # v = np.random.rand(d,1)

    # A = torch.from_numpy(A).cuda()
    # v = torch.from_numpy(v).cuda()
    
    start_time_cpu = time.process_time()
    start_time_wall = time.time()
    
    torch.solve(v.data, A.data)

    print(time.process_time() - start_time_cpu)
    print(time.time() - start_time_wall)

    print('d')
    print(d)
    print('test 2 method')

    '''
    
    

    
    
    if model.name_loss == 'binary classification':
        1
    elif model.name_loss == 'multi-class classification':
        hat_v = get_HV(hat_v, data_, params)
    else:
        print('Error: unknown loss.')
        sys.exit()
    
    # hat_v = np.float32(hat_v)
    
    # hat_v = torch.from_numpy(hat_v)
    
#     hat_v = hat_v.long()
    
#     print('time for solve linear system: ', time.time() - start_time)
    
    # print('hat_v: ', hat_v)

#     hat_v = torch.ones(N2)
    
#     print('hat_v: ', hat_v)
#     print('1 - hat_v: ', 1 - hat_v)

    # compute natural gradient
    

    
#     start_time = time.time()
    
    
    
    delta = compute_sum_J_transpose_V_backp(hat_v, data_, params)
    # delta = model_grad
    # print('test compute_sum_J_transpose_V_backp')
    
#     print('time for compute J transpose V: ', time.time() - start_time)
    
#     print('\n')
    

    
    
        

        
        

        

    delta = get_multiply_scalar(1 / N2, delta)
    
    # print('delta')
    # print(delta)
    
    
    
    delta = get_subtract_torch(model_grad, delta)
    
    delta = get_multiply_scalar(1 / lambda_, delta)
    
        
    p = get_opposite(delta)
   
    data_['p_torch'] = p
        
    return data_



def compute_sum_J_transpose_V(v, data_, params):
    a_grad_momentum = data_['a_grad_momentum'] # N2 * m[l+1]
    h_momentum = data_['h_momentum'] # N2 * m[l]
    
    numlayers = params['numlayers']
    
    delta = []
    # delta['W'] = list(range(numlayers))
    # delta['b'] = list(range(numlayers))

    
    # test_start_time = time.process_time()
    

    # print('time for method 1')
    # print(time.process_time() - test_start_time)

    # ===========

    # test_start_time = time.process_time()

    for l in range(numlayers):
        delta_l = {}
        delta_l['b'] = torch.mv(a_grad_momentum[l].t(), v)
        delta_l['W'] = torch.mm(
            (v[:, None] * a_grad_momentum[l]).t(), h_momentum[l])
        delta.append(delta_l)

    # print('time for method 2')
    # print(time.process_time() - test_start_time)

    # print('test new method')
    # sys.exit()
    
    
    return delta





def update_lambda(p, data_, params):
    true_algorithm = params['algorithm']
    if params['algorithm'] in ['SMW-Fisher-signVAsqrt-p',
                               'SMW-Fisher-VA-p',
                               'SMW-Fisher-momentum-p-sign',
                               'SMW-Fisher-momentum-p',
                               'SMW-Fisher-momentum',
                               'SMW-Fisher-sign']:
        params['algorithm'] = 'SMW-Fisher'
    elif params['algorithm'] in ['kfac-momentum-grad',
                                 'kfac-EF',
                                 'kfac-TR',
                                 'kfac-momentum-grad-TR',
                                 'kfac-CG',
                                 'kfac-momentum-grad-CG',]:
        params['algorithm'] = 'kfac'

    model = data_['model']
    X_mb_N1 = data_['X_mb_N1']
    t_mb_N1 = data_['t_mb_N1']
    loss_N1 = data_['regularized_loss']
    
#     model_grad = data_['model_regularized_grad_used_torch']
    model_grad = data_['model_grad_used_torch']
    
    numlayers = params['numlayers']
    lambda_ = params['lambda_']
    boost = params['boost']
    drop = params['drop']
    
    algorithm = params['algorithm']
    
    
    # compute rho
      

#     [ll_chunk, ~] =...
#             computeLL(paramsp + test_p, indata, outdata, numchunks, targetchunk)

#     print('model.W[1].grad: ', model.W[1].grad)


        
    ll_chunk = get_new_loss(model, p, X_mb_N1, t_mb_N1, params)
    oldll_chunk = loss_N1



   
    
        
        
    if oldll_chunk - ll_chunk < 0:
        rho = float("-inf")
    else:
        if algorithm in ['SMW-Fisher-different-minibatch',
                         'SMW-Fisher',
                         'SMW-GN',
                         'GI-Fisher',
                         'matrix-normal-same-trace',
                         'matrix-normal',
                         'Kron-BFGS-LM',
                         'Kron-BFGS-LM-sqrt']:
            denom = - 0.5 * get_dot_product_torch(model_grad, p)
        elif algorithm in ['SMW-Fisher-batch-grad-momentum-exponential-decay',
                           'ekfac-EF',
                           'kfac',
                           'kfac-test',
                           'kfac-no-max',
                           'kfac-NoMaxNoSqrt',
                           'SMW-Fisher-momentum',
                           'SMW-Fisher-D_t-momentum',
                           'SMW-Fisher-momentum-D_t-momentum',
                           'SMW-Fisher-BD',
                           'RMSprop-individual-grad-no-sqrt-LM',
                           'SMW-Fisher-batch-grad',
                           'SMW-Fisher-batch-grad-momentum']:
#             print('error: should use grad on N1')
#             sys.exit()
            
            denom = computeFV(p, data_, params)
                
            denom = get_dot_product_torch(p, denom)
            
            
            denom = -0.5 * denom
            denom = denom - get_dot_product_torch(model_grad, p)
                
        else:
            print('algorithm')
            print(algorithm)
            print('Error! 1504')
            sys.exit()
        
        rho = (oldll_chunk - ll_chunk) / denom
    
    
    # update lambda   
    if rho < 0.25:
        lambda_ = lambda_ * boost
    elif rho > 0.75:
        lambda_ = lambda_ * drop

    if true_algorithm in ['SMW-Fisher-signVAsqrt-p',
                          'SMW-Fisher-VA-p',
                          'SMW-Fisher-momentum-p-sign',
                          'SMW-Fisher-momentum-p',
                          'SMW-Fisher-momentum',
                          'SMW-Fisher-sign',
                          'kfac-momentum-grad',
                          'kfac-TR',
                          'kfac-momentum-grad-TR',
                          'kfac-EF',
                          'kfac-CG',
                          'kfac-momentum-grad-CG',]:
        params['algorithm'] = true_algorithm
        
    return lambda_

def GI_Fisher_update(data_, params):
    model_grad = data_['model_grad_used']

    # if algorithm == 'SMW-Fisher-momentum':
        # a_grad_momentum = data_['a_grad_momentum']
        # h_momentum = data_['h_momentum']

    # print('model_grad[b][1]')
    # print(model_grad['b'][1])
        
        
    
    N1 = params['N1']
    N2 = params['N2']
    # i = params['i']
    # lambda_ = params['lambda_']
    # numlayers = params['numlayers']
    
    
    
    
    # N2_index = np.random.permutation(N1)[:N2]
    # N2_index = params['N2_index']

    data_ = get_cache_momentum(data_, params)

    # compute 1 / n * J * J^T := G and do cho fac

    # params['lambda_'] = 10**(-8)
    # params['lambda_'] = 10**(-5)
    # params['lambda_'] = 10**(-3)
    G = get_D_t(data_, params)

    # G_cho_fac = scipy.linalg.cho_factor(G)
    G_cho = torch.cholesky(G.data)
    

    # compute J * g
    v = compute_JV(model_grad, data_, params)
    

    # compute G^{-1} * (J * g)
    # hat_v = scipy.linalg.cho_solve(G_cho_fac, v)

    # hat_v, _ = torch.solve(v.data, G.data)
    hat_v = torch.cholesky_solve(v.data, G_cho)
    

    # compute G^{-1} * (G^{-1} * J * g)
    # hat_v = scipy.linalg.cho_solve(G_cho_fac, hat_v)


    # hat_v = np.float32(hat_v)
    # hat_v = torch.from_numpy(hat_v)

    # hat_v, _ = torch.solve(hat_v.data, G.data)
    hat_v = torch.cholesky_solve(hat_v.data, G_cho)



    # compute J^T * (G^{-1} * G^{-1} * J * g)
    delta = compute_sum_J_transpose_V_backp(hat_v, data_, params)

    # dividing by n
    delta = get_multiply_scalar(1 / N2, delta)

    # get minus
    p = get_opppsite(delta, params)
    data_['p'] = p
    return data_, params
   

    
    



def SMW_Fisher_batch_grad_update(data_, params):
    model_grad = data_['model_grad_used']
    
    N_iters = params['N_iters']
    N2 = params['N2']
    lambda_ = params['lambda_']
    numlayers = params['numlayers']

    if params['algorithm'] == 'SMW-Fisher-batch-grad-momentum-exponential-decay' or\
    params['algorithm'] == 'SMW-Fisher-batch-grad-momentum':
        data_ = get_cache_momentum(data_, params)
    elif params['algorithm'] == 'SMW-Fisher-batch-grad':
        1
    else:
        print('Error: need more on cache')
        sys.exit()

    if params['algorithm'] == 'SMW-Fisher-batch-grad-momentum-exponential-decay':
        # print('params[rho_kfac]')
        # print(params['rho_kfac'])

        rho_kfac = params['rho_kfac']
        N_current = len(data_['batch_grads'])
        c_weights = np.asarray(list(range(N_current)))
        c_weights = N_current - 1 - c_weights
        c_weights = np.power(rho_kfac, c_weights)
        c_weights = c_weights * (1 - rho_kfac) / (1 - (rho_kfac**N_current))

        # print('c_weights')
        # print(c_weights)
        # print('np.sum(c_weights)')
        # print(np.sum(c_weights))

        # print('test weight')

    # compute J * g
    if params['algorithm'] == 'SMW-Fisher-batch-grad-momentum-exponential-decay' or\
    params['algorithm'] == 'SMW-Fisher-batch-grad-momentum':
        batch_grads = data_['batch_grads']

        # test_start_time = time.process_time()

        v = np.zeros(len(batch_grads))
        for i in range(len(batch_grads)):
            v[i] = get_dot_product(model_grad, batch_grads[i])

        if params['algorithm'] == 'SMW-Fisher-batch-grad-momentum-exponential-decay':
            v = np.sqrt(c_weights) * v

        # test_time = time.process_time() - test_start_time
        # print('time for method 1')
        # print(test_time)

        # ======

        # batch_grads_test = data_['batch_grads_test']

        # test_start_time = time.process_time()

        # v_test = get_dot_product_batch(model_grad, batch_grads_test, params)

        # test_time = time.process_time() - test_start_time
        # print('time for method 2')
        # print(test_time)

        # ================

        # test_start_time = time.process_time()

        # v_test = np.zeros(len(batch_grads))
        # for i in range(len(batch_grads)):
            # batch_grads_a_grad_i = data_['batch_grads_a_grad'][i]
            # batch_grads_h_i = data_['batch_grads_h'][i]
            # for l in range(numlayers):

                
                # v_test[i] += 1 / N2 * torch.sum(
                    # torch.mm(
                        # batch_grads_a_grad_i[l],
                        # torch.from_numpy(model_grad['W'][l]).float()) * batch_grads_h_i[l])
        
            
                # v_test[i] += 1 / N2 * torch.sum(
                    # torch.from_numpy(model_grad['b'][l]).float() * batch_grads_a_grad_i[l])
                
        # print('max(v_test - v)')
        # print(max(v_test - v))
        # print('min(v_test - v)')
        # print(min(v_test - v))

        # test_time = time.process_time() - test_start_time
        # print('time for method 2')
        # print(test_time)

        

        # print('test new J g')
        # sys.exit()
    elif params['algorithm'] == 'SMW-Fisher-batch-grad':
        model_grad_N2 = data_['model_regularized_grad_N2']
        v = get_dot_product(model_grad, model_grad_N2)
    else:
        print('Error: need more J')
        sys.exit()
        

    # compute D_t
    if params['algorithm'] == 'SMW-Fisher-batch-grad-momentum-exponential-decay' or\
    params['algorithm'] == 'SMW-Fisher-batch-grad-momentum':
        N_iters = params['N_iters']

        # delete old
        if len(data_['D_t_minus_lambda']) == N_iters:
            data_['D_t_minus_lambda'] = data_['D_t_minus_lambda'][1:, 1:]

        # add new
        if len(batch_grads) == 1:
            data_['D_t_minus_lambda'] =\
            np.ones((1,1)) * get_dot_product(batch_grads[0], batch_grads[0])
        else:

            # test_start_time = time.process_time()

            # D_t_i = np.zeros((len(batch_grads), 1))
            # for i in range(len(batch_grads)):
                # D_t_i[i, 0] = get_dot_product(batch_grads[-1], batch_grads[i])

            # test_time = time.process_time() - test_start_time
            # print('time for method 1')
            # print(test_time)

            # ================

            # test_start_time = time.process_time()

            D_t_i = np.zeros((len(batch_grads), 1))
            batch_grads_a_grad_j = data_['batch_grads_a_grad'][-1]
            batch_grads_h_j = data_['batch_grads_h'][-1]
            for i in range(len(batch_grads)):
                batch_grads_a_grad_i = data_['batch_grads_a_grad'][i]
                batch_grads_h_i = data_['batch_grads_h'][i]
                for l in range(numlayers):

                    D_t_i[i, 0] += 1 / (N2**2) * (
                        torch.mul(torch.mm(batch_grads_a_grad_j[l], batch_grads_a_grad_i[l].t()),
                                    torch.mm(batch_grads_h_j[l], batch_grads_h_i[l].t()) + 1)).sum()
                    
            # test_time = time.process_time() - test_start_time
            # print('time for method 2')
            # print(test_time)
            
            # print('max(D_t_i_test - D_t_i)')
            # print(max(D_t_i_test - D_t_i))
            # print('min(D_t_i_test - D_t_i)')
            # print(min(D_t_i_test - D_t_i))
            # print('test new D_t')
            # sys.exit()

            data_['D_t_minus_lambda'] = np.concatenate((data_['D_t_minus_lambda'], D_t_i[:-1]), axis=1)
            data_['D_t_minus_lambda'] = np.concatenate(
                (data_['D_t_minus_lambda'], np.transpose(D_t_i)), axis=0)
            
        if params['algorithm'] == 'SMW-Fisher-batch-grad-momentum-exponential-decay':
            D_t = data_['D_t_minus_lambda'] * np.outer(np.sqrt(c_weights), np.sqrt(c_weights))
        elif params['algorithm'] == 'SMW-Fisher-batch-grad-momentum':
            D_t = 1 / len(data_['D_t_minus_lambda']) * data_['D_t_minus_lambda']
        D_t = D_t + lambda_ * np.eye(len(data_['D_t_minus_lambda']))
            
    elif params['algorithm'] == 'SMW-Fisher-batch-grad':
        D_t = lambda_
        D_t += get_dot_product(model_grad_N2, model_grad_N2)
    else:
        print('Error: need more D_t')
        sys.exit()
        


    # compute D_t^{-1} * (J * g)
    if params['algorithm'] == 'SMW-Fisher-batch-grad-momentum-exponential-decay' or\
    params['algorithm'] == 'SMW-Fisher-batch-grad-momentum':# D_t_cho_fac = scipy.linalg.cho_factor(D_t)
        # hat_v = scipy.linalg.cho_solve(D_t_cho_fac, v)

        # hat_v = torch.solve(v.data, D_t.data).data

        D_t_cho = torch.cholesky(D_t.data)
        hat_v = torch.cholesky_solve(v.data, D_t_cho)
    elif params['algorithm'] == 'SMW-Fisher-batch-grad':
        hat_v = v / D_t
    else:
        print('Error: need more solve')
        sys.exit()
        

    # compute J^T * (D_t^{-1} * (J * g))
    if params['algorithm'] == 'SMW-Fisher-batch-grad-momentum-exponential-decay' or\
    params['algorithm'] == 'SMW-Fisher-batch-grad-momentum':
        if params['algorithm'] == 'SMW-Fisher-batch-grad-momentum-exponential-decay':
            hat_v = hat_v * np.sqrt(c_weights)

        
        batch_grads_test = data_['batch_grads_test']

        # test_start_time = time.process_time()

        p = get_weighted_sum_batch(hat_v, batch_grads_test, params)

        # test_time = time.process_time() - test_start_time
        # print('time for method 2')
        # print(test_time)

        # print('max(test_p[W][0] - p[W][0])')
        # print((test_p['W'][0] - p['W'][0]).max())
        # print('min(test_p[W][0] - p[W][0])')
        # print((test_p['W'][0] - p['W'][0]).min())
        # print('max(test_p[b][1] - p[b][1])')
        # print((test_p['b'][1] - p['b'][1]).max())
        # print('min(test_p[b][1] - p[b][1])')
        # print((test_p['b'][1] - p['b'][1]).min())

        # print('test new J^T')
    elif params['algorithm'] == 'SMW-Fisher-batch-grad':
        p = get_multiply_scalar(hat_v, model_grad_N2)
    else:
        print('Error: need more transpose')
        sys.exit()
        
    
    # rest of SMW
    if params['algorithm'] == 'SMW-Fisher-batch-grad-momentum':
        p = get_multiply_scalar(1 / N_iters, p)

    p = get_subtract(model_grad, p, params)
    
    p = get_multiply_scalar(1 / lambda_, p)

    
    p = get_opposite(p)
    data_['p'] = p
    return data_, params

def get_new_loss(model, p, x, t, params):
    
    model_new = copy.deepcopy(model)

    device = params['device']
    
    for l in range(model_new.numlayers):
        for key in model_new.layers_weight[l]:
            model_new.layers_weight[l][key].data += p[l][key].data
            
            
    

    reduction = 'mean'
    loss = get_regularized_loss_from_x_no_grad(
        model_new, x, t, reduction, params['tau'])

    return loss

def get_dot_product(delta_1, delta_2):
    dot_product = 0
    for l in range(len(delta_1)):
        for key in delta_1[l]:
            dot_product += np.sum(np.multiply(delta_1[l][key], delta_2[l][key]))
    return dot_product

def get_dot_product_blockwise(delta_1, delta_2):
    dot_product = []
    for l in range(len(delta_1)):
        dot_product_l = 0
        for key in delta_1[l]:
            dot_product_l += np.sum(np.multiply(delta_1[l][key], delta_2[l][key]))
        dot_product.append(dot_product_l)
    return dot_product

def get_dot_product_torch(delta_1, delta_2):
    dot_product = 0
    for l in range(len(delta_1)):
        for key in delta_1[l]:
            dot_product += torch.sum(torch.mul(delta_1[l][key], delta_2[l][key]))
    return dot_product

def get_dot_product_blockwise_torch(delta_1, delta_2):
    dot_product = []
    for l in range(len(delta_1)):
        dot_product_l = 0
        for key in delta_1[l]:
            dot_product_l += torch.sum(torch.mul(delta_1[l][key], delta_2[l][key]))
        dot_product.append(dot_product_l)
    return dot_product

def get_dot_product_batch(model_grad, batch_grads_test, params):
    # numlayers = params['numlayers']
    
    dot_product = np.zeros(len(batch_grads_test['W'][0]))
    for l in range(params['numlayers']):
        dot_product += np.sum(
            np.sum(np.multiply(model_grad['W'][l][None, :], batch_grads_test['W'][l]), axis=-1), axis=-1)
        dot_product += np.sum(np.multiply(model_grad['b'][l][None, :], batch_grads_test['b'][l]), axis=-1)
    
    return dot_product

def get_square(delta_1):
    numlayers = len(delta_1)
    sqaure_p = []
    for l in range(numlayers):
        sqaure_p_l = {}
        for key in delta_1[l]:
            sqaure_p_l[key] = np.square(delta_1[l][key])
        sqaure_p.append(sqaure_p_l)  
    return sqaure_p

def get_square_torch(delta_1):
    numlayers = len(delta_1)
    sqaure_p = []
    for l in range(numlayers):
        sqaure_p_l = {}
        for key in delta_1[l]:
            sqaure_p_l[key] = torch.mul(delta_1[l][key], delta_1[l][key])
        sqaure_p.append(sqaure_p_l)  
    return sqaure_p

def get_sqrt(delta_1):
    sqaure_p = []
    for l in range(len(delta_1)):
        sqaure_p_l = {}
        for key in delta_1[l]:
            sqaure_p_l[key] = np.sqrt(delta_1[l][key])
        sqaure_p.append(sqaure_p_l) 
    return sqaure_p

def get_sqrt_torch(delta_1):
    sqaure_p = []
    for l in range(len(delta_1)):
        sqaure_p_l = {}
        for key in delta_1[l]:
            sqaure_p_l[key] = torch.sqrt(delta_1[l][key])
        sqaure_p.append(sqaure_p_l) 
    return sqaure_p

def get_max_with_0(delta_1):
    sqaure_p = []
    for l in range(len(delta_1)):
        sqaure_p_l = {}
        for key in delta_1[l]:
            sqaure_p_l[key] = F.relu(delta_1[l][key])
        sqaure_p.append(sqaure_p_l) 
    return sqaure_p

def get_divide(delta_1, delta_2):
    numlayers = len(delta_1)
    sqaure_p = []
    for l in range(numlayers):
        sqaure_p_l = {}
        for key in delta_1[l]:
            sqaure_p_l[key] = np.divide(delta_1[l][key], delta_2[l][key])
        sqaure_p.append(sqaure_p_l)
    return sqaure_p

def get_divide_torch(delta_1, delta_2):
    numlayers = len(delta_1)
    sqaure_p = []
    for l in range(numlayers):
        sqaure_p_l = {}
        for key in delta_1[l]:
            sqaure_p_l[key] = torch.div(delta_1[l][key], delta_2[l][key])
        sqaure_p.append(sqaure_p_l)
    return sqaure_p

"""
def get_mean(delta, params):
#     import torch
    numlayers = params['numlayers']
    for l in range(numlayers):
        delta[l] = torch.mean(delta[l], dim=0)
    return delta
"""


def computeFV(delta, data_, params):
    
    
    model = data_['model']
    
    N1 = params['N1']
    N2 = params['N2']

    # N2_index = params['N2_index']
    
    algorithm = params['algorithm']
    
#     a_grad_momentum = data_['a_grad_momentum']
#     h_momentum = data_['h_momentum']
    
    
#     import time
#     start_time = time.time()

    
    v = compute_JV(delta, data_, params)
    
#     print('time for FV 1/2: ', time.time() - start_time)

    if algorithm == 'SMW-GN':
        # v = v.data.numpy()
        v = get_HV(v, data_, params)
        # v = torch.from_numpy(v)
    
    

    delta = compute_sum_J_transpose_V_backp(v, data_, params)
    
    
    #############
#     N2 = params['N2']
#     m_L = params['m_L']
#     test_v = torch.zeros(m_L * N2)
#     test_v[0] = 1
    
#     print('print(compute_sum_J_transpose_V_backp(test_v, data_, params)): ', compute_sum_J_transpose_V_backp(test_v, data_, params))
    
#     print('test')
    
    

    
    

    
#     print('delta[1].size(): ', delta[1].size())
    
    
    
#     delta = get_mean(delta, params)
    
    delta = get_multiply_scalar(1 / N2, delta)

#     delta += 
    
    return delta




    
def get_homo_grad(model_grad_N1, params):
    device = params['device']

    homo_model_grad_N1 = []
    for l in range(params['numlayers']):
        if params['layers_params'][l]['name'] == 'fully-connected':
            
            homo_model_grad_N1_l = torch.cat(
                (model_grad_N1[l]['W'], model_grad_N1[l]['b'].unsqueeze(1)), dim=1)
        elif params['layers_params'][l]['name'] in ['conv',
                                                    'conv-no-activation']:
            # take Fashion-MNIST as an example
            # model_grad_N1[l]['W']: 32 * 1 * 5 * 5
            # model_grad_N1[l]['b']: 32
            # 32: conv_out_channels
            # 1: conv_in_channels
            # 5 * 5: conv_kernel_size
            
            homo_model_grad_N1_l = torch.cat(
                (
                    model_grad_N1[l]['W'].flatten(start_dim=1),
                    model_grad_N1[l]['b'].unsqueeze(dim=1)
                ),
                dim=1
            )
            
        elif params['layers_params'][l]['name'] in ['conv-no-bias-no-activation']:
            
            homo_model_grad_N1_l = model_grad_N1[l]['W'].flatten(start_dim=1)
            
        elif params['layers_params'][l]['name'] == 'BN':
            
            homo_model_grad_N1_l = torch.cat(
                (model_grad_N1[l]['W'], model_grad_N1[l]['b'])
            )
            
        else:
            print('Error: unsupported layer when homo grad for ' + params['layers_params'][l]['name'])
            sys.exit()
        homo_model_grad_N1.append(homo_model_grad_N1_l)

    return homo_model_grad_N1  
    





def Kron_SGD_update(data_, params):
    numlayers = params['numlayers']
    
    model_grad = data_['model_regularized_grad_used_torch']
    
    delta = []
    for l in range(numlayers):
        delta_l = {}
        
        mean_a_grad_l = torch.mean(data_['a_grad_N2'][l], dim=0)
        mean_h_l = torch.mean(data_['h_N2'][l], dim=0)
        
        delta_l['W'] = torch.ger(mean_a_grad_l, mean_h_l)
        delta_l['b'] = model_grad[l]['b']
        
#         print('delta_l[W].size()')
#         print(delta_l['W'].size())
        
        delta.append(delta_l)
        
    p = get_opposite(delta)
    data_['p_torch'] = p
        
    
    return data_, params








def HessianAction_scaled_BFGS_update(Kron_BFGS_matrices_l, l, data_, params):
    
    assert params['Kron_BFGS_action_h'] == 'HessianAction-scaled-BFGS'
    
    mean_h_l = torch.mean(data_['h_N2'][l], dim=0).data

    if params['Kron_BFGS_if_homo']:
        mean_h_l = torch.cat(
(
    mean_h_l, 
    torch.ones(1, device=params['device'])
),
dim=0
)

    H_l_h = Kron_BFGS_matrices_l['H']['h']
    s_l_h = torch.mv(H_l_h, mean_h_l)

#     if action_h == 'HessianAction-scaled-BFGS':
    beta_ = params['Kron_BFGS_A_decay']
    s_l_h = s_l_h / beta_
    y_l_h = torch.mv(Kron_BFGS_matrices_l['A_LM'], s_l_h)






    beta_ = params['Kron_BFGS_A_decay']
    H_l_h = H_l_h / beta_

    Kron_BFGS_matrices_l['H']['h'], update_status =\
get_BFGS_formula_v2(H_l_h, s_l_h, y_l_h, mean_h_l, False)
    
    print('torch.norm(Kron_BFGS_matrices_l[H][h])')
    print(torch.norm(Kron_BFGS_matrices_l['H']['h']))

    if update_status != 0:


        sys.exit()
    return Kron_BFGS_matrices_l








def get_BFGS_PowellB0Damping(s_l_a, y_l_a, params):
    
    print('need to move')
    sys.exit()
    
    # B_0 = 1 / gamma * I
    
    delta = params['Kron_BFGS_H_epsilon']
    
    s_T_y = torch.dot(s_l_a, y_l_a)
    y_T_y = torch.dot(y_l_a, y_l_a)
    
    gamma = (y_T_y / s_T_y).item()
    
#     gamma = torch.max(gamma, delta)
    if gamma < delta:
        gamma = delta
    
#     print('gamma')
#     print(gamma)
    
    
    
    alpha = params['Kron_BFGS_H_epsilon']

    s_T_s = torch.dot(s_l_a, s_l_a)
#     s_T_y = torch.dot(s_l_a, y_l_a)

    s_B_s = s_T_s/gamma

    if s_T_y / s_B_s > alpha:
        1
    else:
        theta =  (1-alpha) * s_B_s / (s_B_s - s_T_y)
        y_l_a = theta * y_l_a + (1-theta) * s_l_a/gamma
        
#     sys.exit()
    
    return s_l_a, y_l_a




def get_BFGS_DoubleDamping(s_l_a, y_l_a, l, data_, params):
    
    print('need to move')
    sys.exit()
    
    # DD merged into one step
    
    
    alpha = params['Kron_BFGS_H_epsilon']
        
    Kron_BFGS_matrices_l = data_['Kron_BFGS_matrices'][l]

    if params['Kron_BFGS_action_a'] == 'LBFGS':
        Hy = LBFGS_Hv(
            y_l_a,
            data_['Kron_LBFGS_s_y_pairs']['a_grad'][l],
            params,
            False
        )
    elif params['Kron_BFGS_action_a'] == 'BFGS':
        H_l_a_grad = Kron_BFGS_matrices_l['H']['a_grad']
        Hy = torch.mv(H_l_a_grad ,y_l_a)
    else:
        print('error: not implemented for ' + params['Kron_BFGS_action_a'])
        sys.exit()


    

#     if params['Kron_BFGS_action_a'] == 'LBFGS':

        
#     elif params['Kron_BFGS_action_a'] == 'BFGS':
        
#     else:
#         print('error: not implemented for ' + params['Kron_BFGS_action_a'])
#         sys.exit()
        
    s_T_y = torch.dot(s_l_a, y_l_a)

    yHy = torch.dot(y_l_a, Hy)
    
    s_T_s = torch.dot(s_l_a, s_l_a)
    
    sigma = max(yHy.item(), s_T_s.item())
    
#     print('sigma')
#     print(sigma)
#     sys.exit()

#     if s_T_y / yHy > alpha:
    if s_T_y / sigma > alpha:
        1
    else:
        if yHy >= s_T_s:
            theta =  ((1-alpha) * yHy / (yHy - s_T_y)).item()

#             original_s_l_a = s_l_a

            s_l_a = theta * s_l_a + (1-theta) * Hy
        else:
            theta =  (1-alpha) * s_T_s / (s_T_s - s_T_y)
            y_l_a = theta * y_l_a + (1-theta) * s_l_a
    
    return s_l_a, y_l_a
    




    


def get_block_BFGS_formula(H, s, y):
    
    D = s
    
    D_t_y_inv = torch.mm(D.t(), y).inverse()
    I = torch.eye(H.size()[0]).cuda()
    
    H = torch.mm(torch.mm(D, D_t_y_inv), D.t()) +\
    torch.mm(
        torch.mm(
            I - torch.mm(torch.mm(D, D_t_y_inv), y.t()), H),
        I - torch.mm(torch.mm(y, D_t_y_inv), D.t()))
    
    
    return H

def get_BFGS_formula(H, s, y, g_k):
    
    s = s.data
    y = y.data

    # ger(a, b) = a b^T
    rho_inv = torch.dot(s, y)

    if rho_inv <= 0:
#     if rho_inv <= 10**(-3):
#         print('BFGS not updated (case 1).')
#         print('rho_inv')
#         print(rho_inv)
    
        return H, 1
    elif rho_inv <= 10**(-4) * torch.dot(s, s) * np.sqrt(torch.dot(g_k, g_k).item()):
#         print('BFGS not updated (case 2).')
        return H, 2
    
#     print('rho_inv / (torch.dot(s, s) * np.sqrt(torch.dot(g_k, g_k).item()))')
#     print(rho_inv / (torch.dot(s, s) * np.sqrt(torch.dot(g_k, g_k).item())))

    # sHs = torch.dot(s, torch.mv(H, s))
    # if rho_inv < 0.25 * sHs:
        # theta = (0.75 * sHs) / (sHs - rho_inv)



    rho = 1 / rho_inv

    # s = s / np.sqrt(rho_inv.item())
    # y = y / np.sqrt(rho_inv.item())
    # rho = 1

    Hy = torch.mv(H, y)
    H_new = H.data + (rho**2 * torch.dot(y, torch.mv(H, y)) + rho) * torch.ger(s, s) -\
    rho * (torch.ger(s, Hy) + torch.ger(Hy, s))
    

    
    if torch.norm(H_new) > 2 * torch.norm(H):
#     if torch.norm(H_new) > 5 * torch.norm(H) or\
#     torch.max(torch.isinf(H_new)):
#         print('BFGS not updated (case 3).')
        return H, 3
#         H = H_new
    elif torch.max(torch.isinf(H_new)):
        return H, 4
    else:
        H = H_new

    if torch.max(torch.isinf(H)):
        print('inf in H')
        print('s')
        print(s)
        print('y')
        print(y)
        sys.exit()

    return H, 0








def get_CG(func_A, b, x, max_iter, data_):
    # solve A x = b

    # input:
    # x: initial point

    # https://gist.github.com/sfujiwara/b135e0981d703986b6c2

    # print('x.size()')
    # print(x.size())
    # print('b.size()')
    # print(b.size())
    # print('func_A(x).size()')
    # print(func_A(x).size())

    r = func_A(x) - b
    p = - r
    r_k_norm = torch.sum(r * r)
    i = 0
    while i < max_iter:
        Ap = func_A(p)
        alpha = r_k_norm / torch.sum(p * Ap)

        # if alpha != alpha:
            # print('nan in alpha')
            # print('r_k_norm')
            # print(r_k_norm)
            # print('torch.sum(p * Ap)')
            # print(torch.sum(p * Ap))
            # print('torch.sum(p * p)')
            # print(torch.sum(p * p))
            # print('p')
            # print(p)
            # print('i')
            # print(i)
            # print('torch.sum(Ap * Ap)')
            # print(torch.sum(Ap * Ap))
            # sys.exit()

        x += alpha * p

        # if torch.sum(x != x):
            # print('nan in x')
            # print('x')
            # print(x)
            # print('p')
            # print(p)
            # sys.exit()

        r += alpha * Ap
        r_kplus1_norm = torch.sum(r * r)
        beta = r_kplus1_norm / r_k_norm
        r_k_norm = r_kplus1_norm

        if r_kplus1_norm < 1e-10:
        # if torch.sum(x * func_A(x)) - torch.sum(x * b) < 1e-10:
            break
        p = beta * p - r
        i += 1

    # if torch.sum(x != x):
        # print('nan in x')
        # print('x')
        # print(x)
        # sys.exit()

    return x


def get_safe_division(x, y):
    # if y == 0:
        # return 1e16
    if x == 0 and y == 0:
        print('Error: x = 0 and y = 0 in safe division')
        sys.exit()
    elif x == 0 and y !=0:
        return 0
    elif x != 0 and y == 0:
        return 1e16
    else:
        # print('np.log(x)')
        # print(np.log(x))
        # print('np.log(y)')
        # print(np.log(y))
        # print('np.log(x) - np.log(y)')
        # print(np.log(x) - np.log(y))
        if np.log(x) - np.log(y) < np.log(1e16):
            return np.exp(np.log(x) - np.log(y))
        else:
            return 1e16


def SGD_update(data_, params):
    true_algorithm = params['algorithm']
    if params['algorithm'] in ['SGD-yura-MA',
                               'SGD-yura',
                               'SGD-momentum-yura',
                               'SGD-VA',
                               'SGD-signVAsqrt',
                               'SGD-signVAerf',
                               'SGD-signVA',
                               'SGD-sign']:
        params['algorithm'] = 'SGD'
    elif params['algorithm'] == 'SGD-yura-old':
        params['algorithm'] = 'SGD-yura'

#     model_grad = data_['model_regularized_grad_used_torch']
    model_grad = data_['model_grad_used_torch']
        
    p = get_opposite(model_grad)
    # p_torch = get_opposite(model_grad_torch, params)

    if params['algorithm'] == 'SGD-yura' or\
    params['algorithm'] == 'SGD-yura-BD':
        # alpha = 2
        alpha = 1
        
        print('check whether we should use alpha or alpha_current')
        sys.exit()
        

        if params['i'] == 0:
            if params['algorithm'] == 'SGD-yura':
                lambda_0 = 1
                lambda_k = lambda_0
                theta_k = 10**10
            elif params['algorithm'] == 'SGD-yura-BD':
                lambda_0 = [1] * params['numlayers']
                lambda_k = lambda_0
                theta_k = [10**10] * params['numlayers']
                # print('test')
        else:
            lambda_k_minus_1 = params['yura_lambda']
            theta_k_minus_1 = params['yura_theta']
            weights_k_minus_1 = params['yura_weights']
            grad_k_minus_1 = params['yura_grad']

            
            # get previous grad
            model_new = copy.deepcopy(data_['model'])
            # model_new = get_model(params)


            device = params['device']
            for l in range(model_new.numlayers):
                for key in model_new.layers_weight[l]:

                    # model_new.layers_weight[l][key].data = torch.from_numpy(weights_k_minus_1[l][key]).float().to(device)
                    model_new.layers_weight[l][key].data = weights_k_minus_1[l][key].data
            

            reduction = 'mean'
            loss = get_regularized_loss_from_x(model_new, data_['X_mb'], data_['t_mb'], reduction)

            model_new.zero_grad()

            loss.backward()

            grad_k_minus_1_torch = get_model_grad(model_new, params)

            

            



            # diff_grad = get_subtract(model_grad, grad_k_minus_1, params)
            diff_grad_torch = get_subtract_torch(model_grad_torch, grad_k_minus_1_torch)

            if params['algorithm'] == 'SGD-yura':

                # diff_weights = get_subtract_torch(data_['model'].layers_weight, weights_k_minus_1)
                # norm_sqaure_diff_weights = get_dot_product_torch(diff_weights, diff_weights).cpu().data.numpy()
                norm_sqaure_diff_weights =\
                get_dot_product_torch(data_['p_torch'], data_['p_torch']) * (params['alpha']**2)
                norm_sqaure_diff_weights_np = norm_sqaure_diff_weights.cpu().data.numpy()

                # norm_sqaure_diff_grad = get_dot_product(diff_grad, diff_grad)

                norm_sqaure_diff_grad_torch = get_dot_product_torch(diff_grad_torch, diff_grad_torch)
                norm_sqaure_diff_grad_np = norm_sqaure_diff_grad_torch.cpu().data.numpy()

                # print('norm_sqaure_diff_grad')
                # print(norm_sqaure_diff_grad)
                # print('norm_sqaure_diff_grad_torch')
                # print(norm_sqaure_diff_grad_torch)
                # sys.exit()

                L_k_inv = np.sqrt(get_safe_division(norm_sqaure_diff_weights_np,
                    norm_sqaure_diff_grad_np))
                lambda_k = min(np.sqrt(1 + theta_k_minus_1 / 10) * lambda_k_minus_1, L_k_inv / alpha)

                if lambda_k == 0:
                    print('Warning: lambda_k == 0')
                    lambda_k = lambda_k_minus_1
                    # print('norm_sqaure_diff_grad_np')
                    # print(norm_sqaure_diff_grad_np)
                    # print('norm_sqaure_diff_grad_torch')
                    # print(norm_sqaure_diff_grad_torch)
                    # print('norm_sqaure_diff_weights_np')
                    # print(norm_sqaure_diff_weights_np)
                    # print('norm_sqaure_diff_weights')
                    # print(norm_sqaure_diff_weights)
                    # sys.exit()

                theta_k = lambda_k / lambda_k_minus_1

                # print('lambda_k')
                # print(lambda_k)
                # print('np.sqrt(1 + theta_k_minus_1 / 10) * lambda_k_minus_1')
                # print(np.sqrt(1 + theta_k_minus_1 / 10) * lambda_k_minus_1)
                # print('L_k_inv / alpha')
                # print(L_k_inv / alpha)
                # print('L_k_inv')
                # print(L_k_inv)
                # print('norm_sqaure_diff_weights_np')
                # print(norm_sqaure_diff_weights_np)
                # print('norm_sqaure_diff_grad_np')
                # print(norm_sqaure_diff_grad_np)
            elif params['algorithm'] == 'SGD-yura-BD':
                lambda_k = []
                theta_k = []

                norm_sqaure_diff_weights =\
                get_dot_product_blockwise_torch(data_['p_torch'], data_['p_torch']) * (params['alpha']**2)
                norm_sqaure_diff_weights = [element_.cpu().data.numpy() for element_ in norm_sqaure_diff_weights]

                norm_sqaure_diff_grad = get_dot_product_blockwise(diff_grad, diff_grad)

                for l in range(params['numlayers']):

                    



                    L_k_inv = np.sqrt(get_safe_division(norm_sqaure_diff_weights[l], norm_sqaure_diff_grad[l]))
                    lambda_k_l = min(
                        np.sqrt(1 + theta_k_minus_1[l] / 10) * lambda_k_minus_1[l], L_k_inv / alpha)
                    
                    if lambda_k_l == 0:
                        print('Error: lambda_k_l == 0')
                        print('l')
                        print(l)
                        print('lambda_k_l')
                        print(lambda_k_l)
                        print('np.sqrt(1 + theta_k_minus_1[l] / 10) * lambda_k_minus_1[l]')
                        print(np.sqrt(1 + theta_k_minus_1[l] / 10) * lambda_k_minus_1[l])
                        print('L_k_inv / alpha')
                        print(L_k_inv / alpha)
                        print('L_k_inv')
                        print(L_k_inv)
                        print('norm_sqaure_diff_weights[l]')
                        print(norm_sqaure_diff_weights[l])
                        sys.exit()

                    if lambda_k_minus_1[l] == 0:
                        print('Error: lambda_k_minus_1[l] == 0')
                        print('l')
                        print(l)
                        print('lambda_k_minus_1[l]')
                        print(lambda_k_minus_1[l])
                        sys.exit()

                    theta_k_l = lambda_k_l / lambda_k_minus_1[l]

                    lambda_k.append(lambda_k_l)
                    theta_k.append(theta_k_l)
                    # print('test')

                

        if params['algorithm'] == 'SGD-yura':    
            p = get_multiply_scalar(lambda_k, p)
        elif params['algorithm'] == 'SGD-yura-BD':
            p = get_multiply_scalar_blockwise(lambda_k, p)

        params['yura_lambda'] = lambda_k
        params['yura_theta'] = theta_k
        params['yura_weights'] = copy.deepcopy(data_['model'].layers_weight)
        params['yura_grad'] = copy.deepcopy(model_grad)
    elif params['algorithm'] in ['SGD']:
        1
    else:
        print('Error: unkown algo when yura')
        sys.exit()
    
    if params['algorithm'] in ['SGD-LRdecay']:
        
        print('error: should not reach here')
        
        sys.exit()
        
#         if params['epoch'] > 0 and params['epoch'] % params['num_epoch_to_decay'] == 0 and params['i'] % params['iter_per_epoch'] == 0:
#             params['alpha_current'] *= params['lr_decay_rate']
        params['alpha_current'] =\
    params['alpha'] *\
    (params['lr_decay_rate'] ** (params['epoch'] // params['num_epoch_to_decay']))
    elif params['algorithm'] in ['SGD']:
        pass
    else:
        print('params[algorithm]')
        print(params['algorithm'])
        sys.exit()


    data_['p_torch'] = p
    

    if true_algorithm in ['SGD-yura', 
                          'SGD-yura-MA', 
                          'SGD-momentum-yura',
                          'SGD-momentum',
                          'SGD-VA',
                          'SGD-signVA',
                          'SGD-signVAerf',
                          'SGD-signVAsqrt',
                          'SGD-sign']:
        params['algorithm'] = true_algorithm
    elif true_algorithm == 'SGD-yura-old':
        params['algorithm'] = true_algorithm
    
    
        
    return data_

def RMSprop_update(data_, params):
    
#     print('move to another file')
    
#     model_grad = data_['model_grad_used']
    model_grad = data_['model_grad_used_torch']

    algorithm = params['algorithm']
    
    if algorithm in ['Adam',
                     'Adam-noWarmStart']:
        beta_1 = params['momentum_gradient_rho']
        
        assert params['momentum_gradient_rho'] == params['momentum_gradient_dampening']
        
        i = params['i']
        
        model_grad = get_multiply_scalar(1 / (1 - beta_1**(i+1)), model_grad)
        
    elif algorithm in ['RMSprop',
                       'RMSprop-warmStart']:
        1
    else:
        print('error: check if bias correction for grad for ' + algorithm)
        sys.exit()

    if algorithm == 'RMSprop-individual-grad-no-sqrt-LM':
        epsilon = params['lambda_']
    elif algorithm in ['RMSprop-individual-grad-no-sqrt-Fisher',
                       'RMSprop-individual-grad-no-sqrt',
                       'RMSprop-individual-grad',
                       'RMSprop-no-sqrt',
                       'RMSprop',
                       'RMSprop-warmStart',
                       'RMSprop-test',
                       'Adam',
                       'Adam-test',
                       'Adam-noWarmStart']:
        epsilon = params['RMSprop_epsilon']
    else:
        print('Error: undefined epsilon.')
        sys.exit()
    
#     print('params.keys()')
#     print(params.keys())
    
#     print('RMSprop_beta_2 in params')
#     print('RMSprop_beta_2' in params)
    
#     print('params[RMSprop_beta_2]')
#     print(params['RMSprop_beta_2'])
    
#     sys.exit()
    
#     beta_2 = 0.9
    beta_2 = params['RMSprop_beta_2']
    
#     print('beta_2')
#     print(beta_2)
    
#     sys.exit()
        
    
    if algorithm in ['RMSprop',
                     'RMSprop-warmStart',
                     'RMSprop-test',
                     'Adam',
                     'Adam-test',
                     'Adam-noWarmStart',
                     'RMSprop-no-sqrt']:
        data_['RMSprop_momentum_2'] =\
        get_plus_torch(
            get_multiply_scalar(beta_2, data_['RMSprop_momentum_2']), 
            get_multiply_scalar(1-beta_2, get_square_torch(model_grad)))
    elif algorithm == 'RMSprop-individual-grad' or\
    algorithm == 'RMSprop-individual-grad-no-sqrt' or\
    algorithm == 'RMSprop-individual-grad-no-sqrt-Fisher' or\
    algorithm == 'RMSprop-individual-grad-no-sqrt-LM':
        a_grad_N2 = data_['a_grad_N2']
        h_N2 = data_['h_N2']

        model = data_['model']

        N2 = params['N2']

        for l in range(model.numlayers):
            if params['layers_params'][l]['name'] == 'fully-connected':

                h_l_square = torch.mul(h_N2[l], h_N2[l])
                a_grad_l_square = torch.mul(a_grad_N2[l], a_grad_N2[l]) # N2 * m_l

                W_l_square = torch.mm(h_l_square.t(), a_grad_l_square) / N2

                data_['RMSprop_momentum_2'][l]['W'] =\
                beta_2 * data_['RMSprop_momentum_2'][l]['W'] +\
                (1-beta_2) * W_l_square.t().cpu().data.numpy()

                data_['RMSprop_momentum_2'][l]['b'] =\
                beta_2 * data_['RMSprop_momentum_2'][l]['b'] +\
                (1-beta_2) * torch.mean(a_grad_l_square, dim=0).cpu().data.numpy()
            elif params['layers_params'][l]['name'] == 'conv':
                print('h_N2[l].size')
                print(h_N2[l].size())
                print('a_grad_N2[l].size')
                print(a_grad_N2[l].size())
                print('model_grad[l][W].shape()')
                print(model_grad[l]['W'].shape)

                h_N2_l_pad = F.pad(h_N2[l], (2,2,2,2))

                print('h_N2_l_pad.size()')
                print(h_N2_l_pad.size())

                # print('range(model_grad[l][W].size()[0])')
                # print(range(model_grad[l]['W'].size()[0]))

                for i in range(model_grad[l]['W'].shape[0]):
                    for j in range(model_grad[l]['W'].shape[1]):
                        for test_h in range(model_grad[l]['W'].shape[2]):
                            for test_w in range(model_grad[l]['W'].shape[3]):
                                print('i, j, test_h, test_w')
                                print(i, j, test_h, test_w)

                                # print('model_grad[l][W][i, j, test_h, test_w]')
                                # print(model_grad[l]['W'][i, j, test_h, test_w])

                                # print('torch.from_numpy(model_grad[l][W][i, j, test_h, test_w])')
                                # print(torch.from_numpy(model_grad[l]['W'][i, j, test_h, test_w]))

                                # print('torch.from_numpy(model_grad[l][W][i, j, test_h, test_w]).float()')
                                # print(torch.from_numpy(model_grad[l]['W'][i, j, test_h, test_w]).float())

                                # print('torch.from_numpy(model_grad[l][W][i, j, test_h, test_w]).float().cuda()')
                                # print(torch.from_numpy(model_grad[l]['W'][i, j, test_h, test_w]).float().cuda())

                                print('torch.sum(torch.mean(torch.mul(a_grad_N2[l][:, i], h_N2_l_pad[:, j, test_h: test_h+28, test_w: test_w+28]), dim=0)) -\
                                      torch.from_numpy(model_grad[l][W][i, j, test_h, test_w]).float().cuda()')
                                print(torch.sum(torch.mean(torch.mul(a_grad_N2[l][:, i], h_N2_l_pad[:, j, test_h: test_h+28, test_w: test_w+28]), dim=0)) -\
                                      model_grad[l]['W'][i, j, test_h, test_w])

                                # sys.exit()




                for i in range(len(h_N2[l])):
                    print('h_N2_l_pad[i].size()')
                    print(h_N2_l_pad[i].size())
                    print('a_grad_N2[l][i].size()')
                    print(a_grad_N2[l][i].size())

                    h_N2_l_pad_i_expand = torch.unsqueeze(h_N2_l_pad[i], 0)

                    print('h_N2_l_pad_i_expand.size()')
                    print(h_N2_l_pad_i_expand.size())

                    sys.exit()
            else:
                print('Error: unknown layer when update rmsprop')
                sys.exit()
    else:
        print('Error: unsupported algorithm.')
        sys.exit()
        
    if algorithm in ['Adam',
                     'Adam-test',
                     'Adam-noWarmStart']:
        
        i = params['i']
        
        model_grad_second_moment = get_multiply_scalar(1 / (1 - beta_2**(i+1)), data_['RMSprop_momentum_2'])
        
    elif algorithm in ['RMSprop',
                       'RMSprop-warmStart']:
        model_grad_second_moment = data_['RMSprop_momentum_2']
    else:
        print('error: check if bias correction for grad for ' + algorithm)
        sys.exit()
    
        


    if algorithm in ['RMSprop',
                     'RMSprop-warmStart',
                     'RMSprop-test',
                     'Adam',
                     'Adam-test',
                     'Adam-noWarmStart',
                     'RMSprop-individual-grad']:
        # epsilon = 10**(-8)

        # epsilon = 10**(-4)
        
        

        p = get_divide_torch(
            model_grad, 
            get_plus_scalar(epsilon, get_sqrt_torch(model_grad_second_moment)))
    elif algorithm == 'RMSprop-individual-grad-no-sqrt' or\
    algorithm == 'RMSprop-no-sqrt' or\
    algorithm == 'RMSprop-individual-grad-no-sqrt-Fisher' or\
    algorithm == 'RMSprop-individual-grad-no-sqrt-LM':

        # epsilon = 10**(-4)
        # print('test epsilon')

        
        p = get_divide(
            model_grad, 
            get_plus_scalar(epsilon, model_grad_second_moment))

    else:
        print('Error: unsupported algorithm 2.')
        sys.exit()
    
    p = get_opposite(p)
    
    

    


    data_['p_torch'] = p
    return data_

def update_parameter(p_torch, model, params):
    numlayers = params['numlayers']
    
#     alpha = params['alpha']
    alpha = params['alpha_current']
    
    device = params['device']

    
    for l in range(numlayers):
        
        for name_variable in model.layers_weight[l].keys():
            
            if params['weight_decay'] != 0:
                model.layers_weight[l][name_variable].data *= (1 - alpha*params['weight_decay'])
            
            model.layers_weight[l][name_variable].data += alpha * p_torch[l][name_variable].data
        
#         if params['layers_params'][l]['name'] in ['fully-connected',
#                                                   'conv',
#                                                   'conv-no-activation',
#                                                   '1d-conv',
#                                                   'BN']:

#             model.layers_weight[l]['W'].data += alpha * p_torch[l]['W'].data
#             model.layers_weight[l]['b'].data += alpha * p_torch[l]['b'].data

#         elif params['layers_params'][l]['name'] in ['conv-no-bias-no-activation']:

#             model.layers_weight[l]['W'].data += alpha * p_torch[l]['W'].data
#         else:
#             print('params[layers_params][l][name]')
#             print(params['layers_params'][l]['name'])
#             print('Error: layer not supported when update parameter')
#             sys.exit()
        
    return model

# input_data.py

"""Functions for downloading and reading MNIST data."""

import os
# import urllib
# import urllib.request
import numpy as np


import sys












    
    
def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot



    
def load_subsampled_imagenet(train_dir):
    
    train_path = train_dir + '/' + 'YiRen_imagenet_sample/train/'
    transform = transforms.Compose(
        [transforms.Resize((256,256)), transforms.ToTensor()]
    )

    # in case you want to normalize images
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                      std=[0.229, 0.224, 0.225])
    # transform = transforms.Compose(
    #     [transforms.Resize((256,256)), transforms.ToTensor(), normalize]
    # )

    imagenet_data = datasets.ImageFolder(train_path, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        imagenet_data,
#         batch_size=4373,
        batch_size=200,
        shuffle=True,
        num_workers=0
    )
    
    return data_loader
    








def get_post_activation(pre_, activation):
    if activation == 'relu':
        post_ = F.relu(pre_)
    elif activation == 'sigmoid':
        post_ = torch.sigmoid(pre_)
    elif activation == 'tanh':
        post_ = torch.tanh(pre_)
    elif activation == 'linear':
        post_ = pre_
    else:
        print('Error: unsupported activation for ' + activation)
        sys.exit()
    return post_

def get_layer_forward(input_, layer_, activation_, layer_params):
    if layer_params['name'] == 'fully-connected':
            
        a_ = layer_(input_)
        h_ = get_post_activation(a_, activation_)
        a_.retain_grad()
        
        output_ = h_
        pre_ = a_
        
    elif layer_params['name'] in ['conv',
                                  '1d-conv']:
        
        
        a_ = layer_(input_)
        
#         print('a_.size()')
#         print(a_.size())
        
#         print('torch.norm(a_)')
#         print(torch.norm(a_))
        
        
        
        
        
        
        h_ = get_post_activation(a_, activation_)
        

        a_.retain_grad()
        
        output_ = h_
        pre_ = a_
        
    elif layer_params['name'] in ['conv-no-activation',
                                  'conv-no-bias-no-activation']:
        a_ = layer_(input_)
        
#         print('a_.size()')
#         print(a_.size())
        
#         print('torch.norm(a_)')
#         print(torch.norm(a_))
        
        
        a_.retain_grad()
        output_ = a_
        pre_ = a_
        
    elif layer_params['name'] in ['BN']:
        a_ = layer_(input_)
        a_.retain_grad()
        output_ = a_
        pre_ = a_
        
    else:
        print('layer_params[name]')
        print(layer_params['name'])
        print('Error: unkown layer')
        sys.exit()
    

    return output_, pre_
    # return a_, h_

def get_layers_params(name_model, layersizes, activations, params):
    if name_model == 'fully-connected':
        layers_ = []
        for l in range(len(layersizes) - 1):
            layer_i = {}
            layer_i['name'] = 'fully-connected'
            layer_i['input_size'] = layersizes[l]
            layer_i['output_size'] = layersizes[l+1]
            layer_i['activation'] = activations[l]
            layers_.append(layer_i)
    elif name_model == 'simple-CNN':
        # https://arxiv.org/pdf/1910.05446.pdf
        # https://arxiv.org/pdf/1811.03600.pdf

        # "same" padding:
        # i.e. H_in = H_out
        # by https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        # H_out = H_in + 2 * padding - dilation * (kernel_size - 1)
        # (when stride = 1)
        # Hence, since dilation = 1, H_in = H_out => padding = (kernel_size - 1) / 2
        # this is also endorsed by https://discuss.pytorch.org/t/same-convolution-in-pytorch/19937
        
#         print('need to accomadate GAP')
#         sys.exit()

        layersizes = [32, 64, 1024]

        layers_ = []

        layer_1 = {}
        layer_1['name'] = 'conv'
        
        
        if params['name_dataset'] == 'Subsampled-ImageNet-simple-CNN':
            layer_1['conv_in_channels'] = 3
        elif params['name_dataset'] in ['Fashion-MNIST',
                                        'Fashion-MNIST-N1-60',
                                        'Fashion-MNIST-N1-60-no-regularization',
                                        'Fashion-MNIST-N1-256-no-regularization',
                                        'Fashion-MNIST-GAP-N1-60-no-regularization']:
            layer_1['conv_in_channels'] = 1
        else:
            print('error: need to check conv_in_channels for ' + params['name_dataset'])
            sys.exit()
            
        
        layer_1['conv_out_channels'] = layersizes[0]
        layer_1['conv_kernel_size'] = 5
        layer_1['conv_stride'] = 1
        layer_1['conv_padding'] = int((layer_1['conv_kernel_size'] - 1)/2)
        
        
#         layer_1['activation'] = activations[0]
        layer_1['activation'] = 'relu'
    
        layers_.append(layer_1)
        
        layer_1 = {}
        layer_1['name'] = 'max_pool'
        layer_1['max_pool_kernel_size'] = 2
        layer_1['max_pool_stride'] = 2
        
        layers_.append(layer_1)
        

        layer_2 = {}
        layer_2['name'] = 'conv'
        layer_2['conv_in_channels'] = layersizes[0]
        layer_2['conv_out_channels'] = layersizes[1]
        layer_2['conv_kernel_size'] = 5
        layer_2['conv_stride'] = 1
        layer_2['conv_padding'] = int((layer_2['conv_kernel_size'] - 1)/2)
        
#         layer_2['activation'] = activations[1]
        layer_2['activation'] = 'relu'
        
        layers_.append(layer_2)
        
        if params['name_dataset'] == 'Fashion-MNIST-GAP-N1-60-no-regularization':
            # https://teaching.pages.centralesupelec.fr/deeplearning-lectures-build/00-pytorch-fashionMnist.html
            
            layer_ = {}
            layer_['name'] = 'global_average_pooling'
#             layer_['max_pool_kernel_size'] = 2
#             layer_['max_pool_stride'] = 2

            layers_.append(layer_)
    
        elif params['name_dataset'] in ['Fashion-MNIST-N1-60-no-regularization',
                                        'Fashion-MNIST-N1-256-no-regularization']:
            
            layer_2 = {}
            layer_2['name'] = 'max_pool'
            layer_2['max_pool_kernel_size'] = 2
            layer_2['max_pool_stride'] = 2

            layers_.append(layer_2)


            layer_5 = {}
            layer_5['name'] = 'flatten'

            layers_.append(layer_5)
    
        else:
            print('error: need to check for ' + params['name_dataset'])
            sys.exit()
        
            
        

        layer_3 = {}
        layer_3['name'] = 'fully-connected'
        
        if params['name_dataset'] == 'Subsampled-ImageNet-simple-CNN':
            layer_3['input_size'] = 64 * 64 * layersizes[1]
        elif params['name_dataset'] in ['Fashion-MNIST',
                                        'Fashion-MNIST-N1-60',
                                        'Fashion-MNIST-N1-60-no-regularization',
                                        'Fashion-MNIST-N1-256-no-regularization']:
            layer_3['input_size'] = 7 * 7 * layersizes[1]
        elif params['name_dataset'] in ['Fashion-MNIST-GAP-N1-60-no-regularization']:
            layer_3['input_size'] = layersizes[1]
        else:
            print('error: need to check input_size for ' + params['name_dataset'])
            sys.exit()
            
        
        layer_3['output_size'] = layersizes[2]
        
        layer_3['activation'] = 'relu'
#         layer_3['activation'] = activations[2]

        layers_.append(layer_3)
        

        layer_4 = {}
        layer_4['name'] = 'fully-connected'
        layer_4['input_size'] = layer_3['output_size']
        
        if params['name_dataset'] == 'Subsampled-ImageNet-simple-CNN':
            layer_4['output_size'] = 200
        elif params['name_dataset'] in ['Fashion-MNIST',
                                        'Fashion-MNIST-N1-60',
                                        'Fashion-MNIST-N1-60-no-regularization',
                                        'Fashion-MNIST-N1-256-no-regularization',
                                        'Fashion-MNIST-GAP-N1-60-no-regularization']:
            layer_4['output_size'] = 10
        else:
            print('error: need to check output_size for ' + params['name_dataset'])
            sys.exit()
            
        
#         layer_4['activation'] = activations[3]
        layer_4['activation'] = 'linear'

        layers_.append(layer_4)
        
        
    elif name_model == 'CNN':
#         layersizes = [96, 192, 192, 10]
        layersizes = [96, 192, 192, 100]
        
        layers_ = []
        
        for l in range(3):
        
            layer_l = {}
            layer_l['name'] = 'conv'
            
            if l == 0:
                layer_l['conv_in_channels'] = 3
            else:
                layer_l['conv_in_channels'] = layersizes[l-1]
            
            
            layer_l['conv_out_channels'] = layersizes[l]
            
            layer_l['conv_kernel_size'] = 5
            layer_l['conv_stride'] = 1
            layer_l['conv_padding'] = int((layer_l['conv_kernel_size'] - 1)/2)
            
#             layer_l['activation'] = activations[l]
            layer_l['activation'] = 'relu'
    
            layers_.append(layer_l)
            
            layer_l = {}
            layer_l['name'] = 'max_pool'
            layer_l['max_pool_kernel_size'] = 2
            layer_l['max_pool_stride'] = 2
            layers_.append(layer_l)
            
        layer_5 = {}
        layer_5['name'] = 'flatten'
        layers_.append(layer_5)
            
        layer_4 = {}
        layer_4['name'] = 'fully-connected'
        layer_4['input_size'] = layersizes[2] * 4 * 4
        layer_4['output_size'] = layersizes[3]
        
#         layer_4['activation'] = activations[3]
        layer_4['activation'] = 'linear'
        
#         layer_4['if_flatten'] = True
        
        layers_.append(layer_4)
    elif name_model == '1d-CNN':
        # https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/
        
#         model = Sequential()

#         model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))

#         model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
#         model.add(Dropout(0.5))
#         model.add(MaxPooling1D(pool_size=2))

#         model.add(Flatten())
#         model.add(Dense(100, activation='relu'))

#         model.add(Dense(n_outputs, activation='softmax'))

#         model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        

        
        layersizes = [64, 64, 100, 6]
        
        layers_ = []
        
        for l in range(2):
        
            layer_l = {}
            layer_l['name'] = '1d-conv'
            
            if l == 0:
                layer_l['conv_in_channels'] = 1
            else:
                layer_l['conv_in_channels'] = layersizes[l-1]
            
            
            layer_l['conv_out_channels'] = layersizes[l]
            
            layer_l['conv_kernel_size'] = 3
            layer_l['conv_stride'] = 1
            layer_l['conv_padding'] = 0
            
#             if l == 0: 
#                 layer_l['if_max_pool'] = False
#             elif l == 1:
#                 layer_l['if_max_pool'] = True
#                 layer_l['max_pool_kernel_size'] = 2
#                 layer_l['max_pool_stride'] = 0
            
            
            layer_l['activation'] = activations[l]
            
            layers_.append(layer_l)
            
        layer_6 = {}
        layer_6['name'] = 'max_pool_1d'
        layer_6['max_pool_kernel_size'] = 2
        layer_6['max_pool_stride'] = 0
        layers_.append(layer_6)
            
        layer_5 = {}
        layer_5['name'] = 'flatten'
        layers_.append(layer_5)
            
        layer_3 = {}
        layer_3['name'] = 'fully-connected'
        layer_3['input_size'] = layersizes[1] * 278
        layer_3['output_size'] = layersizes[2]
        layer_3['activation'] = activations[2]
        
        layers_.append(layer_3)
        
        layer_4 = {}
        layer_4['name'] = 'fully-connected'
        layer_4['input_size'] = layersizes[2]
        layer_4['output_size'] = layersizes[3]
        layer_4['activation'] = activations[3]
#         layer_4['if_flatten'] = False
        
        layers_.append(layer_4)
        
    elif name_model == 'AllCNNC':
        # https://arxiv.org/pdf/1412.6806.pdf
        # https://github.com/mateuszbuda/ALL-CNN/blob/master/ALL_CNN_C.png
        
        # except that in the following, for the last 3 * 3 conv in Table 1,
        # we use padding = 1
        
        # see also:
        # 'CNN' in https://arxiv.org/pdf/1910.05446.pdf
        
        layers_ = []
        
        layers_ = add_conv_block(layers_, 3, 96, 3, 1, 1, params)
        layers_ = add_conv_block(layers_, 96, 96, 3, 1, 1, params)
        layers_ = add_conv_block(layers_, 96, 96, 3, 2, 1, params)
        
        layers_ = add_conv_block(layers_, 96, 192, 3, 1, 1, params)
        layers_ = add_conv_block(layers_, 192, 192, 3, 1, 1, params)
        layers_ = add_conv_block(layers_, 192, 192, 3, 2, 1, params)
        
        layers_ = add_conv_block(layers_, 192, 192, 3, 1, 1, params)
        
        layers_ = add_conv_block(layers_, 192, 192, 1, 1, 0, params)
        

        
        
        if params['name_dataset'] == 'CIFAR-100-onTheFly-AllCNNC':
            layers_ = add_conv_block(layers_, 192, 100, 1, 1, 0, params)
        elif params['name_dataset'] in ['CIFAR-10-AllCNNC',
                                        'CIFAR-10-N1-128-AllCNNC',
                                        'CIFAR-10-N1-512-AllCNNC']:
            layers_ = add_conv_block(layers_, 192, 10, 1, 1, 0, params)
        else:
            print('params[name_dataset]')
            print(params['name_dataset'])
            sys.exit()
        
            
        
        layer_ = {}
        layer_['name'] = 'global_average_pooling'
        layers_.append(layer_)
        
    elif name_model == 'ConvPoolCNNC':
        # https://arxiv.org/pdf/1412.6806.pdf
        
        layers_ = []
        
        layers_ = add_conv_block(layers_, 3, 96, params)
        
        layers_ = add_conv_block(layers_, 96, 96, params)
        
        layers_ = add_conv_block(layers_, 96, 96, params)
        
        
        
        sys.exit()
        
    elif name_model in ['ResNet32']:
        
        # ResNet 32 in sec 4.2 and Table 6 of https://arxiv.org/pdf/1512.03385.pdf
        
        # see also:
        # https://pytorch.org/vision/stable/models.html
        
        # see also:
        # https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
        
        # see also:
        # https://github.com/km1414/CNN-models/blob/master/resnet-32/resnet-32.py
        # (this one seems to have bias)
        
        # in the NoBias mode, conv layers don't have bias, including the first conv
        # see e.g. https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html#resnet34
        # (this webpage is no longer available)
        
        
        
        
        if params['name_dataset'] in ['CIFAR-10-onTheFly-ResNet32-BN',
                                      'CIFAR-10-onTheFly-ResNet32-BN-BNshortcut',
                                      'CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly',
                                      'CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly-NoBias',
                                      'CIFAR-10-onTheFly-N1-128-ResNet32-BN-BNshortcutDownsampleOnly-NoBias',
                                      'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',
                                      'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias-no-regularization',
                                      'CIFAR-100-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',]:
            if_BNNoAffine = False
        elif params['name_dataset'] in ['CIFAR-10-onTheFly-ResNet32-BNNoAffine',
                                        'CIFAR-10-onTheFly-ResNet32-BNNoAffine-NoBias',
                                        'CIFAR-10-onTheFly-N1-128-ResNet32-BNNoAffine-PaddingShortcutDownsampleOnly-NoBias-no-regularization',]:
            if_BNNoAffine = True
        else:
            print('params[name_dataset]')
            print(params['name_dataset'])
            sys.exit()
            
        if params['name_dataset'] in ['CIFAR-10-onTheFly-N1-128-ResNet32-BNNoAffine-PaddingShortcutDownsampleOnly-NoBias-no-regularization',
                                      'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',
                                      'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias-no-regularization',
                                      'CIFAR-100-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',]:
            shortcut_type = 'padding'
            if_BN_shortcut = None
        elif params['name_dataset'] in ['CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly-NoBias',
                                        'CIFAR-10-onTheFly-N1-128-ResNet32-BN-BNshortcutDownsampleOnly-NoBias',]:
            shortcut_type = 'conv'
            if_BN_shortcut = True
        elif params['name_dataset'] in ['CIFAR-10-onTheFly-ResNet32-BNNoAffine-NoBias',]:
            shortcut_type = 'conv'
            if_BN_shortcut = False
        else:
            print('params[name_dataset]')
            print(params['name_dataset'])
            sys.exit()
            
            if params['name_dataset'] in ['CIFAR-10-onTheFly-ResNet32-BN-BNshortcut',
                                          'CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly',]:
                if_BN_shortcut = True
            elif params['name_dataset'] in ['CIFAR-10-onTheFly-ResNet32-BNNoAffine',
                                            'CIFAR-10-onTheFly-ResNet32-BN']:
                if_BN_shortcut = False
            else:
                print('params[name_dataset]')
                print(params['name_dataset'])

                sys.exit()
            
        if params['name_dataset'] in ['CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly',
                                      'CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly-NoBias',
                                      'CIFAR-10-onTheFly-N1-128-ResNet32-BNNoAffine-PaddingShortcutDownsampleOnly-NoBias-no-regularization',
                                      'CIFAR-10-onTheFly-N1-128-ResNet32-BN-BNshortcutDownsampleOnly-NoBias',
                                      'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',
                                      'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias-no-regularization',
                                      'CIFAR-100-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',]:
            if_downsample_only = True
        elif params['name_dataset'] in ['CIFAR-10-onTheFly-ResNet32-BNNoAffine-NoBias',]:
            if_downsample_only = False
        else:
            print('params[name_dataset]')
            print(params['name_dataset'])

            sys.exit()
            
            
        if params['name_dataset'] in ['CIFAR-10-onTheFly-ResNet32-BNNoAffine-NoBias',
                                      'CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly-NoBias',
                                      'CIFAR-10-onTheFly-N1-128-ResNet32-BNNoAffine-PaddingShortcutDownsampleOnly-NoBias-no-regularization',
                                      'CIFAR-10-onTheFly-N1-128-ResNet32-BN-BNshortcutDownsampleOnly-NoBias',
                                      'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',
                                      'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias-no-regularization',
                                      'CIFAR-100-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',]:
            if_conv_bias = False
        elif params['name_dataset'] in ['CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly']:
            if_conv_bias = True
        else:
            print('params[name_dataset]')
            print(params['name_dataset'])

            sys.exit()
            
        

        
        
        layers_ = []
        
        layers_ = add_conv_block(layers_, 3, 16, 3, 1, 1, params)
        
        layers_ = add_res_block(layers_, 16, 16, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 16, 16, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 16, 16, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 16, 16, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 16, 16, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        
        layers_ = add_res_block(layers_, 16, 32, 2, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 32, 32, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 32, 32, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 32, 32, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 32, 32, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        
        layers_ = add_res_block(layers_, 32, 64, 2, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 64, 64, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 64, 64, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 64, 64, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 64, 64, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        
        layer_ = {}
        layer_['name'] = 'global_average_pooling'
        layers_.append(layer_)
        
        layer_ = {}
        layer_['name'] = 'fully-connected'
        layer_['input_size'] = 64
        
        if params['name_dataset'] in ['CIFAR-100-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',]:
            layer_['output_size'] = 100
        elif params['name_dataset'] in ['CIFAR-10-onTheFly-N1-128-ResNet32-BNNoAffine-PaddingShortcutDownsampleOnly-NoBias-no-regularization',
                                        'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',
                                        'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias-no-regularization',]:
            layer_['output_size'] = 10
        else:
            print('params[name_dataset]')
            print(params['name_dataset'])
            sys.exit()
        
            
        
        layer_['activation'] = 'linear'
        layers_.append(layer_)
        
    elif name_model in ['ResNet34']:
        
        # ResNet 34 in Fig 3 of https://arxiv.org/pdf/1512.03385.pdf
        
        # see also
        # https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/resnet.py
        
        if params['name_dataset'] == 'CIFAR-100-onTheFly-ResNet34-BNNoAffine':
            if_BNNoAffine = True
            shortcut_type = 'conv'
            if_BN_shortcut = False
            if_downsample_only = False
            if_conv_bias = True
        elif params['name_dataset'] == 'CIFAR-100-onTheFly-ResNet34-BN':
            if_BNNoAffine = False
            shortcut_type = 'conv'
            if_BN_shortcut = False
            if_downsample_only = False
            if_conv_bias = True
        elif params['name_dataset'] == 'CIFAR-100-onTheFly-ResNet34-BN-BNshortcut':
            if_BNNoAffine = False
            shortcut_type = 'conv'
            if_BN_shortcut = True
            if_downsample_only = False
            if_conv_bias = True
        elif params['name_dataset'] == 'CIFAR-100-onTheFly-ResNet34-BN-BNshortcutDownsampleOnly':
            if_BNNoAffine = False
            shortcut_type = 'conv'
            if_BN_shortcut = True
            if_downsample_only = True
            if_conv_bias = True
        elif params['name_dataset'] in ['CIFAR-100-onTheFly-ResNet34-BN-BNshortcutDownsampleOnly-NoBias',
                                        'CIFAR-100-onTheFly-N1-128-ResNet34-BN-BNshortcutDownsampleOnly-NoBias',]:
            if_BNNoAffine = False
            shortcut_type = 'conv'
            if_BN_shortcut = True
            if_downsample_only = True
            if_conv_bias = False
        elif params['name_dataset'] == 'CIFAR-100-onTheFly-N1-128-ResNet34-BN-PaddingShortcutDownsampleOnly-NoBias':
            if_BNNoAffine = False
            shortcut_type = 'padding'
            if_BN_shortcut = None
            if_downsample_only = True
            if_conv_bias = False
        else:
            print('params[name_dataset]')
            print(params['name_dataset'])
            sys.exit()
        
        
        layers_ = []
        
        layers_ = add_conv_block(layers_, 3, 64, 3, 1, 1, params)
        
        
        
        layers_ = add_res_block(layers_, 64, 64, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 64, 64, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 64, 64, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        
        layers_ = add_res_block(layers_, 64, 128, 2, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 128, 128, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 128, 128, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 128, 128, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        
        layers_ = add_res_block(layers_, 128, 256, 2, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 256, 256, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 256, 256, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 256, 256, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 256, 256, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 256, 256, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        
        layers_ = add_res_block(layers_, 256, 512, 2, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 512, 512, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 512, 512, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        
        
        
        layer_ = {}
        layer_['name'] = 'global_average_pooling'
        layers_.append(layer_)
        
#         sys.exit()
        
        layer_ = {}
        layer_['name'] = 'fully-connected'
        layer_['input_size'] = 512
        layer_['output_size'] = 100
        layer_['activation'] = 'linear'
        layers_.append(layer_)
    
    elif name_model == 'vgg16':
        # referece:
        # import torchvision.models as models
        # vgg16 = models.vgg16()
        # print(vgg16.eval())
        
        # model D in https://arxiv.org/pdf/1409.1556.pdf
        
        # when use BN (or BNNoAffine), bias term is NOT omitted in conv
        # see e.g. https://pytorch.org/docs/stable/_modules/torchvision/models/vgg.html#vgg16_bn
        # see also https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py
        
        
        if params['name_dataset'] == 'Subsampled-ImageNet-vgg16':
            print('error: need to check all below')
            sys.exit()
            
        if params['name_dataset'] == 'CIFAR-10-vgg16':
            print('error: not supported anymore')
            sys.exit()
            
            
        
        
        if params['name_dataset'] == 'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPool':
            print('error: not supported anymore')
            sys.exit()
        
        
        
        layers_ = []
        
        layers_ = add_conv_block(layers_, 3, 64, 3, 1, 1, params)
        
        layers_ = add_conv_block(layers_, 64, 64, 3, 1, 1, params)
        
#     (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        layer_ = {}
        layer_['name'] = 'max_pool'
        layer_['max_pool_kernel_size'] = 2
        layer_['max_pool_stride'] = 2
        layers_.append(layer_)
        
        layers_ = add_conv_block(layers_, 64, 128, 3, 1, 1, params)
        
        layers_ = add_conv_block(layers_, 128, 128, 3, 1, 1, params)
        
#         sys.exit()
        
#     (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        layer_ = {}
        layer_['name'] = 'max_pool'
        layer_['max_pool_kernel_size'] = 2
        layer_['max_pool_stride'] = 2
        layers_.append(layer_)
        
        layers_ = add_conv_block(layers_, 128, 256, 3, 1, 1, params)
        
        layers_ = add_conv_block(layers_, 256, 256, 3, 1, 1, params)
        
        layers_ = add_conv_block(layers_, 256, 256, 3, 1, 1, params)
        
#         sys.exit()
        
#     (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        layer_ = {}
        layer_['name'] = 'max_pool'
        layer_['max_pool_kernel_size'] = 2
        layer_['max_pool_stride'] = 2
        layers_.append(layer_)

#         layer_2 = {}
#         layer_2['name'] = 'conv-no-activation'
#         layer_2['conv_in_channels'] = 256
#         layer_2['conv_out_channels'] = 512
#         layer_2['conv_kernel_size'] = 3
#         layer_2['conv_stride'] = 1
#         layer_2['conv_padding'] = 1
#         layer_2['activation'] = None
#         layers_.append(layer_2)
        
#         layer_2 = {}
#         layer_2['name'] = 'relu'
#         layers_.append(layer_2)
        
        layers_ = add_conv_block(layers_, 256, 512, 3, 1, 1, params)
        
#         sys.exit()

#         layer_2 = {}
#         layer_2['name'] = 'conv-no-activation'
#         layer_2['conv_in_channels'] = 512
#         layer_2['conv_out_channels'] = 512
#         layer_2['conv_kernel_size'] = 3
#         layer_2['conv_stride'] = 1
#         layer_2['conv_padding'] = 1
#         layer_2['activation'] = None
#         layers_.append(layer_2)
        
#         layer_2 = {}
#         layer_2['name'] = 'relu'
#         layers_.append(layer_2)
        
        layers_ = add_conv_block(layers_, 512, 512, 3, 1, 1, params)
        
#         sys.exit()

#         layer_2 = {}
#         layer_2['name'] = 'conv-no-activation'
#         layer_2['conv_in_channels'] = 512
#         layer_2['conv_out_channels'] = 512
#         layer_2['conv_kernel_size'] = 3
#         layer_2['conv_stride'] = 1
#         layer_2['conv_padding'] = 1
#         layer_2['activation'] = None
#         layers_.append(layer_2)
        
#         layer_2 = {}
#         layer_2['name'] = 'relu'
#         layers_.append(layer_2)
        
        layers_ = add_conv_block(layers_, 512, 512, 3, 1, 1, params)
        
#         sys.exit()
        
#     (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        layer_ = {}
        layer_['name'] = 'max_pool'
        layer_['max_pool_kernel_size'] = 2
        layer_['max_pool_stride'] = 2
        layers_.append(layer_)

#         layer_2 = {}
#         layer_2['name'] = 'conv-no-activation'
#         layer_2['conv_in_channels'] = 512
#         layer_2['conv_out_channels'] = 512
#         layer_2['conv_kernel_size'] = 3
#         layer_2['conv_stride'] = 1
#         layer_2['conv_padding'] = 1
#         layer_2['activation'] = None
#         layers_.append(layer_2)
        
#         layer_2 = {}
#         layer_2['name'] = 'relu'
#         layers_.append(layer_2)
        
        layers_ = add_conv_block(layers_, 512, 512, 3, 1, 1, params)
        
#         sys.exit()
        
#         layer_2 = {}
#         layer_2['name'] = 'conv-no-activation'
#         layer_2['conv_in_channels'] = 512
#         layer_2['conv_out_channels'] = 512
#         layer_2['conv_kernel_size'] = 3
#         layer_2['conv_stride'] = 1
#         layer_2['conv_padding'] = 1
#         layer_2['activation'] = None
#         layers_.append(layer_2)
        
#         layer_2 = {}
#         layer_2['name'] = 'relu'
#         layers_.append(layer_2)
        
        layers_ = add_conv_block(layers_, 512, 512, 3, 1, 1, params)
        
        layers_ = add_conv_block(layers_, 512, 512, 3, 1, 1, params)
        
#         sys.exit()
        
#     (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        layer_ = {}
        layer_['name'] = 'max_pool'
        layer_['max_pool_kernel_size'] = 2
        layer_['max_pool_stride'] = 2
        layers_.append(layer_)
        
#         if params['name_dataset'] in ['CIFAR-10-vgg16-NoAdaptiveAvgPoolNoDropout',
#                                       'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
#                                       'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
#                                       'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPool',
#                                       'CIFAR-10-onTheFly-N1-512-vgg16-NoAdaptiveAvgPoolNoDropout',
#                                       'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
#                                       'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-NoBias',
#                                       'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',
#                                       'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout',
#                                       'CIFAR-10-NoAugmentation-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
#                                       'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
#                                       'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',
#                                       'CIFAR-100-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout',
#                                       'CIFAR-100-NoAugmentation-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
#                                       'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
#                                       'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine-no-regularization',
#                                       'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
#                                       'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-no-regularization',
#                                       'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
#                                       'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine']:
#             1
#         elif params['name_dataset'] == 'CIFAR-10-vgg16':
#             layer_ = {}
#             layer_['name'] = 'AdaptiveAvgPool2d'
#             layer_['AdaptiveAvgPool2d_output_size'] = (7,7)
#             layers_.append(layer_)
#         else:
#             print('error: need to check for ' + params['name_dataset'])
#             sys.exit()
  
    

        
        layer_5 = {}
        layer_5['name'] = 'flatten'
        layers_.append(layer_5)
        

        
        if params['name_dataset'] == 'CIFAR-100-onTheFly-vgg16-NoLinear-BN-no-regularization':
            layer_ = {}
            layer_['name'] = 'fully-connected'
            layer_['input_size'] = 512
            layer_['output_size'] = 100
            layer_['activation'] = 'linear'
            layers_.append(layer_)
        elif params['name_dataset'] == 'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-no-regularization':
        
        

            layer_ = {}
            layer_['name'] = 'fully-connected'

    #         if params['name_dataset'] in ['CIFAR-10-vgg16-NoAdaptiveAvgPoolNoDropout',
    #                                       'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
    #                                       'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
    #                                       'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPool',
    #                                       'CIFAR-10-onTheFly-N1-512-vgg16-NoAdaptiveAvgPoolNoDropout',
    #                                       'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
    #                                       'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-NoBias',
    #                                       'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',
    #                                       'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout',
    #                                       'CIFAR-10-NoAugmentation-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
    #                                       'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
    #                                       'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',
    #                                       'CIFAR-100-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout',
    #                                       'CIFAR-100-NoAugmentation-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
    #                                       'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
    #                                       'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine-no-regularization',
    #                                       'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
    #                                       'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-no-regularization',
    #                                       'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
    #                                       'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine']:
    #             layer_['input_size'] = 512
    #         elif params['name_dataset'] == 'CIFAR-10-vgg16':
    #             layer_['input_size'] = 25088
    #         else:
    #             print('error: need to check for ' + params['name_dataset'])
    #             sys.exit()

            layer_['input_size'] = 512
            layer_['output_size'] = 4096
            layer_['activation'] = 'relu'
            layers_.append(layer_)

    #         if params['name_dataset'] in ['CIFAR-10-vgg16-NoAdaptiveAvgPoolNoDropout',
    #                                       'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
    #                                       'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
    #                                       'CIFAR-10-onTheFly-N1-512-vgg16-NoAdaptiveAvgPoolNoDropout',
    #                                       'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
    #                                       'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-NoBias',
    #                                       'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',
    #                                       'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout',
    #                                       'CIFAR-10-NoAugmentation-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
    #                                       'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
    #                                       'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',
    #                                       'CIFAR-100-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout',
    #                                       'CIFAR-100-NoAugmentation-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
    #                                       'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
    #                                       'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine-no-regularization',
    #                                       'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
    #                                       'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-no-regularization',
    #                                       'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
    #                                       'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine']:
    #             1
    #         elif params['name_dataset'] in ['CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPool']:
    #             layer_ = {}
    #             layer_['name'] = 'dropout'
    #             layer_['dropout_p'] = 0.5
    #             layers_.append(layer_)

    #         else:
    #             print('error: need to check for ' + params['name_dataset'])
    #             sys.exit()



            layer_ = {}
            layer_['name'] = 'fully-connected'
            layer_['input_size'] = 4096
            layer_['output_size'] = 4096
            layer_['activation'] = 'relu'
            layers_.append(layer_)

    #         if params['name_dataset'] in ['CIFAR-10-vgg16-NoAdaptiveAvgPoolNoDropout',
    #                                       'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
    #                                       'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
    #                                       'CIFAR-10-onTheFly-N1-512-vgg16-NoAdaptiveAvgPoolNoDropout',
    #                                       'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
    #                                       'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-NoBias',
    #                                       'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',
    #                                       'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout',
    #                                       'CIFAR-10-NoAugmentation-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
    #                                       'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
    #                                       'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',
    #                                       'CIFAR-100-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout',
    #                                       'CIFAR-100-NoAugmentation-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
    #                                       'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
    #                                       'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine-no-regularization',
    #                                       'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
    #                                       'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-no-regularization',
    #                                       'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
    #                                       'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine']:
    #             1
    #         elif params['name_dataset'] in ['CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPool']:
    #             layer_ = {}
    #             layer_['name'] = 'dropout'
    #             layer_['dropout_p'] = 0.5
    #             layers_.append(layer_)
    #         else:
    #             print('error: need to check for ' + params['name_dataset'])
    #             sys.exit()




            layer_ = {}
            layer_['name'] = 'fully-connected'
            layer_['input_size'] = 4096

            if params['name_dataset'] in ['CIFAR-10-vgg16-NoAdaptiveAvgPoolNoDropout',
                                          'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
                                          'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
                                          'CIFAR-10-onTheFly-N1-512-vgg16-NoAdaptiveAvgPoolNoDropout',
                                          'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
                                          'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-NoBias',
                                          'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',
                                          'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout',
                                          'CIFAR-10-NoAugmentation-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
                                          'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
                                          'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',]:
                layer_['output_size'] = 10
            elif params['name_dataset'] in ['CIFAR-100-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout',
                                            'CIFAR-100-NoAugmentation-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
                                            'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
                                            'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine-no-regularization',
                                            'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
                                            'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-no-regularization',
                                            'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
                                            'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine']:
                layer_['output_size'] = 100
            else:
                print('error: need to check for ' + params['name_dataset'])
                sys.exit()
                layer_['output_size'] = 1000


            layer_['activation'] = 'linear'
            layers_.append(layer_)
            
        else:
            print('params[name_dataset]')
            print(params['name_dataset'])
            sys.exit()
        
    elif name_model in ['vgg11']:
        
        
        
        # referece:
        # https://arxiv.org/pdf/1409.1556.pdf
        # Table 1, model A
        
        
        
        layers_ = []
        
        
        
        layer_2 = {}
        layer_2['name'] = 'conv'
        layer_2['conv_in_channels'] = 3
        layer_2['conv_out_channels'] = 64
        layer_2['conv_kernel_size'] = 3
        layer_2['conv_stride'] = 1
        layer_2['conv_padding'] = 1
        layer_2['activation'] = 'relu'
        layers_.append(layer_2)
        
        layer_ = {}
        layer_['name'] = 'max_pool'
        layer_['max_pool_kernel_size'] = 2
        layer_['max_pool_stride'] = 2
        layers_.append(layer_)
        
        # working here
        

        layer_2 = {}
        layer_2['name'] = 'conv'
        layer_2['conv_in_channels'] = 64
        layer_2['conv_out_channels'] = 128
        layer_2['conv_kernel_size'] = 3
        layer_2['conv_stride'] = 1
        layer_2['conv_padding'] = 1
        layer_2['activation'] = 'relu'
        layers_.append(layer_2)
        

        layer_ = {}
        layer_['name'] = 'max_pool'
        layer_['max_pool_kernel_size'] = 2
        layer_['max_pool_stride'] = 2
        layers_.append(layer_)
        
        # working here
        

        layer_2 = {}
        layer_2['name'] = 'conv'
        layer_2['conv_in_channels'] = 128
        layer_2['conv_out_channels'] = 256
        layer_2['conv_kernel_size'] = 3
        layer_2['conv_stride'] = 1
        layer_2['conv_padding'] = 1
        layer_2['activation'] = 'relu'
        layers_.append(layer_2)
        

        layer_2 = {}
        layer_2['name'] = 'conv'
        layer_2['conv_in_channels'] = 256
        layer_2['conv_out_channels'] = 256
        layer_2['conv_kernel_size'] = 3
        layer_2['conv_stride'] = 1
        layer_2['conv_padding'] = 1
        layer_2['activation'] = 'relu'
        layers_.append(layer_2)
        

        layer_ = {}
        layer_['name'] = 'max_pool'
        layer_['max_pool_kernel_size'] = 2
        layer_['max_pool_stride'] = 2
        layers_.append(layer_)
        
        # working here
        
        
        

        layer_2 = {}
        layer_2['name'] = 'conv'
        layer_2['conv_in_channels'] = 256
        layer_2['conv_out_channels'] = 512
        layer_2['conv_kernel_size'] = 3
        layer_2['conv_stride'] = 1
        layer_2['conv_padding'] = 1
        layer_2['activation'] = 'relu'
        layers_.append(layer_2)
        

        layer_2 = {}
        layer_2['name'] = 'conv'
        layer_2['conv_in_channels'] = 512
        layer_2['conv_out_channels'] = 512
        layer_2['conv_kernel_size'] = 3
        layer_2['conv_stride'] = 1
        layer_2['conv_padding'] = 1
        layer_2['activation'] = 'relu'
        layers_.append(layer_2)
        

        layer_ = {}
        layer_['name'] = 'max_pool'
        layer_['max_pool_kernel_size'] = 2
        layer_['max_pool_stride'] = 2
        layers_.append(layer_)
        
        # start to not work here
        
        

        layer_2 = {}
        layer_2['name'] = 'conv'
        layer_2['conv_in_channels'] = 512
        layer_2['conv_out_channels'] = 512
        layer_2['conv_kernel_size'] = 3
        layer_2['conv_stride'] = 1
        layer_2['conv_padding'] = 1
        layer_2['activation'] = 'relu'
        layers_.append(layer_2)
        

        layer_2 = {}
        layer_2['name'] = 'conv'
        layer_2['conv_in_channels'] = 512
        layer_2['conv_out_channels'] = 512
        layer_2['conv_kernel_size'] = 3
        layer_2['conv_stride'] = 1
        layer_2['conv_padding'] = 1
        layer_2['activation'] = 'relu'
        layers_.append(layer_2)
        

        layer_ = {}
        layer_['name'] = 'max_pool'
        layer_['max_pool_kernel_size'] = 2
        layer_['max_pool_stride'] = 2
        layers_.append(layer_)
        

        
#         if params['name_dataset'] == 'CIFAR-10-vgg16-NoAdaptiveAvgPoolNoDropout':
#             1
#         elif params['name_dataset'] == 'CIFAR-10-vgg16':
#             layer_ = {}
#             layer_['name'] = 'AdaptiveAvgPool2d'
#             layer_['AdaptiveAvgPool2d_output_size'] = (7,7)
#             layers_.append(layer_)
#         else:
#             print('error: need to check for ' + params['name_dataset'])
#             sys.exit()
  
    
        
        
        layer_5 = {}
        layer_5['name'] = 'flatten'
        layers_.append(layer_5)
        
#         sys.exit()

#         layer_ = {}
#         layer_['name'] = 'dropout'
#         layer_['dropout_p'] = 0.5
#         layers_.append(layer_)
        

        layer_ = {}
        layer_['name'] = 'fully-connected'
        layer_['input_size'] = 512
#         layer_['input_size'] = 3072
#         layer_['input_size'] = 16384
#         layer_['input_size'] = 8192
#         layer_['input_size'] = 4096
#         layer_['input_size'] = 2048
        layer_['output_size'] = 4096
#         layer_['output_size'] = 512
        layer_['activation'] = 'relu'
        layers_.append(layer_)
        
#         if params['name_dataset'] == 'CIFAR-10-vgg16-NoAdaptiveAvgPoolNoDropout':
#             1
#         elif params['name_dataset'] == 'CIFAR-10-vgg16':
#             layer_ = {}
#             layer_['name'] = 'dropout'
#             layer_['dropout_p'] = 0.5
#             layers_.append(layer_)
            
#         else:
#             print('error: need to check')
#             sys.exit()

#         layer_ = {}
#         layer_['name'] = 'dropout'
#         layer_['dropout_p'] = 0.5
#         layers_.append(layer_)

        layer_ = {}
        layer_['name'] = 'fully-connected'
        layer_['input_size'] = 4096
#         layer_['input_size'] = 512
        layer_['output_size'] = 4096
#         layer_['output_size'] = 512
        layer_['activation'] = 'relu'
        layers_.append(layer_)
        
#         if params['name_dataset'] == 'CIFAR-10-vgg16-NoAdaptiveAvgPoolNoDropout':
#             1
#         elif params['name_dataset'] == 'CIFAR-10-vgg16':
#             layer_ = {}
#             layer_['name'] = 'dropout'
#             layer_['dropout_p'] = 0.5
#             layers_.append(layer_)
#         else:
#             print('error: need to check')
#             sys.exit()
        


        layer_ = {}
        layer_['name'] = 'fully-connected'
        layer_['input_size'] = 4096
#         layer_['input_size'] = 512
        layer_['output_size'] = 10
        layer_['activation'] = 'linear'
        layers_.append(layer_)
        
        
    
        
        
  

    
    else:
        print('Error: unknown model name in get_layers_params for ' + name_model)
        sys.exit()
    return layers_



class Model_3(nn.Module):
    def __init__(self, params):

        name_dataset = params['name_dataset']


        super(Model_3, self).__init__()
    
        self.name_loss = params['name_loss']

        if name_dataset in ['MNIST-autoencoder',
                            'MNIST-autoencoder-no-regularization',
                            'MNIST-autoencoder-N1-1000',
                            'MNIST-autoencoder-N1-1000-no-regularization',
                            'MNIST-autoencoder-N1-1000-sum-loss-no-regularization',
                            'MNIST-autoencoder-relu-N1-1000-sum-loss-no-regularization',
                            'MNIST-autoencoder-relu-N1-1000-sum-loss',
                            'MNIST-autoencoder-relu-N1-100-sum-loss',
                            'MNIST-autoencoder-relu-N1-500-sum-loss',
                            'MNIST-autoencoder-relu-N1-1-sum-loss',
                            'MNIST-autoencoder-reluAll-N1-1-sum-loss',
                            'MNIST-autoencoder-N1-1000-sum-loss',
                            'CURVES-autoencoder',
                            'CURVES-autoencoder-no-regularization',
                            'CURVES-autoencoder-sum-loss-no-regularization',
                            'CURVES-autoencoder-sum-loss',
                            'CURVES-autoencoder-relu-sum-loss-no-regularization',
                            'CURVES-autoencoder-relu-sum-loss',
                            'CURVES-autoencoder-relu-N1-100-sum-loss',
                            'CURVES-autoencoder-relu-N1-500-sum-loss',
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
                            'FacesMartens-autoencoder-relu-no-regularization',
                            'FacesMartens-autoencoder-relu-N1-500',
                            'FacesMartens-autoencoder-relu-N1-100',
                            'MNIST',
                            'MNIST-no-regularization',
                            'MNIST-N1-1000',
                            'MNIST-one-layer',
                            'DownScaledMNIST-no-regularization',
                            'DownScaledMNIST-N1-1000-no-regularization',
                            'webspam',
                            'CIFAR',
                            'CIFAR-deep',
                            'sythetic-linear-regression',
                            'sythetic-linear-regression-N1-1']:
            
            self.name_model = 'fully-connected'
            
        elif name_dataset in ['Fashion-MNIST',
                              'Fashion-MNIST-N1-60',
                              'Fashion-MNIST-N1-60-no-regularization',
                              'Fashion-MNIST-N1-256-no-regularization',
                              'Fashion-MNIST-GAP-N1-60-no-regularization',
                              'STL-10-simple-CNN',
                              'Subsampled-ImageNet-simple-CNN']:
            # https://arxiv.org/pdf/1910.05446.pdf
            self.name_model = 'simple-CNN'
        elif name_dataset in ['CIFAR-100',
                              'CIFAR-100-NoAugmentation']:
            self.name_model = 'CNN'
        elif name_dataset in ['CIFAR-10-AllCNNC',
                              'CIFAR-10-N1-128-AllCNNC',
                              'CIFAR-10-N1-512-AllCNNC',
                              'CIFAR-100-onTheFly-AllCNNC']:
            self.name_model = 'AllCNNC'
        elif name_dataset == 'CIFAR-10-ConvPoolCNNC':
            self.name_model = 'ConvPoolCNNC'
        elif name_dataset == 'UCI-HAR':
            self.name_model = '1d-CNN'
        elif name_dataset in ['CIFAR-10-vgg16',
                              'CIFAR-10-vgg16-NoAdaptiveAvgPoolNoDropout',
                              'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
                              'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
                              'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPool',
                              'CIFAR-10-onTheFly-N1-512-vgg16-NoAdaptiveAvgPoolNoDropout',
                              'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
                              'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-NoBias',
                              'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',
                              'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout',
                              'CIFAR-10-NoAugmentation-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
                              'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
                              'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',
                              'CIFAR-10-vgg16-GAP',
                              'CIFAR-100-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout',
                              'CIFAR-100-NoAugmentation-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
                              'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
                              'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine-no-regularization',
                              'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
                              'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-no-regularization',
                              'CIFAR-100-onTheFly-vgg16-NoLinear-BN-no-regularization',
                              'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
                              'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',
                              'Subsampled-ImageNet-vgg16',]:
            self.name_model = 'vgg16'
        elif name_dataset in ['CIFAR-10-vgg11',
                              'CIFAR-10-NoAugmentation-vgg11']:
            self.name_model = 'vgg11'
        elif name_dataset in ['CIFAR-10-onTheFly-ResNet32-BNNoAffine',
                              'CIFAR-10-onTheFly-ResNet32-BN',
                              'CIFAR-10-onTheFly-ResNet32-BN-BNshortcut',
                              'CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly',
                              'CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly-NoBias',
                              'CIFAR-10-onTheFly-N1-128-ResNet32-BNNoAffine-PaddingShortcutDownsampleOnly-NoBias-no-regularization',
                              'CIFAR-10-onTheFly-N1-128-ResNet32-BN-BNshortcutDownsampleOnly-NoBias',
                              'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',
                              'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias-no-regularization',
                              'CIFAR-10-onTheFly-ResNet32-BNNoAffine-NoBias',
                              'CIFAR-100-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',]:
            self.name_model = 'ResNet32'
        elif name_dataset in ['CIFAR-100-onTheFly-ResNet34-BNNoAffine',
                              'CIFAR-100-onTheFly-ResNet34-BN',
                              'CIFAR-100-onTheFly-ResNet34-BN-BNshortcut',
                              'CIFAR-100-onTheFly-ResNet34-BN-BNshortcutDownsampleOnly',
                              'CIFAR-100-onTheFly-ResNet34-BN-BNshortcutDownsampleOnly-NoBias',
                              'CIFAR-100-onTheFly-N1-128-ResNet34-BN-BNshortcutDownsampleOnly-NoBias',
                              'CIFAR-100-onTheFly-N1-128-ResNet34-BN-PaddingShortcutDownsampleOnly-NoBias',]:
            self.name_model = 'ResNet34'
        else:
            print('Error: unkown dataset')
            sys.exit()



        if self.name_model == 'fully-connected':
            if name_dataset in ['MNIST',
                                'MNIST-no-regularization',
                                'MNIST-N1-1000']:
                layersizes = [784, 500, 10]
                self.activations_all = ['sigmoid', 'linear']
            elif name_dataset in ['DownScaledMNIST-no-regularization',
                                  'DownScaledMNIST-N1-1000-no-regularization']:
                # https://arxiv.org/pdf/1503.05671.pdf
                layersizes = [256, 20, 20, 20, 20, 20, 10]
                self.activations_all = ['tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'linear']
            elif name_dataset == 'MNIST-one-layer':
                layersizes = [784, 10]
#                 self.activations = ['linear']
                self.activations_all = ['linear']
            elif name_dataset in ['sythetic-linear-regression',
                                  'sythetic-linear-regression-N1-1']:
                layersizes = [100, 50] # http://proceedings.mlr.press/v70/zhou17a/zhou17a.pdf

                self.activations_all = ['linear']
            elif name_dataset == 'CIFAR':
                
                layersizes = [3072, 400, 400, 10]
                self.activations = ['sigmoid', 'sigmoid', 'linear']
                
            elif name_dataset == 'CIFAR-deep':
                
                layersizes = [3072, 128, 128, 128, 128, 10]
#                 self.activations = ['relu', 'relu', 'relu', 'relu', 'linear']
                self.activations_all = ['relu', 'relu', 'relu', 'relu', 'linear']
                
            elif name_dataset == 'Fashion-MNIST':
                # self.layersizes = [784, 400, 400, 10]
                layersizes = [784, 400, 400, 10]
                self.activations = ['sigmoid', 'sigmoid', 'linear']
            elif name_dataset == 'webspam':
                
                layersizes = [254, 400, 400, 1]
                self.activations = ['sigmoid', 'sigmoid', 'linear']
            elif name_dataset in ['MNIST-autoencoder',
                                  'MNIST-autoencoder-no-regularization',
                                  'MNIST-autoencoder-N1-1000',
                                  'MNIST-autoencoder-N1-1000-no-regularization',
                                  'MNIST-autoencoder-N1-1000-sum-loss-no-regularization',
                                  'MNIST-autoencoder-N1-1000-sum-loss']:
                # reference: https://arxiv.org/pdf/1301.3641.pdf,
                # https://www.cs.toronto.edu/~hinton/science.pdf
                
                layersizes = [784, 1000, 500, 250, 30, 250, 500, 1000, 784]
                self.activations_all =\
                ['sigmoid', 'sigmoid', 'sigmoid', 'linear', 'sigmoid', 'sigmoid', 'sigmoid', 'linear']
                
            elif name_dataset in ['MNIST-autoencoder-relu-N1-1000-sum-loss-no-regularization',
                                  'MNIST-autoencoder-relu-N1-1000-sum-loss',
                                  'MNIST-autoencoder-relu-N1-100-sum-loss',
                                  'MNIST-autoencoder-relu-N1-500-sum-loss',
                                  'MNIST-autoencoder-relu-N1-1-sum-loss']:
                
                layersizes = [784, 1000, 500, 250, 30, 250, 500, 1000, 784]
                self.activations_all =\
                ['relu', 'relu', 'relu', 'linear', 'relu', 'relu', 'relu', 'linear']
                
            elif name_dataset in ['MNIST-autoencoder-reluAll-N1-1-sum-loss']:
                
                layersizes = [784, 1000, 500, 250, 30, 250, 500, 1000, 784]
                self.activations_all =\
                ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear']
            
            elif name_dataset in ['CURVES-autoencoder',
                                  'CURVES-autoencoder-no-regularization',
                                  'CURVES-autoencoder-sum-loss-no-regularization',
                                  'CURVES-autoencoder-sum-loss']:
                # https://www.cs.toronto.edu/~hinton/science.pdf
                # https://arxiv.org/pdf/1301.3641.pdf
                
                layersizes =\
                [784, 400, 200, 100, 50, 25, 6, 25, 50, 100, 200, 400, 784]
                self.activations_all =\
                ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid',
                 'linear',
                 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'linear']
                
            elif name_dataset in ['CURVES-autoencoder-relu-sum-loss-no-regularization',
                                  'CURVES-autoencoder-relu-sum-loss',
                                  'CURVES-autoencoder-relu-N1-100-sum-loss',
                                  'CURVES-autoencoder-relu-N1-500-sum-loss']:
                
                layersizes =\
                [784, 400, 200, 100, 50, 25, 6, 25, 50, 100, 200, 400, 784]
                self.activations_all =\
                ['relu', 'relu', 'relu', 'relu', 'relu',
                 'linear',
                 'relu', 'relu', 'relu', 'relu', 'relu', 'linear']
                
            elif name_dataset in ['CURVES-autoencoder-Botev',
                                  'CURVES-autoencoder-Botev-sum-loss-no-regularization']:
                # https://arxiv.org/pdf/1706.03662.pdf
                layersizes = [784, 1000, 500, 250, 30, 250, 500, 1000, 784]
                self.activations_all =\
                ['sigmoid', 'sigmoid', 'sigmoid', 'linear', 'sigmoid', 'sigmoid', 'sigmoid', 'linear']
                
                
            elif name_dataset == 'CURVES-autoencoder-shallow':
                # https://www.cs.toronto.edu/~hinton/science.pdf
                
                layersizes = [784, 532, 6, 532, 784]
                self.activations =\
                ['sigmoid', 'linear', 'sigmoid', 'linear']
                
            elif name_dataset in ['FACES-autoencoder',
                                  'FACES-autoencoder-no-regularization',
                                  'FACES-autoencoder-sum-loss-no-regularization',
                                  'FACES-autoencoder-sum-loss']:
                # https://www.cs.toronto.edu/~hinton/science.pdf
                layersizes = [625, 2000, 1000, 500, 30,
                             500, 1000, 2000, 625]
                self.activations_all =\
                ['sigmoid', 'sigmoid', 'sigmoid', 'linear',
                 'sigmoid', 'sigmoid', 'sigmoid', 'linear']
                
            elif name_dataset in ['FACES-autoencoder-relu-sum-loss-no-regularization',
                                  'FACES-autoencoder-relu-sum-loss',
                                  'FacesMartens-autoencoder-relu',
                                  'FacesMartens-autoencoder-relu-no-regularization',
                                  'FacesMartens-autoencoder-relu-N1-500',
                                  'FacesMartens-autoencoder-relu-N1-100']:
                layersizes = [625, 2000, 1000, 500, 30,
                             500, 1000, 2000, 625]
                self.activations_all =\
                ['relu', 'relu', 'relu', 'linear',
                 'relu', 'relu', 'relu', 'linear']
                
                
                
            else:
                print('Dateset not supported!')
                sys.exit()
            
        elif self.name_model == 'simple-CNN':
            layersizes = []
#             self.activations_all = ['relu', '', 'relu', '', '', 'relu', 'linear']
            self.activations_all = []
        elif self.name_model == 'CNN':
            layersizes = []
#             self.activations = ['relu', 'relu', 'relu', 'linear']
#             self.activations_all = ['relu', '',  'relu', '',  'relu', '', '', 'linear']
            self.activations_all = []
        elif self.name_model == 'AllCNNC':
            layersizes = []
            self.activations_all = []
        elif self.name_model == 'ConvPoolCNNC':
            layersizes = []
            self.activations_all = []
        elif self.name_model == '1d-CNN':
            layersizes = []
#             self.activations_all = ['relu', 'relu', '', '', 'relu', 'linear']
            self.activations_all = ['relu', 'relu', 'relu', 'linear']
        elif self.name_model == 'vgg16':
            layersizes = []
            self.activations_all = []
        elif self.name_model == 'vgg11':
            layersizes = []
            self.activations_all = []
        elif self.name_model == 'ResNet32':
            layersizes = []
            self.activations_all = []
        elif self.name_model == 'ResNet34':
            layersizes = []
            self.activations_all = []
        else:
            print('Error: model name not supported for ' + self.name_model)
            sys.exit()
            
        self.layersizes = layersizes

        layers_params = get_layers_params(self.name_model, layersizes, self.activations_all, params)
        
        # self.layers: with weights
        # self.layers_all
        # self.layers_no_weights: to be deprecated
        

#         self.numlayers = len(layers_params)

        
#         self.layers = list(range(self.numlayers))
        self.layers_all = []
        
#         self.layers_no_weights = list(range(self.numlayers))
#         self.layers_no_weights = []

#         for l in range(self.numlayers):
        for l in range(len(layers_params)):
            
            if layers_params[l]['name'] == 'fully-connected':
#                 self.layers[l] =\
                self.layers_all.append(
                nn.Linear(layers_params[l]['input_size'], layers_params[l]['output_size'], bias=True)
                )
    
#                 self.layers_no_weights.append([])
                
            elif layers_params[l]['name'] in ['conv-no-bias-no-activation']:

                self.layers_all.append(
                    nn.Conv2d(
                        in_channels=layers_params[l]['conv_in_channels'],
                        out_channels=layers_params[l]['conv_out_channels'],
                        kernel_size=layers_params[l]['conv_kernel_size'],
                        stride=layers_params[l]['conv_stride'],
                        padding=layers_params[l]['conv_padding'],
                        bias=False
                    )
                )
                
            elif layers_params[l]['name'] in ['conv',
                                              'conv-no-activation']:

                self.layers_all.append(
                nn.Conv2d(
                    in_channels=layers_params[l]['conv_in_channels'],
                    out_channels=layers_params[l]['conv_out_channels'],
                    kernel_size=layers_params[l]['conv_kernel_size'],
                    stride=layers_params[l]['conv_stride'],
                    padding=layers_params[l]['conv_padding'])
                )
                
            elif layers_params[l]['name'] in ['ResBlock-BNNoAffine',
                                              'ResBlock-BNNoAffine-identityShortcut-NoBias',
                                              'ResBlock-BNNoAffine-PaddingShortcut-NoBias',
                                              'ResBlock-BN',
                                              'ResBlock-BN-BNshortcut',
                                              'ResBlock-BN-identityShortcut',
                                              'ResBlock-BN-identityShortcut-NoBias',
                                              'ResBlock-BN-BNshortcut-NoBias',
                                              'ResBlock-BN-PaddingShortcut-NoBias',]:
                
                self.layers_all.append([])
                
                # conv
                self.layers_all[-1].append(
                    nn.Conv2d(
                        in_channels=layers_params[l]['conv1']['conv_in_channels'],
                        out_channels=layers_params[l]['conv1']['conv_out_channels'],
                        kernel_size=layers_params[l]['conv1']['conv_kernel_size'],
                        stride=layers_params[l]['conv1']['conv_stride'],
                        padding=layers_params[l]['conv1']['conv_padding'],
                        bias=layers_params[l]['conv1']['conv_bias']
                    )
                )
                
                # BN or BNNoAffine
                if layers_params[l]['name'] in ['ResBlock-BNNoAffine',
                                                'ResBlock-BNNoAffine-identityShortcut-NoBias',
                                                'ResBlock-BNNoAffine-PaddingShortcut-NoBias',]:
                    self.layers_all[-1].append(
                        nn.BatchNorm2d(layers_params[l]['BNNoAffine1']['num_features'], affine=False).to(params['device'])
                    )
                elif layers_params[l]['name'] in ['ResBlock-BN',
                                                  'ResBlock-BN-BNshortcut',
                                                  'ResBlock-BN-identityShortcut',
                                                  'ResBlock-BN-identityShortcut-NoBias',
                                                  'ResBlock-BN-BNshortcut-NoBias',
                                                  'ResBlock-BN-PaddingShortcut-NoBias']:
                    self.layers_all[-1].append(
                        nn.BatchNorm2d(layers_params[l]['BN1']['num_features'], affine=True).to(params['device'])
                    )
                else:
                    print('layers_params[l][name]')
                    print(layers_params[l]['name'])
                    sys.exit()
                
                # conv
                self.layers_all[-1].append(
                    nn.Conv2d(
                        in_channels=layers_params[l]['conv2']['conv_in_channels'],
                        out_channels=layers_params[l]['conv2']['conv_out_channels'],
                        kernel_size=layers_params[l]['conv2']['conv_kernel_size'],
                        stride=layers_params[l]['conv2']['conv_stride'],
                        padding=layers_params[l]['conv2']['conv_padding'],
                        bias=layers_params[l]['conv2']['conv_bias']
                    )
                )
                
                # BN or BNNoAffine
                
                
                if layers_params[l]['name'] in ['ResBlock-BNNoAffine',
                                                'ResBlock-BNNoAffine-identityShortcut-NoBias',
                                                'ResBlock-BNNoAffine-PaddingShortcut-NoBias',]:
                    self.layers_all[-1].append(
                        nn.BatchNorm2d(layers_params[l]['BNNoAffine2']['num_features'], affine=False).to(params['device'])
                    )
                elif layers_params[l]['name'] in ['ResBlock-BN',
                                                  'ResBlock-BN-BNshortcut',
                                                  'ResBlock-BN-identityShortcut',
                                                  'ResBlock-BN-identityShortcut-NoBias',
                                                  'ResBlock-BN-BNshortcut-NoBias',
                                                  'ResBlock-BN-PaddingShortcut-NoBias']:
                    self.layers_all[-1].append(
                        nn.BatchNorm2d(layers_params[l]['BN2']['num_features'], affine=True).to(params['device'])
                    )
                else:
                    print('layers_params[l][name]')
                    print(layers_params[l]['name'])
                    sys.exit()
                
                if layers_params[l]['name'] in ['ResBlock-BNNoAffine-identityShortcut-NoBias',
                                                'ResBlock-BNNoAffine-PaddingShortcut-NoBias',
                                                'ResBlock-BN-identityShortcut',
                                                'ResBlock-BN-identityShortcut-NoBias',
                                                'ResBlock-BN-PaddingShortcut-NoBias',]:
                    1
                elif layers_params[l]['name'] in ['ResBlock-BNNoAffine',
                                                  'ResBlock-BN',
                                                  'ResBlock-BN-BNshortcut',
                                                  'ResBlock-BN-BNshortcut-NoBias']:
                    # 1*1 conv
                    self.layers_all[-1].append(
                        nn.Conv2d(
                            in_channels=layers_params[l]['conv3']['conv_in_channels'],
                            out_channels=layers_params[l]['conv3']['conv_out_channels'],
                            kernel_size=layers_params[l]['conv3']['conv_kernel_size'],
                            stride=layers_params[l]['conv3']['conv_stride'],
                            padding=layers_params[l]['conv3']['conv_padding'],
                            bias=layers_params[l]['conv3']['conv_bias']
                        )
                    )

                    if layers_params[l]['name'] in ['ResBlock-BN-BNshortcut',
                                                    'ResBlock-BN-BNshortcut-NoBias']:
                        self.layers_all[-1].append(
                            nn.BatchNorm2d(layers_params[l]['BN3']['num_features'], affine=True).to(params['device'])
                        )
                else:
                    print('layers_params[l][name]')
                    print(layers_params[l]['name'])
                    sys.exit()
                
                    

                    
            elif layers_params[l]['name'] == '1d-conv':
#                 self.layers[l] =\
                self.layers_all.append(
                nn.Conv1d(
                    in_channels=layers_params[l]['conv_in_channels'],
                    out_channels=layers_params[l]['conv_out_channels'],
                    kernel_size=layers_params[l]['conv_kernel_size'],
                    stride=layers_params[l]['conv_stride'],
                    padding=layers_params[l]['conv_padding'])
                )
#                 if layers_params[l]['if_max_pool']:
#                     self.layers_no_weights.append(
#                     nn.MaxPool1d(
#                         kernel_size=layers_params[l]['max_pool_kernel_size'],
#                         stride=layers_params[l]['max_pool_stride'])
#                     )
#                 else:
#                 self.layers_no_weights.append([])
                
            elif layers_params[l]['name'] == 'flatten':
                self.layers_all.append(nn.Flatten())
#                 self.layers_no_weights.append(
#                 []
#                 )
                
            elif layers_params[l]['name'] == 'max_pool_1d':
                self.layers_all.append(
                    nn.MaxPool1d(
                        kernel_size=layers_params[l]['max_pool_kernel_size'],
                        stride=layers_params[l]['max_pool_stride'])
                )
#                 self.layers_no_weights.append(
#                 []
#                 )
                
            elif layers_params[l]['name'] == 'max_pool':
                self.layers_all.append(
                    nn.MaxPool2d(
                        kernel_size=layers_params[l]['max_pool_kernel_size'],
                        stride=layers_params[l]['max_pool_stride'])
                )

            elif layers_params[l]['name'] == 'AdaptiveAvgPool2d':
                self.layers_all.append(
                    nn.AdaptiveAvgPool2d(
                        output_size=layers_params[l]['AdaptiveAvgPool2d_output_size']
                    )
                )
            
            elif layers_params[l]['name'] == 'dropout':
                self.layers_all.append(
                    nn.Dropout(
                        p=layers_params[l]['dropout_p'], 
                        inplace=False
                    )
                    )
                
            elif layers_params[l]['name'] == 'global_average_pooling':
                self.layers_all.append([])
                
            elif layers_params[l]['name'] == 'relu':
                self.layers_all.append([])
                
            elif layers_params[l]['name'] == 'BN':
                self.layers_all.append(
                    nn.BatchNorm2d(layers_params[l]['num_features'])
                )
                
            elif layers_params[l]['name'] == 'BNNoAffine':
                
                self.layers_all.append(
                    nn.BatchNorm2d(layers_params[l]['num_features'], affine=False).to(params['device'])
                )
                
#                 sys.exit()
                
                    
            else:
                print('Error: layer unsupported for ' + layers_params[l]['name'])
                sys.exit()
                
                
#         print('self.layers_all')
#         print(self.layers_all)
        
#         sys.exit()

        

        self.layers_weight = []
#         for l in range(self.numlayers):
        for l in range(len(layers_params)):
            
            if layers_params[l]['name'] in ['fully-connected',
                                            'conv',
                                            'conv-no-activation',
                                            '1d-conv']:
                layers_weight_l = {}
                layers_weight_l['W'] = self.layers_all[l].weight
                layers_weight_l['b'] = self.layers_all[l].bias
                self.layers_weight.append(layers_weight_l)
                
            elif layers_params[l]['name'] in ['conv-no-bias-no-activation']:
                layers_weight_l = {}
                layers_weight_l['W'] = self.layers_all[l].weight
                
#                 print('self.layers_all[l].bias')
#                 print(self.layers_all[l].bias)
                
#                 sys.exit()
                
#                 layers_weight_l['b'] = self.layers_all[l].bias
                self.layers_weight.append(layers_weight_l)
                
            elif layers_params[l]['name'] in ['ResBlock-BNNoAffine',
                                              'ResBlock-BNNoAffine-identityShortcut-NoBias',
                                              'ResBlock-BNNoAffine-PaddingShortcut-NoBias',
                                              'ResBlock-BN',
                                              'ResBlock-BN-BNshortcut',
                                              'ResBlock-BN-identityShortcut',
                                              'ResBlock-BN-identityShortcut-NoBias',
                                              'ResBlock-BN-BNshortcut-NoBias',
                                              'ResBlock-BN-PaddingShortcut-NoBias',]:
                
                layers_weight_l = {}
                layers_weight_l['W'] = self.layers_all[l][0].weight
                
                if layers_params[l]['name'] in ['ResBlock-BNNoAffine-identityShortcut-NoBias',
                                                'ResBlock-BNNoAffine-PaddingShortcut-NoBias',
                                                'ResBlock-BN-identityShortcut-NoBias',
                                                'ResBlock-BN-BNshortcut-NoBias',
                                                'ResBlock-BN-PaddingShortcut-NoBias']:
                    pass
                elif layers_params[l]['name'] in ['ResBlock-BNNoAffine',
                                                  'ResBlock-BN',
                                                  'ResBlock-BN-identityShortcut',
                                                  'ResBlock-BN-BNshortcut']:
                    layers_weight_l['b'] = self.layers_all[l][0].bias
                else:
                    print('layers_params[l][name]')
                    print(layers_params[l]['name'])
                    sys.exit()
                
                    
                    
                self.layers_weight.append(layers_weight_l)
                
                if layers_params[l]['name'] in ['ResBlock-BN',
                                                'ResBlock-BN-BNshortcut',
                                                'ResBlock-BN-identityShortcut',
                                                'ResBlock-BN-identityShortcut-NoBias',
                                                'ResBlock-BN-BNshortcut-NoBias',
                                                'ResBlock-BN-PaddingShortcut-NoBias']:
                    layers_weight_l = {}
                    layers_weight_l['W'] = self.layers_all[l][1].weight
                    layers_weight_l['b'] = self.layers_all[l][1].bias
                    self.layers_weight.append(layers_weight_l)
                
                layers_weight_l = {}
                layers_weight_l['W'] = self.layers_all[l][2].weight
                
                if layers_params[l]['name'] in ['ResBlock-BNNoAffine-identityShortcut-NoBias',
                                                'ResBlock-BNNoAffine-PaddingShortcut-NoBias',
                                                'ResBlock-BN-identityShortcut-NoBias',
                                                'ResBlock-BN-BNshortcut-NoBias',
                                                'ResBlock-BN-PaddingShortcut-NoBias']:
                    pass
                elif layers_params[l]['name'] in ['ResBlock-BNNoAffine',
                                                  'ResBlock-BN',
                                                  'ResBlock-BN-identityShortcut',
                                                  'ResBlock-BN-BNshortcut']:
                    layers_weight_l['b'] = self.layers_all[l][2].bias
                else:
                    print('layers_params[l][name]')
                    print(layers_params[l]['name'])
                    sys.exit()
                    
                self.layers_weight.append(layers_weight_l)
                
                if layers_params[l]['name'] in ['ResBlock-BN',
                                                'ResBlock-BN-BNshortcut',
                                                'ResBlock-BN-identityShortcut',
                                                'ResBlock-BN-identityShortcut-NoBias',
                                                'ResBlock-BN-BNshortcut-NoBias',
                                                'ResBlock-BN-PaddingShortcut-NoBias']:
                    layers_weight_l = {}
                    layers_weight_l['W'] = self.layers_all[l][3].weight
                    layers_weight_l['b'] = self.layers_all[l][3].bias
                    self.layers_weight.append(layers_weight_l)
                    
                    
                if layers_params[l]['name'] in ['ResBlock-BNNoAffine-identityShortcut-NoBias',
                                                'ResBlock-BNNoAffine-PaddingShortcut-NoBias',
                                                'ResBlock-BN-identityShortcut',
                                                'ResBlock-BN-identityShortcut-NoBias',
                                                'ResBlock-BN-PaddingShortcut-NoBias']:
                    1
                else:
                
                    layers_weight_l = {}
                    layers_weight_l['W'] = self.layers_all[l][4].weight
                    if layers_params[l]['name'] == 'ResBlock-BN-BNshortcut-NoBias':
                        pass
                    elif layers_params[l]['name'] in ['ResBlock-BNNoAffine',
                                                      'ResBlock-BN',
                                                      'ResBlock-BN-BNshortcut']:
                        layers_weight_l['b'] = self.layers_all[l][4].bias
                    else:
                        print('layers_params[l][name]')
                        print(layers_params[l]['name'])
                        sys.exit()
                        
                    self.layers_weight.append(layers_weight_l)


                    if layers_params[l]['name'] in ['ResBlock-BN-BNshortcut',
                                                    'ResBlock-BN-BNshortcut-NoBias']:
                        layers_weight_l = {}
                        layers_weight_l['W'] = self.layers_all[l][5].weight
                        
#                         if layers_params[l]['name'] == 'ResBlock-BN-BNshortcut-NoBias':
#                             pass
#                         else:
#                             sys.exit()
                        layers_weight_l['b'] = self.layers_all[l][5].bias
                        self.layers_weight.append(layers_weight_l)
                
            elif layers_params[l]['name'] == 'BN':
#                 print('self.layers_all[l]')
#                 print(self.layers_all[l])
                
#                 print('self.layers_all[l].weight.size()')
#                 print(self.layers_all[l].weight.size())
                
#                 print('self.layers_all[l].bias.size()')
#                 print(self.layers_all[l].bias.size())
#                 sys.exit()
                
                layers_weight_l = {}
                layers_weight_l['W'] = self.layers_all[l].weight
                layers_weight_l['b'] = self.layers_all[l].bias
                self.layers_weight.append(layers_weight_l)
            elif layers_params[l]['name'] in ['flatten',
                                              'max_pool',
                                              'max_pool_1d',
                                              'AdaptiveAvgPool2d',
                                              'dropout',
                                              'global_average_pooling',
                                              'relu',
                                              'BNNoAffine']:
                1
            else:
                print('layers_params[l][name]')
                print(layers_params[l]['name'])
                print('Error: layer unsupported when define weight for ' + layers_params[l]['name'])
                sys.exit()
            




        
        

                
        
        # filter out the layers with no weights
        self.layers_params_all = layers_params
        layers_params = []
        self.layers = []
        for l in range(len(self.layers_params_all)):
            if self.layers_params_all[l]['name'] in ['fully-connected',
                                                     'conv',
                                                     'conv-no-activation',
                                                     'conv-no-bias-no-activation',
                                                     '1d-conv',
                                                     'BN']:
                
#                 print('self.layers_params_all[l]')
#                 print(self.layers_params_all[l])
                
#                 sys.exit()
                
                layers_params.append(self.layers_params_all[l])
                self.layers.append(self.layers_all[l])
                
            elif self.layers_params_all[l]['name'] in ['ResBlock-BNNoAffine',
                                                       'ResBlock-BNNoAffine-identityShortcut-NoBias',
                                                       'ResBlock-BNNoAffine-PaddingShortcut-NoBias',
                                                       'ResBlock-BN',
                                                       'ResBlock-BN-BNshortcut',
                                                       'ResBlock-BN-identityShortcut',
                                                       'ResBlock-BN-identityShortcut-NoBias',
                                                       'ResBlock-BN-BNshortcut-NoBias',
                                                       'ResBlock-BN-PaddingShortcut-NoBias',]:
                
                if self.layers_params_all[l]['name'] in ['ResBlock-BNNoAffine-identityShortcut-NoBias',
                                                         'ResBlock-BNNoAffine-PaddingShortcut-NoBias',
                                                         'ResBlock-BN-identityShortcut-NoBias',
                                                         'ResBlock-BN-BNshortcut-NoBias',
                                                         'ResBlock-BN-PaddingShortcut-NoBias']:
                    layers_params.append(
                        {**{'name': 'conv-no-bias-no-activation'}, **self.layers_params_all[l]['conv1']}
                    )
                elif self.layers_params_all[l]['name'] in ['ResBlock-BNNoAffine',
                                                           'ResBlock-BN',
                                                           'ResBlock-BN-identityShortcut',
                                                           'ResBlock-BN-BNshortcut']:
                    
                    layers_params.append(
                        {**{'name': 'conv-no-activation'}, **self.layers_params_all[l]['conv1']}
                    )
                else:
                    print('self.layers_params_all[l][name]')
                    print(self.layers_params_all[l]['name'])
                    sys.exit()
            
    
                    
                
                if self.layers_params_all[l]['name'] in ['ResBlock-BN',
                                                         'ResBlock-BN-BNshortcut',
                                                         'ResBlock-BN-identityShortcut',
                                                         'ResBlock-BN-identityShortcut-NoBias',
                                                         'ResBlock-BN-BNshortcut-NoBias',
                                                         'ResBlock-BN-PaddingShortcut-NoBias']:
                    layers_params.append(
                        {**{'name': 'BN'}, **self.layers_params_all[l]['BN1']}
                    )
                

                if self.layers_params_all[l]['name'] in ['ResBlock-BNNoAffine-identityShortcut-NoBias',
                                                         'ResBlock-BNNoAffine-PaddingShortcut-NoBias',
                                                         'ResBlock-BN-identityShortcut-NoBias',
                                                         'ResBlock-BN-BNshortcut-NoBias',
                                                         'ResBlock-BN-PaddingShortcut-NoBias']:
                    layers_params.append(
                        {**{'name': 'conv-no-bias-no-activation'}, **self.layers_params_all[l]['conv2']}
                    )
                elif self.layers_params_all[l]['name'] in ['ResBlock-BNNoAffine',
                                                           'ResBlock-BN',
                                                           'ResBlock-BN-identityShortcut',
                                                           'ResBlock-BN-BNshortcut',
                                                           'ResBlock-BN-PaddingShortcut-NoBias']:
                    layers_params.append(
                        {**{'name': 'conv-no-activation'}, **self.layers_params_all[l]['conv2']}
                    )
                else:
                    print('self.layers_params_all[l][name]')
                    print(self.layers_params_all[l]['name'])
                    sys.exit()
                    
                
                if self.layers_params_all[l]['name'] in ['ResBlock-BN',
                                                         'ResBlock-BN-BNshortcut',
                                                         'ResBlock-BN-identityShortcut',
                                                         'ResBlock-BN-identityShortcut-NoBias',
                                                         'ResBlock-BN-BNshortcut-NoBias',
                                                         'ResBlock-BN-PaddingShortcut-NoBias']:
                    layers_params.append(
                        {**{'name': 'BN'}, **self.layers_params_all[l]['BN2']}
                    )
                
                if self.layers_params_all[l]['name'] in ['ResBlock-BNNoAffine-identityShortcut-NoBias',
                                                         'ResBlock-BNNoAffine-PaddingShortcut-NoBias',
                                                         'ResBlock-BN-identityShortcut',
                                                         'ResBlock-BN-identityShortcut-NoBias',
                                                         'ResBlock-BN-PaddingShortcut-NoBias']:
                    1
                else:
                    if self.layers_params_all[l]['name'] == 'ResBlock-BN-BNshortcut-NoBias':
                        layers_params.append(
                            {**{'name': 'conv-no-bias-no-activation'}, **self.layers_params_all[l]['conv3']}
                        )
                    elif self.layers_params_all[l]['name'] in ['ResBlock-BNNoAffine',
                                                               'ResBlock-BN',
                                                               'ResBlock-BN-BNshortcut']:
                        layers_params.append(
                            {**{'name': 'conv-no-activation'}, **self.layers_params_all[l]['conv3']}
                        )
                    else:
                        print('self.layers_params_all[l][name]')
                        print(self.layers_params_all[l]['name'])
                        sys.exit()
                        

                    if self.layers_params_all[l]['name'] in ['ResBlock-BN-BNshortcut',
                                                             'ResBlock-BN-BNshortcut-NoBias']:
                        layers_params.append(
                            {**{'name': 'BN'}, **self.layers_params_all[l]['BN3']}
                        )

                self.layers.append(self.layers_all[l][0])
                
                if self.layers_params_all[l]['name'] in ['ResBlock-BN',
                                                         'ResBlock-BN-BNshortcut',
                                                         'ResBlock-BN-identityShortcut',
                                                         'ResBlock-BN-identityShortcut-NoBias',
                                                         'ResBlock-BN-BNshortcut-NoBias',
                                                         'ResBlock-BN-PaddingShortcut-NoBias']:
                    self.layers.append(self.layers_all[l][1])
                
                self.layers.append(self.layers_all[l][2])
                
                if self.layers_params_all[l]['name'] in ['ResBlock-BN',
                                                         'ResBlock-BN-BNshortcut',
                                                         'ResBlock-BN-identityShortcut',
                                                         'ResBlock-BN-identityShortcut-NoBias',
                                                         'ResBlock-BN-BNshortcut-NoBias',
                                                         'ResBlock-BN-PaddingShortcut-NoBias']:
                    self.layers.append(self.layers_all[l][3])
                
                if self.layers_params_all[l]['name'] in ['ResBlock-BNNoAffine-PaddingShortcut-NoBias',
                                                         'ResBlock-BNNoAffine-identityShortcut-NoBias',
                                                         'ResBlock-BN-identityShortcut',
                                                         'ResBlock-BN-identityShortcut-NoBias',
                                                         'ResBlock-BN-PaddingShortcut-NoBias']:
                    1
                else:
                    self.layers.append(self.layers_all[l][4])

                    if self.layers_params_all[l]['name'] in ['ResBlock-BN-BNshortcut',
                                                             'ResBlock-BN-BNshortcut-NoBias']:
                        self.layers.append(self.layers_all[l][5])
                
            elif self.layers_params_all[l]['name'] in ['flatten',
                                                       'max_pool',
                                                       'max_pool_1d',
                                                       'AdaptiveAvgPool2d',
                                                       'dropout',
                                                       'global_average_pooling',
                                                       'relu',
                                                       'BNNoAffine']:
                1
            else:
                print('error: unkown layers_params_all[l][name]: ' + self.layers_params_all[l]['name'])
                sys.exit()
                

                
        
        import torch.nn.init as init
        
        for l in range(len(layers_params)):
            
            if layers_params[l]['name'] == 'fully-connected':
                
                if params['initialization_pkg'] == 'numpy':
                
                    np_W_l =\
                    (np.random.uniform(size=self.layers_weight[l]['W'].size()) * 2 - 1) *\
                    np.sqrt(2 / (layers_params[l]['input_size'] + layers_params[l]['output_size']))

                    np_W_l = torch.from_numpy(np_W_l)

                    self.layers_weight[l]['W'].data = np_W_l.type(torch.FloatTensor)
                
                elif params['initialization_pkg'] == 'torch':
                    
                    self.layers_weight[l]['W'].data =\
                    (
                        torch.distributions.uniform.Uniform(low=0, high=1).sample(
                            sample_shape=self.layers_weight[l]['W'].size()
                        )
                        * 2 - 1
                    ) *\
                    np.sqrt(
                        2 / (layers_params[l]['input_size'] + layers_params[l]['output_size'])
                    )
                    
                elif params['initialization_pkg'] in ['default',
                                                      'normal']:
                    # U(-sqrt(k), sqrt(k)), where k = 1 / in_featurest
                    
                    pass
                    
                elif params['initialization_pkg'] == 'kaiming_normal':
                    
#                     print('torch.norm(self.layers_weight[l][W])')
#                     print(torch.norm(self.layers_weight[l]['W']))
                    
                    init.kaiming_normal_(self.layers_weight[l]['W'])
                    
#                     print('torch.norm(self.layers_weight[l][W])')
#                     print(torch.norm(self.layers_weight[l]['W']))
                    
#                     sys.exit()
                    
                else:
                    print('error: unknown params[initialization_pkg] for ' + params['initialization_pkg'])
                    sys.exit()
                

            elif layers_params[l]['name'] in ['conv',
                                              'conv-no-activation',
                                              'conv-no-bias-no-activation']:
                
                if params['initialization_pkg'] == 'normal':
            
                    # https://github.com/chengyangfu/pytorch-vgg-cifar10
                    
                    if layers_params[l]['name'] == 'conv-no-bias-no-activation':
                        pass
                    elif layers_params[l]['name'] in ['conv-no-activation',
                                                      'conv']:
                        self.layers_weight[l]['b'].data.zero_()
                    else:
                        print('layers_params[l][name]')
                        print(layers_params[l]['name'])
                        sys.exit()

                    self.layers_weight[l]['W'].data.normal_(
                        0, 
                        math.sqrt(2. / (layers_params[l]['conv_kernel_size']**2 * layers_params[l]['conv_out_channels']))
                    )
                elif params['initialization_pkg'] == 'default':
                    # use default initialization
                    1
                elif params['initialization_pkg'] == 'kaiming_normal':
                    
                    # the default arguments are equivalent to relu
                    
                    init.kaiming_normal_(self.layers_weight[l]['W'])
                    
                else:
                    print('error: need to check for ' + params['initialization_pkg'])
                    sys.exit()
                    
                    
            elif layers_params[l]['name'] in ['ResBlock-BNNoAffine']:
                
                print('should not reach here')
                
                sys.exit()
                
                print('need to check how to initialize')
                    
                    
                
            elif layers_params[l]['name'] in ['1d-conv']:
                
                sys.exit()
                
                # use default initialization
                1

            elif layers_params[l]['name'] in ['BN']:
                1

            else:
                print('layers_params[l][name]')
                print(layers_params[l]['name'])
                print('Error: layer unsupported when initialization.')
                sys.exit()
                
                
#         sys.exit()
        
        
        self.layers_params = layers_params
        
        self.numlayers = len(layers_params)
        self.numlayers_all = len(self.layers_params_all)
        
        self.layers = nn.ModuleList(self.layers)

        
            

    
    def forward(self, x):
        
        
#         a = list(range(self.numlayers))
        a = []
        
#         h = list(range(self.numlayers))
        h = []

        input_ = x


        for l in range(self.numlayers_all):
            
#             print('input_.size()')
#             print(input_.size())
            
#             print('self.layers_params_all[l][name]')
#             print(self.layers_params_all[l]['name'])
            
            if self.layers_params_all[l]['name'] in ['fully-connected',
                                                     'conv',
                                                     'conv-no-activation',
                                                     'conv-no-bias-no-activation',
                                                     '1d-conv',
                                                     'BN']:

#                 input_, a_l =\
#                 get_layer_error: need to accomadate GAP(
#                     input_, self.layers_all[l], self.activations_all[l], self.layers_params_all[l])
            
                if self.layers_params_all[l]['name'] in ['conv-no-activation',
                                                         'conv',
                                                         'conv-no-bias-no-activation',
                                                         'fully-connected']:

                    h.append(input_)
                
                elif self.layers_params_all[l]['name'] == 'BN':
                    
                    h.append([])
                    
#                     print('input_.size()')
#                     print(input_.size())
                    
#                     print('self.layers_all[l](input_).size()')
#                     print(self.layers_all[l](input_).size())
                    
#                     test_output_affine_true = self.layers_all[l](input_)
                    
#                     print('torch.norm(self.layers_all[l](input_))')
#                     print(torch.norm(self.layers_all[l](input_)))
                    
                    
                    
#                     print('self.layers_all[l].affine')
#                     print(self.layers_all[l].affine)

#                     print('self.layers_all[l].weight.size()')
#                     print(self.layers_all[l].weight.size())
                    
#                     print('torch.norm(self.layers_all[l](input_))')
#                     print(torch.norm(self.layers_all[l](input_)))
                    
#                     print('torch.norm(self.layers_all[l](input_) - test_output_affine_true)')
#                     print(torch.norm(self.layers_all[l](input_) - test_output_affine_true))
                    
#                     print('self.layers_all[l].weight[0].item()')
#                     print(self.layers_all[l].weight[0].item())
                    
#                     if np.abs(self.layers_all[l].weight[0].item() - 1.0) > 1e-3:
                        
#                         print('np.abs(self.layers_all[l].weight[0].item() - 1.0)')
#                         print(np.abs(self.layers_all[l].weight[0].item() - 1.0))
                    
#                         sys.exit()
                    
                else:
                    print('error: need to check for ' + self.layers_params_all[l]['name'])
                    sys.exit()
                
                
                input_, a_l =\
                get_layer_forward(
                    input_, self.layers_all[l], self.layers_params_all[l]['activation'], self.layers_params_all[l])

                a.append(a_l)
                
                
            elif self.layers_params_all[l]['name'] in ['ResBlock-BNNoAffine',
                                                       'ResBlock-BNNoAffine-identityShortcut-NoBias',
                                                       'ResBlock-BNNoAffine-PaddingShortcut-NoBias',
                                                       'ResBlock-BN',
                                                       'ResBlock-BN-BNshortcut',
                                                       'ResBlock-BN-identityShortcut',
                                                       'ResBlock-BN-identityShortcut-NoBias',
                                                       'ResBlock-BN-BNshortcut-NoBias',
                                                       'ResBlock-BN-PaddingShortcut-NoBias',]:
                
                h.append([])
                h.append([])
                
                a.append([])
                a.append([])
                
                if self.layers_params_all[l]['name'] == 'ResBlock-BN':
                    h.append([])
                    h.append([])

                    a.append([])
                    a.append([])
                    
                    h.append([])

                    a.append([])
                    
                    index_mapping = [-1, -5, -4, -3, -2]
                elif self.layers_params_all[l]['name'] in ['ResBlock-BN-BNshortcut',
                                                           'ResBlock-BN-BNshortcut-NoBias']:
                    h.append([])
                    h.append([])
                    h.append([])

                    a.append([])
                    a.append([])
                    a.append([])
                    
                    h.append([])

                    a.append([])
                    
                    index_mapping = [-2, -6, -5, -4, -3]
                elif self.layers_params_all[l]['name'] in ['ResBlock-BN-identityShortcut',
                                                           'ResBlock-BN-identityShortcut-NoBias',
                                                           'ResBlock-BN-PaddingShortcut-NoBias']:
                    h.append([])

                    a.append([])
                    
                    h.append([])

                    a.append([])
                    
                    index_mapping = [None, -4, -3, -2, -1]
                elif self.layers_params_all[l]['name'] in ['ResBlock-BNNoAffine-identityShortcut-NoBias',
                                                           'ResBlock-BNNoAffine-PaddingShortcut-NoBias',]:
                
                    index_mapping = [None, -2, None, -1, None]
                    
                elif self.layers_params_all[l]['name'] == 'ResBlock-BNNoAffine':
                    index_mapping = [-1, -3, None, -2, None]
                    
                
                if self.layers_params_all[l]['name'] in ['ResBlock-BNNoAffine-identityShortcut-NoBias',
                                                         'ResBlock-BN-identityShortcut',
                                                         'ResBlock-BN-identityShortcut-NoBias']:
                    input_shortcut = input_
                elif self.layers_params_all[l]['name'] in ['ResBlock-BNNoAffine-PaddingShortcut-NoBias',
                                                           'ResBlock-BN-PaddingShortcut-NoBias',]:

                    input_shortcut = F.pad(input_[:, :, ::2, ::2], (0,0,0,0,input_.size(1)//2,input_.size(1)//2))
                    
                else:

                    h[index_mapping[0]] = input_
                    input_shortcut, a_l = get_layer_forward(
                        input_, self.layers_all[l][4], None, {'name':'conv-no-activation'})
                    a[index_mapping[0]] = a_l

                    if self.layers_params_all[l]['name'] in ['ResBlock-BN-BNshortcut',
                                                             'ResBlock-BN-BNshortcut-NoBias']:
                        input_shortcut, a_l =\
                        get_layer_forward(
                            input_shortcut, self.layers_all[l][5], None, {'name':'BN'})
                        a[-1] = a_l
                
                
                # conv
                h[index_mapping[1]] = input_
                input_, a_l =\
                get_layer_forward(
                    input_, self.layers_all[l][0], None, {'name':'conv-no-activation'})
                # no-bias is not needed here
                a[index_mapping[1]] = a_l
                
                # BN or BNNoAffine
                if index_mapping[2] == None:
                    input_ = self.layers_all[l][1](input_)
                else:
                    input_, a_l =\
                    get_layer_forward(
                        input_, self.layers_all[l][1], None, {'name':'BN'})
                    a[index_mapping[2]] = a_l
                
                # relu
                input_ = F.relu(input_)
                
                # conv
                h[index_mapping[3]] = input_
                input_, a_l =\
                get_layer_forward(
                    input_, self.layers_all[l][2], None, {'name':'conv-no-activation'})
                a[index_mapping[3]] = a_l
                
                # BN or BNNoAffine
                if index_mapping[4] == None:
                    input_ = self.layers_all[l][3](input_)
                else:
                    input_, a_l =\
                    get_layer_forward(
                        input_, self.layers_all[l][3], None, {'name':'BN'})
                    a[index_mapping[4]] = a_l
                
#                 if self.layers_params_all[l]['name'] in ['ResBlock-BN-PaddingShortcut-NoBias']:
#                     sys.exit()
                
                # add
                input_ = input_ + input_shortcut
                
                # relu
                input_ = F.relu(input_)
                
            elif self.layers_params_all[l]['name'] in ['flatten',
                                                       'max_pool',
                                                       'max_pool_1d',
                                                       'AdaptiveAvgPool2d',
                                                       'dropout',
                                                       'BNNoAffine']:

                

                input_ = self.layers_all[l](input_)
                
            elif self.layers_params_all[l]['name'] == 'global_average_pooling':
                
#                 print('input_.size()')
#                 print(input_.size())
                
                input_ = torch.mean(input_, dim=(2,3))
                
#                 print('input_.size()')
#                 print(input_.size())
                
#                 sys.exit()
    
            elif self.layers_params_all[l]['name'] == 'relu':
                input_ = F.relu(input_)
            else:
                print('error: unknown self.layers_params_all[l][name]: ' + self.layers_params_all[l]['name'])
                sys.exit()
                
#             print('input_.size() in forward')
#             print(input_.size())
                
            
            
            
            # max pooling

#             if not self.layers_params_all[l]['name'] == 'fully-connected':
#                 if 'if_max_pool' in self.layers_params_all[l] and\
#                 self.layers_params_all[l]['if_max_pool']:
#                     input_ = self.layers_no_weights[l](input_)
                    



        # a[l]: pre-activation of the current layer (the one that requires grad)
        # h[l]: input of the current layer (post-activation of the previous layer in the fully-connected case)
        
#         sys.exit()
        
        return input_, a, h

    
    

def get_model(params):
    model = Model_3(params)
    if params['if_gpu']:
        model.to(params['device'])
    return model


def get_A_A_T_kfac(a, h, l, params):
    
    layers_params = params['layers_params']
    device = params['device']
    
    kernel_size = layers_params[l]['conv_kernel_size']
    padding = layers_params[l]['conv_padding']


    if layers_params[l]['name'] == '1d-conv':
        size_test_2_A_j = kernel_size *\
layers_params[l]['conv_in_channels']
    elif layers_params[l]['name'] == 'conv':
        
        size_test_2_A_j = kernel_size**2 *\
layers_params[l]['conv_in_channels']
    
    if params['Kron_BFGS_if_homo']:
        size_test_2_A_j += 1


    test_2_A_j = torch.zeros(
        size_test_2_A_j, size_test_2_A_j, device=device
    )

    if layers_params[l]['name'] == '1d-conv':
        # 1d-conv: a[l]: M * I * |T|
        h_l_padded = F.pad(h[l], (padding, padding), "constant", 0)
        
        h_homo_ones = torch.ones(h_l_padded.size(0), 1 ,device=device)
        
        

        for t in range(a[l].size(2)):
            # a[l].size(2) = |T|
            
            h_l_t = h_l_padded[:, :, t:t+kernel_size].data

            # in the flatten, delta changes the fastest
            h_l_t_flatten = h_l_t.flatten(start_dim=1)
            
            if params['Kron_BFGS_if_homo']:
                h_l_t_flatten = torch.cat((h_l_t_flatten, h_homo_ones), dim=1)

            test_2_A_j += torch.mm(h_l_t_flatten.t(), h_l_t_flatten)
    elif layers_params[l]['name'] == 'conv':
        # 2d-conv: a[l]: M * I * |T|, where |T| has two dimensions
        h_l_padded = F.pad(
            h[l], (padding, padding, padding, padding), "constant", 0
        )
        
        h_homo_ones = torch.ones(h_l_padded.size(0), 1 ,device=device)

        for t1 in range(a[l].size(2)):
            for t2 in range(a[l].size(3)):
                
#                 print('h_l_padded.size()')
#                 print(h_l_padded.size())
                
                h_l_t =\
h_l_padded[:, :, t1:t1+kernel_size, t2:t2+kernel_size].data

#                 print('h_l_t.size()')
#                 print(h_l_t.size())

                h_l_t_flatten = h_l_t.flatten(start_dim=1)
                
                if params['Kron_BFGS_if_homo']:
                    h_l_t_flatten = torch.cat((h_l_t_flatten, h_homo_ones), dim=1)
                
#                 if l == 1:
#                     sys.exit()
    
                test_2_A_j += torch.mm(h_l_t_flatten.t(), h_l_t_flatten)
                
                
    return test_2_A_j




def get_A_A_T_kron_bfgs_v5(h, l, params):
    
    layers_params = params['layers_params']
    device = params['device']
                        
    in_channels = layers_params[l]['conv_in_channels']
    kernel_size = layers_params[l]['conv_kernel_size']
    padding = layers_params[l]['conv_padding']

    if layers_params[l]['name'] == '1d-conv':

        test_5_A_j = torch.zeros(
            in_channels,
            kernel_size,
            in_channels,
            kernel_size,
            device=device
        )

        h_l_padded = F.pad(h[l].data, (padding, padding), "constant", 0)
        
        T = h_l_padded.size(2)-kernel_size+1
        
        weight_conv1d = torch.ones(1, 1, T, device=device)

        for delta_2 in range(-kernel_size+1, kernel_size):
            
            
#             h_l_diff_1 = h_l_padded[:, :, np.maximum(0,delta_2):
#                     np.minimum(h_l_padded.size(2),h_l_padded.size(2)+delta_2)]
            
            if delta_2 > 0:
                h_l_diff_1 = h_l_padded[:, :, delta_2:h_l_padded.size(2)]
                h_l_diff_2 = h_l_padded[:, :, 0:h_l_padded.size(2)-delta_2]
            else:
                h_l_diff_1 = h_l_padded[:, :, 0:h_l_padded.size(2)+delta_2]
                h_l_diff_2 = h_l_padded[:, :, -delta_2:h_l_padded.size(2)]
            
            h_l_square = torch.einsum('ijl,ikl->jkl', h_l_diff_1.data, h_l_diff_2.data)
            
#             weight_conv1d = torch.ones(1, 1, T, device=device)
            
            h_l_square_conv = F.conv1d(
                h_l_square.view(h_l_square.size(0) * h_l_square.size(1), 1, -1),
                weight_conv1d
            ).view(h_l_square.size(0), h_l_square.size(1), -1)
            
            if delta_2 > 0:
                start_index = [0, delta_2]
            else:
                start_index = [-delta_2, 0]

            for delta in range(kernel_size-np.abs(delta_2)):
                test_5_A_j[
                    :, start_index[0] + delta, :, start_index[1] + delta
                ] = h_l_square_conv[:, :, delta].t()

            



#         print('test_5_A_j.size()')
#         print(test_5_A_j.size())
                
        test_5_A_j = test_5_A_j.view(
        test_5_A_j.size(0) * test_5_A_j.size(1),
        test_5_A_j.size(2) * test_5_A_j.size(3),
    )
        
#         print('test_5_A_j.size()')
#         print(test_5_A_j.size())
        
        if params['Kron_BFGS_if_homo']:
            homo_test_5_A_j = torch.zeros(
                test_5_A_j.size(0)+1, test_5_A_j.size(1)+1, device=device
            )
            
            homo_test_5_A_j[:-1, :-1] = test_5_A_j

            sum_h_l_padded = torch.sum(h_l_padded, dim=0)
            sum_h_l_padded = sum_h_l_padded.unsqueeze(1)
            weight_conv1d = torch.ones(1, 1, T, device=device) 
            homo_test_5_A_j[-1, :-1] =\
            F.conv1d(sum_h_l_padded, weight_conv1d).view(-1)
            
            homo_test_5_A_j[:-1, -1] = homo_test_5_A_j[-1, :-1]
            
            homo_test_5_A_j[-1, -1] = T * h_l_padded.size(0)
            
#             print('homo_test_5_A_j.requires_grad')
#             print(homo_test_5_A_j.requires_grad)
            
            test_5_A_j = homo_test_5_A_j

    elif layers_params[l]['name'] == 'conv':
        test_5_A_j = torch.zeros(
            in_channels,
            kernel_size,
            kernel_size,
            in_channels,
            kernel_size,
            kernel_size,
            device=device
        )

        h_l_padded = F.pad(
            h[l].data, (padding, padding, padding, padding), "constant", 0
        )
        
        
#         print('h_l_padded.size()')
#         print(h_l_padded.size())
#         sys.exit()
        T0 = h_l_padded.size(2)-kernel_size+1
        T1 = h_l_padded.size(3)-kernel_size+1
        
        
        weight_conv1d = torch.ones(1, 1, h_l_padded.size(2)-kernel_size+1, h_l_padded.size(3)-kernel_size+1, device=device)
        
        
        for delta_2_0 in range(-kernel_size+1, kernel_size):
            for delta_2_1 in range(-kernel_size+1, kernel_size):
                
                
                if delta_2_0 > 0:
                    h_l_diff_1 = h_l_padded[:, :, delta_2_0:h_l_padded.size(2)]
                    h_l_diff_2 = h_l_padded[:, :, 0:h_l_padded.size(2)-delta_2_0]
                else:
                    h_l_diff_1 = h_l_padded[:, :, 0:h_l_padded.size(2)+delta_2_0]
                    h_l_diff_2 = h_l_padded[:, :, -delta_2_0:h_l_padded.size(2)]
                    
#                 print('h_l_diff_1.size()')
#                 print(h_l_diff_1.size())
                    
                if delta_2_1 > 0:
                    h_l_diff_1 = h_l_diff_1[:, :, :, delta_2_1:h_l_padded.size(3)]
                    h_l_diff_2 = h_l_diff_2[:, :, :, 0:h_l_padded.size(3)-delta_2_1]
                else:
                    h_l_diff_1 = h_l_diff_1[:, :, :, 0:h_l_padded.size(3)+delta_2_1]
                    h_l_diff_2 = h_l_diff_2[:, :, :, -delta_2_1:h_l_padded.size(3)]

                
                
                h_l_square = torch.einsum(
                    'ijlt,iklt->jklt', h_l_diff_1.data, h_l_diff_2.data
                )
                
#                 print('h_l_diff_1.size()')
#                 print(h_l_diff_1.size())
                
                
                t1=0
                t2=0
                kfac_h_l_t =\
h_l_padded[:, :, t1:t1+kernel_size, t2:t2+kernel_size].data



#                 kfac_h_l_t_flatten = kfac_h_l_t.flatten(start_dim=1)
#                 torch.mm(kfac_h_l_t_flatten.t(), kfac_h_l_t_flatten)
                
#                 h_l_diff_1 = h_l_diff_1.reshape(h_l_diff_1.size(0), h_l_diff_1.size(1), h_l_diff_1.size(2) * h_l_diff_1.size(3))
                
#                 h_l_diff_2 = h_l_diff_2.reshape(h_l_diff_2.size(0), h_l_diff_2.size(1), h_l_diff_2.size(2) * h_l_diff_2.size(3))
                
#                 h_l_diff_1 = h_l_diff_1.permute(2,1,0)
#                 h_l_diff_2 = h_l_diff_2.permute(2,0,1)
                
#                 h_l_diff_1 = h_l_diff_1.data
#                 h_l_diff_2 = h_l_diff_2.data
                
#                 torch.bmm(h_l_diff_1, h_l_diff_2)
                
#                 sys.exit()



                
#                 weight_conv1d = torch.ones(1, 1, h_l_padded.size(2)-kernel_size+1, h_l_padded.size(3)-kernel_size+1, device=device)
                
                h_l_square_conv = F.conv1d(
                    h_l_square.view(h_l_square.size(0) * h_l_square.size(1), 1, *(h_l_square.size()[2:])),
                    weight_conv1d
                )
                
                
                h_l_square_conv = h_l_square_conv.view(h_l_square.size(0), h_l_square.size(1), *(h_l_square_conv.size()[2:]))
                
#                 print('h_l_square_conv.size()')
#                 print(h_l_square_conv.size())
            
                if delta_2_0 > 0:
                    start_index_0 = [0, delta_2_0]
                else:
                    start_index_0 = [-delta_2_0, 0]

                if delta_2_1 > 0:
                    start_index_1 = [0, delta_2_1]
                else:
                    start_index_1 = [-delta_2_1, 0]

                for delta_0 in range(kernel_size-np.abs(delta_2_0)):
                    for delta_1 in range(kernel_size-np.abs(delta_2_1)):
                        test_5_A_j[
                            :, start_index_0[0] + delta_0, start_index_1[0] + delta_1,
                            :, start_index_0[1] + delta_0, start_index_1[1] + delta_1
                        ] = h_l_square_conv[:, :, delta_0, delta_1].t()

        test_5_A_j = test_5_A_j.view(
        test_5_A_j.size(0) * test_5_A_j.size(1) * test_5_A_j.size(2),
        test_5_A_j.size(3) * test_5_A_j.size(4) * test_5_A_j.size(5),
    )  

        
#         print('test_5_A_j.requires_grad before homo')
#         print(test_5_A_j.requires_grad)
        
        
        if params['Kron_BFGS_if_homo']:
            homo_test_5_A_j = torch.zeros(
                test_5_A_j.size(0)+1, test_5_A_j.size(1)+1, device=device
            )
            
            homo_test_5_A_j[:-1, :-1] = test_5_A_j
            
#             print('h_l_padded.size()')
#             print(h_l_padded.size())

            sum_h_l_padded = torch.sum(h_l_padded, dim=0)
            
#             print('sum_h_l_padded.size()')
#             print(sum_h_l_padded.size())
            
            sum_h_l_padded = sum_h_l_padded.unsqueeze(1)

#             print('homo_test_5_A_j.requires_grad before conv1d')
#             print(homo_test_5_A_j.requires_grad)
            
            weight_conv1d = torch.ones(1, 1, T0, T1, device=device) 
            
#             print('sum_h_l_padded.requires_grad')
#             print(sum_h_l_padded.requires_grad)
#             print('weight_conv1d.requires_grad')
#             print(weight_conv1d.requires_grad)
            
            homo_test_5_A_j[-1, :-1] =\
            F.conv1d(sum_h_l_padded, weight_conv1d).view(-1)
            
#             print('homo_test_5_A_j.requires_grad after conv1d')
#             print(homo_test_5_A_j.requires_grad)
            
            homo_test_5_A_j[:-1, -1] = homo_test_5_A_j[-1, :-1]
            
            homo_test_5_A_j[-1, -1] = T0 * T1 * h_l_padded.size(0)
            
            test_5_A_j = homo_test_5_A_j
        
#     print('test_5_A_j.requires_grad')
#     print(test_5_A_j.requires_grad)
        
        
    return test_5_A_j








def get_A_A_T_kron_bfgs(h, l, params):
    
    layers_params = params['layers_params']
    device = params['device']
                        
    in_channels = layers_params[l]['conv_in_channels']
    kernel_size = layers_params[l]['conv_kernel_size']
    padding = layers_params[l]['conv_padding']

    if layers_params[l]['name'] == '1d-conv':

        test_5_A_j = torch.zeros(
            in_channels,
            kernel_size,
            in_channels,
            kernel_size,
            device=device
        )

        h_l_padded = F.pad(h[l], (padding, padding), "constant", 0)

        for delta_2 in range(-kernel_size+1, kernel_size):
            h_l_diff = h_l_padded[
                :,
                :,
                np.remainder(np.asarray(range(h[l].size(2)))-delta_2,h[l].size(2))
                                 ]


            cutting_indices_2 = np.arange(
                    np.maximum(0,delta_2),
                    np.minimum(h_l_padded.size(2),h_l_padded.size(2)+delta_2)
                )


            cutting_indices =\
            np.ix_(
                np.asarray(range(h_l_padded.size(0))),
                np.asarray(range(h_l_padded.size(1))),
                cutting_indices_2
            )

            h_l_diff_1 = h_l_padded[cutting_indices]
            h_l_diff_2 = h_l_diff[cutting_indices]
            einsum_ = torch.einsum('ijl,ikl->ijkl', h_l_diff_1.data, h_l_diff_2.data)


            h_l_square = torch.sum(einsum_, dim=0)

            sum_h_l_square = torch.sum(h_l_square, dim=-1)
            for delta in range(kernel_size-np.abs(delta_2)):

                # the length of minus part is:
                # kernel_size-np.abs(delta_2) - 1

                indices_dim_2 = np.arange(0, h_l_square.size(2))
                indcies_dim_2_included =\
                np.arange(0, h_l_square.size(2) - (kernel_size-np.abs(delta_2) - 1)) + delta

                indcies_dim_2_excludes =\
                np.setdiff1d(indices_dim_2, indcies_dim_2_included)

                if delta_2 > 0:
                    start_index = [0, delta_2]
                else:
                    start_index = [-delta_2, 0]

                test_5_A_j[
                    :, start_index[0] + delta, :, start_index[1] + delta
                ] = (
                    sum_h_l_square-\
                    torch.sum(
                        h_l_square[
                            :, :, indcies_dim_2_excludes
                        ],
                        dim=-1
                    )
                ).t()

        test_5_A_j = test_5_A_j.view(
        test_5_A_j.size(0) * test_5_A_j.size(1),
        test_5_A_j.size(2) * test_5_A_j.size(3),
    )

    elif layers_params[l]['name'] == 'conv':
        test_5_A_j = torch.zeros(
            in_channels,
            kernel_size,
            kernel_size,
            in_channels,
            kernel_size,
            kernel_size,
            device=device
        )

        h_l_padded = F.pad(
            h[l], (padding, padding, padding, padding), "constant", 0
        )
        for delta_2_0 in range(-kernel_size+1, kernel_size):
            for delta_2_1 in range(-kernel_size+1, kernel_size):

                diff_indices =\
                np.ix_(
                    np.asarray(range(h_l_padded.size(0))),
                    np.asarray(range(h_l_padded.size(1))),
                    np.remainder(
                        np.asarray(range(h_l_padded.size(2)))-delta_2_0,
                        h_l_padded.size(2)
                    ),
                    np.remainder(
                        np.asarray(range(h_l_padded.size(3)))-delta_2_1,
                        h_l_padded.size(3)
                    )
                )

                h_l_diff = h_l_padded[diff_indices]

                cutting_indices_2 = np.arange(
                    np.maximum(0,delta_2_0),
                    np.minimum(h_l_padded.size(2),h_l_padded.size(2)+delta_2_0)
                )

                cutting_indices_3 = np.arange(
                    np.maximum(0,delta_2_1),
                    np.minimum(h_l_padded.size(3),h_l_padded.size(3)+delta_2_1)
                )


                cutting_indices =\
                np.ix_(
                    np.asarray(range(h_l_padded.size(0))),
                    np.asarray(range(h_l_padded.size(1))),
                    cutting_indices_2,
                    cutting_indices_3
                )

                h_l_diff_1 = h_l_padded[cutting_indices]

                h_l_diff_2 = h_l_diff[cutting_indices]

                einsum_ = torch.einsum(
                    'ijlt,iklt->ijklt', h_l_diff_1.data, h_l_diff_2.data
                )

                h_l_square = torch.sum(einsum_, dim=0)
                
#                 print('h_l_square.size()')
#                 print(h_l_square.size())
#                 sys.exit()
                
                sum_h_l_square_dim_2 = torch.sum(h_l_square, dim=[2]) # dim 3 remains open
                sum_h_l_square_dim_3 = torch.sum(h_l_square, dim=[3])

                sum_h_l_square = torch.sum(h_l_square, dim=[-1, -2])


                for delta_0 in range(kernel_size-np.abs(delta_2_0)):
                    for delta_1 in range(kernel_size-np.abs(delta_2_1)):

                        if delta_2_0 > 0:
                            start_index_0 = [0, delta_2_0]
                        else:
                            start_index_0 = [-delta_2_0, 0]

                        if delta_2_1 > 0:
                            start_index_1 = [0, delta_2_1]
                        else:
                            start_index_1 = [-delta_2_1, 0]


                        indices_dim_2 = np.arange(0, h_l_square.size(2))
                        indcies_dim_2_included =\
                        np.arange(0, h_l_square.size(2) - (kernel_size-np.abs(delta_2_0) - 1)) + delta_0

                        indcies_dim_2_excludes =\
                        np.setdiff1d(indices_dim_2, indcies_dim_2_included)

                        indices_dim_3 = np.arange(0, h_l_square.size(3))
                        indcies_dim_3_included =\
                        np.arange(0, h_l_square.size(3) - (kernel_size-np.abs(delta_2_1) - 1)) + delta_1

                        indcies_dim_3_excludes =\
                        np.setdiff1d(indices_dim_3, indcies_dim_3_included)


                        # this should be used for sum_h_l_square_dim_2
                        slicing_indices_minus_1 =\
                np.ix_(
                    np.asarray(range(h_l_square.size(0))),
                    np.asarray(range(h_l_square.size(1))),
                    indcies_dim_3_excludes
                )

                        # this should be used for sum_h_l_square_dim_3
                        slicing_indices_minus_2 =\
                np.ix_(
                    np.asarray(range(h_l_square.size(0))),
                    np.asarray(range(h_l_square.size(1))),
                    indcies_dim_2_excludes
                )

                        slicing_indices_plus =\
                np.ix_(
                    np.asarray(range(h_l_square.size(0))),
                    np.asarray(range(h_l_square.size(1))),
                    indcies_dim_2_excludes,
                    indcies_dim_3_excludes
                )
                        
#                         print('sum_h_l_square_dim_2.size()')
#                         print(sum_h_l_square_dim_2.size())
                        
#                         print('sum_h_l_square_dim_3.size()')
#                         print(sum_h_l_square_dim_3.size())
                        
#                         sys.exit()


                        test_5_A_j[
                            :, 
                            start_index_0[0]+delta_0,
                            start_index_1[0]+delta_1,
                            :,
                            start_index_0[1]+delta_0,
                            start_index_1[1]+delta_1
                        ] = sum_h_l_square            
        
                        test_5_A_j[
                            :, 
                            start_index_0[0]+delta_0,
                            start_index_1[0]+delta_1,
                            :,
                            start_index_0[1]+delta_0,
                            start_index_1[1]+delta_1
                        ] -=\
                        torch.sum(sum_h_l_square_dim_2[slicing_indices_minus_1], dim=[-1])
                   
        
                        test_5_A_j[
                            :, 
                            start_index_0[0]+delta_0,
                            start_index_1[0]+delta_1,
                            :,
                            start_index_0[1]+delta_0,
                            start_index_1[1]+delta_1
                        ] -=\
                        torch.sum(sum_h_l_square_dim_3[slicing_indices_minus_2], dim=[-1])
                        
                
            
                        test_5_A_j[
                            :, 
                            start_index_0[0]+delta_0,
                            start_index_1[0]+delta_1,
                            :,
                            start_index_0[1]+delta_0,
                            start_index_1[1]+delta_1
                        ] +=\
                        torch.sum(h_l_square[slicing_indices_plus], dim=[-2, -1])
                
                        
    
                        test_5_A_j[
                            :, 
                            start_index_0[0]+delta_0,
                            start_index_1[0]+delta_1,
                            :,
                            start_index_0[1]+delta_0,
                            start_index_1[1]+delta_1
                        ] =\
                        copy.deepcopy(test_5_A_j[
                            :, 
                            start_index_0[0]+delta_0,
                            start_index_1[0]+delta_1,
                            :,
                            start_index_0[1]+delta_0,
                            start_index_1[1]+delta_1
                        ].t())
        
                        '''
                        if l == 1:
                            
                        
                            print('(\
                        sum_h_l_square\
                       -torch.sum(sum_h_l_square_dim_2[slicing_indices_minus_1], dim=[-1])\
                       -torch.sum(sum_h_l_square_dim_3[slicing_indices_minus_2], dim=[-1])\
                       +torch.sum(h_l_square[slicing_indices_plus], dim=[-2, -1])\
                    ).t() -\
                            test_5_A_j[\
                                :, \
                                start_index_0[0]+delta_0,\
                                start_index_1[0]+delta_1,\
                                :,\
                                start_index_0[1]+delta_0,\
                                start_index_1[1]+delta_1\
                            ]')
                            
                            print((
                        sum_h_l_square\
                       -torch.sum(sum_h_l_square_dim_2[slicing_indices_minus_1], dim=[-1])\
                       -torch.sum(sum_h_l_square_dim_3[slicing_indices_minus_2], dim=[-1])\
                       +torch.sum(h_l_square[slicing_indices_plus], dim=[-2, -1])
                    ).t() -\
                            test_5_A_j[
                                :, 
                                start_index_0[0]+delta_0,
                                start_index_1[0]+delta_1,
                                :,
                                start_index_0[1]+delta_0,
                                start_index_1[1]+delta_1
                            ])
                            
                            print('test_5_A_j[\
                                :, \
                                start_index_0[0]+delta_0,\
                                start_index_1[0]+delta_1,\
                                :,\
                                start_index_0[1]+delta_0,\
                                start_index_1[1]+delta_1\
                            ].size()')
                            print(test_5_A_j[
                                :, 
                                start_index_0[0]+delta_0,
                                start_index_1[0]+delta_1,
                                :,
                                start_index_0[1]+delta_0,
                                start_index_1[1]+delta_1
                            ].size())
                            
                            sys.exit()
                            


                        test_5_A_j[
                            :, 
                            start_index_0[0]+delta_0,
                            start_index_1[0]+delta_1,
                            :,
                            start_index_0[1]+delta_0,
                            start_index_1[1]+delta_1
                        ] = (
                    sum_h_l_square\
                   -torch.sum(sum_h_l_square_dim_2[slicing_indices_minus_1], dim=[-1])\
                   -torch.sum(sum_h_l_square_dim_3[slicing_indices_minus_2], dim=[-1])\
                   +torch.sum(h_l_square[slicing_indices_plus], dim=[-2, -1])
                ).t()
                '''
                
    
#         if l == 1:
#             sys.exit()

        test_5_A_j = test_5_A_j.view(
        test_5_A_j.size(0) * test_5_A_j.size(1) * test_5_A_j.size(2),
        test_5_A_j.size(3) * test_5_A_j.size(4) * test_5_A_j.size(5),
    )  
        
    return test_5_A_j



def train_initialization(data_, params, args):
    algorithm = params['algorithm']
    
    if params['N2'] > params['N1']:
        print('Error! 1432')
        sys.exit()
        
    params['if_grafting'] = args['if_grafting']
    
    params['weight_decay'] = args['weight_decay']
        
    if params['if_lr_decay']:
        params['num_epoch_to_decay'] = args['num_epoch_to_decay']
        params['lr_decay_rate'] = args['lr_decay_rate']
        
#     import utils_git.utils_kbfgs as utils_kbfgs
    import utils_git.utils_shampoo as utils_shampoo
    import utils_git.utils_kfac as utils_kfac

    if algorithm in utils_kfac.list_algorithm:
        
        params['kfac_if_svd'] = args['kfac_if_svd']
        
        params['kfac_if_update_BN'] = args['kfac_if_update_BN']
        params['kfac_if_BN_grad_direction'] = args['kfac_if_BN_grad_direction']
        
        if params['kfac_if_update_BN'] == False and params['weight_decay'] != 0:
            print('error: only work if weight_decay == 0')
            sys.exit()
        
        if algorithm in ['kfac-no-max-no-LM',
                         'kfac-warmStart-no-max-no-LM',
                         'kfac-correctFisher-warmStart-no-max-no-LM',
                         'kfac-correctFisher-warmStart-NoMaxNoSqrt-no-LM',
                         'kfac-correctFisher-warmStart-lessInverse-no-max-no-LM',
                         'kfac-correctFisher-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                         'kfac-warmStart-lessInverse-no-max-no-LM',
                         'kfac-warmStart-lessInverse-NoMaxNoSqrt-no-LM']:
            params['Kron_BFGS_if_homo'] = True
        
        if algorithm in ['ekfac-EF-VA',
                         'ekfac-EF',
                         'kfac-EF']:
            print('error: need to check warm start')
            sys.exit()

        if params['algorithm'] in ['kfac-TR',
                                   'kfac-momentum-grad-TR']:
            params['TR_max_iter'] = args['TR_max_iter']
        if params['algorithm'] in ['kfac-CG',
                                   'kfac-momentum-grad-CG']:
            params['CG_max_iter'] = args['CG_max_iter']
            
        if params['algorithm'] in ['kfac-no-max-no-LM',
                                   'kfac-warmStart-no-max-no-LM',
                                   'kfac-correctFisher-warmStart-no-max-no-LM',
                                   'kfac-correctFisher-warmStart-NoMaxNoSqrt-no-LM',
                                   'kfac-correctFisher-warmStart-lessInverse-no-max-no-LM',
                                   'kfac-correctFisher-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                                   'kfac-warmStart-lessInverse-no-max-no-LM',
                                   'kfac-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                                   'kfac-NoMaxNoSqrt-no-LM']:
            params['kfac_damping_lambda'] = args['kfac_damping_lambda']
            
        if params['algorithm'] == 'kfac-no-max-epsilon-A-G-no-LM':
            params['kfac_A_epsilon'] = args['kfac_A_epsilon']
            params['kfac_G_epsilon'] = args['kfac_G_epsilon']
            
        

        device = params['device']
        layersizes = params['layersizes']
        numlayers = params['numlayers']
        
        layers_params = params['layers_params']

        A = []  # KFAC A
        G = []  # KFAC G


        for l in range(numlayers):
            if params['layers_params'][l]['name'] == 'fully-connected':
                
                input_size = params['layers_params'][l]['input_size']
                output_size = params['layers_params'][l]['output_size']
#                 assert layersizes[l] == input_size
#                 assert layersizes[l+1] == output_size
            
#                 A.append(torch.zeros(layersizes[l] + 1, layersizes[l] + 1, device=device))
                
                A.append(torch.zeros(input_size + 1, input_size + 1, device=device))
                
                
#                 G.append(torch.zeros(layersizes[l+1], layersizes[l+1], device=device))
                G.append(torch.zeros(output_size, output_size, device=device))
                
            elif params['layers_params'][l]['name'] in ['conv',
                                                        'conv-no-activation',
                                                        'conv-no-bias-no-activation',]:
            
                size_A = layers_params[l]['conv_in_channels'] *\
                layers_params[l]['conv_kernel_size']**2
                
                if params['layers_params'][l]['name'] in ['conv',
                                                          'conv-no-activation',]:
                
                    size_A += 1
            
                A.append(torch.zeros(size_A, size_A, device=device))
                
                size_G = layers_params[l]['conv_out_channels']
                
                G.append(torch.zeros(size_G, size_G, device=device))
            elif params['layers_params'][l]['name'] in ['BN']:
                
#                 print('layers_params[l]')
#                 print(layers_params[l])
                
#                 sys.exit()
                
                A.append([])
                
                size_G = layers_params[l]['num_features'] * 2
                
                G.append(torch.zeros(size_G, size_G, device=device))
            else:
                print('Error: unsupported layer when initial cache for ' + params['layers_params'][l]['name'])
                sys.exit()

        data_['A'] = A
        data_['G'] = G


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
            
#             print('params[kfac_if_svd]')
#             print(params['kfac_if_svd'])
            
#             sys.exit()
            
            if params['kfac_if_svd']:
                
                U_A, U_G = numlayers * [0], numlayers * [0]
                
                data_['U_A'] = U_A
                data_['U_G'] = U_G
                
                s_A, s_G = numlayers * [0], numlayers * [0]
                
                data_['s_A'] = s_A
                data_['s_G'] = s_G
                
            else:
            
                A_inv, G_inv = numlayers * [0], numlayers * [0]

                data_['A_inv'] = A_inv
                data_['G_inv'] = G_inv

        if params['algorithm'] == 'ekfac-EF-VA' or\
        params['algorithm'] == 'ekfac-EF':
            U_A, U_G = model.numlayers * [0], model.numlayers * [0]
            data_['U_A'] = U_A
            data_['U_G'] = U_G

            data_['ekfac_s'] = get_zero_torch(params)

            if params['algorithm'] == 'ekfac-EF-VA':
                data_['ekfac_m'] = get_zero_torch(params)

        params['kfac_inverse_update_freq'] = args['kfac_inverse_update_freq']
        params['kfac_cov_update_freq'] = args['kfac_cov_update_freq']
        params['kfac_rho'] = args['kfac_rho']
        
        get_warm_start(data_, params)
        
    elif algorithm in ['RMSprop',
                       'RMSprop-warmStart',
                       'RMSprop-test',
                       'Adam',
                       'Adam-test',
                       'Adam-noWarmStart',
                       'RMSprop-no-sqrt',
                       'RMSprop-individual-grad',
                       'RMSprop-individual-grad-no-sqrt',
                       'RMSprop-individual-grad-no-sqrt-Fisher',
                       'RMSprop-individual-grad-no-sqrt-LM']:
        
        params['RMSprop_epsilon'] = args['RMSprop_epsilon']
        
        params['RMSprop_beta_2'] = args['RMSprop_beta_2']
        
        data_['RMSprop_momentum_2'] = get_zero_torch(params)
        
        if algorithm in ['RMSprop-warmStart',
                         'Adam',
                         'Adam-test']:
        
            N1 = params['N1']
            device = params['device']
            model = data_['model']
            if N1 < params['num_train_data']:
                # i.e. stochastic setting

                i = 0 # position of training data
                j = 0 # position of mini-batch

                while i + N1 <= params['num_train_data']:



                    X_mb, t_mb = data_['dataset'].train.next_batch(N1)
                    X_mb = torch.from_numpy(X_mb).to(device)
                    t_mb = torch.from_numpy(t_mb).to(device)



                    z, a, h = model.forward(X_mb)
                    loss = get_loss_from_z(model, z, t_mb, reduction='mean') # not regularized

                    model.zero_grad()
                    loss.backward()

                    model_grad = get_model_grad(model, params)

                    if params['if_regularized_grad'] and params['tau'] != 0:


                        model_grad = get_plus_torch(
                        model_grad,
                        get_multiply_scalar_no_grad(params['tau'], model.layers_weight)
                        )
                    else:
                        1

                    i += N1
                    j += 1




    #                 sys.exit()

    #                 for l in range(numlayers):
                        # bar_A_j = 1 / j * (A_1 + ... + A_j)
                        # bar_A_j = (j-1) / j * bar_A_{j-1} + 1 / j * A_j

    #                     homo_h_l = torch.cat((h[l], torch.ones(N1, 1, device=device)), dim=1)

    #                     A_j = 1/N1 * torch.mm(homo_h_l.t(), homo_h_l).data

    #                     data_['A'][l] *= (j-1)/j
                    data_['RMSprop_momentum_2'] = get_multiply_scalar(
                            (j-1)/j, data_['RMSprop_momentum_2']
                        )

    #                     data_['A'][l] += 1/j * A_j
                    data_['RMSprop_momentum_2'] = get_plus_torch(
                            data_['RMSprop_momentum_2'],
                            get_multiply_scalar(1/j, get_square_torch(model_grad))
                        )

    
                
            

        
        
    elif params['algorithm'] in utils_shampoo.list_algorithm:
        '''
    elif params['algorithm'] in ['shampoo',
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
                                 'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-lessInverse',]:
                                 '''
        
        params['if_Hessian_action'] = args['if_Hessian_action']
        
#         if not params['if_Hessian_action']:

        params['shampoo_inverse_freq'] = args['shampoo_inverse_freq']
    
        params['shampoo_update_freq'] = args['shampoo_update_freq']
        
        params['shampoo_decay'] = args['shampoo_decay']
        params['shampoo_weight'] = args['shampoo_weight']
        
        if params['algorithm'] in ['matrix-normal-allVariables-warmStart-MaxEigDamping',
                                   'matrix-normal-same-trace-allVariables-warmStart-AvgEigDamping',
                                   'matrix-normal-same-trace-allVariables-warmStart-MaxEigDamping',]:
            pass
        elif params['algorithm'] in ['shampoo-allVariables-warmStart',
                                     'shampoo-allVariables-warmStart-lessInverse',
                                     'shampoo-allVariables-filterFlattening-warmStart',
                                     'shampoo-allVariables-filterFlattening-warmStart-lessInverse',
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
            params['shampoo_epsilon'] = args['shampoo_epsilon']
        else:
            print('params[algorithm]')
            print(params['algorithm'])
            sys.exit()
        
            
        
        numlayers = params['numlayers']
        
        data_['shampoo_H'] = []
        for l in range(numlayers):
            data_['shampoo_H'].append({})
    
        data_['shampoo_H_LM_minus_2k'] = []
        for l in range(numlayers):
            data_['shampoo_H_LM_minus_2k'].append({})
    
        data_['shampoo_H_trace'] = []
        for l in range(numlayers):
            data_['shampoo_H_trace'].append({})
            
        if params['algorithm'] in ['matrix-normal-allVariables-warmStart',
                                   'matrix-normal-allVariables-warmStart-MaxEigDamping',
                                   'matrix-normal-allVariables-warmStart-noPerDimDamping',
                                   'matrix-normal-same-trace-warmStart',
                                   'matrix-normal-same-trace-warmStart-noPerDimDamping',
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
                                   'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart',
                                   'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-lessInverse',
                                   'shampoo-allVariables-warmStart',
                                   'shampoo-allVariables-warmStart-lessInverse',
                                   'shampoo-allVariables-filterFlattening-warmStart',
                                   'shampoo-allVariables-filterFlattening-warmStart-lessInverse',]:
            params['if_warm_start'] = True
        elif params['algorithm'] in ['matrix-normal-allVariables',
                                     'matrix-normal-same-trace',
                                     'matrix-normal-same-trace-allVariables',
                                     'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping',
                                     'shampoo-allVariables']:
            params['if_warm_start'] = False
        else:
            print('error: need to check for ' + params['algorithm'])
            sys.exit()
            
        
        if params['if_warm_start']:
            get_warm_start(data_, params)
            
    elif algorithm in ['Fisher-BD',]:
        
        params['Fisher_BD_damping'] = args['Fisher_BD_damping']
        
        data_['block_Fisher'] = []
        for l in range(params['numlayers']):
            data_['block_Fisher'].append([])
        
        get_warm_start(data_, params)

    elif params['algorithm'] in ['SMW-Fisher-batch-grad-momentum-exponential-decay',
                                 'SMW-Fisher-batch-grad-momentum']:
        params['N_iters'] = 30

        from collections import deque
        data_['batch_grads'] = deque()
        data_['batch_grads_a_grad'] = deque()
        data_['batch_grads_h'] = deque()

        print('test batch_grads_a_grad')

        data_['batch_grads_test'] = []

        print('test batch_grads_test')


        data_['D_t_minus_lambda'] = []

    elif params['algorithm'] == 'SMW-Fisher-momentum':
        a_grad_momentum = []
        h_momentum = []

        layersizes = data_['model'].layersizes

        for l in range(model.numlayers):
            a_grad_momentum.append(
                torch.zeros(N2, layersizes[l+1], device=params['device']))
            h_momentum.append(torch.zeros(N2, layersizes[l], device=params['device']))

        data_['a_grad_momentum'] = a_grad_momentum
        data_['h_momentum'] = h_momentum

        D_t_inv = np.zeros((N2, N2))
        data_['D_t_inv'] = D_t_inv

    elif params['algorithm'] == 'SMW-Fisher-D_t-momentum':
        data_['J_J_transpose'] = np.float32(np.zeros((N2, N2)))

    elif params['algorithm'] == 'SMW-Fisher-momentum-D_t-momentum':
        a_grad_momentum = []
        h_momentum = []

        layersizes = params['layersizes']

        for l in range(model.numlayers):
            a_grad_momentum.append(torch.zeros(N2, layersizes[l+1]))
            h_momentum.append(torch.zeros(N2, layersizes[l]))

        data_['a_grad_momentum'] = a_grad_momentum
        data_['h_momentum'] = h_momentum

        data_['J_J_transpose'] = np.float32(np.zeros((N2, N2)))

    elif algorithm in ['SMW-Fisher-signVAsqrt-p',
                       'SMW-Fisher-VA-p',
                       'SMW-Fisher-momentum-p-sign',
                       'SMW-Fisher-momentum-p',
                       'SMW-Fisher-sign',
                       'SMW-Fisher-different-minibatch',
                       'SMW-Fisher',
                       'SGD-VA',
                       'SGD-yura-BD',
                       'SGD-yura-old',
                       'SGD-yura',
                       'SGD-sign',
                       'SGD-signVAerf',
                       'SGD-signVA',
                       'SGD-signVAsqrt',
                       'SGD-momentum-yura',
                       'SGD-momentum',
                       'SGD',
                       'SMW-GN',
                       'GI-Fisher',
                       'SMW-Fisher-BD',
                       'Kron-SGD',
                       'BFGS',
                       'BFGS-homo']:
        
        pass
    
        
    elif algorithm == 'Kron-BFGS-1st-layer-only':
        print('Error: this algo is abandoned')
        sys.exit()
        
    elif algorithm == 'SGD-yura-MA':
        params['yura_lambda_second_term_MA'] = 0
        params['yura_lambda_second_term_MA_weight'] =\
        args['yura_lambda_second_term_MA_weight']
    elif params['algorithm'] == 'SMW-Fisher-batch-grad':
        params['N_iters'] = 30
    else:
        print('Error: momentum variables not defined for ' + params['algorithm'])
        sys.exit()
        
        
    return data_, params





def train(args):
    
    print('\n')
    print('learning rate = {}'.format(args['alpha']))
    
    assert os.path.isdir(args['home_path'])
    
#     import datetime
#     current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
#     with open(args['home_path'] + 'notebooks/lr_record_' + current_time + '.txt', 'w') as fp:
#         fp.write('algorithm = {}\n'.format(args['algorithm']))
#         fp.write('learning rate = {}\n'.format(args['alpha']))
        

        

#     print('args')
#     print(args)
    
    from utils_git.utils_kfac import kfac_update
#     from utils_git.utils_kbfgs import Kron_BFGS_update_v2
    from utils_git.utils_shampoo import shampoo_update
    
    params = {}
    
    from utils_git.utils_shampoo import list_algorithm as list_algorithm_shampoo
    
    params['list_algorithm_shampoo'] = list_algorithm_shampoo
    
    from utils_git.utils_kfac import list_algorithm as list_algorithm_kfac
    
    params['list_algorithm_kfac'] = list_algorithm_kfac
    
    torch.cuda.empty_cache()
    
    params['initialization_pkg'] = args['initialization_pkg']
    
#     seed_number = 9999
    seed_number = args['seed_number']
    
    
    params['seed_number'] = seed_number

    np.random.seed(seed_number)
    torch.manual_seed(seed_number)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    num_threads = args['num_threads']
    params['num_threads'] = num_threads
    
#     print('num_threads')
#     print(num_threads)
    
#     print('num_threads == float(inf)')
#     print(num_threads == float('inf'))
    
    if num_threads == float('inf'):
        1
    else:
    
        torch.set_num_threads(num_threads)
        assert torch.get_num_threads() == num_threads
        
#     print('torch.get_num_threads()')
#     print(torch.get_num_threads())
    
    
    
#     sys.exit()


    

    params['home_path'] = args['home_path']
    params['if_gpu'] = args['if_gpu']
    params['if_test_mode'] = args['if_test_mode']

    if params['if_gpu'] and torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"  
    device = torch.device(dev)  
    params['device'] = device
    

    params['algorithm'] = args['algorithm']

    
    
    
    
    matrix_name = args['matrix_name']
    params['matrix_name'] = matrix_name
    
    params['if_records'] = {}

    params['if_record_sgd_norm'] = args['if_record_sgd_norm']
    params['if_record_p_norm'] = args['if_record_p_norm']
    params['if_record_kron_bfgs_cosine'] = args['if_record_kron_bfgs_cosine']
    params['if_record_kfac_p_norm'] = args['if_record_kfac_p_norm']
    params['if_record_kfac_p_cosine'] = args['if_record_kfac_p_cosine']
    params['if_record_res_grad_norm'] = args['if_record_res_grad_norm']
    params['if_record_res_grad_random_norm'] = args['if_record_res_grad_random_norm']
    params['if_record_res_grad_grad_norm'] = args['if_record_res_grad_grad_norm']
    params['if_record_res_grad_norm_per_iter'] = args['if_record_res_grad_norm_per_iter']
    params['if_record_sgn_norm'] = args['if_record_sgn_norm']
    
    if params['if_test_mode']:
        if 'if_record_kron_bfgs_update_status' in args:
            params['if_record_kron_bfgs_update_status'] =\
            args['if_record_kron_bfgs_update_status']
        if 'if_record_kron_bfgs_matrix_norm_per_iter' in args:
            params['if_record_kron_bfgs_matrix_norm_per_iter'] =\
            args['if_record_kron_bfgs_matrix_norm_per_iter']
        if 'if_record_loss_per_iter' in args:
            params['if_record_loss_per_iter'] =\
            args['if_record_loss_per_iter']
        if 'if_record_kfac_G_inv_norm_per_iter' in args:
            params['if_record_kfac_G_inv_norm_per_iter'] =\
            args['if_record_kfac_G_inv_norm_per_iter']
        if 'if_record_kfac_G_inv_norm_per_epoch' in args:
            params['if_record_kfac_G_inv_norm_per_epoch'] =\
            args['if_record_kfac_G_inv_norm_per_epoch']
            
        if 'if_record_kfac_G_norm_per_epoch' in args:
            params['if_records']['if_record_kfac_G_norm_per_epoch'] =\
            args['if_record_kfac_G_norm_per_epoch']
        else:
            params['if_records']['if_record_kfac_G_norm_per_epoch'] = False
            
        if 'if_record_kfac_G_twoNorm_per_epoch' in args:
            params['if_records']['if_record_kfac_G_twoNorm_per_epoch'] =\
            args['if_record_kfac_G_twoNorm_per_epoch']
        else:
            params['if_records']['if_record_kfac_G_twoNorm_per_epoch'] = False
            
        if 'if_record_kfac_A_twoNorm_per_epoch' in args:
            params['if_records']['if_record_kfac_A_twoNorm_per_epoch'] =\
            args['if_record_kfac_A_twoNorm_per_epoch']
        else:
            params['if_records']['if_record_kfac_A_twoNorm_per_epoch'] = False
            
        if 'if_record_kron_bfgs_A_twoNorm_per_epoch' in args:
            params['if_records']['if_record_kron_bfgs_A_twoNorm_per_epoch'] =\
            args['if_record_kron_bfgs_A_twoNorm_per_epoch']
        else:
            params['if_records']['if_record_kron_bfgs_A_twoNorm_per_epoch'] = False
            
        if 'if_record_kron_bfgs_G_LM_twoNorm_per_epoch' in args:
            params['if_records']['if_record_kron_bfgs_G_LM_twoNorm_per_epoch'] =\
            args['if_record_kron_bfgs_G_LM_twoNorm_per_epoch']
        else:
            params['if_records']['if_record_kron_bfgs_G_LM_twoNorm_per_epoch'] = False
            
        if 'if_record_kron_bfgs_Hg_twoNorm_per_epoch' in args:
            params['if_records']['if_record_kron_bfgs_Hg_twoNorm_per_epoch'] =\
            args['if_record_kron_bfgs_Hg_twoNorm_per_epoch']
        else:
            params['if_records']['if_record_kron_bfgs_Hg_twoNorm_per_epoch'] = False
            
        if 'if_record_kron_bfgs_Ha_twoNorm_per_epoch' in args:
            params['if_records']['if_record_kron_bfgs_Ha_twoNorm_per_epoch'] =\
            args['if_record_kron_bfgs_Ha_twoNorm_per_epoch']
        else:
            params['if_records']['if_record_kron_bfgs_Ha_twoNorm_per_epoch'] = False
            
        if 'if_record_kron_bfgs_matrix_norm' in args:
            params['if_records']['if_record_kron_bfgs_matrix_norm'] =\
            args['if_record_kron_bfgs_matrix_norm']
        else:
            params['if_records']['if_record_kron_bfgs_matrix_norm'] = False
            
        if 'if_record_layerWiseHessian_twoNorm_per_epoch' in args:
            params['if_records']['if_record_layerWiseHessian_twoNorm_per_epoch'] =\
            args['if_record_layerWiseHessian_twoNorm_per_epoch']
        else:
            params['if_records']['if_record_layerWiseHessian_twoNorm_per_epoch'] = False
            
        if 'if_record_inverseLayerWiseHessian_twoNorm_per_epoch' in args:
            params['if_records']['if_record_inverseLayerWiseHessian_twoNorm_per_epoch'] =\
            args['if_record_inverseLayerWiseHessian_twoNorm_per_epoch']
        else:
            params['if_records']['if_record_inverseLayerWiseHessian_twoNorm_per_epoch'] = False
            
        if 'if_record_inverseLayerWiseHessian_LM_twoNorm_per_epoch' in args:
            params['if_records']['if_record_inverseLayerWiseHessian_LM_twoNorm_per_epoch'] =\
            args['if_record_inverseLayerWiseHessian_LM_twoNorm_per_epoch']
        else:
            params['if_records']['if_record_inverseLayerWiseHessian_LM_twoNorm_per_epoch'] = False
            
        if 'if_record_inverseLayerWiseHessian_LM_MA_twoNorm_per_epoch' in args:
            params['if_records']['if_record_inverseLayerWiseHessian_LM_MA_twoNorm_per_epoch'] =\
            args['if_record_inverseLayerWiseHessian_LM_MA_twoNorm_per_epoch']
        else:
            params['if_records']['if_record_inverseLayerWiseHessian_LM_MA_twoNorm_per_epoch'] = False
            
        if 'if_record_kfac_F_twoNorm_per_epoch' in args:
            params['if_record_kfac_F_twoNorm_per_epoch'] =\
            args['if_record_kfac_F_twoNorm_per_epoch']
        if 'if_record_kron_bfgs_norm_s_y_per_iter' in args:
            params['if_record_kron_bfgs_norm_s_y_per_iter'] =\
            args['if_record_kron_bfgs_norm_s_y_per_iter']
        if 'if_record_kron_bfgs_sTy_per_iter' in args:
            params['if_record_kron_bfgs_sTy_per_iter'] =\
            args['if_record_kron_bfgs_sTy_per_iter']
        if 'if_record_kron_bfgs_damping_status' in args:
            params['if_record_kron_bfgs_damping_status'] =\
            args['if_record_kron_bfgs_damping_status']
        if 'if_record_kron_bfgs_check_damping' in args:
            params['if_record_kron_bfgs_check_damping'] =\
            args['if_record_kron_bfgs_check_damping']
            
            
            
    
    
    
    
    
    
    
    
    

    
    params['if_max_epoch'] = args['if_max_epoch']
    params['max_epoch/time'] = args['max_epoch/time']

    if_max_epoch = args['if_max_epoch'] # 0 means max_time
    if if_max_epoch:
        max_epoch = args['max_epoch/time']
    else:
        max_time = args['max_epoch/time']
        
    

    record_epoch = args['record_epoch']
    
    
    name_dataset = args['dataset']
    params['name_dataset'] = name_dataset

    params['name_loss'] = args['name_loss']
    

    params['momentum_gradient_rho'] = args['momentum_gradient_rho']
    params['momentum_gradient_dampening'] = args['momentum_gradient_dampening']



    

    # Model
    model = get_model(params)

    

    params['name_model'] = model.name_model
    params['layersizes'] = model.layersizes

    print('name_loss:')
    print(model.name_loss)

    

    print('Model created.')


    params['layers_params'] = model.layers_params
    
    params['N1'] = args['N1']
    params['N2'] = args['N2']


    data_ = {}


    if name_dataset in ['CIFAR-10-NoAugmentation-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
                        'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
                        'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
                        'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPool',
                        'CIFAR-10-onTheFly-N1-512-vgg16-NoAdaptiveAvgPoolNoDropout',
                        'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
                        'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-NoBias',
                        'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',
                        'CIFAR-10-onTheFly-ResNet32-BNNoAffine',
                        'CIFAR-10-onTheFly-ResNet32-BN',
                        'CIFAR-10-onTheFly-ResNet32-BN-BNshortcut',
                        'CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly',
                        'CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly-NoBias',
                        'CIFAR-10-onTheFly-N1-128-ResNet32-BNNoAffine-PaddingShortcutDownsampleOnly-NoBias-no-regularization',
                        'CIFAR-10-onTheFly-N1-128-ResNet32-BN-BNshortcutDownsampleOnly-NoBias',
                        'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',
                        'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias-no-regularization',
                        'CIFAR-100-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',
                        'CIFAR-100-onTheFly-ResNet34-BNNoAffine',
                        'CIFAR-100-onTheFly-ResNet34-BN',
                        'CIFAR-100-onTheFly-ResNet34-BN-BNshortcut',
                        'CIFAR-100-onTheFly-ResNet34-BN-BNshortcutDownsampleOnly',
                        'CIFAR-100-onTheFly-ResNet34-BN-BNshortcutDownsampleOnly-NoBias',
                        'CIFAR-100-onTheFly-N1-128-ResNet34-BN-BNshortcutDownsampleOnly-NoBias',
                        'CIFAR-100-onTheFly-N1-128-ResNet34-BN-PaddingShortcutDownsampleOnly-NoBias',
                        'CIFAR-10-AllCNNC',
                        'CIFAR-10-N1-128-AllCNNC',
                        'CIFAR-10-N1-512-AllCNNC',
                        'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
                        'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine-no-regularization',
                        'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
                        'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-no-regularization',
                        'CIFAR-100-onTheFly-vgg16-NoLinear-BN-no-regularization',
                        'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
                        'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',
                        'CIFAR-100-onTheFly-AllCNNC']:
        params['if_dataset_onTheFly'] = True
    elif name_dataset in ['Fashion-MNIST-N1-60-no-regularization',
                          'Fashion-MNIST-N1-256-no-regularization',
                          'CIFAR-100',
                          'CIFAR-100-NoAugmentation',
                          'CIFAR-100-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout',
                          'CIFAR-100-NoAugmentation-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
                          'CIFAR-10-NoAugmentation-vgg11',
                          'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout',
                          'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
                          'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',
                          'DownScaledMNIST-N1-1000-no-regularization',
                          'MNIST-autoencoder-relu-N1-1000-sum-loss',
                          'MNIST-autoencoder-relu-N1-1000-sum-loss-no-regularization',
                          'MNIST-autoencoder-relu-N1-100-sum-loss',
                          'CURVES-autoencoder-relu-sum-loss',
                          'CURVES-autoencoder-relu-sum-loss-no-regularization',
                          'CURVES-autoencoder-relu-N1-100-sum-loss',
                          'FacesMartens-autoencoder-relu',
                          'FacesMartens-autoencoder-relu-no-regularization',
                          'FacesMartens-autoencoder-relu-N1-100',]:
        params['if_dataset_onTheFly'] = False
    else:
        print('error: need to check if on the fly for ' + name_dataset)
        sys.exit()

    from utils_git.utils_data import read_data_sets

    
    
    if not params['if_dataset_onTheFly']:
        
        dataset = read_data_sets(name_dataset, params['name_model'], params['home_path'], one_hot=False)
    
#     sys.exit()

#         dataset = read_data_sets(name_dataset, params['name_model'], params['home_path'], one_hot=False)
    
    
    



        
        
        
        
        X_train = dataset.train.images
        t_train = dataset.train.labels

        print('For X_train:')
        get_statistics(X_train)

        

        print('X_train.shape')
        print(X_train.shape)
        print('t_train.shape')
        print(t_train.shape)

        




        params['num_train_data'] = len(t_train)
        
        X_test = dataset.test.images
        t_test = dataset.test.labels

#     print('For X_test:')
#     get_statistics(X_test)

#     print('X_test.shape')
#     print(X_test.shape)
#     print('t_test.shape')
#     print(t_test.shape)
    
    
    if params['if_dataset_onTheFly']:
        
#         print('dataset')
#         print(dataset)
#         sys.exit()
        
        from utils_git.utils_data import read_data_sets_v2
        
#         dataset = read_data_sets_v2(name_dataset, dataset, params)
        dataset = read_data_sets_v2(name_dataset, params)
    


        params['num_train_data'] = dataset.num_train_data
    
    
        
        

    
    
    
    data_['dataset'] = dataset
    
    

    params['alpha'] = args['alpha']
    params['alpha_current'] = params['alpha']
    
    
    params['numlayers'] = model.numlayers






    

    data_['model'] = model

    params = get_params(params, args)
    
    params['tau'] = args['tau']
    

    
    data_, params = train_initialization(data_, params, args)
    
#     print('N1 in params')
#     print('N1' in params)
#     sys.exit()
    
    
#     if params['algorithm'] in ['Kron-BFGS',
#                                'Kron-BFGS-no-norm-gate',
#                                'Kron-BFGS-Hessian-action',
#                                'Kron-BFGS-wrong',
#                                'Kron-BFGS-wrong-Hessian-action',
#                                'Kron-BFGS-no-norm-gate-damping',
#                                'Kron-BFGS-no-norm-gate-Shiqian-damping',
#                                'Kron-BFGS-no-norm-gate-momentum-s-y-damping']:
#         params['if_record_kron_bfgs_matrix_norm'] = True

    if params['if_momentum_gradient']:
#         data_['model_regularized_grad_momentum'] = get_zero_torch(params)
        
        data_['model_grad_momentum'] = get_zero_torch(params)
    
    if params['if_Adam']:
        
        print('error: should not reach here')
        sys.exit()

        params['Adam_beta_1'] = 0.9
        params['Adam_beta_2'] = 0.999
        params['Adam_epsilon'] = 10**(-8)

        data_['model_grad_Adam_momentum_1'] = get_zero(params)
        
        data_['model_grad_Adam_momentum_2'] = get_zero(params)

    

    

    

    

    if params['if_yura']:
        params['yura_lambda_0'] = 1

    if params['if_momentum_p']:
        # data_['p_momentum'] = get_zero(params)
        data_['p_momentum_torch'] = get_zero_torch(params)

    if params['if_VA_p'] or\
    params['if_signVAsqrt'] or\
    params['if_signVA'] or\
    params['if_signVAerf']:
        data_['p_momentum_1_torch'] = get_zero_torch(params)
        data_['p_momentum_2_torch'] = get_zero_torch(params)

    if params['if_LM']:
            
        boost = 1.01
        drop = 1 / 1.01
        params['boost'] = boost
        params['drop'] = drop




    # Visualization stuffs
    # len_record = int(max_epoch / record_epoch)

    epochs = [0]
    timesCPU = [0]
    timesWallClock = [0]
    
    if not params['if_dataset_onTheFly']:
        train_losses = []
        train_unregularized_losses = []
        train_acces = []
    
    train_unregularized_minibatch_losses = []
    train_minibatch_acces = []

    test_acces = []
    test_losses = []
    reduction = 'mean'
    
    
    
    if params['if_dataset_onTheFly']:
            test_loss_0, _, test_acc_0 = get_regularized_loss_and_acc_from_x_whole_dataset_with_generator(
                model, dataset.test_generator, reduction, params
            )
    else:
        test_loss_0, _, test_acc_0 = get_regularized_loss_and_acc_from_x_whole_dataset(
            model, X_test, t_test, reduction, params
        )
    
#     print('test_loss_0, test_acc_0')
#     print(test_loss_0, test_acc_0)
    
#     sys.exit()
    
    test_losses.append(test_loss_0)
    test_acces.append(test_acc_0)
    
    
    print('test_loss_0, test_acc_0')
    print(test_loss_0, test_acc_0)

    if params['if_LM']:
        lambdas = []
        lambdas.append(params['lambda_'])
    if params['if_yura']:
        yura_lambdas = []
        yura_lambdas.append(params['yura_lambda_0'])
    
    if params['if_test_mode']:
        if params['if_record_sgd_norm']:
            sgd_norms = []
        if params['if_record_p_norm']:
            p_norms = []
        if params['if_record_kfac_p_norm']:
            kfac_p_norms = []
            data_['kfac_p_norms'] = kfac_p_norms
        if params['if_record_kfac_p_cosine']:
            kfac_p_cosines = []
            data_['kfac_p_cosines'] = kfac_p_cosines
        if params['if_record_kron_bfgs_cosine']:
            kron_bfgs_cosines = []
            data_['kron_bfgs_cosines'] = kron_bfgs_cosines
        if params['if_record_res_grad_norm']:
            res_grad_norms = []
            data_['res_grad_norms'] = res_grad_norms
        if params['if_record_res_grad_random_norm']:
            res_grad_random_norms = []
            data_['res_grad_random_norms'] = res_grad_random_norms
        if params['if_record_res_grad_grad_norm']:
            res_grad_grad_norms = []
            data_['res_grad_grad_norms'] = res_grad_grad_norms
        if params['if_record_res_grad_norm_per_iter']:
            res_grad_norms_per_iter = []
            data_['res_grad_norms_per_iter'] = res_grad_norms_per_iter
        if 'if_record_kron_bfgs_update_status' in params and\
        params['if_record_kron_bfgs_update_status']:
            data_['kron_bfgs_update_status'] = []
        if 'if_record_kron_bfgs_matrix_norm_per_iter' in params and\
        params['if_record_kron_bfgs_matrix_norm_per_iter']:
            data_['kron_bfgs_matrix_norms_per_iter'] = []
        if 'if_record_loss_per_iter' in params and\
            params['if_record_loss_per_iter'] == True:
            data_['losses_per_iter'] = []
        if 'if_record_kfac_G_inv_norm_per_iter' in params and\
            params['if_record_kfac_G_inv_norm_per_iter'] == True:
            data_['kfac_G_inv_norms_per_iter'] = []
        if 'if_record_kfac_G_inv_norm_per_epoch' in params and\
            params['if_record_kfac_G_inv_norm_per_epoch'] == True:
            data_['kfac_G_inv_norms_per_epoch'] = []

        if params['if_records']['if_record_kfac_G_norm_per_epoch']:
            data_['kfac_G_norms_per_epoch'] = []
        if params['if_records']['if_record_kfac_G_twoNorm_per_epoch']:
            data_['kfac_G_twoNorms_per_epoch'] = []
        if params['if_records']['if_record_kfac_A_twoNorm_per_epoch']:
            data_['kfac_A_twoNorms_per_epoch'] = []
        if params['if_records']['if_record_kron_bfgs_A_twoNorm_per_epoch']:
            data_['kron_bfgs_A_twoNorms_per_epoch'] = []
        if params['if_records']['if_record_kron_bfgs_G_LM_twoNorm_per_epoch']:
            data_['kron_bfgs_G_LM_twoNorms_per_epoch'] = []
        if params['if_records']['if_record_kron_bfgs_Hg_twoNorm_per_epoch']:
            data_['kron_bfgs_Hg_twoNorms_per_epoch'] = []
        if params['if_records']['if_record_kron_bfgs_Ha_twoNorm_per_epoch']:
            data_['kron_bfgs_Ha_twoNorms_per_epoch'] = []
        if params['if_records']['if_record_layerWiseHessian_twoNorm_per_epoch'] == True:
            data_['layerWiseHessian_twoNorms_per_epoch'] = []
        if params['if_records']['if_record_inverseLayerWiseHessian_twoNorm_per_epoch'] == True:
            data_['inverseLayerWiseHessian_twoNorms_per_epoch'] = []
        if params['if_records']['if_record_inverseLayerWiseHessian_LM_twoNorm_per_epoch'] == True:
            data_['inverseLayerWiseHessian_LM_twoNorms_per_epoch'] = []
        if params['if_records']['if_record_inverseLayerWiseHessian_LM_MA_twoNorm_per_epoch'] == True:
            data_['inverseLayerWiseHessian_LM_MA_twoNorms_per_epoch'] = []
            
        if 'if_record_kfac_F_twoNorm_per_epoch' in params and\
            params['if_record_kfac_F_twoNorm_per_epoch'] == True:
            data_['kfac_F_twoNorms_per_epoch'] = []
        if 'if_record_kron_bfgs_norm_s_y_per_iter' in params and\
            params['if_record_kron_bfgs_norm_s_y_per_iter'] == True:
            data_['kron_bfgs_norms_s_y_per_iter'] = {}
            data_['kron_bfgs_norms_s_y_per_iter']['s'] = []
            data_['kron_bfgs_norms_s_y_per_iter']['y'] = []
        if 'if_record_kron_bfgs_sTy_per_iter' in params and\
            params['if_record_kron_bfgs_sTy_per_iter'] == True:
            data_['kron_bfgs_sTy_per_iter'] = []
        if 'if_record_kron_bfgs_damping_status' in params and\
            params['if_record_kron_bfgs_damping_status'] == True:
            data_['kron_bfgs_damping_statuses'] = {}
        if 'if_record_kron_bfgs_check_damping' in params and\
            params['if_record_kron_bfgs_check_damping'] == True:
            data_['kron_bfgs_check_dampings'] = []
            
        if params['if_records']['if_record_kron_bfgs_matrix_norm'] == True:
            data_['kron_bfgs_matrix_norms'] = []

    if params['if_record_sgn_norm']:
        sgn_norms = []
        data_['model_grad_full'] = get_full_grad(model, X_train, t_train, params)
        
    print('params[if_dataset_onTheFly]')
    print(params['if_dataset_onTheFly'])
    
    if not params['if_dataset_onTheFly']:
        
        reduction = 'mean'
        loss_0, unregularized_loss_0, acc_0 = get_regularized_loss_and_acc_from_x_whole_dataset(model, X_train, t_train, reduction, params)
        

        
        train_losses.append(loss_0)
        train_unregularized_losses.append(unregularized_loss_0)
        train_acces.append(acc_0)
    
        print('loss_0, unregularized_loss_0, acc_0')
        print(loss_0, unregularized_loss_0, acc_0)


    N1 = params['N1']
    
#     print('len(t_train)')
#     print(len(t_train))
    
#     print('params[num_train_data]')
#     print(params['num_train_data'])
    
#     sys.exit()
        

#     iter_per_epoch = int(len(t_train) / N1)
    iter_per_epoch = int(params['num_train_data'] / N1)
    
    params['iter_per_epoch'] = iter_per_epoch

#     iter_per_record = int(np.floor(len(t_train) * record_epoch / N1))
    iter_per_record = int(np.floor(params['num_train_data'] * record_epoch / N1))
    
    
#     if params['if_test_mode']:
#         i = 0
#         while i + N1 <= params['num_train_data']:
#             data_['dataset'].train.next_batch(N1)
#             i += N1

    # Training
    print('Begin training...')
    epoch = -1
    i = -1
    
#     model_test = copy.deepcopy(model)
    
    while not get_if_stop(args, i+1, iter_per_epoch, timesCPU):
        
#         torch.cuda.empty_cache()

        i += 1
        params['i'] = i
        
#         if params['if_test_mode']:
        
#             torch.save(
#                 {'model_state_dict': data_['model'].state_dict()},
#                 'saved_model/saved_model_start_' + str(params['i'])
#             )
            
#             print('save model (start) to disk')
        
#         if params['if_test_mode']:
        
#             checkpoint = torch.load(
#                 'saved_model/saved_model_start_' + str(params['i']),
#                 map_location=torch.device(device)
#             )
    
#             data_['model'].load_state_dict(checkpoint['model_state_dict'])
        
#             print('load model (start) from disk')

        if i % iter_per_record == 0:
            start_time_wall_clock = time.time()
            start_time_cpu = time.process_time()
            epoch += 1
            params['epoch'] = epoch

        # get minibatch
        X_mb, t_mb = dataset.train.next_batch(N1)
        
#         if i % iter_per_record == 0:
            
#             print('i')
#             print(i)
            
#             print('t_mb')
#             print(t_mb)
        
#         X_mb = torch.from_numpy(X_mb).to(device)
        
        if not params['if_dataset_onTheFly']:
            X_mb = torch.from_numpy(X_mb)
        X_mb = X_mb.to(device)
        

        

        
#         t_mb = torch.from_numpy(t_mb).to(device)
        
        if not params['if_dataset_onTheFly']:
            t_mb = torch.from_numpy(t_mb)
        t_mb = t_mb.to(device)

    
    
        

        # Forward
        z, a, h = model.forward(X_mb)
        
#         print('z.size()')
#         print(z.size())
        
#         print('t_mb.size()')
#         print(t_mb.size())
        
#         sys.exit()
        
        reduction = 'mean'
#         loss = get_regularized_loss_from_z(
#             model, z, t_mb, reduction, params['tau'])
        loss = get_loss_from_z(
            model, z, t_mb, reduction)
        
        if i == 0:
            
#             print('loss')
#             print(loss)
            
#             sys.exit()
            
            unregularized_minibatch_loss_i = loss.item()
        
            train_unregularized_minibatch_losses.append(
            unregularized_minibatch_loss_i)
            
#             print('get_acc_from_z(model, params, z, t_mb)')
#             print(get_acc_from_z(model, params, z, t_mb))
            
            minibatch_acc_i = get_acc_from_z(model, params, z, t_mb)
            
#             sys.exit()
            train_minibatch_acces.append(minibatch_acc_i)
            
        else:
            
#             sys.exit()
            
            minibatch_acc_i =\
            0.9 * minibatch_acc_i + 0.1 * get_acc_from_z(model, params, z, t_mb)
            
            unregularized_minibatch_loss_i =\
            0.9 * unregularized_minibatch_loss_i + 0.1 * loss.item()
    
#         if params['if_test_mode']:
#             print('loss.item()')
#             print(loss.item())
    
        if params['if_test_mode']:
            if 'if_record_loss_per_iter' in params and\
                params['if_record_loss_per_iter'] == True:
                
                tau = params['tau']
                
                if tau == 0:
                    data_['losses_per_iter'].append(loss.item())
                else:
#                 if params['tau'] != 0:

                    data_['losses_per_iter'].append(
                        loss.item() +\
                    0.5 * tau *\
    get_dot_product_torch(model.layers_weight, model.layers_weight).item()
                    )

#                     print('error: not working for tau != 0')
#                     sys.exit()


        # backward and gradient
        model.zero_grad()
        
        if params['if_second_order_algorithm'] and params['matrix_name'] == 'Fisher-correct':
            loss.backward(retain_graph=True)
        else:
            loss.backward()
        
        
        model_grad_torch = get_model_grad(model, params)
        
        data_['model_grad_torch_unregularized'] = model_grad_torch
        
        model_grad_torch =\
        from_unregularized_grad_to_regularized_grad(model_grad_torch, data_, params)
        
        
        if torch.sum(z != z):
            print('Error: nan in z')
            # print('torch.sum(X_mb != X_mb)')
            # print(torch.sum(X_mb != X_mb))
            print('i')
            print(i)
            print('get_if_nan(model.layers_weight)')
            print(get_if_nan(model.layers_weight))
            for l in range(len(model.layers_weight)):
                for key in model.layers_weight[l]:
                    print('torch.max(model.layers_weight[l][key])')
                    print(torch.max(model.layers_weight[l][key]))
                    print('torch.min(model.layers_weight[l][key])')
                    print(torch.min(model.layers_weight[l][key]))
            for l in range(len(a)):
                print('torch.max(a[l])')
                print(torch.max(a[l]))
                print('torch.min(a[l])')
                print(torch.min(a[l]))
#             for l in range(len(h)):
                
#                 print('h[l]')
#                 print(h[l])
                
#                 print('torch.max(h[l])')
#                 print(torch.max(h[l]))
#                 print('torch.min(h[l])')
#                 print(torch.min(h[l]))
            break
        

        data_['model_grad_torch'] = model_grad_torch # regularized
        
#         print('torch.norm(X_mb)')
#         print(torch.norm(X_mb))
        
#         print('data_[model_grad_torch][-1][b][:10] usual backprop')
#         print(data_['model_grad_torch'][-1]['b'][:10])
        
#         print('get_dot_product_torch(data_[model_grad_torch], data_[model_grad_torch])')
#         print(get_dot_product_torch(data_['model_grad_torch'], data_['model_grad_torch']))
        
        
        # add regularization
#         model_grad_torch = get_plus_scalar(params['tau'], model_grad_torch)
        
        

        if get_if_nan(model_grad_torch):
            print('Error: nan in model_grad_torch')
            for l in range(len(model_grad_torch)):
                for key in model_grad_torch[l]:
                    print('torch.max(model_grad_torch[l][key])')
                    print(torch.max(model_grad_torch[l][key]))
                    print('torch.min(model_grad_torch[l][key])')
                    print(torch.min(model_grad_torch[l][key]))
            for l in range(len(model.layers_weight)):
                for key in model.layers_weight[l]:
                    print('torch.max(model.layers_weight[l][key])')
                    print(torch.max(model.layers_weight[l][key]))
                    print('torch.min(model.layers_weight[l][key])')
                    print(torch.min(model.layers_weight[l][key]))

            break




        
        

        
        if params['if_test_mode']:
            if params['if_record_sgd_norm']:
                sgd_norms.append(
                    np.sqrt(get_dot_product_torch(model_regularized_grad_torch, model_regularized_grad_torch).item()))
        
        if params['if_record_sgn_norm']:
            sgn_ = get_subtract_torch(model_regularized_grad_torch, data_['model_grad_full'])
            sgn_norms.append(np.sqrt(get_dot_product_torch(sgn_, sgn_).item()))
            
        if params['if_momentum_gradient']:
            # rho = min(1-1/(i+1), 0.9)
            rho = params['momentum_gradient_rho']
            
            # mimic torch.optim.SGD
#             dampening = rho
            dampening = params['momentum_gradient_dampening']
            
            data_['model_grad_momentum'] =\
            get_plus_torch(
                get_multiply_scalar(rho, data_['model_grad_momentum']),
                get_multiply_scalar(1 - dampening, model_grad_torch)) # regularized

        if params['if_momentum_gradient']:
            data_['model_grad_used_torch'] =\
            data_['model_grad_momentum']
        else:
        
            data_['model_grad_used_torch'] = model_grad_torch
            
#         if params['if_test_mode']:
            
#             assert params['if_momentum_gradient']

#             if not os.path.exists('saved_model/' + params['algorithm']):
#                 os.makedirs('saved_model/' + params['algorithm'])
            
#             torch.save(
#                 {'model_grad_momentum': data_['model_grad_momentum']},
#                 'saved_model/' + params['algorithm'] + '/saved_grad_' + str(params['i'])
#             )
        
#             print('save grad to disk')


        # get second order caches
        if params['if_second_order_algorithm']:
            data_['X_mb'] = X_mb
            data_['t_mb'] = t_mb
            
#             if i % 10 == 0:
            
            data_ = get_second_order_caches(z, a, h, data_, params)
        

        

        model = data_['model']

        if params['if_LM']:
            data_['regularized_loss'] = loss
            data_['t_mb_N1'] = t_mb
            lambda_minus_tau = params['lambda_']
            params['lambda_'] = params['lambda_'] + params['tau']
            
#         print('params[if_lr_decay]')
#         print(params['if_lr_decay'])
        
#         sys.exit()
        
        if params['if_lr_decay']:
            params['alpha_current'] =\
            params['alpha'] *\
            (params['lr_decay_rate'] ** (params['epoch'] // params['num_epoch_to_decay']))

        algorithm = params['algorithm']

        if algorithm in ['ekfac-EF-VA',
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
                         'kfac-EF',
                         'Fisher-block']:    
            data_, params = kfac_update(data_, params)
            
            if params['kfac_svd_failed']:
                
                print('i')
                print(i)
                
                print('error: kfac_svd_failed')
                
                break

        elif algorithm == 'SMW-Fisher-signVAsqrt-p' or\
        algorithm == 'SMW-Fisher-VA-p' or\
        algorithm == 'SMW-Fisher-momentum-p-sign' or\
        algorithm == 'SMW-Fisher-momentum-p' or\
        algorithm == 'SMW-Fisher-sign' or\
        algorithm == 'SMW-Fisher-different-minibatch' or\
        algorithm == 'SMW-Fisher' or\
        algorithm == 'SMW-Fisher-momentum' or\
        algorithm == 'SMW-Fisher-D_t-momentum' or\
        algorithm == 'SMW-Fisher-momentum-D_t-momentum':

            data_, params = SMW_Fisher_update(data_, params)
        elif algorithm in ['shampoo',
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
            data_, params = shampoo_update(data_, params)
        elif algorithm in ['Fisher-BD']:
            data_, params = Fisher_BD_update(data_, params)
        elif algorithm in ['SMW-Fisher-batch-grad-momentum-exponential-decay',
                           'SMW-Fisher-batch-grad',
                           'SMW-Fisher-batch-grad-momentum']:
            data_, params = SMW_Fisher_batch_grad_update(data_, params)
        elif algorithm in ['SGD-VA',
                           'SGD-signVAsqrt',
                           'SGD-signVAerf',
                           'SGD-signVA',
                           'SGD-yura-BD',
                           'SGD-yura-old',
                           'SGD-yura',
                           'SGD-yura-MA',
                           'SGD-sign',
                           'SGD-momentum-yura',
                           'SGD-momentum',
                           'SGD',]:
            data_ = SGD_update(data_, params)
        elif algorithm in ['RMSprop',
                           'RMSprop-warmStart',
                           'RMSprop-test',
                           'Adam',
                           'Adam-test',
                           'Adam-noWarmStart',
                           'RMSprop-no-sqrt',
                           'RMSprop-individual-grad',
                           'RMSprop-individual-grad-no-sqrt',
                           'RMSprop-individual-grad-no-sqrt-Fisher',
                           'RMSprop-individual-grad-no-sqrt-LM']:
            data_ = RMSprop_update(data_, params)
        elif algorithm == 'SMW-GN':
            data_ = SMW_GN_update(data_, params)
        elif algorithm == 'GI-Fisher':
            data_, params = GI_Fisher_update(data_, params)
        elif algorithm == 'SMW-Fisher-BD':
            data_, params = SMW_Fisher_BD_update(data_, params)
        elif algorithm in ['Kron-BFGS',
                           'Kron-BFGS-no-norm-gate',
                           'Kron-BFGS-no-norm-gate-momentum-s-y',
                           'Kron-BFGS-no-norm-gate-momentum-s-y-damping',
                           'Kron-BFGS-no-norm-gate-damping',
                           'Kron-BFGS-no-norm-gate-Shiqian-damping',
                           'Kron-BFGS-homo-no-norm-gate-damping',
                           'Kron-BFGS-homo-no-norm-gate-Shiqian-damping',
                           'Kron-BFGS-homo-no-norm-gate-Powell-H-damping',
                           'Kron-BFGS-homo-no-norm-gate-PowellBDamping',
                           'Kron-BFGS-homo-no-norm-gate-PowellHDampingV2',
                           'Kron-BFGS-homo-no-norm-gate-Powell-double-damping',
                           'Kron-BFGS-homo-no-norm-gate-PowellDoubleDampingV2',
                           'Kron-BFGS-homo-no-norm-gate-momentum-s-y-damping',
                           'Kron-BFGS-homo-no-norm-gate-momentum-s-y-Powell-double-damping',
                           'Kron-BFGS-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping',
                           'Kron-BFGS-homo-no-norm-gate-HessianActionV2-momentum-s-y-Powell-double-damping',
                           'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-Powell-double-damping',
                           'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-DDV2',
                           'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2IdentityInitial-momentum-s-y-DDV2',
                           'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2',
                           'Kron-BFGS-homo-no-norm-gate-miniBatchANotDamped-HessianActionV2-momentum-s-y-DDV2',
                           'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-DDV2',
                           'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-SqrtT',
                           'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-KFACSplitting',
                           'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-DDV2-extraStep',
                           'Kron-(L)BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-Powell-double-damping',
                           'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2IdentityInitial-momentum-s-y-Powell-double-damping',
                           'Kron-BFGS(L)-homo-no-norm-gate-HessianActionV2-momentum-s-y-Powell-double-damping',
                           'Kron-BFGS(L)-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-Powell-double-damping',
                           'Kron-BFGS(L)-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-DDV2',
                           'Kron-BFGS(L)-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2',
                           'Kron-BFGS-homo-no-norm-gate-scaledHessianAction-momentum-s-y-Powell-double-damping',
                           'Kron-BFGS-homo-no-norm-gate-HessianActionIdentityInitial-momentum-s-y-Powell-double-damping',
                           'Kron-LBFGS-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping',
                           'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping',
                           'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-PowellDoubleDampingSkip',
                           'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-DoubleDamping',
                           'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-H-damping',
                           'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-B0-damping',
                           'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-Shiqian-damping',
                           'Kron-(L)BFGS-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping',
                           'Kron-LBFGS-homo-no-norm-gate-momentum-s-y-Powell-double-damping',
                           'Kron-BFGS-homo-no-norm-gate-Hessian-action-Powell-double-damping',
                           'Kron-BFGS-homo-identity',
                           'Kron-BFGS-wrong',
                           'Kron-BFGS-homo',
                           'Kron-BFGS-Hessian-action',
                           'Kron-BFGS-wrong-Hessian-action',
                           'Kron-BFGS-LM',
                           'Kron-BFGS-LM-sqrt',
                           'Kron-BFGS-1st-layer-only']:
            data_, params = Kron_BFGS_update_v2(data_, params)
            
            if params['nan_detected']:
                break
            
        elif algorithm in ['Kron-BFGS-block']:
            data_, params = Kron_BFGS_update(data_, params)
        elif algorithm == 'Kron-SGD':
            data_, params = Kron_SGD_update(data_, params)
        elif algorithm in ['BFGS',
                           'BFGS-homo']:
            data_, params = BFGS_update(data_, params)
        else:
            print('Error: updating direction not defined for ' + algorithm)
            sys.exit()

        if params['if_LM']:
            params['lambda_'] = lambda_minus_tau


        p_torch = data_['p_torch']
        
#         if params['if_test_mode']:

#             if not os.path.exists('saved_model/' + params['algorithm']):
#                 os.makedirs('saved_model/' + params['algorithm'])
            
#             torch.save(
#                 {'p_torch': p_torch},
#                 'saved_model/' + params['algorithm'] + '/saved_p_' + str(params['i'])
#             )
        
#             print('save p to disk')
        

        if params['if_test_mode']:
            if params['if_record_p_norm']:
                p_norms.append(
                    np.sqrt(get_dot_product_torch(p_torch, p_torch).item()))

        if params['if_LM']:

            lambda_ = update_lambda(p_torch, data_, params)
            params['lambda_'] = lambda_

        

        if params['if_momentum_p']:
            rho_momentum_p = 0.9
            data_['p_momentum_torch'] = get_plus_torch(\
                                           get_multiply_scalar(rho_momentum_p, data_['p_momentum_torch']),
                                           get_multiply_scalar(1 - rho_momentum_p, p_torch))
            # p = data_['p_momentum']
            p_torch = data_['p_momentum_torch']

        if params['if_VA_p']:
            rho_momentum_p = 0.9
            data_['p_momentum_1'] = get_plus(\
                                           get_multiply_scalar(rho_momentum_p, data_['p_momentum_1']),
                                           get_multiply_scalar(1 - rho_momentum_p, p))
            data_['p_momentum_2'] = get_plus(\
                                           get_multiply_scalar(rho_momentum_p, data_['p_momentum_2']),
                                           get_multiply_scalar(1 - rho_momentum_p, get_square(p)))
            p = get_divide(\
                           get_multiply(data_['p_momentum_1'], get_square(data_['p_momentum_1'])),
                           get_plus_scalar(10**(-8), data_['p_momentum_2']))
            
        if params['if_signVAsqrt'] or\
        params['if_signVA'] or\
        params['if_signVAerf']:
            rho_momentum_p = 0.9
            data_['p_momentum_1_torch'] = get_plus_torch(\
                                           get_multiply_scalar(rho_momentum_p, data_['p_momentum_1_torch']),
                                           get_multiply_scalar(1 - rho_momentum_p, p_torch))
            data_['p_momentum_2_torch'] = get_plus_torch(\
                                           get_multiply_scalar(rho_momentum_p, data_['p_momentum_2_torch']),
                                           get_multiply_scalar(1 - rho_momentum_p, get_square_torch(p_torch)))
            if params['if_signVAsqrt']:
                p_torch = get_divide_torch(\
                            data_['p_momentum_1_torch'],
                            get_sqrt_torch(get_plus_scalar(10**(-8), data_['p_momentum_2_torch'])))
            elif params['if_signVA']:
                p_torch = get_divide(\
                           get_multiply(get_sign(data_['p_momentum_1']), get_square(data_['p_momentum_1'])),
                           get_plus_scalar(10**(-8), data_['p_momentum_2']))
            elif params['if_signVAerf']:
                if params['i'] > 100:
                    relative_variance = get_plus_scalar(
                        -1,
                        get_divide_torch(
                            data_['p_momentum_2_torch'],
                            get_plus_scalar(10**(-8), get_square_torch(data_['p_momentum_1_torch']))))
                    

                    
                    # print('data_[p_momentum_2_torch][-1][W]')
                    # print(data_['p_momentum_2_torch'][-1]['W'])

                    p_torch = get_multiply_torch(
                        get_sign_torch(data_['p_momentum_1_torch']),
                        get_erf(
                            get_reciprocal(
                                get_multiply_scalar(
                                    np.sqrt(2),
                                    get_sqrt_torch(get_max_with_0(relative_variance)))))
                    )
                else:
                    p_torch = data_['p_momentum_1_torch']

                
            
        

        if params['if_sign']:
            p = get_sign(p)
            

        if params['if_Adam']:
            
            print('error: deprecated, must be false')
            sys.exit()

            p, data_ = get_Adam_direction(p, data_, params)

        if params['if_yura']:
            p_torch = get_yura(p_torch, data_, params)
            
        

        model = update_parameter(p_torch, model, params)
        
#         l_debug = 1
        
#         print('p_torch[l_debug][W]')
#         print(p_torch[l_debug]['W'])
        
#         print('model.layers_weight[l_debug][W]')
#         print(model.layers_weight[l_debug]['W'])
        
#         l_debug = 3
#         print('model.layers_weight[l_debug][W]')
#         print(model.layers_weight[l_debug]['W'])
        
#         if params['if_test_mode']:
        
#             torch.save(
#                 {'model_state_dict': data_['model'].state_dict()},
#                 'saved_model/saved_model_end_' + str(params['i'])
#             )
            
#             print('save model (end) to disk')
        
#         if params['if_test_mode']:
        
#             checkpoint = torch.load(
#                 'saved_model/saved_model_end_' + str(params['i']),
#                 map_location=torch.device(device)
#             )
    
#             data_['model'].load_state_dict(checkpoint['model_state_dict'])
        
#             print('load model (end) from disk')

        if get_if_nan(model.layers_weight):
            print('Error: nan in model.layers_weight')
            break

    #     print('time 7/8: ', time.time() - start_time)
    
#         print('(i+1) % iter_per_record')
#         print((i+1) % iter_per_record)
#         print('iter_per_record')
#         print(iter_per_record)

        if (i+1) % iter_per_record == 0:
        
            if params['if_test_mode']:
                if 'if_record_kfac_G_inv_norm_per_epoch' in params and\
                params['if_record_kfac_G_inv_norm_per_epoch']:
                    
                    data_['kfac_G_inv_norms_per_epoch'].append(
                        [torch.norm(G_inv_l).item() for G_inv_l in data_['G_inv']]
                    )
                    # torch.norm() is Fro-norm here
                    
            if params['if_test_mode']:
                if params['if_records']['if_record_kfac_G_norm_per_epoch']:
                    
                    # data_['G'] is without LM
                    data_['kfac_G_norms_per_epoch'].append(
                        [torch.norm(G_l).item() for G_l in data_['G']]
                    )
                    
#                     print('[print(torch.norm(G_l).item()) for G_l in data_[G]]')
#                     [print(torch.norm(G_l).item()) for G_l in data_['G']]
                    
            if params['if_test_mode']:
                if params['if_records']['if_record_kfac_G_twoNorm_per_epoch']:
                    
                    # data_['G'] is without LM
                    data_['kfac_G_twoNorms_per_epoch'].append(
                        [np.linalg.norm(G_l.cpu().data.numpy(), ord=2) for G_l in data_['G']]
                    )
                    
            if params['if_test_mode']:
                if params['if_records']['if_record_kfac_A_twoNorm_per_epoch']:
                    
                    # data_['A'] is without LM
                    data_['kfac_A_twoNorms_per_epoch'].append(
                        [np.linalg.norm(A_l.cpu().data.numpy(), ord=2) for A_l in data_['A']]
                    )
                    
                    
            if params['if_test_mode']:
                if params['if_records']['if_record_kron_bfgs_A_twoNorm_per_epoch']:
                   
                    
                    # without LM
                    data_['kron_bfgs_A_twoNorms_per_epoch'].append(
                        [
                            np.linalg.norm(
                                Kron_BFGS_matrices_l['A'].cpu().data.numpy(), ord=2
                            ) for Kron_BFGS_matrices_l in data_['Kron_BFGS_matrices']
                        ]
                    )
                    
            if params['if_test_mode']:
                if params['if_records']['if_record_kron_bfgs_G_LM_twoNorm_per_epoch']:
                    
                    if params['algorithm'] == 'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping':
                        data_['kron_bfgs_G_LM_twoNorms_per_epoch'].append([])
                        for l in range(len(data_['Kron_LBFGS_s_y_pairs']['a_grad'])):
                            size_identity_v =\
                            params['layers_params'][l]['output_size']
 
                            identity_v = torch.eye(size_identity_v).cuda()
                    
                            data_['kron_bfgs_G_LM_twoNorms_per_epoch'][-1].append(
                                np.linalg.norm(
                                    LBFGS_Hv(identity_v, data_['Kron_LBFGS_s_y_pairs']['a_grad'][l], params, False).inverse().cpu().data.numpy(),
                                    ord=2
                                )
                            )
                    else:
                        print('error: not implemented')
                        sys.exit()
                    
                        data_['kron_bfgs_G_LM_twoNorms_per_epoch'].append(
                        [
                                 np.linalg.norm(
                                Kron_BFGS_matrices_l['H']['a_grad'].inverse().cpu().data.numpy(), ord=2
                            )
                            for Kron_BFGS_matrices_l in data_['Kron_BFGS_matrices']
                        ]
                    )
                    
            if params['if_test_mode']:
                if params['if_records']['if_record_kron_bfgs_Hg_twoNorm_per_epoch']:
                    
                    if params['algorithm'] == 'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping':
                        data_['kron_bfgs_Hg_twoNorms_per_epoch'].append([])
                        for l in range(len(data_['Kron_LBFGS_s_y_pairs']['a_grad'])):
                            size_identity_v =\
                            params['layers_params'][l]['output_size']
 
                            identity_v = torch.eye(size_identity_v).cuda()
                    
                            data_['kron_bfgs_Hg_twoNorms_per_epoch'][-1].append(
                                np.linalg.norm(
                                    LBFGS_Hv(identity_v, data_['Kron_LBFGS_s_y_pairs']['a_grad'][l], params, False).cpu().data.numpy(),
                                    ord=2
                                )
                            )
#                             data_['kron_bfgs_Hg_twoNorms_per_epoch'][-1].append(
#                                 np.linalg.norm(
#                                     LBFGS_Hv(identity_v, data_['Kron_LBFGS_s_y_pairs']['a'][l], params, False).cpu().data.numpy(),
#                                     ord=2
#                                 )
#                             )
                    else:
                        print('error: not implemented')
                        sys.exit()
                    
                        data_['kron_bfgs_Hg_twoNorms_per_epoch'].append(
                        [
                                 np.linalg.norm(Kron_BFGS_matrices_l['H']['a_grad'].cpu().data.numpy(), ord=2)
                            for Kron_BFGS_matrices_l in data_['Kron_BFGS_matrices']
                        ]
                    )
                    
            if params['if_test_mode']:
                if params['if_records']['if_record_kron_bfgs_Ha_twoNorm_per_epoch']:
                    
#                     print('data_[Kron_BFGS_matrices][0][H].keys()')
#                     print(data_['Kron_BFGS_matrices'][0]['H'].keys())
                    
#                     sys.exit()
                    
                    data_['kron_bfgs_Ha_twoNorms_per_epoch'].append(
                        [np.linalg.norm(Kron_BFGS_matrices_l['H']['h'].cpu().data.numpy(), ord=2) for Kron_BFGS_matrices_l in data_['Kron_BFGS_matrices']]
                    )
                    
                    
            if params['if_test_mode']:
                if 'if_record_kfac_F_twoNorm_per_epoch' in params and\
                params['if_record_kfac_F_twoNorm_per_epoch']:
                    
                    # without LM
                    # see https://math.stackexchange.com/questions/2342156/matrix-norm-of-kronecker-product
                    data_['kfac_F_twoNorms_per_epoch'].append(
                         [np.linalg.norm(A_l.cpu().data.numpy(), ord=2) * np.linalg.norm(G_l.cpu().data.numpy(), ord=2) for A_l, G_l in zip(data_['A'], data_['G'])]
                    )
#                     data_['kfac_F_twoNorms_per_epoch'].append(
#                          [torch.norm(A_l, p=2).item() * torch.norm(G_l, p=2).item()for A_l, G_l in zip(data_['A'], data_['G'])]
#                     )
            
    
            if params['if_test_mode'] and params['if_records']['if_record_kron_bfgs_matrix_norm'] == True:
                
                kron_bfgs_matrix_norms_i = []
                
                for l in range(len(data_['Kron_BFGS_matrices'])):
                    kron_bfgs_matrix_norms_i_l = {}
                    
#                     kron_bfgs_matrix_norms_i_l['a_grad'] =\
#                     torch.norm(data_['Kron_BFGS_matrices'][l]['H']['a_grad']).item()
                    
                    kron_bfgs_matrix_norms_i_l['a_grad'] = {}
                    kron_bfgs_matrix_norms_i_l['a_grad']['fro'] =\
                    torch.norm(data_['Kron_BFGS_matrices'][l]['H']['a_grad'], p='fro').item()
                    kron_bfgs_matrix_norms_i_l['a_grad']['2'] =\
                    np.linalg.norm(data_['Kron_BFGS_matrices'][l]['H']['a_grad'].cpu().numpy(), ord=2)
                    
                    try:
                        kron_bfgs_matrix_norms_i_l['a_grad']['max_eig'] =\
                    torch.symeig(data_['Kron_BFGS_matrices'][l]['H']['a_grad'])[0][-1].item()
                    except:
                        kron_bfgs_matrix_norms_i_l['a_grad']['max_eig'] = float('nan')
                    
                    kron_bfgs_matrix_norms_i.append(kron_bfgs_matrix_norms_i_l)
                    
                data_['kron_bfgs_matrix_norms'].append(kron_bfgs_matrix_norms_i)

    
            if params['if_test_mode'] and\
        (params['if_records']['if_record_layerWiseHessian_twoNorm_per_epoch'] or\
                params['if_records']['if_record_inverseLayerWiseHessian_twoNorm_per_epoch'] or\
        params['if_records']['if_record_inverseLayerWiseHessian_LM_twoNorm_per_epoch'] or\
        params['if_records']['if_record_inverseLayerWiseHessian_LM_MA_twoNorm_per_epoch']):
            
#                 print('len(data_[inverseLayerWiseHessian_LM_MA_twoNorms_per_epoch])')
#                 print(len(data_['inverseLayerWiseHessian_LM_MA_twoNorms_per_epoch']))
                    
#                 sys.exit()

                from utils_git.utils_hessian import compute_hessian

                true_layer_wise_hessian = compute_hessian(X_mb, t_mb, data_, params)
                
                if params['if_records']['if_record_inverseLayerWiseHessian_LM_MA_twoNorm_per_epoch']:
                    
                    print('torch.norm(X_mb)')
                    print(torch.norm(X_mb))
                    
                    if len(data_['inverseLayerWiseHessian_LM_MA_twoNorms_per_epoch']) == 0:
                        true_layer_wise_hessian_MA = true_layer_wise_hessian
                    else:
#                         print('len(true_layer_wise_hessian_MA)')
#                         print(len(true_layer_wise_hessian_MA))
                        
#                         print('len(true_layer_wise_hessian)')
#                         print(len(true_layer_wise_hessian))
                        
                        assert len(true_layer_wise_hessian_MA) == len(true_layer_wise_hessian)
                        
                        for l in range(len(true_layer_wise_hessian_MA)):
                            true_layer_wise_hessian_MA[l] =\
                        0.9 * true_layer_wise_hessian_MA[l] +\
                        0.1 * true_layer_wise_hessian[l]
                    
                    
                    
#                         sys.exit()
                        
                    data_['inverseLayerWiseHessian_LM_MA_twoNorms_per_epoch'].append([])
                        
                    for B_l in true_layer_wise_hessian_MA:
                        
                        lambda_hessian_LM =\
                        params['Kron_BFGS_A_LM_epsilon'] * params['Kron_BFGS_H_epsilon']
                        
                        B_l_LM = B_l + lambda_hessian_LM * np.eye(B_l.shape[0])
                        
                        data_['inverseLayerWiseHessian_LM_MA_twoNorms_per_epoch'][-1].append(
                        np.linalg.norm(np.linalg.inv(B_l_LM), ord=2)
                        )
                    
        
                

                if params['if_records']['if_record_layerWiseHessian_twoNorm_per_epoch']:
                    data_['layerWiseHessian_twoNorms_per_epoch'].append(
                    [np.linalg.norm(B_l, ord=2) for B_l in true_layer_wise_hessian]
                )
                
                if params['if_records']['if_record_inverseLayerWiseHessian_twoNorm_per_epoch']:
                    data_['inverseLayerWiseHessian_twoNorms_per_epoch'].append(
                    [np.linalg.norm(np.linalg.inv(B_l), ord=2) for B_l in true_layer_wise_hessian]
                )
                    
                if params['if_records']['if_record_inverseLayerWiseHessian_LM_twoNorm_per_epoch']:
                    data_['inverseLayerWiseHessian_LM_twoNorms_per_epoch'].append([])
                    
                    for B_l in true_layer_wise_hessian:
                        
                        lambda_hessian_LM =\
                        params['Kron_BFGS_A_LM_epsilon'] * params['Kron_BFGS_H_epsilon']
                        
                        B_l_LM = B_l + lambda_hessian_LM * np.eye(B_l.shape[0])
                        
                        data_['inverseLayerWiseHessian_LM_twoNorms_per_epoch'][-1].append(
                        np.linalg.norm(np.linalg.inv(B_l_LM), ord=2)
                        )
                    
#                     data_['inverseLayerWiseHessian_LM_twoNorms_per_epoch'].append(
#                     [np.linalg.norm(np.linalg.inv(B_l), ord=2) for B_l in true_layer_wise_hessian]
#                 )
                    
            
            

            import datetime
            import pytz
            my_date = datetime.datetime.now(pytz.timezone('US/Eastern'))
            my_date = my_date.strftime("%d/%m/%Y %H:%M:%S")
            print("date and time =", my_date)

            timesCPU_i = time.process_time() - start_time_cpu
            timesWallClock_i = time.time() - start_time_wall_clock
            
            


            
            if not params['if_dataset_onTheFly']:
                
                reduction = 'mean'
                loss_i, unregularized_loss_i, acc_i = get_regularized_loss_and_acc_from_x_whole_dataset(model, X_train, t_train, reduction, params)
            

            
            
            if params['if_dataset_onTheFly']:    
                if math.isnan(unregularized_minibatch_loss_i):
                    print('Warning: unregularized_minibatch_loss_i is NAN.')
                    break
            else:
                
                if math.isnan(loss_i):
                    print('Warning: loss_i is NAN.')
                    break

            timesCPU.append(timesCPU_i)
            timesWallClock.append(timesWallClock_i)
            if epoch > 0:
                timesCPU[-1] = timesCPU[-1] + timesCPU[-2]
                timesWallClock[-1] = timesWallClock[-1] + timesWallClock[-2]
            
            if not params['if_dataset_onTheFly']:
                train_losses.append(loss_i)
                train_unregularized_losses.append(unregularized_loss_i)
                train_acces.append(acc_i)
            
            train_unregularized_minibatch_losses.append(
                unregularized_minibatch_loss_i
            )
#             sys.exit()
            train_minibatch_acces.append(minibatch_acc_i)
            
            

            reduction = 'mean'
            
            if params['if_dataset_onTheFly']:
                test_loss_i, test_unregularized_loss_i, test_acc_i =\
                get_regularized_loss_and_acc_from_x_whole_dataset_with_generator(model, dataset.test_generator, reduction, params)
            else:
                test_loss_i, test_unregularized_loss_i, test_acc_i =\
                get_regularized_loss_and_acc_from_x_whole_dataset(model, X_test, t_test, reduction, params)
            
            test_losses.append(test_loss_i)
            test_acces.append(test_acc_i)
            
            if params['if_LM']:
                lambdas.append(params['lambda_'])
                print('lambda = ', lambdas[-1])
            if params['if_yura']:
                yura_lambdas.append(params['yura_lambda'])
                print('yura-lambda = ', yura_lambdas[-1])
            epochs.append((epoch + 1) * record_epoch)
            



            
#             print('Learning rate: {0:.5f}'.format(params['alpha']))
            print('Current learning rate: {0:.5f}'.format(params['alpha_current']))
            
            print('Iter-{0:.3f}'.format(epochs[-1]))
            
            if not params['if_dataset_onTheFly']:
                print('Training loss: {0:.3f}'.format(train_losses[-1]))
                print('Training unregularized loss: {0:.3f}'.format(train_unregularized_losses[-1]))
                print('Training accuracy: {0:.3f}'.format(train_acces[-1]))
                
#                 print('train_unregularized_losses[-1]')
#                 print(train_unregularized_losses[-1])
            
            print('Training unregularized minibatch loss: {0:.3f}'.format(train_unregularized_minibatch_losses[-1]))
            print('Training minibatch acc: {0:.3f}'.format(train_minibatch_acces[-1]))
            
            

#             print('Testing loss: {0:.3f}'.format(test_losses[-1]))
            print('Testing unregularized loss: {0:.3f}'.format(test_unregularized_loss_i))
            print('Testing accuracy: {0:.3f}'.format(test_acces[-1]))
            
#             print('train_acces[-1]')
#             print(train_acces[-1])
            
            



            if epoch > 0:
                print('elapsed cpu time: ', timesCPU[-1] - timesCPU[-2])
                print('elapsed wall-clock time: ', timesWallClock[-1] - timesWallClock[-2])
            else:
                print('elapsed cpu time: ', timesCPU[-1])
                print('elapsed wall-clock time: ', timesWallClock[-1])

            


            
            # values_virtual_memory = psutil.virtual_memory()
            # print('total (GB):')
            # print(values_virtual_memory.total >> 30)
            # print('available (GB):')
            # print(values_virtual_memory.available >> 30)
            # print('percent (%):')
            # print(values_virtual_memory.percent)

            import gc
            gc.collect()

            torch.cuda.empty_cache()

            
            values_virtual_memory = psutil.virtual_memory()

            print('total (GB): {}, available (GB): {}, percent (%): {}'.format(
                values_virtual_memory.total >> 30, values_virtual_memory.available >> 30,\
                values_virtual_memory.percent
            ))
            


#             if params['if_gpu']:
            if params['device'] == 'cuda:0':
                
                print_gpu_usage(params)
                
                

            if 0:
                if params['algorithm'] == 'SMW-Fisher-different-minibatch' or\
                params['algorithm'] == 'SMW-Fisher' or\
                params['algorithm'] == 'SMW-Fisher-batch-grad-momentum-exponential-decay' or\
                params['algorithm'] == 'SMW-Fisher-batch-grad-momentum' or\
                params['algorithm'] == 'SMW-Fisher-batch-grad' or\
                params['algorithm'] == 'SGD' or\
                params['algorithm'] == 'ekfac-EF-VA' or\
                params['algorithm'] == 'ekfac-EF' or\
                params['algorithm'] == 'kfac':
                    1
                elif params['algorithm'] == 'RMSprop-individual-grad-no-sqrt-LM' or\
                params['algorithm'] == 'RMSprop-individual-grad-no-sqrt-Fisher' or\
                params['algorithm'] == 'RMSprop-individual-grad-no-sqrt' or\
                params['algorithm'] == 'RMSprop-individual-grad' or\
                params['algorithm'] == 'RMSprop-no-sqrt' or\
                params['algorithm'] == 'RMSprop':
                    for l in range(model.numlayers):
                        if params['layers_params'][l]['name'] == 'fully-connected' or\
                        params['layers_params'][l]['name'] == 'conv':
                            if np.max(data_['RMSprop_momentum_2'][l]['W']) < 10**(-100):
                                print('Warning: values too small.')
                                print('max value:')
                                print(np.max(data_['RMSprop_momentum_2'][l]['W']))
                            else:
                                fig, ax = plt.subplots()

                                # plt.ticklabel_format(style='sci', axis='x')
                                ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

                                ax.hist(data_['RMSprop_momentum_2'][l]['W'].flatten())
                                plt.xticks(rotation=90)

                                # ax.ticklabel_format(style='sci')
                            
                                plt.show()
                        else:
                            print('Error: unknown layer when show v_t')
                            sys.exit()

                    print('test v_t')
                else:
                    print('Error: need more v_t')
                    sys.exit()
                    

            print('\n')



        # model = get_model_grad_zerod(model)

    print('Begin saving results...')
    
    params['algorithm'] = params['true_algorithm']

    

    name_algorithm_with_params = get_name_algorithm_with_params(params)

    name_result = name_dataset + '/' + name_algorithm_with_params + '/'



    epochs = np.asarray(epochs)
    timesCPU = np.asarray(timesCPU)
    timesWallClock = np.asarray(timesWallClock)
    
    if not params['if_dataset_onTheFly']:
        train_losses = np.asarray(train_losses)
        train_unregularized_losses = np.asarray(train_unregularized_losses)
        train_acces = np.asarray(train_acces)
    
    train_unregularized_minibatch_losses = np.asarray(train_unregularized_minibatch_losses)
    train_minibatch_acces = np.asarray(train_minibatch_acces)
    
    
    test_losses = np.asarray(test_losses)
    test_acces = np.asarray(test_acces)
    dict_result = {'train_unregularized_minibatch_losses': train_unregularized_minibatch_losses,
                   'train_minibatch_acces': train_minibatch_acces,
                   'test_losses': test_losses,
                   'test_acces': test_acces,
                   'timesCPU': timesCPU,
                   'timesWallClock': timesWallClock,
                   'epochs': epochs}
    if not params['if_dataset_onTheFly']:
        dict_result.update(
        {'train_losses': train_losses,
         'train_unregularized_losses': train_unregularized_losses,
         'train_acces': train_acces}
    )
    if params['if_LM']:
        lambdas = np.asarray(lambdas)
        dict_result['lambdas'] = lambdas
    if params['if_yura']:
        yura_lambdas = np.asarray(yura_lambdas)
        dict_result['yura_lambdas'] = yura_lambdas
        

    

    if params['if_test_mode']:
        if params['if_record_sgd_norm']:
            sgd_norms = np.asarray(sgd_norms)
            dict_result['sgd_norms'] = sgd_norms
        if params['if_record_p_norm']:
            p_norms = np.asarray(p_norms)
            dict_result['p_norms'] = p_norms
        if params['if_record_kron_bfgs_cosine']:
            kron_bfgs_cosines = data_['kron_bfgs_cosines']
#             kfac_p_norms = np.asarray(kfac_p_norms)
            dict_result['kron_bfgs_cosines'] = kron_bfgs_cosines
        if params['if_record_kfac_p_norm']:
            kfac_p_norms = data_['kfac_p_norms']
            kfac_p_norms = np.asarray(kfac_p_norms)
            dict_result['kfac_p_norms'] = kfac_p_norms
        if params['if_record_kfac_p_cosine']:
            kfac_p_cosines = data_['kfac_p_cosines']
            kfac_p_cosines = np.asarray(kfac_p_cosines)
            dict_result['kfac_p_cosines'] = kfac_p_cosines
        if params['if_record_res_grad_norm']:
            res_grad_norms = data_['res_grad_norms']
            res_grad_norms = np.asarray(res_grad_norms)
            dict_result['res_grad_norms'] = res_grad_norms
        if params['if_record_res_grad_random_norm']:
            res_grad_random_norms = data_['res_grad_random_norms']
            res_grad_random_norms = np.asarray(res_grad_random_norms)
            dict_result['res_grad_random_norms'] = res_grad_random_norms
        if params['if_record_res_grad_grad_norm']:
            res_grad_grad_norms = data_['res_grad_grad_norms']
            res_grad_grad_norms = np.asarray(res_grad_grad_norms)
            dict_result['res_grad_grad_norms'] = res_grad_grad_norms
        if params['if_record_res_grad_norm_per_iter']:
            res_grad_norms_per_iter = data_['res_grad_norms_per_iter']
            res_grad_norms_per_iter = np.asarray(res_grad_norms_per_iter)
            dict_result['res_grad_norms_per_iter'] = res_grad_norms_per_iter
        if 'if_record_kron_bfgs_matrix_norm_per_iter' in params and\
            params['if_record_kron_bfgs_matrix_norm_per_iter'] == True:
            dict_result['kron_bfgs_matrix_norms_per_iter'] = data_['kron_bfgs_matrix_norms_per_iter']
            
        if 'if_record_kron_bfgs_damping_status' in params and\
            params['if_record_kron_bfgs_damping_status'] == True:
            dict_result['kron_bfgs_damping_statuses'] = data_['kron_bfgs_damping_statuses']
            
        if 'if_record_kron_bfgs_check_damping' in params and\
            params['if_record_kron_bfgs_check_damping'] == True:
            dict_result['kron_bfgs_check_dampings'] = data_['kron_bfgs_check_dampings']
        
        
            
        if 'if_record_kron_bfgs_update_status' in params and\
            params['if_record_kron_bfgs_update_status'] == True:
            dict_result['kron_bfgs_update_status'] = data_['kron_bfgs_update_status']
            
        if 'if_record_loss_per_iter' in params and\
            params['if_record_loss_per_iter'] == True:
            dict_result['losses_per_iter'] = data_['losses_per_iter']
            
        if 'if_record_kfac_G_inv_norm_per_iter' in params and\
        params['if_record_kfac_G_inv_norm_per_iter']:
            dict_result['kfac_G_inv_norms_per_iter'] = data_['kfac_G_inv_norms_per_iter']
            
        if 'if_record_kfac_G_inv_norm_per_epoch' in params and\
        params['if_record_kfac_G_inv_norm_per_epoch']:
            dict_result['kfac_G_inv_norms_per_epoch'] = data_['kfac_G_inv_norms_per_epoch']
            

        if params['if_records']['if_record_kfac_G_norm_per_epoch']:
            dict_result['kfac_G_norms_per_epoch'] = data_['kfac_G_norms_per_epoch']
        if params['if_records']['if_record_kfac_G_twoNorm_per_epoch']:
            dict_result['kfac_G_twoNorms_per_epoch'] = data_['kfac_G_twoNorms_per_epoch']
        if params['if_records']['if_record_kfac_A_twoNorm_per_epoch']:
            dict_result['kfac_A_twoNorms_per_epoch'] = data_['kfac_A_twoNorms_per_epoch']
        if params['if_records']['if_record_kron_bfgs_A_twoNorm_per_epoch']:
            dict_result['kron_bfgs_A_twoNorms_per_epoch'] = data_['kron_bfgs_A_twoNorms_per_epoch']
        if params['if_records']['if_record_kron_bfgs_G_LM_twoNorm_per_epoch']:
            dict_result['kron_bfgs_G_LM_twoNorms_per_epoch'] = data_['kron_bfgs_G_LM_twoNorms_per_epoch']
        if params['if_records']['if_record_kron_bfgs_Hg_twoNorm_per_epoch']:
            dict_result['kron_bfgs_Hg_twoNorms_per_epoch'] = data_['kron_bfgs_Hg_twoNorms_per_epoch']
        if params['if_records']['if_record_kron_bfgs_Ha_twoNorm_per_epoch']:
            dict_result['kron_bfgs_Ha_twoNorms_per_epoch'] = data_['kron_bfgs_Ha_twoNorms_per_epoch']
        if params['if_records']['if_record_layerWiseHessian_twoNorm_per_epoch']:
            dict_result['layerWiseHessian_twoNorms_per_epoch'] = data_['layerWiseHessian_twoNorms_per_epoch']
        if params['if_records']['if_record_inverseLayerWiseHessian_twoNorm_per_epoch']:
            dict_result['inverseLayerWiseHessian_twoNorms_per_epoch'] = data_['inverseLayerWiseHessian_twoNorms_per_epoch']
        if params['if_records']['if_record_inverseLayerWiseHessian_LM_twoNorm_per_epoch']:
            dict_result['inverseLayerWiseHessian_LM_twoNorms_per_epoch'] = data_['inverseLayerWiseHessian_LM_twoNorms_per_epoch']
        if params['if_records']['if_record_inverseLayerWiseHessian_LM_MA_twoNorm_per_epoch']:
            dict_result['inverseLayerWiseHessian_LM_MA_twoNorms_per_epoch'] = data_['inverseLayerWiseHessian_LM_MA_twoNorms_per_epoch']
        if params['if_records']['if_record_kron_bfgs_matrix_norm'] == True:
            dict_result['kron_bfgs_matrix_norms'] = data_['kron_bfgs_matrix_norms']
            
        if 'if_record_kfac_F_twoNorm_per_epoch' in params and\
        params['if_record_kfac_F_twoNorm_per_epoch']:
            dict_result['kfac_F_twoNorms_per_epoch'] = data_['kfac_F_twoNorms_per_epoch']
            
        if 'if_record_kron_bfgs_norm_s_y_per_iter' in params and\
        params['if_record_kron_bfgs_norm_s_y_per_iter']:
            dict_result['kron_bfgs_norms_s_y_per_iter'] = data_['kron_bfgs_norms_s_y_per_iter']
            
        if 'if_record_kron_bfgs_sTy_per_iter' in params and\
        params['if_record_kron_bfgs_sTy_per_iter']:
            dict_result['kron_bfgs_sTy_per_iter'] = data_['kron_bfgs_sTy_per_iter']

    if params['if_record_sgn_norm']:
        sgn_norms = np.asarray(sgn_norms)
        dict_result['sgn_norms'] = sgn_norms
        

    params_saved = {}
    for key_ in params['keys_params_saved']:
        params_saved[key_] = params[key_]
    dict_result['params'] = params_saved

    


    path_to_goolge_drive_dir = params['home_path'] + 'result/'
        
    
    os.makedirs(path_to_goolge_drive_dir + name_result, exist_ok = True)
    
#     print('dict_result[params]')
#     print(dict_result['params'])
#     sys.exit()

    fake_args = {}
    fake_args['algorithm_dict'] = {}
    fake_args['algorithm_dict']['name'] = params['algorithm']
    # for key in dict_result['params']:
    fake_args['algorithm_dict']['params'] = dict_result['params']
    fake_args['home_path'] = params['home_path']
    fake_args['N1'] = params['N1']
    fake_args['N2'] = params['N2']
    fake_args['if_gpu'] = params['if_gpu']
    fake_args['dataset'] = name_dataset
    fake_args['name_loss'] = params['name_loss']
    
    fake_args['list_lr'] = [params['alpha']]
    
#     fake_args['tuning_criterion'] = 'train_loss'
    fake_args['tuning_criterion'] = 'test_acc'
    # does not matter because 
    # presumably, there will be at most 1 old pkl
    
    _, _, old_pkl_name = get_best_params(fake_args, if_plot=False)

    if old_pkl_name != None:
        print('Remove old result:')
        
#         print(path_to_goolge_drive_dir + name_result + old_pkl_name)
        print(name_result + old_pkl_name)
    
        os.remove(path_to_goolge_drive_dir + name_result + old_pkl_name)

    import datetime        

    filename_result_with_time =\
    'result_' + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") +'.pkl'
    
    print('dict_result.keys()')
    print(dict_result.keys())
    
    print('dict_result[params].keys()')
    print(dict_result['params'].keys())
    
    print('dict_result[params')
    print(dict_result['params'])
    
    with open(path_to_goolge_drive_dir + name_result + filename_result_with_time, 'wb') as output_result:
        pickle.dump(dict_result, output_result)


#     print('Saved at ' + path_to_goolge_drive_dir + name_result + filename_result_with_time)
    print('Saved at ' + name_result + filename_result_with_time)

    return name_result, data_, dict_result['params']


def print_gpu_usage(params):
    device = params['device']
    
#     print('device')
#     print(device)
    
#     sys.exit()

    gpu_total_memory = torch.cuda.get_device_properties(device).total_memory
    gpu_cached = torch.cuda.memory_reserved(device)
    gpu_allocated = torch.cuda.memory_allocated(device)
    # f = c-a  # free inside cache

    print('total GPU memory: {0:.3f} GB, cached: {1:.3f} GB, allocated: {2:.3f} GB'.format(
        gpu_total_memory * 1e-9, gpu_cached * 1e-9, gpu_allocated * 1e-9))



def get_sort_profile():

    filepath = 'lprof0.txt'
    with open(filepath) as fp:
        line = fp.readline()
        cnt = 1
        list_percent_time = []
        list_line_1 = []
        while line:
            if cnt <=9:
                print(line)
            
            line_1 = line.strip().split()
            if len(line_1) > 5 and\
            line_1[0].replace('.','',1).isdigit() and\
            line_1[1].replace('Error: check if need to save shampoo.','',1).isdigit() and\
            line_1[2].replace('.','',1).isdigit() and\
            line_1[3].replace('.','',1).isdigit() and\
            line_1[4].replace('.','',1).isdigit():
                list_percent_time.append(float(line_1[4]))
                list_line_1.append(line)
            
            # print(line_1[0])
            line = fp.readline()
            cnt += 1


    list_percent_time = np.asarray(list_percent_time)

    argsort_list_percent_time = np.argsort(-list_percent_time)
    for i in argsort_list_percent_time:
        print(list_line_1[i])
