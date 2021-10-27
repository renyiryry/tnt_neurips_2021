def get_params(params, args):
    params['true_algorithm'] = params['algorithm']
    
    algorithm = params['algorithm']
    


    
    if algorithm == 'SGD-LRdecay-momentum':
        params['if_lr_decay'] = True
        params['algorithm'] = 'SGD-momentum'
    elif algorithm == 'Adam-noWarmStart-momentum-grad-LRdecay':
        params['if_lr_decay'] = True
        params['algorithm'] = 'Adam-noWarmStart-momentum-grad'
        
    elif algorithm == 'kfac-warmStart-lessInverse-no-max-no-LM-momentum-grad-LRdecay':
        params['if_lr_decay'] = True
        params['algorithm'] = 'kfac-warmStart-lessInverse-no-max-no-LM-momentum-grad'
        
    elif algorithm == 'kfac-correctFisher-warmStart-no-max-no-LM-momentum-grad-LRdecay':
        params['if_lr_decay'] = True
        params['algorithm'] = 'kfac-correctFisher-warmStart-no-max-no-LM-momentum-grad'
        
    elif algorithm == 'kfac-correctFisher-warmStart-NoMaxNoSqrt-no-LM-momentum-grad-LRdecay':
        params['if_lr_decay'] = True
        params['algorithm'] = 'kfac-correctFisher-warmStart-NoMaxNoSqrt-no-LM-momentum-grad'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-unregularized-grad-momentum-grad-LRdecay':
        params['if_lr_decay'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-unregularized-grad-momentum-grad'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-regularized-grad-momentum-grad-LRdecay':
        params['if_lr_decay'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-regularized-grad-momentum-grad'
        
    elif algorithm == 'shampoo-allVariables-filterFlattening-warmStart-momentum-grad-LRdecay':
        params['if_lr_decay'] = True
        params['algorithm'] = 'shampoo-allVariables-filterFlattening-warmStart-momentum-grad'
        
    elif algorithm == 'shampoo-allVariables-filterFlattening-warmStart-lessInverse-momentum-grad-LRdecay':
        params['if_lr_decay'] = True
        params['algorithm'] = 'shampoo-allVariables-filterFlattening-warmStart-lessInverse-momentum-grad'
        
    elif algorithm == 'matrix-normal-same-trace-allVariables-warmStart-momentum-grad-LRdecay':
        params['if_lr_decay'] = True
        params['algorithm'] = 'matrix-normal-same-trace-allVariables-warmStart-momentum-grad'
    elif algorithm == 'matrix-normal-same-trace-allVariables-KFACReshaping-warmStart-momentum-grad-LRdecay':
        params['if_lr_decay'] = True
        params['algorithm'] = 'matrix-normal-same-trace-allVariables-KFACReshaping-warmStart-momentum-grad'
    
    elif algorithm == 'matrix-normal-correctFisher-allVariables-filterFlattening-warmStart-lessInverse-momentum-grad-LRdecay':
        params['if_lr_decay'] = True
        params['algorithm'] = 'matrix-normal-correctFisher-allVariables-filterFlattening-warmStart-lessInverse-momentum-grad'
    elif algorithm == 'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-momentum-grad-LRdecay':
        params['if_lr_decay'] = True
        params['algorithm'] = 'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-momentum-grad'
    elif algorithm == 'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-lessInverse-momentum-grad-LRdecay':
        params['if_lr_decay'] = True
        params['algorithm'] = 'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-lessInverse-momentum-grad'
    elif algorithm == 'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-TraceWithEpsilonDamping-momentum-grad-LRdecay':
        params['if_lr_decay'] = True
        params['algorithm'] = 'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-TraceWithEpsilonDamping-momentum-grad'
        
    elif algorithm == 'matrix-normal-correctFisher-same-trace-allVariables-filterFlattening-warmStart-momentum-grad-LRdecay':
        params['if_lr_decay'] = True
        params['algorithm'] = 'matrix-normal-correctFisher-same-trace-allVariables-filterFlattening-warmStart-momentum-grad'
        
    elif algorithm == 'matrix-normal-correctFisher-same-trace-allVariables-filterFlattening-warmStart-lessInverse-momentum-grad-LRdecay':
        params['if_lr_decay'] = True
        params['algorithm'] = 'matrix-normal-correctFisher-same-trace-allVariables-filterFlattening-warmStart-lessInverse-momentum-grad'
        
    elif algorithm == 'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-momentum-grad-LRdecay':
        params['if_lr_decay'] = True
        params['algorithm'] = 'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-momentum-grad'
    elif algorithm == 'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-lessInverse-momentum-grad-LRdecay':
        params['if_lr_decay'] = True
        params['algorithm'] = 'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-lessInverse-momentum-grad'
        
    elif algorithm == 'matrix-normal-EF-same-trace-allVariables-filterFlattening-warmStart-momentum-grad-LRdecay':
        params['if_lr_decay'] = True
        params['algorithm'] = 'matrix-normal-EF-same-trace-allVariables-filterFlattening-warmStart-momentum-grad'
        
    elif algorithm in ['SGD-momentum',
                       'Adam-noWarmStart-momentum-grad',
                       'Fisher-BD',
                       'Fisher-BD-momentum-grad',
                       'kfac-warmStart-no-max-no-LM-momentum-grad',
                       'kfac-warmStart-lessInverse-no-max-no-LM-momentum-grad',
                       'kfac-correctFisher-warmStart-NoMaxNoSqrt-no-LM-momentum-grad',
                       'kfac-correctFisher-warmStart-no-max-no-LM-momentum-grad',
                       'kfac-correctFisher-warmStart-lessInverse-no-max-no-LM-momentum-grad',
                       'kfac-correctFisher-warmStart-lessInverse-NoMaxNoSqrt-no-LM-momentum-grad',
                       'Kron-BFGS-homo-no-norm-gate-HessianActionV2-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-regularized-grad-momentum-grad',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-Sqrt-regularized-grad-momentum-grad',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-SqrtT-regularized-grad-momentum-grad',
                       'shampoo-allVariables-warmStart-momentum-grad',
                       'shampoo-allVariables-warmStart-lessInverse-momentum-grad',
                       'shampoo-allVariables-filterFlattening-warmStart-momentum-grad',
                       'shampoo-allVariables-filterFlattening-warmStart-lessInverse-momentum-grad',
                       'matrix-normal-allVariables-warmStart-MaxEigDamping-momentum-grad',
                       'matrix-normal-same-trace-allVariables-warmStart-momentum-grad',
                       'matrix-normal-same-trace-allVariables-warmStart-AvgEigDamping-momentum-grad',
                       'matrix-normal-same-trace-allVariables-warmStart-MaxEigDamping-momentum-grad',
                       'matrix-normal-same-trace-allVariables-filterFlattening-warmStart-momentum-grad',
                       'matrix-normal-same-trace-allVariables-KFACReshaping-warmStart-momentum-grad',
                       'matrix-normal-correctFisher-allVariables-filterFlattening-warmStart-lessInverse-momentum-grad',
                       'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-momentum-grad',
                       'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-MaxEigWithEpsilonDamping-momentum-grad',
                       'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-AvgEigWithEpsilonDamping-momentum-grad',
                       'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-TraceWithEpsilonDamping-momentum-grad',
                       'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-momentum-grad',
                       'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-momentum-grad',
                       'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-lessInverse-momentum-grad',
                       'matrix-normal-EF-same-trace-allVariables-filterFlattening-warmStart-momentum-grad',]:
        params['if_lr_decay'] = False
    else:
        print('algorithm')
        print(algorithm)
        sys.exit()
    
    # get rid of momentum-grad
    algorithm = params['algorithm']
    if algorithm == 'Kron-BFGS-LM-unregularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-LM-unregularized-grad'
        
    elif algorithm == 'Kron-BFGS-LM-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-LM-regularized-grad'
        
    elif algorithm == 'Kron-BFGS-LM-sqrt-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-LM-sqrt-regularized-grad'
        
    elif algorithm == 'Kron-BFGS-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-regularized-grad'
        
    elif algorithm == 'Kron-BFGS-no-norm-gate-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-no-norm-gate-regularized-grad'
        
    elif algorithm == 'Kron-BFGS-no-norm-gate-momentum-s-y-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-no-norm-gate-momentum-s-y-regularized-grad'
        
    elif algorithm == 'Kron-BFGS-no-norm-gate-momentum-s-y-damping-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-no-norm-gate-momentum-s-y-damping-regularized-grad'
        
    elif algorithm == 'Kron-BFGS-no-norm-gate-momentum-s-y-damping-unregularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-no-norm-gate-momentum-s-y-damping-unregularized-grad'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-momentum-s-y-damping-unregularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-momentum-s-y-damping-unregularized-grad'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-momentum-s-y-damping-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-momentum-s-y-damping-regularized-grad'
        
    elif algorithm == 'Kron-BFGS-no-norm-gate-damping-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-no-norm-gate-damping-regularized-grad'
        
    elif algorithm == 'Kron-BFGS-no-norm-gate-Shiqian-damping-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-no-norm-gate-Shiqian-damping-regularized-grad'
        
    elif algorithm == 'Kron-BFGS-no-norm-gate-Shiqian-damping-unregularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-no-norm-gate-Shiqian-damping-unregularized-grad'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-Shiqian-damping-unregularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-Shiqian-damping-unregularized-grad'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-Powell-H-damping-unregularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-Powell-H-damping-unregularized-grad'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-PowellBDamping-unregularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-PowellBDamping-unregularized-grad'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-PowellHDampingV2-unregularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-PowellHDampingV2-unregularized-grad'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-Powell-H-damping-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-Powell-H-damping-regularized-grad'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-Powell-double-damping-unregularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-Powell-double-damping-unregularized-grad'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-PowellDoubleDampingV2-unregularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-PowellDoubleDampingV2-unregularized-grad'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-momentum-s-y-Powell-double-damping-unregularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-momentum-s-y-Powell-double-damping-unregularized-grad'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-momentum-s-y-Powell-double-damping-regularized-grad'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping-regularized-grad'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-HessianActionV2-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-HessianActionV2-momentum-s-y-Powell-double-damping-regularized-grad'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-Powell-double-damping-regularized-grad'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-DDV2-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-DDV2-regularized-grad'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2IdentityInitial-momentum-s-y-DDV2-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2IdentityInitial-momentum-s-y-DDV2-regularized-grad'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-unregularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-unregularized-grad'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-regularized-grad'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-miniBatchANotDamped-HessianActionV2-momentum-s-y-DDV2-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-miniBatchANotDamped-HessianActionV2-momentum-s-y-DDV2-regularized-grad'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-DDV2-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-DDV2-regularized-grad'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-Sqrt-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-Sqrt-regularized-grad'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-SqrtT-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-SqrtT-regularized-grad'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-KFACSplitting-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-KFACSplitting-regularized-grad'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-DDV2-extraStep-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-DDV2-extraStep-regularized-grad'
        
    elif algorithm == 'Kron-(L)BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-(L)BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-Powell-double-damping-regularized-grad'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2IdentityInitial-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2IdentityInitial-momentum-s-y-Powell-double-damping-regularized-grad'
        
    elif algorithm == 'Kron-BFGS(L)-homo-no-norm-gate-HessianActionV2-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS(L)-homo-no-norm-gate-HessianActionV2-momentum-s-y-Powell-double-damping-regularized-grad'
        
    elif algorithm == 'Kron-BFGS(L)-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS(L)-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-Powell-double-damping-regularized-grad'
        
    elif algorithm == 'Kron-BFGS(L)-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-DDV2-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS(L)-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-DDV2-regularized-grad'
        
    elif algorithm == 'Kron-BFGS(L)-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS(L)-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-regularized-grad'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping-doubleGrad-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping-doubleGrad-regularized-grad'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-scaledHessianAction-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-scaledHessianAction-momentum-s-y-Powell-double-damping-regularized-grad'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-HessianActionIdentityInitial-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-HessianActionIdentityInitial-momentum-s-y-Powell-double-damping-regularized-grad'
        
    elif algorithm == 'Kron-LBFGS-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-LBFGS-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping-regularized-grad'
        
    elif algorithm == 'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping-regularized-grad'
        
    elif algorithm == 'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-PowellDoubleDampingSkip-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-PowellDoubleDampingSkip-regularized-grad'
        
    elif algorithm == 'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-DoubleDamping-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-DoubleDamping-regularized-grad'
        
    elif algorithm == 'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-H-damping-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-H-damping-regularized-grad'
        
    elif algorithm == 'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-B0-damping-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-B0-damping-regularized-grad'
        
    elif algorithm == 'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-Shiqian-damping-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-Shiqian-damping-regularized-grad'
        
    elif algorithm == 'Kron-(L)BFGS-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-(L)BFGS-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping-regularized-grad'
        
    elif algorithm == 'Kron-LBFGS-homo-no-norm-gate-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-LBFGS-homo-no-norm-gate-momentum-s-y-Powell-double-damping-regularized-grad'
        
    elif algorithm == 'Kron-BFGS-homo-identity-unregularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-identity-unregularized-grad'
        
    elif algorithm == 'Kron-BFGS-homo-identity-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-identity-regularized-grad'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-Powell-double-damping-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-Powell-double-damping-regularized-grad'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-Hessian-action-Powell-double-damping-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-Hessian-action-Powell-double-damping-regularized-grad'
        
    elif algorithm == 'Kron-BFGS-no-norm-gate-damping-unregularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-no-norm-gate-damping-unregularized-grad'
    
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-damping-unregularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-damping-unregularized-grad'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-damping-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-damping-regularized-grad'
        
    elif algorithm == 'Kron-BFGS-no-norm-gate-unregularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-no-norm-gate-unregularized-grad'
        
    elif algorithm == 'Kron-BFGS-unregularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-unregularized-grad'
        
    elif algorithm == 'Kron-BFGS-wrong-unregularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-wrong-unregularized-grad'
        
    elif algorithm == 'Kron-BFGS-homo-regularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-regularized-grad'
        
    elif algorithm == 'Kron-BFGS-homo-unregularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-homo-unregularized-grad'
        
    elif algorithm == 'Kron-BFGS-Hessian-action-unregularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-Hessian-action-unregularized-grad'
        
    elif algorithm == 'Kron-BFGS-wrong-Hessian-action-unregularized-grad-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-wrong-Hessian-action-unregularized-grad'
        
    elif algorithm == 'Kron-BFGS-LM-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Kron-BFGS-LM'
        
    elif algorithm == 'Fisher-BD-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Fisher-BD'
        
    elif algorithm == 'kfac-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'kfac'
        
    elif algorithm == 'kfac-momentum-grad-test':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'kfac-test'
        
    elif algorithm == 'kfac-no-max-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'kfac-no-max'
        
    elif algorithm == 'kfac-NoMaxNoSqrt-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'kfac-NoMaxNoSqrt'
        
    elif algorithm == 'kfac-NoMaxNoSqrt-no-LM-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'kfac-NoMaxNoSqrt-no-LM'
        
    elif algorithm == 'kfac-no-max-no-LM-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'kfac-no-max-no-LM'
        
    elif algorithm == 'kfac-warmStart-no-max-no-LM-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'kfac-warmStart-no-max-no-LM'
        
    elif algorithm == 'kfac-warmStart-lessInverse-no-max-no-LM-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'kfac-warmStart-lessInverse-no-max-no-LM'
        
    elif algorithm == 'kfac-warmStart-lessInverse-NoMaxNoSqrt-no-LM-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'kfac-warmStart-lessInverse-NoMaxNoSqrt-no-LM'
        
    elif algorithm == 'kfac-correctFisher-warmStart-no-max-no-LM-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'kfac-correctFisher-warmStart-no-max-no-LM'
        
    elif algorithm == 'kfac-correctFisher-warmStart-NoMaxNoSqrt-no-LM-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'kfac-correctFisher-warmStart-NoMaxNoSqrt-no-LM'
        
    elif algorithm == 'kfac-correctFisher-warmStart-lessInverse-no-max-no-LM-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'kfac-correctFisher-warmStart-lessInverse-no-max-no-LM'
        
    elif algorithm == 'kfac-correctFisher-warmStart-lessInverse-NoMaxNoSqrt-no-LM-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'kfac-correctFisher-warmStart-lessInverse-NoMaxNoSqrt-no-LM'
        
    elif algorithm == 'kfac-no-max-epsilon-A-G-no-LM-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'kfac-no-max-epsilon-A-G-no-LM'
        
    elif algorithm == 'SGD-momentum':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'SGD'
    
    elif algorithm == 'SGD-LRdecay-momentum':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'SGD-LRdecay'
        
    elif algorithm == 'RMSprop-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'RMSprop'
        
    elif algorithm == 'RMSprop-warmStart-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'RMSprop-warmStart'
        
    elif algorithm == 'RMSprop-momentum-grad-test':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'RMSprop-test'
        
    elif algorithm == 'Adam-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Adam'
        
    elif algorithm == 'Adam-momentum-grad-test':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Adam-test'
        
    elif algorithm == 'Adam-noWarmStart-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Adam-noWarmStart'
        
    elif algorithm == 'shampoo-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'shampoo'
        
    elif algorithm == 'shampoo-allVariables-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'shampoo-allVariables'
        
    elif algorithm == 'shampoo-allVariables-warmStart-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'shampoo-allVariables-warmStart'
        
    elif algorithm == 'shampoo-allVariables-warmStart-lessInverse-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'shampoo-allVariables-warmStart-lessInverse'
        
    elif algorithm == 'shampoo-allVariables-filterFlattening-warmStart-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'shampoo-allVariables-filterFlattening-warmStart'
        
    elif algorithm == 'shampoo-allVariables-filterFlattening-warmStart-lessInverse-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'shampoo-allVariables-filterFlattening-warmStart-lessInverse'
        
    elif algorithm == 'shampoo-no-sqrt-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'shampoo-no-sqrt'
        
    elif algorithm == 'shampoo-no-sqrt-Fisher-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'shampoo-no-sqrt-Fisher'
        
    elif algorithm == 'shampoo-no-sqrt-Fisher-momentum-grad-test':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'shampoo-no-sqrt-Fisher-test'
        
    elif algorithm == 'matrix-normal-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'matrix-normal'
        
    elif algorithm == 'matrix-normal-allVariables-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'matrix-normal-allVariables'
        
    elif algorithm == 'matrix-normal-allVariables-warmStart-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'matrix-normal-allVariables-warmStart'
        
    elif algorithm == 'matrix-normal-allVariables-warmStart-MaxEigDamping-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'matrix-normal-allVariables-warmStart-MaxEigDamping'
        
    elif algorithm == 'matrix-normal-allVariables-warmStart-noPerDimDamping-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'matrix-normal-allVariables-warmStart-noPerDimDamping'
        
    elif algorithm == 'matrix-normal-same-trace-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'matrix-normal-same-trace'
        
    elif algorithm == 'matrix-normal-same-trace-warmStart-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'matrix-normal-same-trace-warmStart'
        
    elif algorithm == 'matrix-normal-same-trace-warmStart-noPerDimDamping-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'matrix-normal-same-trace-warmStart-noPerDimDamping'
        
    elif algorithm == 'matrix-normal-same-trace-allVariables-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'matrix-normal-same-trace-allVariables'
        
    elif algorithm == 'matrix-normal-same-trace-allVariables-warmStart-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'matrix-normal-same-trace-allVariables-warmStart'
        
    elif algorithm == 'matrix-normal-same-trace-allVariables-warmStart-AvgEigDamping-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'matrix-normal-same-trace-allVariables-warmStart-AvgEigDamping'
        
    elif algorithm == 'matrix-normal-same-trace-allVariables-warmStart-MaxEigDamping-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'matrix-normal-same-trace-allVariables-warmStart-MaxEigDamping'
        
    elif algorithm == 'matrix-normal-same-trace-allVariables-filterFlattening-warmStart-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'matrix-normal-same-trace-allVariables-filterFlattening-warmStart'
        
    elif algorithm == 'matrix-normal-same-trace-allVariables-KFACReshaping-warmStart-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'matrix-normal-same-trace-allVariables-KFACReshaping-warmStart'
        
    elif algorithm == 'matrix-normal-same-trace-allVariables-warmStart-noPerDimDamping-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'matrix-normal-same-trace-allVariables-warmStart-noPerDimDamping'
        
    elif algorithm == 'matrix-normal-correctFisher-allVariables-filterFlattening-warmStart-lessInverse-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'matrix-normal-correctFisher-allVariables-filterFlattening-warmStart-lessInverse'
        
    elif algorithm == 'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart'
        
    elif algorithm == 'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-lessInverse-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-lessInverse'
        
    elif algorithm == 'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-MaxEigWithEpsilonDamping-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-MaxEigWithEpsilonDamping'
        
    elif algorithm == 'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-AvgEigWithEpsilonDamping-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-AvgEigWithEpsilonDamping'
        
    elif algorithm == 'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-TraceWithEpsilonDamping-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-TraceWithEpsilonDamping'
        
    elif algorithm == 'matrix-normal-correctFisher-same-trace-allVariables-filterFlattening-warmStart-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'matrix-normal-correctFisher-same-trace-allVariables-filterFlattening-warmStart'
        
    elif algorithm == 'matrix-normal-correctFisher-same-trace-allVariables-filterFlattening-warmStart-lessInverse-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'matrix-normal-correctFisher-same-trace-allVariables-filterFlattening-warmStart-lessInverse'
        
    elif algorithm == 'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping'
        
    elif algorithm == 'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart'
        
    elif algorithm == 'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-lessInverse-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-lessInverse'
        
    elif algorithm == 'matrix-normal-EF-same-trace-allVariables-filterFlattening-warmStart-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'matrix-normal-EF-same-trace-allVariables-filterFlattening-warmStart'
        
    else:   
        if algorithm in ['SGD',
                         'SGD-yura-MA',
                         'SGD-yura',
                         'SMW-Fisher',
                         'SMW-Fisher-momentum',
                         'SMW-GN',
                         'shampoo',
                         'matrix-normal-same-trace',
                         'kfac-TR',
                         'kfac-CG',
                         'kfac',
                         'kfac-test',
                         'kfac-no-max',
                         'kfac-no-max-no-LM',
                         'ekfac-EF',
                         'Kron-BFGS',
                         'Kron-BFGS-no-norm-gate-regularized-grad',
                         'Kron-BFGS-no-norm-gate-momentum-s-y-regularized-grad',
                         'Kron-BFGS-no-norm-gate-momentum-s-y-damping-regularized-grad',
                         'Kron-BFGS-no-norm-gate-damping-regularized-grad',
                         'Kron-BFGS-no-norm-gate-Shiqian-damping-regularized-grad',
                         'Kron-BFGS-unregularized-grad',
                         'Kron-BFGS-wrong-unregularized-grad',
                         'Kron-BFGS-regularized-grad',
                         'Kron-BFGS-homo-regularized-grad',
                         'Kron-BFGS-LM',
                         'Kron-BFGS-LM-unregularized-grad',
                         'Kron-BFGS-LM-regularized-grad',
                         'Kron-BFGS-LM-sqrt-regularized-grad',
                         'Kron-BFGS-Hessian-action',
                         'Kron-BFGS-Hessian-action-unregularized-grad',
                         'Kron-BFGS-1st-layer-only',
                         'Kron-BFGS-block',
                         'Kron-SGD',
                         'Kron-SGD-test',
                         'BFGS',
                         'BFGS-homo']:
            params['if_momentum_gradient'] = False
        elif algorithm in ['matrix-normal-LM-momentum-grad',
                           'matrix-normal-same-trace-LM-momentum-grad',
                           'kfac-momentum-grad-CG',
                           'kfac-momentum-grad-TR',
                           'Kron-BFGS-momentum-grad',
                           'Kron-BFGS-Hessian-action-momentum-grad']:
            params['if_momentum_gradient'] = True
        else:
            print('Error: unkown if momentum gradient for ' + algorithm)
            sys.exit()
    
    # get rid of (un)regularized grad
    algorithm = params['algorithm']
    if algorithm == 'Kron-BFGS-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS'
        
    elif algorithm == 'Kron-BFGS-no-norm-gate-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS-no-norm-gate'
        
    elif algorithm == 'Kron-BFGS-no-norm-gate-momentum-s-y-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS-no-norm-gate-momentum-s-y'
        
    elif algorithm == 'Kron-BFGS-no-norm-gate-momentum-s-y-damping-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS-no-norm-gate-momentum-s-y-damping'
        
    elif algorithm == 'Kron-BFGS-no-norm-gate-momentum-s-y-damping-unregularized-grad':
        params['if_regularized_grad'] = False
        params['algorithm'] = 'Kron-BFGS-no-norm-gate-momentum-s-y-damping'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-momentum-s-y-damping-unregularized-grad':
        params['if_regularized_grad'] = False
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-momentum-s-y-damping'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-momentum-s-y-damping-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-momentum-s-y-damping'
        
    elif algorithm == 'Kron-BFGS-no-norm-gate-damping-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS-no-norm-gate-damping'
        
    elif algorithm == 'Kron-BFGS-no-norm-gate-Shiqian-damping-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS-no-norm-gate-Shiqian-damping'
        
    elif algorithm == 'Kron-BFGS-no-norm-gate-Shiqian-damping-unregularized-grad':
        params['if_regularized_grad'] = False
        params['algorithm'] = 'Kron-BFGS-no-norm-gate-Shiqian-damping'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-Shiqian-damping-unregularized-grad':
        params['if_regularized_grad'] = False
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-Shiqian-damping'
    
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-Powell-H-damping-unregularized-grad':
        params['if_regularized_grad'] = False
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-Powell-H-damping'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-PowellBDamping-unregularized-grad':
        params['if_regularized_grad'] = False
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-PowellBDamping'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-PowellHDampingV2-unregularized-grad':
        params['if_regularized_grad'] = False
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-PowellHDampingV2'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-Powell-H-damping-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-Powell-H-damping'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-Powell-double-damping-unregularized-grad':
        params['if_regularized_grad'] = False
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-Powell-double-damping'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-PowellDoubleDampingV2-unregularized-grad':
        params['if_regularized_grad'] = False
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-PowellDoubleDampingV2'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-momentum-s-y-Powell-double-damping-unregularized-grad':
        params['if_regularized_grad'] = False
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-momentum-s-y-Powell-double-damping'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-momentum-s-y-Powell-double-damping-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-momentum-s-y-Powell-double-damping'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-HessianActionV2-momentum-s-y-Powell-double-damping-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-HessianActionV2-momentum-s-y-Powell-double-damping'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-Powell-double-damping-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-Powell-double-damping'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-DDV2-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-DDV2'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2IdentityInitial-momentum-s-y-DDV2-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2IdentityInitial-momentum-s-y-DDV2'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-unregularized-grad':
        params['if_regularized_grad'] = False
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-miniBatchANotDamped-HessianActionV2-momentum-s-y-DDV2-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-miniBatchANotDamped-HessianActionV2-momentum-s-y-DDV2'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-DDV2-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-DDV2'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-Sqrt-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-Sqrt'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-SqrtT-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-SqrtT'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-KFACSplitting-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-KFACSplitting'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-DDV2-extraStep-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-DDV2-extraStep'
        
    elif algorithm == 'Kron-(L)BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-Powell-double-damping-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-(L)BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-Powell-double-damping'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2IdentityInitial-momentum-s-y-Powell-double-damping-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2IdentityInitial-momentum-s-y-Powell-double-damping'
        
    elif algorithm == 'Kron-BFGS(L)-homo-no-norm-gate-HessianActionV2-momentum-s-y-Powell-double-damping-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS(L)-homo-no-norm-gate-HessianActionV2-momentum-s-y-Powell-double-damping'
        
    elif algorithm == 'Kron-BFGS(L)-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-Powell-double-damping-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS(L)-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-Powell-double-damping'
        
    elif algorithm == 'Kron-BFGS(L)-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-DDV2-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS(L)-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-DDV2'
        
    elif algorithm == 'Kron-BFGS(L)-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS(L)-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping-doubleGrad-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping-doubleGrad'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-scaledHessianAction-momentum-s-y-Powell-double-damping-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-scaledHessianAction-momentum-s-y-Powell-double-damping'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-HessianActionIdentityInitial-momentum-s-y-Powell-double-damping-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-HessianActionIdentityInitial-momentum-s-y-Powell-double-damping'
        
    elif algorithm == 'Kron-LBFGS-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-LBFGS-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping'
        
    elif algorithm == 'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping'
        
    elif algorithm == 'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-PowellDoubleDampingSkip-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-PowellDoubleDampingSkip'
        
    elif algorithm == 'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-DoubleDamping-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-DoubleDamping'
        
    elif algorithm == 'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-H-damping-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-H-damping'
        
    elif algorithm == 'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-B0-damping-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-B0-damping'
        
    elif algorithm == 'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-Shiqian-damping-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-Shiqian-damping'
        
    elif algorithm == 'Kron-(L)BFGS-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-(L)BFGS-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping'
        
    elif algorithm == 'Kron-LBFGS-homo-no-norm-gate-momentum-s-y-Powell-double-damping-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-LBFGS-homo-no-norm-gate-momentum-s-y-Powell-double-damping'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-Powell-double-damping-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-Powell-double-damping'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-Hessian-action-Powell-double-damping-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-Hessian-action-Powell-double-damping'
        
    elif algorithm == 'Kron-BFGS-homo-identity-unregularized-grad':
        params['if_regularized_grad'] = False
        params['algorithm'] = 'Kron-BFGS-homo-identity'
        
    elif algorithm == 'Kron-BFGS-homo-identity-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS-homo-identity'
        
    elif algorithm == 'Kron-BFGS-no-norm-gate-damping-unregularized-grad':
        params['if_regularized_grad'] = False
        params['algorithm'] = 'Kron-BFGS-no-norm-gate-damping'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-damping-unregularized-grad':
        params['if_regularized_grad'] = False
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-damping'
        
    elif algorithm == 'Kron-BFGS-homo-no-norm-gate-damping-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-damping'
        
    elif algorithm == 'Kron-BFGS-no-norm-gate-unregularized-grad':
        params['if_regularized_grad'] = False
        params['algorithm'] = 'Kron-BFGS-no-norm-gate'
        
    elif algorithm == 'Kron-BFGS-unregularized-grad':
        params['if_regularized_grad'] = False
        params['algorithm'] = 'Kron-BFGS'
        
    elif algorithm == 'Kron-BFGS-wrong-unregularized-grad':
        params['if_regularized_grad'] = False
        params['algorithm'] = 'Kron-BFGS-wrong'
        
    elif algorithm == 'Kron-BFGS-homo-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS-homo'
        
    elif algorithm == 'Kron-BFGS-homo-unregularized-grad':
        params['if_regularized_grad'] = False
        params['algorithm'] = 'Kron-BFGS-homo'
        
    elif algorithm == 'Kron-BFGS-LM-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS-LM'
        
    elif algorithm == 'Kron-BFGS-LM-sqrt-regularized-grad':
        params['if_regularized_grad'] = True
        params['algorithm'] = 'Kron-BFGS-LM-sqrt'
        
    elif algorithm == 'Kron-BFGS-Hessian-action-unregularized-grad':
        params['if_regularized_grad'] = False
        params['algorithm'] = 'Kron-BFGS-Hessian-action'
        
    elif algorithm == 'Kron-BFGS-wrong-Hessian-action-unregularized-grad':
        params['if_regularized_grad'] = False
        params['algorithm'] = 'Kron-BFGS-wrong-Hessian-action'
    else:
        if algorithm in ['Kron-BFGS-LM-unregularized-grad']:
            params['if_regularized_grad'] = False
        elif algorithm in ['SGD',
                           'RMSprop',
                           'RMSprop-warmStart',
                           'Adam',
                           'Adam-test',
                           'Adam-noWarmStart',
                           'shampoo',
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
                           'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-lessInverse',
                           'matrix-normal-EF-same-trace-allVariables-filterFlattening-warmStart',
                           'BFGS',
                           'BFGS-homo',
                           'Fisher-BD',
                           'kfac',
                           'kfac-no-max',
                           'kfac-NoMaxNoSqrt',
                           'kfac-NoMaxNoSqrt-no-LM',
                           'kfac-no-max-no-LM',
                           'kfac-warmStart-no-max-no-LM',
                           'kfac-correctFisher-warmStart-NoMaxNoSqrt-no-LM',
                           'kfac-correctFisher-warmStart-no-max-no-LM',
                           'kfac-correctFisher-warmStart-lessInverse-no-max-no-LM',
                           'kfac-correctFisher-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                           'kfac-warmStart-lessInverse-no-max-no-LM',
                           'kfac-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                           'kfac-no-max-epsilon-A-G-no-LM']:
            params['if_regularized_grad'] = True
        else:
            print('Error: unkown if_regularized_grad for ' + algorithm)
            sys.exit()
            
    algorithm = params['algorithm']  

    if algorithm == 'Kron-BFGS-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping-doubleGrad':
        params['if_double_grad'] = True
        params['algorithm'] = 'Kron-BFGS-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping'
    elif algorithm in ['SGD',
                       'RMSprop-warmStart',
                       'Adam-noWarmStart',
                       'Fisher-BD',
                       'kfac',
                       'kfac-no-max-no-LM',
                       'kfac-warmStart-no-max-no-LM',
                       'kfac-correctFisher-warmStart-no-max-no-LM',
                       'kfac-correctFisher-warmStart-NoMaxNoSqrt-no-LM',
                       'kfac-correctFisher-warmStart-lessInverse-no-max-no-LM',
                       'kfac-correctFisher-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                       'kfac-warmStart-lessInverse-no-max-no-LM',
                       'kfac-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                       'kfac-NoMaxNoSqrt-no-LM',
                       'kfac-no-max-epsilon-A-G-no-LM',
                       'shampoo',
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
                       'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-lessInverse',
                       'matrix-normal-EF-same-trace-allVariables-filterFlattening-warmStart',
                       'matrix-normal-same-trace-allVariables-warmStart-noPerDimDamping',
                       'Kron-BFGS-homo-no-norm-gate-momentum-s-y-Powell-double-damping',
                       'Kron-BFGS-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping',
                       'Kron-BFGS-homo-no-norm-gate-HessianActionV2-momentum-s-y-Powell-double-damping',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-Powell-double-damping',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-DDV2',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2IdentityInitial-momentum-s-y-DDV2',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchANotDamped-HessianActionV2-momentum-s-y-DDV2',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-DDV2',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-Sqrt',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-SqrtT',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-KFACSplitting',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-DDV2-extraStep',
                       'Kron-(L)BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-Powell-double-damping',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2IdentityInitial-momentum-s-y-Powell-double-damping',
                       'Kron-BFGS(L)-homo-no-norm-gate-HessianActionV2-momentum-s-y-Powell-double-damping',
                       'Kron-BFGS(L)-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-Powell-double-damping',
                       'Kron-BFGS(L)-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-DDV2',
                       'Kron-BFGS(L)-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2',
                       'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping']:
        params['if_double_grad'] = False
    else:
        print('error: need to check if_double_grad for ' + algorithm)
        sys.exit()
    
    algorithm = params['algorithm']

    if algorithm in ['SGD-VA',
                     'SGD-yura-BD',
                     'SGD-yura-old',
                     'SGD-yura-MA',
                     'SGD-yura',
                     'SGD-signVAsqrt',
                     'SGD-signVAerf',
                     'SGD-signVA',
                     'SGD-sign',
                     'SGD-momentum-yura',
                     'SGD-momentum',
                     'SGD',
                     'shampoo',
                     'shampoo-allVariables',
                     'shampoo-allVariables-warmStart',
                     'shampoo-allVariables-warmStart-lessInverse',
                     'shampoo-allVariables-filterFlattening-warmStart',
                     'shampoo-allVariables-filterFlattening-warmStart-lessInverse',
                     'shampoo-no-sqrt',
                     'matrix-normal-EF-same-trace-allVariables-filterFlattening-warmStart',
                     'RMSprop',
                     'RMSprop-warmStart',
                     'Adam',
                     'Adam-test',
                     'Adam-noWarmStart',
                     'RMSprop-no-sqrt',
                     'BFGS',
                     'BFGS-homo']:
        params['if_second_order_algorithm'] = False
    elif algorithm in ['SMW-Fisher-signVAsqrt-p',
                       'SMW-Fisher-VA-p',
                       'SMW-Fisher-momentum-p-sign',
                       'SMW-Fisher-momentum-p',
                       'SMW-Fisher-sign',
                       'SMW-Fisher-different-minibatch',
                       'SMW-Fisher',
                       'SMW-Fisher-batch-grad-momentum-exponential-decay',
                       'SMW-Fisher-momentum',
                       'ekfac-EF-VA',
                       'ekfac-EF',
                       'Fisher-BD',
                       'kfac-TR',
                       'kfac-CG',
                       'kfac-momentum-grad-CG',
                       'kfac-momentum-grad-TR',
                       'kfac',
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
                       'kfac-no-max-momentum-grad',
                       'kfac-EF',
                       'shampoo-no-sqrt-Fisher',
                       'shampoo-no-sqrt-Fisher-test',
                       'matrix-normal',
                       'matrix-normal-allVariables',
                       'matrix-normal-allVariables-warmStart',
                       'matrix-normal-allVariables-warmStart-MaxEigDamping',
                       'matrix-normal-allVariables-warmStart-noPerDimDamping',
                       'matrix-normal-momentum-grad',
                       'matrix-normal-LM-momentum-grad',
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
                       'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-lessInverse',
                       'SMW-Fisher-momentum-D_t-momentum',
                       'GI-Fisher',
                       'SMW-GN',
                       'SMW-Fisher-BD',
                       'RMSprop-individual-grad',
                       'RMSprop-individual-grad-no-sqrt',
                       'RMSprop-individual-grad-no-sqrt-Fisher',
                       'RMSprop-individual-grad-no-sqrt-LM',
                       'SMW-Fisher-batch-grad-momentum',
                       'SMW-Fisher-batch-grad',
                       'Kron-BFGS',
                       'Kron-BFGS-no-norm-gate',
                       'Kron-BFGS-no-norm-gate-momentum-s-y',
                       'Kron-BFGS-no-norm-gate-momentum-s-y-damping',
                       'Kron-BFGS-homo-no-norm-gate-momentum-s-y-damping',
                       'Kron-BFGS-no-norm-gate-damping',
                       'Kron-BFGS-homo-no-norm-gate-damping',
                       'Kron-BFGS-no-norm-gate-Shiqian-damping',
                       'Kron-BFGS-homo-no-norm-gate-Shiqian-damping',
                       'Kron-BFGS-homo-no-norm-gate-Powell-H-damping',
                       'Kron-BFGS-homo-no-norm-gate-PowellBDamping',
                       'Kron-BFGS-homo-no-norm-gate-PowellHDampingV2',
                       'Kron-BFGS-homo-no-norm-gate-Powell-double-damping',
                       'Kron-BFGS-homo-no-norm-gate-PowellDoubleDampingV2',
                       'Kron-BFGS-homo-no-norm-gate-momentum-s-y-Powell-double-damping',
                       'Kron-BFGS-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping',
                       'Kron-BFGS-homo-no-norm-gate-HessianActionV2-momentum-s-y-Powell-double-damping',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-Powell-double-damping',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-DDV2',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2IdentityInitial-momentum-s-y-DDV2',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchANotDamped-HessianActionV2-momentum-s-y-DDV2',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-DDV2',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-Sqrt',
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
                       'Kron-BFGS-momentum-grad',
                       'Kron-BFGS-unregularized-grad',
                       'Kron-BFGS-unregularized-grad-momentum-grad',
                       'Kron-BFGS-Hessian-action',
                       'Kron-BFGS-wrong-Hessian-action',
                       'Kron-BFGS-Hessian-action-unregularized-grad',
                       'Kron-BFGS-Hessian-action-momentum-grad',
                       'Kron-BFGS-LM',
                       'Kron-BFGS-LM-sqrt',
                       'Kron-BFGS-LM-unregularized-grad',
                       'Kron-BFGS-LM-momentum-grad',
                       'Kron-BFGS-1st-layer-only',
                       'Kron-BFGS-block',
                       'Kron-SGD']:
        params['if_second_order_algorithm'] = True
    else:
        print('Error: unknown if_second_order_algorithm for ' + algorithm)
        sys.exit()

    if algorithm in ['SMW-Fisher-signVAsqrt-p',
                     'SMW-Fisher-VA-p',
                   'SMW-Fisher-momentum-p-sign',
                   'SMW-Fisher-momentum-p',
                   'SMW-Fisher-sign',
                   'SMW-Fisher-different-minibatch',
                   'SMW-Fisher',
                   'SMW-Fisher-batch-grad-momentum-exponential-decay',
                   'SMW-Fisher-momentum',
                   'SMW-Fisher-momentum-D_t-momentum',
                   'GI-Fisher',
                   'kfac-TR',
                   'kfac-CG',
                   'kfac-momentum-grad-CG',
                   'kfac-momentum-grad-TR',
                   'kfac-momentum-grad',
                   'kfac',
                   'kfac-test',
                   'kfac-no-max',
                   'kfac-NoMaxNoSqrt',
                   'kfac-no-max-momentum-grad',
                   'kfac-EF',
                   'SMW-GN',
                   'SMW-Fisher-BD',
                   'RMSprop-individual-grad-no-sqrt-LM',
                   'SMW-Fisher-batch-grad-momentum',
                   'SMW-Fisher-batch-grad',
                   'matrix-normal-LM-momentum-grad',
                   'matrix-normal-same-trace-LM-momentum-grad',]:
        params['if_LM'] = True
    elif algorithm in ['Kron-BFGS-LM',
                       'Kron-BFGS-LM-sqrt',
                       'Kron-BFGS-LM-unregularized-grad',
                       'Kron-BFGS-LM-momentum-grad']:
        
        print('error: should use Kron_BFGS_LM')
        
        sys.exit()
        
        params['if_LM'] = True
    elif algorithm in ['ekfac-EF-VA',
                       'ekfac-EF',
                       'SGD-VA',
                       'SGD-yura-BD',
                       'SGD-yura-old',
                       'SGD-yura-MA',
                       'SGD-yura',
                       'SGD-signVAsqrt',
                       'SGD-signVAerf',
                       'SGD-signVA',
                       'SGD-sign',
                       'SGD-momentum-yura',
                       'SGD-momentum',
                       'SGD',
                       'shampoo',
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
                       'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-lessInverse',
                       'matrix-normal-EF-same-trace-allVariables-filterFlattening-warmStart',
                       'matrix-normal-same-trace-allVariables-warmStart-noPerDimDamping',
                       'RMSprop',
                       'RMSprop-warmStart',
                       'Adam',
                       'Adam-test',
                       'Adam-noWarmStart',
                       'RMSprop-no-sqrt',
                       'RMSprop-individual-grad',
                       'Fisher-BD',
                       'kfac-no-max-no-LM',
                       'kfac-warmStart-no-max-no-LM',
                       'kfac-correctFisher-warmStart-no-max-no-LM',
                       'kfac-correctFisher-warmStart-NoMaxNoSqrt-no-LM',
                       'kfac-correctFisher-warmStart-lessInverse-no-max-no-LM',
                       'kfac-correctFisher-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                       'kfac-warmStart-lessInverse-no-max-no-LM',
                       'kfac-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                       'kfac-NoMaxNoSqrt-no-LM',
                       'kfac-no-max-epsilon-A-G-no-LM',
                       'Kron-BFGS',
                       'Kron-BFGS-no-norm-gate',
                       'Kron-BFGS-no-norm-gate-momentum-s-y',
                       'Kron-BFGS-no-norm-gate-momentum-s-y-damping',
                       'Kron-BFGS-homo-no-norm-gate-momentum-s-y-damping',
                       'Kron-BFGS-no-norm-gate-damping',
                       'Kron-BFGS-homo-no-norm-gate-damping',
                       'Kron-BFGS-no-norm-gate-Shiqian-damping',
                       'Kron-BFGS-homo-no-norm-gate-Shiqian-damping',
                       'Kron-BFGS-homo-no-norm-gate-Powell-H-damping',
                       'Kron-BFGS-homo-no-norm-gate-PowellBDamping',
                       'Kron-BFGS-homo-no-norm-gate-PowellHDampingV2',
                       'Kron-BFGS-homo-no-norm-gate-Powell-double-damping',
                       'Kron-BFGS-homo-no-norm-gate-PowellDoubleDampingV2',
                       'Kron-BFGS-homo-no-norm-gate-momentum-s-y-Powell-double-damping',
                       'Kron-BFGS-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping',
                       'Kron-BFGS-homo-no-norm-gate-HessianActionV2-momentum-s-y-Powell-double-damping',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-Powell-double-damping',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-DDV2',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2IdentityInitial-momentum-s-y-DDV2',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchANotDamped-HessianActionV2-momentum-s-y-DDV2',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-DDV2',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-Sqrt',
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
                       'Kron-BFGS-momentum-grad',
                       'Kron-BFGS-unregularized-grad',
                       'Kron-BFGS-unregularized-grad-momentum-grad',
                       'Kron-BFGS-Hessian-action',
                       'Kron-BFGS-wrong-Hessian-action',
                       'Kron-BFGS-Hessian-action-unregularized-grad',
                       'Kron-BFGS-Hessian-action-momentum-grad',
                       'Kron-BFGS-1st-layer-only',
                       'Kron-BFGS-block',
                       'Kron-SGD',
                       'BFGS',
                       'BFGS-homo']:
        params['if_LM'] = False
    elif algorithm == 'RMSprop-individual-grad-no-sqrt' or\
    algorithm == 'RMSprop-individual-grad-no-sqrt-Fisher':
        params['if_LM'] = False
    else:
        print('Error: unknown if_LM')
        sys.exit()

    if algorithm in ['SMW-Fisher-batch-grad-momentum-exponential-decay',
                     'SMW-Fisher-batch-grad',
                     'SMW-Fisher-batch-grad-momentum',
                     'shampoo-no-sqrt-Fisher',
                     'shampoo-no-sqrt-Fisher-test',
                     'matrix-normal',
                     'matrix-normal-allVariables',
                     'matrix-normal-allVariables-warmStart',
                     'matrix-normal-allVariables-warmStart-MaxEigDamping',
                     'matrix-normal-allVariables-warmStart-noPerDimDamping',
                     'matrix-normal-LM-momentum-grad',
                     'matrix-normal-same-trace',
                     'matrix-normal-same-trace-warmStart',
                     'matrix-normal-same-trace-warmStart-noPerDimDamping',
                     'matrix-normal-same-trace-allVariables',
                     'matrix-normal-same-trace-allVariables-warmStart',
                     'matrix-normal-same-trace-allVariables-warmStart-noPerDimDamping',
                     'matrix-normal-same-trace-allVariables-warmStart-AvgEigDamping',
                     'matrix-normal-same-trace-allVariables-warmStart-MaxEigDamping',
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
        params['if_model_grad_N2'] = True
    elif algorithm in ['SGD-VA',
                       'SGD-signVAsqrt',
                       'SGD-signVAerf',
                       'SGD-signVA',
                       'SGD-sign',
                       'SGD-momentum-yura',
                       'SGD-momentum',
                       'SGD',
                       'SGD-yura-BD',
                       'SGD-yura-old',
                       'SGD-yura-MA',
                       'SGD-yura',
                       'SMW-Fisher-signVAsqrt-p',
                       'SMW-Fisher-VA-p',
                       'SMW-Fisher-momentum-p-sign',
                       'SMW-Fisher-momentum-p',
                       'SMW-Fisher-sign',
                       'SMW-Fisher-different-minibatch',
                       'SMW-Fisher',
                       'SMW-Fisher-momentum',
                       'SMW-GN',
                       'shampoo',
                       'shampoo-allVariables',
                       'shampoo-allVariables-warmStart',
                       'shampoo-allVariables-warmStart-lessInverse',
                       'shampoo-allVariables-filterFlattening-warmStart',
                       'shampoo-allVariables-filterFlattening-warmStart-lessInverse',
                       'shampoo-no-sqrt',
                       'matrix-normal-EF-same-trace-allVariables-filterFlattening-warmStart',
                       'RMSprop-individual-grad-no-sqrt-Fisher',
                       'RMSprop-individual-grad-no-sqrt',
                       'RMSprop-individual-grad',
                       'RMSprop',
                       'RMSprop-warmStart',
                       'Adam',
                       'Adam-test',
                       'Adam-noWarmStart',
                       'Fisher-BD',
                       'kfac-TR',
                       'kfac-CG',
                       'kfac-momentum-grad-CG',
                       'kfac-momentum-grad-TR',
                       'kfac-momentum-grad',
                       'kfac',
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
                       'kfac-no-max-momentum-grad',
                       'kfac-EF',
                       'ekfac-EF',
                       'ekfac-EF-VA',
                       'Kron-BFGS',
                       'Kron-BFGS-no-norm-gate',
                       'Kron-BFGS-no-norm-gate-momentum-s-y',
                       'Kron-BFGS-no-norm-gate-momentum-s-y-damping',
                       'Kron-BFGS-homo-no-norm-gate-momentum-s-y-damping',
                       'Kron-BFGS-no-norm-gate-damping',
                       'Kron-BFGS-homo-no-norm-gate-damping',
                       'Kron-BFGS-no-norm-gate-Shiqian-damping',
                       'Kron-BFGS-homo-no-norm-gate-Shiqian-damping',
                       'Kron-BFGS-homo-no-norm-gate-Powell-H-damping',
                       'Kron-BFGS-homo-no-norm-gate-PowellBDamping',
                       'Kron-BFGS-homo-no-norm-gate-PowellHDampingV2',
                       'Kron-BFGS-homo-no-norm-gate-Powell-double-damping',
                       'Kron-BFGS-homo-no-norm-gate-PowellDoubleDampingV2',
                       'Kron-BFGS-homo-no-norm-gate-momentum-s-y-Powell-double-damping',
                       'Kron-BFGS-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping',
                       'Kron-BFGS-homo-no-norm-gate-HessianActionV2-momentum-s-y-Powell-double-damping',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-Powell-double-damping',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-DDV2',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2IdentityInitial-momentum-s-y-DDV2',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchANotDamped-HessianActionV2-momentum-s-y-DDV2',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-DDV2',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-Sqrt',
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
                       'Kron-BFGS-LM',
                       'Kron-BFGS-LM-sqrt',
                       'Kron-BFGS-LM-unregularized-grad',
                       'Kron-BFGS-momentum-grad',
                       'Kron-BFGS-unregularized-grad',
                       'Kron-BFGS-unregularized-grad-momentum-grad',
                       'Kron-BFGS-LM-momentum-grad',
                       'Kron-BFGS-Hessian-action',
                       'Kron-BFGS-wrong-Hessian-action',
                       'Kron-BFGS-Hessian-action-unregularized-grad',
                       'Kron-BFGS-Hessian-action-momentum-grad',
                       'Kron-BFGS-1st-layer-only',
                       'Kron-BFGS-block',
                       'Kron-SGD',
                       'BFGS',
                       'BFGS-homo']:
        params['if_model_grad_N2'] = False
    else:
        print('Error: check if need model_grad_N2')
        sys.exit()

    

    if algorithm == 'ekfac-EF-VA' or\
    algorithm == 'ekfac-EF' or\
    algorithm == 'RMSprop-individual-grad-no-sqrt-LM' or\
    algorithm == 'RMSprop-individual-grad-no-sqrt-Fisher' or\
    algorithm == 'RMSprop-individual-grad-no-sqrt' or\
    algorithm == 'RMSprop-individual-grad' or\
    algorithm == 'RMSprop-no-sqrt':
        params['lambda_'] = args['RMSprop_epsilon']
#         params['tau'] = 0
    elif algorithm in ['SMW-Fisher-signVAsqrt-p',
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
                       'SMW-GN',
                       'kfac-TR',
                       'kfac-CG',
                       'kfac-momentum-grad-CG',
                       'kfac-momentum-grad-TR',
                       'kfac-momentum-grad',
                       'kfac',
                       'kfac-test',
                       'kfac-no-max',
                       'kfac-NoMaxNoSqrt',
                       'kfac-no-max-momentum-grad',
                       'kfac-EF',
                       'matrix-normal-LM-momentum-grad',
                       'matrix-normal-same-trace-LM-momentum-grad',
                       'Kron-BFGS-LM',
                       'Kron-BFGS-LM-sqrt',
                       'Kron-BFGS-LM-unregularized-grad',
                       'Kron-BFGS-LM-momentum-grad']:
        params['lambda_'] = args['lambda_']
#         params['tau'] = args['tau']
    elif algorithm in ['SGD-VA',
                       'SGD-yura-BD',
                       'SGD-yura-old',
                       'SGD-yura-MA',
                       'SGD-yura',
                       'SGD-signVAsqrt',
                       'SGD-signVAerf',
                       'SGD-signVA',
                       'SGD-sign',
                       'shampoo',
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
                       'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-lessInverse',
                       'matrix-normal-EF-same-trace-allVariables-filterFlattening-warmStart',
                       'matrix-normal-same-trace-allVariables-warmStart-noPerDimDamping',
                       'SGD-momentum-yura',
                       'SGD-momentum',
                       'SGD',
                       'RMSprop',
                       'RMSprop-warmStart',
                       'RMSprop-test',
                       'Adam',
                       'Adam-test',
                       'Adam-noWarmStart',
                       'Fisher-BD',
                       'kfac-no-max-no-LM',
                       'kfac-warmStart-no-max-no-LM',
                       'kfac-correctFisher-warmStart-no-max-no-LM',
                       'kfac-correctFisher-warmStart-NoMaxNoSqrt-no-LM',
                       'kfac-correctFisher-warmStart-lessInverse-no-max-no-LM',
                       'kfac-correctFisher-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                       'kfac-warmStart-lessInverse-no-max-no-LM',
                       'kfac-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                       'kfac-NoMaxNoSqrt-no-LM',
                       'kfac-no-max-epsilon-A-G-no-LM',
                       'Kron-BFGS',
                       'Kron-BFGS-no-norm-gate',
                       'Kron-BFGS-no-norm-gate-momentum-s-y',
                       'Kron-BFGS-no-norm-gate-momentum-s-y-damping',
                       'Kron-BFGS-homo-no-norm-gate-momentum-s-y-damping',
                       'Kron-BFGS-no-norm-gate-Shiqian-damping',
                       'Kron-BFGS-homo-no-norm-gate-Shiqian-damping',
                       'Kron-BFGS-homo-no-norm-gate-Powell-H-damping',
                       'Kron-BFGS-homo-no-norm-gate-PowellBDamping',
                       'Kron-BFGS-homo-no-norm-gate-PowellHDampingV2',
                       'Kron-BFGS-homo-no-norm-gate-Powell-double-damping',
                       'Kron-BFGS-homo-no-norm-gate-PowellDoubleDampingV2',
                       'Kron-BFGS-homo-no-norm-gate-momentum-s-y-Powell-double-damping',
                       'Kron-BFGS-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping',
                       'Kron-BFGS-homo-no-norm-gate-HessianActionV2-momentum-s-y-Powell-double-damping',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-Powell-double-damping',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-DDV2',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2IdentityInitial-momentum-s-y-DDV2',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchANotDamped-HessianActionV2-momentum-s-y-DDV2',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-DDV2',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-Sqrt',
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
                       'Kron-BFGS-no-norm-gate-damping',
                       'Kron-BFGS-homo-no-norm-gate-damping',
                       'Kron-BFGS-wrong',
                       'Kron-BFGS-homo',
                       'Kron-BFGS-momentum-grad',
                       'Kron-BFGS-unregularized-grad',
                       'Kron-BFGS-unregularized-grad-momentum-grad',
                       'Kron-BFGS-Hessian-action',
                       'Kron-BFGS-wrong-Hessian-action',
                       'Kron-BFGS-Hessian-action-unregularized-grad',
                       'Kron-BFGS-Hessian-action-momentum-grad',
                       'Kron-BFGS-1st-layer-only',
                       'Kron-BFGS-block',
                       'Kron-SGD',
                       'BFGS',
                       'BFGS-homo']:
        1
    else:
        print('Error: need check lambda')
        sys.exit()

    if algorithm == 'SMW-Fisher-different-minibatch':
        params['if_different_minibatch'] = True
    elif algorithm in ['RMSprop-individual-grad-no-sqrt-LM',
                       'RMSprop-individual-grad-no-sqrt-Fisher',
                       'RMSprop-individual-grad-no-sqrt-EF',
                       'RMSprop-individual-grad-no-sqrt',
                       'RMSprop-individual-grad',
                       'RMSprop-no-sqrt',
                       'RMSprop',
                       'RMSprop-warmStart',
                       'RMSprop-test',
                       'Adam',
                       'Adam-test',
                       'Adam-noWarmStart',
                       'SMW-Fisher-signVAsqrt-p',
                       'SMW-Fisher-VA-p',
                       'SMW-Fisher-sign',
                       'SMW-Fisher-momentum-p-sign',
                       'SMW-Fisher-momentum-p',
                       'SMW-Fisher',
                       'SMW-Fisher-momentum',
                       'SMW-GN',
                       'shampoo-no-sqrt-Fisher',
                       'shampoo-no-sqrt-Fisher-test',
                       'matrix-normal',
                       'matrix-normal-allVariables',
                       'matrix-normal-allVariables-warmStart',
                       'matrix-normal-allVariables-warmStart-MaxEigDamping',
                       'matrix-normal-allVariables-warmStart-noPerDimDamping',
                       'matrix-normal-LM-momentum-grad',
                       'matrix-normal-same-trace',
                       'matrix-normal-same-trace-warmStart',
                       'matrix-normal-same-trace-warmStart-noPerDimDamping',
                       'matrix-normal-same-trace-allVariables',
                       'matrix-normal-same-trace-allVariables-warmStart',
                       'matrix-normal-same-trace-allVariables-warmStart-AvgEigDamping',
                       'matrix-normal-same-trace-allVariables-warmStart-MaxEigDamping',
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
                       'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-lessInverse',
                       'matrix-normal-EF-same-trace-allVariables-filterFlattening-warmStart',
                       'matrix-normal-same-trace-allVariables-warmStart-noPerDimDamping',
                       'ekfac-EF-VA',
                       'ekfac-EF',
                       'Fisher-BD',
                       'kfac-TR',
                       'kfac-CG',
                       'kfac-momentum-grad-CG',
                       'kfac-momentum-grad-TR',
                       'kfac-momentum-grad',
                       'kfac',
                       'kfac-no-max',
                       'kfac-NoMaxNoSqrt',
                       'kfac-NoMaxNoSqrt-no-LM',
                       'kfac-no-max-momentum-grad',
                       'kfac-EF',
                       'kfac-no-max-no-LM',
                       'kfac-warmStart-no-max-no-LM',
                       'kfac-correctFisher-warmStart-no-max-no-LM',
                       'kfac-correctFisher-warmStart-NoMaxNoSqrt-no-LM',
                       'kfac-correctFisher-warmStart-lessInverse-no-max-no-LM',
                       'kfac-correctFisher-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                       'kfac-warmStart-lessInverse-no-max-no-LM',
                       'kfac-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                       'kfac-no-max-epsilon-A-G-no-LM',
                       'SGD-VA',
                       'SGD-signVAsqrt',
                       'SGD-signVA',
                       'SGD-sign',
                       'Kron-BFGS',
                       'Kron-BFGS-no-norm-gate',
                       'Kron-BFGS-no-norm-gate-momentum-s-y',
                       'Kron-BFGS-no-norm-gate-momentum-s-y-damping',
                       'Kron-BFGS-homo-no-norm-gate-momentum-s-y-damping',
                       'Kron-BFGS-no-norm-gate-Shiqian-damping',
                       'Kron-BFGS-homo-no-norm-gate-Shiqian-damping',
                       'Kron-BFGS-homo-no-norm-gate-Powell-H-damping',
                       'Kron-BFGS-homo-no-norm-gate-PowellBDamping',
                       'Kron-BFGS-homo-no-norm-gate-PowellHDampingV2',
                       'Kron-BFGS-homo-no-norm-gate-Powell-double-damping',
                       'Kron-BFGS-homo-no-norm-gate-PowellDoubleDampingV2',
                       'Kron-BFGS-homo-no-norm-gate-momentum-s-y-Powell-double-damping',
                       'Kron-BFGS-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping',
                       'Kron-BFGS-homo-no-norm-gate-HessianActionV2-momentum-s-y-Powell-double-damping',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-Powell-double-damping',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2-momentum-s-y-DDV2',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchA-HessianActionV2IdentityInitial-momentum-s-y-DDV2',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchANotDamped-HessianActionV2-momentum-s-y-DDV2',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-DDV2',
                       'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-momentum-s-y-DDV2-Sqrt',
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
                       'Kron-BFGS-no-norm-gate-damping',
                       'Kron-BFGS-homo-no-norm-gate-damping',
                       'Kron-BFGS-wrong',
                       'Kron-BFGS-homo',
                       'Kron-BFGS-LM',
                       'Kron-BFGS-LM-sqrt',
                       'Kron-BFGS-LM-unregularized-grad',
                       'Kron-BFGS-momentum-grad',
                       'Kron-BFGS-unregularized-grad',
                       'Kron-BFGS-unregularized-grad-momentum-grad',
                       'Kron-BFGS-LM-momentum-grad',
                       'Kron-BFGS-Hessian-action',
                       'Kron-BFGS-wrong-Hessian-action',
                       'Kron-BFGS-Hessian-action-unregularized-grad',
                       'Kron-BFGS-Hessian-action-momentum-grad',
                       'Kron-BFGS-1st-layer-only',
                       'Kron-BFGS-block',
                       'Kron-SGD']:
        params['if_different_minibatch'] = False
    elif algorithm in ['SGD-signVAerf',
                       'SGD-yura-BD',
                       'SGD-yura-old',
                       'SGD-yura-MA',
                       'SGD-yura',
                       'shampoo',
                       'shampoo-allVariables',
                       'shampoo-allVariables-warmStart',
                       'shampoo-allVariables-warmStart-lessInverse',
                       'shampoo-allVariables-filterFlattening-warmStart',
                       'shampoo-allVariables-filterFlattening-warmStart-lessInverse',
                       'shampoo-no-sqrt',
                       'SGD-momentum-yura',
                       'SGD-momentum',
                       'SGD',
                       'BFGS',
                       'BFGS-homo']:
        1 
    else:
        print('Error: need check different minibatch')
        sys.exit()
        
    if algorithm == 'SGD-sign' or\
    algorithm == 'SMW-Fisher-sign':
        print('error: not supported.')
        sys.exit()
    else:
        params['if_sign'] = False

    if algorithm in ['SGD-momentum-yura',
                     'SMW-Fisher-momentum-p-sign',
                     'SMW-Fisher-momentum-p']:
        
        print('error: not supported.')
        sys.exit()
        
        params['if_momentum_p'] = True
    else:
        params['if_momentum_p'] = False

    if algorithm in ['SGD-VA',
                     'SMW-Fisher-VA-p']:
        
        print('error: not supported.')
        sys.exit()
        
        params['if_VA_p'] = True
    else:
        params['if_VA_p'] = False

    if algorithm in ['SMW-Fisher-signVAsqrt-p',
                     'SGD-signVAsqrt']:
        
        print('error: not supported.')
        sys.exit()
        
        params['if_signVAsqrt'] = True
    else:
        params['if_signVAsqrt'] = False

    if algorithm in ['SGD-signVA']:
        
        print('error: not supported.')
        sys.exit()
        
        params['if_signVA'] = True
    else:
        params['if_signVA'] = False

    if algorithm in ['SGD-yura',
                     'SGD-yura-MA',
                     'SGD-momentum-yura']:
        
        print('error: not supported.')
        sys.exit()
        
        params['if_yura'] = True
    else:
        params['if_yura'] = False

    params['if_signVAerf'] = False


    params['if_Adam'] = False
    


    params['keys_params_saved'] = []

    params['keys_params_saved'].append('if_test_mode')
    params['keys_params_saved'].append('tau')
    params['keys_params_saved'].append('seed_number')
    params['keys_params_saved'].append('num_threads')
    
    params['keys_params_saved'].append('initialization_pkg')
    params['keys_params_saved'].append('N1')
    params['keys_params_saved'].append('N2')
    
    params['keys_params_saved'].append('if_max_epoch')
    params['keys_params_saved'].append('max_epoch/time')
    
    params['keys_params_saved'].append('momentum_gradient_rho')
    params['keys_params_saved'].append('momentum_gradient_dampening')
    
    params['keys_params_saved'].append('if_grafting')
    
    params['keys_params_saved'].append('weight_decay')
    
    if params['if_lr_decay']:
        params['keys_params_saved'].append('num_epoch_to_decay')
        params['keys_params_saved'].append('lr_decay_rate')

    if params['algorithm'] == 'SGD-yura-MA':
        params['keys_params_saved'].append('yura_lambda_second_term_MA_weight')
        
        

    

    
    if params['algorithm'] in ['RMSprop',
                               'RMSprop-warmStart',
                               'RMSprop-test',
                               'Adam',
                               'Adam-test',
                               'Adam-noWarmStart']:
        params['keys_params_saved'].append('RMSprop_epsilon')
        params['keys_params_saved'].append('RMSprop_beta_2')
        
    import utils_git.utils_kfac as utils_kfac
    
    if params['algorithm'] in utils_kfac.list_algorithm:
        params = utils_kfac.get_saved_params_kfac(params)
        

        
    import utils_git.utils_kbfgs as utils_kbfgs
    
    if params['algorithm'] in utils_kbfgs.list_algorithm:
        params = utils_kbfgs.get_saved_params_kbfgs(params, args)
    
    
    
        

    
#     if params['algorithm'] in ['matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart',
#                                'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-lessInverse',]:
#         params['keys_params_saved'].append('if_Hessian_action')
#     else:
#         print('params[algorithm]')
#         print(params['algorithm'])
#         sys.exit()
        
        
    if params['algorithm'] in ['shampoo',
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
                               'matrix-normal-LM-momentum-grad',
                               'matrix-normal-same-trace',
                               'matrix-normal-same-trace-warmStart',
                               'matrix-normal-same-trace-warmStart-noPerDimDamping',
                               'matrix-normal-same-trace-allVariables',
                               'matrix-normal-same-trace-allVariables-warmStart',
                               'matrix-normal-same-trace-allVariables-warmStart-AvgEigDamping',
                               'matrix-normal-same-trace-allVariables-warmStart-MaxEigDamping',
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
                               'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-lessInverse',
                               'matrix-normal-EF-same-trace-allVariables-filterFlattening-warmStart',
                               'matrix-normal-same-trace-allVariables-warmStart-noPerDimDamping']:
        
        params['keys_params_saved'].append('shampoo_inverse_freq')
        params['keys_params_saved'].append('shampoo_update_freq')
        
        params['keys_params_saved'].append('shampoo_decay')
        params['keys_params_saved'].append('shampoo_weight')
        
        params['keys_params_saved'].append('if_Hessian_action')
        
        params['keys_params_saved'].append('shampoo_if_coupled_newton')
    

    if params['algorithm'] in ['shampoo',
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
                               'matrix-normal-allVariables-warmStart-noPerDimDamping',
                               'matrix-normal-same-trace',
                               'matrix-normal-same-trace-warmStart',
                               'matrix-normal-same-trace-warmStart-noPerDimDamping',
                               'matrix-normal-same-trace-allVariables',
                               'matrix-normal-same-trace-allVariables-warmStart',
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
                               'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-lessInverse',
                               'matrix-normal-EF-same-trace-allVariables-filterFlattening-warmStart',]:
        params['keys_params_saved'].append('shampoo_epsilon')
        
        
#     print('params[algorithm]')
#     print(params['algorithm'])
    
#     sys.exit()
    
    if params['algorithm'] == 'Fisher-BD':
        params['keys_params_saved'].append('Fisher_BD_damping')
    

    if algorithm in ['kfac-momentum-grad',
                       'kfac-EF']:
        params['algorithm'] = 'kfac'
    elif algorithm == 'kfac-no-max-momentum-grad':
        params['algorithm'] = 'kfac-no-max'
    elif algorithm in ['Kron-BFGS-momentum-grad',
                       'Kron-BFGS-unregularized-grad',
                       'Kron-BFGS-unregularized-grad-momentum-grad']:
        params['algorithm'] = 'Kron-BFGS'
    elif algorithm in ['Kron-BFGS-Hessian-action-momentum-grad']:
        params['algorithm'] = 'Kron-BFGS-Hessian-action'
    elif algorithm in ['Kron-BFGS-LM-momentum-grad',
                       'Kron-BFGS-LM-unregularized-grad']:
        params['algorithm'] = 'Kron-BFGS-LM'
    


    return params