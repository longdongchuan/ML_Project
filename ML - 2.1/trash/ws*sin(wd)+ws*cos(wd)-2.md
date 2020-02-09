# 模型预测


```python
import numpy as np
import pandas as pd
from module.utils import *
from ngboost.learners import *
%config InlineBackend.figure_format='retina'
```


```python
transform='ws*sin(wd)+ws*cos(wd)'
X_train, X_test, Y_train, Y_test, Y_scaler = get_data(hour_num=2, transform=transform,
                                            drop_time=True, scale=True, return_y_scaler=True)
Pred_df = Y_test
```

    get_data(hour_num=2, transform='ws*sin(wd)+ws*cos(wd)', drop_time=True, scale=True)
    
    Input space:  Index(['ws*sin(wd)', 'ws*cos(wd)', 'ws*sin(wd)-1', 'ws*sin(wd)-2',
           'ws*cos(wd)-1', 'ws*cos(wd)-2', 'wind_power-1', 'wind_power-2'],
          dtype='object')


## default_linear_learner


```python
Y_pred = model_test(Base=default_linear_learner(alpha=0.1),
           X_train=X_train, X_test=X_test,
           Y_train=Y_train, Y_test=Y_test,
          n_estimators=500, verbose_eval=100,
          plot_predict=True, return_y_pred=True)
Y_pred.name = 'default_linear_learner'
Pred_df = pd.concat([Pred_df, Y_pred], axis=1)
pd.Series(np.zeros(len(Pred_df)), index=Pred_df.index).plot(color='k')
del Y_pred
```

    NGBRegressor(Base=Ridge(alpha=0.1, copy_X=True, fit_intercept=True,
                            max_iter=None, normalize=False, random_state=None,
                            solver='auto', tol=0.001),
                 Dist=<class 'ngboost.distns.normal.Normal'>,
                 Score=<class 'ngboost.scores.MLE'>, learning_rate=0.01,
                 minibatch_frac=1.0, n_estimators=500, natural_gradient=True,
                 tol=0.0001, verbose=True, verbose_eval=100) 
    
    [iter 0] loss=0.0535 val_loss=0.0000 scale=0.5000 norm=0.2703
    [iter 100] loss=-0.3356 val_loss=0.0000 scale=0.1250 norm=0.0578
    [iter 200] loss=-0.3587 val_loss=0.0000 scale=0.0312 norm=0.0142
    [iter 300] loss=-0.3602 val_loss=0.0000 scale=0.0078 norm=0.0035
    [iter 400] loss=-0.3602 val_loss=0.0000 scale=0.0020 norm=0.0009
    
    Test MSE 0.0459599941990834
    Test NLL -0.21670528342918716



![png](output_4_1.png)


## default_tree_learner


```python
Y_pred = model_test(Base=default_tree_learner(depth=6),
           X_train=X_train, X_test=X_test,
           Y_train=Y_train, Y_test=Y_test,
          n_estimators=1000, verbose_eval=200,
          plot_predict=True, return_y_pred=True)
Y_pred.name = 'default_tree_learner'
Pred_df = pd.concat([Pred_df, Y_pred], axis=1)
pd.Series(np.zeros(len(Pred_df)), index=Pred_df.index).plot(color='k')
del Y_pred
```

    NGBRegressor(Base=DecisionTreeRegressor(criterion='friedman_mse', max_depth=6,
                                            max_features=None, max_leaf_nodes=None,
                                            min_impurity_decrease=0.0,
                                            min_impurity_split=None,
                                            min_samples_leaf=1, min_samples_split=2,
                                            min_weight_fraction_leaf=0.0,
                                            presort=False, random_state=None,
                                            splitter='best'),
                 Dist=<class 'ngboost.distns.normal.Normal'>,
                 Score=<class 'ngboost.scores.MLE'>, learning_rate=0.01,
                 minibatch_frac=1.0, n_estimators=1000, natural_gradient=True,
                 tol=0.0001, verbose=True, verbose_eval=200) 
    
    [iter 0] loss=0.0535 val_loss=0.0000 scale=0.5000 norm=0.2703
    [iter 200] loss=-0.3005 val_loss=0.0000 scale=0.1250 norm=0.0499
    [iter 400] loss=-0.3102 val_loss=0.0000 scale=0.0156 norm=0.0061
    [iter 600] loss=-0.3107 val_loss=0.0000 scale=0.0020 norm=0.0008
    [iter 800] loss=-0.3107 val_loss=0.0000 scale=0.0010 norm=0.0004
    
    Test MSE 0.0437378222295984
    Test NLL -0.22405840697408172



![png](output_6_1.png)


## lasso_learner


```python
Y_pred = model_test(Base=lasso_learner(alpha=0.01),
           n_estimators=500, verbose_eval=100, Score=CRPS,
           X_train=X_train, X_test=X_test,
           Y_train=Y_train, Y_test=Y_test,
          plot_predict=True, return_y_pred=True)
Y_pred.name = 'lasso_learner'
Pred_df = pd.concat([Pred_df, Y_pred], axis=1)
pd.Series(np.zeros(len(Pred_df)), index=Pred_df.index).plot(color='k')
del Y_pred
```

    NGBRegressor(Base=Lasso(alpha=0.01, copy_X=True, fit_intercept=True,
                            max_iter=1000, normalize=False, positive=False,
                            precompute=False, random_state=None, selection='cyclic',
                            tol=0.0001, warm_start=False),
                 Dist=<class 'ngboost.distns.normal.Normal'>,
                 Score=<class 'ngboost.scores.CRPS'>, learning_rate=0.01,
                 minibatch_frac=1.0, n_estimators=500, natural_gradient=True,
                 tol=0.0001, verbose=True, verbose_eval=100) 
    
    [iter 0] loss=0.1430 val_loss=0.0000 scale=0.2500 norm=0.6104
    [iter 100] loss=0.0925 val_loss=0.0000 scale=0.0156 norm=0.0472
    [iter 200] loss=0.0903 val_loss=0.0000 scale=0.0039 norm=0.0121
    [iter 300] loss=0.0902 val_loss=0.0000 scale=0.0010 norm=0.0030
    [iter 400] loss=0.0902 val_loss=0.0000 scale=0.0002 norm=0.0008
    
    Test MSE 0.04592888927667521
    Test NLL -0.22022107942825958



![png](output_8_1.png)


## linear_svr_learner


```python
Y_pred = model_test(Base=linear_svr_learner(epsilon=0.0, 
                                   C=0.05, 
                                   max_iter=10000),
           X_train=X_train, X_test=X_test,
           Y_train=Y_train, Y_test=Y_test,
          n_estimators=500, verbose_eval=100,
          plot_predict=True, return_y_pred=True)
Y_pred.name = 'linear_svr_learner'
Pred_df = pd.concat([Pred_df, Y_pred], axis=1)
pd.Series(np.zeros(len(Pred_df)), index=Pred_df.index).plot(color='k')
del Y_pred
```

    NGBRegressor(Base=LinearSVR(C=0.05, dual=True, epsilon=0.0, fit_intercept=True,
                                intercept_scaling=1.0, loss='epsilon_insensitive',
                                max_iter=1000, random_state=None, tol=0.0001,
                                verbose=0),
                 Dist=<class 'ngboost.distns.normal.Normal'>,
                 Score=<class 'ngboost.scores.MLE'>, learning_rate=0.01,
                 minibatch_frac=1.0, n_estimators=500, natural_gradient=True,
                 tol=0.0001, verbose=True, verbose_eval=100) 
    
    [iter 0] loss=0.0535 val_loss=0.0000 scale=1.0000 norm=0.5406
    [iter 100] loss=-0.3002 val_loss=0.0000 scale=0.1250 norm=0.0608
    [iter 200] loss=-0.3249 val_loss=0.0000 scale=0.0312 norm=0.0147
    [iter 300] loss=-0.3264 val_loss=0.0000 scale=0.0078 norm=0.0036
    [iter 400] loss=-0.3265 val_loss=0.0000 scale=0.0020 norm=0.0009
    
    Test MSE 0.04371492730170478
    Test NLL -0.2414046672830109



![png](output_10_1.png)


## kernel_ridge_learner


```python
Y_pred = model_test(Base=kernel_ridge_learner(alpha=0.5, 
                                    kernel="poly",
                                    degree=3),
           X_train=X_train, X_test=X_test,
           Y_train=Y_train, Y_test=Y_test,
          n_estimators=500, verbose_eval=10,
          plot_predict=True, return_y_pred=True)
Y_pred.name = 'kernel_ridge_learner'
Pred_df = pd.concat([Pred_df, Y_pred], axis=1)
pd.Series(np.zeros(len(Pred_df)), index=Pred_df.index).plot(color='k')
del Y_pred
```

    NGBRegressor(Base=KernelRidge(alpha=0.5, coef0=1, degree=3, gamma=None,
                                  kernel='poly', kernel_params=None),
                 Dist=<class 'ngboost.distns.normal.Normal'>,
                 Score=<class 'ngboost.scores.MLE'>, learning_rate=0.01,
                 minibatch_frac=1.0, n_estimators=500, natural_gradient=True,
                 tol=0.0001, verbose=True, verbose_eval=10) 
    
    [iter 0] loss=0.0535 val_loss=0.0000 scale=0.5000 norm=0.2703
    [iter 10] loss=-0.0525 val_loss=0.0000 scale=0.5000 norm=0.2318
    [iter 20] loss=-0.1136 val_loss=0.0000 scale=0.5000 norm=0.2176
    [iter 30] loss=-0.1589 val_loss=0.0000 scale=0.5000 norm=0.2127
    [iter 40] loss=-0.1969 val_loss=0.0000 scale=0.5000 norm=0.2125
    [iter 50] loss=-0.2304 val_loss=0.0000 scale=0.5000 norm=0.2148
    [iter 60] loss=-0.2534 val_loss=0.0000 scale=0.2500 norm=0.1087
    [iter 70] loss=-0.2677 val_loss=0.0000 scale=0.2500 norm=0.1096
    [iter 80] loss=-0.2810 val_loss=0.0000 scale=0.2500 norm=0.1105
    [iter 90] loss=-0.2933 val_loss=0.0000 scale=0.2500 norm=0.1113
    [iter 100] loss=-0.3045 val_loss=0.0000 scale=0.2500 norm=0.1119
    [iter 110] loss=-0.3130 val_loss=0.0000 scale=0.1250 norm=0.0562
    [iter 120] loss=-0.3175 val_loss=0.0000 scale=0.1250 norm=0.0563
    [iter 130] loss=-0.3216 val_loss=0.0000 scale=0.1250 norm=0.0563
    [iter 140] loss=-0.3254 val_loss=0.0000 scale=0.1250 norm=0.0564
    [iter 150] loss=-0.3287 val_loss=0.0000 scale=0.1250 norm=0.0564
    [iter 160] loss=-0.3316 val_loss=0.0000 scale=0.1250 norm=0.0563
    [iter 170] loss=-0.3331 val_loss=0.0000 scale=0.0625 norm=0.0281
    [iter 180] loss=-0.3342 val_loss=0.0000 scale=0.0625 norm=0.0281
    [iter 190] loss=-0.3353 val_loss=0.0000 scale=0.0625 norm=0.0281
    [iter 200] loss=-0.3362 val_loss=0.0000 scale=0.0625 norm=0.0281
    [iter 210] loss=-0.3370 val_loss=0.0000 scale=0.0625 norm=0.0280
    [iter 220] loss=-0.3376 val_loss=0.0000 scale=0.0312 norm=0.0140
    [iter 230] loss=-0.3379 val_loss=0.0000 scale=0.0312 norm=0.0140
    [iter 240] loss=-0.3382 val_loss=0.0000 scale=0.0312 norm=0.0140
    [iter 250] loss=-0.3384 val_loss=0.0000 scale=0.0312 norm=0.0140
    [iter 260] loss=-0.3386 val_loss=0.0000 scale=0.0312 norm=0.0139
    [iter 270] loss=-0.3388 val_loss=0.0000 scale=0.0312 norm=0.0139
    [iter 280] loss=-0.3389 val_loss=0.0000 scale=0.0156 norm=0.0070
    [iter 290] loss=-0.3390 val_loss=0.0000 scale=0.0156 norm=0.0070
    [iter 300] loss=-0.3390 val_loss=0.0000 scale=0.0156 norm=0.0069
    [iter 310] loss=-0.3391 val_loss=0.0000 scale=0.0156 norm=0.0069
    [iter 320] loss=-0.3391 val_loss=0.0000 scale=0.0156 norm=0.0069
    [iter 330] loss=-0.3392 val_loss=0.0000 scale=0.0078 norm=0.0035
    [iter 340] loss=-0.3392 val_loss=0.0000 scale=0.0078 norm=0.0035
    [iter 350] loss=-0.3392 val_loss=0.0000 scale=0.0078 norm=0.0035
    [iter 360] loss=-0.3392 val_loss=0.0000 scale=0.0078 norm=0.0035
    [iter 370] loss=-0.3392 val_loss=0.0000 scale=0.0078 norm=0.0035
    [iter 380] loss=-0.3392 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 390] loss=-0.3392 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 400] loss=-0.3392 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 410] loss=-0.3392 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 420] loss=-0.3392 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 430] loss=-0.3392 val_loss=0.0000 scale=0.0020 norm=0.0009
    [iter 440] loss=-0.3392 val_loss=0.0000 scale=0.0020 norm=0.0009
    [iter 450] loss=-0.3392 val_loss=0.0000 scale=0.0020 norm=0.0009
    [iter 460] loss=-0.3392 val_loss=0.0000 scale=0.0020 norm=0.0009
    [iter 470] loss=-0.3392 val_loss=0.0000 scale=0.0020 norm=0.0009
    [iter 480] loss=-0.3392 val_loss=0.0000 scale=0.0010 norm=0.0004
    [iter 490] loss=-0.3392 val_loss=0.0000 scale=0.0010 norm=0.0004
    
    Test MSE 0.04155964756568224
    Test NLL -0.2588380344480677



![png](output_12_1.png)


## esn_ridge_learner


```python
Y_pred = model_test(Base=esn_ridge_learner(n_readout=1000,
                                  n_components=100,
                                  alpha=0.01),
           X_train=X_train, X_test=X_test,
           Y_train=Y_train, Y_test=Y_test,
          n_estimators=500, verbose_eval=10,
          plot_predict=True, return_y_pred=True)
Y_pred.name = 'esn_ridge_learner'
Pred_df = pd.concat([Pred_df, Y_pred], axis=1)
pd.Series(np.zeros(len(Pred_df)), index=Pred_df.index).plot(color='k')
del Y_pred
```

    NGBRegressor(Base=<ngboost.esn_learners.ESN_Ridge_learner object at 0x1a17fb2780>,
                 Dist=<class 'ngboost.distns.normal.Normal'>,
                 Score=<class 'ngboost.scores.MLE'>, learning_rate=0.01,
                 minibatch_frac=1.0, n_estimators=500, natural_gradient=True,
                 tol=0.0001, verbose=True, verbose_eval=10) 
    
    [iter 0] loss=0.0535 val_loss=0.0000 scale=0.5000 norm=0.2703
    [iter 10] loss=-0.0528 val_loss=0.0000 scale=0.5000 norm=0.2305
    [iter 20] loss=-0.1131 val_loss=0.0000 scale=0.5000 norm=0.2156
    [iter 30] loss=-0.1570 val_loss=0.0000 scale=0.5000 norm=0.2098
    [iter 40] loss=-0.1930 val_loss=0.0000 scale=0.5000 norm=0.2084
    [iter 50] loss=-0.2243 val_loss=0.0000 scale=0.2500 norm=0.1048
    [iter 60] loss=-0.2400 val_loss=0.0000 scale=0.2500 norm=0.1054
    [iter 70] loss=-0.2534 val_loss=0.0000 scale=0.2500 norm=0.1060
    [iter 80] loss=-0.2659 val_loss=0.0000 scale=0.2500 norm=0.1067
    [iter 90] loss=-0.2777 val_loss=0.0000 scale=0.2500 norm=0.1074
    [iter 100] loss=-0.2885 val_loss=0.0000 scale=0.2500 norm=0.1081
    [iter 110] loss=-0.2983 val_loss=0.0000 scale=0.2500 norm=0.1086
    [iter 120] loss=-0.3041 val_loss=0.0000 scale=0.1250 norm=0.0545
    [iter 130] loss=-0.3082 val_loss=0.0000 scale=0.1250 norm=0.0546
    [iter 140] loss=-0.3120 val_loss=0.0000 scale=0.1250 norm=0.0546
    [iter 150] loss=-0.3155 val_loss=0.0000 scale=0.1250 norm=0.0547
    [iter 160] loss=-0.3186 val_loss=0.0000 scale=0.1250 norm=0.0547
    [iter 170] loss=-0.3214 val_loss=0.0000 scale=0.1250 norm=0.0547
    [iter 180] loss=-0.3235 val_loss=0.0000 scale=0.0625 norm=0.0273
    [iter 190] loss=-0.3246 val_loss=0.0000 scale=0.0625 norm=0.0273
    [iter 200] loss=-0.3256 val_loss=0.0000 scale=0.0625 norm=0.0273
    [iter 210] loss=-0.3265 val_loss=0.0000 scale=0.0625 norm=0.0273
    [iter 220] loss=-0.3273 val_loss=0.0000 scale=0.0625 norm=0.0273
    [iter 230] loss=-0.3281 val_loss=0.0000 scale=0.0625 norm=0.0272
    [iter 240] loss=-0.3286 val_loss=0.0000 scale=0.0312 norm=0.0136
    [iter 250] loss=-0.3289 val_loss=0.0000 scale=0.0312 norm=0.0136
    [iter 260] loss=-0.3291 val_loss=0.0000 scale=0.0312 norm=0.0136
    [iter 270] loss=-0.3294 val_loss=0.0000 scale=0.0312 norm=0.0136
    [iter 280] loss=-0.3296 val_loss=0.0000 scale=0.0312 norm=0.0136
    [iter 290] loss=-0.3297 val_loss=0.0000 scale=0.0312 norm=0.0136
    [iter 300] loss=-0.3299 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 310] loss=-0.3299 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 320] loss=-0.3300 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 330] loss=-0.3300 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 340] loss=-0.3301 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 350] loss=-0.3301 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 360] loss=-0.3302 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 370] loss=-0.3302 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 380] loss=-0.3302 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 390] loss=-0.3302 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 400] loss=-0.3302 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 410] loss=-0.3302 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 420] loss=-0.3302 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 430] loss=-0.3302 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 440] loss=-0.3302 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 450] loss=-0.3303 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 460] loss=-0.3303 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 470] loss=-0.3303 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 480] loss=-0.3303 val_loss=0.0000 scale=0.0005 norm=0.0002
    [iter 490] loss=-0.3303 val_loss=0.0000 scale=0.0010 norm=0.0004
    
    Test MSE 0.042610436874713005
    Test NLL -0.24086176100202053



![png](output_14_1.png)


## esn_kernel_ridge_learner


```python
Y_pred = model_test(Base=esn_kernel_ridge_learner(n_readout=1000,
                                         n_components=100,
                                         alpha=1, 
                                         kernel='poly',
                                         degree=3),
           X_train=X_train, X_test=X_test,
           Y_train=Y_train, Y_test=Y_test,
          n_estimators=500, verbose_eval=5,
          plot_predict=True, return_y_pred=True)
Y_pred.name = 'esn_kernel_ridge_learner'
Pred_df = pd.concat([Pred_df, Y_pred], axis=1)
pd.Series(np.zeros(len(Pred_df)), index=Pred_df.index).plot(color='k')
del Y_pred
```

    NGBRegressor(Base=<ngboost.esn_learners.ESN_kernel_ridge_learner object at 0x1a1981e828>,
                 Dist=<class 'ngboost.distns.normal.Normal'>,
                 Score=<class 'ngboost.scores.MLE'>, learning_rate=0.01,
                 minibatch_frac=1.0, n_estimators=500, natural_gradient=True,
                 tol=0.0001, verbose=True, verbose_eval=5) 
    
    [iter 0] loss=0.0535 val_loss=0.0000 scale=1.0000 norm=0.5406
    [iter 5] loss=-0.0337 val_loss=0.0000 scale=1.0000 norm=0.4826
    [iter 10] loss=-0.0785 val_loss=0.0000 scale=0.5000 norm=0.2302
    [iter 15] loss=-0.1089 val_loss=0.0000 scale=0.5000 norm=0.2246
    [iter 20] loss=-0.1355 val_loss=0.0000 scale=0.5000 norm=0.2212
    [iter 25] loss=-0.1592 val_loss=0.0000 scale=0.5000 norm=0.2195
    [iter 30] loss=-0.1808 val_loss=0.0000 scale=0.5000 norm=0.2188
    [iter 35] loss=-0.2009 val_loss=0.0000 scale=0.5000 norm=0.2189
    [iter 40] loss=-0.2196 val_loss=0.0000 scale=0.5000 norm=0.2196
    [iter 45] loss=-0.2371 val_loss=0.0000 scale=0.5000 norm=0.2207
    [iter 50] loss=-0.2503 val_loss=0.0000 scale=0.2500 norm=0.1108
    [iter 55] loss=-0.2581 val_loss=0.0000 scale=0.2500 norm=0.1112
    [iter 60] loss=-0.2657 val_loss=0.0000 scale=0.2500 norm=0.1115
    [iter 65] loss=-0.2729 val_loss=0.0000 scale=0.2500 norm=0.1119
    [iter 70] loss=-0.2799 val_loss=0.0000 scale=0.2500 norm=0.1122
    [iter 75] loss=-0.2865 val_loss=0.0000 scale=0.2500 norm=0.1126
    [iter 80] loss=-0.2927 val_loss=0.0000 scale=0.2500 norm=0.1129
    [iter 85] loss=-0.2987 val_loss=0.0000 scale=0.2500 norm=0.1132
    [iter 90] loss=-0.3043 val_loss=0.0000 scale=0.2500 norm=0.1134
    [iter 95] loss=-0.3095 val_loss=0.0000 scale=0.2500 norm=0.1136
    [iter 100] loss=-0.3134 val_loss=0.0000 scale=0.1250 norm=0.0569
    [iter 105] loss=-0.3157 val_loss=0.0000 scale=0.1250 norm=0.0569
    [iter 110] loss=-0.3179 val_loss=0.0000 scale=0.1250 norm=0.0570
    [iter 115] loss=-0.3200 val_loss=0.0000 scale=0.1250 norm=0.0570
    [iter 120] loss=-0.3220 val_loss=0.0000 scale=0.1250 norm=0.0570
    [iter 125] loss=-0.3240 val_loss=0.0000 scale=0.1250 norm=0.0570
    [iter 130] loss=-0.3258 val_loss=0.0000 scale=0.1250 norm=0.0570
    [iter 135] loss=-0.3275 val_loss=0.0000 scale=0.1250 norm=0.0570
    [iter 140] loss=-0.3290 val_loss=0.0000 scale=0.1250 norm=0.0570
    [iter 145] loss=-0.3305 val_loss=0.0000 scale=0.1250 norm=0.0570
    [iter 150] loss=-0.3318 val_loss=0.0000 scale=0.1250 norm=0.0569
    [iter 155] loss=-0.3327 val_loss=0.0000 scale=0.0625 norm=0.0285
    [iter 160] loss=-0.3333 val_loss=0.0000 scale=0.0625 norm=0.0284
    [iter 165] loss=-0.3338 val_loss=0.0000 scale=0.0625 norm=0.0284
    [iter 170] loss=-0.3344 val_loss=0.0000 scale=0.0625 norm=0.0284
    [iter 175] loss=-0.3349 val_loss=0.0000 scale=0.0625 norm=0.0284
    [iter 180] loss=-0.3353 val_loss=0.0000 scale=0.0625 norm=0.0284
    [iter 185] loss=-0.3358 val_loss=0.0000 scale=0.0625 norm=0.0284
    [iter 190] loss=-0.3362 val_loss=0.0000 scale=0.0625 norm=0.0284
    [iter 195] loss=-0.3366 val_loss=0.0000 scale=0.0625 norm=0.0283
    [iter 200] loss=-0.3369 val_loss=0.0000 scale=0.0625 norm=0.0283
    [iter 205] loss=-0.3372 val_loss=0.0000 scale=0.0312 norm=0.0142
    [iter 210] loss=-0.3374 val_loss=0.0000 scale=0.0312 norm=0.0141
    [iter 215] loss=-0.3375 val_loss=0.0000 scale=0.0312 norm=0.0141
    [iter 220] loss=-0.3377 val_loss=0.0000 scale=0.0312 norm=0.0141
    [iter 225] loss=-0.3378 val_loss=0.0000 scale=0.0312 norm=0.0141
    [iter 230] loss=-0.3379 val_loss=0.0000 scale=0.0312 norm=0.0141
    [iter 235] loss=-0.3380 val_loss=0.0000 scale=0.0312 norm=0.0141
    [iter 240] loss=-0.3381 val_loss=0.0000 scale=0.0312 norm=0.0141
    [iter 245] loss=-0.3382 val_loss=0.0000 scale=0.0312 norm=0.0141
    [iter 250] loss=-0.3383 val_loss=0.0000 scale=0.0312 norm=0.0141
    [iter 255] loss=-0.3384 val_loss=0.0000 scale=0.0312 norm=0.0141
    [iter 260] loss=-0.3385 val_loss=0.0000 scale=0.0156 norm=0.0070
    [iter 265] loss=-0.3385 val_loss=0.0000 scale=0.0156 norm=0.0070
    [iter 270] loss=-0.3386 val_loss=0.0000 scale=0.0156 norm=0.0070
    [iter 275] loss=-0.3386 val_loss=0.0000 scale=0.0156 norm=0.0070
    [iter 280] loss=-0.3386 val_loss=0.0000 scale=0.0156 norm=0.0070
    [iter 285] loss=-0.3387 val_loss=0.0000 scale=0.0156 norm=0.0070
    [iter 290] loss=-0.3387 val_loss=0.0000 scale=0.0156 norm=0.0070
    [iter 295] loss=-0.3387 val_loss=0.0000 scale=0.0156 norm=0.0070
    [iter 300] loss=-0.3387 val_loss=0.0000 scale=0.0156 norm=0.0070
    [iter 305] loss=-0.3388 val_loss=0.0000 scale=0.0156 norm=0.0070
    [iter 310] loss=-0.3388 val_loss=0.0000 scale=0.0078 norm=0.0035
    [iter 315] loss=-0.3388 val_loss=0.0000 scale=0.0156 norm=0.0070
    [iter 320] loss=-0.3388 val_loss=0.0000 scale=0.0078 norm=0.0035
    [iter 325] loss=-0.3388 val_loss=0.0000 scale=0.0078 norm=0.0035
    [iter 330] loss=-0.3388 val_loss=0.0000 scale=0.0078 norm=0.0035
    [iter 335] loss=-0.3388 val_loss=0.0000 scale=0.0078 norm=0.0035
    [iter 340] loss=-0.3388 val_loss=0.0000 scale=0.0078 norm=0.0035
    [iter 345] loss=-0.3388 val_loss=0.0000 scale=0.0078 norm=0.0035
    [iter 350] loss=-0.3388 val_loss=0.0000 scale=0.0078 norm=0.0035
    [iter 355] loss=-0.3388 val_loss=0.0000 scale=0.0020 norm=0.0009
    [iter 360] loss=-0.3389 val_loss=0.0000 scale=0.0039 norm=0.0018
    [iter 365] loss=-0.3389 val_loss=0.0000 scale=0.0020 norm=0.0009
    [iter 370] loss=-0.3389 val_loss=0.0000 scale=0.0078 norm=0.0035
    [iter 375] loss=-0.3389 val_loss=0.0000 scale=0.0078 norm=0.0035
    [iter 380] loss=-0.3389 val_loss=0.0000 scale=0.0039 norm=0.0018
    [iter 385] loss=-0.3389 val_loss=0.0000 scale=0.0039 norm=0.0018
    [iter 390] loss=-0.3389 val_loss=0.0000 scale=0.0078 norm=0.0035
    [iter 395] loss=-0.3389 val_loss=0.0000 scale=0.0078 norm=0.0035
    [iter 400] loss=-0.3389 val_loss=0.0000 scale=0.0020 norm=0.0009
    [iter 405] loss=-0.3389 val_loss=0.0000 scale=0.0039 norm=0.0018
    [iter 410] loss=-0.3389 val_loss=0.0000 scale=0.0078 norm=0.0035
    [iter 415] loss=-0.3389 val_loss=0.0000 scale=0.0001 norm=0.0001
    [iter 420] loss=-0.3389 val_loss=0.0000 scale=0.0039 norm=0.0018
    [iter 425] loss=-0.3389 val_loss=0.0000 scale=0.0039 norm=0.0018
    [iter 430] loss=-0.3389 val_loss=0.0000 scale=0.0001 norm=0.0001
    [iter 435] loss=-0.3389 val_loss=0.0000 scale=0.0001 norm=0.0001
    [iter 440] loss=-0.3389 val_loss=0.0000 scale=0.0010 norm=0.0004
    [iter 445] loss=-0.3389 val_loss=0.0000 scale=0.0020 norm=0.0009
    [iter 450] loss=-0.3389 val_loss=0.0000 scale=0.0010 norm=0.0004
    [iter 455] loss=-0.3389 val_loss=0.0000 scale=0.0001 norm=0.0001
    [iter 460] loss=-0.3389 val_loss=0.0000 scale=0.0020 norm=0.0009
    [iter 465] loss=-0.3389 val_loss=0.0000 scale=0.0020 norm=0.0009
    [iter 470] loss=-0.3389 val_loss=0.0000 scale=0.0001 norm=0.0001
    [iter 475] loss=-0.3389 val_loss=0.0000 scale=0.0001 norm=0.0001
    [iter 480] loss=-0.3389 val_loss=0.0000 scale=0.0020 norm=0.0009
    [iter 485] loss=-0.3389 val_loss=0.0000 scale=0.0001 norm=0.0001
    [iter 490] loss=-0.3389 val_loss=0.0000 scale=0.0001 norm=0.0001
    [iter 495] loss=-0.3389 val_loss=0.0000 scale=0.0039 norm=0.0018
    
    Test MSE 0.04445203788173143
    Test NLL -0.23067758984893777



![png](output_16_1.png)


## esn_linear_svr_learner


```python
Y_pred = model_test(Base=esn_linear_svr_learner(n_readout=1000,
                                         n_components=100,
                                         epsilon=0.0,
                                         C=0.02,
                                         max_iter=1000),
           X_train=X_train, X_test=X_test,
           Y_train=Y_train, Y_test=Y_test,
          n_estimators=500, verbose_eval=10,
          plot_predict=True, return_y_pred=True)
Y_pred.name = 'esn_linear_svr_learner'
Pred_df = pd.concat([Pred_df, Y_pred], axis=1)
pd.Series(np.zeros(len(Pred_df)), index=Pred_df.index).plot(color='k')
del Y_pred
```

    NGBRegressor(Base=<ngboost.esn_learners.ESN_linear_svr_learner object at 0x102503f98>,
                 Dist=<class 'ngboost.distns.normal.Normal'>,
                 Score=<class 'ngboost.scores.MLE'>, learning_rate=0.01,
                 minibatch_frac=1.0, n_estimators=500, natural_gradient=True,
                 tol=0.0001, verbose=True, verbose_eval=10) 
    


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 0] loss=0.0535 val_loss=0.0000 scale=1.0000 norm=0.5406


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 10] loss=-0.0563 val_loss=0.0000 scale=0.5000 norm=0.2447


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 20] loss=-0.1060 val_loss=0.0000 scale=0.5000 norm=0.2368


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 30] loss=-0.1483 val_loss=0.0000 scale=0.5000 norm=0.2334


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 40] loss=-0.1855 val_loss=0.0000 scale=0.5000 norm=0.2331


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 50] loss=-0.2138 val_loss=0.0000 scale=0.2500 norm=0.1168


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 60] loss=-0.2290 val_loss=0.0000 scale=0.2500 norm=0.1171


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 70] loss=-0.2428 val_loss=0.0000 scale=0.2500 norm=0.1175


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 80] loss=-0.2553 val_loss=0.0000 scale=0.2500 norm=0.1179


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 90] loss=-0.2664 val_loss=0.0000 scale=0.2500 norm=0.1182


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 100] loss=-0.2728 val_loss=0.0000 scale=0.1250 norm=0.0592


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 110] loss=-0.2773 val_loss=0.0000 scale=0.1250 norm=0.0593


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 120] loss=-0.2813 val_loss=0.0000 scale=0.1250 norm=0.0593


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 130] loss=-0.2849 val_loss=0.0000 scale=0.1250 norm=0.0593


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 140] loss=-0.2880 val_loss=0.0000 scale=0.1250 norm=0.0593


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 150] loss=-0.2905 val_loss=0.0000 scale=0.0625 norm=0.0296


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 160] loss=-0.2917 val_loss=0.0000 scale=0.0625 norm=0.0296


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 170] loss=-0.2928 val_loss=0.0000 scale=0.0625 norm=0.0296


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 180] loss=-0.2938 val_loss=0.0000 scale=0.0625 norm=0.0296


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 190] loss=-0.2946 val_loss=0.0000 scale=0.0625 norm=0.0296


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 200] loss=-0.2952 val_loss=0.0000 scale=0.0625 norm=0.0295


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 210] loss=-0.2957 val_loss=0.0000 scale=0.0312 norm=0.0148


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 220] loss=-0.2960 val_loss=0.0000 scale=0.0312 norm=0.0148


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 230] loss=-0.2962 val_loss=0.0000 scale=0.0312 norm=0.0147


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 240] loss=-0.2965 val_loss=0.0000 scale=0.0312 norm=0.0147


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 250] loss=-0.2967 val_loss=0.0000 scale=0.0156 norm=0.0074


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 260] loss=-0.2968 val_loss=0.0000 scale=0.0312 norm=0.0147


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 270] loss=-0.2969 val_loss=0.0000 scale=0.0156 norm=0.0074


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 280] loss=-0.2970 val_loss=0.0000 scale=0.0156 norm=0.0074


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 290] loss=-0.2971 val_loss=0.0000 scale=0.0078 norm=0.0037


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 300] loss=-0.2971 val_loss=0.0000 scale=0.0078 norm=0.0037


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 310] loss=-0.2972 val_loss=0.0000 scale=0.0078 norm=0.0037


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 320] loss=-0.2972 val_loss=0.0000 scale=0.0156 norm=0.0073


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 330] loss=-0.2972 val_loss=0.0000 scale=0.0078 norm=0.0037


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 340] loss=-0.2972 val_loss=0.0000 scale=0.0078 norm=0.0037


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 350] loss=-0.2972 val_loss=0.0000 scale=0.0078 norm=0.0037


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 360] loss=-0.2973 val_loss=0.0000 scale=0.0078 norm=0.0037


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 370] loss=-0.2973 val_loss=0.0000 scale=0.0156 norm=0.0073


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 380] loss=-0.2973 val_loss=0.0000 scale=0.0020 norm=0.0009


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 390] loss=-0.2973 val_loss=0.0000 scale=0.0078 norm=0.0037


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 400] loss=-0.2973 val_loss=0.0000 scale=0.0039 norm=0.0018


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 410] loss=-0.2973 val_loss=0.0000 scale=0.0078 norm=0.0037


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 420] loss=-0.2973 val_loss=0.0000 scale=0.0010 norm=0.0005


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 430] loss=-0.2973 val_loss=0.0000 scale=0.0001 norm=0.0001


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 440] loss=-0.2973 val_loss=0.0000 scale=0.0001 norm=0.0001


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 450] loss=-0.2973 val_loss=0.0000 scale=0.0039 norm=0.0018


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 460] loss=-0.2973 val_loss=0.0000 scale=0.0020 norm=0.0009


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 470] loss=-0.2973 val_loss=0.0000 scale=0.0039 norm=0.0018


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 480] loss=-0.2973 val_loss=0.0000 scale=0.0002 norm=0.0001


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 490] loss=-0.2973 val_loss=0.0000 scale=0.0001 norm=0.0001


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    
    Test MSE 0.04635982853530269
    Test NLL -0.2191469547525955



![png](output_18_103.png)



```python
filename = 'ws*sin(wd)+ws*cos(wd)-2.csv'
Pred_df.to_csv('/Users/apple/Documents/ML_Project/ML - 2.1/result/csv/'+filename)
Pred_df = pd.DataFrame(Y_scaler.inverse_transform(Pred_df), columns=Pred_df.columns)
Pred_df.to_csv('/Users/apple/Documents/ML_Project/ML - 2.1/result/inverse/'+filename)
```


```python

```
