# 模型预测


```python
import numpy as np
import pandas as pd
from module.utils import *
from ngboost.learners import *
%config InlineBackend.figure_format='retina'
```


```python
transform='sin+cos'
X_train, X_test, Y_train, Y_test, Y_scaler = get_data(hour_num=2, transform=transform,
                                            drop_time=True, scale=True, return_y_scaler=True)
Pred_df = Y_test
```

    get_data(hour_num=2, transform='sin+cos', drop_time=True, scale=True)
    
    Input space:  Index(['wind_speed', 'sin(wd)', 'cos(wd)', 'wind_speed-1', 'wind_speed-2',
           'sin(wd)-1', 'sin(wd)-2', 'cos(wd)-1', 'cos(wd)-2', 'wind_power-1',
           'wind_power-2'],
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
    [iter 100] loss=-0.3445 val_loss=0.0000 scale=0.1250 norm=0.0573
    [iter 200] loss=-0.3701 val_loss=0.0000 scale=0.0312 norm=0.0140
    [iter 300] loss=-0.3717 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 400] loss=-0.3718 val_loss=0.0000 scale=0.0020 norm=0.0009
    
    Test MSE 0.036976530642298444
    Test NLL -0.29826037311224973



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
    [iter 200] loss=-0.3029 val_loss=0.0000 scale=0.1250 norm=0.0482
    [iter 400] loss=-0.3134 val_loss=0.0000 scale=0.0156 norm=0.0059
    [iter 600] loss=-0.3138 val_loss=0.0000 scale=0.0039 norm=0.0015
    [iter 800] loss=-0.3138 val_loss=0.0000 scale=0.0010 norm=0.0004
    
    Test MSE 0.04508044404964825
    Test NLL -0.2013187187281285



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
    [iter 100] loss=0.0920 val_loss=0.0000 scale=0.0156 norm=0.0471
    [iter 200] loss=0.0896 val_loss=0.0000 scale=0.0039 norm=0.0121
    [iter 300] loss=0.0895 val_loss=0.0000 scale=0.0010 norm=0.0030
    [iter 400] loss=0.0895 val_loss=0.0000 scale=0.0002 norm=0.0008
    
    Test MSE 0.03147666587008985
    Test NLL -0.3346283205221507



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
    [iter 100] loss=-0.2521 val_loss=0.0000 scale=0.2500 norm=0.1250
    [iter 200] loss=-0.2980 val_loss=0.0000 scale=0.0625 norm=0.0299
    [iter 300] loss=-0.3038 val_loss=0.0000 scale=0.0156 norm=0.0072
    [iter 400] loss=-0.3042 val_loss=0.0000 scale=0.0078 norm=0.0036
    
    Test MSE 0.023650557676016346
    Test NLL -0.383163003381697



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
    [iter 10] loss=-0.0552 val_loss=0.0000 scale=0.5000 norm=0.2311
    [iter 20] loss=-0.1180 val_loss=0.0000 scale=0.5000 norm=0.2162
    [iter 30] loss=-0.1640 val_loss=0.0000 scale=0.5000 norm=0.2106
    [iter 40] loss=-0.2021 val_loss=0.0000 scale=0.5000 norm=0.2094
    [iter 50] loss=-0.2352 val_loss=0.0000 scale=0.5000 norm=0.2106
    [iter 60] loss=-0.2533 val_loss=0.0000 scale=0.2500 norm=0.1060
    [iter 70] loss=-0.2674 val_loss=0.0000 scale=0.2500 norm=0.1066
    [iter 80] loss=-0.2805 val_loss=0.0000 scale=0.2500 norm=0.1072
    [iter 90] loss=-0.2927 val_loss=0.0000 scale=0.2500 norm=0.1077
    [iter 100] loss=-0.3038 val_loss=0.0000 scale=0.2500 norm=0.1082
    [iter 110] loss=-0.3127 val_loss=0.0000 scale=0.1250 norm=0.0542
    [iter 120] loss=-0.3173 val_loss=0.0000 scale=0.1250 norm=0.0543
    [iter 130] loss=-0.3214 val_loss=0.0000 scale=0.1250 norm=0.0543
    [iter 140] loss=-0.3253 val_loss=0.0000 scale=0.1250 norm=0.0543
    [iter 150] loss=-0.3287 val_loss=0.0000 scale=0.1250 norm=0.0543
    [iter 160] loss=-0.3318 val_loss=0.0000 scale=0.1250 norm=0.0542
    [iter 170] loss=-0.3342 val_loss=0.0000 scale=0.0625 norm=0.0271
    [iter 180] loss=-0.3354 val_loss=0.0000 scale=0.0625 norm=0.0270
    [iter 190] loss=-0.3365 val_loss=0.0000 scale=0.0625 norm=0.0270
    [iter 200] loss=-0.3375 val_loss=0.0000 scale=0.0625 norm=0.0269
    [iter 210] loss=-0.3384 val_loss=0.0000 scale=0.0625 norm=0.0269
    [iter 220] loss=-0.3392 val_loss=0.0000 scale=0.0625 norm=0.0268
    [iter 230] loss=-0.3398 val_loss=0.0000 scale=0.0312 norm=0.0134
    [iter 240] loss=-0.3401 val_loss=0.0000 scale=0.0312 norm=0.0134
    [iter 250] loss=-0.3404 val_loss=0.0000 scale=0.0312 norm=0.0134
    [iter 260] loss=-0.3406 val_loss=0.0000 scale=0.0312 norm=0.0134
    [iter 270] loss=-0.3408 val_loss=0.0000 scale=0.0312 norm=0.0133
    [iter 280] loss=-0.3410 val_loss=0.0000 scale=0.0312 norm=0.0133
    [iter 290] loss=-0.3412 val_loss=0.0000 scale=0.0156 norm=0.0067
    [iter 300] loss=-0.3412 val_loss=0.0000 scale=0.0156 norm=0.0066
    [iter 310] loss=-0.3413 val_loss=0.0000 scale=0.0156 norm=0.0066
    [iter 320] loss=-0.3414 val_loss=0.0000 scale=0.0156 norm=0.0066
    [iter 330] loss=-0.3414 val_loss=0.0000 scale=0.0156 norm=0.0066
    [iter 340] loss=-0.3415 val_loss=0.0000 scale=0.0156 norm=0.0066
    [iter 350] loss=-0.3415 val_loss=0.0000 scale=0.0078 norm=0.0033
    [iter 360] loss=-0.3415 val_loss=0.0000 scale=0.0078 norm=0.0033
    [iter 370] loss=-0.3415 val_loss=0.0000 scale=0.0078 norm=0.0033
    [iter 380] loss=-0.3415 val_loss=0.0000 scale=0.0078 norm=0.0033
    [iter 390] loss=-0.3415 val_loss=0.0000 scale=0.0078 norm=0.0033
    [iter 400] loss=-0.3415 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 410] loss=-0.3415 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 420] loss=-0.3416 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 430] loss=-0.3416 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 440] loss=-0.3416 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 450] loss=-0.3416 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 460] loss=-0.3416 val_loss=0.0000 scale=0.0020 norm=0.0008
    [iter 470] loss=-0.3416 val_loss=0.0000 scale=0.0020 norm=0.0008
    [iter 480] loss=-0.3416 val_loss=0.0000 scale=0.0020 norm=0.0008
    [iter 490] loss=-0.3416 val_loss=0.0000 scale=0.0020 norm=0.0008
    
    Test MSE 0.038267685057435226
    Test NLL -0.28408286453214643



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

    NGBRegressor(Base=<ngboost.esn_learners.ESN_Ridge_learner object at 0x1a1f7df2e8>,
                 Dist=<class 'ngboost.distns.normal.Normal'>,
                 Score=<class 'ngboost.scores.MLE'>, learning_rate=0.01,
                 minibatch_frac=1.0, n_estimators=500, natural_gradient=True,
                 tol=0.0001, verbose=True, verbose_eval=10) 
    
    [iter 0] loss=0.0535 val_loss=0.0000 scale=0.5000 norm=0.2703
    [iter 10] loss=-0.0555 val_loss=0.0000 scale=0.5000 norm=0.2313
    [iter 20] loss=-0.1191 val_loss=0.0000 scale=0.5000 norm=0.2161
    [iter 30] loss=-0.1656 val_loss=0.0000 scale=0.5000 norm=0.2100
    [iter 40] loss=-0.2035 val_loss=0.0000 scale=0.5000 norm=0.2081
    [iter 50] loss=-0.2299 val_loss=0.0000 scale=0.2500 norm=0.1042
    [iter 60] loss=-0.2450 val_loss=0.0000 scale=0.2500 norm=0.1044
    [iter 70] loss=-0.2590 val_loss=0.0000 scale=0.2500 norm=0.1048
    [iter 80] loss=-0.2720 val_loss=0.0000 scale=0.2500 norm=0.1051
    [iter 90] loss=-0.2840 val_loss=0.0000 scale=0.2500 norm=0.1054
    [iter 100] loss=-0.2948 val_loss=0.0000 scale=0.2500 norm=0.1057
    [iter 110] loss=-0.3031 val_loss=0.0000 scale=0.1250 norm=0.0529
    [iter 120] loss=-0.3076 val_loss=0.0000 scale=0.1250 norm=0.0530
    [iter 130] loss=-0.3118 val_loss=0.0000 scale=0.1250 norm=0.0530
    [iter 140] loss=-0.3156 val_loss=0.0000 scale=0.1250 norm=0.0530
    [iter 150] loss=-0.3191 val_loss=0.0000 scale=0.1250 norm=0.0529
    [iter 160] loss=-0.3223 val_loss=0.0000 scale=0.1250 norm=0.0529
    [iter 170] loss=-0.3251 val_loss=0.0000 scale=0.1250 norm=0.0528
    [iter 180] loss=-0.3270 val_loss=0.0000 scale=0.0625 norm=0.0263
    [iter 190] loss=-0.3282 val_loss=0.0000 scale=0.0625 norm=0.0263
    [iter 200] loss=-0.3292 val_loss=0.0000 scale=0.0625 norm=0.0263
    [iter 210] loss=-0.3301 val_loss=0.0000 scale=0.0625 norm=0.0262
    [iter 220] loss=-0.3310 val_loss=0.0000 scale=0.0625 norm=0.0262
    [iter 230] loss=-0.3317 val_loss=0.0000 scale=0.0625 norm=0.0261
    [iter 240] loss=-0.3323 val_loss=0.0000 scale=0.0312 norm=0.0130
    [iter 250] loss=-0.3326 val_loss=0.0000 scale=0.0312 norm=0.0130
    [iter 260] loss=-0.3328 val_loss=0.0000 scale=0.0312 norm=0.0130
    [iter 270] loss=-0.3331 val_loss=0.0000 scale=0.0312 norm=0.0130
    [iter 280] loss=-0.3333 val_loss=0.0000 scale=0.0312 norm=0.0130
    [iter 290] loss=-0.3335 val_loss=0.0000 scale=0.0312 norm=0.0130
    [iter 300] loss=-0.3337 val_loss=0.0000 scale=0.0312 norm=0.0130
    [iter 310] loss=-0.3338 val_loss=0.0000 scale=0.0156 norm=0.0065
    [iter 320] loss=-0.3339 val_loss=0.0000 scale=0.0156 norm=0.0065
    [iter 330] loss=-0.3339 val_loss=0.0000 scale=0.0156 norm=0.0065
    [iter 340] loss=-0.3340 val_loss=0.0000 scale=0.0156 norm=0.0065
    [iter 350] loss=-0.3340 val_loss=0.0000 scale=0.0156 norm=0.0065
    [iter 360] loss=-0.3341 val_loss=0.0000 scale=0.0156 norm=0.0065
    [iter 370] loss=-0.3341 val_loss=0.0000 scale=0.0078 norm=0.0032
    [iter 380] loss=-0.3341 val_loss=0.0000 scale=0.0078 norm=0.0032
    [iter 390] loss=-0.3341 val_loss=0.0000 scale=0.0078 norm=0.0032
    [iter 400] loss=-0.3341 val_loss=0.0000 scale=0.0078 norm=0.0032
    [iter 410] loss=-0.3341 val_loss=0.0000 scale=0.0078 norm=0.0032
    [iter 420] loss=-0.3342 val_loss=0.0000 scale=0.0078 norm=0.0032
    [iter 430] loss=-0.3342 val_loss=0.0000 scale=0.0039 norm=0.0016
    [iter 440] loss=-0.3342 val_loss=0.0000 scale=0.0039 norm=0.0016
    [iter 450] loss=-0.3342 val_loss=0.0000 scale=0.0039 norm=0.0016
    [iter 460] loss=-0.3342 val_loss=0.0000 scale=0.0039 norm=0.0016
    [iter 470] loss=-0.3342 val_loss=0.0000 scale=0.0039 norm=0.0016
    [iter 480] loss=-0.3342 val_loss=0.0000 scale=0.0001 norm=0.0001
    [iter 490] loss=-0.3342 val_loss=0.0000 scale=0.0005 norm=0.0002
    
    Test MSE 0.03746203552280819
    Test NLL -0.2868544199950934



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

    NGBRegressor(Base=<ngboost.esn_learners.ESN_kernel_ridge_learner object at 0x108f9eac8>,
                 Dist=<class 'ngboost.distns.normal.Normal'>,
                 Score=<class 'ngboost.scores.MLE'>, learning_rate=0.01,
                 minibatch_frac=1.0, n_estimators=500, natural_gradient=True,
                 tol=0.0001, verbose=True, verbose_eval=5) 
    
    [iter 0] loss=0.0535 val_loss=0.0000 scale=0.5000 norm=0.2703
    [iter 5] loss=-0.0062 val_loss=0.0000 scale=0.5000 norm=0.2491
    [iter 10] loss=-0.0503 val_loss=0.0000 scale=0.5000 norm=0.2364
    [iter 15] loss=-0.0856 val_loss=0.0000 scale=0.5000 norm=0.2280
    [iter 20] loss=-0.1150 val_loss=0.0000 scale=0.5000 norm=0.2227
    [iter 25] loss=-0.1408 val_loss=0.0000 scale=0.5000 norm=0.2194
    [iter 30] loss=-0.1640 val_loss=0.0000 scale=0.5000 norm=0.2175
    [iter 35] loss=-0.1851 val_loss=0.0000 scale=0.5000 norm=0.2167
    [iter 40] loss=-0.2045 val_loss=0.0000 scale=0.5000 norm=0.2166
    [iter 45] loss=-0.2227 val_loss=0.0000 scale=0.5000 norm=0.2170
    [iter 50] loss=-0.2395 val_loss=0.0000 scale=0.5000 norm=0.2178
    [iter 55] loss=-0.2492 val_loss=0.0000 scale=0.2500 norm=0.1092
    [iter 60] loss=-0.2569 val_loss=0.0000 scale=0.2500 norm=0.1094
    [iter 65] loss=-0.2643 val_loss=0.0000 scale=0.2500 norm=0.1097
    [iter 70] loss=-0.2714 val_loss=0.0000 scale=0.2500 norm=0.1100
    [iter 75] loss=-0.2783 val_loss=0.0000 scale=0.2500 norm=0.1102
    [iter 80] loss=-0.2848 val_loss=0.0000 scale=0.2500 norm=0.1105
    [iter 85] loss=-0.2910 val_loss=0.0000 scale=0.2500 norm=0.1107
    [iter 90] loss=-0.2970 val_loss=0.0000 scale=0.2500 norm=0.1110
    [iter 95] loss=-0.3026 val_loss=0.0000 scale=0.2500 norm=0.1112
    [iter 100] loss=-0.3079 val_loss=0.0000 scale=0.2500 norm=0.1113
    [iter 105] loss=-0.3128 val_loss=0.0000 scale=0.1250 norm=0.0557
    [iter 110] loss=-0.3151 val_loss=0.0000 scale=0.1250 norm=0.0557
    [iter 115] loss=-0.3174 val_loss=0.0000 scale=0.1250 norm=0.0558
    [iter 120] loss=-0.3195 val_loss=0.0000 scale=0.1250 norm=0.0558
    [iter 125] loss=-0.3215 val_loss=0.0000 scale=0.1250 norm=0.0558
    [iter 130] loss=-0.3235 val_loss=0.0000 scale=0.1250 norm=0.0558
    [iter 135] loss=-0.3254 val_loss=0.0000 scale=0.1250 norm=0.0557
    [iter 140] loss=-0.3271 val_loss=0.0000 scale=0.1250 norm=0.0557
    [iter 145] loss=-0.3288 val_loss=0.0000 scale=0.1250 norm=0.0557
    [iter 150] loss=-0.3303 val_loss=0.0000 scale=0.1250 norm=0.0557
    [iter 155] loss=-0.3317 val_loss=0.0000 scale=0.1250 norm=0.0556
    [iter 160] loss=-0.3330 val_loss=0.0000 scale=0.0625 norm=0.0278
    [iter 165] loss=-0.3336 val_loss=0.0000 scale=0.0625 norm=0.0278
    [iter 170] loss=-0.3343 val_loss=0.0000 scale=0.0625 norm=0.0278
    [iter 175] loss=-0.3348 val_loss=0.0000 scale=0.0625 norm=0.0277
    [iter 180] loss=-0.3354 val_loss=0.0000 scale=0.0625 norm=0.0277
    [iter 185] loss=-0.3359 val_loss=0.0000 scale=0.0625 norm=0.0277
    [iter 190] loss=-0.3364 val_loss=0.0000 scale=0.0625 norm=0.0277
    [iter 195] loss=-0.3369 val_loss=0.0000 scale=0.0625 norm=0.0277
    [iter 200] loss=-0.3373 val_loss=0.0000 scale=0.0625 norm=0.0277
    [iter 205] loss=-0.3377 val_loss=0.0000 scale=0.0625 norm=0.0276
    [iter 210] loss=-0.3381 val_loss=0.0000 scale=0.0625 norm=0.0276
    [iter 215] loss=-0.3384 val_loss=0.0000 scale=0.0625 norm=0.0276
    [iter 220] loss=-0.3388 val_loss=0.0000 scale=0.0312 norm=0.0138
    [iter 225] loss=-0.3389 val_loss=0.0000 scale=0.0625 norm=0.0275
    [iter 230] loss=-0.3391 val_loss=0.0000 scale=0.0312 norm=0.0138
    [iter 235] loss=-0.3392 val_loss=0.0000 scale=0.0312 norm=0.0138
    [iter 240] loss=-0.3393 val_loss=0.0000 scale=0.0312 norm=0.0138
    [iter 245] loss=-0.3395 val_loss=0.0000 scale=0.0312 norm=0.0137
    [iter 250] loss=-0.3396 val_loss=0.0000 scale=0.0312 norm=0.0137
    [iter 255] loss=-0.3397 val_loss=0.0000 scale=0.0312 norm=0.0137
    [iter 260] loss=-0.3398 val_loss=0.0000 scale=0.0312 norm=0.0137
    [iter 265] loss=-0.3399 val_loss=0.0000 scale=0.0312 norm=0.0137
    [iter 270] loss=-0.3400 val_loss=0.0000 scale=0.0312 norm=0.0137
    [iter 275] loss=-0.3400 val_loss=0.0000 scale=0.0312 norm=0.0137
    [iter 280] loss=-0.3401 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 285] loss=-0.3401 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 290] loss=-0.3402 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 295] loss=-0.3402 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 300] loss=-0.3403 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 305] loss=-0.3403 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 310] loss=-0.3403 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 315] loss=-0.3403 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 320] loss=-0.3404 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 325] loss=-0.3404 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 330] loss=-0.3404 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 335] loss=-0.3404 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 340] loss=-0.3404 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 345] loss=-0.3404 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 350] loss=-0.3404 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 355] loss=-0.3404 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 360] loss=-0.3405 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 365] loss=-0.3405 val_loss=0.0000 scale=0.0005 norm=0.0002
    [iter 370] loss=-0.3405 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 375] loss=-0.3405 val_loss=0.0000 scale=0.0001 norm=0.0001
    [iter 380] loss=-0.3405 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 385] loss=-0.3405 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 390] loss=-0.3405 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 395] loss=-0.3405 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 400] loss=-0.3405 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 405] loss=-0.3405 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 410] loss=-0.3405 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 415] loss=-0.3405 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 420] loss=-0.3405 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 425] loss=-0.3405 val_loss=0.0000 scale=0.0001 norm=0.0001
    [iter 430] loss=-0.3405 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 435] loss=-0.3405 val_loss=0.0000 scale=0.0020 norm=0.0009
    [iter 440] loss=-0.3405 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 445] loss=-0.3405 val_loss=0.0000 scale=0.0020 norm=0.0009
    [iter 450] loss=-0.3405 val_loss=0.0000 scale=0.0001 norm=0.0001
    [iter 455] loss=-0.3405 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 460] loss=-0.3405 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 465] loss=-0.3405 val_loss=0.0000 scale=0.0020 norm=0.0009
    [iter 470] loss=-0.3405 val_loss=0.0000 scale=0.0005 norm=0.0002
    [iter 475] loss=-0.3405 val_loss=0.0000 scale=0.0010 norm=0.0004
    [iter 480] loss=-0.3405 val_loss=0.0000 scale=0.0001 norm=0.0001
    [iter 485] loss=-0.3405 val_loss=0.0000 scale=0.0001 norm=0.0001
    [iter 490] loss=-0.3405 val_loss=0.0000 scale=0.0001 norm=0.0001
    [iter 495] loss=-0.3405 val_loss=0.0000 scale=0.0001 norm=0.0001
    
    Test MSE 0.03827606546641854
    Test NLL -0.2871522937482334



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

    NGBRegressor(Base=<ngboost.esn_learners.ESN_linear_svr_learner object at 0x1a1fad96d8>,
                 Dist=<class 'ngboost.distns.normal.Normal'>,
                 Score=<class 'ngboost.scores.MLE'>, learning_rate=0.01,
                 minibatch_frac=1.0, n_estimators=500, natural_gradient=True,
                 tol=0.0001, verbose=True, verbose_eval=10) 
    
    [iter 0] loss=0.0535 val_loss=0.0000 scale=1.0000 norm=0.5406
    [iter 10] loss=-0.0352 val_loss=0.0000 scale=0.5000 norm=0.2472
    [iter 20] loss=-0.0855 val_loss=0.0000 scale=0.5000 norm=0.2382
    [iter 30] loss=-0.1288 val_loss=0.0000 scale=0.5000 norm=0.2329
    [iter 40] loss=-0.1660 val_loss=0.0000 scale=0.5000 norm=0.2306
    [iter 50] loss=-0.1886 val_loss=0.0000 scale=0.2500 norm=0.1152
    [iter 60] loss=-0.2032 val_loss=0.0000 scale=0.2500 norm=0.1153
    [iter 70] loss=-0.2166 val_loss=0.0000 scale=0.2500 norm=0.1155
    [iter 80] loss=-0.2290 val_loss=0.0000 scale=0.2500 norm=0.1157
    [iter 90] loss=-0.2401 val_loss=0.0000 scale=0.2500 norm=0.1160
    [iter 100] loss=-0.2490 val_loss=0.0000 scale=0.1250 norm=0.0581
    [iter 110] loss=-0.2535 val_loss=0.0000 scale=0.1250 norm=0.0582
    [iter 120] loss=-0.2578 val_loss=0.0000 scale=0.1250 norm=0.0582
    [iter 130] loss=-0.2618 val_loss=0.0000 scale=0.1250 norm=0.0583
    [iter 140] loss=-0.2654 val_loss=0.0000 scale=0.1250 norm=0.0583
    [iter 150] loss=-0.2687 val_loss=0.0000 scale=0.1250 norm=0.0583
    [iter 160] loss=-0.2717 val_loss=0.0000 scale=0.1250 norm=0.0582
    [iter 170] loss=-0.2741 val_loss=0.0000 scale=0.1250 norm=0.0582
    [iter 180] loss=-0.2755 val_loss=0.0000 scale=0.0625 norm=0.0291
    [iter 190] loss=-0.2770 val_loss=0.0000 scale=0.0625 norm=0.0290
    [iter 200] loss=-0.2780 val_loss=0.0000 scale=0.0625 norm=0.0290
    [iter 210] loss=-0.2790 val_loss=0.0000 scale=0.0625 norm=0.0290
    [iter 220] loss=-0.2799 val_loss=0.0000 scale=0.0625 norm=0.0289
    [iter 230] loss=-0.2807 val_loss=0.0000 scale=0.0625 norm=0.0289
    [iter 240] loss=-0.2814 val_loss=0.0000 scale=0.0625 norm=0.0289
    [iter 250] loss=-0.2819 val_loss=0.0000 scale=0.0312 norm=0.0144
    [iter 260] loss=-0.2823 val_loss=0.0000 scale=0.0312 norm=0.0144
    [iter 270] loss=-0.2825 val_loss=0.0000 scale=0.0312 norm=0.0144
    [iter 280] loss=-0.2828 val_loss=0.0000 scale=0.0312 norm=0.0144
    [iter 290] loss=-0.2830 val_loss=0.0000 scale=0.0312 norm=0.0144
    [iter 300] loss=-0.2832 val_loss=0.0000 scale=0.0312 norm=0.0143
    [iter 310] loss=-0.2834 val_loss=0.0000 scale=0.0312 norm=0.0143
    [iter 320] loss=-0.2835 val_loss=0.0000 scale=0.0312 norm=0.0143
    [iter 330] loss=-0.2836 val_loss=0.0000 scale=0.0312 norm=0.0143
    [iter 340] loss=-0.2837 val_loss=0.0000 scale=0.0078 norm=0.0036
    [iter 350] loss=-0.2838 val_loss=0.0000 scale=0.0156 norm=0.0071
    [iter 360] loss=-0.2838 val_loss=0.0000 scale=0.0156 norm=0.0071
    [iter 370] loss=-0.2838 val_loss=0.0000 scale=0.0156 norm=0.0071
    [iter 380] loss=-0.2839 val_loss=0.0000 scale=0.0078 norm=0.0036
    [iter 390] loss=-0.2839 val_loss=0.0000 scale=0.0078 norm=0.0036
    [iter 400] loss=-0.2839 val_loss=0.0000 scale=0.0078 norm=0.0036
    [iter 410] loss=-0.2840 val_loss=0.0000 scale=0.0078 norm=0.0036
    [iter 420] loss=-0.2840 val_loss=0.0000 scale=0.0078 norm=0.0036
    [iter 430] loss=-0.2840 val_loss=0.0000 scale=0.0005 norm=0.0002
    [iter 440] loss=-0.2840 val_loss=0.0000 scale=0.0001 norm=0.0001
    [iter 450] loss=-0.2840 val_loss=0.0000 scale=0.0010 norm=0.0004
    [iter 460] loss=-0.2840 val_loss=0.0000 scale=0.0010 norm=0.0004
    [iter 470] loss=-0.2840 val_loss=0.0000 scale=0.0001 norm=0.0001
    [iter 480] loss=-0.2840 val_loss=0.0000 scale=0.0020 norm=0.0009


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 490] loss=-0.2840 val_loss=0.0000 scale=0.0078 norm=0.0036
    
    Test MSE 0.032988082604124065
    Test NLL -0.32993186460938617



![png](output_18_3.png)



```python
filename = 'ws+sin(wd)+cos(wd)-2.csv'
Pred_df.to_csv('/Users/apple/Documents/ML_Project/ML - 2.1/result/csv/'+filename)
Pred_df = pd.DataFrame(Y_scaler.inverse_transform(Pred_df), columns=Pred_df.columns)
Pred_df.to_csv('/Users/apple/Documents/ML_Project/ML - 2.1/result/inverse/'+filename)
```


```python

```
