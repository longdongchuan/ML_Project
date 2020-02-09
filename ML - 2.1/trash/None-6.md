# 数据准备

## 连续区间


```python
import numpy as np
import pandas as pd
from utils import *
from ngboost.learners import *
```

输出连续区间


```python
path = '/Users/apple/Documents/ML_Project/ML - 2.1/Data/国际西班牙数据.csv'
data = load_data(path, add_time=True, describe=False)

index = data[data['wind_speed'].isna() |
    data['wind_direction'].isna() |
    data['wind_power'].isna()]['wind_power'].index.tolist()
a = index[0]
b=-1
for i,x in enumerate(index):
    if i<len(index)-1:
        if index[i+1] > index[i]+1:
            print('Continue: [',b+1, ',', a-1,']','len:',a-b-2,
                '\tNan: [',a, ',',index[i], '] len:',index[i]-a+1)     
            a = index[i+1]
            b = index[i]
    else: 
        a=index[-15]
        print('Continue: [',b+1, ',', a-1,']','len:',a-b-2,
                '\tNan: [',a, ',',index[i], '] len:',index[i]-a+1)     
        a=len(data)
        b=index[i]
        print('Continue: [',b+1, ',', a-1,']','len:',a-b-2)

del data
```

    Continue: [ 0 , 975 ] len: 975 	Nan: [ 976 , 976 ] len: 1
    Continue: [ 977 , 2216 ] len: 1239 	Nan: [ 2217 , 2221 ] len: 5
    Continue: [ 2222 , 3498 ] len: 1276 	Nan: [ 3499 , 3560 ] len: 62
    Continue: [ 3561 , 3572 ] len: 11 	Nan: [ 3573 , 3589 ] len: 17
    Continue: [ 3590 , 4314 ] len: 724 	Nan: [ 4315 , 4401 ] len: 87
    Continue: [ 4402 , 6255 ] len: 1853 	Nan: [ 6256 , 6273 ] len: 18
    Continue: [ 6274 , 6375 ] len: 101 	Nan: [ 6376 , 6376 ] len: 1
    Continue: [ 6377 , 6417 ] len: 40 	Nan: [ 6418 , 6425 ] len: 8
    Continue: [ 6426 , 10427 ] len: 4001 	Nan: [ 10428 , 10447 ] len: 20
    Continue: [ 10448 , 13432 ] len: 2984 	Nan: [ 13433 , 13434 ] len: 2
    Continue: [ 13435 , 13976 ] len: 541 	Nan: [ 13977 , 13985 ] len: 9
    Continue: [ 13986 , 14000 ] len: 14 	Nan: [ 14001 , 14009 ] len: 9
    Continue: [ 14010 , 14024 ] len: 14 	Nan: [ 14025 , 14033 ] len: 9
    Continue: [ 14034 , 14048 ] len: 14 	Nan: [ 14049 , 14057 ] len: 9
    Continue: [ 14058 , 14072 ] len: 14 	Nan: [ 14073 , 14077 ] len: 5
    Continue: [ 14078 , 14387 ] len: 309 	Nan: [ 14388 , 14388 ] len: 1
    Continue: [ 14389 , 17872 ] len: 3483 	Nan: [ 17873 , 17887 ] len: 15
    Continue: [ 17888 , 18265 ] len: 377


## 可视化


```python
%config InlineBackend.figure_format='retina'
from tqdm.notebook import tqdm as tqdm
for day in tqdm(np.arange(1,2)):
    plot_module1(year=2017, month=10, day=day, figsize=(8,13), 
                 save_fig=False, close_fig=True)
for day in tqdm(np.arange(1,2)):
    plot_module2(year=2017, month=10, day=day, figsize=(8,10), 
                 save_fig=False, close_fig=True)
```


    HBox(children=(FloatProgress(value=0.0, max=1.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=1.0), HTML(value='')))


    


# 模型预测


```python
import numpy as np
import pandas as pd
from module.utils import *
from ngboost.learners import *
%config InlineBackend.figure_format='retina'
```


```python
transform=None
X_train, X_test, Y_train, Y_test, Y_scaler = get_data(hour_num=6, transform=transform,
                                            drop_time=True, scale=True, return_y_scaler=True)
Pred_df = Y_test
```

    get_data(hour_num=6, transform='None', drop_time=True, scale=True)
    
    Input space:  Index(['wind_speed', 'wind_direction', 'wind_speed-1', 'wind_speed-2',
           'wind_speed-3', 'wind_speed-4', 'wind_speed-5', 'wind_speed-6',
           'wind_direction-1', 'wind_direction-2', 'wind_direction-3',
           'wind_direction-4', 'wind_direction-5', 'wind_direction-6',
           'wind_power-1', 'wind_power-2', 'wind_power-3', 'wind_power-4',
           'wind_power-5', 'wind_power-6'],
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
    
    [iter 0] loss=0.0537 val_loss=0.0000 scale=0.5000 norm=0.2702
    [iter 100] loss=-0.3438 val_loss=0.0000 scale=0.1250 norm=0.0571
    [iter 200] loss=-0.3690 val_loss=0.0000 scale=0.0312 norm=0.0139
    [iter 300] loss=-0.3706 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 400] loss=-0.3707 val_loss=0.0000 scale=0.0020 norm=0.0009
    
    Test MSE 0.03673669379839538
    Test NLL -0.30001758834976244



![png](output_10_1.png)


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
    
    [iter 0] loss=0.0537 val_loss=0.0000 scale=0.5000 norm=0.2702
    [iter 200] loss=-0.3026 val_loss=0.0000 scale=0.1250 norm=0.0480
    [iter 400] loss=-0.3133 val_loss=0.0000 scale=0.0156 norm=0.0058
    [iter 600] loss=-0.3138 val_loss=0.0000 scale=0.0039 norm=0.0015
    [iter 800] loss=-0.3138 val_loss=0.0000 scale=0.0005 norm=0.0002
    
    Test MSE 0.04437069264792852
    Test NLL -0.22179703232153028



![png](output_12_1.png)


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
    
    [iter 0] loss=0.1430 val_loss=0.0000 scale=0.2500 norm=0.6105
    [iter 100] loss=0.0922 val_loss=0.0000 scale=0.0156 norm=0.0470
    [iter 200] loss=0.0897 val_loss=0.0000 scale=0.0039 norm=0.0121
    [iter 300] loss=0.0896 val_loss=0.0000 scale=0.0010 norm=0.0030
    [iter 400] loss=0.0896 val_loss=0.0000 scale=0.0002 norm=0.0008
    
    Test MSE 0.0313971811940047
    Test NLL -0.33499919929388194



![png](output_14_1.png)


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
    
    [iter 0] loss=0.0537 val_loss=0.0000 scale=1.0000 norm=0.5405
    [iter 100] loss=-0.2474 val_loss=0.0000 scale=0.2500 norm=0.1248
    [iter 200] loss=-0.2932 val_loss=0.0000 scale=0.0625 norm=0.0300
    [iter 300] loss=-0.2997 val_loss=0.0000 scale=0.0312 norm=0.0145
    [iter 400] loss=-0.3003 val_loss=0.0000 scale=0.0078 norm=0.0036
    
    Test MSE 0.020575707089261725
    Test NLL -0.39794916091661864



![png](output_16_1.png)


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
    
    [iter 0] loss=0.0537 val_loss=0.0000 scale=0.5000 norm=0.2702
    [iter 10] loss=-0.0546 val_loss=0.0000 scale=0.5000 norm=0.2318
    [iter 20] loss=-0.1181 val_loss=0.0000 scale=0.5000 norm=0.2172
    [iter 30] loss=-0.1650 val_loss=0.0000 scale=0.5000 norm=0.2117
    [iter 40] loss=-0.2039 val_loss=0.0000 scale=0.5000 norm=0.2106
    [iter 50] loss=-0.2377 val_loss=0.0000 scale=0.5000 norm=0.2118
    [iter 60] loss=-0.2546 val_loss=0.0000 scale=0.2500 norm=0.1065
    [iter 70] loss=-0.2690 val_loss=0.0000 scale=0.2500 norm=0.1071
    [iter 80] loss=-0.2824 val_loss=0.0000 scale=0.2500 norm=0.1077
    [iter 90] loss=-0.2947 val_loss=0.0000 scale=0.2500 norm=0.1082
    [iter 100] loss=-0.3058 val_loss=0.0000 scale=0.2500 norm=0.1086
    [iter 110] loss=-0.3143 val_loss=0.0000 scale=0.1250 norm=0.0544
    [iter 120] loss=-0.3189 val_loss=0.0000 scale=0.1250 norm=0.0545
    [iter 130] loss=-0.3231 val_loss=0.0000 scale=0.1250 norm=0.0545
    [iter 140] loss=-0.3269 val_loss=0.0000 scale=0.1250 norm=0.0545
    [iter 150] loss=-0.3304 val_loss=0.0000 scale=0.1250 norm=0.0545
    [iter 160] loss=-0.3334 val_loss=0.0000 scale=0.1250 norm=0.0544
    [iter 170] loss=-0.3357 val_loss=0.0000 scale=0.0625 norm=0.0271
    [iter 180] loss=-0.3369 val_loss=0.0000 scale=0.0625 norm=0.0271
    [iter 190] loss=-0.3380 val_loss=0.0000 scale=0.0625 norm=0.0271
    [iter 200] loss=-0.3390 val_loss=0.0000 scale=0.0625 norm=0.0270
    [iter 210] loss=-0.3398 val_loss=0.0000 scale=0.0625 norm=0.0270
    [iter 220] loss=-0.3406 val_loss=0.0000 scale=0.0625 norm=0.0269
    [iter 230] loss=-0.3411 val_loss=0.0000 scale=0.0312 norm=0.0134
    [iter 240] loss=-0.3414 val_loss=0.0000 scale=0.0312 norm=0.0134
    [iter 250] loss=-0.3417 val_loss=0.0000 scale=0.0312 norm=0.0134
    [iter 260] loss=-0.3419 val_loss=0.0000 scale=0.0312 norm=0.0134
    [iter 270] loss=-0.3421 val_loss=0.0000 scale=0.0312 norm=0.0134
    [iter 280] loss=-0.3423 val_loss=0.0000 scale=0.0312 norm=0.0134
    [iter 290] loss=-0.3424 val_loss=0.0000 scale=0.0156 norm=0.0067
    [iter 300] loss=-0.3425 val_loss=0.0000 scale=0.0156 norm=0.0067
    [iter 310] loss=-0.3426 val_loss=0.0000 scale=0.0156 norm=0.0067
    [iter 320] loss=-0.3426 val_loss=0.0000 scale=0.0156 norm=0.0067
    [iter 330] loss=-0.3427 val_loss=0.0000 scale=0.0156 norm=0.0067
    [iter 340] loss=-0.3427 val_loss=0.0000 scale=0.0078 norm=0.0033
    [iter 350] loss=-0.3427 val_loss=0.0000 scale=0.0078 norm=0.0033
    [iter 360] loss=-0.3428 val_loss=0.0000 scale=0.0078 norm=0.0033
    [iter 370] loss=-0.3428 val_loss=0.0000 scale=0.0078 norm=0.0033
    [iter 380] loss=-0.3428 val_loss=0.0000 scale=0.0078 norm=0.0033
    [iter 390] loss=-0.3428 val_loss=0.0000 scale=0.0078 norm=0.0033
    [iter 400] loss=-0.3428 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 410] loss=-0.3428 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 420] loss=-0.3428 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 430] loss=-0.3428 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 440] loss=-0.3428 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 450] loss=-0.3428 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 460] loss=-0.3428 val_loss=0.0000 scale=0.0020 norm=0.0008
    [iter 470] loss=-0.3428 val_loss=0.0000 scale=0.0020 norm=0.0008
    [iter 480] loss=-0.3428 val_loss=0.0000 scale=0.0020 norm=0.0008
    [iter 490] loss=-0.3428 val_loss=0.0000 scale=0.0020 norm=0.0008
    
    Test MSE 0.037804915863731935
    Test NLL -0.28926778383139756



![png](output_18_1.png)


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

    NGBRegressor(Base=<ngboost.esn_learners.ESN_Ridge_learner object at 0x1a1ec17358>,
                 Dist=<class 'ngboost.distns.normal.Normal'>,
                 Score=<class 'ngboost.scores.MLE'>, learning_rate=0.01,
                 minibatch_frac=1.0, n_estimators=500, natural_gradient=True,
                 tol=0.0001, verbose=True, verbose_eval=10) 
    
    [iter 0] loss=0.0537 val_loss=0.0000 scale=0.5000 norm=0.2702
    [iter 10] loss=-0.0553 val_loss=0.0000 scale=0.5000 norm=0.2305
    [iter 20] loss=-0.1180 val_loss=0.0000 scale=0.5000 norm=0.2151
    [iter 30] loss=-0.1637 val_loss=0.0000 scale=0.5000 norm=0.2089
    [iter 40] loss=-0.2010 val_loss=0.0000 scale=0.5000 norm=0.2068
    [iter 50] loss=-0.2253 val_loss=0.0000 scale=0.2500 norm=0.1034
    [iter 60] loss=-0.2400 val_loss=0.0000 scale=0.2500 norm=0.1036
    [iter 70] loss=-0.2539 val_loss=0.0000 scale=0.2500 norm=0.1039
    [iter 80] loss=-0.2667 val_loss=0.0000 scale=0.2500 norm=0.1042
    [iter 90] loss=-0.2784 val_loss=0.0000 scale=0.2500 norm=0.1045
    [iter 100] loss=-0.2892 val_loss=0.0000 scale=0.2500 norm=0.1048
    [iter 110] loss=-0.2984 val_loss=0.0000 scale=0.1250 norm=0.0525
    [iter 120] loss=-0.3028 val_loss=0.0000 scale=0.1250 norm=0.0525
    [iter 130] loss=-0.3069 val_loss=0.0000 scale=0.1250 norm=0.0525
    [iter 140] loss=-0.3106 val_loss=0.0000 scale=0.1250 norm=0.0525
    [iter 150] loss=-0.3141 val_loss=0.0000 scale=0.1250 norm=0.0524
    [iter 160] loss=-0.3172 val_loss=0.0000 scale=0.1250 norm=0.0524
    [iter 170] loss=-0.3201 val_loss=0.0000 scale=0.1250 norm=0.0523
    [iter 180] loss=-0.3223 val_loss=0.0000 scale=0.0625 norm=0.0261
    [iter 190] loss=-0.3234 val_loss=0.0000 scale=0.0625 norm=0.0261
    [iter 200] loss=-0.3245 val_loss=0.0000 scale=0.0625 norm=0.0260
    [iter 210] loss=-0.3254 val_loss=0.0000 scale=0.0625 norm=0.0260
    [iter 220] loss=-0.3263 val_loss=0.0000 scale=0.0625 norm=0.0260
    [iter 230] loss=-0.3271 val_loss=0.0000 scale=0.0625 norm=0.0259
    [iter 240] loss=-0.3278 val_loss=0.0000 scale=0.0625 norm=0.0259
    [iter 250] loss=-0.3283 val_loss=0.0000 scale=0.0312 norm=0.0129
    [iter 260] loss=-0.3285 val_loss=0.0000 scale=0.0312 norm=0.0129
    [iter 270] loss=-0.3288 val_loss=0.0000 scale=0.0312 norm=0.0129
    [iter 280] loss=-0.3290 val_loss=0.0000 scale=0.0312 norm=0.0129
    [iter 290] loss=-0.3292 val_loss=0.0000 scale=0.0312 norm=0.0129
    [iter 300] loss=-0.3294 val_loss=0.0000 scale=0.0312 norm=0.0128
    [iter 310] loss=-0.3296 val_loss=0.0000 scale=0.0156 norm=0.0064
    [iter 320] loss=-0.3296 val_loss=0.0000 scale=0.0156 norm=0.0064
    [iter 330] loss=-0.3297 val_loss=0.0000 scale=0.0156 norm=0.0064
    [iter 340] loss=-0.3298 val_loss=0.0000 scale=0.0156 norm=0.0064
    [iter 350] loss=-0.3298 val_loss=0.0000 scale=0.0156 norm=0.0064
    [iter 360] loss=-0.3299 val_loss=0.0000 scale=0.0156 norm=0.0064
    [iter 370] loss=-0.3299 val_loss=0.0000 scale=0.0156 norm=0.0064
    [iter 380] loss=-0.3299 val_loss=0.0000 scale=0.0156 norm=0.0064
    [iter 390] loss=-0.3300 val_loss=0.0000 scale=0.0078 norm=0.0032
    [iter 400] loss=-0.3300 val_loss=0.0000 scale=0.0078 norm=0.0032
    [iter 410] loss=-0.3300 val_loss=0.0000 scale=0.0078 norm=0.0032
    [iter 420] loss=-0.3300 val_loss=0.0000 scale=0.0078 norm=0.0032
    [iter 430] loss=-0.3300 val_loss=0.0000 scale=0.0078 norm=0.0032
    [iter 440] loss=-0.3300 val_loss=0.0000 scale=0.0039 norm=0.0016
    [iter 450] loss=-0.3300 val_loss=0.0000 scale=0.0039 norm=0.0016
    [iter 460] loss=-0.3300 val_loss=0.0000 scale=0.0039 norm=0.0016
    [iter 470] loss=-0.3300 val_loss=0.0000 scale=0.0039 norm=0.0016
    [iter 480] loss=-0.3300 val_loss=0.0000 scale=0.0039 norm=0.0016
    [iter 490] loss=-0.3300 val_loss=0.0000 scale=0.0039 norm=0.0016
    
    Test MSE 0.039748718687679575
    Test NLL -0.26696970747689736



![png](output_20_1.png)


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

    NGBRegressor(Base=<ngboost.esn_learners.ESN_kernel_ridge_learner object at 0x1a21d72358>,
                 Dist=<class 'ngboost.distns.normal.Normal'>,
                 Score=<class 'ngboost.scores.MLE'>, learning_rate=0.01,
                 minibatch_frac=1.0, n_estimators=500, natural_gradient=True,
                 tol=0.0001, verbose=True, verbose_eval=5) 
    
    [iter 0] loss=0.0537 val_loss=0.0000 scale=0.5000 norm=0.2702
    [iter 5] loss=-0.0063 val_loss=0.0000 scale=0.5000 norm=0.2485
    [iter 10] loss=-0.0500 val_loss=0.0000 scale=0.5000 norm=0.2353
    [iter 15] loss=-0.0847 val_loss=0.0000 scale=0.5000 norm=0.2267
    [iter 20] loss=-0.1137 val_loss=0.0000 scale=0.5000 norm=0.2213
    [iter 25] loss=-0.1389 val_loss=0.0000 scale=0.5000 norm=0.2181
    [iter 30] loss=-0.1614 val_loss=0.0000 scale=0.5000 norm=0.2161
    [iter 35] loss=-0.1820 val_loss=0.0000 scale=0.5000 norm=0.2152
    [iter 40] loss=-0.2010 val_loss=0.0000 scale=0.5000 norm=0.2150
    [iter 45] loss=-0.2187 val_loss=0.0000 scale=0.5000 norm=0.2153
    [iter 50] loss=-0.2352 val_loss=0.0000 scale=0.5000 norm=0.2160
    [iter 55] loss=-0.2446 val_loss=0.0000 scale=0.2500 norm=0.1083
    [iter 60] loss=-0.2521 val_loss=0.0000 scale=0.2500 norm=0.1085
    [iter 65] loss=-0.2593 val_loss=0.0000 scale=0.2500 norm=0.1088
    [iter 70] loss=-0.2663 val_loss=0.0000 scale=0.2500 norm=0.1090
    [iter 75] loss=-0.2730 val_loss=0.0000 scale=0.2500 norm=0.1093
    [iter 80] loss=-0.2794 val_loss=0.0000 scale=0.2500 norm=0.1095
    [iter 85] loss=-0.2856 val_loss=0.0000 scale=0.2500 norm=0.1098
    [iter 90] loss=-0.2915 val_loss=0.0000 scale=0.2500 norm=0.1100
    [iter 95] loss=-0.2970 val_loss=0.0000 scale=0.2500 norm=0.1102
    [iter 100] loss=-0.3022 val_loss=0.0000 scale=0.2500 norm=0.1104
    [iter 105] loss=-0.3071 val_loss=0.0000 scale=0.1250 norm=0.0553
    [iter 110] loss=-0.3099 val_loss=0.0000 scale=0.1250 norm=0.0553
    [iter 115] loss=-0.3121 val_loss=0.0000 scale=0.1250 norm=0.0553
    [iter 120] loss=-0.3142 val_loss=0.0000 scale=0.1250 norm=0.0553
    [iter 125] loss=-0.3162 val_loss=0.0000 scale=0.1250 norm=0.0553
    [iter 130] loss=-0.3182 val_loss=0.0000 scale=0.1250 norm=0.0553
    [iter 135] loss=-0.3200 val_loss=0.0000 scale=0.1250 norm=0.0553
    [iter 140] loss=-0.3218 val_loss=0.0000 scale=0.1250 norm=0.0553
    [iter 145] loss=-0.3235 val_loss=0.0000 scale=0.1250 norm=0.0553
    [iter 150] loss=-0.3251 val_loss=0.0000 scale=0.1250 norm=0.0553
    [iter 155] loss=-0.3266 val_loss=0.0000 scale=0.1250 norm=0.0553
    [iter 160] loss=-0.3280 val_loss=0.0000 scale=0.1250 norm=0.0552
    [iter 165] loss=-0.3293 val_loss=0.0000 scale=0.1250 norm=0.0552
    [iter 170] loss=-0.3301 val_loss=0.0000 scale=0.0625 norm=0.0276
    [iter 175] loss=-0.3307 val_loss=0.0000 scale=0.0625 norm=0.0276
    [iter 180] loss=-0.3312 val_loss=0.0000 scale=0.0625 norm=0.0275
    [iter 185] loss=-0.3318 val_loss=0.0000 scale=0.0625 norm=0.0275
    [iter 190] loss=-0.3323 val_loss=0.0000 scale=0.0625 norm=0.0275
    [iter 195] loss=-0.3327 val_loss=0.0000 scale=0.0625 norm=0.0275
    [iter 200] loss=-0.3332 val_loss=0.0000 scale=0.0625 norm=0.0275
    [iter 205] loss=-0.3336 val_loss=0.0000 scale=0.0625 norm=0.0275
    [iter 210] loss=-0.3340 val_loss=0.0000 scale=0.0625 norm=0.0274
    [iter 215] loss=-0.3344 val_loss=0.0000 scale=0.0625 norm=0.0274
    [iter 220] loss=-0.3347 val_loss=0.0000 scale=0.0625 norm=0.0274
    [iter 225] loss=-0.3350 val_loss=0.0000 scale=0.0312 norm=0.0137
    [iter 230] loss=-0.3352 val_loss=0.0000 scale=0.0312 norm=0.0137
    [iter 235] loss=-0.3353 val_loss=0.0000 scale=0.0312 norm=0.0137
    [iter 240] loss=-0.3355 val_loss=0.0000 scale=0.0312 norm=0.0137
    [iter 245] loss=-0.3356 val_loss=0.0000 scale=0.0312 norm=0.0137
    [iter 250] loss=-0.3357 val_loss=0.0000 scale=0.0312 norm=0.0136
    [iter 255] loss=-0.3358 val_loss=0.0000 scale=0.0312 norm=0.0136
    [iter 260] loss=-0.3359 val_loss=0.0000 scale=0.0312 norm=0.0136
    [iter 265] loss=-0.3360 val_loss=0.0000 scale=0.0312 norm=0.0136
    [iter 270] loss=-0.3361 val_loss=0.0000 scale=0.0312 norm=0.0136
    [iter 275] loss=-0.3362 val_loss=0.0000 scale=0.0312 norm=0.0136
    [iter 280] loss=-0.3363 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 285] loss=-0.3363 val_loss=0.0000 scale=0.0312 norm=0.0136
    [iter 290] loss=-0.3364 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 295] loss=-0.3364 val_loss=0.0000 scale=0.0312 norm=0.0136
    [iter 300] loss=-0.3364 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 305] loss=-0.3365 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 310] loss=-0.3365 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 315] loss=-0.3365 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 320] loss=-0.3366 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 325] loss=-0.3366 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 330] loss=-0.3366 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 335] loss=-0.3366 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 340] loss=-0.3366 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 345] loss=-0.3367 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 350] loss=-0.3367 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 355] loss=-0.3367 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 360] loss=-0.3367 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 365] loss=-0.3367 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 370] loss=-0.3367 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 375] loss=-0.3367 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 380] loss=-0.3367 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 385] loss=-0.3367 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 390] loss=-0.3367 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 395] loss=-0.3367 val_loss=0.0000 scale=0.0020 norm=0.0008
    [iter 400] loss=-0.3367 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 405] loss=-0.3367 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 410] loss=-0.3367 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 415] loss=-0.3367 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 420] loss=-0.3367 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 425] loss=-0.3367 val_loss=0.0000 scale=0.0001 norm=0.0001
    [iter 430] loss=-0.3367 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 435] loss=-0.3368 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 440] loss=-0.3368 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 445] loss=-0.3368 val_loss=0.0000 scale=0.0001 norm=0.0001
    [iter 450] loss=-0.3368 val_loss=0.0000 scale=0.0001 norm=0.0001
    [iter 455] loss=-0.3368 val_loss=0.0000 scale=0.0001 norm=0.0001
    [iter 460] loss=-0.3368 val_loss=0.0000 scale=0.0005 norm=0.0002
    [iter 465] loss=-0.3368 val_loss=0.0000 scale=0.0010 norm=0.0004
    [iter 470] loss=-0.3368 val_loss=0.0000 scale=0.0020 norm=0.0008
    [iter 475] loss=-0.3368 val_loss=0.0000 scale=0.0001 norm=0.0001
    [iter 480] loss=-0.3368 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 485] loss=-0.3368 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 490] loss=-0.3368 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 495] loss=-0.3368 val_loss=0.0000 scale=0.0002 norm=0.0001
    
    Test MSE 0.03718129113537457
    Test NLL -0.2959785240482578



![png](output_22_1.png)


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

    NGBRegressor(Base=<ngboost.esn_learners.ESN_linear_svr_learner object at 0x1a20b59f28>,
                 Dist=<class 'ngboost.distns.normal.Normal'>,
                 Score=<class 'ngboost.scores.MLE'>, learning_rate=0.01,
                 minibatch_frac=1.0, n_estimators=500, natural_gradient=True,
                 tol=0.0001, verbose=True, verbose_eval=10) 
    
    [iter 0] loss=0.0537 val_loss=0.0000 scale=0.5000 norm=0.2702


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 10] loss=-0.0187 val_loss=0.0000 scale=0.5000 norm=0.2485


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 20] loss=-0.0744 val_loss=0.0000 scale=0.5000 norm=0.2361
    [iter 30] loss=-0.1175 val_loss=0.0000 scale=0.5000 norm=0.2302


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 40] loss=-0.1532 val_loss=0.0000 scale=0.5000 norm=0.2282
    [iter 50] loss=-0.1749 val_loss=0.0000 scale=0.5000 norm=0.2281
    [iter 60] loss=-0.1907 val_loss=0.0000 scale=0.2500 norm=0.1142


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 70] loss=-0.2039 val_loss=0.0000 scale=0.2500 norm=0.1145
    [iter 80] loss=-0.2161 val_loss=0.0000 scale=0.2500 norm=0.1148
    [iter 90] loss=-0.2269 val_loss=0.0000 scale=0.2500 norm=0.1152


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 100] loss=-0.2368 val_loss=0.0000 scale=0.2500 norm=0.1156


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 110] loss=-0.2432 val_loss=0.0000 scale=0.1250 norm=0.0579


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 120] loss=-0.2473 val_loss=0.0000 scale=0.1250 norm=0.0580
    [iter 130] loss=-0.2512 val_loss=0.0000 scale=0.1250 norm=0.0580


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 140] loss=-0.2548 val_loss=0.0000 scale=0.1250 norm=0.0581


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 150] loss=-0.2582 val_loss=0.0000 scale=0.1250 norm=0.0581


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 160] loss=-0.2612 val_loss=0.0000 scale=0.1250 norm=0.0581


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 170] loss=-0.2641 val_loss=0.0000 scale=0.1250 norm=0.0581


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 180] loss=-0.2664 val_loss=0.0000 scale=0.1250 norm=0.0580


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 190] loss=-0.2678 val_loss=0.0000 scale=0.0625 norm=0.0290


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 200] loss=-0.2692 val_loss=0.0000 scale=0.0625 norm=0.0290


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 210] loss=-0.2702 val_loss=0.0000 scale=0.0625 norm=0.0289


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 220] loss=-0.2712 val_loss=0.0000 scale=0.0625 norm=0.0289


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 230] loss=-0.2721 val_loss=0.0000 scale=0.0625 norm=0.0289


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 240] loss=-0.2729 val_loss=0.0000 scale=0.0625 norm=0.0288
    [iter 250] loss=-0.2735 val_loss=0.0000 scale=0.0625 norm=0.0288


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 260] loss=-0.2741 val_loss=0.0000 scale=0.0312 norm=0.0144
    [iter 270] loss=-0.2746 val_loss=0.0000 scale=0.0312 norm=0.0144


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 280] loss=-0.2748 val_loss=0.0000 scale=0.0312 norm=0.0144
    [iter 290] loss=-0.2751 val_loss=0.0000 scale=0.0312 norm=0.0143


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 300] loss=-0.2753 val_loss=0.0000 scale=0.0312 norm=0.0143
    [iter 310] loss=-0.2755 val_loss=0.0000 scale=0.0312 norm=0.0143
    [iter 320] loss=-0.2757 val_loss=0.0000 scale=0.0312 norm=0.0143
    [iter 330] loss=-0.2759 val_loss=0.0000 scale=0.0156 norm=0.0071
    [iter 340] loss=-0.2760 val_loss=0.0000 scale=0.0156 norm=0.0071
    [iter 350] loss=-0.2761 val_loss=0.0000 scale=0.0156 norm=0.0071


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 360] loss=-0.2762 val_loss=0.0000 scale=0.0156 norm=0.0071
    [iter 370] loss=-0.2763 val_loss=0.0000 scale=0.0156 norm=0.0071


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 380] loss=-0.2763 val_loss=0.0000 scale=0.0078 norm=0.0036
    [iter 390] loss=-0.2764 val_loss=0.0000 scale=0.0156 norm=0.0071


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 400] loss=-0.2764 val_loss=0.0000 scale=0.0078 norm=0.0036


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 410] loss=-0.2764 val_loss=0.0000 scale=0.0078 norm=0.0036


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 420] loss=-0.2765 val_loss=0.0000 scale=0.0020 norm=0.0009


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 430] loss=-0.2765 val_loss=0.0000 scale=0.0078 norm=0.0036
    [iter 440] loss=-0.2765 val_loss=0.0000 scale=0.0039 norm=0.0018


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 450] loss=-0.2765 val_loss=0.0000 scale=0.0156 norm=0.0071


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 460] loss=-0.2766 val_loss=0.0000 scale=0.0078 norm=0.0036


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 470] loss=-0.2766 val_loss=0.0000 scale=0.0078 norm=0.0036
    [iter 480] loss=-0.2766 val_loss=0.0000 scale=0.0078 norm=0.0035
    [iter 490] loss=-0.2766 val_loss=0.0000 scale=0.0078 norm=0.0035


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    
    Test MSE 0.03689292820955263
    Test NLL -0.2978679342914011



![png](output_24_63.png)



```python
# Pred_df = pd.DataFrame(Y_scaler.inverse_transform(Pred_df), columns=Pred_df.columns)
Pred_df.to_csv('/Users/apple/Documents/ML_Project/ML - 2.1/result/'+'None'+'.csv')
```


```python

```
