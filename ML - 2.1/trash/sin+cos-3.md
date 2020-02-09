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
transform='sin+cos'
X_train, X_test, Y_train, Y_test = get_data(hour_num=3, transform=transform,
                                            drop_time=True, scale=True)
Pred_df = Y_test
```

    get_data(hour_num=3, transform='sin+cos', drop_time=True, scale=True)
    


## default_linear_learner


```python
Y_pred = model_test(Base=default_linear_learner(alpha=0.1),
           X_train=X_train, X_test=X_test,
           Y_train=Y_train, Y_test=Y_test,
          n_estimators=500, verbose_eval=100,
          plot_predict=True, return_y_pred=True)
Y_pred.name = 'default_linear_learner'
Pred_df = pd.concat([Pred_df, Y_pred], axis=1)
del Y_pred
```

    NGBRegressor(Base=Ridge(alpha=0.1, copy_X=True, fit_intercept=True,
                            max_iter=None, normalize=False, random_state=None,
                            solver='auto', tol=0.001),
                 Dist=<class 'ngboost.distns.normal.Normal'>,
                 Score=<class 'ngboost.scores.MLE'>, learning_rate=0.01,
                 minibatch_frac=1.0, n_estimators=500, natural_gradient=True,
                 tol=0.0001, verbose=True, verbose_eval=100) 
    
    [iter 0] loss=0.0536 val_loss=0.0000 scale=0.5000 norm=0.2703
    [iter 100] loss=-0.3445 val_loss=0.0000 scale=0.1250 norm=0.0572
    [iter 200] loss=-0.3701 val_loss=0.0000 scale=0.0312 norm=0.0139
    [iter 300] loss=-0.3717 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 400] loss=-0.3718 val_loss=0.0000 scale=0.0020 norm=0.0009
    
    Test MSE 0.03713388479853643
    Test NLL -0.2970978540733858



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
    
    [iter 0] loss=0.0536 val_loss=0.0000 scale=0.5000 norm=0.2703
    [iter 200] loss=-0.3027 val_loss=0.0000 scale=0.1250 norm=0.0481
    [iter 400] loss=-0.3134 val_loss=0.0000 scale=0.0156 norm=0.0059
    [iter 600] loss=-0.3139 val_loss=0.0000 scale=0.0039 norm=0.0015
    [iter 800] loss=-0.3139 val_loss=0.0000 scale=0.0005 norm=0.0002
    
    Test MSE 0.04328362249027503
    Test NLL -0.23415528342656414



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
    [iter 100] loss=0.0921 val_loss=0.0000 scale=0.0156 norm=0.0471
    [iter 200] loss=0.0896 val_loss=0.0000 scale=0.0039 norm=0.0121
    [iter 300] loss=0.0895 val_loss=0.0000 scale=0.0010 norm=0.0030
    [iter 400] loss=0.0895 val_loss=0.0000 scale=0.0002 norm=0.0008
    
    Test MSE 0.03148231334075362
    Test NLL -0.3345488208984631



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
    
    [iter 0] loss=0.0536 val_loss=0.0000 scale=1.0000 norm=0.5406
    [iter 100] loss=-0.2476 val_loss=0.0000 scale=0.2500 norm=0.1253
    [iter 200] loss=-0.2946 val_loss=0.0000 scale=0.0625 norm=0.0300
    [iter 300] loss=-0.3010 val_loss=0.0000 scale=0.0312 norm=0.0145
    [iter 400] loss=-0.3016 val_loss=0.0000 scale=0.0078 norm=0.0036
    
    Test MSE 0.02178728250308147
    Test NLL -0.3917017670634337



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
del Y_pred
```

    NGBRegressor(Base=KernelRidge(alpha=0.5, coef0=1, degree=3, gamma=None,
                                  kernel='poly', kernel_params=None),
                 Dist=<class 'ngboost.distns.normal.Normal'>,
                 Score=<class 'ngboost.scores.MLE'>, learning_rate=0.01,
                 minibatch_frac=1.0, n_estimators=500, natural_gradient=True,
                 tol=0.0001, verbose=True, verbose_eval=10) 
    
    [iter 0] loss=0.0536 val_loss=0.0000 scale=0.5000 norm=0.2703
    [iter 10] loss=-0.0550 val_loss=0.0000 scale=0.5000 norm=0.2315
    [iter 20] loss=-0.1182 val_loss=0.0000 scale=0.5000 norm=0.2167
    [iter 30] loss=-0.1647 val_loss=0.0000 scale=0.5000 norm=0.2111
    [iter 40] loss=-0.2032 val_loss=0.0000 scale=0.5000 norm=0.2099
    [iter 50] loss=-0.2367 val_loss=0.0000 scale=0.5000 norm=0.2111
    [iter 60] loss=-0.2535 val_loss=0.0000 scale=0.2500 norm=0.1061
    [iter 70] loss=-0.2677 val_loss=0.0000 scale=0.2500 norm=0.1067
    [iter 80] loss=-0.2810 val_loss=0.0000 scale=0.2500 norm=0.1073
    [iter 90] loss=-0.2933 val_loss=0.0000 scale=0.2500 norm=0.1078
    [iter 100] loss=-0.3045 val_loss=0.0000 scale=0.2500 norm=0.1082
    [iter 110] loss=-0.3135 val_loss=0.0000 scale=0.1250 norm=0.0543
    [iter 120] loss=-0.3181 val_loss=0.0000 scale=0.1250 norm=0.0543
    [iter 130] loss=-0.3223 val_loss=0.0000 scale=0.1250 norm=0.0543
    [iter 140] loss=-0.3261 val_loss=0.0000 scale=0.1250 norm=0.0543
    [iter 150] loss=-0.3296 val_loss=0.0000 scale=0.1250 norm=0.0543
    [iter 160] loss=-0.3327 val_loss=0.0000 scale=0.1250 norm=0.0542
    [iter 170] loss=-0.3351 val_loss=0.0000 scale=0.0625 norm=0.0270
    [iter 180] loss=-0.3363 val_loss=0.0000 scale=0.0625 norm=0.0270
    [iter 190] loss=-0.3374 val_loss=0.0000 scale=0.0625 norm=0.0270
    [iter 200] loss=-0.3384 val_loss=0.0000 scale=0.0625 norm=0.0269
    [iter 210] loss=-0.3392 val_loss=0.0000 scale=0.0625 norm=0.0269
    [iter 220] loss=-0.3400 val_loss=0.0000 scale=0.0625 norm=0.0268
    [iter 230] loss=-0.3406 val_loss=0.0000 scale=0.0312 norm=0.0134
    [iter 240] loss=-0.3409 val_loss=0.0000 scale=0.0312 norm=0.0134
    [iter 250] loss=-0.3412 val_loss=0.0000 scale=0.0312 norm=0.0134
    [iter 260] loss=-0.3414 val_loss=0.0000 scale=0.0312 norm=0.0133
    [iter 270] loss=-0.3416 val_loss=0.0000 scale=0.0312 norm=0.0133
    [iter 280] loss=-0.3418 val_loss=0.0000 scale=0.0312 norm=0.0133
    [iter 290] loss=-0.3419 val_loss=0.0000 scale=0.0156 norm=0.0067
    [iter 300] loss=-0.3420 val_loss=0.0000 scale=0.0156 norm=0.0066
    [iter 310] loss=-0.3421 val_loss=0.0000 scale=0.0156 norm=0.0066
    [iter 320] loss=-0.3421 val_loss=0.0000 scale=0.0156 norm=0.0066
    [iter 330] loss=-0.3422 val_loss=0.0000 scale=0.0156 norm=0.0066
    [iter 340] loss=-0.3422 val_loss=0.0000 scale=0.0156 norm=0.0066
    [iter 350] loss=-0.3422 val_loss=0.0000 scale=0.0078 norm=0.0033
    [iter 360] loss=-0.3422 val_loss=0.0000 scale=0.0078 norm=0.0033
    [iter 370] loss=-0.3423 val_loss=0.0000 scale=0.0078 norm=0.0033
    [iter 380] loss=-0.3423 val_loss=0.0000 scale=0.0078 norm=0.0033
    [iter 390] loss=-0.3423 val_loss=0.0000 scale=0.0078 norm=0.0033
    [iter 400] loss=-0.3423 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 410] loss=-0.3423 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 420] loss=-0.3423 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 430] loss=-0.3423 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 440] loss=-0.3423 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 450] loss=-0.3423 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 460] loss=-0.3423 val_loss=0.0000 scale=0.0020 norm=0.0008
    [iter 470] loss=-0.3423 val_loss=0.0000 scale=0.0020 norm=0.0008
    [iter 480] loss=-0.3423 val_loss=0.0000 scale=0.0020 norm=0.0008
    [iter 490] loss=-0.3423 val_loss=0.0000 scale=0.0020 norm=0.0008
    
    Test MSE 0.03835423687608188
    Test NLL -0.2842250617936376



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
del Y_pred
```

    NGBRegressor(Base=<ngboost.esn_learners.ESN_Ridge_learner object at 0x1a23062860>,
                 Dist=<class 'ngboost.distns.normal.Normal'>,
                 Score=<class 'ngboost.scores.MLE'>, learning_rate=0.01,
                 minibatch_frac=1.0, n_estimators=500, natural_gradient=True,
                 tol=0.0001, verbose=True, verbose_eval=10) 
    
    [iter 0] loss=0.0536 val_loss=0.0000 scale=0.5000 norm=0.2703
    [iter 10] loss=-0.0548 val_loss=0.0000 scale=0.5000 norm=0.2320
    [iter 20] loss=-0.1188 val_loss=0.0000 scale=0.5000 norm=0.2171
    [iter 30] loss=-0.1660 val_loss=0.0000 scale=0.5000 norm=0.2111
    [iter 40] loss=-0.2045 val_loss=0.0000 scale=0.5000 norm=0.2092
    [iter 50] loss=-0.2314 val_loss=0.0000 scale=0.2500 norm=0.1048
    [iter 60] loss=-0.2467 val_loss=0.0000 scale=0.2500 norm=0.1050
    [iter 70] loss=-0.2609 val_loss=0.0000 scale=0.2500 norm=0.1054
    [iter 80] loss=-0.2740 val_loss=0.0000 scale=0.2500 norm=0.1057
    [iter 90] loss=-0.2861 val_loss=0.0000 scale=0.2500 norm=0.1060
    [iter 100] loss=-0.2970 val_loss=0.0000 scale=0.2500 norm=0.1063
    [iter 110] loss=-0.3059 val_loss=0.0000 scale=0.1250 norm=0.0532
    [iter 120] loss=-0.3103 val_loss=0.0000 scale=0.1250 norm=0.0532
    [iter 130] loss=-0.3145 val_loss=0.0000 scale=0.1250 norm=0.0532
    [iter 140] loss=-0.3182 val_loss=0.0000 scale=0.1250 norm=0.0532
    [iter 150] loss=-0.3217 val_loss=0.0000 scale=0.1250 norm=0.0532
    [iter 160] loss=-0.3248 val_loss=0.0000 scale=0.1250 norm=0.0531
    [iter 170] loss=-0.3275 val_loss=0.0000 scale=0.0625 norm=0.0265
    [iter 180] loss=-0.3288 val_loss=0.0000 scale=0.0625 norm=0.0265
    [iter 190] loss=-0.3299 val_loss=0.0000 scale=0.0625 norm=0.0264
    [iter 200] loss=-0.3310 val_loss=0.0000 scale=0.0625 norm=0.0264
    [iter 210] loss=-0.3319 val_loss=0.0000 scale=0.0625 norm=0.0263
    [iter 220] loss=-0.3327 val_loss=0.0000 scale=0.0625 norm=0.0263
    [iter 230] loss=-0.3334 val_loss=0.0000 scale=0.0625 norm=0.0262
    [iter 240] loss=-0.3340 val_loss=0.0000 scale=0.0312 norm=0.0131
    [iter 250] loss=-0.3343 val_loss=0.0000 scale=0.0312 norm=0.0131
    [iter 260] loss=-0.3346 val_loss=0.0000 scale=0.0312 norm=0.0131
    [iter 270] loss=-0.3348 val_loss=0.0000 scale=0.0312 norm=0.0131
    [iter 280] loss=-0.3350 val_loss=0.0000 scale=0.0312 norm=0.0130
    [iter 290] loss=-0.3352 val_loss=0.0000 scale=0.0156 norm=0.0065
    [iter 300] loss=-0.3353 val_loss=0.0000 scale=0.0156 norm=0.0065
    [iter 310] loss=-0.3354 val_loss=0.0000 scale=0.0156 norm=0.0065
    [iter 320] loss=-0.3354 val_loss=0.0000 scale=0.0156 norm=0.0065
    [iter 330] loss=-0.3355 val_loss=0.0000 scale=0.0156 norm=0.0065
    [iter 340] loss=-0.3356 val_loss=0.0000 scale=0.0156 norm=0.0065
    [iter 350] loss=-0.3356 val_loss=0.0000 scale=0.0156 norm=0.0065
    [iter 360] loss=-0.3356 val_loss=0.0000 scale=0.0156 norm=0.0065
    [iter 370] loss=-0.3357 val_loss=0.0000 scale=0.0078 norm=0.0032
    [iter 380] loss=-0.3357 val_loss=0.0000 scale=0.0078 norm=0.0032
    [iter 390] loss=-0.3357 val_loss=0.0000 scale=0.0078 norm=0.0032
    [iter 400] loss=-0.3357 val_loss=0.0000 scale=0.0078 norm=0.0032
    [iter 410] loss=-0.3357 val_loss=0.0000 scale=0.0078 norm=0.0032
    [iter 420] loss=-0.3357 val_loss=0.0000 scale=0.0039 norm=0.0016
    [iter 430] loss=-0.3357 val_loss=0.0000 scale=0.0039 norm=0.0016
    [iter 440] loss=-0.3357 val_loss=0.0000 scale=0.0078 norm=0.0032
    [iter 450] loss=-0.3357 val_loss=0.0000 scale=0.0039 norm=0.0016
    [iter 460] loss=-0.3357 val_loss=0.0000 scale=0.0001 norm=0.0001
    [iter 470] loss=-0.3357 val_loss=0.0000 scale=0.0020 norm=0.0008
    [iter 480] loss=-0.3357 val_loss=0.0000 scale=0.0001 norm=0.0001
    [iter 490] loss=-0.3357 val_loss=0.0000 scale=0.0020 norm=0.0008
    
    Test MSE 0.036405167241842715
    Test NLL -0.29301947273185736



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
del Y_pred
```

    NGBRegressor(Base=<ngboost.esn_learners.ESN_kernel_ridge_learner object at 0x1a24a4cd30>,
                 Dist=<class 'ngboost.distns.normal.Normal'>,
                 Score=<class 'ngboost.scores.MLE'>, learning_rate=0.01,
                 minibatch_frac=1.0, n_estimators=500, natural_gradient=True,
                 tol=0.0001, verbose=True, verbose_eval=5) 
    
    [iter 0] loss=0.0536 val_loss=0.0000 scale=0.5000 norm=0.2703
    [iter 5] loss=-0.0056 val_loss=0.0000 scale=0.5000 norm=0.2494
    [iter 10] loss=-0.0499 val_loss=0.0000 scale=0.5000 norm=0.2363
    [iter 15] loss=-0.0851 val_loss=0.0000 scale=0.5000 norm=0.2279
    [iter 20] loss=-0.1145 val_loss=0.0000 scale=0.5000 norm=0.2227
    [iter 25] loss=-0.1402 val_loss=0.0000 scale=0.5000 norm=0.2193
    [iter 30] loss=-0.1632 val_loss=0.0000 scale=0.5000 norm=0.2174
    [iter 35] loss=-0.1841 val_loss=0.0000 scale=0.5000 norm=0.2166
    [iter 40] loss=-0.2035 val_loss=0.0000 scale=0.5000 norm=0.2164
    [iter 45] loss=-0.2215 val_loss=0.0000 scale=0.5000 norm=0.2167
    [iter 50] loss=-0.2384 val_loss=0.0000 scale=0.5000 norm=0.2175
    [iter 55] loss=-0.2480 val_loss=0.0000 scale=0.2500 norm=0.1090
    [iter 60] loss=-0.2557 val_loss=0.0000 scale=0.2500 norm=0.1093
    [iter 65] loss=-0.2630 val_loss=0.0000 scale=0.2500 norm=0.1095
    [iter 70] loss=-0.2702 val_loss=0.0000 scale=0.2500 norm=0.1098
    [iter 75] loss=-0.2770 val_loss=0.0000 scale=0.2500 norm=0.1101
    [iter 80] loss=-0.2835 val_loss=0.0000 scale=0.2500 norm=0.1103
    [iter 85] loss=-0.2897 val_loss=0.0000 scale=0.2500 norm=0.1106
    [iter 90] loss=-0.2956 val_loss=0.0000 scale=0.2500 norm=0.1108
    [iter 95] loss=-0.3012 val_loss=0.0000 scale=0.2500 norm=0.1110
    [iter 100] loss=-0.3064 val_loss=0.0000 scale=0.2500 norm=0.1112
    [iter 105] loss=-0.3114 val_loss=0.0000 scale=0.1250 norm=0.0556
    [iter 110] loss=-0.3137 val_loss=0.0000 scale=0.1250 norm=0.0557
    [iter 115] loss=-0.3159 val_loss=0.0000 scale=0.1250 norm=0.0557
    [iter 120] loss=-0.3180 val_loss=0.0000 scale=0.1250 norm=0.0557
    [iter 125] loss=-0.3201 val_loss=0.0000 scale=0.1250 norm=0.0557
    [iter 130] loss=-0.3220 val_loss=0.0000 scale=0.1250 norm=0.0557
    [iter 135] loss=-0.3239 val_loss=0.0000 scale=0.1250 norm=0.0557
    [iter 140] loss=-0.3257 val_loss=0.0000 scale=0.1250 norm=0.0557
    [iter 145] loss=-0.3273 val_loss=0.0000 scale=0.1250 norm=0.0557
    [iter 150] loss=-0.3289 val_loss=0.0000 scale=0.1250 norm=0.0556
    [iter 155] loss=-0.3304 val_loss=0.0000 scale=0.1250 norm=0.0556
    [iter 160] loss=-0.3317 val_loss=0.0000 scale=0.1250 norm=0.0555
    [iter 165] loss=-0.3326 val_loss=0.0000 scale=0.0625 norm=0.0278
    [iter 170] loss=-0.3332 val_loss=0.0000 scale=0.0625 norm=0.0277
    [iter 175] loss=-0.3337 val_loss=0.0000 scale=0.0625 norm=0.0277
    [iter 180] loss=-0.3343 val_loss=0.0000 scale=0.0625 norm=0.0277
    [iter 185] loss=-0.3348 val_loss=0.0000 scale=0.0625 norm=0.0277
    [iter 190] loss=-0.3353 val_loss=0.0000 scale=0.0625 norm=0.0277
    [iter 195] loss=-0.3358 val_loss=0.0000 scale=0.0625 norm=0.0277
    [iter 200] loss=-0.3362 val_loss=0.0000 scale=0.0625 norm=0.0276
    [iter 205] loss=-0.3366 val_loss=0.0000 scale=0.0625 norm=0.0276
    [iter 210] loss=-0.3370 val_loss=0.0000 scale=0.0625 norm=0.0276
    [iter 215] loss=-0.3374 val_loss=0.0000 scale=0.0625 norm=0.0276
    [iter 220] loss=-0.3377 val_loss=0.0000 scale=0.0312 norm=0.0138
    [iter 225] loss=-0.3380 val_loss=0.0000 scale=0.0312 norm=0.0138
    [iter 230] loss=-0.3381 val_loss=0.0000 scale=0.0312 norm=0.0138
    [iter 235] loss=-0.3382 val_loss=0.0000 scale=0.0312 norm=0.0137
    [iter 240] loss=-0.3384 val_loss=0.0000 scale=0.0312 norm=0.0137
    [iter 245] loss=-0.3385 val_loss=0.0000 scale=0.0312 norm=0.0137
    [iter 250] loss=-0.3386 val_loss=0.0000 scale=0.0312 norm=0.0137
    [iter 255] loss=-0.3387 val_loss=0.0000 scale=0.0312 norm=0.0137
    [iter 260] loss=-0.3388 val_loss=0.0000 scale=0.0312 norm=0.0137
    [iter 265] loss=-0.3389 val_loss=0.0000 scale=0.0312 norm=0.0137
    [iter 270] loss=-0.3390 val_loss=0.0000 scale=0.0312 norm=0.0137
    [iter 275] loss=-0.3391 val_loss=0.0000 scale=0.0312 norm=0.0137
    [iter 280] loss=-0.3391 val_loss=0.0000 scale=0.0312 norm=0.0137
    [iter 285] loss=-0.3392 val_loss=0.0000 scale=0.0312 norm=0.0137
    [iter 290] loss=-0.3392 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 295] loss=-0.3392 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 300] loss=-0.3393 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 305] loss=-0.3393 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 310] loss=-0.3393 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 315] loss=-0.3393 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 320] loss=-0.3394 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 325] loss=-0.3394 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 330] loss=-0.3394 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 335] loss=-0.3394 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 340] loss=-0.3394 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 345] loss=-0.3394 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 350] loss=-0.3394 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 355] loss=-0.3395 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 360] loss=-0.3395 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 365] loss=-0.3395 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 370] loss=-0.3395 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 375] loss=-0.3395 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 380] loss=-0.3395 val_loss=0.0000 scale=0.0020 norm=0.0009
    [iter 385] loss=-0.3395 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 390] loss=-0.3395 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 395] loss=-0.3395 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 400] loss=-0.3395 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 405] loss=-0.3395 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 410] loss=-0.3395 val_loss=0.0000 scale=0.0020 norm=0.0009
    [iter 415] loss=-0.3395 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 420] loss=-0.3395 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 425] loss=-0.3395 val_loss=0.0000 scale=0.0001 norm=0.0001
    [iter 430] loss=-0.3395 val_loss=0.0000 scale=0.0020 norm=0.0009
    [iter 435] loss=-0.3395 val_loss=0.0000 scale=0.0020 norm=0.0009
    [iter 440] loss=-0.3395 val_loss=0.0000 scale=0.0010 norm=0.0004
    [iter 445] loss=-0.3395 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 450] loss=-0.3395 val_loss=0.0000 scale=0.0010 norm=0.0004
    [iter 455] loss=-0.3395 val_loss=0.0000 scale=0.0001 norm=0.0001
    [iter 460] loss=-0.3395 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 465] loss=-0.3395 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 470] loss=-0.3395 val_loss=0.0000 scale=0.0020 norm=0.0009
    [iter 475] loss=-0.3395 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 480] loss=-0.3395 val_loss=0.0000 scale=0.0020 norm=0.0008
    [iter 485] loss=-0.3395 val_loss=0.0000 scale=0.0020 norm=0.0008
    [iter 490] loss=-0.3395 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 495] loss=-0.3395 val_loss=0.0000 scale=0.0001 norm=0.0001
    
    Test MSE 0.03862410587577653
    Test NLL -0.28260315662842006



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
del Y_pred
```

    NGBRegressor(Base=<ngboost.esn_learners.ESN_linear_svr_learner object at 0x1a2237d8d0>,
                 Dist=<class 'ngboost.distns.normal.Normal'>,
                 Score=<class 'ngboost.scores.MLE'>, learning_rate=0.01,
                 minibatch_frac=1.0, n_estimators=500, natural_gradient=True,
                 tol=0.0001, verbose=True, verbose_eval=10) 
    
    [iter 0] loss=0.0536 val_loss=0.0000 scale=1.0000 norm=0.5406
    [iter 10] loss=-0.0331 val_loss=0.0000 scale=0.5000 norm=0.2462
    [iter 20] loss=-0.0827 val_loss=0.0000 scale=0.5000 norm=0.2374
    [iter 30] loss=-0.1266 val_loss=0.0000 scale=0.5000 norm=0.2315
    [iter 40] loss=-0.1626 val_loss=0.0000 scale=0.5000 norm=0.2297
    [iter 50] loss=-0.1863 val_loss=0.0000 scale=0.2500 norm=0.1148
    [iter 60] loss=-0.2006 val_loss=0.0000 scale=0.2500 norm=0.1150
    [iter 70] loss=-0.2135 val_loss=0.0000 scale=0.2500 norm=0.1153
    [iter 80] loss=-0.2254 val_loss=0.0000 scale=0.2500 norm=0.1156
    [iter 90] loss=-0.2363 val_loss=0.0000 scale=0.2500 norm=0.1160
    [iter 100] loss=-0.2456 val_loss=0.0000 scale=0.1250 norm=0.0581
    [iter 110] loss=-0.2505 val_loss=0.0000 scale=0.1250 norm=0.0582
    [iter 120] loss=-0.2547 val_loss=0.0000 scale=0.1250 norm=0.0583


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 130] loss=-0.2586 val_loss=0.0000 scale=0.1250 norm=0.0583
    [iter 140] loss=-0.2622 val_loss=0.0000 scale=0.1250 norm=0.0583
    [iter 150] loss=-0.2656 val_loss=0.0000 scale=0.1250 norm=0.0583
    [iter 160] loss=-0.2686 val_loss=0.0000 scale=0.1250 norm=0.0583
    [iter 170] loss=-0.2710 val_loss=0.0000 scale=0.1250 norm=0.0583
    [iter 180] loss=-0.2727 val_loss=0.0000 scale=0.0625 norm=0.0291
    [iter 190] loss=-0.2738 val_loss=0.0000 scale=0.0625 norm=0.0291
    [iter 200] loss=-0.2749 val_loss=0.0000 scale=0.0625 norm=0.0291
    [iter 210] loss=-0.2759 val_loss=0.0000 scale=0.0625 norm=0.0290
    [iter 220] loss=-0.2768 val_loss=0.0000 scale=0.0625 norm=0.0290
    [iter 230] loss=-0.2776 val_loss=0.0000 scale=0.0625 norm=0.0290
    [iter 240] loss=-0.2784 val_loss=0.0000 scale=0.0625 norm=0.0289
    [iter 250] loss=-0.2789 val_loss=0.0000 scale=0.0625 norm=0.0289


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 260] loss=-0.2793 val_loss=0.0000 scale=0.0625 norm=0.0289
    [iter 270] loss=-0.2797 val_loss=0.0000 scale=0.0312 norm=0.0144
    [iter 280] loss=-0.2799 val_loss=0.0000 scale=0.0312 norm=0.0144
    [iter 290] loss=-0.2801 val_loss=0.0000 scale=0.0625 norm=0.0288


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 300] loss=-0.2804 val_loss=0.0000 scale=0.0156 norm=0.0072
    [iter 310] loss=-0.2805 val_loss=0.0000 scale=0.0156 norm=0.0072
    [iter 320] loss=-0.2807 val_loss=0.0000 scale=0.0156 norm=0.0072
    [iter 330] loss=-0.2808 val_loss=0.0000 scale=0.0156 norm=0.0072
    [iter 340] loss=-0.2809 val_loss=0.0000 scale=0.0156 norm=0.0072
    [iter 350] loss=-0.2809 val_loss=0.0000 scale=0.0156 norm=0.0072
    [iter 360] loss=-0.2810 val_loss=0.0000 scale=0.0156 norm=0.0072
    [iter 370] loss=-0.2810 val_loss=0.0000 scale=0.0156 norm=0.0072
    [iter 380] loss=-0.2810 val_loss=0.0000 scale=0.0156 norm=0.0072
    [iter 390] loss=-0.2811 val_loss=0.0000 scale=0.0156 norm=0.0072


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 400] loss=-0.2811 val_loss=0.0000 scale=0.0156 norm=0.0071
    [iter 410] loss=-0.2811 val_loss=0.0000 scale=0.0312 norm=0.0143
    [iter 420] loss=-0.2812 val_loss=0.0000 scale=0.0156 norm=0.0071
    [iter 430] loss=-0.2812 val_loss=0.0000 scale=0.0039 norm=0.0018
    [iter 440] loss=-0.2812 val_loss=0.0000 scale=0.0020 norm=0.0009
    [iter 450] loss=-0.2812 val_loss=0.0000 scale=0.0001 norm=0.0001
    [iter 460] loss=-0.2812 val_loss=0.0000 scale=0.0078 norm=0.0036
    [iter 470] loss=-0.2813 val_loss=0.0000 scale=0.0156 norm=0.0071
    [iter 480] loss=-0.2813 val_loss=0.0000 scale=0.0001 norm=0.0001
    [iter 490] loss=-0.2813 val_loss=0.0000 scale=0.0078 norm=0.0036
    
    Test MSE 0.036098413254226504
    Test NLL -0.30795730608864474



![png](output_24_9.png)



```python
Pred_df.to_csv('/Users/apple/Documents/ML_Project/ML - 2.1/result/'+transform+'-3.csv')
```
