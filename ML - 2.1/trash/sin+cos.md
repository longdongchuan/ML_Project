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
X_train, X_test, Y_train, Y_test = get_data(hour_num=0, transform=transform,
                                            drop_time=True, scale=True)
Pred_df = Y_test
```

    get_data(hour_num=0, transform='sin+cos', drop_time=True, scale=True)
    


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
    
    [iter 0] loss=0.0535 val_loss=0.0000 scale=0.5000 norm=0.2703
    [iter 100] loss=-0.2825 val_loss=0.0000 scale=0.1250 norm=0.0631
    [iter 200] loss=-0.3046 val_loss=0.0000 scale=0.0312 norm=0.0159
    [iter 300] loss=-0.3059 val_loss=0.0000 scale=0.0078 norm=0.0040
    [iter 400] loss=-0.3060 val_loss=0.0000 scale=0.0020 norm=0.0010
    
    Test MSE 0.024902636629026444
    Test NLL -0.3565953933102744



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
    
    [iter 0] loss=0.0535 val_loss=0.0000 scale=0.5000 norm=0.2703
    [iter 200] loss=-0.2979 val_loss=0.0000 scale=0.0625 norm=0.0283
    [iter 400] loss=-0.3055 val_loss=0.0000 scale=0.0078 norm=0.0035
    [iter 600] loss=-0.3057 val_loss=0.0000 scale=0.0020 norm=0.0009
    [iter 800] loss=-0.3057 val_loss=0.0000 scale=0.0002 norm=0.0001
    
    Test MSE 0.042196495413924104
    Test NLL -0.224710780783675



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
    
    [iter 0] loss=0.1430 val_loss=0.0000 scale=0.2500 norm=0.6104
    [iter 100] loss=0.0993 val_loss=0.0000 scale=0.0156 norm=0.0492
    [iter 200] loss=0.0975 val_loss=0.0000 scale=0.0039 norm=0.0128
    [iter 300] loss=0.0974 val_loss=0.0000 scale=0.0010 norm=0.0032
    [iter 400] loss=0.0974 val_loss=0.0000 scale=0.0002 norm=0.0008
    
    Test MSE 0.021723381197709522
    Test NLL -0.37213340352302826



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
    
    [iter 0] loss=0.0535 val_loss=0.0000 scale=0.5000 norm=0.2703
    [iter 100] loss=-0.1015 val_loss=0.0000 scale=0.2500 norm=0.1586
    [iter 200] loss=-0.1446 val_loss=0.0000 scale=0.0625 norm=0.0397
    [iter 300] loss=-0.1504 val_loss=0.0000 scale=0.0156 norm=0.0098
    [iter 400] loss=-0.1507 val_loss=0.0000 scale=0.0039 norm=0.0024
    
    Test MSE 0.020059426550962392
    Test NLL -0.37803360183299034



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
    
    [iter 0] loss=0.0535 val_loss=0.0000 scale=1.0000 norm=0.5407
    [iter 10] loss=-0.0832 val_loss=0.0000 scale=1.0000 norm=0.4721
    [iter 20] loss=-0.1328 val_loss=0.0000 scale=0.5000 norm=0.2341
    [iter 30] loss=-0.1683 val_loss=0.0000 scale=0.5000 norm=0.2363
    [iter 40] loss=-0.2004 val_loss=0.0000 scale=0.5000 norm=0.2404
    [iter 50] loss=-0.2252 val_loss=0.0000 scale=0.2500 norm=0.1223
    [iter 60] loss=-0.2387 val_loss=0.0000 scale=0.2500 norm=0.1235
    [iter 70] loss=-0.2512 val_loss=0.0000 scale=0.2500 norm=0.1247
    [iter 80] loss=-0.2626 val_loss=0.0000 scale=0.2500 norm=0.1259
    [iter 90] loss=-0.2728 val_loss=0.0000 scale=0.2500 norm=0.1269
    [iter 100] loss=-0.2791 val_loss=0.0000 scale=0.1250 norm=0.0637
    [iter 110] loss=-0.2832 val_loss=0.0000 scale=0.1250 norm=0.0639
    [iter 120] loss=-0.2869 val_loss=0.0000 scale=0.1250 norm=0.0641
    [iter 130] loss=-0.2902 val_loss=0.0000 scale=0.1250 norm=0.0642
    [iter 140] loss=-0.2931 val_loss=0.0000 scale=0.1250 norm=0.0643
    [iter 150] loss=-0.2952 val_loss=0.0000 scale=0.0625 norm=0.0322
    [iter 160] loss=-0.2962 val_loss=0.0000 scale=0.0625 norm=0.0322
    [iter 170] loss=-0.2972 val_loss=0.0000 scale=0.0625 norm=0.0322
    [iter 180] loss=-0.2981 val_loss=0.0000 scale=0.0625 norm=0.0322
    [iter 190] loss=-0.2988 val_loss=0.0000 scale=0.0625 norm=0.0322
    [iter 200] loss=-0.2994 val_loss=0.0000 scale=0.0312 norm=0.0161
    [iter 210] loss=-0.2997 val_loss=0.0000 scale=0.0312 norm=0.0161
    [iter 220] loss=-0.2999 val_loss=0.0000 scale=0.0312 norm=0.0161
    [iter 230] loss=-0.3001 val_loss=0.0000 scale=0.0312 norm=0.0161
    [iter 240] loss=-0.3003 val_loss=0.0000 scale=0.0312 norm=0.0161
    [iter 250] loss=-0.3005 val_loss=0.0000 scale=0.0156 norm=0.0080
    [iter 260] loss=-0.3005 val_loss=0.0000 scale=0.0156 norm=0.0080
    [iter 270] loss=-0.3006 val_loss=0.0000 scale=0.0156 norm=0.0080
    [iter 280] loss=-0.3006 val_loss=0.0000 scale=0.0156 norm=0.0080
    [iter 290] loss=-0.3007 val_loss=0.0000 scale=0.0156 norm=0.0080
    [iter 300] loss=-0.3007 val_loss=0.0000 scale=0.0078 norm=0.0040
    [iter 310] loss=-0.3007 val_loss=0.0000 scale=0.0078 norm=0.0040
    [iter 320] loss=-0.3008 val_loss=0.0000 scale=0.0078 norm=0.0040
    [iter 330] loss=-0.3008 val_loss=0.0000 scale=0.0078 norm=0.0040
    [iter 340] loss=-0.3008 val_loss=0.0000 scale=0.0078 norm=0.0040
    [iter 350] loss=-0.3008 val_loss=0.0000 scale=0.0039 norm=0.0020
    [iter 360] loss=-0.3008 val_loss=0.0000 scale=0.0039 norm=0.0020
    [iter 370] loss=-0.3008 val_loss=0.0000 scale=0.0039 norm=0.0020
    [iter 380] loss=-0.3008 val_loss=0.0000 scale=0.0039 norm=0.0020
    [iter 390] loss=-0.3008 val_loss=0.0000 scale=0.0039 norm=0.0020
    [iter 400] loss=-0.3008 val_loss=0.0000 scale=0.0020 norm=0.0010
    [iter 410] loss=-0.3008 val_loss=0.0000 scale=0.0020 norm=0.0010
    [iter 420] loss=-0.3008 val_loss=0.0000 scale=0.0020 norm=0.0010
    [iter 430] loss=-0.3008 val_loss=0.0000 scale=0.0020 norm=0.0010
    [iter 440] loss=-0.3008 val_loss=0.0000 scale=0.0020 norm=0.0010
    [iter 450] loss=-0.3008 val_loss=0.0000 scale=0.0010 norm=0.0005
    [iter 460] loss=-0.3008 val_loss=0.0000 scale=0.0010 norm=0.0005
    [iter 470] loss=-0.3008 val_loss=0.0000 scale=0.0010 norm=0.0005
    [iter 480] loss=-0.3008 val_loss=0.0000 scale=0.0010 norm=0.0005
    [iter 490] loss=-0.3008 val_loss=0.0000 scale=0.0010 norm=0.0005
    
    Test MSE 0.030964958061800198
    Test NLL -0.3139016653566659



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

    NGBRegressor(Base=<ngboost.esn_learners.ESN_Ridge_learner object at 0x1a229e8940>,
                 Dist=<class 'ngboost.distns.normal.Normal'>,
                 Score=<class 'ngboost.scores.MLE'>, learning_rate=0.01,
                 minibatch_frac=1.0, n_estimators=500, natural_gradient=True,
                 tol=0.0001, verbose=True, verbose_eval=10) 
    
    [iter 0] loss=0.0535 val_loss=0.0000 scale=1.0000 norm=0.5407
    [iter 10] loss=-0.0781 val_loss=0.0000 scale=0.5000 norm=0.2345
    [iter 20] loss=-0.1226 val_loss=0.0000 scale=0.5000 norm=0.2305
    [iter 30] loss=-0.1602 val_loss=0.0000 scale=0.5000 norm=0.2309
    [iter 40] loss=-0.1938 val_loss=0.0000 scale=0.5000 norm=0.2335
    [iter 50] loss=-0.2240 val_loss=0.0000 scale=0.2500 norm=0.1187
    [iter 60] loss=-0.2381 val_loss=0.0000 scale=0.2500 norm=0.1198
    [iter 70] loss=-0.2514 val_loss=0.0000 scale=0.2500 norm=0.1208
    [iter 80] loss=-0.2635 val_loss=0.0000 scale=0.2500 norm=0.1218
    [iter 90] loss=-0.2747 val_loss=0.0000 scale=0.2500 norm=0.1226
    [iter 100] loss=-0.2846 val_loss=0.0000 scale=0.2500 norm=0.1234
    [iter 110] loss=-0.2895 val_loss=0.0000 scale=0.1250 norm=0.0618
    [iter 120] loss=-0.2936 val_loss=0.0000 scale=0.1250 norm=0.0620
    [iter 130] loss=-0.2973 val_loss=0.0000 scale=0.1250 norm=0.0621
    [iter 140] loss=-0.3006 val_loss=0.0000 scale=0.1250 norm=0.0621
    [iter 150] loss=-0.3035 val_loss=0.0000 scale=0.1250 norm=0.0621
    [iter 160] loss=-0.3056 val_loss=0.0000 scale=0.0625 norm=0.0311
    [iter 170] loss=-0.3067 val_loss=0.0000 scale=0.0625 norm=0.0311
    [iter 180] loss=-0.3077 val_loss=0.0000 scale=0.0625 norm=0.0311
    [iter 190] loss=-0.3086 val_loss=0.0000 scale=0.0625 norm=0.0310
    [iter 200] loss=-0.3094 val_loss=0.0000 scale=0.0625 norm=0.0310
    [iter 210] loss=-0.3100 val_loss=0.0000 scale=0.0312 norm=0.0155
    [iter 220] loss=-0.3104 val_loss=0.0000 scale=0.0312 norm=0.0155
    [iter 230] loss=-0.3106 val_loss=0.0000 scale=0.0312 norm=0.0155
    [iter 240] loss=-0.3109 val_loss=0.0000 scale=0.0312 norm=0.0155
    [iter 250] loss=-0.3111 val_loss=0.0000 scale=0.0312 norm=0.0155
    [iter 260] loss=-0.3112 val_loss=0.0000 scale=0.0312 norm=0.0155
    [iter 270] loss=-0.3113 val_loss=0.0000 scale=0.0312 norm=0.0154
    [iter 280] loss=-0.3114 val_loss=0.0000 scale=0.0156 norm=0.0077
    [iter 290] loss=-0.3115 val_loss=0.0000 scale=0.0156 norm=0.0077
    [iter 300] loss=-0.3115 val_loss=0.0000 scale=0.0156 norm=0.0077
    [iter 310] loss=-0.3116 val_loss=0.0000 scale=0.0078 norm=0.0039
    [iter 320] loss=-0.3116 val_loss=0.0000 scale=0.0078 norm=0.0039
    [iter 330] loss=-0.3116 val_loss=0.0000 scale=0.0156 norm=0.0077
    [iter 340] loss=-0.3116 val_loss=0.0000 scale=0.0078 norm=0.0039
    [iter 350] loss=-0.3116 val_loss=0.0000 scale=0.0039 norm=0.0019
    [iter 360] loss=-0.3116 val_loss=0.0000 scale=0.0156 norm=0.0077
    [iter 370] loss=-0.3116 val_loss=0.0000 scale=0.0078 norm=0.0039
    [iter 380] loss=-0.3116 val_loss=0.0000 scale=0.0039 norm=0.0019
    [iter 390] loss=-0.3117 val_loss=0.0000 scale=0.0020 norm=0.0010
    [iter 400] loss=-0.3117 val_loss=0.0000 scale=0.0039 norm=0.0019
    [iter 410] loss=-0.3117 val_loss=0.0000 scale=0.0039 norm=0.0019
    [iter 420] loss=-0.3117 val_loss=0.0000 scale=0.0039 norm=0.0019
    [iter 430] loss=-0.3117 val_loss=0.0000 scale=0.0039 norm=0.0019
    [iter 440] loss=-0.3117 val_loss=0.0000 scale=0.0039 norm=0.0019
    [iter 450] loss=-0.3117 val_loss=0.0000 scale=0.0039 norm=0.0019
    [iter 460] loss=-0.3117 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 470] loss=-0.3117 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 480] loss=-0.3117 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 490] loss=-0.3117 val_loss=0.0000 scale=0.0002 norm=0.0001
    
    Test MSE 0.0386812205016771
    Test NLL -0.20830790066114968



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

    NGBRegressor(Base=<ngboost.esn_learners.ESN_kernel_ridge_learner object at 0x1a229e2320>,
                 Dist=<class 'ngboost.distns.normal.Normal'>,
                 Score=<class 'ngboost.scores.MLE'>, learning_rate=0.01,
                 minibatch_frac=1.0, n_estimators=500, natural_gradient=True,
                 tol=0.0001, verbose=True, verbose_eval=5) 
    
    [iter 0] loss=0.0535 val_loss=0.0000 scale=0.5000 norm=0.2703
    [iter 5] loss=-0.0197 val_loss=0.0000 scale=1.0000 norm=0.5010
    [iter 10] loss=-0.0623 val_loss=0.0000 scale=0.5000 norm=0.2429
    [iter 15] loss=-0.0886 val_loss=0.0000 scale=0.5000 norm=0.2399
    [iter 20] loss=-0.1124 val_loss=0.0000 scale=0.5000 norm=0.2385
    [iter 25] loss=-0.1343 val_loss=0.0000 scale=0.5000 norm=0.2381
    [iter 30] loss=-0.1547 val_loss=0.0000 scale=0.5000 norm=0.2385
    [iter 35] loss=-0.1738 val_loss=0.0000 scale=0.5000 norm=0.2396
    [iter 40] loss=-0.1918 val_loss=0.0000 scale=0.5000 norm=0.2411
    [iter 45] loss=-0.2089 val_loss=0.0000 scale=0.5000 norm=0.2429
    [iter 50] loss=-0.2232 val_loss=0.0000 scale=0.2500 norm=0.1223
    [iter 55] loss=-0.2308 val_loss=0.0000 scale=0.2500 norm=0.1229
    [iter 60] loss=-0.2381 val_loss=0.0000 scale=0.2500 norm=0.1234
    [iter 65] loss=-0.2450 val_loss=0.0000 scale=0.2500 norm=0.1239
    [iter 70] loss=-0.2517 val_loss=0.0000 scale=0.2500 norm=0.1244
    [iter 75] loss=-0.2579 val_loss=0.0000 scale=0.2500 norm=0.1249
    [iter 80] loss=-0.2638 val_loss=0.0000 scale=0.2500 norm=0.1254
    [iter 85] loss=-0.2695 val_loss=0.0000 scale=0.2500 norm=0.1258
    [iter 90] loss=-0.2747 val_loss=0.0000 scale=0.2500 norm=0.1262
    [iter 95] loss=-0.2795 val_loss=0.0000 scale=0.2500 norm=0.1266
    [iter 100] loss=-0.2823 val_loss=0.0000 scale=0.1250 norm=0.0634
    [iter 105] loss=-0.2844 val_loss=0.0000 scale=0.1250 norm=0.0634
    [iter 110] loss=-0.2865 val_loss=0.0000 scale=0.1250 norm=0.0635
    [iter 115] loss=-0.2884 val_loss=0.0000 scale=0.1250 norm=0.0636
    [iter 120] loss=-0.2903 val_loss=0.0000 scale=0.1250 norm=0.0636
    [iter 125] loss=-0.2920 val_loss=0.0000 scale=0.1250 norm=0.0637
    [iter 130] loss=-0.2936 val_loss=0.0000 scale=0.1250 norm=0.0637
    [iter 135] loss=-0.2951 val_loss=0.0000 scale=0.1250 norm=0.0638
    [iter 140] loss=-0.2965 val_loss=0.0000 scale=0.1250 norm=0.0638
    [iter 145] loss=-0.2978 val_loss=0.0000 scale=0.1250 norm=0.0638
    [iter 150] loss=-0.2988 val_loss=0.0000 scale=0.0625 norm=0.0319
    [iter 155] loss=-0.2994 val_loss=0.0000 scale=0.0625 norm=0.0319
    [iter 160] loss=-0.2999 val_loss=0.0000 scale=0.0625 norm=0.0319
    [iter 165] loss=-0.3004 val_loss=0.0000 scale=0.0625 norm=0.0319
    [iter 170] loss=-0.3009 val_loss=0.0000 scale=0.0625 norm=0.0319
    [iter 175] loss=-0.3013 val_loss=0.0000 scale=0.0625 norm=0.0319
    [iter 180] loss=-0.3017 val_loss=0.0000 scale=0.0625 norm=0.0319
    [iter 185] loss=-0.3021 val_loss=0.0000 scale=0.0625 norm=0.0319
    [iter 190] loss=-0.3025 val_loss=0.0000 scale=0.0625 norm=0.0319
    [iter 195] loss=-0.3028 val_loss=0.0000 scale=0.0625 norm=0.0319
    [iter 200] loss=-0.3030 val_loss=0.0000 scale=0.0625 norm=0.0319
    [iter 205] loss=-0.3032 val_loss=0.0000 scale=0.0312 norm=0.0159
    [iter 210] loss=-0.3033 val_loss=0.0000 scale=0.0312 norm=0.0159
    [iter 215] loss=-0.3034 val_loss=0.0000 scale=0.0312 norm=0.0159
    [iter 220] loss=-0.3036 val_loss=0.0000 scale=0.0312 norm=0.0159
    [iter 225] loss=-0.3037 val_loss=0.0000 scale=0.0312 norm=0.0159
    [iter 230] loss=-0.3037 val_loss=0.0000 scale=0.0312 norm=0.0159
    [iter 235] loss=-0.3038 val_loss=0.0000 scale=0.0312 norm=0.0159
    [iter 240] loss=-0.3039 val_loss=0.0000 scale=0.0312 norm=0.0159
    [iter 245] loss=-0.3040 val_loss=0.0000 scale=0.0312 norm=0.0159
    [iter 250] loss=-0.3040 val_loss=0.0000 scale=0.0156 norm=0.0080
    [iter 255] loss=-0.3041 val_loss=0.0000 scale=0.0156 norm=0.0080
    [iter 260] loss=-0.3041 val_loss=0.0000 scale=0.0156 norm=0.0080
    [iter 265] loss=-0.3041 val_loss=0.0000 scale=0.0156 norm=0.0080
    [iter 270] loss=-0.3042 val_loss=0.0000 scale=0.0156 norm=0.0080
    [iter 275] loss=-0.3042 val_loss=0.0000 scale=0.0156 norm=0.0080
    [iter 280] loss=-0.3042 val_loss=0.0000 scale=0.0156 norm=0.0080
    [iter 285] loss=-0.3042 val_loss=0.0000 scale=0.0156 norm=0.0080
    [iter 290] loss=-0.3043 val_loss=0.0000 scale=0.0078 norm=0.0040
    [iter 295] loss=-0.3043 val_loss=0.0000 scale=0.0156 norm=0.0080
    [iter 300] loss=-0.3043 val_loss=0.0000 scale=0.0078 norm=0.0040
    [iter 305] loss=-0.3043 val_loss=0.0000 scale=0.0078 norm=0.0040
    [iter 310] loss=-0.3043 val_loss=0.0000 scale=0.0078 norm=0.0040
    [iter 315] loss=-0.3043 val_loss=0.0000 scale=0.0078 norm=0.0040
    [iter 320] loss=-0.3043 val_loss=0.0000 scale=0.0039 norm=0.0020
    [iter 325] loss=-0.3043 val_loss=0.0000 scale=0.0078 norm=0.0040
    [iter 330] loss=-0.3043 val_loss=0.0000 scale=0.0039 norm=0.0020
    [iter 335] loss=-0.3043 val_loss=0.0000 scale=0.0078 norm=0.0040
    [iter 340] loss=-0.3043 val_loss=0.0000 scale=0.0039 norm=0.0020
    [iter 345] loss=-0.3043 val_loss=0.0000 scale=0.0078 norm=0.0040
    [iter 350] loss=-0.3043 val_loss=0.0000 scale=0.0039 norm=0.0020
    [iter 355] loss=-0.3044 val_loss=0.0000 scale=0.0039 norm=0.0020
    [iter 360] loss=-0.3044 val_loss=0.0000 scale=0.0020 norm=0.0010
    [iter 365] loss=-0.3044 val_loss=0.0000 scale=0.0039 norm=0.0020
    [iter 370] loss=-0.3044 val_loss=0.0000 scale=0.0078 norm=0.0040
    [iter 375] loss=-0.3044 val_loss=0.0000 scale=0.0020 norm=0.0010
    [iter 380] loss=-0.3044 val_loss=0.0000 scale=0.0005 norm=0.0002
    [iter 385] loss=-0.3044 val_loss=0.0000 scale=0.0039 norm=0.0020
    [iter 390] loss=-0.3044 val_loss=0.0000 scale=0.0010 norm=0.0005
    [iter 395] loss=-0.3044 val_loss=0.0000 scale=0.0039 norm=0.0020
    [iter 400] loss=-0.3044 val_loss=0.0000 scale=0.0020 norm=0.0010
    [iter 405] loss=-0.3044 val_loss=0.0000 scale=0.0039 norm=0.0020
    [iter 410] loss=-0.3044 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 415] loss=-0.3044 val_loss=0.0000 scale=0.0039 norm=0.0020
    [iter 420] loss=-0.3044 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 425] loss=-0.3044 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 430] loss=-0.3044 val_loss=0.0000 scale=0.0039 norm=0.0020
    [iter 435] loss=-0.3044 val_loss=0.0000 scale=0.0020 norm=0.0010
    [iter 440] loss=-0.3044 val_loss=0.0000 scale=0.0020 norm=0.0010
    [iter 445] loss=-0.3044 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 450] loss=-0.3044 val_loss=0.0000 scale=0.0005 norm=0.0002
    [iter 455] loss=-0.3044 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 460] loss=-0.3044 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 465] loss=-0.3044 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 470] loss=-0.3044 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 475] loss=-0.3044 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 480] loss=-0.3044 val_loss=0.0000 scale=0.0078 norm=0.0040
    [iter 485] loss=-0.3044 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 490] loss=-0.3044 val_loss=0.0000 scale=0.0020 norm=0.0010
    [iter 495] loss=-0.3044 val_loss=0.0000 scale=0.0002 norm=0.0001
    
    Test MSE 0.023998196293202663
    Test NLL -0.36135169330173



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

    NGBRegressor(Base=<ngboost.esn_learners.ESN_linear_svr_learner object at 0x1a25c8d828>,
                 Dist=<class 'ngboost.distns.normal.Normal'>,
                 Score=<class 'ngboost.scores.MLE'>, learning_rate=0.01,
                 minibatch_frac=1.0, n_estimators=500, natural_gradient=True,
                 tol=0.0001, verbose=True, verbose_eval=10) 
    
    [iter 0] loss=0.0535 val_loss=0.0000 scale=0.5000 norm=0.2703
    [iter 10] loss=0.0405 val_loss=0.0000 scale=0.5000 norm=0.2762
    [iter 20] loss=0.0240 val_loss=0.0000 scale=0.5000 norm=0.2819
    [iter 30] loss=0.0036 val_loss=0.0000 scale=0.5000 norm=0.2874
    [iter 40] loss=-0.0197 val_loss=0.0000 scale=0.5000 norm=0.2927
    [iter 50] loss=-0.0457 val_loss=0.0000 scale=0.5000 norm=0.2975
    [iter 60] loss=-0.0699 val_loss=0.0000 scale=0.2500 norm=0.1508
    [iter 70] loss=-0.0835 val_loss=0.0000 scale=0.2500 norm=0.1519
    [iter 80] loss=-0.0965 val_loss=0.0000 scale=0.2500 norm=0.1529
    [iter 90] loss=-0.1093 val_loss=0.0000 scale=0.2500 norm=0.1537
    [iter 100] loss=-0.1211 val_loss=0.0000 scale=0.2500 norm=0.1544
    [iter 110] loss=-0.1304 val_loss=0.0000 scale=0.1250 norm=0.0775
    [iter 120] loss=-0.1351 val_loss=0.0000 scale=0.1250 norm=0.0776
    [iter 130] loss=-0.1395 val_loss=0.0000 scale=0.1250 norm=0.0777
    [iter 140] loss=-0.1435 val_loss=0.0000 scale=0.1250 norm=0.0778
    [iter 150] loss=-0.1473 val_loss=0.0000 scale=0.1250 norm=0.0779
    [iter 160] loss=-0.1507 val_loss=0.0000 scale=0.0625 norm=0.0389
    [iter 170] loss=-0.1527 val_loss=0.0000 scale=0.0625 norm=0.0389
    [iter 180] loss=-0.1540 val_loss=0.0000 scale=0.0625 norm=0.0389
    [iter 190] loss=-0.1553 val_loss=0.0000 scale=0.0625 norm=0.0389
    [iter 200] loss=-0.1564 val_loss=0.0000 scale=0.0625 norm=0.0389
    [iter 210] loss=-0.1574 val_loss=0.0000 scale=0.0625 norm=0.0388
    [iter 220] loss=-0.1584 val_loss=0.0000 scale=0.0625 norm=0.0388
    [iter 230] loss=-0.1591 val_loss=0.0000 scale=0.0312 norm=0.0194
    [iter 240] loss=-0.1596 val_loss=0.0000 scale=0.0625 norm=0.0387
    [iter 250] loss=-0.1601 val_loss=0.0000 scale=0.0312 norm=0.0194
    [iter 260] loss=-0.1603 val_loss=0.0000 scale=0.0312 norm=0.0193
    [iter 270] loss=-0.1606 val_loss=0.0000 scale=0.0312 norm=0.0193
    [iter 280] loss=-0.1608 val_loss=0.0000 scale=0.0156 norm=0.0097
    [iter 290] loss=-0.1610 val_loss=0.0000 scale=0.0312 norm=0.0193
    [iter 300] loss=-0.1611 val_loss=0.0000 scale=0.0078 norm=0.0048
    [iter 310] loss=-0.1612 val_loss=0.0000 scale=0.0078 norm=0.0048
    [iter 320] loss=-0.1612 val_loss=0.0000 scale=0.0156 norm=0.0096
    [iter 330] loss=-0.1613 val_loss=0.0000 scale=0.0156 norm=0.0096
    [iter 340] loss=-0.1613 val_loss=0.0000 scale=0.0156 norm=0.0096
    [iter 350] loss=-0.1614 val_loss=0.0000 scale=0.0078 norm=0.0048
    [iter 360] loss=-0.1614 val_loss=0.0000 scale=0.0156 norm=0.0096
    [iter 370] loss=-0.1615 val_loss=0.0000 scale=0.0078 norm=0.0048
    [iter 380] loss=-0.1615 val_loss=0.0000 scale=0.0078 norm=0.0048
    [iter 390] loss=-0.1615 val_loss=0.0000 scale=0.0078 norm=0.0048
    [iter 400] loss=-0.1615 val_loss=0.0000 scale=0.0039 norm=0.0024
    [iter 410] loss=-0.1615 val_loss=0.0000 scale=0.0001 norm=0.0001
    [iter 420] loss=-0.1615 val_loss=0.0000 scale=0.0078 norm=0.0048
    [iter 430] loss=-0.1615 val_loss=0.0000 scale=0.0005 norm=0.0003
    [iter 440] loss=-0.1615 val_loss=0.0000 scale=0.0001 norm=0.0001
    [iter 450] loss=-0.1616 val_loss=0.0000 scale=0.0039 norm=0.0024
    [iter 460] loss=-0.1616 val_loss=0.0000 scale=0.0001 norm=0.0001
    [iter 470] loss=-0.1616 val_loss=0.0000 scale=0.0039 norm=0.0024
    [iter 480] loss=-0.1616 val_loss=0.0000 scale=0.0020 norm=0.0012
    [iter 490] loss=-0.1616 val_loss=0.0000 scale=0.0020 norm=0.0012
    
    Test MSE 0.019522124314945034
    Test NLL -0.3856432474766308



![png](output_24_1.png)



```python
Pred_df.to_csv('/Users/apple/Documents/ML_Project/ML - 2.1/result/'+transform+'.csv')
```
