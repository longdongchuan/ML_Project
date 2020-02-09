# 数据准备

## 连续区间


```
import numpy as np
import pandas as pd
from utils import *
from ngboost.learners import *
```

输出连续区间


```
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


```
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


```
import numpy as np
import pandas as pd
from module.utils import *
from ngboost.learners import *
%config InlineBackend.figure_format='retina'
```


```
transform=None
X_train, X_test, Y_train, Y_test = get_data(hour_num=3, transform=transform,
                                            drop_time=True, scale=True)
Pred_df = Y_test
```

    get_data(hour_num=3, transform='None', drop_time=True, scale=True)
    
    Input space:  Index(['wind_speed', 'wind_direction', 'wind_speed-1', 'wind_speed-2',
           'wind_speed-3', 'wind_direction-1', 'wind_direction-2',
           'wind_direction-3', 'wind_power-1', 'wind_power-2', 'wind_power-3'],
          dtype='object')


## default_linear_learner


```
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
    [iter 100] loss=-0.3438 val_loss=0.0000 scale=0.1250 norm=0.0571
    [iter 200] loss=-0.3694 val_loss=0.0000 scale=0.0312 norm=0.0139
    [iter 300] loss=-0.3710 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 400] loss=-0.3711 val_loss=0.0000 scale=0.0020 norm=0.0009
    
    Test MSE 0.03700262671904601
    Test NLL -0.2981769482187339



![png](output_10_1.png)


## default_tree_learner


```
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
    [iter 400] loss=-0.3135 val_loss=0.0000 scale=0.0156 norm=0.0059
    [iter 600] loss=-0.3139 val_loss=0.0000 scale=0.0039 norm=0.0015
    [iter 800] loss=-0.3139 val_loss=0.0000 scale=0.0020 norm=0.0007
    
    Test MSE 0.043376918386505536
    Test NLL -0.2279749345158323



![png](output_12_1.png)


## lasso_learner


```
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
    [iter 100] loss=0.0922 val_loss=0.0000 scale=0.0156 norm=0.0470
    [iter 200] loss=0.0897 val_loss=0.0000 scale=0.0039 norm=0.0121
    [iter 300] loss=0.0896 val_loss=0.0000 scale=0.0010 norm=0.0030
    [iter 400] loss=0.0895 val_loss=0.0000 scale=0.0002 norm=0.0008
    
    Test MSE 0.03137745895624452
    Test NLL -0.33521301343449406



![png](output_14_1.png)


## linear_svr_learner


```
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
    [iter 100] loss=-0.2488 val_loss=0.0000 scale=0.2500 norm=0.1250
    [iter 200] loss=-0.2949 val_loss=0.0000 scale=0.0625 norm=0.0299
    [iter 300] loss=-0.3011 val_loss=0.0000 scale=0.0312 norm=0.0145
    [iter 400] loss=-0.3017 val_loss=0.0000 scale=0.0078 norm=0.0036
    
    Test MSE 0.02228136666658328
    Test NLL -0.390679135676004



![png](output_16_1.png)


## kernel_ridge_learner


```
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
    [iter 10] loss=-0.0549 val_loss=0.0000 scale=0.5000 norm=0.2308
    [iter 20] loss=-0.1173 val_loss=0.0000 scale=0.5000 norm=0.2160
    [iter 30] loss=-0.1631 val_loss=0.0000 scale=0.5000 norm=0.2104
    [iter 40] loss=-0.2011 val_loss=0.0000 scale=0.5000 norm=0.2093
    [iter 50] loss=-0.2344 val_loss=0.0000 scale=0.5000 norm=0.2105
    [iter 60] loss=-0.2525 val_loss=0.0000 scale=0.2500 norm=0.1059
    [iter 70] loss=-0.2667 val_loss=0.0000 scale=0.2500 norm=0.1065
    [iter 80] loss=-0.2799 val_loss=0.0000 scale=0.2500 norm=0.1071
    [iter 90] loss=-0.2922 val_loss=0.0000 scale=0.2500 norm=0.1077
    [iter 100] loss=-0.3034 val_loss=0.0000 scale=0.2500 norm=0.1082
    [iter 110] loss=-0.3129 val_loss=0.0000 scale=0.1250 norm=0.0542
    [iter 120] loss=-0.3174 val_loss=0.0000 scale=0.1250 norm=0.0543
    [iter 130] loss=-0.3216 val_loss=0.0000 scale=0.1250 norm=0.0543
    [iter 140] loss=-0.3255 val_loss=0.0000 scale=0.1250 norm=0.0543
    [iter 150] loss=-0.3289 val_loss=0.0000 scale=0.1250 norm=0.0543
    [iter 160] loss=-0.3320 val_loss=0.0000 scale=0.1250 norm=0.0542
    [iter 170] loss=-0.3345 val_loss=0.0000 scale=0.0625 norm=0.0270
    [iter 180] loss=-0.3357 val_loss=0.0000 scale=0.0625 norm=0.0270
    [iter 190] loss=-0.3368 val_loss=0.0000 scale=0.0625 norm=0.0270
    [iter 200] loss=-0.3378 val_loss=0.0000 scale=0.0625 norm=0.0269
    [iter 210] loss=-0.3386 val_loss=0.0000 scale=0.0625 norm=0.0269
    [iter 220] loss=-0.3394 val_loss=0.0000 scale=0.0625 norm=0.0268
    [iter 230] loss=-0.3400 val_loss=0.0000 scale=0.0312 norm=0.0134
    [iter 240] loss=-0.3403 val_loss=0.0000 scale=0.0312 norm=0.0134
    [iter 250] loss=-0.3406 val_loss=0.0000 scale=0.0312 norm=0.0134
    [iter 260] loss=-0.3408 val_loss=0.0000 scale=0.0312 norm=0.0133
    [iter 270] loss=-0.3410 val_loss=0.0000 scale=0.0312 norm=0.0133
    [iter 280] loss=-0.3412 val_loss=0.0000 scale=0.0312 norm=0.0133
    [iter 290] loss=-0.3414 val_loss=0.0000 scale=0.0156 norm=0.0067
    [iter 300] loss=-0.3414 val_loss=0.0000 scale=0.0156 norm=0.0066
    [iter 310] loss=-0.3415 val_loss=0.0000 scale=0.0156 norm=0.0066
    [iter 320] loss=-0.3416 val_loss=0.0000 scale=0.0156 norm=0.0066
    [iter 330] loss=-0.3416 val_loss=0.0000 scale=0.0156 norm=0.0066
    [iter 340] loss=-0.3416 val_loss=0.0000 scale=0.0156 norm=0.0066
    [iter 350] loss=-0.3417 val_loss=0.0000 scale=0.0078 norm=0.0033
    [iter 360] loss=-0.3417 val_loss=0.0000 scale=0.0078 norm=0.0033
    [iter 370] loss=-0.3417 val_loss=0.0000 scale=0.0078 norm=0.0033
    [iter 380] loss=-0.3417 val_loss=0.0000 scale=0.0078 norm=0.0033
    [iter 390] loss=-0.3417 val_loss=0.0000 scale=0.0078 norm=0.0033
    [iter 400] loss=-0.3417 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 410] loss=-0.3417 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 420] loss=-0.3417 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 430] loss=-0.3417 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 440] loss=-0.3418 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 450] loss=-0.3418 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 460] loss=-0.3418 val_loss=0.0000 scale=0.0020 norm=0.0008
    [iter 470] loss=-0.3418 val_loss=0.0000 scale=0.0020 norm=0.0008
    [iter 480] loss=-0.3418 val_loss=0.0000 scale=0.0020 norm=0.0008
    [iter 490] loss=-0.3418 val_loss=0.0000 scale=0.0020 norm=0.0008
    
    Test MSE 0.038000771543233475
    Test NLL -0.2858545972071557



![png](output_18_1.png)


## esn_ridge_learner


```
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

    NGBRegressor(Base=<ngboost.esn_learners.ESN_Ridge_learner object at 0x1a205ada20>,
                 Dist=<class 'ngboost.distns.normal.Normal'>,
                 Score=<class 'ngboost.scores.MLE'>, learning_rate=0.01,
                 minibatch_frac=1.0, n_estimators=500, natural_gradient=True,
                 tol=0.0001, verbose=True, verbose_eval=10) 
    
    [iter 0] loss=0.0536 val_loss=0.0000 scale=0.5000 norm=0.2703
    [iter 10] loss=-0.0556 val_loss=0.0000 scale=0.5000 norm=0.2293
    [iter 20] loss=-0.1173 val_loss=0.0000 scale=0.5000 norm=0.2136
    [iter 30] loss=-0.1619 val_loss=0.0000 scale=0.5000 norm=0.2070
    [iter 40] loss=-0.1979 val_loss=0.0000 scale=0.5000 norm=0.2048
    [iter 50] loss=-0.2214 val_loss=0.0000 scale=0.2500 norm=0.1024
    [iter 60] loss=-0.2358 val_loss=0.0000 scale=0.2500 norm=0.1026
    [iter 70] loss=-0.2493 val_loss=0.0000 scale=0.2500 norm=0.1029
    [iter 80] loss=-0.2618 val_loss=0.0000 scale=0.2500 norm=0.1032
    [iter 90] loss=-0.2733 val_loss=0.0000 scale=0.2500 norm=0.1035
    [iter 100] loss=-0.2839 val_loss=0.0000 scale=0.2500 norm=0.1038
    [iter 110] loss=-0.2935 val_loss=0.0000 scale=0.1250 norm=0.0520
    [iter 120] loss=-0.2984 val_loss=0.0000 scale=0.1250 norm=0.0520
    [iter 130] loss=-0.3025 val_loss=0.0000 scale=0.1250 norm=0.0521
    [iter 140] loss=-0.3063 val_loss=0.0000 scale=0.1250 norm=0.0521
    [iter 150] loss=-0.3098 val_loss=0.0000 scale=0.1250 norm=0.0520
    [iter 160] loss=-0.3130 val_loss=0.0000 scale=0.1250 norm=0.0520
    [iter 170] loss=-0.3159 val_loss=0.0000 scale=0.1250 norm=0.0519
    [iter 180] loss=-0.3185 val_loss=0.0000 scale=0.1250 norm=0.0519
    [iter 190] loss=-0.3200 val_loss=0.0000 scale=0.0625 norm=0.0259
    [iter 200] loss=-0.3211 val_loss=0.0000 scale=0.0625 norm=0.0259
    [iter 210] loss=-0.3220 val_loss=0.0000 scale=0.0625 norm=0.0258
    [iter 220] loss=-0.3229 val_loss=0.0000 scale=0.0625 norm=0.0258
    [iter 230] loss=-0.3237 val_loss=0.0000 scale=0.0625 norm=0.0258
    [iter 240] loss=-0.3245 val_loss=0.0000 scale=0.0625 norm=0.0257
    [iter 250] loss=-0.3251 val_loss=0.0000 scale=0.0625 norm=0.0257
    [iter 260] loss=-0.3254 val_loss=0.0000 scale=0.0312 norm=0.0128
    [iter 270] loss=-0.3257 val_loss=0.0000 scale=0.0312 norm=0.0128
    [iter 280] loss=-0.3259 val_loss=0.0000 scale=0.0312 norm=0.0128
    [iter 290] loss=-0.3261 val_loss=0.0000 scale=0.0312 norm=0.0128
    [iter 300] loss=-0.3263 val_loss=0.0000 scale=0.0312 norm=0.0128
    [iter 310] loss=-0.3265 val_loss=0.0000 scale=0.0312 norm=0.0128
    [iter 320] loss=-0.3267 val_loss=0.0000 scale=0.0312 norm=0.0127
    [iter 330] loss=-0.3268 val_loss=0.0000 scale=0.0156 norm=0.0064
    [iter 340] loss=-0.3268 val_loss=0.0000 scale=0.0156 norm=0.0064
    [iter 350] loss=-0.3269 val_loss=0.0000 scale=0.0156 norm=0.0064
    [iter 360] loss=-0.3269 val_loss=0.0000 scale=0.0156 norm=0.0064
    [iter 370] loss=-0.3270 val_loss=0.0000 scale=0.0156 norm=0.0064
    [iter 380] loss=-0.3270 val_loss=0.0000 scale=0.0156 norm=0.0063
    [iter 390] loss=-0.3270 val_loss=0.0000 scale=0.0156 norm=0.0063
    [iter 400] loss=-0.3271 val_loss=0.0000 scale=0.0078 norm=0.0032
    [iter 410] loss=-0.3271 val_loss=0.0000 scale=0.0078 norm=0.0032
    [iter 420] loss=-0.3271 val_loss=0.0000 scale=0.0078 norm=0.0032
    [iter 430] loss=-0.3271 val_loss=0.0000 scale=0.0078 norm=0.0032
    [iter 440] loss=-0.3271 val_loss=0.0000 scale=0.0078 norm=0.0032
    [iter 450] loss=-0.3271 val_loss=0.0000 scale=0.0039 norm=0.0016
    [iter 460] loss=-0.3271 val_loss=0.0000 scale=0.0039 norm=0.0016
    [iter 470] loss=-0.3271 val_loss=0.0000 scale=0.0039 norm=0.0016
    [iter 480] loss=-0.3271 val_loss=0.0000 scale=0.0039 norm=0.0016
    [iter 490] loss=-0.3271 val_loss=0.0000 scale=0.0020 norm=0.0008
    
    Test MSE 0.03797721586201863
    Test NLL -0.28290549828093725



![png](output_20_1.png)


## esn_kernel_ridge_learner


```
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

    NGBRegressor(Base=<ngboost.esn_learners.ESN_kernel_ridge_learner object at 0x1a205de278>,
                 Dist=<class 'ngboost.distns.normal.Normal'>,
                 Score=<class 'ngboost.scores.MLE'>, learning_rate=0.01,
                 minibatch_frac=1.0, n_estimators=500, natural_gradient=True,
                 tol=0.0001, verbose=True, verbose_eval=5) 
    
    [iter 0] loss=0.0536 val_loss=0.0000 scale=0.5000 norm=0.2703
    [iter 5] loss=-0.0064 val_loss=0.0000 scale=0.5000 norm=0.2486
    [iter 10] loss=-0.0503 val_loss=0.0000 scale=0.5000 norm=0.2354
    [iter 15] loss=-0.0852 val_loss=0.0000 scale=0.5000 norm=0.2270
    [iter 20] loss=-0.1144 val_loss=0.0000 scale=0.5000 norm=0.2214
    [iter 25] loss=-0.1398 val_loss=0.0000 scale=0.5000 norm=0.2181
    [iter 30] loss=-0.1624 val_loss=0.0000 scale=0.5000 norm=0.2162
    [iter 35] loss=-0.1832 val_loss=0.0000 scale=0.5000 norm=0.2153
    [iter 40] loss=-0.2022 val_loss=0.0000 scale=0.5000 norm=0.2151
    [iter 45] loss=-0.2199 val_loss=0.0000 scale=0.5000 norm=0.2154
    [iter 50] loss=-0.2365 val_loss=0.0000 scale=0.5000 norm=0.2161
    [iter 55] loss=-0.2458 val_loss=0.0000 scale=0.2500 norm=0.1083
    [iter 60] loss=-0.2534 val_loss=0.0000 scale=0.2500 norm=0.1086
    [iter 65] loss=-0.2607 val_loss=0.0000 scale=0.2500 norm=0.1088
    [iter 70] loss=-0.2677 val_loss=0.0000 scale=0.2500 norm=0.1091
    [iter 75] loss=-0.2744 val_loss=0.0000 scale=0.2500 norm=0.1094
    [iter 80] loss=-0.2808 val_loss=0.0000 scale=0.2500 norm=0.1096
    [iter 85] loss=-0.2870 val_loss=0.0000 scale=0.2500 norm=0.1099
    [iter 90] loss=-0.2928 val_loss=0.0000 scale=0.2500 norm=0.1101
    [iter 95] loss=-0.2984 val_loss=0.0000 scale=0.2500 norm=0.1103
    [iter 100] loss=-0.3036 val_loss=0.0000 scale=0.2500 norm=0.1105
    [iter 105] loss=-0.3085 val_loss=0.0000 scale=0.2500 norm=0.1106
    [iter 110] loss=-0.3113 val_loss=0.0000 scale=0.1250 norm=0.0553
    [iter 115] loss=-0.3136 val_loss=0.0000 scale=0.1250 norm=0.0554
    [iter 120] loss=-0.3157 val_loss=0.0000 scale=0.1250 norm=0.0554
    [iter 125] loss=-0.3178 val_loss=0.0000 scale=0.1250 norm=0.0554
    [iter 130] loss=-0.3198 val_loss=0.0000 scale=0.1250 norm=0.0554
    [iter 135] loss=-0.3216 val_loss=0.0000 scale=0.1250 norm=0.0554
    [iter 140] loss=-0.3234 val_loss=0.0000 scale=0.1250 norm=0.0554
    [iter 145] loss=-0.3251 val_loss=0.0000 scale=0.1250 norm=0.0554
    [iter 150] loss=-0.3266 val_loss=0.0000 scale=0.1250 norm=0.0553
    [iter 155] loss=-0.3281 val_loss=0.0000 scale=0.1250 norm=0.0553
    [iter 160] loss=-0.3295 val_loss=0.0000 scale=0.1250 norm=0.0553
    [iter 165] loss=-0.3307 val_loss=0.0000 scale=0.0625 norm=0.0276
    [iter 170] loss=-0.3314 val_loss=0.0000 scale=0.0625 norm=0.0276
    [iter 175] loss=-0.3320 val_loss=0.0000 scale=0.0625 norm=0.0276
    [iter 180] loss=-0.3325 val_loss=0.0000 scale=0.0625 norm=0.0276
    [iter 185] loss=-0.3330 val_loss=0.0000 scale=0.0625 norm=0.0276
    [iter 190] loss=-0.3335 val_loss=0.0000 scale=0.0625 norm=0.0275
    [iter 195] loss=-0.3340 val_loss=0.0000 scale=0.0625 norm=0.0275
    [iter 200] loss=-0.3344 val_loss=0.0000 scale=0.0625 norm=0.0275
    [iter 205] loss=-0.3349 val_loss=0.0000 scale=0.0625 norm=0.0275
    [iter 210] loss=-0.3352 val_loss=0.0000 scale=0.0625 norm=0.0275
    [iter 215] loss=-0.3356 val_loss=0.0000 scale=0.0625 norm=0.0274
    [iter 220] loss=-0.3359 val_loss=0.0000 scale=0.0625 norm=0.0274
    [iter 225] loss=-0.3362 val_loss=0.0000 scale=0.0312 norm=0.0137
    [iter 230] loss=-0.3364 val_loss=0.0000 scale=0.0312 norm=0.0137
    [iter 235] loss=-0.3366 val_loss=0.0000 scale=0.0312 norm=0.0137
    [iter 240] loss=-0.3367 val_loss=0.0000 scale=0.0312 norm=0.0137
    [iter 245] loss=-0.3368 val_loss=0.0000 scale=0.0312 norm=0.0137
    [iter 250] loss=-0.3369 val_loss=0.0000 scale=0.0312 norm=0.0137
    [iter 255] loss=-0.3371 val_loss=0.0000 scale=0.0312 norm=0.0137
    [iter 260] loss=-0.3372 val_loss=0.0000 scale=0.0312 norm=0.0136
    [iter 265] loss=-0.3373 val_loss=0.0000 scale=0.0312 norm=0.0136
    [iter 270] loss=-0.3374 val_loss=0.0000 scale=0.0312 norm=0.0136
    [iter 275] loss=-0.3374 val_loss=0.0000 scale=0.0312 norm=0.0136
    [iter 280] loss=-0.3375 val_loss=0.0000 scale=0.0312 norm=0.0136
    [iter 285] loss=-0.3376 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 290] loss=-0.3376 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 295] loss=-0.3377 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 300] loss=-0.3377 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 305] loss=-0.3377 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 310] loss=-0.3377 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 315] loss=-0.3378 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 320] loss=-0.3378 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 325] loss=-0.3378 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 330] loss=-0.3378 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 335] loss=-0.3379 val_loss=0.0000 scale=0.0156 norm=0.0068
    [iter 340] loss=-0.3379 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 345] loss=-0.3379 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 350] loss=-0.3379 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 355] loss=-0.3379 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 360] loss=-0.3379 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 365] loss=-0.3379 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 370] loss=-0.3379 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 375] loss=-0.3379 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 380] loss=-0.3380 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 385] loss=-0.3380 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 390] loss=-0.3380 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 395] loss=-0.3380 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 400] loss=-0.3380 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 405] loss=-0.3380 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 410] loss=-0.3380 val_loss=0.0000 scale=0.0078 norm=0.0034
    [iter 415] loss=-0.3380 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 420] loss=-0.3380 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 425] loss=-0.3380 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 430] loss=-0.3380 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 435] loss=-0.3380 val_loss=0.0000 scale=0.0001 norm=0.0001
    [iter 440] loss=-0.3380 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 445] loss=-0.3380 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 450] loss=-0.3380 val_loss=0.0000 scale=0.0010 norm=0.0004
    [iter 455] loss=-0.3380 val_loss=0.0000 scale=0.0001 norm=0.0001
    [iter 460] loss=-0.3380 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 465] loss=-0.3380 val_loss=0.0000 scale=0.0001 norm=0.0001
    [iter 470] loss=-0.3380 val_loss=0.0000 scale=0.0010 norm=0.0004
    [iter 475] loss=-0.3380 val_loss=0.0000 scale=0.0020 norm=0.0008
    [iter 480] loss=-0.3380 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 485] loss=-0.3380 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 490] loss=-0.3380 val_loss=0.0000 scale=0.0039 norm=0.0017
    [iter 495] loss=-0.3380 val_loss=0.0000 scale=0.0020 norm=0.0008
    
    Test MSE 0.038281426594666
    Test NLL -0.2847464490864984



![png](output_22_1.png)


## esn_linear_svr_learner


```
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

    NGBRegressor(Base=<ngboost.esn_learners.ESN_linear_svr_learner object at 0x108504cc0>,
                 Dist=<class 'ngboost.distns.normal.Normal'>,
                 Score=<class 'ngboost.scores.MLE'>, learning_rate=0.01,
                 minibatch_frac=1.0, n_estimators=500, natural_gradient=True,
                 tol=0.0001, verbose=True, verbose_eval=10) 
    
    [iter 0] loss=0.0536 val_loss=0.0000 scale=1.0000 norm=0.5406
    [iter 10] loss=-0.0274 val_loss=0.0000 scale=0.5000 norm=0.2461
    [iter 20] loss=-0.0815 val_loss=0.0000 scale=0.5000 norm=0.2347


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 30] loss=-0.1247 val_loss=0.0000 scale=0.5000 norm=0.2289
    [iter 40] loss=-0.1603 val_loss=0.0000 scale=0.5000 norm=0.2268
    [iter 50] loss=-0.1819 val_loss=0.0000 scale=0.2500 norm=0.1134
    [iter 60] loss=-0.1958 val_loss=0.0000 scale=0.2500 norm=0.1136
    [iter 70] loss=-0.2088 val_loss=0.0000 scale=0.2500 norm=0.1138
    [iter 80] loss=-0.2206 val_loss=0.0000 scale=0.2500 norm=0.1142
    [iter 90] loss=-0.2311 val_loss=0.0000 scale=0.2500 norm=0.1146
    [iter 100] loss=-0.2397 val_loss=0.0000 scale=0.2500 norm=0.1150
    [iter 110] loss=-0.2450 val_loss=0.0000 scale=0.1250 norm=0.0576
    [iter 120] loss=-0.2492 val_loss=0.0000 scale=0.1250 norm=0.0577
    [iter 130] loss=-0.2531 val_loss=0.0000 scale=0.1250 norm=0.0577
    [iter 140] loss=-0.2567 val_loss=0.0000 scale=0.1250 norm=0.0578
    [iter 150] loss=-0.2600 val_loss=0.0000 scale=0.1250 norm=0.0578
    [iter 160] loss=-0.2630 val_loss=0.0000 scale=0.1250 norm=0.0578
    [iter 170] loss=-0.2658 val_loss=0.0000 scale=0.1250 norm=0.0578
    [iter 180] loss=-0.2680 val_loss=0.0000 scale=0.0625 norm=0.0289
    [iter 190] loss=-0.2694 val_loss=0.0000 scale=0.0625 norm=0.0289
    [iter 200] loss=-0.2705 val_loss=0.0000 scale=0.0625 norm=0.0289
    [iter 210] loss=-0.2716 val_loss=0.0000 scale=0.0625 norm=0.0288


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 220] loss=-0.2726 val_loss=0.0000 scale=0.0625 norm=0.0288
    [iter 230] loss=-0.2735 val_loss=0.0000 scale=0.0625 norm=0.0288


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 240] loss=-0.2743 val_loss=0.0000 scale=0.0625 norm=0.0288
    [iter 250] loss=-0.2750 val_loss=0.0000 scale=0.0625 norm=0.0287


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 260] loss=-0.2755 val_loss=0.0000 scale=0.0312 norm=0.0144
    [iter 270] loss=-0.2759 val_loss=0.0000 scale=0.0625 norm=0.0287
    [iter 280] loss=-0.2762 val_loss=0.0000 scale=0.0312 norm=0.0143
    [iter 290] loss=-0.2765 val_loss=0.0000 scale=0.0312 norm=0.0143
    [iter 300] loss=-0.2768 val_loss=0.0000 scale=0.0312 norm=0.0143
    [iter 310] loss=-0.2770 val_loss=0.0000 scale=0.0312 norm=0.0143
    [iter 320] loss=-0.2772 val_loss=0.0000 scale=0.0312 norm=0.0143
    [iter 330] loss=-0.2774 val_loss=0.0000 scale=0.0312 norm=0.0143
    [iter 340] loss=-0.2775 val_loss=0.0000 scale=0.0156 norm=0.0071
    [iter 350] loss=-0.2776 val_loss=0.0000 scale=0.0312 norm=0.0142
    [iter 360] loss=-0.2777 val_loss=0.0000 scale=0.0156 norm=0.0071
    [iter 370] loss=-0.2777 val_loss=0.0000 scale=0.0156 norm=0.0071
    [iter 380] loss=-0.2778 val_loss=0.0000 scale=0.0156 norm=0.0071
    [iter 390] loss=-0.2778 val_loss=0.0000 scale=0.0312 norm=0.0142
    [iter 400] loss=-0.2779 val_loss=0.0000 scale=0.0020 norm=0.0009
    [iter 410] loss=-0.2779 val_loss=0.0000 scale=0.0156 norm=0.0071
    [iter 420] loss=-0.2780 val_loss=0.0000 scale=0.0078 norm=0.0035
    [iter 430] loss=-0.2780 val_loss=0.0000 scale=0.0156 norm=0.0071
    [iter 440] loss=-0.2780 val_loss=0.0000 scale=0.0039 norm=0.0018


    //anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    [iter 450] loss=-0.2780 val_loss=0.0000 scale=0.0156 norm=0.0071
    [iter 460] loss=-0.2781 val_loss=0.0000 scale=0.0156 norm=0.0071
    [iter 470] loss=-0.2781 val_loss=0.0000 scale=0.0001 norm=0.0001
    [iter 480] loss=-0.2781 val_loss=0.0000 scale=0.0078 norm=0.0035
    [iter 490] loss=-0.2781 val_loss=0.0000 scale=0.0078 norm=0.0035
    
    Test MSE 0.03521229848127299
    Test NLL -0.31523119881622824



![png](output_24_11.png)



```
Pred_df.to_csv('/Users/apple/Documents/ML_Project/ML - 2.1/result/'+transform+'-3.csv')
```
