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
transform='ws*sin(wd)+ws*cos(wd)'
X_train, X_test, Y_train, Y_test = get_data(hour_num=0, transform=transform,
                                            drop_time=True, scale=True)
Pred_df = Y_test
```

    get_data(hour_num=0, transform='ws*sin(wd)+ws*cos(wd)', drop_time=True, scale=True)
    


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
    
    [iter 0] loss=0.0535 val_loss=0.0000 scale=0.2500 norm=0.1352
    [iter 100] loss=0.0524 val_loss=0.0000 scale=0.2500 norm=0.1350
    [iter 200] loss=0.0524 val_loss=0.0000 scale=0.1250 norm=0.0675
    [iter 300] loss=0.0524 val_loss=0.0000 scale=0.0625 norm=0.0338
    [iter 400] loss=0.0524 val_loss=0.0000 scale=0.0312 norm=0.0169
    
    Test MSE 0.05932252097751367
    Test NLL 0.008632585860007869



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
    
    [iter 0] loss=0.0535 val_loss=0.0000 scale=1.0000 norm=0.5407
    [iter 200] loss=-0.3170 val_loss=0.0000 scale=0.0625 norm=0.0290
    [iter 400] loss=-0.3220 val_loss=0.0000 scale=0.0020 norm=0.0009
    [iter 600] loss=-0.3221 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 800] loss=-0.3221 val_loss=0.0000 scale=0.0002 norm=0.0001
    
    Test MSE 0.04237594754733673
    Test NLL -0.21648586131146536



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
    
    [iter 0] loss=0.1430 val_loss=0.0000 scale=0.5000 norm=1.2208
    [iter 100] loss=0.1415 val_loss=0.0000 scale=0.0625 norm=0.1667
    [iter 200] loss=0.1415 val_loss=0.0000 scale=0.0156 norm=0.0428
    [iter 300] loss=0.1414 val_loss=0.0000 scale=0.0039 norm=0.0108
    [iter 400] loss=0.1414 val_loss=0.0000 scale=0.0010 norm=0.0027
    
    Test MSE 0.06411462031624857
    Test NLL 0.04803610477379679



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
    
    [iter 0] loss=0.0535 val_loss=0.0000 scale=0.0156 norm=0.0084
    [iter 100] loss=0.0534 val_loss=0.0000 scale=0.0039 norm=0.0021
    [iter 200] loss=0.0534 val_loss=0.0000 scale=0.0005 norm=0.0003
    [iter 300] loss=0.0534 val_loss=0.0000 scale=0.0005 norm=0.0003
    [iter 400] loss=0.0534 val_loss=0.0000 scale=0.0002 norm=0.0001
    
    Test MSE 0.0593597895274197
    Test NLL 0.008806623473430192



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
    [iter 10] loss=-0.0685 val_loss=0.0000 scale=1.0000 norm=0.4805
    [iter 20] loss=-0.1310 val_loss=0.0000 scale=0.5000 norm=0.2374
    [iter 30] loss=-0.1606 val_loss=0.0000 scale=0.5000 norm=0.2397
    [iter 40] loss=-0.1865 val_loss=0.0000 scale=0.5000 norm=0.2434
    [iter 50] loss=-0.2090 val_loss=0.0000 scale=0.5000 norm=0.2479
    [iter 60] loss=-0.2198 val_loss=0.0000 scale=0.2500 norm=0.1253
    [iter 70] loss=-0.2287 val_loss=0.0000 scale=0.2500 norm=0.1265
    [iter 80] loss=-0.2367 val_loss=0.0000 scale=0.2500 norm=0.1277
    [iter 90] loss=-0.2437 val_loss=0.0000 scale=0.2500 norm=0.1288
    [iter 100] loss=-0.2494 val_loss=0.0000 scale=0.1250 norm=0.0649
    [iter 110] loss=-0.2520 val_loss=0.0000 scale=0.1250 norm=0.0652
    [iter 120] loss=-0.2544 val_loss=0.0000 scale=0.1250 norm=0.0654
    [iter 130] loss=-0.2565 val_loss=0.0000 scale=0.1250 norm=0.0656
    [iter 140] loss=-0.2583 val_loss=0.0000 scale=0.1250 norm=0.0658
    [iter 150] loss=-0.2599 val_loss=0.0000 scale=0.1250 norm=0.0660
    [iter 160] loss=-0.2607 val_loss=0.0000 scale=0.0625 norm=0.0331
    [iter 170] loss=-0.2613 val_loss=0.0000 scale=0.0625 norm=0.0331
    [iter 180] loss=-0.2619 val_loss=0.0000 scale=0.0625 norm=0.0332
    [iter 190] loss=-0.2623 val_loss=0.0000 scale=0.0625 norm=0.0332
    [iter 200] loss=-0.2627 val_loss=0.0000 scale=0.0625 norm=0.0333
    [iter 210] loss=-0.2630 val_loss=0.0000 scale=0.0312 norm=0.0166
    [iter 220] loss=-0.2632 val_loss=0.0000 scale=0.0312 norm=0.0167
    [iter 230] loss=-0.2633 val_loss=0.0000 scale=0.0312 norm=0.0167
    [iter 240] loss=-0.2634 val_loss=0.0000 scale=0.0312 norm=0.0167
    [iter 250] loss=-0.2635 val_loss=0.0000 scale=0.0312 norm=0.0167
    [iter 260] loss=-0.2636 val_loss=0.0000 scale=0.0156 norm=0.0083
    [iter 270] loss=-0.2637 val_loss=0.0000 scale=0.0156 norm=0.0083
    [iter 280] loss=-0.2637 val_loss=0.0000 scale=0.0156 norm=0.0083
    [iter 290] loss=-0.2637 val_loss=0.0000 scale=0.0156 norm=0.0083
    [iter 300] loss=-0.2638 val_loss=0.0000 scale=0.0156 norm=0.0083
    [iter 310] loss=-0.2638 val_loss=0.0000 scale=0.0156 norm=0.0084
    [iter 320] loss=-0.2638 val_loss=0.0000 scale=0.0078 norm=0.0042
    [iter 330] loss=-0.2638 val_loss=0.0000 scale=0.0078 norm=0.0042
    [iter 340] loss=-0.2638 val_loss=0.0000 scale=0.0078 norm=0.0042
    [iter 350] loss=-0.2638 val_loss=0.0000 scale=0.0078 norm=0.0042
    [iter 360] loss=-0.2638 val_loss=0.0000 scale=0.0078 norm=0.0042
    [iter 370] loss=-0.2638 val_loss=0.0000 scale=0.0039 norm=0.0021
    [iter 380] loss=-0.2638 val_loss=0.0000 scale=0.0039 norm=0.0021
    [iter 390] loss=-0.2638 val_loss=0.0000 scale=0.0039 norm=0.0021
    [iter 400] loss=-0.2638 val_loss=0.0000 scale=0.0039 norm=0.0021
    [iter 410] loss=-0.2638 val_loss=0.0000 scale=0.0039 norm=0.0021
    [iter 420] loss=-0.2638 val_loss=0.0000 scale=0.0020 norm=0.0010
    [iter 430] loss=-0.2638 val_loss=0.0000 scale=0.0020 norm=0.0010
    [iter 440] loss=-0.2638 val_loss=0.0000 scale=0.0020 norm=0.0010
    [iter 450] loss=-0.2638 val_loss=0.0000 scale=0.0020 norm=0.0010
    [iter 460] loss=-0.2638 val_loss=0.0000 scale=0.0020 norm=0.0010
    [iter 470] loss=-0.2638 val_loss=0.0000 scale=0.0020 norm=0.0010
    [iter 480] loss=-0.2638 val_loss=0.0000 scale=0.0010 norm=0.0005
    [iter 490] loss=-0.2638 val_loss=0.0000 scale=0.0010 norm=0.0005
    
    Test MSE 0.020917298398912716
    Test NLL -0.36772470462107276



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

    NGBRegressor(Base=<ngboost.esn_learners.ESN_Ridge_learner object at 0x1a20c12518>,
                 Dist=<class 'ngboost.distns.normal.Normal'>,
                 Score=<class 'ngboost.scores.MLE'>, learning_rate=0.01,
                 minibatch_frac=1.0, n_estimators=500, natural_gradient=True,
                 tol=0.0001, verbose=True, verbose_eval=10) 
    
    [iter 0] loss=0.0535 val_loss=0.0000 scale=1.0000 norm=0.5407
    [iter 10] loss=-0.0827 val_loss=0.0000 scale=1.0000 norm=0.4746
    [iter 20] loss=-0.1321 val_loss=0.0000 scale=0.5000 norm=0.2348
    [iter 30] loss=-0.1655 val_loss=0.0000 scale=0.5000 norm=0.2366
    [iter 40] loss=-0.1949 val_loss=0.0000 scale=0.5000 norm=0.2401
    [iter 50] loss=-0.2180 val_loss=0.0000 scale=0.2500 norm=0.1220
    [iter 60] loss=-0.2296 val_loss=0.0000 scale=0.2500 norm=0.1232
    [iter 70] loss=-0.2398 val_loss=0.0000 scale=0.2500 norm=0.1243
    [iter 80] loss=-0.2490 val_loss=0.0000 scale=0.2500 norm=0.1255
    [iter 90] loss=-0.2573 val_loss=0.0000 scale=0.2500 norm=0.1265
    [iter 100] loss=-0.2629 val_loss=0.0000 scale=0.1250 norm=0.0636
    [iter 110] loss=-0.2659 val_loss=0.0000 scale=0.1250 norm=0.0639
    [iter 120] loss=-0.2687 val_loss=0.0000 scale=0.1250 norm=0.0641
    [iter 130] loss=-0.2712 val_loss=0.0000 scale=0.1250 norm=0.0643
    [iter 140] loss=-0.2734 val_loss=0.0000 scale=0.1250 norm=0.0645
    [iter 150] loss=-0.2751 val_loss=0.0000 scale=0.1250 norm=0.0646
    [iter 160] loss=-0.2761 val_loss=0.0000 scale=0.0625 norm=0.0323
    [iter 170] loss=-0.2768 val_loss=0.0000 scale=0.0625 norm=0.0324
    [iter 180] loss=-0.2774 val_loss=0.0000 scale=0.0625 norm=0.0324
    [iter 190] loss=-0.2780 val_loss=0.0000 scale=0.0625 norm=0.0324
    [iter 200] loss=-0.2785 val_loss=0.0000 scale=0.0625 norm=0.0325
    [iter 210] loss=-0.2787 val_loss=0.0000 scale=0.0312 norm=0.0162
    [iter 220] loss=-0.2789 val_loss=0.0000 scale=0.0312 norm=0.0162
    [iter 230] loss=-0.2790 val_loss=0.0000 scale=0.0312 norm=0.0163
    [iter 240] loss=-0.2792 val_loss=0.0000 scale=0.0312 norm=0.0163
    [iter 250] loss=-0.2793 val_loss=0.0000 scale=0.0312 norm=0.0163
    [iter 260] loss=-0.2794 val_loss=0.0000 scale=0.0156 norm=0.0081
    [iter 270] loss=-0.2795 val_loss=0.0000 scale=0.0156 norm=0.0081
    [iter 280] loss=-0.2795 val_loss=0.0000 scale=0.0156 norm=0.0081
    [iter 290] loss=-0.2796 val_loss=0.0000 scale=0.0078 norm=0.0041
    [iter 300] loss=-0.2796 val_loss=0.0000 scale=0.0156 norm=0.0081
    [iter 310] loss=-0.2796 val_loss=0.0000 scale=0.0020 norm=0.0010
    [iter 320] loss=-0.2796 val_loss=0.0000 scale=0.0078 norm=0.0041
    [iter 330] loss=-0.2797 val_loss=0.0000 scale=0.0078 norm=0.0041
    [iter 340] loss=-0.2797 val_loss=0.0000 scale=0.0078 norm=0.0041
    [iter 350] loss=-0.2797 val_loss=0.0000 scale=0.0020 norm=0.0010
    [iter 360] loss=-0.2797 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 370] loss=-0.2797 val_loss=0.0000 scale=0.0078 norm=0.0041
    [iter 380] loss=-0.2797 val_loss=0.0000 scale=0.0039 norm=0.0020
    [iter 390] loss=-0.2797 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 400] loss=-0.2797 val_loss=0.0000 scale=0.0078 norm=0.0041
    [iter 410] loss=-0.2797 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 420] loss=-0.2797 val_loss=0.0000 scale=0.0039 norm=0.0020
    [iter 430] loss=-0.2797 val_loss=0.0000 scale=0.0078 norm=0.0041
    [iter 440] loss=-0.2797 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 450] loss=-0.2797 val_loss=0.0000 scale=0.0078 norm=0.0041
    [iter 460] loss=-0.2797 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 470] loss=-0.2797 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 480] loss=-0.2797 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 490] loss=-0.2797 val_loss=0.0000 scale=0.0039 norm=0.0020
    
    Test MSE 0.02130404152699136
    Test NLL -0.36056684039587056



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

    NGBRegressor(Base=<ngboost.esn_learners.ESN_kernel_ridge_learner object at 0x10b1a0be0>,
                 Dist=<class 'ngboost.distns.normal.Normal'>,
                 Score=<class 'ngboost.scores.MLE'>, learning_rate=0.01,
                 minibatch_frac=1.0, n_estimators=500, natural_gradient=True,
                 tol=0.0001, verbose=True, verbose_eval=5) 
    
    [iter 0] loss=0.0535 val_loss=0.0000 scale=1.0000 norm=0.5407
    [iter 5] loss=0.0501 val_loss=0.0000 scale=1.0000 norm=0.5384
    [iter 10] loss=0.0468 val_loss=0.0000 scale=1.0000 norm=0.5367
    [iter 15] loss=0.0438 val_loss=0.0000 scale=1.0000 norm=0.5356
    [iter 20] loss=0.0407 val_loss=0.0000 scale=1.0000 norm=0.5348
    [iter 25] loss=0.0377 val_loss=0.0000 scale=1.0000 norm=0.5345
    [iter 30] loss=0.0350 val_loss=0.0000 scale=1.0000 norm=0.5345
    [iter 35] loss=0.0323 val_loss=0.0000 scale=1.0000 norm=0.5348
    [iter 40] loss=0.0298 val_loss=0.0000 scale=1.0000 norm=0.5353
    [iter 45] loss=0.0273 val_loss=0.0000 scale=1.0000 norm=0.5360
    [iter 50] loss=0.0243 val_loss=0.0000 scale=1.0000 norm=0.5367
    [iter 55] loss=0.0212 val_loss=0.0000 scale=1.0000 norm=0.5375
    [iter 60] loss=0.0177 val_loss=0.0000 scale=1.0000 norm=0.5384
    [iter 65] loss=0.0145 val_loss=0.0000 scale=1.0000 norm=0.5398
    [iter 70] loss=0.0115 val_loss=0.0000 scale=1.0000 norm=0.5413
    [iter 75] loss=0.0087 val_loss=0.0000 scale=1.0000 norm=0.5432
    [iter 80] loss=0.0058 val_loss=0.0000 scale=1.0000 norm=0.5452
    [iter 85] loss=0.0027 val_loss=0.0000 scale=1.0000 norm=0.5471
    [iter 90] loss=0.0002 val_loss=0.0000 scale=1.0000 norm=0.5496
    [iter 95] loss=-0.0028 val_loss=0.0000 scale=1.0000 norm=0.5519
    [iter 100] loss=-0.0055 val_loss=0.0000 scale=1.0000 norm=0.5545
    [iter 105] loss=-0.0084 val_loss=0.0000 scale=1.0000 norm=0.5571
    [iter 110] loss=-0.0112 val_loss=0.0000 scale=1.0000 norm=0.5599
    [iter 115] loss=-0.0139 val_loss=0.0000 scale=1.0000 norm=0.5628
    [iter 120] loss=-0.0165 val_loss=0.0000 scale=1.0000 norm=0.5659
    [iter 125] loss=-0.0186 val_loss=0.0000 scale=1.0000 norm=0.5693
    [iter 130] loss=-0.0208 val_loss=0.0000 scale=1.0000 norm=0.5726
    [iter 135] loss=-0.0230 val_loss=0.0000 scale=1.0000 norm=0.5758
    [iter 140] loss=-0.0252 val_loss=0.0000 scale=1.0000 norm=0.5791
    [iter 145] loss=-0.0275 val_loss=0.0000 scale=1.0000 norm=0.5824
    [iter 150] loss=-0.0293 val_loss=0.0000 scale=1.0000 norm=0.5859
    [iter 155] loss=-0.0316 val_loss=0.0000 scale=1.0000 norm=0.5892
    [iter 160] loss=-0.0332 val_loss=0.0000 scale=1.0000 norm=0.5923
    [iter 165] loss=-0.0348 val_loss=0.0000 scale=1.0000 norm=0.5949
    [iter 170] loss=-0.0361 val_loss=0.0000 scale=1.0000 norm=0.5985
    [iter 175] loss=-0.0372 val_loss=0.0000 scale=1.0000 norm=0.6017
    [iter 180] loss=-0.0387 val_loss=0.0000 scale=1.0000 norm=0.6051
    [iter 185] loss=-0.0399 val_loss=0.0000 scale=1.0000 norm=0.6076
    [iter 190] loss=-0.0420 val_loss=0.0000 scale=1.0000 norm=0.6105
    [iter 195] loss=-0.0432 val_loss=0.0000 scale=0.5000 norm=0.3065
    [iter 200] loss=-0.0442 val_loss=0.0000 scale=0.5000 norm=0.3078
    [iter 205] loss=-0.0454 val_loss=0.0000 scale=1.0000 norm=0.6180
    [iter 210] loss=-0.0462 val_loss=0.0000 scale=1.0000 norm=0.6200
    [iter 215] loss=-0.0479 val_loss=0.0000 scale=1.0000 norm=0.6228
    [iter 220] loss=-0.0491 val_loss=0.0000 scale=0.5000 norm=0.3127
    [iter 225] loss=-0.0498 val_loss=0.0000 scale=0.5000 norm=0.3136
    [iter 230] loss=-0.0514 val_loss=0.0000 scale=0.5000 norm=0.3147
    [iter 235] loss=-0.0517 val_loss=0.0000 scale=0.5000 norm=0.3154
    [iter 240] loss=-0.0521 val_loss=0.0000 scale=0.2500 norm=0.1581
    [iter 245] loss=-0.0525 val_loss=0.0000 scale=0.5000 norm=0.3169
    [iter 250] loss=-0.0530 val_loss=0.0000 scale=0.5000 norm=0.3177
    [iter 255] loss=-0.0537 val_loss=0.0000 scale=0.5000 norm=0.3184
    [iter 260] loss=-0.0543 val_loss=0.0000 scale=1.0000 norm=0.6383
    [iter 265] loss=-0.0549 val_loss=0.0000 scale=0.5000 norm=0.3199
    [iter 270] loss=-0.0559 val_loss=0.0000 scale=1.0000 norm=0.6414
    [iter 275] loss=-0.0565 val_loss=0.0000 scale=1.0000 norm=0.6428
    [iter 280] loss=-0.0571 val_loss=0.0000 scale=1.0000 norm=0.6440
    [iter 285] loss=-0.0582 val_loss=0.0000 scale=0.5000 norm=0.3229
    [iter 290] loss=-0.0586 val_loss=0.0000 scale=0.2500 norm=0.1617
    [iter 295] loss=-0.0592 val_loss=0.0000 scale=0.2500 norm=0.1620
    [iter 300] loss=-0.0594 val_loss=0.0000 scale=0.2500 norm=0.1622
    [iter 305] loss=-0.0595 val_loss=0.0000 scale=0.0020 norm=0.0013
    [iter 310] loss=-0.0599 val_loss=0.0000 scale=0.5000 norm=0.3252
    [iter 315] loss=-0.0603 val_loss=0.0000 scale=0.2500 norm=0.1629
    [iter 320] loss=-0.0605 val_loss=0.0000 scale=0.5000 norm=0.3260
    [iter 325] loss=-0.0610 val_loss=0.0000 scale=0.5000 norm=0.3265
    [iter 330] loss=-0.0616 val_loss=0.0000 scale=0.0625 norm=0.0409
    [iter 335] loss=-0.0618 val_loss=0.0000 scale=0.0020 norm=0.0013
    [iter 340] loss=-0.0621 val_loss=0.0000 scale=0.1250 norm=0.0819
    [iter 345] loss=-0.0621 val_loss=0.0000 scale=0.0020 norm=0.0013
    [iter 350] loss=-0.0625 val_loss=0.0000 scale=0.2500 norm=0.1641
    [iter 355] loss=-0.0628 val_loss=0.0000 scale=0.2500 norm=0.1642
    [iter 360] loss=-0.0631 val_loss=0.0000 scale=0.5000 norm=0.3289
    [iter 365] loss=-0.0632 val_loss=0.0000 scale=0.2500 norm=0.1646
    [iter 370] loss=-0.0636 val_loss=0.0000 scale=0.0020 norm=0.0013
    [iter 375] loss=-0.0641 val_loss=0.0000 scale=0.5000 norm=0.3299
    [iter 380] loss=-0.0645 val_loss=0.0000 scale=1.0000 norm=0.6606
    [iter 385] loss=-0.0651 val_loss=0.0000 scale=0.0020 norm=0.0013
    [iter 390] loss=-0.0652 val_loss=0.0000 scale=0.5000 norm=0.3308
    [iter 395] loss=-0.0654 val_loss=0.0000 scale=0.5000 norm=0.3311
    [iter 400] loss=-0.0656 val_loss=0.0000 scale=0.0625 norm=0.0414
    [iter 405] loss=-0.0660 val_loss=0.0000 scale=0.5000 norm=0.3316
    [iter 410] loss=-0.0663 val_loss=0.0000 scale=0.5000 norm=0.3320
    [iter 415] loss=-0.0667 val_loss=0.0000 scale=1.0000 norm=0.6645
    [iter 420] loss=-0.0669 val_loss=0.0000 scale=0.5000 norm=0.3324
    [iter 425] loss=-0.0670 val_loss=0.0000 scale=0.1250 norm=0.0831
    [iter 430] loss=-0.0673 val_loss=0.0000 scale=0.5000 norm=0.3329
    [iter 435] loss=-0.0674 val_loss=0.0000 scale=0.0020 norm=0.0013
    [iter 440] loss=-0.0676 val_loss=0.0000 scale=0.2500 norm=0.1666
    [iter 445] loss=-0.0679 val_loss=0.0000 scale=0.2500 norm=0.1667
    [iter 450] loss=-0.0681 val_loss=0.0000 scale=0.0625 norm=0.0417
    [iter 455] loss=-0.0685 val_loss=0.0000 scale=0.5000 norm=0.3339
    [iter 460] loss=-0.0687 val_loss=0.0000 scale=0.0020 norm=0.0013
    [iter 465] loss=-0.0695 val_loss=0.0000 scale=0.2500 norm=0.1672
    [iter 470] loss=-0.0695 val_loss=0.0000 scale=0.2500 norm=0.1673
    [iter 475] loss=-0.0697 val_loss=0.0000 scale=0.5000 norm=0.3348
    [iter 480] loss=-0.0698 val_loss=0.0000 scale=0.0020 norm=0.0013
    [iter 485] loss=-0.0702 val_loss=0.0000 scale=1.0000 norm=0.6704
    [iter 490] loss=-0.0706 val_loss=0.0000 scale=0.0020 norm=0.0013
    [iter 495] loss=-0.0706 val_loss=0.0000 scale=0.5000 norm=0.3354
    
    Test MSE 0.06326687907600909
    Test NLL -0.17487266935865922



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

    NGBRegressor(Base=<ngboost.esn_learners.ESN_linear_svr_learner object at 0x1a21132f98>,
                 Dist=<class 'ngboost.distns.normal.Normal'>,
                 Score=<class 'ngboost.scores.MLE'>, learning_rate=0.01,
                 minibatch_frac=1.0, n_estimators=500, natural_gradient=True,
                 tol=0.0001, verbose=True, verbose_eval=10) 
    
    [iter 0] loss=0.0535 val_loss=0.0000 scale=0.0312 norm=0.0169
    [iter 10] loss=0.0534 val_loss=0.0000 scale=0.0312 norm=0.0169
    [iter 20] loss=0.0534 val_loss=0.0000 scale=0.0312 norm=0.0170
    [iter 30] loss=0.0533 val_loss=0.0000 scale=0.0156 norm=0.0085
    [iter 40] loss=0.0533 val_loss=0.0000 scale=0.0156 norm=0.0085
    [iter 50] loss=0.0533 val_loss=0.0000 scale=0.0156 norm=0.0085
    [iter 60] loss=0.0533 val_loss=0.0000 scale=0.0156 norm=0.0085
    [iter 70] loss=0.0533 val_loss=0.0000 scale=0.0156 norm=0.0086
    [iter 80] loss=0.0533 val_loss=0.0000 scale=0.0078 norm=0.0043
    [iter 90] loss=0.0533 val_loss=0.0000 scale=0.0156 norm=0.0086
    [iter 100] loss=0.0532 val_loss=0.0000 scale=0.0078 norm=0.0043
    [iter 110] loss=0.0532 val_loss=0.0000 scale=0.0039 norm=0.0021
    [iter 120] loss=0.0532 val_loss=0.0000 scale=0.0020 norm=0.0011
    [iter 130] loss=0.0532 val_loss=0.0000 scale=0.0156 norm=0.0086
    [iter 140] loss=0.0532 val_loss=0.0000 scale=0.0078 norm=0.0043
    [iter 150] loss=0.0532 val_loss=0.0000 scale=0.0020 norm=0.0011
    [iter 160] loss=0.0532 val_loss=0.0000 scale=0.0039 norm=0.0021
    [iter 170] loss=0.0532 val_loss=0.0000 scale=0.0039 norm=0.0021
    [iter 180] loss=0.0532 val_loss=0.0000 scale=0.0020 norm=0.0011
    [iter 190] loss=0.0532 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 200] loss=0.0532 val_loss=0.0000 scale=0.0039 norm=0.0022
    [iter 210] loss=0.0532 val_loss=0.0000 scale=0.0039 norm=0.0022
    [iter 220] loss=0.0532 val_loss=0.0000 scale=0.0078 norm=0.0043
    [iter 230] loss=0.0532 val_loss=0.0000 scale=0.0020 norm=0.0011
    [iter 240] loss=0.0532 val_loss=0.0000 scale=0.0078 norm=0.0043
    [iter 250] loss=0.0532 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 260] loss=0.0532 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 270] loss=0.0532 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 280] loss=0.0532 val_loss=0.0000 scale=0.0039 norm=0.0022
    [iter 290] loss=0.0532 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 300] loss=0.0532 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 310] loss=0.0532 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 320] loss=0.0532 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 330] loss=0.0532 val_loss=0.0000 scale=0.0039 norm=0.0022
    [iter 340] loss=0.0532 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 350] loss=0.0532 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 360] loss=0.0532 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 370] loss=0.0532 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 380] loss=0.0532 val_loss=0.0000 scale=0.0078 norm=0.0043
    [iter 390] loss=0.0532 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 400] loss=0.0532 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 410] loss=0.0532 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 420] loss=0.0532 val_loss=0.0000 scale=0.0039 norm=0.0022
    [iter 430] loss=0.0532 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 440] loss=0.0532 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 450] loss=0.0532 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 460] loss=0.0532 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 470] loss=0.0532 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 480] loss=0.0532 val_loss=0.0000 scale=0.0002 norm=0.0001
    [iter 490] loss=0.0532 val_loss=0.0000 scale=0.0002 norm=0.0001
    
    Test MSE 0.059434927947876935
    Test NLL 0.008945629464314415



![png](output_24_1.png)



```python
Pred_df.to_csv('/Users/apple/Documents/ML_Project/ML - 2.1/result/'+transform+'.csv')
```
