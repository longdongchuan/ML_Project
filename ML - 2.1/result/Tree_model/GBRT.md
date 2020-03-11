[toc]

# GBRT

## 1. 西班牙数据集

train index: [6426, 10427]   train_len: 4000
test index: [14389, 15390]  test_len: 1000

- **输入特征：**

```python
'wind_speed', 'sin(wd)', 'cos(wd)', 【t期】
'wind_speed-1', 'sin(wd)-1','cos(wd)-1', 'wind_power-1'【t-1期】
```

- **输出：**wind_power

### 1.1 寻找最大深度

max_depth = 2

<img src="/Users/apple/Documents/ML_Project/ML - 2.1/result/Tree_model/figure_GBRT/Spain_max_depth.png" alt="Spain_max_depth" style="zoom:67%;" />

### 1.2 n_estimators

<img src="/Users/apple/Documents/ML_Project/ML - 2.1/result/Tree_model/figure_GBRT/Spain_n_estimators.png" alt="Spain_n_estimators" style="zoom:67%;" />

最终设置：

```python
GradientBoostingRegressor(alpha=0.9, criterion='mse', init=None,
                          learning_rate=0.1, loss='ls', max_depth=2,
                          max_features=None, max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, n_estimators=200,
                          n_iter_no_change=None, presort='auto',
                          random_state=None, subsample=1.0, tol=0.0001,
                          validation_fraction=0.1, verbose=0, warm_start=False) 
```

test mse: 0.0025435651610870007

![Spain_predict](/Users/apple/Documents/ML_Project/ML - 2.1/result/Tree_model/figure_GBRT/Spain_predict.png)

## 2. 美国数据集

train index: [3001, 7002]   train_len: 4000
test index: [2000, 3001]  test_len: 1000

- **输入特征：**

```python
'wind_speed', 'sin(wd)', 'cos(wd)', 【t期】
'wind_speed-1', 'sin(wd)-1','cos(wd)-1', 'wind_power-1'【t-1期】
```

- **输出：**wind_power

### 2.1 寻找最大深度

max depth = 6

<img src="/Users/apple/Documents/ML_Project/ML - 2.1/result/Tree_model/figure_GBRT/US_max_depth.png" alt="US_max_depth" style="zoom:67%;" />

### 2.2 n_estimators

<img src="/Users/apple/Documents/ML_Project/ML - 2.1/result/Tree_model/figure_GBRT/US_n_estimators.png" alt="US_n_estimators" style="zoom:67%;" />

最终设置：

```python
GradientBoostingRegressor(alpha=0.9, criterion='mse', init=None,
                          learning_rate=0.1, loss='ls', max_depth=6,
                          max_features=None, max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, n_estimators=200,
                          n_iter_no_change=None, presort='auto',
                          random_state=None, subsample=1.0, tol=0.0001,
                          validation_fraction=0.1, verbose=0, warm_start=False) 
```

test mse : 2.1468578057334267e-05

![US_predict](/Users/apple/Documents/ML_Project/ML - 2.1/result/Tree_model/figure_GBRT/US_predict.png)