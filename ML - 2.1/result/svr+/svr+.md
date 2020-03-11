[toc]

# SVR+

## 1. 西班牙数据集

### 1.1 Privileged Information = Wind_Power_train ( t )

train index: [6426, 10427]   train_len: 4000
test index: [14389, 15390]  test_len: 1000

- **输入特征：**

```python
'wind_speed', 'sin(wd)', 'cos(wd)', 【t期】
'wind_speed-1', 'sin(wd)-1','cos(wd)-1', 'wind_power-1'【t-1期】
```

- **Privileged Information**:  Wind_Power ( t )【即 Y_train】
- svr+ 参数设置（默认参数）：
  - 【训练时间过长 不易调参；若将训练数据长度降至 2000，测试 mse 为0.03421，非常不理想】

``` Python
Parameters = {'C': 10,
              'gamma_corSpace': 10,
              'gamma_rbf': 1,
              'gamma_rbf_corSpace': 10,
              'epsilon': 0.1,
              'tol': 0.0001}
```

- **输出：** Wind_Power

训练时间：65 mins

**测试 mse：0.01667742097965347**

![Spain_wp_4000](/Users/apple/Documents/ML_Project/ML - 2.1/result/svr+/figure/Spain_wp_4000.png)

### 1.2 Privileged Information = Wind_Speed_train ( t ), sin(Wind_Direction_train) ( t )

train index: [6426, 10427]   train_len: 4000
test index: [14389, 15390]  test_len: 1000

- **输入特征：**

```python
'wind_speed', 'sin(wd)', 'cos(wd)', 【t期】
'wind_speed-1', 'sin(wd)-1','cos(wd)-1', 'wind_power-1'【t-1期】
```

- **Privileged Information**:  Wind_Speed_train ( t ),  sin(Wind_Direction_train) ( t )
- svr+ 参数设置（默认参数）

``` Python
Parameters = {'C': 10,
              'gamma_corSpace': 10,
              'gamma_rbf': 1,
              'gamma_rbf_corSpace': 10,
              'epsilon': 0.1,
              'tol': 0.0001}
```

- **输出：** Wind_Power

训练时间：57 mins

**测试 mse：** 0.03748050291635869

![Spain_ws_sin(wd)_4000](/Users/apple/Documents/ML_Project/ML - 2.1/result/svr+/figure/Spain_ws_sin(wd)_4000.png)

## 2. 美国数据集

### 1.1 Privileged Information = Wind_Power_train ( t )

train index: [3001, 7002] train_len: 4000
test index: [2000, 3001] test_len: 1000

- **输入特征：**

```python
'wind_speed', 'sin(wd)', 'cos(wd)', 【t期】
'wind_speed-1', 'sin(wd)-1','cos(wd)-1', 'wind_power-1'【t-1期】
```

- **Privileged Information**:  Wind_Power ( t )【即 Y_train】
- svr+ 参数设置（默认参数）：
  - 【训练时间过长 不易调参；若将训练数据长度降至 2000，测试 mse 为0.03421，非常不理想】

``` Python
Parameters = {'C': 10,
              'gamma_corSpace': 10,
              'gamma_rbf': 1,
              'gamma_rbf_corSpace': 10,
              'epsilon': 0.1,
              'tol': 0.0001}
```

- **输出：** Wind_Power

训练时间：53 mins

**测试 mse：**0.03130679323067965

![US_wp_4000](/Users/apple/Documents/ML_Project/ML - 2.1/result/svr+/figure/US_wp_4000.png)

### 1.2 Privileged Information = Wind_Speed_train ( t ), sin(Wind_Direction_train) ( t )

train index: [6426, 10427]   train_len: 4000
test index: [14389, 15390]  test_len: 1000

- **输入特征：**

```python
'wind_speed', 'sin(wd)', 'cos(wd)', 【t期】
'wind_speed-1', 'sin(wd)-1','cos(wd)-1', 'wind_power-1'【t-1期】
```

- **Privileged Information**:  Wind_Speed_train ( t ),  sin(Wind_Direction_train) ( t )
- svr+ 参数设置（默认参数）

``` Python
Parameters = {'C': 10,
              'gamma_corSpace': 10,
              'gamma_rbf': 1,
              'gamma_rbf_corSpace': 10,
              'epsilon': 0.1,
              'tol': 0.0001}
```

- **输出：** Wind_Power

训练时间：65 mins

**测试 mse：** 0.11394138700050119

![US_ws_sin(wd)_4000](/Users/apple/Documents/ML_Project/ML - 2.1/result/svr+/figure/US_ws_sin(wd)_4000.png)

# 