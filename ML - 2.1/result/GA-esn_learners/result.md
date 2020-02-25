# 1. 遗传寻优

**遗传参数设置：**

1. 编码方式：Encoding = 'RI'
2.  种群规模：NIND = 30
3. 算法模板：ea.soea_SEGA_templet
4. 最大进化代数：MAXGEN = 100
5. “进化停滞”判断阈值：trappedValue = 1e-6
6.  进化停滞计数器最大上限值：maxTrappedCount = 20
7. 交叉验证折数：CV=10
8. 训练集+验证集：[ 6426 , 10427 ] len: 4001【**四折交叉验证**，验证集长度1000】
9. 测试集：[ 14389 , 15388 ] len: 1000
10. 输入空间：wind_speed、 sin(wind_direction)、 cos(wind_direction)
11. 预测：wind_power

## 1.1 esn_ridge_learner

首先对 ridge_learner 岭回归的 alpha 值（岭参数）进行设置，在 [0,1]，间隔 0.01 上进行测试，得到 ridge_learner 的最优 alpha 值为 1。

<img style="float:center" src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/ridge_learner_alpha.png" width="400" >

**寻优结果：**（alpha=1）

1. 最优MSE：0.014756220500052944（验证集长度400）

2. **Test mse：0.016489124026034547（测试集长度1000）**

3. **最优控制变量值：**（变量搜索范围）

   **n_readout=8287**  				 	                (1, 10000] int

   **n_components=24** 			                       (1, 2000] int

   **damping = 0.665457107410011**		       (0, 1] float

   **weight_scaling = 0.6273128470518949**  (0.5, 1] float 

4. 有效进化代数：49

5. 最优的一代是第 29 代

6. 评价次数：1490

7. 使用时间： 4947 秒

**esn 默认参数：**（alpha=1）

1. n_readout=1000
2. n_components=100
3. damping = 0.5
4. weight_scaling = 0.9
5. **Test mse：0.01642650566799493（测试集长度1000）**



## 1.2 esn_lasso_learner

首先对 Lasso 回归的 alpha 值进行设置，在 [0,1]，间隔 0.01 上进行测试，得到 ridge_learner 的最优 alpha 值为 0.01。

<img style="float:center" src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/lasso_learner_alpha.png" width="400" >

**寻优结果：**（alpha=0.01）

1. 最优MSE：（验证集长度400）

2. **Test mse：（测试集长度1000）**

3. **最优控制变量值：**（变量搜索范围）

   **n_readout=**  				 	                (1, 10000] int

   **n_components=** 			                       (1, 2000] int

   **damping =**		       (0, 1] float

   **weight_scaling = **  (0.5, 1] float 

4. 有效进化代数：

5. 最优的一代是第  代

6. 评价次数：

7. 使用时间：  秒

**esn 默认参数：**（alpha=0.01）

1. n_readout=1000
2. n_components=100
3. damping = 0.5
4. weight_scaling = 0.9
5. **Test mse：0.01965754538004085（测试集长度1000）**

# 2. 模型对比

模型组合方式：
- 1. **Base**
    - 1.1: default_tree_learner
    - 1.2: default_linear_learner(ridge)
    - 1.3: lasso_learner
    - 1.4: kernel_ridge_learner
    - 1.5: linear_svr_learner
  
- 2. **ESN + Base:**
    - 2.1 esn_ridge_learner
    - 2.2 esn_lasso_learner
    - 2.3 esn_kernel_ridge_learner
    - 2.4 esn_linear_svr_learner
  
- 3. **NGBoost(Base):**
    - model_test(Base)

- 4. **NGBoost(ESN + Base):**
    - model_test(ESN + Base)

- 5. **ESN + NGBoost(Base):**
    - esn_model_test(Base)
    
  

## 2.1 ridge 模型

| 模型     | 1. ridge  | 2. esn+ridge                                                 | 3. ngboost(ridge)                                            | 4. ngboost(ens+ridge)                                        | 5. esn+ngboost(ridge)                                        |
| -------- | :-------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 参数设置 | alpha=1   | alpha=1<br>n_readout=8287<br/>n_components=24<br/>damping = 0.66547671<br/>weight_scaling = 0.62731284 | alpha=1<br>n_estimators=500<br>learning_rate=0.01<br>Score=MLE | alpha=1<br>n_estimators=500<br>learning_rate=0.01<br>Score=MLE<br>n_readout=8287<br>n_components=24<br>damping = 0.66547671<br>weight_scaling = 0.62731284 | alpha=1<br/>n_estimators=500<br/>learning_rate=0.01<br/>Score=CRPS<br/>n_readout=8287<br/>n_components=24<br/>damping = 0.66547671<br/>weight_scaling = 0.62731284 |
| MSE      | 0.0163812 | 0.0165920                                                    | 0.0126616826                                                 | 0.012901196                                                  | 0.20159886102                                                |

与默认 esn 设置参数结果对比：

| 模型         | 2. esn+ridge                                                 | 4. ngboost(ens+ridge)                                        | 5. esn+ngboost(ridge)                                        |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | :----------------------------------------------------------- |
| 默认参数设置 | alpha=1<br/>n_readout=1000<br/>n_components=24<br/>damping = 0.66547671<br/>weight_scaling = 0.62731284 | alpha=1<br/>n_estimators=500<br/>learning_rate=0.01<br/>Score=MLE<br/>n_readout=1000<br/>n_components=24<br/>damping = 0.66547671<br/>weight_scaling = 0.62731284 | alpha=1<br/>n_estimators=500<br/>learning_rate=0.01<br/>Score=CRPS<br/>n_readout=1000<br/>n_components=24<br/>damping = 0.66547671<br/>weight_scaling = 0.62731284 |
| MSE          | 0.0164265056                                                 | 0.01247351456                                                | 0.14218040                                                   |

**在 alpha=1 下，遗传寻优后的 esn 参数值的表现甚至不如默认参数**



## 2.2 lasso 模型

| 模型     | 1. lasso    | 2. esn+lasso                                                 | 3. ngboost(lasso)                                            | 4. ngboost(ens+lasso)                                        | 5. esn+ngboost(lasso)                                        |
| -------- | :---------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 参数设置 | alpha=1     | alpha=1<br>n_readout=8287<br/>n_components=24<br/>damping = 0.66547671<br/>weight_scaling = 0.62731284 | alpha=1<br>n_estimators=500<br>learning_rate=0.01<br>Score=MLE | alpha=1<br>n_estimators=500<br>learning_rate=0.01<br>Score=MLE<br>n_readout=8287<br>n_components=24<br>damping = 0.66547671<br>weight_scaling = 0.62731284 | alpha=1<br/>n_estimators=500<br/>learning_rate=0.01<br/>Score=CRPS<br/>n_readout=8287<br/>n_components=24<br/>damping = 0.66547671<br/>weight_scaling = 0.62731284 |
| MSE      | 0.013779816 |                                                              | 0.0136203388                                                 |                                                              |                                                              |

与默认 esn 设置参数结果对比：

| 模型         | 2. esn+lasso                                                 | 4. ngboost(ens+lasso)                                        | 5. esn+ngboost(lasso)                                        |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | :----------------------------------------------------------- |
| 默认参数设置 | alpha=1<br/>n_readout=1000<br/>n_components=24<br/>damping = 0.66547671<br/>weight_scaling = 0.62731284 | alpha=1<br/>n_estimators=500<br/>learning_rate=0.01<br/>Score=MLE<br/>n_readout=1000<br/>n_components=24<br/>damping = 0.66547671<br/>weight_scaling = 0.62731284 | alpha=1<br/>n_estimators=500<br/>learning_rate=0.01<br/>Score=CRPS<br/>n_readout=1000<br/>n_components=24<br/>damping = 0.66547671<br/>weight_scaling = 0.62731284 |
| MSE          | 0.019657545                                                  | 0.0155364018                                                 | 0.22048135                                                   |