# 1. 遗传寻优设置

1. 编码方式：Encoding = 'RI'
2.  种群规模：NIND = 30
3. 算法模板：ea.soea_SEGA_templet
4. 最大进化代数：MAXGEN = 100
5. “进化停滞”判断阈值：trappedValue = 1e-6
6.  进化停滞计数器最大上限值：maxTrappedCount = 20
7. 交叉验证折数：CV=10
8. 训练集+验证集：[ 6426 , 10427 ] len: 4001【**四折交叉验证**，验证集长度1000】
9. 测试集：[ 14389 , 15388 ] len: 1000
10. **输入空间：wind_speed、 sin(wind_direction)、 cos(wind_direction)**
11. 预测：wind_power

# 2. 模型组合方式

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

# 3. 结果

## 3.1 esn_ridge_learner

**esn 默认参数：**

1. n_readout=1000
2. n_components=100
3. damping = 0.5
4. weight_scaling = 0.9
5. alpha = 1（在 ridge 上的最优参数）
6. **Test mse：0.01642650566799493（测试集长度1000）**

**寻优结果：**

1. 最优MSE：0.015390469160833288（验证集长度1000）

2. **Test mse：0.016440854213573736（测试集长度1000）**

3. **最优控制变量值：**（变量搜索范围）

   **n_readout=3462**  				 	                     (1, 10000] int

   **n_components=23** 			                          (1, 2000] int

   **damping = 0.26215546327467487**		       (0, 1] float

   **weight_scaling = 0.6234509481681756**      (0.5, 1] float 

   **alpha = 0.4649085531487292**                       (0, 1] float

4. 有效进化代数：38

5. 最优的一代是第 18 代

6. 评价次数：1140

7. 使用时间： 3297 秒

**模型对比：**

| 模型             | 1. ridge                                                     | 2.1 esn+ridge [default]                                      | 4.1 ngboost(ens+ridge)[default]                              | 5.1 esn+ngboost(ridge)[default]                              |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | :----------------------------------------------------------- | ------------------------------------------------------------ |
| **默认参数设置** | **alpha=1**                                                  | **alpha=1 n_readout=1000 n_components=100 damping = 0.5 weight_scaling = 0.9** | **alpha=1** <br>n_estimators=500 learning_rate=0.01 Score=MLE **n_readout=1000 n_components=100 damping = 0.5 weight_scaling = 0.9** | **alpha=1** <br>n_estimators=500 learning_rate=0.01 Score=CRPS **n_readout=1000 n_components=100 damping = 0.5 weight_scaling = 0.9** |
| **MSE**          | 0.0163812                                                    | 0.01641301                                                   | 0.01287494                                                   | 0.3714271                                                    |
| **图**           | ![1. ridge_default](file:///Users/apple/Documents/ML_Project/ML%20-%202.1/result/GA-esn_learners/ridge_figure/1.%20ridge_default.png?lastModify=1582644016) | ![2.1 esn+ridge_defualt](file:///Users/apple/Documents/ML_Project/ML%20-%202.1/result/GA-esn_learners/ridge_figure/2.1%20esn+ridge_defualt.png?lastModify=1582644016) | ![4.1 ngboost+ridge_default](file:///Users/apple/Documents/ML_Project/ML%20-%202.1/result/GA-esn_learners/ridge_figure/4.1%20ngboost+ridge_default.png?lastModify=1582644016) | ![5.1 esn+ngboost+ridge_default](file:///Users/apple/Documents/ML_Project/ML%20-%202.1/result/GA-esn_learners/ridge_figure/5.1%20esn+ngboost+ridge_default.png?lastModify=1582644016) |
| **模型**         | **3. ngboost(ridge)**                                        | **2.2 esn+ridge [GA]**                                       | **4.2 ngboost(ens+ridge) [GA]**                              | **5.2 esn+ngboost(ridge) [GA]**                              |
| **参数设置**     | **alpha=1** n_estimators=500 learning_rate=0.01 Score=MLE    | **alpha=0.4649085531 n_readout=3462 n_components=23 damping = 0.26215546 weight_scaling = 0.623450948** | **alpha=0.4649085531** n_estimators=500 learning_rate=0.01 Score=MLE **n_readout=3462 n_components=23 damping = 0.26215546 weight_scaling = 0.623450948** | **alpha=0.4649085531** n_estimators=500 learning_rate=0.01 Score=CRPS **n_readout=3462 n_components=23 damping = 0.26215546 weight_scaling = 0.623450948** |
| **MSE**          | 0.0126617                                                    | 0.016440854                                                  | 0.013522998                                                  | 0.25163414                                                   |
| **图**           | ![3. ngboost+esn+ridge_default](file:///Users/apple/Documents/ML_Project/ML%20-%202.1/result/GA-esn_learners/ridge_figure/3.%20ngboost+esn+ridge_default.png?lastModify=1582644016) | ![2.2 esn+ridge_GA](file:///Users/apple/Documents/ML_Project/ML%20-%202.1/result/GA-esn_learners/ridge_figure/2.2%20esn+ridge_GA.png?lastModify=1582644016) | ![4.2 ngboost+ridge_GA](file:///Users/apple/Documents/ML_Project/ML%20-%202.1/result/GA-esn_learners/ridge_figure/4.2%20ngboost+ridge_GA.png?lastModify=1582644016) | ![5.2 esn+ngboost+ridge_GA](file:///Users/apple/Documents/ML_Project/ML%20-%202.1/result/GA-esn_learners/ridge_figure/5.2%20esn+ngboost+ridge_GA.png?lastModify=1582644016) |

**对比 X.1 （默认参数）模型与 X.2 （遗传寻优参数）模型，遗传寻优后的 esn 参数值的表现甚至不如默认参数**



