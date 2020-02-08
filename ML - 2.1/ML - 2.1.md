- [x] 从数据中选取一段连续的、没有缺失值的数据记录。把数据分成**训练集**，**测试集**（如果数据量多，可以加入验证集），**看看训练集和测试集的分布是否差不多**。

预测：**wind power**

影响因素：**wind direction**（角度数据） 和 **wind speed**

1. 如果用角度变量，考虑沿着风速分解，即 **wscos(wd) **

2. 做预测的时候，输入要考虑**滞后性**，具体是过去几阶的影响，可以用相关性判断一下，或者用energy distance 这个判断	**（当前时刻的 wind power 不仅受 wind direction 和wind speed 影响，还受过去的wind power的影响。也受过去的wind speed ，direction 的影响）**

<img style="float:center" src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/WechatIMG90.jpeg" width="520" >

<img style="float:center" src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/WechatIMG91.png" width="520" >

---

然后，在ngboost模型中，把基模型，用

- [x] 决策树（**default_tree_learner**）

- [x] linear model（**default_linear_learner**）

- [x] **lasso_learner**

- [x] 加惩罚的linear model（**default_linear_learner**）

- [x] kernel ridge model（**kernel_ridge_learner**）

- [x] suppor vecor regrssion（**linear_svr_learner**）

- [x] reservoir computing+linear model（**esn_ridge_learner**）

- [x] reservoir computing+penalized linear model（**esn_ridge_learner**）

- [x] reservoir computing+kernel ridge model（**esn_kernel_ridge_learner**）

- [x] reservoir computing+suppor vecor regrssion（**esn_linear_svr_learner**）

  

**输入：**

1. $WS_{t} \quad WD_{t}$
2. $WS_{t} \quad cos(WD_{t})$
3. $WS_{t} \quad sin(WD_{t})$
4. $WS_{t} \quad cos(WD_{t}), \quad WS_{t-1} \quad cos(WD_{t-1}) \quad WP_{t-1},..., \quad WS_{t-3} \quad cos(WD_{t-3}) \quad WP_{t-3}$ 
5. $WS_{t} \quad cos(WD_{t}), \quad WS_{t-1} \quad cos(WD_{t-1}) \quad WP_{t-1},..., \quad WS_{t-6} \quad cos(WD_{t-6}) \quad WP_{t-6}$ 
6. $WS_{t} \times cos(WD_{t})$ 
7. $WS_{t} \times sin(WD_{t})$ 
8. $WS_{t} \times cos(WD_{t}), \quad WS_{t-1} \times cos(WD_{t-1}) \quad WP_{t-1},..., \quad WS_{t-3} \times cos(WD_{t-3}) \quad WP_{t-3}$ 

**输出：** $WP_{t}$

**train：** [ 6426 , 10427 ] len: 4001

**test：** [ 14389 , 17872 ] len: 3483

<img style="float:center" src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/Unknown.png">

| model \ MSE                | ws+wd         | cos(wd)       | sin(wd)       | cos(wd)-3 | cos(wd)-6 |
| -------------------------- | ------------- | ------------- | ------------- | --------- | --------- |
| default_linear_learner     | 3. 0.0312     | 4. 0.0249     | 3. 0.0249     | 2. 0.0371 | 1. 0.0368 |
| default_tree_learner       | 4. 0.0403     | 6. 0.0408     | 6. 0.0421     | 6. 0.4089 | 4. 0.0423 |
| **linear_svr_learner**     | **1. 0.0245** | **1. 0.0189** | **2. 0.0198** | 7. 0.0593 | 5. 0.0594 |
| kernel_ridge_learner       | 6. 0.0478     | 5. 0.0340     | 5. 0.0341     | 4. 0.0386 | 3. 0.0383 |
| esn_ridge_learner          | 7. 0.0482     | 7. 0.0417     | 7. 0.0521     | 3. 0.0379 | 6. 0.9389 |
| esn_kernel_ridge_learner   | 5. 0.0442     | **3. 0.0248** | **4. 0.0257** | 5. 0.0387 | 2. 0.0367 |
| **esn_linear_svr_learner** | **2. 0.0275** | **2. 0.0216** | **1. 0.0171** | 1. 0.0368 | 无法收敛  |

| model \ MSE              | ws*cos(wd)    | ws*sin(wd)    | ws*cos(wd)-3  |      |      |
| ------------------------ | ------------- | ------------- | ------------- | ---- | ---- |
| default_linear_learner   | 5. 0.0633     | 7. 0.0632     | 5. 0.0593     |      |      |
| default_tree_learner     | 6. 0.0633     | 5. 0.0593     | 4. 0.0528     |      |      |
| linear_svr_learner       | 3. 0.0593     | 4. 0.0592     | 6. 0.0593     |      |      |
| kernel_ridge_learner     | 2. 0.0474     | 2. 0.0456     | 2. 0.0315     |      |      |
| **esn_ridge_learner**    | **1. 0.0263** | **1. 0.0258** | **1. 0.0276** |      |      |
| esn_kernel_ridge_learner | 7. 0.0704     | 3. 0.0520     | 3. 0.0464     |      |      |
| esn_linear_svr_learner   | 4. 0.0594     | 6. 0.0593     | 7. 0.0594     |      |      |



---

- [x] 把那个 deep learning 运行，deep learning 的中间结构可以简化