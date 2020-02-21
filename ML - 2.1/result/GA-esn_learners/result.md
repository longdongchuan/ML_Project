# esn_ridge_learner

**遗传参数设置：**

1. 编码方式：Encoding = 'RI'
2.  种群规模：NIND = 30
3. 算法模板：ea.soea_SEGA_templet
4. 最大进化代数：MAXGEN = 100
5. “进化停滞”判断阈值：trappedValue = 1e-6
6.  进化停滞计数器最大上限值：maxTrappedCount = 20
7. 交叉验证折数：CV=10
8. 训练集：[ 6426 , 10427 ] len: 4001【十折交叉验证，验证集长度400】
9. 测试集：[ 14389 , 15388 ] len: 1000
10. 输入空间：wind_power、 sin(wind_direction)、 cos(wind_direction)
11. 预测：wind_speed

---

**结果1：**

<img src="/Users/apple/Documents/ML_Project/ML - 2.1/module/GA/Result3/Unknown.png" alt="Unknown" style="zoom: 50%;" />

1. 最优MSE：0.014613530279146575（验证集长度400）

2. **Test mse：0.01656127540590186（测试集长度1000）**

3. **最优控制变量值：**（变量搜索范围）

   n_readout=7746  				 	 (1, 10000] int

   n_components=21 				   (1, 2000] int

   damping = 0.184439471 		  (0, 1] float

   weight_scaling = 0.449944076 (0, 1] float

4. 有效进化代数：41

5. 最优的一代是第 21 代

6. 评价次数：1230

7. 使用时间：5213 秒

**结果2：**

<img src="/Users/apple/Documents/ML_Project/ML - 2.1/module/GA/Result4/Unknown.png" alt="Unknown" style="zoom: 50%;" />

1. 最优MSE：0.014936368910068318（验证集长度400）

2. **Test mse：0.0164576028774735（测试集长度1000）**

3. **最优控制变量值：**（变量搜索范围）

   n_readout=464  				 	  (1, 10000] int

   n_components=30 			      (1, 2000] int

   damping = 0.617584854		  (0, 1] float

   weight_scaling = 0.9465386  **(0.5, 1]** float **# 与上方设置唯一不同**

4. 有效进化代数：50

5. 最优的一代是第 30 代

6. 评价次数：1500

7. 使用时间： 5654 秒

**esn 默认参数：**

1. n_readout=1000
2. n_components=100
3. damping = 0.5
4. weight_scaling = 0.9
5. **Test mse：0.016665708498075166（测试集长度1000）**

---

# 模型比较

- **ridge**
    - alpha=0.01
- **MSE = 0.016642**
- **esn_ridge_learner**
    - n_readout=464
    - n_components=30
    - damping=0.61758485
    - weight_scaling=0.94653868
    - alpha=0.01
    - **MSE = 0.016483**

- **esn_ridge_learner**（default）
  - n_readout=1000
  - n_components=100
  - damping=0.5
  - weight_scaling=0.9
  - alpha=0.01
  - **MSE = 0.016665**

- **ngboost**
    - **base=ridge**
      - alpha=0.01
    - n_estimators=500
    - learning_rate=0.01
- **MSE = 0.012642**
- **ngboost**
    - **base=esn_ridge_learner**
      - n_readout=464
      - n_components=30
      - damping=0.61758485
      - weight_scaling=0.94653868
      - alpha=0.01
    - n_estimators=500
    - learning_rate=0.01
    - **MSE = 0.0132529**

- **ngboost**
  - **base=esn_ridge_learner**（default）
    - n_readout=1000
    - n_components=100
    - damping=0.5
    - weight_scaling=0.9
    - alpha=0.01
  - n_estimators=500
  - learning_rate=0.01
  - **MSE =  0.0138924**