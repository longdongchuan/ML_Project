**模型：**

1. 决策树（**default_tree_learner**）
2. 加惩罚的linear model（**default_linear_learner**）
3. kernel ridge model（**kernel_ridge_learner**）
4. suppor vecor regrssion（**linear_svr_learner**）
5. reservoir computing+penalized linear model（**esn_ridge_learner**）
6. reservoir computing+kernel ridge model（**esn_kernel_ridge_learner**）
7. reservoir computing+suppor vecor regrssion（**esn_linear_svr_learner**）

**输入：**

​    \#                           'ws**sin(wd)', 'ws**cos(wd)', 'ws*sin(wd)+ws*cos(wd)']

1. $WS_{t} \quad WD_{t}$ **[None]**
2. $WS_{t} \quad cos(WD_{t})$ **[cos]**
3. $WS_{t} \quad sin(WD_{t})$ **[sin]**
4. $WS_{t} \quad cos(WD_{t}), \quad WS_{t-1} \quad cos(WD_{t-1}) \quad WP_{t-1},..., \quad WS_{t-3} \quad cos(WD_{t-3}) \quad WP_{t-3}$ **[cos, 3]**
5. $WS_{t} \quad cos(WD_{t}), \quad WS_{t-1} \quad cos(WD_{t-1}) \quad WP_{t-1},...,  \quad WS_{t-6} \quad cos(WD_{t-6}) \quad WP_{t-6}$ **[cos, 6]**
6. $WS_{t} \quad sin(WD_{t}) \quad cos(WD_{t})$ **[sin+cos]**
7. $WS_{t} \times cos(WD_{t})$ **[ws*cos(wd)]**
8. $WS_{t} \times sin(WD_{t})$ **[ws*sin(wd)]**
9. $WS_{t} \times cos(WD_{t}), \quad WS_{t-1} \times cos(WD_{t-1}) \quad WP_{t-1},..., \quad WS_{t-3} \times cos(WD_{t-3}) \quad WP_{t-3}$ **[ws*cos(wd), 3]**
10. $WS_{t} \times sin(WD_{t}), \quad WS_{t} \times cos(WD_{t})$ **[ws\*sin(wd)+ws\*cos(wd)]**

**输出：** $WP_{t}$

**train：** [ 6426 , 10427 ] len: 4001

**test：** [ 14389 , 17872 ] len: 3483

<img style="float:center" src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/Unknown.png">

---

### $WS_{t} \quad sin(WD_{t})$ 图

1. 决策树（**default_tree_learner**）

   ![1. default_tree_learner](/Users/apple/Documents/ML_Project/ML - 2.1/result/sin_plot/1. default_tree_learner.png)

   

2. 加惩罚的linear model（**default_linear_learner**）

   ![2. default_linear_learner](/Users/apple/Documents/ML_Project/ML - 2.1/result/sin_plot/2. default_linear_learner.png)

3. lasso

   ![3. lasso_learner](/Users/apple/Documents/ML_Project/ML - 2.1/result/sin_plot/3. lasso_learner.png)

4. kernel ridge model（**kernel_ridge_learner**）

   ![4. kernel_ridge_learner](/Users/apple/Documents/ML_Project/ML - 2.1/result/sin_plot/4. kernel_ridge_learner.png)

5. suppor vecor regrssion（**linear_svr_learner**）

   ![5. linear_svr_learner](/Users/apple/Documents/ML_Project/ML - 2.1/result/sin_plot/5. linear_svr_learner.png)

6. reservoir computing+penalized linear model（**esn_ridge_learner**）

   ![6. esn_ridge_learner](/Users/apple/Documents/ML_Project/ML - 2.1/result/sin_plot/6. esn_ridge_learner.png)

7. reservoir computing+kernel ridge model（**esn_kernel_ridge_learner**）

   ![7. esn_kernel_ridge_learner](/Users/apple/Documents/ML_Project/ML - 2.1/result/sin_plot/7. esn_kernel_ridge_learner.png)

8. reservoir computing+suppor vecor regrssion（**esn_linear_svr_learner**）

   ![8. esn_linear_svr_learner](/Users/apple/Documents/ML_Project/ML - 2.1/result/sin_plot/8. esn_linear_svr_learner.png)



