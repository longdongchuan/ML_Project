**模型：**

1. 决策树（**default_tree_learner**）
2. 加惩罚的linear model（**default_linear_learner**）
3. **lasso_learner**
4. kernel ridge model（**kernel_ridge_learner**）
5. suppor vecor regrssion（**linear_svr_learner**）
6. reservoir computing+penalized linear model（**esn_ridge_learner**）
7. reservoir computing+kernel ridge model（**esn_kernel_ridge_learner**）
8. reservoir computing+suppor vecor regrssion（**esn_linear_svr_learner**）

**输入：**

1. $WS_{t} \quad WD_{t}$
2. $WS_{t} \quad WD_{t}$ || $WS_{t-1} \quad WD_{t-1} \quad WP_{t-1}$ || ... || $WS_{t-3} \quad WD_{t-3} \quad WP_{t-3}$  **~ (t-3)**
3. $WS_{t} \quad WD_{t}$   **~ (t-6)**
4. $WS_{t} \quad sin(WD_{t})$ 
5. $WS_{t} \quad cos(WD_{t})$
6. $WS_{t} \quad cos(WD_{t})$ **~ (t-3)**
7. $WS_{t} \quad cos(WD_{t})$ **~ (t-6)**
8. $WS_{t} \quad sin(WD_{t}) \quad cos(WD_{t})$ 
9. $WS_{t} \quad sin(WD_{t}) \quad cos(WD_{t})$  **~ (t-3)**
10. $WS_{t} \times sin(WD_{t})$
11. $WS_{t} \times cos(WD_{t})$
12. $WS_{t} \times cos(WD_{t})$ **~ (t-3)**
13. $WS_{t} \times sin(WD_{t}), \quad WS_{t} \times cos(WD_{t})$ 
14. $WS_{t} \times sin(WD_{t}), \quad WS_{t} \times cos(WD_{t})$  **~ (t-3)**

**输出：** $WP_{t}$

**train：** [ 6426 , 10427 ] len: 4001

**test：** [ 14389 , 17872 ] len: 3483

![Unknown](/Users/apple/Documents/ML_Project/ML - 2.1/result/plot/Unknown.png)