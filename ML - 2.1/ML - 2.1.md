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
- [x] 加惩罚的linear model（**default_linear_learner**）
- [x] kernel ridge model（**kernel_ridge_learner**）
- [x] suppor vecor regrssion（**linear_svr_learner**）
- [x] reservoir computing+linear model（**esn_ridge_learner**）
- [x] reservoir computing+penalized linear model（**esn_ridge_learner**）
- [x] reservoir computing+kernel ridge model（**esn_kernel_ridge_learner**）
- [x] reservoir computing+suppor vecor regrssion（**esn_linear_svr_learner**）

1. linear_svr_learner（0.0245-->0.0189）
2. esn_linear_svr_learner（0.0275-->0.0216）
3. default_linear_learner（0.0312-->0.0249）
4. default_tree_learner（0.04029-->0.0408）
5. esn_kernel_ridge_learner（0.0442-->0.0248）
6. kernel_ridge_learner（0.0478-->0.03400）
7. esn_ridge_learner（0.0482-->0.04177）



1. linear_svr_learner（0.0245-->0.0189）
2. esn_linear_svr_learner（0.0275-->0.0216）
3. esn_kernel_ridge_learner（0.0442-->0.0248）
4. default_linear_learner（0.0312-->0.0249）
5. kernel_ridge_learner（0.0478-->0.03400）
6. default_tree_learner（0.04029-->0.0408）
7. esn_ridge_learner（0.0482-->0.04177）

---

- [x] 把那个 deep learning 运行，deep learning 的中间结构可以简化