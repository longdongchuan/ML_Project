[TOC]

# 1. Natural Gradient Boosting (NGboost)

- 论文 [NGBoost Natural Gradient Boosting for Probabilistic Prediction](https://arxiv.org/abs/1910.03225)
- 官网 https://stanfordmlgroup.github.io/projects/ngboost/ 

**Base Learner**（Decision Tree）

The most common choice is Decision Trees, which tend to work well on structured inputs. 

**Probability Distribution**（Normal）

The distribution needs to be compatible with the output type. For e.g. Normal distribution for real valued outputs, Bernoulli for binary outputs.

**Scoring rule**（MLE）

Maximum Likelihood Estimation is an obvious choice. More robust rules such as Continuous Ranked Probability Score are also suitable.

<img style="float:center" src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/20200114222001.png" width="520" >

<img style="float:center" src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/20200114222029.png" width="320" >

# 2. gptp_multi_output

This toolkit is used to implement **multivariate Gaussian process regression (MV-GPR)** and **multivariate Student-t process regression (MV-TPR)**.

The main function is **gptp_general.m**. It can return GPR, TPR, MV-GPR, MV-TPR and their comparisons. There are four useful sub-functions:

- gp_solve_gpml.m
- tp_solve_gpml.m，
- mvgp_solve_gpml.m
- mvtp_solve_gpml.m

These four functions are used to solve GPR, TPR, MV-GPR and MV-TPR, respectively.

## 2.1 Gaussian Processs (GP) 

- [sklearn 1.7. Gaussian Processes](https://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process)

- https://sklearn.apachecn.org/docs/0.21.3/8.html

- [从数学到实现，全面回顾高斯过程中的函数最优化](https://blog.csdn.net/rlnLo2pNEfx9c/article/details/79545435)

- [浅析高斯过程回归（Gaussian process regression）](https://blog.csdn.net/qq_20195745/article/details/82721666)
- 论文 [Gaussian processes](http://cs229.stanford.edu/section/cs229-gaussian_processes.pdf)

**高斯过程 (GP)** 是一种常用的监督学习方法，旨在解决**回归**问题和**概率分类**问题。

高斯过程模型的**优点**如下：

- 预测内插了观察结果（至少对于正则核）。
- 预测结果是概率形式的（高斯形式的）。这样的话，人们可以计算得到经验置信区间并且据此来判断是否需要修改（在线拟合，自适应）在一些区域的预测值。
- 通用性: 可以指定不同的:[内核(kernels)](https://sklearn.apachecn.org/#175-高斯过程内核)。虽然该函数提供了常用的内核，但是也可以指定自定义内核。

高斯过程模型的**缺点**包括：

- 它们不稀疏，例如，模型通常使用整个样本/特征信息来进行预测。
- 高维空间模型会失效，高维也就是指特征的数量超过几十个。

# 3. Natural-Parameter Networks (NPN)

- 论文 [Natural-Parameter Networks: A Class of Probabilistic Neural Networks](http://wanghao.in/paper/NIPS16_NPN.pdf) 

**Neural networks v.s. natural-parameter-networks in two figures:**

- Distributions as first-class citizens:

<img style="float:center" src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/nn-vs-npn.png" width="520" >

- Closed-form operations to handle uncertainty:

<img style="float:center" src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/nn-vs-npn-op.png" width="520" >

# 4. Echo State Networks (ESN)

论文 [Echo State Networks](http://www.scholarpedia.org/article/Echo_state_network)

官网 [Simple Echo State Network implementations](https://mantas.info/code/simple_esn/)

[机器学习：回声状态网络(Echo State Networks)](https://blog.csdn.net/minemine999/article/details/80861863)

[回声状态网络(ESN)教程](https://blog.csdn.net/cassiePython/article/details/80389394)

[How to Understand the Spectral Radius of ESN](https://blog.csdn.net/dasimao/article/details/89635932)

<img style="float:center" src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/20180521104711625.png" width="520" >

# 5. Deep Ensembles

论文 [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://arxiv.org/pdf/1612.01474.pdf)

GitHub [deep-ensembles-uncertainty](https://github.com/vvanirudh/deep-ensembles-uncertainty)

<img style="float:center" src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/Screen Shot 2020-01-17 at 16.19.10.png" width="520" >