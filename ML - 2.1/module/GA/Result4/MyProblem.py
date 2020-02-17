# -*- coding: utf-8 -*-
# esn_linear_svr_learner
import numpy as np
import geatpy as ea
import multiprocessing as mp
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool
from sklearn.model_selection import cross_val_score

import sys
sys.path.append('/Users/apple/Documents/ML_Project/ML - 2.1/module')
from utils import *
from ngboost.learners import *
from sklearn.metrics import mean_squared_error

class MyProblem(ea.Problem): # 继承Problem父类
    def __init__(self, PoolType): # PoolType是取值为'Process'或'Thread'的字符串
        name = 'MyProblem' # 初始化name（函数名称，可以随意设置）
        M = 1 # 初始化M（目标维数）
        maxormins = [1] # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 4 # 初始化Dim（决策变量维数）
        varTypes = [1, 1, 0, 0] # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [1, 1, 0, 0.5] # 决策变量下界
        ub = [10000, 2000, 1, 1] # 决策变量上界
        lbin = [0] * Dim # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        # 目标函数计算中用到的一些数据
        X_train, X_test, Y_train, Y_test = get_data(hour_num=0, transform='sin+cos',
                                            drop_time=True, scale=True)
        np.where(X_train==0, X_train, 0.001)
        np.where(Y_train==0, Y_train, 0.001)
        self.data = X_train # 训练集的特征数据（归一化）
        self.dataTarget = Y_train
        # 设置用多线程还是多进程
        self.PoolType = PoolType
        if self.PoolType == 'Thread':
            self.pool = ThreadPool(2) # 设置池的大小
        elif self.PoolType == 'Process':
            num_cores = int(mp.cpu_count()) # 获得计算机的核心数
            self.pool = ProcessPool(num_cores) # 设置池的大小
    
    def aimFunc(self, pop): # 目标函数，采用多线程加速计算
        Vars = pop.Phen.astype(int) # 得到决策变量矩阵
        args = list(zip(list(range(pop.sizes)), [Vars] * pop.sizes, [self.data] * pop.sizes, [self.dataTarget] * pop.sizes))
        if self.PoolType == 'Thread':
            pop.ObjV = np.array(list(self.pool.map(subAimFunc, args)))
        elif self.PoolType == 'Process':
            result = self.pool.map_async(subAimFunc, args)
            result.wait()
            pop.ObjV = np.array(result.get())
    
    def test(self, n_readout, n_components, damping, weight_scaling): # 代入优化后的参数对测试集进行检验
        # 读取测试集数据
        X_train, X_test, Y_train, Y_test = get_data(hour_num=0, transform='sin+cos',
                                            drop_time=True, scale=True)
        np.where(X_test==0, X_test, 0.001)
        np.where(Y_test==0, Y_test, 0.001)
        data_test = X_test[:1000] # 测试集的特征数据（归一化）
        dataTarget_test = Y_test[:1000] # 测试集的标签数据

        esn = esn_ridge_learner(
                n_readout=n_readout,
                n_components=n_components,
                damping = damping,
                weight_scaling = weight_scaling,
                alpha=0.01).fit(self.data, self.dataTarget) # 创建分类器对象并用训练集的数据拟合分类器模型

        # esn = esn_linear_svr_learner(
        #         n_readout=n_readout,
        #         n_components=n_components,
        #         damping = damping,
        #         weight_scaling = weight_scaling,
        #         epsilon=0.0,
        #         C=0.02,
        #         max_iter=10000).fit(self.data, self.dataTarget) # 创建分类器对象并用训练集的数据拟合分类器模型

        # test Mean Squared Error
        dataTarget_predict = esn.predict(data_test) # 采用训练好的分类器对象对测试集数据进行预测
        test_MSE = mean_squared_error(dataTarget_predict, dataTarget_test)
        print('\nTest MSE', test_MSE)

        
def subAimFunc(args):
    i = args[0]
    Vars = args[1]
    data = args[2]
    dataTarget = args[3]
    n_readout = int(Vars[i, 0])
    n_components = int(Vars[i, 1])
    damping = Vars[i, 2]
    weight_scaling = Vars[i, 3]
 
    esn = esn_ridge_learner(
            n_readout=n_readout,
            n_components=n_components,
            damping = damping,
            weight_scaling = weight_scaling,
            alpha=0.01) # 创建分类器对象并用训练集的数据拟合分类器模型

    # esn = esn_linear_svr_learner(
    #             n_readout=n_readout,
    #             n_components=n_components,
    #             damping = damping,
    #             weight_scaling = weight_scaling,
    #             epsilon=0.0,
    #             C=0.02,
    #             max_iter=10000) # 创建分类器对象并用训练集的数据拟合分类器模型
    
    excute_time() # 显示执行 CV 的次数
    scores = -cross_val_score(esn, data, dataTarget, cv=10, scoring='neg_mean_squared_error') # 计算交叉验证的得分
    ObjV_i = [scores.mean()] # 把交叉验证的平均得分作为目标函数值
    return ObjV_i