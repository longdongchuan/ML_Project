import sys
sys.path.append('/Users/apple/Documents/ML_Project/ML - 2.1/module')
from utils import *
from ngboost.learners import *
from sklearn.metrics import mean_squared_error
import numpy as np
from tqdm import tqdm

n_readout=5830
n_components=213
damping = 0.83683165
weight_scaling = 0.902327013

test_len = 1000

X_train, X_test, Y_train, Y_test = get_data(hour_num=0, transform='sin+cos',
                                            drop_time=True, scale=True)
mse_list = []
for i in tqdm(range(100)):
	esn = esn_ridge_learner(
                n_readout=n_readout,
                n_components=n_components,
                damping = damping,
                weight_scaling = weight_scaling,
                alpha=0.01).fit(X_train, Y_train)
	Y_pred = esn.predict(X_test)
	mse = mean_squared_error(Y_pred[:test_len], Y_test[:test_len])
	mse_list.append(mse)

print('Test MSE:', np.mean(mse_list)) # 0.0170636245571411

