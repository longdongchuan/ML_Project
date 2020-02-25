import sys
sys.path.append('/Users/apple/Documents/ML_Project/ML - 2.1/module')
from utils import *
from ngboost.learners import *
from sklearn.metrics import mean_squared_error
import numpy as np
from tqdm import tqdm

#n_readout = 1000
#n_components = 100
#damping = 0.5
#weight_scaling = 0.9
#test_len = 1000
#alpha = 0.01


n_readout = 3462
n_components = 23
damping = 0.26215546327467487
weight_scaling = 0.6234509481681756
alpha = 0.4649085531487292
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
                alpha=alpha).fit(X_train, Y_train)
	Y_pred = esn.predict(X_test)
	mse = mean_squared_error(Y_pred[:test_len], Y_test[:test_len])
	mse_list.append(mse)

print('Test MSE:', np.mean(mse_list)) # 0.016665708498075166
