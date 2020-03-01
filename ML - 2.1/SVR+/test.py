import sys
sys.path.append('/Users/apple/Documents/ML_Project/ML - 2.1/module')
import numpy as np
from utils import get_data, get_data2
from svr_plus import svr_plus

Parameters = {}
Parameters['C'] = 10
Parameters['gamma_corSpace'] = 10
Parameters['gamma_rbf'] = 1
Parameters['gamma_rbf_corSpace'] = 10
Parameters['epsilon'] = 0.1
Parameters['tol'] = 1e-4

X_train, X_test, Y_train, Y_test = get_data(hour_num=1, transform='sin+cos')
X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

svr_plus(X_train, X_test, Y_train, Y_test, X_train, Parameters)
