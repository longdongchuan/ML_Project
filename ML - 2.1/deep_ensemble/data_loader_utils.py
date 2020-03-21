import numpy as np
import sys
sys.path.append('/Users/apple/Documents/ML_Project/ML - 2.1/module')
from utils import get_data, get_data2
from simple_esn.simple_esn import SimpleESN
from sklearn.utils import check_random_state

class DataLoader_Spain():

    def __init__(self, args, esn_param=None, box_cox=False):

        X_train, X_test, Y_train, Y_test = get_data(
            hour_num=1, transform='sin+cos',
            train_index=[6426,10427],
            test_index=[14389,15390],
            return_y_scaler=False, 
            box_cox=box_cox)
        
        if esn_param is not None:
            ESN = SimpleESN(
                n_readout=esn_param['n_readout'],
                n_components=esn_param['n_components'],
                damping=esn_param['damping'],
                weight_scaling=esn_param['weight_scaling'], 
                discard_steps=esn_param['discard_steps'], 
                random_state=check_random_state(esn_param['random_state']))
            ESN.fit(X_train)
            X_train = ESN.transform(X_train)
            X_test = ESN.transform(X_test)
            print('ESN param:', esn_param)

        self.xs = np.array(X_train, dtype=np.float32)
        self.ys = np.array(Y_train, dtype=np.float32).reshape(len(Y_train), 1)
        self.test_xs = np.array(X_test, dtype=np.float32)
        self.test_ys = np.array(Y_test, dtype=np.float32).reshape(len(Y_test), 1)

        # Standardize input features
        self.input_mean = np.mean(self.xs, 0)
        self.input_std = np.std(self.xs, 0)
        # self.xs = (self.xs - self.input_mean)/self.input_std

        # Target mean and std
        self.target_mean = np.mean(self.ys, 0)[0]
        self.target_std = np.std(self.ys, 0)[0]

        self.batch_size = args.batch_size

    def next_batch(self):

        indices = np.random.choice(np.arange(len(self.xs)), size=self.batch_size)
        x = self.xs[indices, :]
        y = self.ys[indices, :]

        return x, y

    def get_data(self):

        return self.xs, self.ys

    def get_test_data(self):

        return self.test_xs, self.test_ys


class DataLoader_US():

    def __init__(self, args, esn_param=None, box_cox=False):
        
        X_train, X_test, Y_train, Y_test= get_data2(
            hour_num=1, transform='sin+cos',
            train_index=[3001,7002],
            test_index=[2000,3001],
            return_y_scaler=False,
            drop_else=True, 
            box_cox=box_cox)
        
        if esn_param is not None:
            ESN = SimpleESN(
                n_readout=esn_param['n_readout'],
                n_components=esn_param['n_components'],
                damping=esn_param['damping'],
                weight_scaling=esn_param['weight_scaling'], 
                discard_steps=esn_param['discard_steps'], 
                random_state=check_random_state(esn_param['random_state']))
            ESN.fit(X_train)
            X_train = ESN.transform(X_train)
            X_test = ESN.transform(X_test)
            print('ESN param:', esn_param)

        self.xs = np.array(X_train, dtype=np.float32)
        self.ys = np.array(Y_train, dtype=np.float32).reshape(len(Y_train), 1)
        self.test_xs = np.array(X_test, dtype=np.float32)
        self.test_ys = np.array(Y_test, dtype=np.float32).reshape(len(Y_test), 1)

        # Standardize input features
        self.input_mean = np.mean(self.xs, 0)
        self.input_std = np.std(self.xs, 0)
        # self.xs = (self.xs - self.input_mean)/self.input_std

        # Target mean and std
        self.target_mean = np.mean(self.ys, 0)[0]
        self.target_std = np.std(self.ys, 0)[0]

        self.batch_size = args.batch_size

    def next_batch(self):

        indices = np.random.choice(np.arange(len(self.xs)), size=self.batch_size)
        x = self.xs[indices, :]
        y = self.ys[indices, :]

        return x, y

    def get_data(self):

        return self.xs, self.ys

    def get_test_data(self):

        return self.test_xs, self.test_ys        