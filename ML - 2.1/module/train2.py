from utils import *
from ngboost.learners import *

X_train, X_test, Y_train, Y_test = get_data(hour_num=0, transform='sin+cos',
                                            drop_time=True, scale=True)
model_test(Base=esn_linear_svr_learner(n_readout=1000,
                                        n_components=100,
                                        epsilon=0.0,
                                        C=0.02,
                                        max_iter=10000),
           X_train=X_train, X_test=X_test,
           Y_train=Y_train, Y_test=Y_test,
          n_estimators=500, verbose_eval=10)

print(Base.get_params())