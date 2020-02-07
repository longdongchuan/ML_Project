from utils import *
from ngboost.learners import *

X_train, X_test, Y_train, Y_test = get_data(hour_num=0, transform='cos',
                                            drop_time=True, scale=True)
model_test(Base=default_linear_learner(alpha=0.1),
           X_train=X_train, X_test=X_test,
           Y_train=Y_train, Y_test=Y_test,
          n_estimators=500, verbose_eval=100)