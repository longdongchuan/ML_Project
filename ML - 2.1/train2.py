from utils import *
from ngboost.learners import *

X_train, X_test, Y_train, Y_test = get_data(hour_num=6, transform='cos',
                                            drop_time=True, scale=True)
# ## esn_kernel_ridge_learner

# In[6]:


model_test(Base=esn_kernel_ridge_learner(n_readout=1000,
                                         n_components=100,
                                         alpha=1, 
                                         kernel='poly',
                                         degree=3),
           X_train=X_train, X_test=X_test,
           Y_train=Y_train, Y_test=Y_test,
          n_estimators=500, verbose_eval=5)

