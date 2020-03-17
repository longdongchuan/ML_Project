#---------default learners--------#
from sklearn.tree import DecisionTreeRegressor
def default_tree_learner(depth=3):
    return DecisionTreeRegressor(
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=depth,
        splitter='best')


from sklearn.linear_model import Ridge
def default_linear_learner(alpha=0.0):
    return Ridge(alpha=alpha)


#---------self-define learners--------#
from sklearn.linear_model import Lasso
def lasso_learner(alpha=0.5, max_iter=1000):
    return Lasso(
        alpha=alpha, 
        fit_intercept=True, 
        normalize=False, 
        precompute=False, 
        copy_X=True, 
        max_iter=max_iter, 
        tol=0.0001, 
        warm_start=False, 
        positive=False, 
        random_state=None, 
        selection='cyclic')


from sklearn.kernel_ridge import KernelRidge
def kernel_ridge_learner(alpha=1, kernel="poly", degree=3):
    return KernelRidge(
        alpha=alpha, 
        kernel=kernel, 
        gamma=None, 
        degree=degree, 
        coef0=1, 
        kernel_params=None)


from sklearn.svm import LinearSVR
def linear_svr_learner(epsilon=0.0, C=1.0, max_iter=1000):
    return LinearSVR(
        epsilon=epsilon, 
        tol=1e-4, 
        C=C, 
        loss='epsilon_insensitive', 
        fit_intercept=True, 
        intercept_scaling=1., 
        dual=True, 
        verbose=0, 
        random_state=None,
        max_iter=max_iter)


#---------self-define ESN-Based learners--------#
from ngboost.esn_learners import ESN_Ridge_learner
def esn_ridge_learner(
    n_readout=1000, 
    n_components=100, 
    damping=0.5,
    weight_scaling=0.9, 
    discard_steps=0, 
    random_state=None, 
    alpha=0.01):
    return ESN_Ridge_learner(
        n_readout=n_readout, 
        n_components=n_components, 
        damping=damping,
        weight_scaling=weight_scaling, 
        discard_steps=discard_steps, 
        random_state=random_state, 
        alpha=alpha)


from ngboost.esn_learners import ESN_Lasso_learner
def esn_lasso_learner(
    n_readout=1000, 
    n_components=100, 
    damping=0.5,
    weight_scaling=0.9, 
    discard_steps=0, 
    random_state=None, 
    alpha=0.01):
    return ESN_Lasso_learner(
        n_readout=n_readout, 
        n_components=n_components, 
        damping=damping,
        weight_scaling=weight_scaling, 
        discard_steps=discard_steps, 
        random_state=random_state, 
        alpha=alpha)  


from ngboost.esn_learners import ESN_kernel_ridge_learner
def esn_kernel_ridge_learner(
    n_readout=1000, 
    n_components=100, 
    damping=0.5,
    weight_scaling=0.9, 
    discard_steps=0, 
    random_state=None, 
    alpha=1, 
    kernel="poly", 
    degree=3):
    return ESN_kernel_ridge_learner(
        n_readout=n_readout, 
        n_components=n_components, 
        damping=damping,
        weight_scaling=weight_scaling, 
        discard_steps=discard_steps, 
        random_state=random_state, 
        alpha=alpha,
        kernel=kernel, 
        degree=degree)

from ngboost.esn_learners import ESN_linear_svr_learner
def esn_linear_svr_learner(
    n_readout=1000, 
    n_components=100, 
    damping=0.5,
    weight_scaling=0.9, 
    discard_steps=0, 
    random_state=None, 
    epsilon=0.0, 
    C=1.0, 
    max_iter=1000):
    return ESN_linear_svr_learner(
        n_readout=n_readout, 
        n_components=n_components, 
        damping=damping,
        weight_scaling=weight_scaling, 
        discard_steps=discard_steps, 
        random_state=random_state, 
        epsilon=epsilon, 
        C=C, 
        max_iter=max_iter)


from ngboost.esn_learners import ESN_decision_tree_learner
def esn_decision_tree_learner(
    n_readout=1000, 
    n_components=100, 
    damping=0.5,
    weight_scaling=0.9, 
    discard_steps=0, 
    random_state=None, 
    criterion="mse", 
    max_depth=None):
    return ESN_decision_tree_learner(
        n_readout=n_readout, 
        n_components=n_components, 
        damping=damping,
        weight_scaling=weight_scaling, 
        discard_steps=discard_steps, 
        random_state=random_state, 
        criterion=criterion, 
        max_depth=max_depth)