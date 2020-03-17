from simple_esn.simple_esn import SimpleESN
from sklearn.utils import check_random_state
import numpy as np

from sklearn.linear_model import Ridge
class ESN_Ridge_learner():
    def __init__(self, n_readout=1000, n_components=100, damping=0.5,
                 weight_scaling=0.9, discard_steps=0, random_state=None, 
                 alpha=0.01):
        self.n_readout = n_readout
        self.n_components = n_components
        self.damping = damping
        self.weight_scaling = weight_scaling
        self.discard_steps = discard_steps
        self.random_state = random_state
        self.alpha = alpha

        self.ESN = SimpleESN(
            n_readout=self.n_readout, 
            n_components=self.n_components, 
            damping=self.damping,
            weight_scaling=self.weight_scaling, 
            discard_steps=self.discard_steps, 
            random_state=check_random_state(self.random_state))
        self.Ridge = Ridge(alpha=self.alpha)

# change y to np.array(y).reshape(-1,) 
    def fit(self, X, y):
        self.ESN.fit(X)
        self.Ridge.fit( self.ESN.transform(X), np.array(y).reshape(-1,) )
        return self

    def predict(self, X):
        return self.Ridge.predict( self.ESN.transform(X) )
    
    def get_params(self, deep=True):
        if deep:
            params = {
                'n_readout': self.n_readout,
                'n_components': self.n_components,
                'damping': self.damping,
                'weight_scaling': self.weight_scaling,
                'discard_steps': self.discard_steps,
                'random_state': self.random_state,
                'alpha': self.alpha}
            return params
        else:
            params = {
                'n_readout': self.n_readout,
                'n_components': self.n_components,
                'damping': self.damping,
                'weight_scaling': self.weight_scaling}
            return params


from sklearn.linear_model import Lasso
class ESN_Lasso_learner():
    def __init__(self, n_readout=1000, n_components=100, damping=0.5,
                 weight_scaling=0.9, discard_steps=0, random_state=None, 
                 alpha=0.01):
        self.n_readout = n_readout
        self.n_components = n_components
        self.damping = damping
        self.weight_scaling = weight_scaling
        self.discard_steps = discard_steps
        self.random_state = random_state
        self.alpha = alpha

        self.ESN = SimpleESN(
            n_readout=self.n_readout, 
            n_components=self.n_components, 
            damping=self.damping,
            weight_scaling=self.weight_scaling, 
            discard_steps=self.discard_steps, 
            random_state=check_random_state(self.random_state))
        self.Lasso = Lasso(alpha=self.alpha)

# change y to np.array(y).reshape(-1,) 
    def fit(self, X, y):
        self.ESN.fit(X)
        self.Lasso.fit( self.ESN.transform(X), np.array(y).reshape(-1,) )
        return self

    def predict(self, X):
        return self.Lasso.predict( self.ESN.transform(X) )
    
    def get_params(self, deep=True):
        if deep:
            params = {
                'n_readout': self.n_readout,
                'n_components': self.n_components,
                'damping': self.damping,
                'weight_scaling': self.weight_scaling,
                'discard_steps': self.discard_steps,
                'random_state': self.random_state,
                'alpha': self.alpha}
            return params
        else:
            params = {
                'n_readout': self.n_readout,
                'n_components': self.n_components,
                'damping': self.damping,
                'weight_scaling': self.weight_scaling}
            return params


from sklearn.kernel_ridge import KernelRidge
class ESN_kernel_ridge_learner():
    def __init__(self, n_readout=1000, n_components=100, damping=0.5,
                 weight_scaling=0.9, discard_steps=0, random_state=None, 
                 alpha=1, kernel="poly", degree=3):
        self.n_readout = n_readout
        self.n_components = n_components
        self.damping = damping
        self.weight_scaling = weight_scaling
        self.discard_steps = discard_steps
        self.random_state = random_state
        self.alpha = alpha
        self.kernel = kernel
        self.degree = degree

        self.ESN = SimpleESN(
            n_readout=self.n_readout, 
            n_components=self.n_components, 
            damping=self.damping,
            weight_scaling=self.weight_scaling, 
            discard_steps=self.discard_steps, 
            random_state=check_random_state(self.random_state))
        self.Kernel_Ridge = KernelRidge(
            alpha=self.alpha, 
            kernel=self.kernel, 
            gamma=None, 
            degree=self.degree, 
            coef0=1, 
            kernel_params=None)

    def fit(self, X, y):
        self.ESN.fit(X)
        self.Kernel_Ridge.fit( self.ESN.transform(X), y )
        return self

    def predict(self, X):
        return self.Kernel_Ridge.predict( self.ESN.transform(X) )

    def get_params(self, deep=True):
        if deep:
            params = {
                'n_readout': self.n_readout,
                'n_components': self.n_components,
                'damping': self.damping,
                'weight_scaling': self.weight_scaling,
                'discard_steps': self.discard_steps,
                'random_state': self.random_state,
                'alpha': self.alpha,
                'kernel': self.kernel,
                'degree': self.degree}
            return params
        else:
            params = {
                'n_readout': self.n_readout,
                'n_components': self.n_components,
                'damping': self.damping,
                'weight_scaling': self.weight_scaling}
            return params


from sklearn.svm import LinearSVR
class ESN_linear_svr_learner():
    def __init__(self, n_readout=1000, n_components=100, damping=0.5,
                 weight_scaling=0.9, discard_steps=0, random_state=None, 
                 epsilon=0.0, C=1.0, max_iter=1000):
        self.n_readout = n_readout
        self.n_components = n_components
        self.damping = damping
        self.weight_scaling = weight_scaling
        self.discard_steps = discard_steps
        self.random_state = random_state
        self.epsilon = epsilon
        self.C = C
        self.max_iter = max_iter


        self.ESN = SimpleESN(
            n_readout=self.n_readout, 
            n_components=self.n_components, 
            damping=self.damping,
            weight_scaling=self.weight_scaling, 
            discard_steps=self.discard_steps, 
            random_state=check_random_state(self.random_state))
        self.Linear_SVR = LinearSVR(
            epsilon=self.epsilon, 
            tol=1e-4, 
            C=self.C, 
            loss='epsilon_insensitive', 
            fit_intercept=True, 
            intercept_scaling=1., 
            dual=True, 
            verbose=0, 
            random_state=None,
            max_iter=self.max_iter)

    def fit(self, X, y):
        self.ESN.fit(X)
        self.Linear_SVR.fit( self.ESN.transform(X), y )
        return self

    def predict(self, X):
        return self.Linear_SVR.predict( self.ESN.transform(X) )

    def get_params(self, deep=True):
        if deep:
            params = {
                'n_readout': self.n_readout,
                'n_components': self.n_components,
                'damping': self.damping,
                'weight_scaling': self.weight_scaling,
                'discard_steps': self.discard_steps,
                'random_state': self.random_state,
                'epsilon': self.epsilon,
                'C': self.C,
                'max_iter': self.max_iter}
            return params
        else:
            params = {
                'n_readout': self.n_readout,
                'n_components': self.n_components,
                'damping': self.damping,
                'weight_scaling': self.weight_scaling}
            return params


from sklearn.tree import DecisionTreeRegressor
class ESN_decision_tree_learner():
    def __init__(self, n_readout=1000, n_components=100, damping=0.5,
                 weight_scaling=0.9, discard_steps=0, random_state=None, 
                 criterion="mse", max_depth=None):
        self.n_readout = n_readout
        self.n_components = n_components
        self.damping = damping
        self.weight_scaling = weight_scaling
        self.discard_steps = discard_steps
        self.random_state = random_state
        self.criterion = criterion
        self.max_depth = max_depth

        self.ESN = SimpleESN(
            n_readout=self.n_readout, 
            n_components=self.n_components, 
            damping=self.damping,
            weight_scaling=self.weight_scaling, 
            discard_steps=self.discard_steps, 
            random_state=check_random_state(self.random_state))
        self.Decision_Tree = DecisionTreeRegressor(
            criterion=self.criterion,
            splitter="best",
            max_depth=self.max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.,
            max_features=None,
            random_state=None,
            max_leaf_nodes=None,
            min_impurity_decrease=0.,
            min_impurity_split=None,
            presort=False)

    def fit(self, X, y):
        self.ESN.fit(X)
        self.Decision_Tree.fit( self.ESN.transform(X), y )
        return self

    def predict(self, X):
        return self.Decision_Tree.predict( self.ESN.transform(X) )

    def get_params(self, deep=True):
        if deep:
            params = {
                'n_readout': self.n_readout,
                'n_components': self.n_components,
                'damping': self.damping,
                'weight_scaling': self.weight_scaling,
                'discard_steps': self.discard_steps,
                'random_state': self.random_state,
                'criterion': self.criterion,
                'max_depth': self.max_depth}
            return params
        else:
            params = {
                'n_readout': self.n_readout,
                'n_components': self.n_components,
                'damping': self.damping,
                'weight_scaling': self.weight_scaling}
            return params

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
