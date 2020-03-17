from simple_esn.simple_esn import SimpleESN
from sklearn.linear_model import Ridge
from sklearn.utils import check_random_state

class ESN_Ridge_learner():
    def __init__(self, n_readout=1000, n_components=100, damping=0.5,
                 weight_scaling=0.9, discard_steps=0, random_state=None, 
                 alpha=0.01):
        self.Ridge = Ridge(alpha=alpha)
        self.ESN = SimpleESN(
            n_readout=n_readout, 
            n_components=n_components, 
            damping=damping,
            weight_scaling=weight_scaling, 
            discard_steps=discard_steps, 
            random_state=check_random_state(random_state))

    def fit(self, X, y):
        self.ESN.fit(X)
        self.Ridge.fit( self.ESN.transform(X), y )
        return self

    def predict(self, X):
        return self.Ridge.predict( self.ESN.transform(X) )