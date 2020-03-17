from simple_esn import SimpleESN
import numpy as np
n_samples, n_features = 10, 5
np.random.seed(0)
X = np.random.randn(n_samples, n_features)
esn = SimpleESN(n_readout = 2)
echoes = esn.fit_transform(X)
