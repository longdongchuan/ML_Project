from warnings import warn

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import logsumexp
from autograd import grad

from scipy.optimize import linear_sum_assignment, minimize
from scipy.special import gammaln, digamma, polygamma

from ssm.primitives import solve_symm_block_tridiag

def compute_state_overlap(z1, z2, K1=None, K2=None):
    assert z1.dtype == int and z2.dtype == int
    assert z1.shape == z2.shape
    assert z1.min() >= 0 and z2.min() >= 0

    K1 = z1.max() + 1 if K1 is None else K1
    K2 = z2.max() + 1 if K2 is None else K2

    overlap = np.zeros((K1, K2))
    for k1 in range(K1):
        for k2 in range(K2):
            overlap[k1, k2] = np.sum((z1 == k1) & (z2 == k2))
    return overlap


def find_permutation(z1, z2, K1=None, K2=None):
    overlap = compute_state_overlap(z1, z2, K1=K1, K2=K2)
    K1, K2 = overlap.shape

    tmp, perm = linear_sum_assignment(-overlap)
    assert np.all(tmp == np.arange(K1)), "All indices should have been matched!"

    # Pad permutation if K1 < K2
    if K1 < K2:
        unused = np.array(list(set(np.arange(K2)) - set(perm)))
        perm = np.concatenate((perm, unused))

    return perm


def rle(stateseq):
    """
    Compute the run length encoding of a discrete state sequence.

    E.g. the state sequence [0, 0, 1, 1, 1, 2, 3, 3]
         would be encoded as ([0, 1, 2, 3], [2, 3, 1, 2])

    [Copied from pyhsmm.util.general.rle]

    Parameters
    ----------
    stateseq : array_like
        discrete state sequence

    Returns
    -------
    ids : array_like
        integer identities of the states

    durations : array_like (int)
        length of time in corresponding state
    """
    pos, = np.where(np.diff(stateseq) != 0)
    pos = np.concatenate(([0],pos+1,[len(stateseq)]))
    return stateseq[pos[:-1]], np.diff(pos)


def random_rotation(n, theta=None):
    if theta is None:
        # Sample a random, slow rotation
        theta = 0.5 * np.pi * np.random.rand()

    if n == 1:
        return np.random.rand() * np.eye(1)

    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    out = np.eye(n)
    out[:2, :2] = rot
    q = np.linalg.qr(np.random.randn(n, n))[0]
    return q.dot(out).dot(q.T)


def ensure_args_are_lists(f):
    def wrapper(self, datas, inputs=None, masks=None, tags=None, **kwargs):
        datas = [datas] if not isinstance(datas, (list, tuple)) else datas

        M = (self.M,) if isinstance(self.M, int) else self.M
        assert isinstance(M, tuple)

        if inputs is None:
            inputs = [np.zeros((data.shape[0],) + M) for data in datas]
        elif not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        if masks is None:
            masks = [np.ones_like(data, dtype=bool) for data in datas]
        elif not isinstance(masks, (list, tuple)):
            masks = [masks]

        if tags is None:
            tags = [None] * len(datas)
        elif not isinstance(tags, (list, tuple)):
            tags = [tags]

        return f(self, datas, inputs=inputs, masks=masks, tags=tags, **kwargs)

    return wrapper


def ensure_variational_args_are_lists(f):
    def wrapper(self, arg0, datas, inputs=None, masks=None, tags=None, **kwargs):
        datas = [datas] if not isinstance(datas, (list, tuple)) else datas

        try:
            M = (self.M,) if isinstance(self.M, int) else self.M
        except:
            # self does not have M if self is a variational posterior object
            # in that case, arg0 is a model, which does have an M parameter
            M = (arg0.M,) if isinstance(arg0.M, int) else arg0.M

        assert isinstance(M, tuple)

        if inputs is None:
            inputs = [np.zeros((data.shape[0],) + M) for data in datas]
        elif not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        if masks is None:
            masks = [np.ones_like(data, dtype=bool) for data in datas]
        elif not isinstance(masks, (list, tuple)):
            masks = [masks]

        if tags is None:
            tags = [None] * len(datas)
        elif not isinstance(tags, (list, tuple)):
            tags = [tags]

        return f(self, arg0, datas, inputs=inputs, masks=masks, tags=tags, **kwargs)

    return wrapper


def ensure_args_not_none(f):
    def wrapper(self, data, input=None, mask=None, tag=None, **kwargs):
        assert data is not None

        M = (self.M,) if isinstance(self.M, int) else self.M
        assert isinstance(M, tuple)
        input = np.zeros((data.shape[0],) + M) if input is None else input

        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return f(self, data, input=input, mask=mask, tag=tag, **kwargs)
    return wrapper


def ensure_slds_args_not_none(f):
    def wrapper(self, variational_mean, data, input=None, mask=None, tag=None, **kwargs):
        assert variational_mean is not None
        assert data is not None
        M = (self.M,) if isinstance(self.M, int) else self.M
        assert isinstance(M, tuple)
        input = np.zeros((data.shape[0],) + M) if input is None else input
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return f(self, variational_mean, data, input=input, mask=mask, tag=tag, **kwargs)
    return wrapper


def logistic(x):
    return 1. / (1 + np.exp(-x))


def logit(p):
    return np.log(p / (1 - p))


def softplus(x):
    return np.log1p(np.exp(x))


def inv_softplus(y):
    return np.log(np.exp(y) - 1)


def one_hot(z, K):
    z = np.atleast_1d(z).astype(int)
    assert np.all(z >= 0) and np.all(z < K)
    shp = z.shape
    N = z.size
    zoh = np.zeros((N, K))
    zoh[np.arange(N), np.arange(K)[np.ravel(z)]] = 1
    zoh = np.reshape(zoh, shp + (K,))
    return zoh


def relu(x):
    return np.maximum(0, x)


def replicate(x, state_map, axis=-1):
    """
    Replicate an array of shape (..., K) according to the given state map
    to get an array of shape (..., R) where R is the total number of states.

    Parameters
    ----------
    x : array_like, shape (..., K)
        The array to be replicated.

    state_map : array_like, shape (R,), int
        The mapping from [0, K) -> [0, R)
    """
    assert state_map.ndim == 1
    assert np.all(state_map >= 0) and np.all(state_map < x.shape[-1])
    return np.take(x, state_map, axis=axis)

def collapse(x, state_map, axis=-1):
    """
    Collapse an array of shape (..., R) to shape (..., K) by summing
    columns that map to the same state in [0, K).

    Parameters
    ----------
    x : array_like, shape (..., R)
        The array to be collapsed.

    state_map : array_like, shape (R,), int
        The mapping from [0, K) -> [0, R)
    """
    R = x.shape[axis]
    assert state_map.ndim == 1 and state_map.shape[0] == R
    K = state_map.max() + 1
    return np.concatenate([np.sum(np.take(x, np.where(state_map == k)[0], axis=axis),
                                  axis=axis, keepdims=True)
                           for k in range(K)], axis=axis)


def newtons_method_block_tridiag_hessian(
    x0, obj, grad_func, hess_func,
    tolerance=1e-4, maxiter=100):
    """
    Newton's method to minimize a positive definite function with a
    block tridiagonal Hessian matrix.
    Algorithm 9.5, Boyd & Vandenberghe, 2004.
    """
    x = x0
    is_converged = False
    count = 0
    while not is_converged:
        H_diag, H_lower_diag = hess_func(x)
        g = grad_func(x)
        dx = -1.0 * solve_symm_block_tridiag(H_diag, H_lower_diag, g)
        lambdasq = np.dot(g.ravel(), -1.0*dx.ravel())
        if lambdasq / 2.0 <= tolerance:
            is_converged = True
            break
        stepsize = backtracking_line_search(x, dx, obj, g)
        x = x + stepsize * dx
        count += 1
        if count > maxiter:
            break

    if not is_converged:
        warn("Newton's method failed to converge in {} iterations. "
             "Final mean abs(dx): {}".format(maxiter, np.mean(np.abs(dx))))

    return x

def backtracking_line_search(x0, dx, obj, g, stepsize = 1.0, min_stepsize=1e-8,
                             alpha=0.2, beta=0.7):
    """
    A backtracking line search for the step size in Newton's method.
    Algorithm 9.2, Boyd & Vandenberghe, 2004.
    - dx is the descent direction
    - g is the gradient evaluated at x0
    - alpha in (0,0.5) is fraction of decrease in objective predicted  by
        a linear extrapolation that we will accept
    - beta in (0,1) is step size reduction factor
    """
    x = x0

    # criterion: stop when f(x + stepsize * dx) < f(x) + \alpha * stepsize * f'(x)^T dx
    f_term = obj(x)
    grad_term = alpha * np.dot(g.ravel(), dx.ravel())

    # decrease stepsize until criterion is met
    # or stop at minimum step size
    while stepsize > min_stepsize:
        fx = obj(x+ stepsize*dx)
        if np.isnan(fx) or fx > f_term + grad_term*stepsize:
            stepsize *= beta
        else:
            break

    return stepsize