# -*- coding: utf-8 -*-
#   - Implementation of SVM, SVM+, SVR, SVR+ using quadratic programming.
#     The code below is based on the publication "A new learning paradigm:
#     Learning using privileged information" of Vapnik and Vashist and was used
#     for our paper "Predicting privileged information for height estimation"
#
#   - For more information please refer to the tutorial which explains in detail
#     how the parameters are defined and how the optimization problem is formulated in each case
#
#   - If you find our code and or our method useful please cite our work:
#     N. Sarafianos, C. Nikou, and I.A. Kakadiaris. ‘‘Predicting Privileged Information for Height Estimation,’’ ICPR 2016
#
#   - If you're interested in a LIBSVM implementation of SVM+ (only for classification) check D. Pechyony's webpage: http://www.cs.technion.ac.il/~pechyony/
#
#   - For bugs/questions feel free to contact me at nsarafia[at]central.uh.edu.

#     Licence:
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.

from cvxopt import matrix, solvers
import numpy as np
from numpy.matlib import repmat
from scipy.spatial.distance import pdist, squareform
import scipy
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import sys
solvers.options['show_progress'] = False
def SVR_Quad(X,Y,K,Parameters):
    l = Y.shape[0]
    Q = np.concatenate((np.concatenate((K,-K),axis=1),np.concatenate((-K,K),axis=1)),axis=0)
    p = np.concatenate((np.squeeze(Parameters['epsilon']*np.ones((l,1))) + Y, np.squeeze(Parameters['epsilon']*np.ones((l,1))) - Y),axis=0)
    A = np.concatenate((np.ones((1,l)),-np.ones((1,l))),axis=1)
    b = np.array([0.])
    
    G = np.concatenate((-np.eye(2*l),(np.eye(2*l))),axis=0)
    h = np.concatenate((np.zeros((2*l,1)),Parameters['C']*np.ones((2*l,1))),axis=0)
    sol=solvers.qp(matrix(Q), matrix(p), matrix(G), matrix(h), matrix(A), matrix(b))

    alpha = np.array(sol['x'][0:l])
    alpha_star = np.array(sol['x'][l:])

    alpha[alpha < Parameters['tol']] = 0
    alpha_star[alpha_star < Parameters['tol']] = 0

    bias = 0
    for n in range(len(alpha)):
        bias += Y[n] - np.sum(np.transpose((alpha_star[:, 0] - alpha[:, 0])[:, None]) * K[n, :])
    bias /= len(alpha)

    clf = {}
    clf['K'] = K
    clf['alpha'] = alpha
    clf['alpha_star'] = alpha_star
    clf['bias'] = bias

    return (clf)

def SVR_Plus_Quad(X,X_star,Y,K,Parameters):
    l = Y.shape[0]
    K_star = rbf_kernel(X_star, Y=None, gamma=Parameters['gamma_rbf_corSpace'])
    
    row1 = np.concatenate((K + K_star/float(Parameters['gamma_corSpace']), -K, K_star/float(Parameters['gamma_corSpace']), np.zeros((l,l))),axis=1)
    row2 = np.concatenate((-K, K + K_star/float(Parameters['gamma_corSpace']), np.zeros((l,l)), K_star/float(Parameters['gamma_corSpace'])),axis=1)
    row3 = np.concatenate((K_star/float(Parameters['gamma_corSpace']), np.zeros((l,l)), K_star/float(Parameters['gamma_corSpace']),np.zeros((l,l))),axis=1)
    row4 = np.concatenate((np.zeros((l,l)), K_star/float(Parameters['gamma_corSpace']), np.zeros((l,l)), K_star/float(Parameters['gamma_corSpace'])),axis=1)
    
    Q = np.concatenate((row1,row2,row3,row4),axis=0)
    b = np.array([[0.],[l*Parameters['C']],[l*Parameters['C']]])
    A1 = np.concatenate((np.ones((1,l)),-np.ones((1,l)),np.zeros((1,2*l))),axis=1)
    A2 = np.concatenate((np.zeros((1,l)),np.ones((1,l)),np.zeros((1,l)),np.ones((1,l))),axis=1)
    A3 = np.concatenate((np.ones((1,l)),np.zeros((1,l)),np.ones((1,l)),np.zeros((1,l))),axis=1)
    A = np.concatenate((A1,A2,A3),axis=0)
    
    p1 = repmat(sum(K_star+np.transpose(K_star)),1,4)*(-Parameters['C']/float(Parameters['gamma_corSpace']))
    p2 = np.concatenate((np.transpose(Y)+Parameters['epsilon']*np.ones((1,l)),-np.transpose(Y)+Parameters['epsilon']*np.ones((1,l)),np.zeros((1,2*l))),axis=1)
    p = np.transpose(p1+p2)
    G = -np.eye(4*l)
    h= np.zeros((4*l,1))

    sol=solvers.qp(matrix(Q), matrix(p), matrix(G), matrix(h), matrix(A), matrix(b))

    alpha = np.array(sol['x'][0:l])
    alpha_star = np.array(sol['x'][l:2*l])
    beta = np.array(sol['x'][2*l:3*l])
    beta_star = np.array(sol['x'][3*l:4*l])
    
    alpha[alpha<Parameters['tol']]=0
    alpha_star[alpha_star<Parameters['tol']]=0
    beta[beta<Parameters['tol']]=0
    beta_star[beta_star<Parameters['tol']]=0

    bias = 0
    for n in range(len(alpha)):
        bias += Y[n] - np.sum(np.transpose((alpha_star[:, 0] - alpha[:, 0])[:, None]) * K[n, :])
    bias /= len(alpha)

    clf = {}
    clf['K'] = K
    clf['alpha'] = alpha
    clf['alpha_star'] = alpha_star
    clf['bias'] = bias

    return (clf)
    
    
def SVM_Quad(X,Y,K,Parameters):
    l = Y.shape[0]
    Q = Y[:,None]*np.transpose(Y)*K
    p = -np.ones((l))
    A = np.transpose(Y)
    b = np.array([0.])
    G = np.concatenate((-np.eye(l),(np.eye(l))),axis=0)
    h = np.concatenate((np.zeros((l,1)),Parameters['C']*np.ones((l,1))),axis=0)
    sol = solvers.qp(matrix(Q), matrix(np.squeeze(p)), matrix(G), matrix(np.squeeze(h)), matrix(np.transpose(A[:,None]).astype(float)), matrix(b)) #matrix(A)

    alpha = np.array(sol['x'])
    sv = alpha > Parameters['tol']
    ind = np.arange(len(alpha))[sv[:, 0]]
    alpha = alpha[sv[:, 0]]
    sv_x = X[sv[:, 0]]
    sv_y = Y[sv[:, 0]]

    bias = 0
    for n in range(len(alpha)):
        bias += sv_y[n] - np.sum(alpha[:, 0] * sv_y * K[ind[n], sv[:, 0]])
    bias /= len(alpha)

    clf = {}
    clf['K'] = K
    clf['sv_x'] = sv_x
    clf['sv_y'] = sv_y
    clf['sv'] = sv
    clf['alpha'] = alpha
    clf['bias'] = bias

    return (clf)


def SVM_Plus_Quad(X,X_star,Y,K,Parameters):
    l = Y.shape[0]
    K_svm = np.multiply(np.transpose(Y)*Y,K)
    K_star = rbf_kernel(X_star, Y=None, gamma=Parameters['gamma_rbf_corSpace'])

    Q1 = np.concatenate((K_svm + K_star/float(Parameters['gamma_corSpace']),K_star/float(Parameters['gamma_corSpace'])),axis=1)
    Q2 = np.concatenate((K_star/float(Parameters['gamma_corSpace']),K_star/float(Parameters['gamma_corSpace'])),axis=1)
    Q = np.concatenate((Q1,Q2),axis=0)
    A = np.concatenate((np.ones((1, 2 * l)), np.concatenate((np.transpose(Y[:, None]), np.zeros((1, l))), axis=1)), axis=0)
    b = np.array([[l*Parameters['C']],[0]])
    G = -np.eye(2*l)
    h = np.zeros((2*l,1))
    
    p = repmat(sum(K_star+np.transpose(K_star)),1,2)*(-Parameters['C']/float(2*Parameters['gamma_corSpace'])) - \
        np.concatenate((np.ones((1,l)),np.zeros((1,l))),axis=1)
    p = np.transpose(p)   
    sol=solvers.qp(matrix(Q,tc='d'), matrix(p,tc='d'), matrix(G,tc='d'), matrix(h,tc='d'), matrix(A,tc='d'), matrix(b,tc='d'))


    alpha = np.array(sol['x'][0:l])
    beta = np.array(sol['x'][l:2*l])
    sv_a = alpha > Parameters['tol']
    sv_b = beta > Parameters['tol']
    ind_a = np.arange(len(alpha))[sv_a[:, 0]]
    ind_b = np.arange(len(beta))[sv_b[:, 0]]

    alpha = alpha[sv_a[:, 0]]
    sv_x = X[sv_a[:, 0]]
    sv_y = Y[sv_a[:, 0]]

    bias = 0
    for n in range(len(alpha)):
        bias += sv_y[n] - np.sum(alpha[:, 0] * sv_y * K[ind_a[n], sv_a[:, 0]])
    bias /= len(alpha)

    clf = {}
    clf['K'] = K
    clf['sv_x'] = sv_x
    clf['sv_y'] = sv_y
    clf['sv'] = sv_a
    clf['alpha'] = alpha
    clf['bias'] = bias

    return (clf)

from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel


def gen_non_lin_separable_data():
    mean1 = [-1, 2]
    mean2 = [1, -1]
    mean3 = [4, -4]
    mean4 = [-4, 4]
    cov = [[1.0, 0.8], [0.8, 1.0]]
    X1 = np.random.multivariate_normal(mean1, cov, 50)
    X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, 50)
    X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
    y2 = np.ones(len(X2)) * -1
    return X1, y1, X2, y2


def split_train(X1, y1, X2, y2):
    X1_train = X1[:90]
    y1_train = y1[:90]
    X2_train = X2[:90]
    y2_train = y2[:90]
    X_train = np.vstack((X1_train, X2_train))
    y_train = np.hstack((y1_train, y2_train))
    return X_train, y_train


def split_test(X1, y1, X2, y2):
    X1_test = X1[90:]
    y1_test = y1[90:]
    X2_test = X2[90:]
    y2_test = y2[90:]
    X_test = np.vstack((X1_test, X2_test))
    y_test = np.hstack((y1_test, y2_test))
    return X_test, y_test

def main():
    # Example of how to run the code
    method = input('Select method: \n 1 for SVR \n 2 for SVR+ \n 3 for SVM \n 4 for SVM+ \n Anything else for Exit ')
    if method ==3 or method == 4: # Classification
        X1, y1, X2, y2 = gen_non_lin_separable_data()
        X, Y = split_train(X1, y1, X2, y2)
        X_test, Y_test = split_test(X1, y1, X2, y2)
    else: # Regression
        n_samples, n_features = 100, 5
        np.random.seed(0)
        Y = np.random.randn(n_samples)
        X = np.random.randn(n_samples, n_features)
        X_test = X[90:, :]
        Y_test = Y[90:]
        X = X[0:90, :]
        Y = Y[0:90]

    # Select Parameters (you should cross-validate these thoroughly)
    Parameters = {}
    Parameters['C'] = 10
    Parameters['gamma_corSpace'] = 10
    Parameters['gamma_rbf'] = 1
    Parameters['gamma_rbf_corSpace'] = 10
    Parameters['epsilon'] = 0.1
    Parameters['tol'] = 1e-4

    K = rbf_kernel(X,Y=None, gamma = Parameters['gamma_rbf'])



    # Training
    if method == 1: # SVR
        clf = SVR_Quad(X,Y,K,Parameters)
    elif method == 2: # SVR+
        Xstar = X # Using X* = X, please replace with your data
        clf = SVR_Plus_Quad(X,Xstar,Y,K,Parameters)
    elif method == 3: # SVM
        clf = SVM_Quad(X,Y,K,Parameters)
    elif method == 4: # SVM+
        Xstar = X  # Using X* = X, please replace with your data
        clf = SVM_Plus_Quad(X,Xstar,Y,K,Parameters)
    else:
        sys.exit()

    if method ==3 or method == 4:
        Kern = rbf_kernel(clf['sv_x'], X_test, gamma=Parameters['gamma_rbf'])
        y_predict = np.dot(clf['alpha'][:, 0] * clf['sv_y'], Kern)
        signed_distance = y_predict + clf['bias']
        Pred = np.sign(signed_distance)
        print ('\nClassification Accuracy: ' + str(accuracy_score(Y_test, Pred)))
    else:
        Kern = rbf_kernel(X, X_test, gamma=Parameters['gamma_rbf'])
        y_predict = np.dot(np.transpose((clf['alpha_star'][:, 0] - clf['alpha'][:, 0])[:, None]) , Kern) + clf['bias']
        print ('\n Mean Squared Error: ' + str(mean_squared_error(Y_test,np.squeeze(y_predict))))

if __name__ == "__main__":
    main()