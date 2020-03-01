from SV_Quad import *
    # Parameters = {}
    # Parameters['C'] = 10
    # Parameters['gamma_corSpace'] = 10
    # Parameters['gamma_rbf'] = 1
    # Parameters['gamma_rbf_corSpace'] = 10
    # Parameters['epsilon'] = 0.1
    # Parameters['tol'] = 1e-4
def svr_plus(X_train, X_test, Y_train, Y_test, X_star,  Parameters):
    K = rbf_kernel(X_train, Y=None, gamma = Parameters['gamma_rbf'])
    reg = SVR_Plus_Quad(X_train, X_star, Y_train, K ,Parameters)
    Kern = rbf_kernel(X_train, X_test, gamma=Parameters['gamma_rbf'])
    y_predict = np.dot(np.transpose((reg['alpha_star'][:, 0] - reg['alpha'][:, 0])[:, None]) , Kern) + reg['bias']
    print ('\n Mean Squared Error: ' + str(mean_squared_error(Y_test,np.squeeze(y_predict))))