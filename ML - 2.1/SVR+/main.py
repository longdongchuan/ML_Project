from SV_Quad import *
def main():
    # Example of how to run the code
    method = 2
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