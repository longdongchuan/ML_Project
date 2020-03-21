import numpy as np
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import ipdb

import sys
sys.path.append('/Users/apple/Documents/ML_Project/ML - 2.1/deep_ensemble')
from model import MLPGaussianRegressor
from model import MLPDropoutGaussianRegressor

from data_loader_utils import DataLoader_US
save_figure_path = '/Users/apple/Documents/ML_Project/ML - 2.1/deep_ensemble/figure/US_result.png'
save_csv_path = '/Users/apple/Documents/ML_Project/ML - 2.1/deep_ensemble/result/US_esn.csv'

def main():

    parser = argparse.ArgumentParser()
    # Ensemble size
    parser.add_argument('--ensemble_size', type=int, default=5,
                        help='Size of the ensemble')
    # Maximum number of iterations
    parser.add_argument('--max_iter', type=int, default=5000,
                        help='Maximum number of iterations')
    # Batch size
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Size of batch')
    # Epsilon for adversarial input perturbation
    parser.add_argument('--epsilon', type=float, default=1e-2,
                        help='Epsilon for adversarial input perturbation')
    # Alpha for trade-off between likelihood score and adversarial score
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Trade off parameter for likelihood score and adversarial score')
    # Learning rate
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='Learning rate for the optimization')
    # Gradient clipping value
    parser.add_argument('--grad_clip', type=float, default=100.,
                        help='clip gradients at this value')
    # Learning rate decay
    parser.add_argument('--decay_rate', type=float, default=0.99,
                        help='Decay rate for learning rate')
    # Dropout rate (keep prob)
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='Keep probability for dropout')
    
    # deep_ensemble param
    args = parser.parse_args()
    # esn_param
    esn_param = {'n_readout': 1000,
                 'n_components': 20, 
                 'damping': 0.5,
                 'weight_scaling': 0.9, 
                 'discard_steps': 0, 
                 'random_state': None}
    # box-cox transform
    box_cox=False
    # start train
    train_ensemble(args, esn_param=esn_param, box_cox=box_cox)


def ensemble_mean_var(ensemble, xs, sess):
    en_mean = 0
    en_var = 0

    for model in ensemble:
        feed = {model.input_data: xs}
        mean, var = sess.run([model.mean, model.var], feed)
        en_mean += mean
        en_var += var + mean**2

    en_mean /= len(ensemble)
    en_var /= len(ensemble)
    en_var -= en_mean**2
    return en_mean, en_var



def train_ensemble(args, esn_param=None, box_cox=False):
    
    # Input data
    dataLoader = DataLoader_US(args, esn_param=esn_param, box_cox=box_cox)
    # Layer sizes
    sizes = [dataLoader.xs.shape[1], 50, 50, 2]

    ensemble = [MLPGaussianRegressor(args, sizes, 'model'+str(i)) for i in range(args.ensemble_size)]

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for model in ensemble:
            sess.run(tf.assign(model.output_mean, dataLoader.target_mean))
            sess.run(tf.assign(model.output_std, dataLoader.target_std))

        for itr in range(args.max_iter):

            for model in ensemble:

                x, y = dataLoader.next_batch()

                feed = {model.input_data: x, model.target_data: y}
                _, nll, m, v = sess.run([model.train_op, model.nll, model.mean, model.var], feed)

                if itr % 100 == 0:
                    sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** (itr/100))))
                    print('itr', itr, 'nll', nll)

        test_ensemble(ensemble, sess, dataLoader)


def test_ensemble(ensemble, sess, dataLoader):
    test_xs, test_ys = dataLoader.get_test_data()
    mean, var = ensemble_mean_var(ensemble, test_xs, sess)
    std = np.sqrt(var)
    upper = mean + 3*std
    lower = mean - 3*std
    
    test_xs_scaled = dataLoader.input_mean + dataLoader.input_std*test_xs

    plt.plot(test_xs_scaled, test_ys, 'b-')
    plt.plot(test_xs_scaled, mean, 'r-')

    plt.fill_between(test_xs_scaled[:, 0], lower[:, 0], upper[:, 0], color='yellow', alpha=0.5)
    plt.show()
    # plt.savefig(save_figure_path)    
    predict_dist = pd.DataFrame({'mean': mean.flatten().tolist(), 'std': std.flatten().tolist()})
    predict_dist.to_csv(save_csv_path)


if __name__ == '__main__':
    main()
