import os
import numpy as np
import pickle
from data_utils import *
from models import *
from optimisers import *
import argparse
from sys import argv
from fl_algs import *
import torch


def get_fname(a):
  
    fname = '_'.join([  k+'-'+str(v) for (k, v) in vars(a).items() 
                        if not v is None])
    return fname + '.pkl'

def save_data(data, fname):
    with open(fname, 'wb') as f:
        pickle.dump(data, f)

def any_in_list(x, y):
    return any(x_i in y for x_i in x)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dset', required=True, choices=['mnist'], 
                        help='Federated dataset')
    parser.add_argument('-alg', required=True, help='Federated optimiser', 
                        choices=[   'fedavg', 'fedavg-adam', 'fedadam', 
                                    'pfedme', 'perfedavg'])
    parser.add_argument('-C', required=True, type=float, 
                        help='Fraction of clients selected per round')
    parser.add_argument('-B', required=True, type=int, help='Client batch size')
    parser.add_argument('-T', required=True, type=int, help='Total rounds')
    parser.add_argument('-E', required=True, type=int, help='Client num epochs')
    parser.add_argument('-device', required=True, choices=['cpu', 'gpu'], 
                        help='Training occurs on this device')
    parser.add_argument('-W', required=True, type=int, 
                        help='Total workers to split data across')
    parser.add_argument('-seed', required=True, type=int, help='Random seed')
    parser.add_argument('-lr', required=True, type=float, 
                        help='Client learning rate')
    parser.add_argument('-noisy_frac', required=True, type=float, 
                        help='Fraction of noisy clients')
    
    # specific arguments for different FL algorithms
    if any_in_list(['fedavg', 'fedavg-adam', 'fedadam'], argv):
        parser.add_argument('-bn_private', choices=['usyb', 'us', 'yb', 'none'],
                            required=True, 
                            help='Patch parameters to keep private')
        
    if any_in_list(['fedadam'], argv):
        parser.add_argument('-server_lr', required=True, type=float, 
                            help='Server learning rate')
    
    if any_in_list(['fedavg-adam', 'fedadam'], argv):
        parser.add_argument('-beta1', required=True, type=float, 
                            help='Only required for FedAdam, 0 <= beta1 < 1')
        parser.add_argument('-beta2', required=True, type=float, 
                            help='Only required for FedAdam, 0 <= beta2 < 1')
        parser.add_argument('-epsilon', required=True, type=float, 
                            help='Only required for FedAdam, 0 < epsilon << 1')
    
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda:0' if args.device=='gpu' else 'cpu')

    # load data 
    print('Loading data...')
    if args.dset == 'mnist':
        train, test = load_mnist('./MNIST_data', args.W, iid=False, 
                                    user_test=True)
        model       = MNISTModel(device)
        noise_std   = 3.0
        steps_per_E = int(np.round(60000 / (args.W * args.B)))

    # add noise to data
    noisy_imgs, noisy_idxs = add_noise_to_frac( train[0], args.noisy_frac, 
                                                noise_std)
    train = (noisy_imgs, train[1])
 
    # convert to pytorch tensors
    feeders   = [   PyTorchDataFeeder(x, torch.float32, y, 'long', device) 
                    for (x, y) in zip(train[0], train[1])]
    test_data = (   [to_tensor(x, device, torch.float32) for x in test[0]],
                    [to_tensor(y, device, 'long') for y in test[1]])

  
    fname             = get_fname(args)
    M                 = int(args.W * args.C)
    K                 = steps_per_E * args.E
    str_to_bn_setting = {'usyb':0, 'yb':1, 'us':2, 'none':3}
    if args.alg in ['fedavg', 'fedavg-adam', 'fedadam']:
        bn_setting = str_to_bn_setting[args.bn_private]
    
    # run experiment
    print('Starting experiment...')
    if args.alg == 'fedavg':
        client_optim = ClientSGD(model.parameters(), lr=args.lr)
        model.set_optim(client_optim)
        data = run_fedavg(  feeders, test_data, model, client_optim, args.T, M, 
                            K, args.B, bn_setting=bn_setting, 
                            noisy_idxs=noisy_idxs)
    elif args.alg == 'fedadam':
        client_optim = ClientSGD(model.parameters(), lr=args.lr)
        model.set_optim(client_optim)
        server_optim = ServerAdam(  model.get_params(), args.server_lr, 
                                    args.beta1, args.beta2, args.epsilon)
        data         = run_fedavg_adam(   feeders, test_data, model, 
                                            server_optim, args.T, M, 
                                            K, args.B, 
                                            bn_setting=bn_setting,
                                            noisy_idxs=noisy_idxs)
                        
    elif args.alg == 'fedavg-adam':
        client_optim = ClientAdam(  model.parameters(), lr=args.lr, 
                                    betas=(args.beta1, args.beta2), 
                                    eps=args.epsilon)
        model.set_optim(client_optim)
        data = run_fedavg(  feeders, test_data, model, client_optim, args.T, M, 
                            K, args.B, bn_setting=bn_setting, 
                            noisy_idxs=noisy_idxs)

    save_data(data, fname)
    print('Data saved to: {}'.format(fname))

if __name__ == '__main__':
    main()
