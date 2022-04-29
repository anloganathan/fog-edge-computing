import numpy as np
import pickle
import torch
from progressbar import progressbar
from models import NumpyModel


def init_stats_arrays(T):
    return tuple(np.zeros(T, dtype=np.float32) for i in range(4))

def run_fedavg( data_feeders, test_data, model, client_opt,  
                T, M, K, B, test_freq=1, bn_setting=0, noisy_idxs=[]):
   
    W = len(data_feeders)
    
    train_errs, train_accs, test_errs, test_accs = init_stats_arrays(T)
    
    # contains private model and optimiser BN vals (if bn_setting != 3)
    user_bn_model_vals = [model.get_bn_vals(setting=bn_setting) for w in range(W)]
    user_bn_optim_vals = [client_opt.get_bn_params(model) for w in range(W)]
    
    # global model/optimiser updated at the end of each round
    round_model = model.get_params()
    round_optim = client_opt.get_params()
    
    # stores accumulated client models/optimisers each round
    round_agg   = model.get_params()
    round_opt_agg = client_opt.get_params()
    
    for t in progressbar(range(T)):
        round_agg = round_agg.zeros_like()
        round_opt_agg = round_opt_agg.zeros_like()
        
        # select round clients and compute their weights for later sum
        user_idxs = np.random.choice(W, M, replace=False)        
        weights   = np.array([data_feeders[u].n_samples for u in user_idxs])
        weights   = weights.astype(np.float32)
        weights   /= np.sum(weights)
        
        round_n_test_users = 0
        
        for (w, user_idx) in zip(weights, user_idxs):
            # download global model/optim, update with private BN params
            model.set_params(round_model)
            client_opt.set_params(round_optim)
            model.set_bn_vals(user_bn_model_vals[user_idx], setting=bn_setting)
            client_opt.set_bn_params(user_bn_optim_vals[user_idx], 
                                        model, setting=bn_setting)
            
            # test local model if not a noisy client
            if (t % test_freq == 0) and (user_idx not in noisy_idxs):
                err, acc = model.test(  test_data[0][user_idx], 
                                        test_data[1][user_idx], 128)
                test_errs[t] += err
                test_accs[t] += acc
                round_n_test_users += 1
            
            # perform local SGD
            for k in range(K):
                x, y = data_feeders[user_idx].next_batch(B)
                err, acc = model.train_step(x, y)
                train_errs[t] += err
                train_accs[t] += acc

            # upload local model/optim to server, store private BN params
            round_agg = round_agg + (model.get_params() * w)
            round_opt_agg = round_opt_agg + (client_opt.get_params() * w)
            user_bn_model_vals[user_idx] = model.get_bn_vals(setting=bn_setting)
            user_bn_optim_vals[user_idx] = client_opt.get_bn_params(model,
                                                setting=bn_setting)
            
        # new global model is weighted sum of client models
        round_model = round_agg.copy()
        round_optim = round_opt_agg.copy()
        
        if t % test_freq == 0:
            test_errs[t] /= round_n_test_users
            test_accs[t] /= round_n_test_users
    
    train_errs /= M * K
    train_accs /= M * K
    
    return train_errs, train_accs, test_errs, test_accs

def run_fedavg_adam( data_feeders, test_data, model, server_opt, T, M, 
                        K, B, test_freq=1, bn_setting=0, noisy_idxs=[]):
    W = len(data_feeders)
    
    train_errs, train_accs, test_errs, test_accs = init_stats_arrays(T)
    
    round_model = NumpyModel(model.get_params())
    round_grads = NumpyModel(model.get_params())
    
    # contains private BN vals (if bn_setting != 3)
    user_bn_model_vals = [model.get_bn_vals(bn_setting) for w in range(W)]
    
    for t in progressbar(range(T)):
        round_grads = round_grads.zeros_like()  # round psuedogradient
        
        # select round clients and compute their weights for later sum
        user_idxs = np.random.choice(W, M, replace=False)
        weights   = np.array([data_feeders[u].n_samples for u in user_idxs])
        weights   = weights.astype(np.float32)
        weights   /= np.sum(weights)
        
        round_n_test_users = 0
        
        for (w, user_idx) in zip(weights, user_idxs):
            # download global model, update local model with private BN params
            model.set_params(round_model)
            model.set_bn_vals(user_bn_model_vals[user_idx], bn_setting)
                        
            # test local model if not a noisy client
            if (t % test_freq == 0) and (user_idx not in noisy_idxs):
                err, acc = model.test(  test_data[0][user_idx], 
                                        test_data[1][user_idx], 128)
                test_errs[t] += err
                test_accs[t] += acc
                round_n_test_users += 1
            
            # perform local SGD
            for k in range(K):
                x, y = data_feeders[user_idx].next_batch(B)
                loss, acc = model.train_step(x, y)
                train_errs[t] += loss
                train_accs[t] += acc

            # upload local model to server, store private BN params
            round_grads = round_grads + ((round_model - model.get_params()) * w)
            user_bn_model_vals[user_idx] = model.get_bn_vals(bn_setting)
        
        # update global model using psuedogradient
        round_model = server_opt.apply_gradients(round_model, round_grads)
        
        if t % test_freq == 0:
            test_errs[t] /= round_n_test_users
            test_accs[t] /= round_n_test_users
    
    train_errs /= M * K
    train_accs /= M * K
    
    return train_errs, train_accs, test_errs, test_accs







