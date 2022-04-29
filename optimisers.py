import numpy as np
import torch
from torch.optim import Optimizer
from models import NumpyModel

class ServerOpt():
    def apply_gradients(self, model, grads):
        raise NotImplementedError()

class ServerAdam(ServerOpt):   
    def __init__(self, params, lr, beta1, beta2, epsilon):
      
        self.m = params.zeros_like()
        self.v = params.zeros_like()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
    def apply_gradients(self, model, grads):
    
        self.m = (self.beta1 * self.m) + (1 - self.beta1) * grads
        self.v = (self.beta2 * self.v) + (1 - self.beta2) * (grads ** 2)
        
        # uses constant learning rate as per AdaptiveFedOpt paper
        return model - (self.m * self.lr) / ((self.v ** 0.5) + self.epsilon)

class ClientOpt():
    def get_params(self):
        raise NotImplementedError()
    def set_params(self, params):     
        raise NotImplementedError()
    def get_bn_params(self, setting=0):
        raise NotImplementedError()  
    def set_bn_params(self, params, setting=0):
        raise NotImplementedError()

class ClientSGD(torch.optim.SGD, ClientOpt):
    def __init__(self, params, lr):
        super(ClientSGD, self).__init__(params, lr)
        
    def get_params(self):
        return NumpyModel([])
        
    def set_params(self, params):
        pass
        
    def get_bn_params(self, model, setting=0):
        return []
        
    def set_bn_params(self, params, model, setting=0):
        pass
        
    def step(self, closure=None, beta=None):
    
        loss = None
        if closure is not None:
            loss = closure

        # apply SGD update rule
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if beta is None:
                    p.data.add_(d_p, alpha=-group['lr'])
                else:     
                    p.data.add_(d_p, alpha=-beta)
        
        return loss

class ClientAdam(torch.optim.Adam, ClientOpt):

    def __init__(   self, params, lr=0.001, betas=(0.9, 0.999), 
                    eps=1e-07, weight_decay=0, amsgrad=False):
       
        super(ClientAdam, self).__init__(   params, lr, betas, eps, 
                                            weight_decay, amsgrad)
    
    def get_bn_params(self, model, setting=0):

        if setting in [2, 3]:
            return []
        
        # order is (weight m, weight v, bias m, bias v)
        params = []
        for bn in model.bn_layers:
            weight = self.state[bn.weight]
            bias = self.state[bn.bias]
            params.append(np.copy(weight['exp_avg'].cpu().numpy()))
            params.append(np.copy(weight['exp_avg_sq'].cpu().numpy()))
            params.append(np.copy(bias['exp_avg'].cpu().numpy()))
            params.append(np.copy(bias['exp_avg_sq'].cpu().numpy()))
        
        return params
        
    def set_bn_params(self, params, model, setting=0):
       
        if setting in [2, 3]:
            return
        
        i = 0
        for bn in model.bn_layers:
            weight = self.state[bn.weight]
            bias = self.state[bn.bias]
            weight['exp_avg'].copy_(torch.tensor(params[i]))
            weight['exp_avg_sq'].copy_(torch.tensor(params[i+1]))
            bias['exp_avg'].copy_(torch.tensor(params[i+2]))
            bias['exp_avg_sq'].copy_(torch.tensor(params[i+3]))
            i += 4
        
    def get_params(self):
       
        params = []
        for key in self.state.keys():
            params.append(self.state[key]['step'])
            params.append(self.state[key]['exp_avg'].cpu().numpy())
            params.append(self.state[key]['exp_avg_sq'].cpu().numpy())
            
        return NumpyModel(params)
    
    def set_params(self, params):
        i = 0
        for key in self.state.keys():
            self.state[key]['step'] = params[i]
            self.state[key]['exp_avg'].copy_(torch.tensor(params[i+1]))
            self.state[key]['exp_avg_sq'].copy_(torch.tensor(params[i+2]))
            i += 3

            