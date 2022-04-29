import torch 
import numpy as np
import operator
import numbers

class FLModel(torch.nn.Module):    
    def __init__(self, device):
        super(FLModel, self).__init__()
        self.optim      = None
        self.device     = device
        self.loss_fn    = None
        self.bn_layers  = []        # any model BN layers must be added to this 

    def set_optim(self, optim, init_optim=True):
        self.optim = optim
        if init_optim:
            self.empty_step()
        
    def empty_step(self):
        raise NotImplementedError()

    def get_params(self):
       
        ps = [np.copy(p.data.cpu().numpy()) for p in list(self.parameters())]
        for bn in self.bn_layers:
            ps.append(np.copy(bn.running_mean.cpu().numpy()))
            ps.append(np.copy(bn.running_var.cpu().numpy()))
        
        return NumpyModel(ps)
    
    def get_bn_vals(self, setting=0):

        if setting not in [0, 1, 2, 3]:
            raise ValueError('Setting must be in: {0, 1, 2, 3}')
    
        vals = []
        
        if setting == 3:
            return vals
        
        with torch.no_grad():
            # add gamma, beta
            if setting in [0, 1]:
                for bn in self.bn_layers:
                    vals.append(np.copy(bn.weight.cpu().numpy()))
                    vals.append(np.copy(bn.bias.cpu().numpy()))
            
            # add mu, sigma
            if setting in [0, 2]:
                for bn in self.bn_layers:
                    vals.append(np.copy(bn.running_mean.cpu().numpy()))
                    vals.append(np.copy(bn.running_var.cpu().numpy()))
        return vals


    def set_bn_vals(self, vals, setting=0):
      
        if setting not in [0, 1, 2, 3]:
            raise ValueError('Setting must be in: {0, 1, 2, 3}')
        
        if setting == 3:
            return
        
        with torch.no_grad():
            i = 0
            # set gamma, beta
            if setting in [0, 1]:
                for bn in self.bn_layers:
                    bn.weight.copy_(torch.tensor(vals[i]))
                    bn.bias.copy_(torch.tensor(vals[i+1]))
                    i += 2
                    
            # set mu, sigma
            if setting in [0, 2]:
                for bn in self.bn_layers:
                    bn.running_mean.copy_(torch.tensor(vals[i]))
                    bn.running_var.copy_(torch.tensor(vals[i+1]))
                    i += 2
    
    def set_params(self, params):
        i = 0
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.tensor(params[i]))
                i += 1
                
            # set mu, sigma
            for bn in self.bn_layers:
                bn.running_mean.copy_(torch.tensor(params[i]))
                bn.running_var.copy_(torch.tensor(params[i+1]))
                i += 2
   
    def forward(self, x):
        raise NotImplementedError()
        
    def calc_acc(self, logits, y):
    
        raise NotImplementedError()
    
    def train_step(self, x, y):
    
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        acc = self.calc_acc(logits, y)
        self.optim.zero_grad()
        loss.backward()        
        self.optim.step()
        
        return loss.item(), acc

    def test(self, x, y, B):
    
        self.eval()
        n_batches = int(np.ceil(x.shape[0] / B))
        loss = 0.0
        acc = 0.0
        
        with torch.no_grad():
            for b in range(n_batches):
                logits = self.forward(x[b*B:(b+1)*B])
                loss += self.loss_fn(logits, y[b*B:(b+1)*B]).item()
                acc += self.calc_acc(logits, y[b*B:(b+1)*B])
        self.train()
        
        return loss/n_batches, acc/n_batches

class MNISTModel(FLModel):
    def __init__(self, device):
       
        super(MNISTModel, self).__init__(device)
        
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        
        self.fc0    = torch.nn.Linear(784, 200).to(device)
        self.relu0  = torch.nn.ReLU().to(device)
        
        self.fc1    = torch.nn.Linear(200, 200).to(device)
        self.relu1  = torch.nn.ReLU().to(device)
        
        self.out    = torch.nn.Linear(200, 10).to(device)

        self.bn0 = torch.nn.BatchNorm1d(200).to(device)        
        
        self.bn_layers = [self.bn0]
        
    def forward(self, x):

        a = self.bn0(self.relu0(self.fc0(x)))
        b = self.relu1(self.fc1(a))
        
        return self.out(b)
        
    def calc_acc(self, logits, y):
        
        return (torch.argmax(logits, dim=1) == y).float().mean()
        
    def empty_step(self):
   
        self.train_step(torch.zeros((2, 784), 
                                    device=self.device, 
                                    dtype=torch.float32), 
                        torch.zeros((2), 
                                    device=self.device,
                                    dtype=torch.int32).long())
                                    
class NumpyModel():
    
    def __init__(self, params):
        self.params = params
        
    def copy(self):
        return NumpyModel([np.copy(p) for p in self.params])
        
    def zeros_like(self):
        return NumpyModel([np.zeros_like(p) for p in self.params])
        
    def _op(self, other, f):
        if np.isscalar(other):
            new_params = [f(p, other) for p in self.params]
            
        elif isinstance(other, NumpyModel):
            new_params = [f(p, o) for (p, o) in zip(self.params, other.params)]
            
        else:
            raise ValueError('Incompatible type for op: {}'.format(other))
        
        return NumpyModel(new_params)
        
        
    def abs(self):
        return NumpyModel([np.absolute(p) for p in self.params])
        
    def __add__(self, other):
        return self._op(other, operator.add)
        
    def __radd__(self, other):
        return self._op(other, operator.add)

    def __sub__(self, other):
        return self._op(other, operator.sub)
        
    def __mul__(self, other):
        return self._op(other, operator.mul)
        
    def __rmul__(self, other):
        return self._op(other, operator.mul)
        
    def __truediv__(self, other):
        return self._op(other, operator.truediv)
        
    def __pow__(self, other):
        return self._op(other, operator.pow)
        
    def __getitem__(self, key):
        return self.params[key]
        
    def __len__(self):
        return len(self.params)
        
    def __iter__(self):
        for p in self.params:
            yield p
