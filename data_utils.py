import gzip 
import numpy as np 
import torch
import pickle

class PyTorchDataFeeder():

    def __init__(   self, x, x_dtype, y, y_dtype, device, 
                    cast_device=None, transform=None):
        if x_dtype == 'long':
            self.x = torch.tensor(  x, device=device, 
                                    requires_grad=False, 
                                    dtype=torch.int32).long()
        else:
            self.x = torch.tensor(  x, device=device, 
                                    requires_grad=False, 
                                    dtype=x_dtype)
        
        if y_dtype == 'long':
            self.y = torch.tensor(  y, device=device, 
                                    requires_grad=False, 
                                    dtype=torch.int32).long()
        else:
            self.y = torch.tensor(  y, device=device, 
                                    requires_grad=False, 
                                    dtype=y_dtype)
        
        self.idx = 0
        self.n_samples = x.shape[0]
        self.cast_device = cast_device
        self.transform = transform
        self.shuffle_data()
        
    def shuffle_data(self):
        ord = torch.randperm(self.n_samples)
        self.x = self.x[ord]
        self.y = self.y[ord]
        
    def next_batch(self, B):
       
        if B == -1:
            x = self.x
            y = self.y
            self.shuffle_data()
            
        elif self.idx + B > self.n_samples:
            # if batch wraps around to start, add some samples from the start
            extra = (self.idx + B) - self.n_samples
            x = torch.cat((self.x[self.idx:], self.x[:extra]))
            y = torch.cat((self.y[self.idx:], self.y[:extra]))
            self.shuffle_data()
            self.idx = extra
            
        else:
            x = self.x[self.idx:self.idx+B]
            y = self.y[self.idx:self.idx+B]
            self.idx += B
            
        if not self.cast_device is None:
            x = x.to(self.cast_device)
            y = y.to(self.cast_device)

        if not self.transform is None:
            x = self.transform(x)

        return x, y



def load_mnist(data_dir, W, iid, user_test=False):
   
    train_x_fname = data_dir + '/train-images-idx3-ubyte.gz'
    train_y_fname = data_dir + '/train-labels-idx1-ubyte.gz'
    test_x_fname = data_dir + '/t10k-images-idx3-ubyte.gz'
    test_y_fname = data_dir + '/t10k-labels-idx1-ubyte.gz'

    # load MNIST files
    with gzip.open(train_x_fname) as f:
        x_train = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
        x_train = x_train.astype(np.float32) / 255.0

    with gzip.open(train_y_fname) as f:
        y_train = np.copy(np.frombuffer(f.read(), np.uint8, offset=8))

    with gzip.open(test_x_fname) as f:
        x_test = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
        x_test = x_test.astype(np.float32) / 255.0        

    with gzip.open(test_y_fname) as f:
        y_test = np.copy(np.frombuffer(f.read(), np.uint8, offset=8))
    
    # split into iid/non-iid and users
    if iid:
        x_train, y_train = co_shuffle_split(x_train, y_train, W)
        if user_test:
            x_test, y_test = co_shuffle_split(x_test, y_test, W)
    
    else:
        x_train, y_train, assign = shard_split(x_train, y_train, W, W*2)
        if user_test:
            x_test, y_test, _ = shard_split(x_test, y_test, W, W*2, assign)
    
    return (x_train, y_train), (x_test, y_test)


def add_noise_to_frac(xs, frac, std):
  
    idxs = np.random.choice(len(xs), int(len(xs)*frac), replace=False)
    
    new_xs = []
    for i in range(len(xs)):
        if i in idxs:
            noisy = xs[i] + np.random.normal(0.0, std, size=xs[i].shape)
            new_xs.append(np.clip(noisy, 0.0, 1.0))
        else:
            new_xs.append(np.copy(xs[i]))
    
    return new_xs, idxs



def co_shuffle_split(x, y, W):
   
    order = np.random.permutation(x.shape[0])
    x_split = np.array_split(x[order], W)
    y_split = np.array_split(y[order], W)
    
    return x_split, y_split



def shard_split(x, y, W, n_shards, assignment=None):
   
    order = np.argsort(y)
    x_sorted = x[order]
    y_sorted = y[order]
    
    # split data into shards of (mostly) the same index 
    x_shards = np.array_split(x_sorted, n_shards)
    y_shards = np.array_split(y_sorted, n_shards)
    
    if assignment is None:
        assignment = np.array_split(np.random.permutation(n_shards), W)
    
    x_sharded = []
    y_sharded = []
    
    # assign each worker two shards from the random assignment
    for w in range(W):
        x_sharded.append(np.concatenate([x_shards[i] for i in assignment[w]]))
        y_sharded.append(np.concatenate([y_shards[i] for i in assignment[w]]))
        
    return x_sharded, y_sharded, assignment


def to_tensor(x, device, dtype):
   
    if dtype == 'long':
        return torch.tensor(x, device=device, 
                            requires_grad=False, 
                            dtype=torch.int32).long()
    else:
        return torch.tensor(x, device=device, 
                            requires_grad=False, 
                            dtype=dtype)
