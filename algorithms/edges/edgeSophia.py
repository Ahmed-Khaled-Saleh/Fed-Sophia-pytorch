import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from algorithms.edges.edgebase import Edgebase
from algorithms.optimizers.optimizer import *
from algorithms.trainmodel.models import *

class edgeSophia(Edgebase):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, alpha, eta, L,
                 local_epochs, optimizer, tau, rho, exp):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, alpha, eta, L,
                         local_epochs)

        self.pre_params = []
        if (model[1] == "linear_regression"):
            self.loss = nn.MSELoss()
        elif model[1] == "logistic_regression":
            self.loss = nn.BCELoss()
        else:
            self.loss = nn.CrossEntropyLoss()#nn.NLLLoss()

        
        # make a list of zeros-like tensors for EMA with the same shape as model params
        # import pdb; pdb.set_trace()
        self.ema_grads = [torch.zeros_like(item, memory_format=torch.preserve_format) for _, item in enumerate(self.model.parameters())]
        self.ema_hess = [torch.zeros_like(item, memory_format=torch.preserve_format) for _, item in enumerate(self.model.parameters())]        
        self.clippings = [torch.zeros_like(item, memory_format=torch.preserve_format) for _, item in enumerate(self.model.parameters())]
        # p  = [param for param in self.model.parameters()]
        # self.ema_grads = [torch.zeros_like(p[0],  memory_format=torch.preserve_format), torch.zeros_like(p[1],  memory_format=torch.preserve_format)]
        # self.ema_grads = [torch.zeros_like(item[i], memory_format=torch.preserve_format) for i, item in enumerate(p)]
        # self.ema_hess = [torch.zeros_like(item[i], memory_format=torch.preserve_format) for i, item in enumerate(p)]
        # self.ema_hess = [torch.zeros_like(p[0],  memory_format=torch.preserve_format), torch.zeros_like(p[1],  memory_format=torch.preserve_format)]
        self.tau = tau
        self.exp = exp
        self.lr = learning_rate
        self.rho = rho
        self.optimizer =  SophiaG(self.model.parameters(), lr=self.lr, rho = self.rho, betas=(0.965, 0.95), weight_decay=0.05, version=0)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=5)
    def get_lr(self, it):
        warmup_iters = 500 # how many steps to warm up for
        lr_decay_iters = 1500 
        min_lr = 0.0001
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return self.lr * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (self.lr - min_lr)
    
    def train(self, epochs, glob_iter):
        self.model.train()
        grad_clip = 1
        
        loss = 0
        iter_num = 0
        for i in range(1, self.local_epochs + 1):
            # self.lr = self.get_lr(glob_iter)
            # for param_group in self.optimizer.param_groups:
            #     param_group['lr'] = self.lr

            for X, Y in self.trainloader:
                logits = self.model(X)
                loss = self.loss(logits, Y) + self.regularize()
                loss.backward()
                _, self.ema_grads, self.clippings = self.optimizer.step()
                self.optimizer.zero_grad()
                

                if grad_clip != 0.0:
                    total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    if total_norm.item() > grad_clip:
                        clip_time += 1
                        print("======================clipping time=======================")
                        self.exp.log_metric("clipping time",clip_time)
                
                if glob_iter % self.tau != 0:
                    continue
                else:
                    # update hessian EMA
                    #X_batch, _ = self.get_next_train_batch()
                    logits = self.model(X) #if self.algorithm != "logistic_regression" else self.model.linear(X)	
                    samp_dist = torch.distributions.Categorical(logits=logits)
                    y_sample = samp_dist.sample()
                    loss_sampled = F.cross_entropy(logits.view(-1, logits.size(-1)), y_sample.view(-1), ignore_index=-1)
                    loss_sampled.backward()
                    self.ema_hess = self.optimizer.update_hessian()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.model.zero_grad()
                iter_num += 1
            # for X, y in self.testloader:
            #     self.model.eval()
            #     logits = self.model(X)
            #     val_loss = self.loss(logits, y) + self.regularize()
            # self.scheduler.step(val_loss)