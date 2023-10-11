import torch
import torch.nn as nn
from algorithms.edges.edgebase import Edgebase
from algorithms.optimizers.optimizer import *
from algorithms.trainmodel.models import *

class edgeSophia(Edgebase):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, alpha, eta, L,
                 local_epochs, optimizer):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, alpha, eta, L,
                         local_epochs)

        self.pre_params = []
        if (model[1] == "linear_regression"):
            self.loss = nn.MSELoss()
        elif model[1] == "logistic_regression":
            self.loss = nn.BCELoss()
        else:
            self.loss = nn.CrossEntropyLoss()#nn.NLLLoss()

        self.optimizer =  SophiaG(self.model.parameters(), lr=learning_rate, rho = 20, betas=(0.90, 0.95), weight_decay=0.0002, version=0)
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
        
        
    def train(self, epochs, glob_iter):
        self.model.train()
        
        tau = 10
        loss = 0
        iter_num = 0
        for i in range(1, self.local_epochs + 1):
            for X, Y in self.trainloader:
                logits = self.model(X)
                loss = self.loss(logits, Y) + self.regularize()
                loss.backward()
                _, self.ema_grads, self.clippings = self.optimizer.step()
                self.optimizer.zero_grad()
                iter_num += 1

                if glob_iter % tau != 0:
                    continue
                else:
                    # update hessian EMA
                    
                    logits = self.model(X)
                    samp_dist = torch.distributions.Categorical(logits=logits)
                    y_sample = samp_dist.sample()
                    loss_sampled = F.cross_entropy(logits.view(-1, logits.size(-1)), y_sample.view(-1), ignore_index=-1)
                    loss_sampled.backward()
                    self.ema_hess = self.optimizer.update_hessian()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.model.zero_grad()