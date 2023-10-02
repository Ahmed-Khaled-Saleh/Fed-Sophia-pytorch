import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from algorithms.edges.edgebase import Edgebase
from algorithms.optimizers.optimizer import *


# Implementation for FedAvg clients

class edgeSophia(Edgebase):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, alpha, eta, L,
                 local_epochs, optimizer):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, alpha, eta, L,
                         local_epochs)

        if (model[1] == "linear_regression"):
            self.loss = nn.MSELoss()
        elif model[1] == "logistic_regression":
            self.loss = nn.BCELoss()
        else:
            self.loss = nn.NLLLoss()

        '''
        AdamW:
            lr=0.001, 
            betas=(0.9, 0.999),
            weight_decay=0.01)

        - Choose lr to be slightly smaller than the learning rate that you would use for AdamW 
            or 3 - 5 times the learning rate that you would use for Lion.
        - Tune rho to make the proportion of the clipped coordinates stable and in a proper range
        - If the loss blows up, slightly decrease the learning rate or increase rho.
        - Always use about 2x larger weight decay than what you would use for AdamW.


        '''
        # lr=0.0005, betas=(0.90, 0.90), rho=0.7, weight_decay=0.02    -> converges
        # lr=0.001, betas=(0.90, 0.90), rho=0.7, weight_decay=0.02    -> converges
        # lr=0.001, betas=(0.90, 0.90), rho=0.8, weight_decay=0.02    -> converges
        # lr=0.003, betas=(0.90, 0.90), rho=0.7, weight_decay=0.02    -> converges
        
        # lr=0.01, betas=(0.90, 0.90), rho=0.5, weight_decay=0.02    -> converges (a9a)
        # lr=0.05, betas=(0.90, 0.90), rho=0.5, weight_decay=0.02   -> converges (phihsing)


        self.optimizer = SophiaG(self.model.parameters(), lr=0.005, betas=(0.965, 0.99), rho=0.001, weight_decay=0.002, version=0)
        p  = [param for param in self.model.parameters()]
        self.ema_grads = [torch.zeros_like(p[0],  memory_format=torch.preserve_format), torch.zeros_like(p[1],  memory_format=torch.preserve_format)]
        self.ema_hess = [torch.zeros_like(p[0],  memory_format=torch.preserve_format), torch.zeros_like(p[1],  memory_format=torch.preserve_format)]
        
    def get_ema_grads(self):
        for i, param in enumerate(self.ema_grads):
            param.detach() 
        return self.ema_grads

    def get_ema_hess(self):
        for i, param in enumerate(self.ema_hess):
            param.detach()        
        return self.ema_hess


    def train(self, epochs, glob_iter):

        # Only update once time
        iter_num = 0
        k = 10
        block_size = 1024
        total_bs = len(self.trainloader)
        bs = total_bs * block_size
        # import pdb; pdb.set_trace()
        for epoch in range(self.local_epochs):
            for X, y in self.trainloaderfull:
                
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                # y = y.to(torch.float32)
                l = self.loss(output, y)
                l.backward()
                self.ema_grads = self.optimizer.step()[1]
                self.optimizer.zero_grad(set_to_none=True)
                iter_num += 1
            

                if glob_iter % k != 0:
                    continue
                else:
                    # update hessian EMA
                    # import pdb;pdb.set_trace()
                    # sample a minibatch from the trainloader
                    logits = self.model.linear(X)
                    samp_dist = torch.distributions.Categorical(logits= logits)
                    y_sample = samp_dist.sample()
                    loss_sampled = F.cross_entropy(logits.view(-1, logits.size(-1)), y_sample.view(-1), ignore_index=-1)
                    loss_sampled.backward()
                    self.ema_hess = self.optimizer.update_hessian()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.model.zero_grad()
                    
            
