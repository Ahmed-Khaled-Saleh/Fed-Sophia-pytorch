
from comet_ml import Experiment
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from algorithms.server.server import Server
from algorithms.trainmodel.models import *
from utils.plot_utils import *
import torch
torch.manual_seed(0)

def main(experiment, dataset, algorithm, model, batch_size, learning_rate, alpha, eta, L, rho, num_glob_iters,
         local_epochs, optimizer, numedges, times, commet, gpu):

    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

    for i in range(times):
        print("---------------Running time:------------",i)

        # Generate model
        if(model == "mclr"):
            if(dataset == "human_activity"):
                model = Mclr_Logistic(561,6).to(device), model
            else:
                model = Mclr_Logistic().to(device), model

        if(model == "linear_regression"):
            model = Linear_Regression(40,1).to(device), model

        if model == "logistic_regression":
            model = Logistic_Regression(300).to(device), model
        
        if model == "MLP" and dataset == "a9a":
            model = DNN( input_dim = 123, output_dim = 2).to(device), model
        if model == "MLP" and dataset == "human_activity":
            model = DNN( input_dim = 561, mid_dim = 561, output_dim = 6).to(device), model
        if model == "MLP" and dataset == "Mnist":
            model = DNN().to(device), model

        # select algorithm
        if(commet):
            experiment.set_name(dataset + "_" + algorithm + "_" + model[1] + "_" + str(batch_size) + "b_" + str(learning_rate) + "lr_" + str(alpha) + "al_" + str(eta) + "eta_" + str(L) + "L_" + str(rho) + "p_" +  str(num_glob_iters) + "ge_"+ str(local_epochs) + "le_"+ str(numedges) +"u")
        server = Server(experiment, device, dataset, algorithm, model, batch_size, learning_rate, alpha, eta,  L, num_glob_iters, local_epochs, optimizer, numedges, i)
        
        server.train()
        server.test()


    

sophia_params = {
    "dataset": "a9a",
    "algorithm": "Sophia",
    "model": "MLP",
    "batch_size": 0,
    "learning_rate": 0.001,
    "alpha": 0.03,
    "eta": 0.5,
    "L": 1e-5,
    "rho": 20,
    "num_glob_iters": 500,
    "local_epochs": 1,
    "optimizer": "Sophia",
    "numedges": 30,
    "times": 1,
    "commet": True,
    "gpu": 0
}

DONE_params = {
    "dataset": "Mnist",
    "algorithm": "DONE",
    "model": "MLP",
    "batch_size": 0,
    "learning_rate": 1,
    "alpha": 0.03,
    "eta": 1.0,
    "L": 1e-3,
    "rho": 0.01,
    "num_glob_iters": 500,
    "local_epochs": 40,
    "optimizer": "DONE",
    "numedges": 30,
    "times": 1,
    "commet": True,
    "gpu": 0
}

Newton_params = {
    "dataset": "Mnist",
    "algorithm": "Newton",
    "model": "MLP",
    "batch_size": 0,
    "learning_rate": 1,
    "alpha": 0.03,
    "eta": 1.0,
    "L": 1e-3,
    "rho": 0.01,
    "num_glob_iters": 500,
    "local_epochs": 40,
    "optimizer": "Newton",
    "numedges": 30,
    "times": 1,
    "commet": True,
    "gpu": 0
}

experiment = Experiment(
        api_key="lhdQnruUATiAZPyU7Qp2zFiVX",
        project_name="sophia-supplement",
        workspace="ahmed-khaled-saleh",
    )
main(experiment, **sophia_params)
