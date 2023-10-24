
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
         local_epochs, optimizer, numedges, times, commet, gpu, tau):

    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

    in_dim = {
        "a9a": 123,
        "Mnist": 784,
        "w8a": 300,
        "Fashion_Mnist": 784,
        "human_activity": 561,
        "phishing": 68,
        "linear_regression": 40,

    }

    out_dim = {
        "a9a": 2,
        "Mnist": 10,
        "Fashion_Mnist": 10,
        "w8a": 2,
        "human_activity": 6,
        "phishing": 2,
    }

    import eco2ai

    pro_name = dataset + "_" + algorithm + "_" + model + "_" + str(batch_size) + "b_" + str(learning_rate) + "lr_" + str(alpha) + "al_" + str(eta) + "eta_" + str(L) + "L_" + str(rho) + "p_" +  str(num_glob_iters) + "ge_"+ str(local_epochs) + "le_"+ str(numedges) +"u" + str(tau) + "tau"
    tracker = eco2ai.Tracker(project_name=pro_name
                            , experiment_description=f"training the {algorithm} model")

    tracker.start()


    for i in range(times):
        print("---------------Running time:------------",i)

        # Generate model
        if(model == "mclr"):
            model = Mclr_CrossEntropy(input_dim = in_dim[dataset], output_dim = out_dim[dataset]).to(device), model

        if(model == "linear_regression"):
            model = Linear_Regression(input_dim = in_dim[dataset]).to(device), model

        if model == "logistic_regression":
            model = Logistic_Regression(input_dim = in_dim[dataset]).to(device), model
        
        if model == "MLP":
            model = DNN(input_dim = in_dim[dataset], output_dim = out_dim[dataset]).to(device), model
        
        if model == 'CNN':
            model = Net().to(device), model

        if(commet):
            experiment.set_name(dataset + "_" + algorithm + "_" + model[1] + "_" + str(batch_size) + "b_" + str(learning_rate) + "lr_" + str(alpha) + "al_" + str(eta) + "eta_" + str(L) + "L_" + str(rho) + "p_" +  str(num_glob_iters) + "ge_"+ str(local_epochs) + "le_"+ str(numedges) +"u" + str(tau) + "tau")
        server = Server(experiment, device, dataset, algorithm, model, batch_size, learning_rate, alpha, eta,  L, num_glob_iters, local_epochs, optimizer, numedges, i, tau, rho)
        
        server.train()
        server.test()
    tracker.stop()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="human_activity", choices=["a9a","w8a","phishing", "Mnist", "Linear_synthetic", "Fashion_Mnist", "Cifar10" ,"human_activity", "Nist"])
    parser.add_argument("--model", type=str, default="MLP", choices=["CNN","MLP","linear_regression", "mclr", "logistic_regression"])
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1, help="Local learning rate")
    parser.add_argument("--alpha", type=float, default=0.03, help="alpha for DONE and Newton using in richason interation")
    parser.add_argument("--eta", type=float, default=1.0, help = "eta is parameter for DANE")
    parser.add_argument("--L", type=float, default=0.02, help="Regularization term")
    parser.add_argument("--rho", type=float, default=0, help="Condition number")
    parser.add_argument("--num_global_iters", type=int, default=100)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="SGD",choices=["SGD"])
    parser.add_argument("--algorithm", type=str, default="Sophia",choices=["Sophia","Sophia-1","Sophia-2","DONE", "GD", "DANE", "Newton", "FedAvg", "GT", "PGT", "FEDL","GIANT"])
    parser.add_argument("--numedges", type=int, default=32,help="Number of Edges per round")
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--commet", type=int, default=1, help="log data to comet")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to run the experiments")
    parser.add_argument("--exp_type", type=str, default="regular", help = "whether hyperparams or regular exp", choices=["regular", "hyperparams"])
    parser.add_argument("--tau", type=int, default=10, help = "tau for Sophia")
    parser.add_argument("--place", type=str, default="cnn-fmnist-new", help = "which exp folder")    
    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm               : {}".format(args.algorithm))
    print("Batch size              : {}".format(args.batch_size))
    print("Learing rate            : {}".format(args.learning_rate))
    print("alpha                   : {}".format(args.alpha))
    print("Subset of edges         : {}".format(args.numedges))
    print("Number of local rounds  : {}".format(args.local_epochs))
    print("Number of global rounds : {}".format(args.num_global_iters))
    print("Dataset                 : {}".format(args.dataset))
    print("Local Model             : {}".format(args.model))
    print("=" * 80)

    if(args.commet):

        name = "sophia_exps" if args.exp_type == "regular" else "sophia-hyper-parameters"
        experiment = Experiment(
            api_key="lhdQnruUATiAZPyU7Qp2zFiVX",
            project_name=args.place,
            workspace="ahmed-khaled-saleh",
            )
        hyper_params = {
            "dataset":args.dataset,
            "algorithm" : args.algorithm,
            "model":args.model,
            "batch_size":args.batch_size,
            "learning_rate":args.learning_rate,
            "alpha" : args.alpha, 
            "L" : args.L,
            "rho" : args.rho,
            "num_glob_iters":args.num_global_iters,
            "local_epochs":args.local_epochs,
            "optimizer": args.optimizer,
            "numusers": args.numedges,
            "times" : args.times,
            "gpu": args.gpu,
            "tau": args.tau
        }
        experiment.log_parameters(hyper_params)
    else:
        experiment = 0

    main(
        experiment= experiment,
        dataset=args.dataset,
        algorithm = args.algorithm,
        model=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        alpha = args.alpha,
        eta = args.eta,
        L = args.L,
        rho = args.rho,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer= args.optimizer,
        numedges=args.numedges,
        times = args.times,
        commet = args.commet,
        gpu=args.gpu,
        tau = args.tau
        )
