import torchvision
import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

import os
from opacus.validators import ModuleValidator
from opacus.grad_sample import GradSampleModule
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, ncx2
from torch.utils.data import TensorDataset, DataLoader
import sys

from glir import GaussianDataLoader
from glir import SimulatedGradientTracer, DirectGradients
from glir import GLiRAttack
from dp_sgd import recursive_fix
from utils import analytical_tpr
from dp_sgd import RandomSubsetDataset
from glir import ClassificationModelGradients, CheckpointListTracer
from ext_datasets.data_loaders import get_datasets 
from transformers import DistilBertForSequenceClassification
## USAGE:
## Call with argument compute_cfd_scores.py <dataset> <num_models> <device> [<layer_inf>]
dataset = sys.argv[1]
num_models = int(sys.argv[2])
use_device = sys.argv[3]

if len(sys.argv) > 4:
    layer_info = sys.argv[4]
else:
    layer_info = None

def load_cifar10_model(num_classes = 10):
    model = resnet18(weights=None, num_classes=1000)
    model.fc = nn.Linear(512, num_classes, bias=True)

    model = ModuleValidator.fix(model)
    #model.load_state_dict(dset_dict["model_plain"])
    model_opacus = GradSampleModule(model, loss_reduction = "mean").to(use_device)
    #model_opacus = model_opacus.eval()
    plist = list(model_opacus.parameters())
    for p in plist[:-10]:
        p.requires_grad_(False)
        
    return model_opacus, lambda model, state: model.load_state_dict(state)

def load_distilbert_model():
    n_layers = 4
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    model.distilbert.transformer.layer = model.distilbert.transformer.layer[:n_layers]
    model = ModuleValidator.fix(model)
    model_opacus = GradSampleModule(model, loss_reduction = "mean").to(use_device)
    model_opacus.train()
    plist = list(model_opacus.parameters())
    for p in plist[:-3]:
        p.requires_grad_(False)
    return model_opacus, lambda model, state: model.load_state_dict(state)




## Set parameters for the datasets.

    
def run_glir_multi_step(params_use, models_use = 5, steps_use = 1, layer_use = None):
    """
        params_use: The attack and dataset paramters. See next cell for an example for the structure.
        models_use: Number of models in the name scheme to load. Make sure the corresponding files exist.
        stept_use: How many SGD steps to consider for the attack
    """
    # Load a model and set up the tracing.
    tot_scores_list = []
    criterion = torch.nn.CrossEntropyLoss().to(use_device)
    for trained_model in range(0, models_use):
        # Create a model
        opacus_model, weight_load = params_use["model_load"]()
        opacus_model = opacus_model.to(use_device)

        print("Loading model no. ", trained_model)
        # Reset to a past training set that is loaded from a logfile
        if layer_use is None:
            layer_use = slice(-params_use["top_k_params"], None)
        else:
            layer_use = layer_use
        

        tracer = CheckpointListTracer(f"{params_use['modelprefix']}{trained_model}.pt", weight_load, onlyuse_layers=layer_use)
        #tracer.update_model_to_next_step(opacus_model)

        train_point_idx = tracer.get_used_sample_idx()
        n_in_out_points = len(train_point_idx)
        print("Number of train samples", n_in_out_points)

        # Create loaders for query points, background points, etc.
        # Background dataset: rest of the training data that was not used for test points
        val_dataset = RandomSubsetDataset(params_use["data_train"], subsample_ratio=params_use["n_grad_estimations"]/len(params_use["data_train"]))
        val_dataset.sample_idx = torch.arange(params_use["n_grad_estimations"])+params_use["samples_use"]
        background_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        ## Setup a trainpoints loader
        traintest_dataset = RandomSubsetDataset(params_use["data_train"], subsample_ratio=params_use["samples_use"]/len(params_use["data_train"]))
        traintest_dataset.sample_idx = torch.arange(params_use["samples_use"])
        base_dataloader = torch.utils.data.DataLoader(traintest_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        ## layers to use
        gradient_comp = ClassificationModelGradients(opacus_model, criterion, cutoff_threshold = params_use["c"], device=use_device, onlyuse_layers=layer_use)
        attack = GLiRAttack(background_loader, gradient_comp, tracer, params_use["d"], params_use["n"], n_background_samples=params_use["n_grad_estimations"], small_var_lim=params_use["var_lim"])
        #grad_estimation = attack._estimate_grads(attack.background_loader, n_estimation_samples=attack.n_background_samples)
        #grad_means = grad_estimation.mean(axis=0, keepdim=True)
        #grad_vars_norm = grad_estimation - grad_means
        #sigma_diag = grad_vars_norm.var(axis=0)
        ## zero-out off diagonal elements.
        #sigma = (grad_vars_norm.t() @ grad_vars_norm)/len(grad_vars_norm) 
        #sigma = sigma*diag_fiter(1, sigma.size(0))
        #print("Min: ", torch.mean(torch.abs(torch.diag(sigma))))
        #sigma = sigma + torch.diag(torch.ones(len(sigma)))*0.001
        #print("Inverting Sigma...")
        #kinv = torch.cholesky_inverse(sigma)
        #print("Sigma inverted.")
        out_scores = attack.compute_glir_attack_scores_w_loader(base_dataloader, n_load=2*n_in_out_points, n_steps=steps_use)
        labels = torch.zeros(params_use["samples_use"])
        labels[train_point_idx] = 1
        tot_scores_list.append((out_scores, labels))
    return tot_scores_list

## Define parameters
data_train, data_test = get_datasets(dataset, full=True)
params_list = {"cifar": {"data_train": data_train, "data_test": data_test, "model_load": load_cifar10_model,
                        "modelprefix": "/mnt/ssd3/tobias/mi_auditing/models_trace/CIFAR10_Cinf_tau0.0_batch32_ep30_resnet_", "n_grad_estimations": 45000, "c": float("inf"),
                        "n": 32, "d": 5120, "samples_use": 4000, "top_k_params": 10, "batch_size": 32, "var_lim": 1e-1, "steps_use": 30},
                "imdb": {"data_train": data_train, "data_test": data_test, "model_load": load_distilbert_model,
                        "modelprefix": "/mnt/ssd3/tobias/mi_auditing/models_trace/imdb_Cinf_tau0.0_batch64_ep8_bert4_", "n_grad_estimations": 5000, "c": float("inf"),
                        "n": 64, "d": 2306, "samples_use": 4000, "top_k_params": 3, "batch_size": 32, "var_lim": 1e-3, "steps_use": 40},
                "cancer": {"data_train": data_train, "data_test": data_test, "model_load": lambda: load_cifar10_model(num_classes=7),
                        "modelprefix": "/mnt/ssd3/tobias/mi_auditing/models_trace/SkinCancer_Cinf_tau0.0_batch64_ep40_resnet_", "n_grad_estimations": 6500, "c": float("inf"),
                        "n": 32, "d": 3584, "samples_use": 2000, "top_k_params": 2, "batch_size": 128, "var_lim": 1e-2, "steps_use": 40}
                }

## Ablation study with conv layers.

batch_size = 128
if dataset == "cifar":
    if layer_info == "conv1":
        layer_use = [-8]
    if layer_info == "conv2":
        layer_use = [-5]
    if layer_info is None:
        layer_use = [-2, -1]
    glir_res = run_glir_multi_step(params_list[dataset], models_use = num_models, steps_use = params_list[dataset]["steps_use"], layer_use=layer_use)
else:
    glir_res = run_glir_multi_step(params_list[dataset], models_use = num_models, steps_use = params_list[dataset]["steps_use"])

mi_scores = torch.stack(list([res[0] for res in glir_res]), axis=0)
mi_labels = torch.stack(list([res[1].long() for res in glir_res]), axis=0)
torch.save((mi_scores, mi_labels), f"results/mi_scores_glir_{dataset}_{layer_info}.pt")
