## Implementation of the Counterfactual-Distance-Attack by Pawelczyk et al. (AISTATS 23)
import sys, os
import numpy as np
import datetime
import torch 
import torch.nn as nn
from opacus import GradSampleModule
from opacus.validators import ModuleValidator
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm
from torchvision.models import resnet18
from ml_models.models import get_model_w_grads
from ext_datasets.data_loaders import get_datasets
from dp_sgd import RandomSubsetDataset
## Call with argument compute_cfd_scores.py <dataset> <num_models> <device>
## Load a CIFAR-10 model

batch_size = 16
latent_lvl = 5 #int(sys.argv[1])
dataset = sys.argv[1]
max_models = int(sys.argv[2])
device = sys.argv[3]

from cfd_attack import ModelLatentExtractor, SCFE, LLMLatentExtractor

TOTAL_TRAIN_POINTS = {"cifar": 4000, "imdb": 4000, "cancer": 2000}
total_train_pts = TOTAL_TRAIN_POINTS[dataset]

#max_models = 200
mi_score_array = torch.zeros(max_models, total_train_pts)
mi_labels_array = torch.zeros(max_models, total_train_pts, dtype=torch.long)

my_train_dataset, _ = get_datasets(dataset)
train_dataset = RandomSubsetDataset(my_train_dataset, subsample_ratio=1.0, n_use_total=total_train_pts)

for k in range(0, max_models):
    model_path = "/mnt/ssd3/tobias/mi_auditing/models_trace"
    #my_model, mi_label_dict = get_model_w_grads(model_path, dataset, k, device)
    print("loaded weights", k)
    score_list = []
    for i in tqdm(range(total_train_pts)):
        if i == 0 or (i % 400 == 0 and dataset=="imdb"):
            my_model, mi_label_dict = get_model_w_grads(model_path, dataset, k, device)
            my_model = my_model._module ## No opacus
            if dataset == "imdb":
                my_le = LLMLatentExtractor(my_model).to(device)
            else:
                my_le = ModelLatentExtractor(my_model, latent_lvl=latent_lvl).to(device)
            my_le.train()
            

        with torch.no_grad():
            if dataset == "imdb":
                inputs = train_dataset[i]
                pred, latents = my_le.get_latents(inputs["input_ids"].to(device), inputs["attention_mask"].to(device))
            else:
                inputs, lbl = train_dataset[i]
                pred, latents = my_le.get_latents(inputs.unsqueeze(0).to(device))
        current_class = pred.argmax().item()
        mySCFE = SCFE(my_le, target_threshold = -1, _lambda = 0.01, use_device=device, max_iter=200, step=0.0)
        dstar, _ = mySCFE.generate_counterfactuals(latents.detach(), current_class = current_class)
        score_list.append(dstar)
        for p in my_model.parameters():
            if p.grad is not None:
                p.grad = None
            
    train_point_idx_use = mi_label_dict["samples_used"]
    all_samples = torch.zeros(total_train_pts)
    all_samples[train_point_idx_use]=1 # set the train indices to zero.
    mi_labels_array[k, :] = all_samples[:total_train_pts]
    mi_score_array[k, :] = torch.tensor(score_list)

torch.save((mi_score_array, mi_labels_array), f"results/mi_scores_cfd_{dataset}_rest.pt")