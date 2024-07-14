import torch
import torchvision
import torchvision.transforms as transforms
import argparse
#from torchvision import datasets, models
from torchvision.models import resnet18, ResNet18_Weights
from opacus.grad_sample import GradSampleModule
import sys
import torch.nn as nn
import math
from torch.optim import Adam, RMSprop, SGD
from dp_sgd import PrivateOptimizer, noisy_train, eval_model, recursive_fix, RandomSubsetDataset
import torch.utils.data as data
import numpy as np
from transformers import BertForSequenceClassification, BertConfig, DistilBertForSequenceClassification
from opacus.validators import ModuleValidator
from transformers import AutoTokenizer
from datasets import load_dataset

""" Args: train_cifar.py C tau runid savepath batch_size epochs model_arch device
    tau can either be a numerical value or MIP<step> / DP<step>, e.g. MIP10 to train a model for the 10th step of the utility experiment. tau is computed automatically in this case.
"""

dset_cache = "."
def arg_parse():
    # add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('C', type=float, help='cropping threshold', default=100)
    parser.add_argument('tau', type=str, help='noise level tau (number) or DP<idx> or MIP<idx> for utility experiment', default="1.0")
    parser.add_argument('runid', type=int, help='number of the run', default=0)
    parser.add_argument("savepath", type=str, help='the path where to save the trained models', default='models')
    parser.add_argument('batch_size', type=int, help='batchsize to use', default=125)
    parser.add_argument('epochs', type=int, help='number of epochs to train', default=0)
    parser.add_argument('model_arch', type=str, help='architecture to use', default="bert6")
    parser.add_argument('--device', type=str, help='device to use for training', default="cuda:0")
    parser.add_argument('--finetune', type=bool, help='fintune a pretrained model, or pretrain a model', default=False)
    parser.add_argument('--shallow', type=bool, help='use this mode to train shallow models for the attack experiment', default=False)
    parser.add_argument('--model_dims', type=int, help='number of trainable parameters. pass correct value if arch is not resnet56', default=650)
    parser.add_argument('--num_train', type=int, help='reduce the number of training points to use to this number', default=200)
    parser.add_argument('--dataset', type=str, help='the dataset to use', default="imdb")
    parser.add_argument('--trace_grads', type=bool, help='whether gradients should be recorded.', default=False)
    parser.add_argument('--record_dims', type=int, help='how many dims of the gradients should be recorded.', default=4)
    parser.add_argument('--record_steps', type=int, help='whether gradients should be recorded every n steps', default=100)
    args = parser.parse_args()
    return args


def setup_bert_model(dim: int=768, hidden_dim: int=3072, n_heads: int=12, n_layers: int=12, device="cuda"):
    config = BertConfig(hidden_size = dim,
                        num_hidden_layers = n_layers,
                        num_attention_heads = n_heads,
                        intermediate_size = hidden_dim,
                        output_attentions=True)
    model = BertForSequenceClassification(config)
    model = ModuleValidator.fix(model)
    model_opacus = GradSampleModule(model, loss_reduction = "mean").to(device)
    return model_opacus, model

def setup_distilbert_model(n_layers: int=6, device="cuda"):
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    model.distilbert.transformer.layer = model.distilbert.transformer.layer[:n_layers]
    model = ModuleValidator.fix(model)
    model_opacus = GradSampleModule(model, loss_reduction = "mean").to(device)
    return model_opacus, model

def preprocess_function(tokenizer, examples, max_seq_len=512):
        """Preprocess function for the dataset"""
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_seq_len, return_tensors='pt')

if __name__ == "__main__":
    args = arg_parse()
    ## Fix random seets
    torch.manual_seed(49*args.runid)

    ds = None
    if args.dataset == "imdb":
        imdb = load_dataset('imdb').with_format('torch', device="cpu") # format to pytorch tensors, but leave data on cpu
        imdb["train"] = imdb["train"].shuffle(seed=42).select(range(4000))
        imdb["test"] = imdb["test"].shuffle(seed=42).select(range(100))
        ds = imdb

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    tokenizer_func = lambda ex: preprocess_function(tokenizer, ex)
    ## Tokenize and prepare dataset.
    ds_tokenized = ds.map(tokenizer_func, batched=True)

    #dataloader_train = torch.utils.data.DataLoader(ds_tokenized['train'], batch_size=args.batch_size, shuffle=True)
    ds_train_split = RandomSubsetDataset(ds_tokenized['train'], subsample_ratio=0.5, n_use_total=args.num_train*2)
    base_trainloader = torch.utils.data.DataLoader(ds_train_split, batch_size=args.batch_size, shuffle=True, num_workers=4)
    base_testloader = torch.utils.data.DataLoader(ds_tokenized['test'], batch_size=args.batch_size, shuffle=False)

    ## Setup model
    n_layers = int(args.model_arch[4:])
    model_opacus, model = setup_distilbert_model(n_layers = n_layers, device=args.device)
    model_opacus.to(args.device)
    nparams = []
    for p in model_opacus.parameters():
        nparams.append(p.numel())
    print("Total param count: ", sum(nparams))
    print("Last params: ", sum(nparams[-args.record_dims:]))

    criterion = nn.CrossEntropyLoss().to(args.device)
    # Private Optimizer and training
    tau = float(args.tau)
    myoptim_adam = PrivateOptimizer(Adam(model_opacus.parameters(), lr=5e-5), model_opacus, C=args.C, tau=tau)
    acc= 0.0
    ep_trained = 0
    max_acc = 0.
    best_epoch = 0
    tot_grad_list = []
    tot_params_list = []
    while ep_trained < args.epochs:
        model_opacus.train()
        if args.trace_grads:
            model_opacus, ep_grad, ep_params = noisy_train(model_opacus, base_trainloader, criterion, myoptim_adam, 1, ep_trained,
                use_device=args.device, collect_stepwise=True, return_n_last_dims=args.record_dims, return_grads_every=args.record_steps)
            tot_grad_list = tot_grad_list + ep_grad
            tot_params_list = tot_params_list + ep_params
        else:
            model_opacus = noisy_train(model_opacus, base_trainloader, criterion, myoptim_adam, 1, ep_trained, use_device=args.device)
        ep_trained += 1

        acc = eval_model(model, base_testloader, use_device=args.device)
        if acc > max_acc:
            max_acc = acc
            best_epoch = ep_trained

    print("FINAL_ACC: ", 100.0*acc, " % ")

    res_dict = {"C": args.C, "tau": args.tau, "tau_eff": tau, "batch_size": args.batch_size, "final_acc": 100*acc, "max_acc": max_acc*100, "best_epoch": best_epoch, "steps": ep_trained*len(base_trainloader)}

    res_dict["samples_used"] = ds_train_split.get_samples_used().tolist()
    #res_dict["model_opacus"] = model_opacus.cpu().state_dict()
    model_opacus = model_opacus.cpu()
    res_dict["stepwise_params"] = tot_params_list
    res_dict["stepwise_grads"] = tot_grad_list
    res_dict["model_plain"] = model_opacus.state_dict()
    #res_dict["model_plain_fc_layer"] = model.fc.state_dict() #model.cpu().state_dict()

    ## Legacy naming convention
    torch.save(res_dict, f"{args.savepath}/{args.dataset}_C{args.C}_tau{args.tau}_batch{args.batch_size}_ep{ep_trained}_{args.model_arch}_{args.runid}.pt")


