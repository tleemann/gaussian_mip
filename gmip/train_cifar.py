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
from gmip.dp_sgd import PrivateOptimizer, noisy_train, eval_model, recursive_fix, RandomSubsetDataset
import torch.utils.data as data
import numpy as np
from gmip.utils import compute_tau

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
    parser.add_argument('model_arch', type=str, help='architecture to use', default="resnet56")
    parser.add_argument('--device', type=str, help='device to use for training', default="cuda:0")
    parser.add_argument('--finetune', type=bool, help='fintune a pretrained model, or pretrain a model', default=True)
    parser.add_argument('--shallow', type=bool, help='use this mode to train shallow models for the attack experiment', default=False)
    parser.add_argument('--trace_grads', type=bool, help='trace and store accumulated gradients for performing attacs', default=False)
    parser.add_argument('--kval', type=int, help='value of K to use for automatic privacy computation', default=650)
    parser.add_argument('--model_dims', type=int, help='number of trainable parameters. pass correct value if arch is not resnet56', default=650)
    parser.add_argument('--num_train', type=int, help='reduce the number of training points to use to this number', default=-1)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    config = arg_parse()
    
        ## Fix random seets
    torch.manual_seed(49*config.runid)

    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))])
    transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))])

    if config.finetune == False:
        dataset = "CIFAR100"
        cifar_train_split = torchvision.datasets.CIFAR100(root=dset_cache, train=True, download=True, transform=transform_train)
        cifar_test = torchvision.datasets.CIFAR100(root=dset_cache, train=False, download=True, transform=transform_test)
    else:
        dataset = "CIFAR10"
        cifar_train = torchvision.datasets.CIFAR10(root=dset_cache, train=True, download=True, transform=transform_train)
        cifar_test = torchvision.datasets.CIFAR10(root=dset_cache, train=False, download=True, transform=transform_test)

    shallow_flag = config.shallow

    share_trainset_use = None
    ratio_train = 1.0
    if "DP" in config.tau or "MIP" in config.tau: # Perf setup
        if "DP" in config.tau:
            index = int(config.tau[2:])
        if "MIP" in config.tau:
            index = int(config.tau[3:])
        mus = np.exp(np.linspace(np.log(0.4), np.log(50), 20))
        mu_use = mus[index]
        print("Target mu:", mu_use)
        K = config.kval
        d = config.model_dims
        N = config.num_train if config.num_train > 0 else len(cifar_train)
        share_trainset_use = N
        T=(N/config.batch_size)*config.epochs
        tau_eff = compute_tau(mu_use, config.C, K, d, N, T, config.batch_size, dp=("DP" in config.tau))
    else:
        tau = float(config.tau)
        if tau != 0.0:
            tau_eff = (tau*C)/math.sqrt(config.batch_size)
        else:
            tau_eff = 0.0

    if shallow_flag:
        share_trainset_use = batch_size*2
        ratio_train = 0.5
    ###
    print(f"Using C={config.C}, tau={tau_eff}, batch_size={config.batch_size}, epochs={config.epochs} as privacy parameters.")
    

    if dataset=="CIFAR10":
        cifar_train_split = RandomSubsetDataset(cifar_train, subsample_ratio=ratio_train, n_use_total=share_trainset_use)
        base_trainloader = torch.utils.data.DataLoader(cifar_train_split, batch_size=config.batch_size, shuffle=True, num_workers=4)
        base_testloader = torch.utils.data.DataLoader(cifar_test, batch_size=config.batch_size, shuffle=False, num_workers=4)
    else:
        base_trainloader = data.DataLoader(cifar_train_split, batch_size=config.batch_size, num_workers=4, sampler = data.RandomSampler(torch.arange(len(cifar_train_split)), replacement=True if share_train==None else False))
        base_testloader = data.DataLoader(cifar_test, batch_size=config.batch_size, shuffle=False, num_workers=4)
    print(len(cifar_train_split), len(cifar_test), cifar_train_split.get_samples_used().tolist()[:10])

    # Splitting: Only take a random half of the trainset. 
    # Setup model

    from opacus.validators import ModuleValidator

    if not config.finetune:
        model = resnet18(weights=ResNet18_Weights.DEFAULT, num_classes=1000)
        model.fc = nn.Linear(512, 100, bias=True)
        model = ModuleValidator.fix(model)
        model_opacus = GradSampleModule(model, loss_reduction = "mean").to(config.device)
    else:
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", f"cifar100_{config.model_arch}", pretrained=True)
        model = model.eval()
        recursive_fix(model)
        for p in model.parameters():
            p.requires_grad_(False)

        if "vgg1" in config.model_arch:
            print("Setting trainable parameters for VGG")
            model.classifier[0] = nn.Linear(512, 100, bias=True)
            model.classifier[3] = nn.Linear(100, 100, bias=True)
            model.classifier[6] = nn.Linear(100, 10, bias=True)
        elif "resnet" in config.model_arch:
            print("Setting trainable parameters for ResNet")
            model.fc = nn.Linear(64, 10, bias=True)
        elif "mobilenetv2" in config.model_arch:
            print("Setting trainable parameters for MobileNet")
            in_feat = model.fc.in_features
            model.classifier[1] = nn.Linear(in_feat, 10, bias=True)
        elif "shufflenet" in config.model_arch:
            print("Setting trainable parameters for Shufflenet")
            in_feat = model.fc.in_features
            model.fc = nn.Linear(in_feat, 10, bias=True)
        elif "repvgg" in config.model_arch:
            print("Setting trainable parameters for RepVGG")
            in_feat = model.linear.in_features
            model.linear = nn.Linear(in_feat, 10, bias=True)

        model_opacus = GradSampleModule(model, loss_reduction = "mean").to(config.device)
        #model_opacus.load_state_dict("models/CIFAR10_base100.pt")
    criterion = nn.CrossEntropyLoss().to(config.device)

    # Private Optimizer


    myoptim_adam = PrivateOptimizer(Adam(model_opacus.parameters(), lr=1e-3), model_opacus, C=config.C, tau=tau_eff)
    acc= 0.0
    ep_trained = 0
    max_acc = 0.
    best_epoch = 0
    while ep_trained < config.epochs:
        model_opacus = noisy_train(model_opacus, base_trainloader, criterion, myoptim_adam, 1, ep_trained, use_device=config.device)
        ep_trained += 1

        acc = eval_model(model, base_testloader, use_device=config.device)
        if acc > max_acc:
            max_acc = acc
            best_epoch = ep_trained

    print("FINAL_ACC: ", 100.0*acc, " % ")

    res_dict = {"C": config.C, "tau": config.tau, "tau_eff": tau_eff, "batch_size": config.batch_size, "final_acc": 100*acc, "max_acc": max_acc*100, "best_epoch": best_epoch, "steps": ep_trained*len(base_trainloader)}

    if config.finetune:
        res_dict["samples_used"] = cifar_train_split.get_samples_used().tolist()
    #res_dict["model_opacus"] = model_opacus.cpu().state_dict()
    model = model.cpu()
    #res_dict["model_plain"] = model.state_dict() 
    res_dict["model_plain_fc_layer"] = model.fc.state_dict() #model.cpu().state_dict()

    ## Legacy naming convention
    torch.save(res_dict, f"{config.savepath}/{dataset}_C{config.C}_tau{config.tau}_batch{config.batch_size}_ep{ep_trained}_{config.model_arch}_{config.runid}.pt")
