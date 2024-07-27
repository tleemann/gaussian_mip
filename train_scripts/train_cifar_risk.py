## Modified CIFAR-10 Train script for the privacy risk prediction task
import torch
import torchvision
import torchvision.transforms as transforms
import argparse
from torchvision.models import resnet18, ResNet18_Weights
from opacus.grad_sample import GradSampleModule
import sys
import torch.nn as nn
import math
from torch.optim import Adam, RMSprop, SGD
from gmip.dp_sgd import PrivateOptimizer, noisy_train, eval_model, recursive_fix, RandomSubsetDataset
import torch.utils.data as data
import numpy as np

from ml_models.resnets import CustomBasicBlock, CustomResNet

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
    parser.add_argument('--finetune', type=bool, help='fintune a pretrained model, or pretrain a model', default=False)
    parser.add_argument('--shallow', type=bool, help='use this mode to train shallow models for the attack experiment', default=False)
    parser.add_argument('--model_dims', type=int, help='number of trainable parameters. pass correct value if arch is not resnet56', default=650)
    parser.add_argument('--num_train', type=int, help='reduce the number of training points to use to this number', default=20000)
    parser.add_argument('--dataset', type=str, help='the dataset to use', default="CIFAR10")
    parser.add_argument('--trace_grads', type=bool, help='whether gradients should be recorded.', default=False)
    parser.add_argument('--record_dims', type=int, help='how many dims of the gradients should be recorded.', default=10)
    parser.add_argument('--record_steps', type=int, help='whether gradients should be recorded every n steps', default=100)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    config = arg_parse()
    
        ## Fix random seets
    torch.manual_seed(49*config.runid)

    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))])
    transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))])

    if config.dataset == "CIFAR100":
        cifar_train_split = torchvision.datasets.CIFAR100(root=dset_cache, train=True, download=True, transform=transform_train)
        cifar_test = torchvision.datasets.CIFAR100(root=dset_cache, train=False, download=True, transform=transform_test)
    else:
        cifar_train = torchvision.datasets.CIFAR10(root=dset_cache, train=True, download=True, transform=transform_test)
        cifar_test = torchvision.datasets.CIFAR10(root=dset_cache, train=False, download=True, transform=transform_test)

    if config.dataset=="CIFAR10":
        cifar_train_split = RandomSubsetDataset(cifar_train, subsample_ratio=0.5, n_use_total=config.num_train*2)
        base_trainloader = torch.utils.data.DataLoader(cifar_train_split, batch_size=config.batch_size, shuffle=True, num_workers=4)
        base_testloader = torch.utils.data.DataLoader(cifar_test, batch_size=config.batch_size, shuffle=False, num_workers=4)
    else:
        base_trainloader = data.DataLoader(cifar_train_split, batch_size=config.batch_size, num_workers=4, sampler = data.RandomSampler(torch.arange(len(cifar_train_split)), replacement=replacement))
        base_testloader = data.DataLoader(cifar_test, batch_size=config.batch_size, shuffle=False, num_workers=4)
    print(len(cifar_train_split), len(cifar_test), cifar_train_split.get_samples_used().tolist()[:10])

    # Splitting: Only take a random half of the trainset. 
    # Setup model

    from opacus.validators import ModuleValidator

    if not config.finetune:
        if "resnet" in config.model_arch:
            model = CustomResNet(block=CustomBasicBlock, layers=[2, 2, 2, 2], num_classes=1000)
            if config.dataset=="CIFAR10":
                model.fc = nn.Linear(512, 10, bias=True)
                print("using 10 classes.")
            else:
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
    tau = float(config.tau)
    myoptim_adam = PrivateOptimizer(Adam(model_opacus.parameters(), lr=1e-3), model_opacus, C=config.C, tau=tau)
    acc= 0.0
    ep_trained = 0
    max_acc = 0.
    best_epoch = 0
    tot_grad_list = []
    tot_params_list = []
    while ep_trained < config.epochs:
        if config.trace_grads:
            model_opacus, ep_grad, ep_params = noisy_train(model_opacus, base_trainloader, criterion, myoptim_adam, 1, ep_trained,
                use_device=config.device, collect_stepwise=True, return_n_last_dims=config.record_dims, return_grads_every=config.record_steps)
            tot_grad_list = tot_grad_list + ep_grad
            tot_params_list = tot_params_list + ep_params
        else:
            model_opacus = noisy_train(model_opacus, base_trainloader, criterion, myoptim_adam, 1, ep_trained, use_device=config.device)
        ep_trained += 1

        acc = eval_model(model, base_testloader, use_device=config.device)
        if acc > max_acc:
            max_acc = acc
            best_epoch = ep_trained

    print("FINAL_ACC: ", 100.0*acc, " % ")

    res_dict = {"C": config.C, "tau": config.tau, "tau_eff": tau, "batch_size": config.batch_size, "final_acc": 100*acc, "max_acc": max_acc*100, "best_epoch": best_epoch, "steps": ep_trained*len(base_trainloader)}

    res_dict["samples_used"] = cifar_train_split.get_samples_used().tolist()
    #res_dict["model_opacus"] = model_opacus.cpu().state_dict()
    model = model.cpu()
    res_dict["stepwise_params"] = tot_params_list
    res_dict["stepwise_grads"] = tot_grad_list
    res_dict["model_plain"] = model.state_dict() 
    #res_dict["model_plain_fc_layer"] = model.fc.state_dict() #model.cpu().state_dict()

    ## Legacy naming convention
    torch.save(res_dict, f"{config.savepath}/{config.dataset}_C{config.C}_tau{config.tau}_batch{config.batch_size}_ep{ep_trained}_{config.model_arch}_{config.runid}.pt")
