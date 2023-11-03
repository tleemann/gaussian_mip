""" Train private models on the tabular datasets. """
""" call train_tabular.py dataset C tau run_id savepath batchsize epochs trace_grads(true/false) dataset_path
    tau can either be a numerical value or MIP<step> / DP<step>, e.g. MIP10 to train a model for the 10th step of the utility experiment
"""
import torch
import torch.nn as nn
import sys
import torch.utils.data as data
from gmip.dp_sgd import RandomSubsetDataset, noisy_train, eval_model, PrivateOptimizer
from torch.optim import Adam
from opacus.grad_sample import GradSampleModule
import math
import numpy as np
from gmip.utils import get_fn, calc_privacy_lvl
from scipy.optimize import fsolve

shallow_flag = False
def create_purchase_base():
    net = nn.Sequential(
        nn.Linear(600, 1024),
        nn.Dropout(p=0.2),
        nn.Tanh(),
        nn.Linear(1024, 128),
        nn.Dropout(p=0.2),
        nn.Tanh(),
        nn.Linear(128, 100)
    )
    #print(net)
    return net, (lambda modelp: modelp._modules["_module"][6].state_dict())

def create_adult_base():
    net = nn.Sequential(
            nn.Linear(12, 512),
            nn.Tanh(),
            nn.Linear(512, 2)
    )
    return net, (lambda modelp: modelp._modules["_module"].state_dict())

if __name__ == "__main__":
    dataset_path = "/ssd/tobias/datasets/"

    dataset = "purchase"
    if len(sys.argv) > 1:
        dataset = sys.argv[1]

    C = 100
    if len(sys.argv) > 2:
        C = float(sys.argv[2])

    tau = 1e-2
    if len(sys.argv) > 3:
        tau = sys.argv[3]

    runid = 0
    if len(sys.argv) > 4:
        runid = int(sys.argv[4])

    savepath="models"
    if len(sys.argv) > 5:
        savepath = sys.argv[5] 

    batch_size = 128
    if len(sys.argv) > 6:
        batch_size = int(sys.argv[6])

    epochs = 30
    if len(sys.argv) > 7:
        epochs = int(sys.argv[7])

    #dataset_path = 30
    trace_grads = False
    replacement = True
    subsample_ratio = 0.5
    if len(sys.argv) > 8:
        if sys.argv[8] == "True" or sys.argv[8] == "true":
            print("Tracing gradients.")
            trace_grads = True
            subsample_ratio = batch_size/(55494.0 if dataset=="purchase" else 43842.0)
            replacement = False
    
    if len(sys.argv) > 9:
        dataset_path = sys.argv[9]

    torch.manual_seed(49*runid)
    n_total = None
    if "DP" in tau or "MIP" in tau: # Ulitiy experiment setup
        if "DP" in tau:
            index = int(tau[2:])
        if "MIP" in tau:
            index = int(tau[3:])
        mus = np.exp(np.linspace(np.log(0.4), np.log(50), 20))
        mu_use = mus[index]
        print("Target mu:", mu_use)
        if dataset == "purchase":
            K=2560
            d=2560
            N=54855
            subsample_ratio = N/55494.0
        elif dataset == "adult":
            K=1026
            d=1026
            N=43000
            subsample_ratio = N/43842.0
        T=(N/batch_size)*epochs
        torch.manual_seed(53*((runid)*20+index))
        if mu_use > calc_privacy_lvl(C, 0.0, T, batch_size, N, d, K) and ("MIP" in tau):
            tau_eff = 0.0
        else:
            func_mip = get_fn(mu_use, dp=("DP" in tau), C=C, K=K, d=d, N=N, T=T, batch_size=batch_size)
            tau_0 = np.array([1.0])
            tau_eff = fsolve(func_mip, tau_0.copy(), factor=0.1)[0]
            if ("MIP" in tau):
                func_dp = get_fn(mu_use, dp=True, C=C, K=K, d=d, N=N, T=T, batch_size=batch_size)
                tau_0 = np.array([1.0])
                tau_eff_dp = fsolve(func_dp, tau_0.copy(), factor=0.1)[0]
                tau_eff = min(tau_eff, tau_eff_dp)

        if shallow_flag:
            subsample_ratio = 0.5
            n_total = 2*batch_size
            replacement = False
    else:
        tau = float(tau)
        if tau != 0.0:
            tau_eff = (tau*C)/math.sqrt(batch_size)
        else:
            tau_eff = 0.0
    print(f"Using C={C}, tau={tau_eff}, batch_size={batch_size}, epochs={epochs} as privacy parameters.")

    try:
        import pandas as pd
        df = pd.read_csv(f"{dataset_path}{dataset}/{dataset}.csv", header=None)
        dataset_tensor = torch.tensor(df.values)
    except:
        # Dont want to install pandas
        # read in the CSV via plain python.
        with open(f"{dataset_path}{dataset}/{dataset}.csv", "r") as file:
            lines = file.readlines()
            records = [[float(value) for value in line.split(",")] for line in lines]
        dataset_tensor = torch.tensor(records)
    print(dataset_tensor.shape)

    labels = dataset_tensor[:, 0].long()
    features = dataset_tensor[:, 1:].float()

    # normalization
    features = ((features-features.mean(axis=0))/(features.std(axis=0)+1e-5))

    if dataset == "purchase":
        ## Finetuning split
        ## Find 20 most common labels
        _, counts = torch.unique(labels, return_counts=True)
        freq_ranks = torch.argsort(np.argsort(-counts))

        #pretrain_dataset = freq_ranks[labels-1] >= 20
        finetune_dataset = freq_ranks[labels-1] < 20
        #labels_pretrain = labels.iloc[pretrain_dataset].values
        #features_pretrain = features.iloc[pretrain_dataset].values
        finetune_labels = labels[finetune_dataset]
        finetune_features = features[finetune_dataset]
        print(len(finetune_features), len(finetune_labels))
        finetune_labels = freq_ranks[finetune_labels-1] # map labels from 0 -> 19.
        test_data = 10000 
    elif dataset == "adult":
        test_data = 5000
        finetune_labels = labels
        finetune_features = features
    else:
        raise ValueError("Unsupported dataset.")

    x_test_pre, y_test = finetune_features[:test_data].float(),  finetune_labels[:test_data].long()
    x_train_pre, y_train = finetune_features[test_data:].float(),  finetune_labels[test_data:].long()
    # Create datasets
    train_data = data.TensorDataset(x_train_pre, y_train)
    test_data = data.TensorDataset(x_test_pre, y_test)
    #print(train_data[100])

    tabular_train_split = RandomSubsetDataset(train_data, subsample_ratio=subsample_ratio, n_use_total=n_total)
    base_trainloader = data.DataLoader(tabular_train_split, batch_size=batch_size, num_workers=4, sampler = data.RandomSampler(torch.arange(len(tabular_train_split)), replacement=replacement))
    base_testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    ## Setup model and optimizer
    use_device = "cuda:0"
    if dataset == "purchase":
        model_finetune, state_fn = create_purchase_base()
        model_finetune = model_finetune.to(use_device)
        model_finetune.load_state_dict(torch.load("tabular/tabular_base998.pt"))
        ## Shut off dropout and change last layer.
        for p in model_finetune.parameters():
            p.requires_grad_(False)
        model_finetune[1].p=0.0
        model_finetune[4].p=0.0
        model_finetune[6] = nn.Linear(128, 20)
    elif dataset == "adult":
        model_finetune, state_fn = create_adult_base()
        for p in model_finetune.parameters():
            p.requires_grad_(False)
        infeat = model_finetune[0].out_features
        model_finetune[2] = nn.Linear(infeat, 2)
        model_finetune = model_finetune.to(use_device)
    model_opacus = GradSampleModule(model_finetune, loss_reduction = "mean").to(use_device)
    criterion = torch.nn.CrossEntropyLoss().to(use_device)
    myoptim_adam = PrivateOptimizer(Adam(model_opacus.parameters(), lr=2e-3), model_opacus, C=C, tau=tau_eff)

    max_acc = 0.0
    best_epoch = 0
    ep_trained = 0
    tot_grad_list = []
    tot_params_list = []
    while ep_trained < epochs:
        if not trace_grads:
            model_opacus = noisy_train(model_opacus, base_trainloader, criterion, myoptim_adam, 1, ep_trained, use_device=use_device)
        else:
            model_opacus, ep_grad, ep_params = noisy_train(model_opacus, base_trainloader, criterion, myoptim_adam, 1,
                    ep_trained, use_device=use_device, collect_stepwise=True, state_dict_fn=state_fn)
            tot_grad_list = tot_grad_list + ep_grad
            tot_params_list = tot_params_list + ep_params
        ep_trained += 1
        acc = eval_model(model_finetune, base_testloader, use_device=use_device)
        if acc > max_acc:
            max_acc = acc
            best_epoch = ep_trained

    print("FINAL_ACC: ", 100.0*acc, " % ")

    res_dict = {"C": C, "tau": tau, "batch_size": batch_size, "final_acc": 100*acc, "max_acc": max_acc*100, "best_epoch": best_epoch, "steps": ep_trained*len(base_trainloader)}


    res_dict["samples_used"] = tabular_train_split.get_samples_used().tolist()
    #res_dict["model_opacus"] = model_opacus.cpu().state_dict()
    model_finetune = model_finetune.cpu()
    res_dict["model_plain"] = model_finetune.state_dict() 
    #res_dict["model_plain_fc_layer"] = model.fc.state_dict() #model.cpu().state_dict()
    if trace_grads == True: 
        res_dict["stepwise_params"] = tot_params_list
        res_dict["stepwise_grads"] = tot_grad_list
    ## Legacy naming convention
    torch.save(res_dict, f"{savepath}/{dataset}_C{C}_tau{tau}_batch{batch_size}_ep{ep_trained}_{runid}{'' if not trace_grads else '_stepwise'}.pt")
