## Evaluate different predictors for predicting loss-based MI success of points.

## possible predictors include cov K, loss, smoothgrad abs mean, grad var

## Real gradients, estimate distribution parameters
import argparse
import json
import torch
import torchvision.transforms as transforms
import torchvision
from gmip.dp_sgd import RandomSubsetDataset
from scipy.stats import ncx2, norm, spearmanr
import numpy as np
from sklearn.metrics import roc_curve
import torch.nn as nn
from opacus.validators import ModuleValidator
from opacus import GradSampleModule
import os
from torchvision.models import resnet18
from copy import deepcopy
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from ml_models.resnets import CustomBasicBlock, CustomResNet
from risk_predictors import compute_cov_mahalanobis, compute_loss, compute_loss_vargrad, compute_shap, compute_lime
from risk_predictors import estimate_hutchinson_traces, vae_reconstruction_loss, compute_input_gradients, compute_confidence, compute_mahalanobis_latents
from ml_models.models import get_model_w_grads
from ext_datasets.data_loaders import get_datasets  

def arg_parse():
    # add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='metrics config file to use')
    parser.add_argument('scores_file', type=str, help='scores_file to use')
    parser.add_argument('dataset', type=str, help='dataset to use')
    parser.add_argument('--device', type=str, help='which device to run on', default='cuda')
    parser.add_argument('--skip_scores', type=bool, help='if passed, only compute the metrics', default=False)
    parser.add_argument('--skip_metrics', type=bool, help='if passed, only compute the scores', default=False)
    
    args = parser.parse_args()
    return args


def analytical_tpr(fpr, mu):
    return 1-norm.cdf(norm.ppf(1-fpr)-mu)
    
def compute_maxacc_mu(fpr_var, tpr_var):
    acc = np.max(0.5*(tpr_var + (1-fpr_var)))
    mu = 2*norm.ppf(acc)
    return acc, mu

def get_model(model_path, model_id, dataset, method_key):
    return get_model_w_grads(model_path, dataset, model_id, method_key, DEVICE)

def get_predictor_aggregator(key: str, args, device):
    """ Return a predictor function.
        There are two types of predictors, model-dependent ones, and model-independent ones (only depend on the data points)
        For model-dependent ones return a func(model, data_loader)
        For model-independent ones return a func(back_ground_dataloader, data_loader)
        key: predictor to use
        args: arguments for the predictor
        device: which device to use. Should match model.
    """
    if "criterion" in args and args["criterion"] == "logit":
        criterion = lambda outputs, labels: outputs[torch.arange(len(outputs)), labels]
    else:
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
    method_args = deepcopy(args)
    if criterion in method_args:
        del method_args["criterion"]

    criterion_sum = lambda outputs, labels: torch.sum(criterion(outputs, labels))

    if key == "loss":
        return lambda model, loader: compute_loss(model, loader, criterion=criterion, use_device=device)
    elif key == "confidence":
        return lambda model, loader: compute_confidence(model, loader, criterion=criterion, use_device=device)
    elif key == "curvature":
        return lambda model, loader: estimate_hutchinson_traces(model, loader, criterion=criterion, use_device=device,
                n_samples=args.get('n_samples', 10), embeddings=args.get('embeddings', False))
    elif key == "grad":
        return lambda model, loader: compute_loss_vargrad(model, loader, criterion=criterion_sum, use_device=device, n_samples=1, var_use=0.0, agg="mean", agg_param="var")
    elif key == "vargrad":
        return lambda model, loader: compute_loss_vargrad(model, loader, criterion=criterion_sum, use_device=device)
    elif key == 'shap':
        return lambda model, loader: compute_shap(model, loader, 
                                                  n_samples=args.get('n_samples', 10),
                                                  var_use=args.get('var_use', 0.01),
                                                  agg=args.get('agg','var'), 
                                                  agg_param=args.get('agg_param','mean'), 
                                                  pretrained_lang_model=args.get('pretrained_lang_model', None),
                                                  use_device=device)
    elif key == 'input_grad':
        return lambda model, loader: compute_input_gradients(model, loader, criterion=criterion_sum,
                                                  n_samples=args.get('n_samples', 10),
                                                  var_use=args.get('var_use', 0.01),
                                                  agg=args.get('agg','var'), 
                                                  agg_param=args.get('agg_param','mean'), 
                                                  use_device=device,
                                                  embeddings=args.get('embeddings', False))
    elif key == 'lime':
        return lambda model, loader: compute_lime(model, loader,
                                                  n_samples=args.get('n_samples', 10),
                                                  agg_glob=args.get('agg_glob', 'mean'),
                                                  kernel_width=args.get('kernel_width', 0.2), 
                                                  feature_selection=args.get('feature_selection', 'auto'), 
                                                  use_device=device)
    elif key == 'mahalanobis_latent':
        return lambda model, bg_loader, loader: compute_mahalanobis_latents(model, bg_loader, loader, use_device=device)
    elif key == "mahalanobis": ## Model-independent ones
        return lambda bgload, asload: compute_cov_mahalanobis(bgload, asload, use_dims=args.get('use_dims', None))
    elif key == "vae_reconstruction_loss":
        return lambda bgload, asload: vae_reconstruction_loss(bgload, asload, use_device=device)
    else:
        raise ValueError(f"Unknown metric {key}.")


def compute_metrics(metric_list, scores_pred, scores_real, prefix=None):
    res_dict = {}
    for metric in metric_list:
        if metric == "spearman": 
            res_val = spearmanr(scores_pred[~scores_pred.isnan()], scores_real[~scores_pred.isnan()]).statistic
        elif metric == "rf_score":
            rf = RandomForestRegressor(max_depth=5)
            rf.fit(scores_pred[~scores_pred.isnan()].reshape(-1, 1), scores_real[~scores_pred.isnan()])
            res_val =  rf.score(scores_pred[~scores_pred.isnan()].reshape(-1, 1), scores_real[~scores_pred.isnan()])
        res_dict[prefix + "_" + metric if prefix else metric] = res_val
    return res_dict

def update_results(metric_results, method_key, method_params, results):
    if method_key not in metric_results:
        metric_results[method_key] = [[deepcopy(method_params), results]]
        return metric_results
    else:
        for res in  metric_results[method_key]:
            if res[0] == method_params: # update
                res[1] = results
                return metric_results
        metric_results[method_key].append([deepcopy(method_params), results])
    return metric_results

def get_results(metric_results, method_key, method_params):
    if method_key not in metric_results:
        return None
    else:
        for res in metric_results[method_key]:
            if res[0] == method_params: # update
                return res[1]
        return None

TOTAL_TRAIN_POINTS = {"cifar": 4000, "imdb": 4000, "cancer": 2000}
DEVICE = None
if __name__ == "__main__":
    config = arg_parse()
    DEVICE = config.device
    device = config.device
    fileconfig = json.load(open(config.config_file))
    dataset = config.dataset
    n_eval_points = TOTAL_TRAIN_POINTS[dataset]
    batch_size = fileconfig["batch_size"]

    ##Setup the prediction dataset
    my_train_dataset, _ = get_datasets(dataset, full=True)
    tot_len = len(my_train_dataset)
    ##Background dataset.
    background_dataset = RandomSubsetDataset(my_train_dataset, subsample_ratio=1.0-(n_eval_points/tot_len))
    background_dataset.sample_idx=torch.arange(tot_len-n_eval_points)+n_eval_points
    assess_dataset = RandomSubsetDataset(my_train_dataset, subsample_ratio=1.0, n_use_total=n_eval_points)
    assess_dataset.sample_idx = torch.arange(n_eval_points)
    background_loader = torch.utils.data.DataLoader(background_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    assess_loader = torch.utils.data.DataLoader(assess_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


    ## Compute mus and maxacc 
    mi_score_array_org, mi_labels_array_org = torch.load(config.scores_file)
    print(mi_score_array_org.shape)

    scores_file = f"{fileconfig['output_path']}/features_{dataset}.pt"
    if not config.skip_scores: 
        model_path = fileconfig["model_path"]
        
        if os.path.exists(scores_file):
            res_dict = torch.load(scores_file)
        else:
            res_dict = {}

        ## Scores / Feature stage
        for method_key, method_params in fileconfig["predictors"]:
            print(method_key)
            print(method_params)
            if method_key in ["mahalanobis", "vae_reconstruction_loss"]: ## Global metrics
                my_method = get_predictor_aggregator(method_key, method_params, device)
                update_results(res_dict, method_key, method_params, my_method(background_loader, assess_loader))
            else: ## Per model estimators
                max_models = fileconfig["num_models"]
                mi_score_array = torch.zeros(max_models, n_eval_points)
                mi_labels_array = torch.zeros(max_models, n_eval_points, dtype=torch.long)
                for model_id in range(len(mi_score_array_org), len(mi_score_array_org)+max_models):
                    my_model, dset_dict = get_model(model_path, model_id, dataset, method_key)
                    if my_model:
                        print("loaded weights", model_id)
                        train_point_idx_use = dset_dict["samples_used"]
                        n_use = len(train_point_idx_use)
                        all_samples = torch.zeros(n_eval_points)
                        all_samples[train_point_idx_use]=1 # set the train indices to zero.
                        my_method = get_predictor_aggregator(method_key, method_params, device)
                        if method_key == "mahalanobis_latent":
                            res = my_method(my_model, background_loader, assess_loader)
                        else:
                            res = my_method(my_model, assess_loader)
                        mi_score_array[model_id-len(mi_score_array_org), :len(res)] = res
                        mi_labels_array[model_id-len(mi_score_array_org), :] = all_samples
                update_results(res_dict, method_key,  method_params, (mi_score_array, mi_labels_array))
            torch.save(res_dict, scores_file)

    ### Metrics
    if not config.skip_metrics:
        results = torch.load(scores_file)
        mus = torch.zeros(n_eval_points)
        maxacc = torch.zeros(n_eval_points)
        tprat01 = torch.zeros(n_eval_points)
        for instance_no in range(n_eval_points):
            mi_labels = mi_labels_array_org[:, instance_no]
            mi_losses = mi_score_array_org[:, instance_no]
            fpr_var, tpr_var, thres = roc_curve(mi_labels, (1 if "cfd" in config.scores_file else -1)*mi_losses)
            # Find larges threshold 
            tprat01[instance_no] = np.max(tpr_var[fpr_var < 0.1])
            maxacc[instance_no], mus[instance_no] = compute_maxacc_mu(fpr_var, tpr_var)
        output_file = fileconfig["output_path"] + "/predict_" + config.scores_file.split("/")[-1][:-3] + ".json"
        if os.path.exists(output_file):
            metric_results = json.load(open(output_file))
        else:
            metric_results = {}
        for method_key, method_params in fileconfig["predictors"]:
            print(method_key, method_params)
            method_values = get_results(results, method_key, method_params)
            if method_key in ["mahalanobis", "vae_reconstruction_loss"]:
                my_results = compute_metrics(fileconfig["metrics"], method_values, maxacc)
            else:
                data = method_values[0]*1e8
                print("recalibrate")
                labels = method_values[1]
                in_data = data.clone()
                out_data = data.clone()
                in_data[labels==0] = 0
                out_data[labels==1] = 0
                in_data = in_data.sum(axis=0)/labels.sum(axis=0)
                out_data = out_data.sum(axis=0)/(1-labels).sum(axis=0)
                all_data = data.mean(axis=0)
                in_dict = compute_metrics(fileconfig["metrics"], in_data, maxacc, prefix="in")
                out_dict = compute_metrics(fileconfig["metrics"], out_data, maxacc, prefix="out")
                all_dict = compute_metrics(fileconfig["metrics"], all_data, maxacc, prefix="all")
                fused_dict = dict(**in_dict, **out_dict, **all_dict)
                my_results = fused_dict
            metric_results = update_results(metric_results, method_key, method_params, my_results)
        json.dump(metric_results, open(output_file, "w"))

        
        






