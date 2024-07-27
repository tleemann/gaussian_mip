## Helper functions for risk predictors
import shap
from lime.lime_image import LimeImageExplainer, ImageExplanation
from transformers import AutoTokenizer
import torch
from ml_models.models import distilbert_latents, distilbert_classification_head
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from ml_models.vae_model import VAE

from gmip.utils import language_kernel_shap

def compute_cov_mahalanobis(base_trainloader, assess_loader, use_dims=None):
    """ Compute data covariance matrix using base_trainloader and 
        compute mahalanobis distance for points in assess_loader
    """
    feature_list = []
    for data in base_trainloader:
        if isinstance(data, dict):
            ## Load a random model 
            distilbert_latents(model, data["input_ids"].to(use_device), data["attention_mask"].to(use_device), embeddings=False)
        else:
            feature_list.append(data[0].reshape(len(data[0]), -1))
            #print(feature_list[-1].shape)
        if use_dims is not None:
            ## Randomly select ~ use_dims dimentions from the latents
            evernth = feature_list[-1].shape[1] // use_dims
            ind_use = torch.arange(feature_list[-1].shape[1] // evernth) * evernth
            feature_list[-1] = feature_list[-1][:, ind_use]
    grad_estimation = torch.cat(feature_list, dim=0)
    #grad_estimation = grad_estimation.reshape(len(grad_estimation), -1)
    grad_means = grad_estimation.mean(axis=0, keepdim=True)
 
    grad_vars_norm = grad_estimation - grad_means
    #sigma_diag = grad_vars_norm.var(axis=0)
    sigma = (grad_vars_norm.t() @ grad_vars_norm)/len(grad_vars_norm)
    print("Inverting Sigma...")
    kinv = torch.cholesky_inverse(sigma)
    print("Computing K...")
    feature_list = []
    for data in assess_loader:
        feature_list.append(data[0].reshape(len(data[0]), -1))
        if use_dims is not None:
            ## Randomly select ~ use_dims dimentions from the latents
            evernth = feature_list[-1].shape[1] // use_dims
            ind_use = torch.arange(feature_list[-1].shape[1] // evernth) * evernth
            feature_list[-1] = feature_list[-1][:, ind_use]
    all_estimation = torch.cat(feature_list, dim=0)
    #all_estimation = all_estimation.reshape(len(all_estimation), -1)
    print(all_estimation.shape)
    all_estimation = all_estimation - grad_means
    Ks = torch.sum(all_estimation * (kinv @ all_estimation.t()).t(), axis=1)
    return Ks

def compute_mahalanobis_latents(model, base_trainloader, assess_loader, use_device="cuda:3"):
    """ Compute the mahalanobis distance for a model with a latent encoder. """
    feature_list = []
    for data in tqdm(base_trainloader, total=len(base_trainloader)):
        with torch.no_grad():
            _, inputs = distilbert_latents(model, data["input_ids"].to(use_device), data["attention_mask"].to(use_device), embeddings=False)
            feature_list.append(inputs.detach())
            #print(inputs.shape)
    grad_estimation = torch.cat(feature_list, dim=0)
    grad_estimation = grad_estimation.reshape(len(grad_estimation), -1)
    grad_means = grad_estimation.mean(axis=0, keepdim=True)
 
    grad_vars_norm = grad_estimation - grad_means
    #sigma_diag = grad_vars_norm.var(axis=0)
    sigma = (grad_vars_norm.t() @ grad_vars_norm)/len(grad_vars_norm)
    print("Inverting Sigma...")
    kinv = torch.cholesky_inverse(sigma)
    print("Computing K...")
    feature_list = []
    for data in assess_loader:
        with torch.no_grad():
            _, inputs = distilbert_latents(model, data["input_ids"].to(use_device),  data["attention_mask"].to(use_device), embeddings=False)
            feature_list.append(inputs)
    all_estimation = torch.cat(feature_list, dim=0)
    all_estimation = all_estimation.reshape(len(all_estimation), -1)
    print(all_estimation.shape)
    all_estimation = all_estimation - grad_means
    Ks = torch.sum(all_estimation * (kinv @ all_estimation.t()).t(), axis=1)
    return Ks


        
def compute_shap(model, testloader, n_samples=20, var_use=0.01, agg='var', agg_param='mean', pretrained_lang_model="distilbert-base-uncased", use_device='cuda:3'):
    model.to(use_device)
    
    var_shap_list = []
    requires_grad(model)
    for i, data in tqdm(enumerate(testloader), total=len(testloader)):
        zero_batch_grad(model)
        if isinstance(data, dict): ## Language dateset                                        
            shap_values = torch.from_numpy(language_kernel_shap(model, 
                                                                data, 
                                                                pretrained_model=pretrained_lang_model, 
                                                                use_device=use_device)).float()
        else:
            inputs, _ = data
            inputs = inputs.to(use_device)
            var_inputs = inputs.unsqueeze(0)*torch.ones(n_samples, 1, 1, 1, 1, device=use_device)
            var_inputs = var_inputs + torch.randn_like(var_inputs)*var_use
            var_inputs = var_inputs.reshape(-1, *var_inputs.shape[2:])

            shap_explainer = shap.DeepExplainer(model, var_inputs)
            shap_values = shap_explainer.shap_values(var_inputs, check_additivity=False)
            shap_values = torch.from_numpy(shap_values.reshape(n_samples, inputs.shape[0], -1)).float()
            
            if agg == "var":
                shap_values = (1e4)*(shap_values.var(dim=0))
            elif agg == "mean":
                shap_values = shap_values.mean(dim=0)
            else:
                raise ValueError("Unknown param aggregation type.")

        if agg_param == "var":
            shap_values = shap_values.var(dim=1)
        elif agg_param == "mean":
            shap_values = shap_values.mean(dim=1)
        else:
            raise ValueError("Unknown global aggregation type.")
        var_shap_list.append(shap_values)
    return torch.cat(var_shap_list, dim=0)
    
def compute_lime(model, testloader, n_samples=20, kernel_width=0.2, feature_selection='auto', agg_glob='mean', use_device='cuda'):
    
    model.to(use_device)
    
    def pred_fn(input):
        """
            Function that encapsulates a torch nn.Module's
            forward function into returning a numpy array
            of probabilities. This is necessary for LIME
        """        
        model.eval()
        input_tensor = torch.from_numpy(input).permute(0, -1, 1, 2).to(use_device)
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()
    
    explainer = LimeImageExplainer(kernel_width=kernel_width,
                                     feature_selection=feature_selection)
    
    lime_scores = []
    for _, data in tqdm(enumerate(testloader), total=len(testloader)):
        with torch.no_grad():
            batched_input, _ = data
            # unfortunately LIME doesn't support batches :-(
            for single_input in batched_input: 
                single_input = single_input.permute(1, 2, 0)
                explanation: ImageExplanation = explainer.explain_instance(single_input.numpy(),
                                                                           classifier_fn=pred_fn,
                                                                           num_samples=n_samples)
                top_label = explanation.top_labels[0]
                feature_contribution = dict(explanation.local_exp[top_label])
                # feature_contribution contains the effect on each segmentation in the
                # image. So, I'm vectorizing these contributions according to the segments
                # In this way if there are bigger portions of the image that contain 
                # contribution weights, we need to reflect this in computing the mean.
                heatmap = np.vectorize(feature_contribution.get)(explanation.segments)
                if agg_glob == 'mean':
                    lime_scores.append(heatmap.mean())
                elif agg_glob == 'var':
                    lime_scores.append(heatmap.var())
                else:
                    raise ValueError("Unkown global aggregation type")
                
    return torch.cat(lime_scores, dim=0)
            
            
    

def vae_reconstruction_loss(base_trainloader, assess_loader, vae_model_file="vae_checkpoints/checkpoint.pth.tar", use_device="cuda:3"):
    """ Compute reconstruction loss of a pre-trained VAE model on the background distribution. """
    print(use_device)
    res = torch.load(vae_model_file, map_location=use_device)
    myvae = VAE()
    myvae.load_state_dict(res["state_dict"])
    myvae = myvae.to(use_device)
    mse_loss = nn.MSELoss(reduction="none")
    with torch.no_grad():
        mse_loss_list = []
        for i, (data, _) in enumerate(assess_loader):
            data = data.to(use_device)
            recon_batch, mu, logvar = myvae(data)
            MSE = mse_loss(recon_batch, data).reshape(len(data), -1)
            mse_loss_list.append(MSE.sum(axis=-1))
        all_losses = torch.cat(mse_loss_list)
    return all_losses.cpu()

## Loss
def compute_loss(model, testloader, criterion, use_device="cuda"):
    """ Eval accuracy of model. """
    correct = 0
    prob = 0.0
    model.eval()
    model.to(use_device)
    loss_list = []
    with torch.no_grad():
        # for data in tqdm(testloader)
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            if isinstance(data, dict): ## Language dateset
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        data[k] = v.to(use_device)
                        batchlen = len(v)
                labels = data["label"]
                outputs = model(data["input_ids"], attention_mask=data["attention_mask"], labels=data["label"])["logits"]
            else:
                inputs, labels = data
                inputs = inputs.to(use_device)
                labels = labels.to(use_device)
                outputs = model(inputs)
            loss_batch = criterion(outputs, labels)
            loss_list.append(loss_batch.detach())
            #print(loss_batch.shape)
    return torch.cat(loss_list, dim=0)

def compute_confidence(model, testloader, criterion, use_device="cuda"):
    """ Eval confidence of the model. No labels are required for this predictor. """
    correct = 0
    prob = 0.0
    model.eval()
    model.to(use_device)
    loss_list = []
    with torch.no_grad():
        # for data in tqdm(testloader)
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            if isinstance(data, dict): ## Language dateset
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        data[k] = v.to(use_device)
                        batchlen = len(v)
                labels = data["label"]
                outputs = model(data["input_ids"], attention_mask=data["attention_mask"], labels=data["label"])["logits"]
            else:
                inputs, labels = data
                inputs = inputs.to(use_device)
                labels = labels.to(use_device)
                outputs = model(inputs)
            ## Define confidence as the log-odds difference between 1st most likely and 2nd most likely.
            scores_sort = outputs.sort(dim=1)[0]
            loss_list.append(scores_sort[:,-1]-scores_sort[:,-2])
            #print(loss_batch.shape)
    return torch.cat(loss_list, dim=0)


def estimate_hutchinson_traces(model, testloader, criterion, use_device="cuda", n_samples = 10, eps=0.01, embeddings=False):
    """ Compute the curvature estimator for the model.
        In case of the IMDB dataset, we suppose that the model wrapper is given and we perturb the latent
        representations to estimate curvature.
    """
    correct = 0
    prob = 0.0
    model.eval()
    model.to(use_device)
    loss_list = []
    with torch.no_grad():
        # for data in tqdm(testloader)
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            if isinstance(data, dict): ## Language dateset
                with torch.no_grad():
                    attention_mask = data["attention_mask"].to(use_device)
                    _, inputs = distilbert_latents(model, data["input_ids"].to(use_device), attention_mask, embeddings=embeddings)
                    labels = data["label"].to(use_device)
                    attention_mask_rep = attention_mask.unsqueeze(0).expand(2*n_samples+1, *attention_mask.shape)
                    #print(attention_mask_rep.shape)
            else:
                inputs, labels = data
                orgshape = inputs.shape[1:4]
                inputs = inputs.to(use_device).reshape(len(inputs), -1)
                labels = labels.to(use_device)
            # Perturb the samples: :n_samples +v, n_samples:2_nsamples -> -v, 2*n_samples original input
            inputs = inputs.unsqueeze(0)*torch.ones(2*n_samples+1, 1, 1, device=use_device)
            rademacher = (1.0-2.0*(torch.rand_like(inputs)< 0.5).float())
            rademacher[n_samples:2*n_samples] = -rademacher[:n_samples]
            rademacher[2*n_samples] = 0.0
            inputs = inputs + rademacher*eps
            #print(inputs.shape)
            
            labels = labels.repeat(2*n_samples+1) 
            with torch.no_grad():
                if isinstance(data, dict):
                    inputs = inputs.reshape(-1, inputs.size(-1))
                    outputs = distilbert_classification_head(model, inputs, attention_mask_rep.reshape(len(inputs), -1), embeddings=embeddings)
                else:
                    inputs = inputs.reshape(-1, *orgshape)
                    outputs = model(inputs)
            loss_batch = criterion(outputs, labels)
            loss_batch = loss_batch.reshape(2*n_samples+1, -1)
            ## Compute the 2nd derivative
            d2 = -2*loss_batch[2*n_samples].unsqueeze(0)-loss_batch[:n_samples]-loss_batch[n_samples:2*n_samples]
            d2 = torch.var(d2/(eps*eps), dim=0)
            #print(d2.shape)
            loss_list.append(d2)
            #print(loss_batch.shape)
    return torch.cat(loss_list, dim=0)


def aggregate_crop_grads(model):
    """ Aggregate gradients of a subbatch and store them in this object. """
    grad_sum = 0.0
    grad_list_all = []
    with torch.no_grad():
        for t in model.parameters():
            if t.requires_grad == False:
                continue
            
            if t.grad_sample is not None:
                if t.grad_sample.shape[0] != 16:
                    print(type(t.grad_sample), type(t), t)
                    print(t.grad_sample.shape)
                grad_list_all.append(t.grad_sample.reshape(len(t.grad_sample), -1))
    grad_list_all = torch.cat(grad_list_all, dim=1)
    return grad_list_all
    
def zero_batch_grad(model):
    for t in model.parameters():
        t.grad_sample = None
        t.grad_summed = None
        t.grad = None
        
def requires_grad(model):
    for n, t in model.named_parameters():
        if n == "_module.distilbert.embeddings.position_embeddings.weight": ## Do not derive positional embeddings, as this yields errors.
            t.requires_grad_(False)
        else:
            t.requires_grad_(True)


def compute_input_gradients(model, testloader, criterion, use_device="cuda", n_samples = 15, var_use = 0.01, agg="var", agg_param="mean", embeddings=False):
    """ Compute gradients and vargrad.
        agg: How to aggregate param-wise over the gradients sampled from the neighborhood
        agg_glob: How to aggregate over the paramters.
    
    """
    correct = 0
    prob = 0.0
    model.train()
    model.to(use_device)
    print(type(model))
    var_score_list = []
    for i, data in tqdm(enumerate(testloader), total=len(testloader)):
        zero_batch_grad(model)
        with torch.no_grad():
            if isinstance(data, dict): ## Language dateset
                attention_mask = data["attention_mask"].to(use_device)
                _, inputs = distilbert_latents(model, data["input_ids"].to(use_device), attention_mask, embeddings=embeddings)
                labels = data["label"].to(use_device)
                attention_mask_rep = attention_mask.unsqueeze(0).expand(n_samples, *attention_mask.shape)
        
            else:
                inputs, labels = data
                orgshape = inputs.shape[1:4]
                inputs = inputs.to(use_device).reshape(len(inputs), -1)
                labels = labels.to(use_device)
            # Perturb the samples: :n_samples +v, n_samples:2_nsamples -> -v, 2*n_samples original input
            inputs = inputs.unsqueeze(0)*torch.ones(n_samples, 1, 1, device=use_device)
        inputs.requires_grad_(True)
        batch_sz = inputs.size(1)
        labels = labels.to(use_device)
        labels = labels.repeat(n_samples) #torch.randint(high=10, size=(inputs.shape[0],)).to(use_device) #
        if isinstance(data, dict):
            inputs = inputs.reshape(-1, inputs.size(-1))
            outputs = distilbert_classification_head(model, inputs, attention_mask_rep.reshape(len(inputs), -1), embeddings=embeddings)
        else:
            inputs = inputs.reshape(-1, *orgshape)
            outputs = model(inputs)
        loss_batch = criterion(outputs, labels)
        grads = torch.autograd.grad(loss_batch, inputs)[0].detach()
        #print(grads.shape)
        grad_list = grads.reshape(n_samples, batch_sz, -1)
        if agg == "var":
            var_scores = (1e4)*(grad_list.var(dim=0))
        elif agg == "mean":
            var_scores = grad_list.mean(dim=0)
        else:
            raise ValueError("Unknown sample aggregation type.")

        if agg_param == "var":
            var_scores = var_scores.var(dim=1)
        elif agg_param == "mean":
            var_scores = var_scores.mean(dim=1)
        else:
            raise ValueError("Unknown param aggregation type.")
        var_score_list.append(var_scores)
    return torch.cat(var_score_list, dim=0)


def compute_loss_vargrad(model, testloader, criterion, use_device="cuda", n_samples = 15, var_use = 0.01, agg="var", agg_param="mean"):
    """ Compute gradients and vargrad.
        agg: How to aggregate param-wise over the gradients sampled from the neighborhood
        agg_glob: How to aggregate over the paramters.
    
    """
    correct = 0
    prob = 0.0
    model.train()
    model.to(use_device)
    print(type(model))
    var_score_list = []
    requires_grad(model)
    for i, data in tqdm(enumerate(testloader), total=len(testloader)):
        zero_batch_grad(model)
        with torch.no_grad():
            if isinstance(data, dict): ## Language dateset
                attention_mask = data["attention_mask"].to(use_device)
                _, inputs = distilbert_latents(model, data["input_ids"].to(use_device), attention_mask, embeddings=True)
                labels = data["label"].to(use_device)
                attention_mask_rep = attention_mask.unsqueeze(0).expand(n_samples, *attention_mask.shape)
            else:
                inputs, labels = data
                orgshape = inputs.shape[1:4]
                inputs = inputs.to(use_device).reshape(len(inputs), -1)
                labels = labels.to(use_device)
            # Perturb the samples: :n_samples +v, n_samples:2_nsamples -> -v, 2*n_samples original input
            inputs = inputs.unsqueeze(0)*torch.ones(n_samples, 1, 1, device=use_device)
            inputs += torch.randn_like(inputs)*var_use

        #inputs.requires_grad_(True)
        batch_sz = inputs.size(1)
        labels = labels.to(use_device)
        labels = labels.repeat(n_samples) #torch.randint(high=10, size=(inputs.shape[0],)).to(use_device) #
        if isinstance(data, dict):
            inputs = inputs.reshape(-1, inputs.size(-1))
            outputs = distilbert_classification_head(model, inputs, attention_mask_rep.reshape(len(inputs), -1), embeddings=True)
        else:
            inputs = inputs.reshape(-1, *orgshape)
            outputs = model(inputs)
        loss_batch = criterion(outputs, labels)
        loss_batch.backward()
        grad_list = aggregate_crop_grads(model).detach()
        grad_list = grad_list.reshape(n_samples, -1, grad_list.shape[-1]).detach()
        if agg == "var":
            var_scores = (1e4)*(grad_list.var(dim=0))
        elif agg == "mean":
            var_scores = grad_list.mean(dim=0)
        else:
            raise ValueError("Unknown sample aggregation type.")

        if agg_param == "var":
            var_scores = var_scores.var(dim=1)
        elif agg_param == "mean":
            var_scores = var_scores.mean(dim=1)
        else:
            raise ValueError("Unknown param aggregation type.")
        var_score_list.append(var_scores)
    return torch.cat(var_score_list, dim=0)


def __random_rows(tensor, n):
    """
    Randomly select n rows from a PyTorch tensor.

    Args:
    - tensor: PyTorch tensor
    - n: Number of rows to select

    Returns:
    - selected_rows: Tensor containing the selected rows
    """
    num_rows = tensor.size(0)
    
    # Generate random indices
    indices = torch.randperm(num_rows)[:n]
    
    # Select rows using the indices
    selected_rows = tensor[indices]
    
    return selected_rows, indices