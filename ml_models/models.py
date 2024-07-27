## Load the ML models from the log files.
import torch
from opacus.validators import ModuleValidator
from opacus import GradSampleModule
from torchvision.models import resnet18
import os
import torch.nn as nn
from ml_models.resnets import CustomBasicBlock, CustomResNet
from transformers import DistilBertForSequenceClassification


def get_model_w_grads(path: str, dset: str, runid: int, method_key, device: str):
    """ Load an ML model from file. """
    if dset == "cifar":
        return get_model_w_grads_cifar(path, runid, method_key, device)
    elif dset == "imdb":
        return get_model_w_grads_imdb(path, runid, method_key, device)
    elif dset == "cancer":
        return get_model_w_grads_cancer(path, runid, method_key, device)


def get_model_w_grads_cifar(path, run_id, method_key, device):
    if method_key == 'shap':
        model = CustomResNet(CustomBasicBlock, layers=[2, 2, 2, 2], num_classes=1000)
    else:
        model = resnet18(weights=None, num_classes=1000)
    model.fc = nn.Linear(512, 10, bias=True)
    target_path = f"{path}/CIFAR10_Cinf_tau0.0_batch32_ep30_resnet_{run_id}.pt"
    print(target_path)
    if os.path.exists(target_path):
        dset_dict = torch.load(target_path, map_location=device)
        model = ModuleValidator.fix(model)
        model.load_state_dict(dset_dict["model_plain"])
        if method_key != 'shap':
            model = GradSampleModule(model, loss_reduction = "mean").to(device)
        model = model.eval()
        return model, dset_dict
    else:
        return None, None


def get_model_w_grads_imdb(path: str, runid: int, method_key, device, opacus=True):
    n_layers = 4
    target_path = f"{path}/imdb_Cinf_tau0.0_batch64_ep8_bert{n_layers}_{runid}.pt"
    if os.path.exists(target_path):
        dset_dict = torch.load(target_path, map_location=device)
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        model.distilbert.transformer.layer = model.distilbert.transformer.layer[:n_layers]
        model = ModuleValidator.fix(model)
        model_opacus = GradSampleModule(model, loss_reduction = "mean").to(device)
        model_opacus.load_state_dict(dset_dict["model_plain"])
        model_opacus = model_opacus.eval()
        return model_opacus, dset_dict
    else:
        return None, None

## Functions to forward distillbert latent_representations.
def distilbert_latents(model, input_ids, attention_mask, embeddings=False):
    """ Return the latent representations. if embeddings==True, the embeddings are returned
    if embeddings == False, the latent represnetation before the classification head is used. """
    
    if embeddings:
        out_embds = model.distilbert.embeddings(input_ids)
        out = model(input_ids=None, attention_mask=attention_mask, inputs_embeds=out_embds)
        return out.logits, out_embds.reshape(len(out_embds), -1)
    else:
        out = model(input_ids, attention_mask, output_hidden_states=True)
    return  out.logits, out.hidden_states[-1][:, 0, :]


def distilbert_classification_head(model, inputs: torch.Tensor, attention_mask = None, embeddings=False):
    if embeddings:
        inputs = inputs.reshape(len(inputs), -1, model._module.config.dim)
        out = model(input_ids=None, attention_mask=attention_mask, inputs_embeds=inputs)
        return out.logits
    else:
        pooled_output = model.pre_classifier(inputs)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        #pooled_output = model_trained.dropout(pooled_output)  # (bs, dim)
        out = model.classifier(pooled_output)  # (bs, num_labels)
    return out

def get_model_w_grads_cancer(path, runid, method_key, device):
    target_path = f"{path}/SkinCancer_Cinf_tau0.0_batch64_ep40_resnet_{runid}.pt"
    print(target_path)
    if os.path.exists(target_path):
        if method_key == 'shap':
            model = CustomResNet(CustomBasicBlock, layers=[2, 2, 2, 2], num_classes=1000)
        else:
            model = resnet18(weights=None, num_classes=1000)
        model.fc = nn.Linear(512, 7, bias=True)
        dset_dict = torch.load(target_path, map_location=device)
        model = ModuleValidator.fix(model)
        model.load_state_dict(dset_dict["model_plain"])
        if method_key != 'shap':
            model = GradSampleModule(model, loss_reduction = "mean").to(device)
        model = model.eval()
        return model, dset_dict
    else:
        return None, None