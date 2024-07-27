import torch
from torch import autograd
from torch.func import grad, grad_and_value, vmap
from datetime import datetime
import torch.nn as nn
import copy
from torch.utils.data import DataLoader
from tqdm import tqdm

MAX_PHYSICAL_BATCH = 20 # Maximum batch size on GPU.

class RandomSubsetDataset():
    def __init__(self, base_dataset, subsample_ratio: float = 0.5, n_use_total=None):
        """ Subsample subsample ratio from n_use_total examples of the dataset."""
        self.base_dataset = base_dataset
        if n_use_total == None:
            n_use_total = len(base_dataset)
        self.num_examples_use = int(n_use_total*subsample_ratio + 1e-2)
        self.sample_idx = torch.randperm(n_use_total)[:self.num_examples_use]
        print("Using sub dataset of size", self.num_examples_use)

    def __len__(self):
        return self.num_examples_use

    def __getitem__(self, idx: int):
        return self.base_dataset[self.sample_idx[idx].item()]

    def get_samples_used(self):
        """ Return the list of samples used for trainin in this datasubset """
        return self.sample_idx
        
## Fix functions for models.
def return_groupnorm_params(batch_norm):
    """ return weights and bias """
    wdivstd = batch_norm._parameters["weight"]/(torch.sqrt(batch_norm._buffers["running_var"]+batch_norm.eps))
    return wdivstd.detach(), (-wdivstd*batch_norm._buffers["running_mean"]+batch_norm._parameters["bias"]).detach()

def get_bn_replacement(batchnorm2d):
    with torch.no_grad():
        fact, bias = return_groupnorm_params(batchnorm2d)
        dim = len(fact)
        replacement = nn.Conv2d(dim, dim, 1)
    replacement._parameters["weight"] = nn.Parameter(torch.diag(fact).cpu().reshape(dim, dim, 1,1))
    replacement._parameters["bias"] = nn.Parameter(bias.cpu())
    return replacement

def recursive_fix(module):
    for name, sub in module.named_children():
        if type(sub) == nn.modules.batchnorm.BatchNorm2d:
            #print(name, "has to be fixed")
            module.__setattr__(name, get_bn_replacement(sub))
        else:
            recursive_fix(sub)


def state_dict_to_cpu(sdict, device = "cpu"):
    for k, v in sdict.items():
        sdict[k] = sdict[k].to(device)
    return sdict


def list_to_cpu(slist, device = "cpu"):
    for i in range(len(slist)):
        slist[i] = slist[i].to(device)
    return slist


def noisy_train(model_priv, trainloader, criterion, optimizer_priv, epoch, start_epoch, 
                scheduler=None, use_device="cuda:0", collect_stepwise=False, state_dict_fn=None, 
                return_n_last_dims=None, return_grads_every=1):
    """
        Our implementation of the Noisy Private Training (Algorithm 1).
        model: model that should be trained
        train_loader: data loader for training
        test_loader: data loader for testing
        criterion: Loss function
        optimizer: Optimizer to use
        epoch: Number of epochs to train
        C: range used for gradient clipping
        tau: noise magnitude
        collect_stepwise: Collect the model parameters and gradients after each training step.
        state_dict_fn: Function that retuns a state dict of trainable parameters for the model
        return_n_last_grads: return only the last n items of the gradients to save memory.
        return_grads_every: return not every gradient and parameters but only every couple of steps.
    """
    # eval the model for a specific modified data set
    # Return accuracy and average true class probability.
    best_acc = 0.0
    best_prob = 0.0
    if collect_stepwise:
        step_grad_list = []
        step_param_list = []
    for epoch in range(start_epoch, start_epoch+epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        correct = 0
        grad_sum2 = 0.0
        model_priv.train()
        print('starting epoch %d'%(epoch+1), datetime.now())
        num_samples = 0
        for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
            batchlen = 0
            # get the inputs; data is a list of [inputs, labels]
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        data[k] = v.to(use_device)
                        batchlen = len(v)
                labels = data["label"]
            else: ## Tuple datasets
                inputs, labels = data
                inputs = inputs.to(use_device)
                labels = labels.to(use_device)
                batchlen =  len(inputs)
            ## Split the batch in physical batches.
            optimizer_priv.zero_grad()
            outputs_list = []
            for subbatch_start in range(0, batchlen, MAX_PHYSICAL_BATCH):
                optimizer_priv.zero_subbatch_grad(model_priv)
                subbatch_end = min(subbatch_start+MAX_PHYSICAL_BATCH, batchlen)
                #print("Using batch", subbatch_start, subbatch_end)
                if isinstance(data, dict):
                    data_batch = {}
                    for k, v in data.items():
                        data_batch[k] = v[subbatch_start:subbatch_end]
                    outputs_batch = model_priv(data_batch["input_ids"], attention_mask=data_batch["attention_mask"], labels=data_batch["label"])["logits"]
                    #print(outputs_batch, data_batch["label"])
                    loss_batch = criterion(outputs_batch, data_batch["label"])
                else: ## Tuple datasets
                    input_batch = inputs[subbatch_start:subbatch_end].clone()
                    labels_batch = labels[subbatch_start:subbatch_end].clone()
                    outputs_batch = model_priv(input_batch)
                    loss_batch = criterion(outputs_batch, labels_batch) # Note that loss should return one element per batch item.
                loss_batch.backward()
                # forward + backward + optimize
                optimizer_priv.aggregate_crop_grads(model_priv)
                running_loss += loss_batch.item()
                outputs_list.append(outputs_batch.detach())

            outputs = torch.cat(outputs_list)
            if collect_stepwise and (i % return_grads_every == 0):
                if state_dict_fn == None:
                    state_dict = model_priv.state_dict()
                else:
                    state_dict = state_dict_fn(model_priv)
                step_param_list.append(state_dict_to_cpu(copy.deepcopy(state_dict)))
                # Perform step
                grad_sum_increment, gradients = optimizer_priv.step(model_priv)
                if return_n_last_dims is not None:
                    step_grad_list.append(list_to_cpu(gradients[-return_n_last_dims:]))
                else:
                    step_grad_list.append(list_to_cpu(gradients))
            else:
                # Perform step
                grad_sum_increment, _ = optimizer_priv.step(model_priv)
            grad_sum2 += grad_sum_increment

            num_samples += batchlen
            # print statistics
            
            
            pred = outputs.max(1, keepdim=True)[1]
            #print(pred, model_priv.classifier.weight)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            #if i % 100 == 0:
            #    print(running_loss/num_samples)
            #    print("Acc:", correct/num_samples)

        print('Training [%d] loss: %.5f; acc: %.5f; grad_sum: %.5f' % (epoch + 1, running_loss / len(trainloader.dataset), correct / len(trainloader.dataset), torch.sqrt(grad_sum2 / len(trainloader.dataset))))
        if scheduler is not None:
            scheduler.step()

    if collect_stepwise:
        return model_priv, step_grad_list, step_param_list
    else:
        return model_priv



class PrivateOptimizer():
    def __init__(self, base_optimizer, model,  C=float("inf"), tau=0.0):
        self.model = model
        self.base_optimizer = base_optimizer
        self.C = C
        self.tau = tau
        self.mygradlist = []
        self.grad_sum = 0.0
        self.batch_len = 0.0

    def aggregate_crop_grads(self, model):
        """ Aggregate gradients of a subbatch and store them in this object. """
        grad_sum = 0.0
        with torch.no_grad():
            for t in model.parameters():
                if t.requires_grad == False:
                    continue
                #print(type(t.grad_sample), type(t))
                if t.grad_sample is not None:
                    grad_sum += torch.sum(t.grad_sample.reshape(len(t.grad_sample), -1).pow(2), dim=1)
        self.batch_len += len(grad_sum)
        target_mult = torch.ones_like(grad_sum)
        target_mult[grad_sum > self.C*self.C] = self.C/torch.sqrt(grad_sum[grad_sum > self.C*self.C])

        self.grad_sum += torch.sum(grad_sum.detach())
        with torch.no_grad():
            for pidx, t in enumerate(model.parameters()):
                if t.requires_grad == False:
                    if len(self.mygradlist) <= pidx:
                        self.mygradlist.append([])
                    continue
                # Crop and store per microbatch sum
                cropgrad = torch.sum(target_mult.reshape([len(target_mult)]+[1]*(len(t.grad_sample.shape)-1))*t.grad_sample, 0).detach()
                if len(self.mygradlist) <= pidx:
                    self.mygradlist.append([cropgrad])
                else:
                    self.mygradlist[pidx].append(cropgrad)

    def step(self, model):
        #print(self.mygradlist)
        # Performing updates
        grad_list = []
        with torch.no_grad():
            for i, (torg, t) in enumerate(zip([p for p in self.base_optimizer.param_groups[0]["params"]], [p for p in model.parameters()])):
                if t.requires_grad == False:
                    continue
                if t.grad_sample is not None:
                    all_grad = torch.stack(self.mygradlist[i], dim=0).sum(dim=0)/self.batch_len
                    torg.grad = all_grad + self.tau*torch.randn(all_grad.shape[1:], device=all_grad.device)
                    grad_list.append(torg.grad.clone())
        #grad_diff = 0.0
        #for i, iref in zip([p for p in model_ref.parameters()], [p for p in self.base_optimizer.param_groups[0]["params"]]):
        #    grad_diff += torch.sum(i.grad-iref.grad)
        #    #print(i.grad.flatten()[:10].detach(), iref.grad.flatten()[:10].detach())
        #print("Grad_diff: ", grad_diff)
        #p1 = next(iter(model.parameters())).clone()
        self.base_optimizer.step()
        #p2 = next(iter(model.parameters()))
        #print("Update:", torch.sum(torch.abs(p1-p2)))
        return self.grad_sum, grad_list

    def zero_subbatch_grad(self, model):
        for t in model.parameters():
            t.grad_sample = None
            t.grad_summed = None
            self.base_optimizer.zero_grad()
            
    def zero_grad(self):
        #self.zero_subbatch_grad()
        self.batch_len = 0
        self.mygradlist = []
        self.base_optimizer.zero_grad()
        self.grad_sum = 0.0


def eval_model(model, testloader, use_device="cuda:0"):
    """ Eval accuracy of model. """
    correct = 0
    prob = 0.0
    model.eval()
    model.to(use_device)
    with torch.no_grad():
        # for data in tqdm(testloader)
        for i, data in enumerate(testloader):
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        data[k] = v.to(use_device)
                labels = data["label"]
                outputs_batch = model(data["input_ids"], attention_mask=data["attention_mask"])["logits"]
                _, predicted = torch.max(outputs_batch, 1)
            else:
                inputs, labels = data
                inputs = inputs.to(use_device)
                labels = labels.to(use_device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            #break
    print('Accuracy of the network on test samples: %.4f %%, average probability:  %.4f' % (
                    100 * correct / len(testloader.dataset), prob / len(testloader.dataset)))
    acc_avg = correct / len(testloader.dataset)
    return acc_avg


