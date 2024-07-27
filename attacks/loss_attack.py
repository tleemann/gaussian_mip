## Attack score computation functions:
from tqdm import tqdm

from torch import nn as nn
import sys
import torch

from ext_datasets.data_loaders import get_datasets
from ml_models.models import get_model_w_grads
from dp_sgd import RandomSubsetDataset

## Loss based attack script.
## USAGE
## loss_attack.py <dataset> <device> <max_models>

## Loss
def compute_loss(model, testloader, use_device="cuda"):
    """ Eval accuracy of model. """
    correct = 0
    prob = 0.0
    model.eval()
    model.to(use_device)
    loss_list = []
    criterion = nn.CrossEntropyLoss(reduction="none").to(use_device)
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


TOTAL_TRAIN_POINTS = {"cifar": 4000, "imdb": 4000, "cancer": 2000}
if __name__ == "__main__":
    dataset = sys.argv[1]
    device = sys.argv[2]
    max_models = int(sys.argv[3])
    total_train_pts = TOTAL_TRAIN_POINTS[dataset]

    batch_size=32
    model_path = "/mnt/ssd3/tobias/mi_auditing/models_trace"
    #model_path="."
    mi_score_array = torch.zeros(max_models, total_train_pts)
    mi_labels_array = torch.zeros(max_models, total_train_pts, dtype=torch.long)
    my_train_dataset, _ = get_datasets(dataset)
    x, y = my_train_dataset[289]
    print(x[2, :10,:10])
    print(y)
    for i in range(max_models):
        my_model, dset_dict = get_model_w_grads(model_path, dataset, i, device)
        my_model.to(device)
        if my_model:
            print("loaded weights", i)
            train_point_idx_use = dset_dict["samples_used"]
            n_use = len(train_point_idx_use)
            total_train_pts = 2*n_use
            train_dataset = RandomSubsetDataset(my_train_dataset, subsample_ratio=1.0, n_use_total=total_train_pts)
            train_dataset.sample_idx = torch.arange(total_train_pts) # No random order 
            base_trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            all_samples = torch.zeros(total_train_pts)
            all_samples[train_point_idx_use]=1 # set the train indices to zero.
            mi_score_array[i,:] = compute_loss(my_model, base_trainloader, use_device=device)
            mi_labels_array[i,:] = all_samples

    torch.save((mi_score_array, mi_labels_array), f"results/mi_scores_loss_{dataset}_test.pt")
