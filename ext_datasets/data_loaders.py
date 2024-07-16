## Implement datasets for the different tasks
import torchvision
import torchvision.transforms as transforms
from ext_datasets.skin_cancer import SkinCancerDataset, preprocess
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

def get_datasets(dataset: str, full=False):
    """ Return train and test datasets. """
    if dataset == "cifar":
        return get_cifar_datasets()
    elif dataset == "cancer":
        return get_cancer_datasets()
    elif dataset == "imdb":
        return get_imdb_datasets(full)
    else:
        raise ValueError(f"Unknown dataset {dataset}")

def get_cifar_datasets():
    transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))])
    cifar_train = torchvision.datasets.CIFAR10(root = ".", train=True, download=True, transform=transform_test)
    cifar_test= torchvision.datasets.CIFAR10(root = ".", train=False, download=True, transform=transform_test)
    return cifar_train, cifar_test

def get_imdb_datasets(full=False):

    def preprocess_function(tokenizer, examples, max_seq_len=512):
        """Preprocess function for the dataset"""
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_seq_len, return_tensors='pt')

    ds = None
    imdb = load_dataset('imdb').with_format('torch', device="cpu") # format to pytorch tensors, but leave data on cpu
    imdb["train"] = imdb["train"].shuffle(seed=42).select(range(8000 if full else 4000))
    imdb["test"] = imdb["test"].shuffle(seed=42).select(range(100))
    del imdb["unsupervised"]
    ds = imdb

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    tokenizer_func = lambda ex: preprocess_function(tokenizer, ex)
    ## Tokenize and prepare dataset.
    ds_tokenized = ds.map(tokenizer_func, batched=True)

    return ds_tokenized["train"], ds_tokenized["test"]

def get_cancer_datasets():
    dfs, norms = preprocess()
    df_train, df_test = dfs
    norm_mean, norm_std = norms
    print(norm_mean)
    test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(norm_mean, norm_std)])
    df_train = df_train.sample(frac=1.0, random_state = 1)
    y = torch.tensor(list([int(df_train['cell_type_idx'].iloc[index]) for index in range(10)]))
    print(y[:10])
    train_dataset = SkinCancerDataset(df=df_train, transform=test_transform)
    test_dataset = SkinCancerDataset(df=df_test, transform=test_transform)
    return train_dataset, test_dataset