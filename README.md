# Gaussian Membership Inference Privacy

This repository contains accompanying code for the paper [Gaussian Membership Inference Privacy](https://openreview.net/forum?id=2NUFe4TZMS) by Tobias Leemann*, Martin Pawelczyk*, and Gjergji Kasneci.

## Downloading datasets
The adult dataset is included in this repository and CIFAR-10 is available via pytorch. To download the purchase dataset, run the script
```
./tabular/download_datasets.sh
```

## Gradient Log Likelihood attack

In our work, we present a novel membership inference attack. We run this attack on models that were trained with the train scripts ```gmip/train_cifar.py``` and ```gmip/train_tabular.py````. An example is in the Notebook ``notebooks/GLiRAttack.ipynb``.

**Preliminary code version. More documentation and code to be added soon.**


## Reference
If you find our work or the ressources provided here useful, please consider citing our work, for instance using the following BibTex entry:

```
@InProceedings{leemann2023gaussian,
  title     = {Gaussian Membership Inference Privacy},
  author    = {Leemann, Tobias and Pawelczyk, Martin and Kasneci, Gjergji},
  booktitle = {37th Conference on Neural Information Processing Systems (NeurIPS)},
  year      = {2023}
}
```
