#!/bin/bash
# Training of models for auditing experiment with GLiR (Figure 2, GLiRAttack.ipynb). Pass a single argument to indicate the dataset ("cifar10", "adult", "purchase")
# Run the script from the main directory, e.g., > ./train_scripts/train_models_util.sh cifar10

SAVEPATH=models
mkdir -p $SAVEPATH
epochs=10
for ((runid=0;runid<=4;runid++));
do

if [ $1 == "cifar10" ]
then
batchsize=500
export PYTHONPATH="."; python3 train_scripts/train_cifar.py --shallow True --trace_grads True inf 0.0 $runid $SAVEPATH $batchsize $epochs resnet56
fi

if [ $1 == "purchase" ]
then
batchsize=1970
export PYTHONPATH="."; python3 train_scripts/train_tabular.py --shallow True --trace_grads True purchase inf 0.0 $runid $SAVEPATH $batchsize $epochs
fi

if [ $1 == "adult" ]
then
batchsize=790
export PYTHONPATH="."; python3 train_scripts/train_tabular.py --shallow True --trace_grads True adult inf 0.0 $runid $SAVEPATH $batchsize $epochs
fi

done
echo "DONE."