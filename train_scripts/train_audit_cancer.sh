#!/bin/bash
# Training of models for auditing experiment with GLiR (Figure 2, GLiRAttack.ipynb). Pass a single argument to indicate the dataset ("cifar10", "adult", "purchase")
# Run the script from the main directory, e.g., > ./train_scripts/train_models_util.sh cifar10

SAVEPATH=/mnt/ssd3/tobias/mi_auditing/models_trace
# mkdir -p $SAVEPATH
epochs=30
batchsize=32
startid=$2
for ((runid=$startid;runid<($startid+50);runid++));
do
export PYTHONPATH="."; python3 train_skin_cancer.py inf 0.0  $runid $SAVEPATH 64 40 resnet --num_train 1000 --trace_grads true --record_steps 20 --device $1
#export PYTHONPATH="."; python3 train_cifar.py inf 0.0 $runid $SAVEPATH $batchsize $epochs resnet --device $1 --num_train 2000 --trace_grads true
#export PYTHONPATH="."; python3 train_cifar.py inf 0.0 $runid $SAVEPATH $batchsize $epochs resnet --device $1 --num_train 2000 --trace_grads true
done
echo "DONE." 