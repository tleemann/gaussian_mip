#!/bin/bash
# Training of models for utility experiment. Pass a single argument to indicate the dataset ("cifar10", "adult", "purchase")
# Run the script from the main directory, e.g., > ./train_scripts/train_models_util.sh cifar10

SAVEPATH=models
mkdir -p $SAVEPATH

for ((prlvl=0;prlvl<20;prlvl++));
do
for ((runid=0;runid<=4;runid++));
do

if [ $1 == "cifar10" ]
then
cutoff=500
batchsize=400
export PYTHONPATH="."; python3 train_scripts/train_cifar.py --num_train 48000 $cutoff DP$prlvl $runid $SAVEPATH $batchsize 10 resnet56
export PYTHONPATH="."; python3 train_scripts/train_cifar.py --num_train 48000 $cutoff MIP$prlvl $runid $SAVEPATH $batchsize 10 resnet56
fi

if [ $1 == "purchase" ]
then
cutoff=2000.0
batchsize=795
kd_val=2580
numtrain=54855
epochs=3
export PYTHONPATH="."; python3 train_scripts/train_tabular.py purchase $cutoff DP$prlvl $runid $SAVEPATH $batchsize $epochs --kval $kd_val --model_dims $kd_val --num_train $numtrain
export PYTHONPATH="."; python3 train_scripts/train_tabular.py purchase $cutoff MIP$prlvl $runid $SAVEPATH $batchsize $epochs --kval $kd_val --model_dims $kd_val --num_train $numtrain
fi

if [ $1 == "adult" ]
then
cutoff=800.0
batchsize=1000
kd_val=1026
numtrain=43000
epochs=20
export PYTHONPATH="."; python3 train_scripts/train_tabular.py adult $cutoff DP$prlvl $runid $SAVEPATH $batchsize $epochs --kval $kd_val --model_dims $kd_val --num_train $numtrain
export PYTHONPATH="."; python3 train_scripts/train_tabular.py adult $cutoff MIP$prlvl $runid $SAVEPATH $batchsize $epochs --kval $kd_val --model_dims $kd_val --num_train $numtrain
fi

done
done
echo "DONE."