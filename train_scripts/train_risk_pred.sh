#!/bin/bash
# Train models for indidivual risk prediction.
# Pass an argument to indicate the dataset/model combination ("cifar10", "bert", "cancer") and the device (e.g., cuda:0)
# Run the script from the main directory, e.g., > ./train_scripts/train_risk_pred.sh cifar10 cuda:0

SAVEPATH=models_trace
mkdir -p $SAVEPATH

startid=0
for ((runid=$startid;runid<($startid+50);runid++));
do

if [ $1 == "cancer" ]
then
export PYTHONPATH="."; python3 train_scripts/train_skin_cancer.py inf 0.0  $runid $SAVEPATH 64 40 resnet --num_train 1000 --trace_grads true --record_steps 20 --device $2
fi

if [ $1 == "bert" ]
then
export PYTHONPATH="."; python3 train_scripts/train_bert.py inf 0.0 $runid $SAVEPATH 64 8 bert4 --num_train 2000 --trace_grad true --record_steps 7 --device $2
fi

if [ $1 == "cifar10" ]
then
export PYTHONPATH="."; python3 train_scripts/train_cifar.py inf 0.0 $runid $SAVEPATH 32 30 resnet --device $2 --num_train 2000 --trace_grads true --record_steps 50
fi

done
echo "DONE."