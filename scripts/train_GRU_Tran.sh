#!/bin/bash

trap "exit" INT
# shellcheck disable=SC2068
run_ver='v1'
task='VA'
kfold=-1
max_epoch=1
#limit_train_batches=0.05
root_project='/media/nguyenhonghai/Data/projects/Affwild2-ABAW3-main3/'
data_dir='/home/nguyenhonghai/datasets/ABAW3/'
train_dir=$root_project'train_logs/'$task'_'$run_ver'/'
cfg_path=$root_project'conf/VA_GRU_Tran.yaml'
kfold_data_dir=$data_dir'/Kfold_Data/'
python -W ignore main.py --cfg $cfg_path \
        OUT_DIR $train_dir \
        N_KFOLD $kfold \
        DATA_LOADER.DATA_DIR $data_dir \
        DATA_LOADER.KFOLD_DATA_DIR $kfold_data_dir \
        OPTIM.MAX_EPOCH $max_epoch \
#        TRAIN.LIMIT_TRAIN_BATCHES $limit_train_batches
sleep 10
