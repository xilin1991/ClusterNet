#!/bin/bash

source ${your_env_path}/anaconda3/bin/activate
conda activate ${your_envs}
export CUDA_VISIBLE_DEVICES=${gpu_id} # your GPU number
export USE_AMP=true

batch_size=16
dataset_dir=${data_dir} # for example, data_dir=your_dataset_folder/DAVIS2017
n_clusters=30
pretrain_path=./checkpoints/sequence_nc${n_clusters}_thres010_BGC
results_path=./results/sequence_nc${n_clusters}_thres010_BGC
threshold=0.1

for seq in $(<${data_dir}/ImageSets/2016/val.txt); do # for example, '<your_dataset_folder/ImageSets/2016/val.txt' in parentheses.
    echo 'sequence: '"${seq}"
    python train_sequence_BGC.py --batch_size ${batch_size} \
        --data_dir ${dataset_dir} \
        --seq_name ${seq} \
        --n_clusters ${n_clusters} \
        --to_rgb \
        --with_gt \
        --pretrain_path ${pretrain_path} \
        --results_path ${results_path} \
        --threshold ${threshold}
done
