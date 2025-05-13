#!/bin/sh

# expert 用 datanum 3最好，medium+replay用datanum 1最好 random用的是0

datatypes=("medium" "medium-replay")  # "expert" "medium" "medium-replay" "random"
dataset_seed=1   # 1 2 3
mix=False

## algo selection
# difftype="SRPO"     # DQL
marltype="JAL"      # Can train JAL, IND, CTDE    seq使用ctde/jal critic
TASK="HalfCheetah-v2"
device=3

# if train on mixed data: --mixed_data

for types in "${datatypes[@]}"
do
    echo "data types: $types, device: $device, data number: $dataset_seed, mix datasets: $mix"
    python pretrain_critic.py --data_type $types --dataset_num $dataset_seed --device $device --srpo_mode $marltype &
done