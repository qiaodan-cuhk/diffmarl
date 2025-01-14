#!/bin/sh

datatypes=("expert" "medium" "medium-replay" "random")  # "expert" "random"
datanums=0    # 1 2 3

## algo selection
# difftype="SRPO"     # DQL
marltype="JAL"      # Can train JAL, IND, CTDE
TASK="HalfCheetah-v2"
device=0

# if train on mixed data: --mixed_data

for types in "${datatypes[@]}"
do
    echo "data types: $types, device: $device, data number: $datanums, mix datasets: True"
    python pretrain_critic.py --data_type $types --device $device --srpo_mode $marltype --mixed_data &
done