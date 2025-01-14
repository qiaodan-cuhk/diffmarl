#!/bin/sh

# 在用seq去pretrain的时候，直接train的是agent1的条件分布，还需要用ind去train得到agent0的分布
datatypes=("expert" "medium" "medium-replay" "random")  # "expert" "random"
datanums=0    # 1 2 3

## algo selection
# difftype="SRPO"     # DQL
marltype="Seq"      # default Seq, Can train JAL, IND, CTDE
TASK="HalfCheetah-v2"
device=0

agent_idx=0 # 用于指定train 0 还是 1
# if train on mixed data: --mixed_data

for types in "${datatypes[@]}"
do
    echo "data types: $types, data number: $datanums, mix datasets: True"
    python pretrain_behavior.py --data_type $types --device $device --srpo_mode $marltype --seq_agent_id $agent_idx --mixed_data &
done