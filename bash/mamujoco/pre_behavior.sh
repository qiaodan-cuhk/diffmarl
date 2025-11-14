#!/bin/sh
# 在用seq去pretrain的时候，直接train的是agent1的条件分布，还需要用ind去train得到agent0的分布
datatypes=("Medium" "Poor")  # Good Medium Poor


## algo selection
marltype="Seq"      # default Seq, Can train JAL, IND, CTDE
TASK="2ant"  # 2ant, 4ant, 2halfcheetah
device=3
# agent_idx=0 # 用于指定train 0 还是 1

device_idx=0
for agent_idx in 0 1
do
    for types in "${datatypes[@]}"
    do
        device=$((device_idx % 4))  # 循环使用 device 0-3
        echo "data types: $types, training agents id: $agent_idx"
        python /home/qiaodan/code/diffmarl/pretrain_behavior.py --data_type $types --device $device --save_interval 10 --srpo_mode $marltype --seq_agent_id $agent_idx &
        device_idx=$((device_idx + 1))
    done
done