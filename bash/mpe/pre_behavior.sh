#!/bin/sh
# 在用seq去pretrain的时候，直接train的是agent1的条件分布，还需要用ind去train得到agent0的分布
# simple spread 数据集用 0 0 0 2
# simple tag 4 0 4 0
# simple world 1 0 1 2

TASK="simple_world" 
devices=(1 2)
declare -A dataset_seeds=(
    ["expert"]=1
    ["medium"]=0
    ["medium-replay"]=1
    ["random"]=2
)

marltype="Seq"      # default Seq, Can train JAL, IND, CTDE
datatypes=("expert" "medium" "medium-replay" "random")  # "medium" "random"  先跑expert和replay
mix=False

# 计数器，用于轮流分配 GPU
counter=0

for types in "${datatypes[@]}"
do
    for agent_idx in {0..2}
    do
        device=${devices[$((counter % 2))]}
        seed=${dataset_seeds[$types]}
        echo "data types: $types, data number: $seed, mix datasets: $mix, training agents id: $agent_idx"

        python /data/qiaodan/code/diffmarl/pretrain_behavior.py \
            --env_id $TASK \
            --data_type $types \
            --dataset_num $seed \
            --device $device \
            --srpo_mode $marltype \
            --seq_agent_id $agent_idx &

        counter=$((counter + 1))
        done
done

wait