#!/bin/sh
# simple spread 数据集用 0 0 0 2
# simple tag 4 0 4 0
# simple world 1 0 1 2

TASK="simple_spread"
declare -A dataset_seeds=(
    ["expert"]=0
    ["medium"]=0
    ["medium-replay"]=0
    ["random"]=0
)

seeds=(100)  # 改为数组形式
# seeds=(42 99 1000 9999 23477)

device=(4 5)
beta=(0.01 0.02 0.05 0.1 0.2 0.5)
datatypes=("expert" "medium" "random")  # "medium-replay" simple spread 爆炸critic

diffusion_epoch_num=149
critic_epoch_num=79

difftype="SRPO"
marltype="SEQ"
counter=0
trainsteps=1000000

# beta=(0.001 0.005 0.01 0.02 0.05 0.1 0.2 0.5)  # medium-replay 在0.02附近表现较好
# beta=(0.01 0.015 0.02 0.025 0.03 0.04)
# 0.05 - 0.25附近 for random
# beta=(0.02)
# beta=(0.0001 0.001 0.01 0.02 0.05 0.1 0.2 0.5 1)

for types in "${datatypes[@]}"
do
    for i in "${seeds[@]}"
    do
        for beta_i in "${beta[@]}"
        do
            dataset_num=${dataset_seeds[$types]}
            current_device=${device[$((counter % ${#device[@]}))]}
            
            critic_load_path="/data/qiaodan/code/diffmarl/SRPO_premodels/$TASK"
            diffusion_load_path="/data/qiaodan/code/diffmarl/SRPO_premodels/$TASK"

            echo "Critic Load Path: $critic_load_path"
            echo "Diffusion Load Path: $diffusion_load_path"
            echo "seed: $i, datatypes: $types, data number: $dataset_num, beta: $beta_i, device: $current_device"
            
            python main.py --env_id $TASK \
                         --data_type $types \
                         --dataset_num $dataset_num \
                         --seed $i \
                         --device $current_device \
                         --difftype $difftype \
                         --marltype $marltype \
                         --critic_load_path $critic_load_path \
                         --diffusion_load_path $diffusion_load_path \
                         --diff_epoch $diffusion_epoch_num \
                         --critic_epoch $critic_epoch_num \
                         --num_steps $trainsteps \
                         --beta $beta_i &

            counter=$((counter + 1))  # 更新counter
        done
    done
done

wait
