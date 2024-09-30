#!/bin/sh
# seeds=(42 99 1000 9999 23477)   #  100

seeds=(44)
datatypes=("expert")  # "expert" "random" "medium-replay"
datanums=0    # 1 2 3

## algo seletion
difftype="SRPO"     # DQL
marltype="SEQ"      # JAL, VD, SEQ
TASK="HalfCheetah-v2"
device=1
# beta=(0.0005 0.001 0.0015 0.002 0.01 0.02 0.05 0.1 0.2 0.5 1)
beta=(0.002)


for i in "${seeds[@]}"
do
    for types in "${datatypes[@]}"
    do
        # actor/critic load path
        critic_load_path="/home/qiaodan/Code/diffmarl/SRPO_premodels/HalfCheetah-v2"
        diffusion_load_path="/home/qiaodan/Code/diffmarl/SRPO_premodels/HalfCheetah-v2"


        # 打印路径以进行调试
        echo "Critic Load Path: $critic_load_path"
        echo "Diffusion Load Path: $diffusion_load_path"



        for num in "${datanums[@]}"
        do
            echo "seed: $i, datatypes: $types, data number: $num"
            python main.py --env_id $TASK --data_type $types --dataset_num $num --seed $i --device $device --difftype $difftype --marltype $marltype --critic_load_path $critic_load_path --diffusion_load_path $diffusion_load_path --beta $beta &
        done
    done
done

# 等待所有后台任务完成
wait

# SRPO instruction
# TASK="walker2d-medium-replay-v2"
# python3 -u train_policy.py --expid ${TASK}-baseline-seed${seed} --env $TASK --seed ${seed} --actor_load_path ./SRPO_model_factory/${TASK}-baseline-seed${seed}/behavior_ckpt200.pth --critic_load_path ./SRPO_model_factory/${TASK}-baseline-seed${seed}/critic_ckpt150.pth