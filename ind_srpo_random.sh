#!/bin/sh   #  100
#专门用于random dataset找ind参数
seeds=(44)
types="random"  # "expert" "random" "medium-replay"
datanums=0    # 1 2 3

## algo seletion
difftype="SRPO"     # DQL
marltype="IND"      # JAL, VD, SEQ
TASK="HalfCheetah-v2"
device=0
beta=(0.15 0.17 0.19 0.21 0.23 0.25)
# beta=(0.0005 0.07 0.15 0.3 0.7 2 3 10)
# beta=(0.002)
# random需要0.2附近

diffusion_epoch_num=149
critic_epoch_num=119


trainsteps=100000

for i in "${seeds[@]}"
do
    for beta_i in "${beta[@]}"
    do
        # actor/critic load path
        critic_load_path="/home/qiaodan/Code/diffmarl/SRPO_premodels/HalfCheetah-v2"
        diffusion_load_path="/home/qiaodan/Code/diffmarl/SRPO_premodels/HalfCheetah-v2"


        # 打印路径以进行调试
        echo "Critic Load Path: $critic_load_path"
        echo "Diffusion Load Path: $diffusion_load_path"

        echo "seed: $i, datatypes: $types, data number: $num"
        python main.py --env_id $TASK --data_type $types --dataset_num $datanums --seed $i --device $device --difftype $difftype --marltype $marltype --critic_load_path $critic_load_path --diffusion_load_path $diffusion_load_path --diff_epoch $diffusion_epoch_num --critic_epoch $critic_epoch_num --num_steps $trainsteps --beta $beta_i &


        # for num in "${datanums[@]}"
        # do
            # echo "seed: $i, datatypes: $types, data number: $num"
            # python main.py --env_id $TASK --data_type $types --dataset_num $num --seed $i --device $device --difftype $difftype --marltype $marltype --critic_load_path $critic_load_path --diffusion_load_path $diffusion_load_path --beta $beta &
        # done
    done
done

# 等待所有后台任务完成
wait

# SRPO instruction
# TASK="walker2d-medium-replay-v2"
# python3 -u train_policy.py --expid ${TASK}-baseline-seed${seed} --env $TASK --seed ${seed} --actor_load_path ./SRPO_model_factory/${TASK}-baseline-seed${seed}/behavior_ckpt200.pth --critic_load_path ./SRPO_model_factory/${TASK}-baseline-seed${seed}/critic_ckpt150.pth