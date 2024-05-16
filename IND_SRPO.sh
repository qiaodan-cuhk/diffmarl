#!/bin/sh
seeds=(100)
datatypes=("random")  # "expert" "random" "medium" "medium-replay"
datanums=0    # 1 2 3
beta=0.02

## algo seletion
difftype="SRPO"     # DQL
marltype="IND"      # JAL, VD, SEQ
TASK="HalfCheetah-v2"
device=1


for i in "${seeds[@]}"
do
    for types in "${datatypes[@]}"
    do
        # actor/critic load path
        critic_load_path="/home/qiaodan/Code/diffmarl/SRPO_premodels/HalfCheetah-v2_${types}"
        diffusion_load_path="/home/qiaodan/Code/diffmarl/SRPO_premodels/HalfCheetah-v2_${types}"

        for num in "${datanums[@]}"
        do
            echo "seed: $i, datatypes: $types, data number: $num"
            python main.py --env_id $TASK --data_type $types --dataset_num $num --seed $i --device $device --difftype $difftype --marltype $marltype --critic_load_path $critic_load_path --diffusion_load_path $diffusion_load_path --beta $beta
        done
    done
done


# python main.py --env_id HalfCheetah-v2 --data_type random --difftype SRPO --marltype IND --critic_load_path /home/qiaodan/Code/diffmarl/SRPO_premodels/HalfCheetah-v2_random --diffusion_load_path /home/qiaodan/Code/diffmarl/SRPO_premodels/HalfCheetah-v2_random --beta 0.02 --seed 100 --device 1


# SRPO instruction
# TASK="walker2d-medium-replay-v2"
# python3 -u train_policy.py --expid ${TASK}-baseline-seed${seed} --env $TASK --seed ${seed} --actor_load_path ./SRPO_model_factory/${TASK}-baseline-seed${seed}/behavior_ckpt200.pth --critic_load_path ./SRPO_model_factory/${TASK}-baseline-seed${seed}/critic_ckpt150.pth