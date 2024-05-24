#!/bin/sh
seeds=(0 37 42 99 400 1000  32987 50000 77777  9000000)
datatypes=("random")  # "expert" "random" "medium" "medium-replay"
datanums=0    # 1 2 3
beta=0  # (0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

## algo seletion
difftype="SRPO"     # DQL
marltype="CTDE"      # JAL, VD, SEQ
TASK="bandit"
device=0

for beta_i in "${beta[@]}"
do
    for i in "${seeds[@]}"
    do
        # actor/critic load path
        critic_load_path="/home/qiaodan/Code/diffmarl/SRPO_premodels/bandit"
        diffusion_load_path="/home/qiaodan/Code/diffmarl/SRPO_premodels/bandit"

        for num in "${datanums[@]}"
        do
            echo "seed: $i, datatypes: $types, data number: $num"
            python check.py --env_id $TASK --marltype $marltype --seed $i --device $device --critic_load_path $critic_load_path --diffusion_load_path $diffusion_load_path --beta $beta_i
        done
    done
done


# python main.py --env_id HalfCheetah-v2 --data_type random --difftype SRPO --marltype IND --critic_load_path /home/qiaodan/Code/diffmarl/SRPO_premodels/HalfCheetah-v2_random --diffusion_load_path /home/qiaodan/Code/diffmarl/SRPO_premodels/HalfCheetah-v2_random --beta 0.02 --seed 100 --device 1


# SRPO instruction
# TASK="walker2d-medium-replay-v2"
# python3 -u train_policy.py --expid ${TASK}-baseline-seed${seed} --env $TASK --seed ${seed} --actor_load_path ./SRPO_model_factory/${TASK}-baseline-seed${seed}/behavior_ckpt200.pth --critic_load_path ./SRPO_model_factory/${TASK}-baseline-seed${seed}/critic_ckpt150.pth