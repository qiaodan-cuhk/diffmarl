#!/bin/sh
seeds=(0 37 42 99 400 1000  32987 50000 77777  9000000)
datatypes=("random")  # "expert" "random" "medium" "medium-replay"
datanums=0    # 1 2 3
beta=0  # (0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)


## algo seletion
difftype="SRPO"     # DQL
marltype=("IND" "CTDE" "SEQ")      # JAL, VD, SEQ
TASK="bandit"
device=0


for beta_i in "${beta[@]}"
do
    for i in "${seeds[@]}"
    do
        # actor/critic load path
        critic_load_path="/data/qiaodan/code/diffmarl/SRPO_premodels/bandit"
        diffusion_load_path="/data/qiaodan/code/diffmarl/SRPO_premodels/bandit"

        for num in "${datanums[@]}"
        do
            echo "seed: $i, datatypes: $types, data number: $num"
            python ablation/abl_bandit.py --env_id $TASK --marltype $marltype --seed $i --device $device --critic_load_path $critic_load_path --diffusion_load_path $diffusion_load_path --beta $beta_i
        done
    done
done


