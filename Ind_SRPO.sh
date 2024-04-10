#!/bin/sh

seeds=( 1 27 42 100 1000 )
datatypes=( "expert" "medium" "medium-replay" "random" )
datanums=( 0 1 2 3 )


## algo seletion
diff="SRPO"     # DQL
marl="IND"      # JAL, VD, SEQ
TASK="simple_spread"

# actor/critic load path: None
for i in "${seeds[@]}"
do
    for types in "${datatypes[@]}"
    do
        for num in "${datanums[@]}"
        do
            echo "seed: $i, datatypes: $types, data number: $num"
            python main.py --env_id $TASK --data_type $types --dataset_num $num --seed $i --device 0 --difftype $diff --marltype $marl
        done
    done
done


# SRPO instruction
# TASK="walker2d-medium-replay-v2"
# python3 -u train_policy.py --expid ${TASK}-baseline-seed${seed} --env $TASK --seed ${seed} --actor_load_path ./SRPO_model_factory/${TASK}-baseline-seed${seed}/behavior_ckpt200.pth --critic_load_path ./SRPO_model_factory/${TASK}-baseline-seed${seed}/critic_ckpt150.pth