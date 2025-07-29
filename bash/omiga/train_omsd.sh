#!/bin/sh
# seeds=(42 99 1000 9999 23477)   #  100
seeds=100


## algo seletion
difftype="SRPO"     # DQL
marltype="SEQ"      
TASK="HalfCheetah-v2"
trainsteps=1000000


device=5
beta=(0.01 0.03 0.05 0.1 0.3)
# medium-expert 和  expert 训练完了
types=("expert") # "expert" "random" "medium-replay"
diffusion_epoch_num=159
critic_epoch_num=119



for i in "${seeds[@]}"
do
    for beta_i in "${beta[@]}"
    do
        # actor/critic load path
        pretrain_model_path="/data/qiaodan/code/diffmarl/pretrain/omiga/"
        # diffusion_load_path="/data/qiaodan/code/diffmarl/pretrain/omiga"


        # 打印路径以进行调试
        echo "Critic Load Path: $critic_load_path"
        echo "Diffusion Load Path: $diffusion_load_path"

        echo "seed: $i, datatypes: $types, beta: $beta_i"
        python /data/qiaodan/code/diffmarl/main.py --env_id $TASK --data_type $types --seed $i --device $device --difftype $difftype --marltype $marltype --pretrain_model_path $pretrain_model_path --diff_epoch $diffusion_epoch_num --critic_epoch $critic_epoch_num --num_steps $trainsteps --beta $beta_i &

    done
done

# 等待所有后台任务完成
wait
