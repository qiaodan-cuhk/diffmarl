#!/bin/sh
# simple spread 数据集用 0 0 0 2
# simple tag 4 0 4 0
# simple world 1 0 1 2

# 手动指定所有参数
TASK="simple_spread"      # 手动指定任务名称
dataset_num=0             # 手动指定 dataset seed
beta_value=0.01          # 手动指定 beta 值
datatype="medium"        # 手动指定 datatype
save_interval=10000      # 手动指定保存间隔

seeds=(999)  # 改为数组形式
device=(1 2 3 4 5)

diffusion_epoch_num=(199)           # 149
critic_epoch_num=(499)    # 99

difftype="SRPO"
marltype="SEQ"
counter=0
trainsteps=100000
annealing_epochs=10

for i in "${seeds[@]}"
do
    for critic_i in "${critic_epoch_num[@]}"
    do
        for diff_i in "${diffusion_epoch_num[@]}"
        do
            current_device=${device[$((counter % ${#device[@]}))]}

            echo "task: $TASK, seed: $i, data type: $datatype, dataset_num: $dataset_num"
            echo "beta: $beta_value, device: $current_device, critic: $critic_i, diff: $diff_i"
            echo "save_interval: $save_interval"
            
            python main.py --env_id $TASK \
                        --data_type $datatype \
                        --dataset_num $dataset_num \
                        --seed $i \
                        --device $current_device \
                        --difftype $difftype \
                        --marltype $marltype \
                        --num_steps $trainsteps \
                        --beta $beta_value \
                        --n_policy_epochs $annealing_epochs \
                        --diff_epoch $diff_i \
                        --critic_epoch $critic_i \
                        --save_interval $save_interval &

            counter=$((counter + 1))  # 更新counter
        done
    done
done

wait