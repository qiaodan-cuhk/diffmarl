#!/bin/sh
# simple spread 数据集用 0 0 0 2
# simple tag 4 0 4 0
# simple world 1 0 1 2

TASKS=("simple_spread" "simple_tag" "simple_world")

# 根据任务设置对应的dataset seed
declare -A dataset_seeds=(
    ["simple_spread"]="0"
    ["simple_tag"]="4"
    ["simple_world"]="1"
)

seeds=(42)  # 改为数组形式
# seeds=(42 99 1000)

device=(1 2 3 4 5)


# 只保留expert的beta值，且为simple spread设置固定值
declare -A type_betas=(
    ["simple_spread"]="0.001"
    ["simple_tag"]="0.005"
    ["simple_world"]="0.005"
)


datatypes=("expert")  

diffusion_epoch_num=(199)           # 149
critic_epoch_num=(199 249 299 349 399 449 499)    # 99

difftype="SRPO"
marltype="SEQ"
counter=0
trainsteps=100000
annealing_epochs=10

for TASK in "${TASKS[@]}"
do
    for i in "${seeds[@]}"
    do
        for critic_i in "${critic_epoch_num[@]}"
        do
            for diff_i in "${diffusion_epoch_num[@]}"
            do
                dataset_num=${dataset_seeds[$TASK]}
                current_device=${device[$((counter % ${#device[@]}))]}
                beta_i=${type_betas[$TASK]}

                echo "task: $TASK, seed: $i, data type: expert, dataset_num: $dataset_num"
                echo "beta: $beta_i, device: $current_device, critic: $critic_i, diff: $diff_i"
                
                python main.py --env_id $TASK \
                            --data_type "expert" \
                            --dataset_num $dataset_num \
                            --seed $i \
                            --device $current_device \
                            --difftype $difftype \
                            --marltype $marltype \
                            --num_steps $trainsteps \
                            --beta $beta_i \
                            --n_policy_epochs $annealing_epochs \
                            --diff_epoch $diff_i \
                            --critic_epoch $critic_i &

                counter=$((counter + 1))  # 更新counter
            done
        done
    done
done

wait
