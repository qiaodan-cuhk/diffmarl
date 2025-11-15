#!/bin/sh

# expert 用 datanum 3最好，medium+replay用datanum 1最好 random用的是0

datatypes=("Good" "Medium" "Poor")  # Good Medium Poor
marltype="JAL"      # JAL, IND
TASK="4ant"  # 2ant, 4ant, 2halfcheetah

tau_list=(0.5 0.7)
temp_list=(3.0 5.0)

device_idx=0
eval_interval=5
save_interval=10
# if train on mixed data: --mixed_data

for types in "${datatypes[@]}"
do
    for tau in "${tau_list[@]}"
    do
        for temp in "${temp_list[@]}"
        do
            device=$((device_idx % 4))  # 循环使用 device 0-3
            echo "tau: $tau, temp: $temp"
            python /home/qiaodan/code/diffmarl/pretrain_critic.py --env_id $TASK --data_type $types --device $device --save_interval $save_interval --eval_interval $eval_interval --tau $tau --temp $temp --srpo_mode $marltype &
            device_idx=$((device_idx + 1))
        done
    done
done

wait  # 等待所有后台任务完成
echo "All training jobs completed!"