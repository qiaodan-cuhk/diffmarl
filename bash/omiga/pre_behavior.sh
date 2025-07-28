#!/bin/sh

# 在用seq去pretrain的时候，直接train的是agent1的条件分布，还需要用ind去train得到agent0的分布
datatypes=("expert")  # "expert" "medium" "medium-replay" "random"

## algo selection
# difftype="SRPO"     # DQL
marltype="Seq"      # default Seq, Can train JAL, IND, CTDE
TASK="HalfCheetah-v2"
devices=(1 2 3 4)  # 可用的GPU设备列表
agent_idx=(0 1 2 3 4 5) # 用于指定train agent id

# 初始化计数器
counter=0
total_devices=${#devices[@]}

for types in "${datatypes[@]}"
do
    for agent in "${agent_idx[@]}"
    do
        # 计算当前应该使用的设备索引
        device_index=$((counter % total_devices))
        current_device=${devices[$device_index]}
        
        echo "data types: $types, device: $current_device, training agent id: $agent"
        python /data/qiaodan/code/diffmarl/pretrain_behavior.py --env_id $TASK --data_type $types --device $current_device --srpo_mode $marltype --seq_agent_id $agent &
        
        # 增加计数器
        counter=$((counter + 1))
    done
done

# 等待所有后台进程完成
wait
echo "所有训练任务完成！"