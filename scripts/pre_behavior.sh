#!/bin/sh

datatypes=("expert" "medium" "medium-replay" "medium-expert")  # "expert" "medium" "medium-replay" "medium-expert"

## algo selection
marltype="Seq"      # default Seq (sequential score function), can train JAL, IND for ablation
TASK="Hopper-v2"
devices=(2 3 4 5)  # devices id
agent_idx=(0 1 2) # trained agent id numbers, used for parallel training 


counter=0
total_devices=${#devices[@]}

for types in "${datatypes[@]}"
do
    for agent in "${agent_idx[@]}"
    do
        device_index=$((counter % total_devices))
        current_device=${devices[$device_index]}
        
        echo "data types: $types, device: $current_device, training agent id: $agent"
        python /data/qiaodan/code/diffmarl/pretrain_behavior.py --env_id $TASK --data_type $types --device $current_device --srpo_mode $marltype --seq_agent_id $agent &
        
        counter=$((counter + 1))
    done
done

wait
echo "All training done!"