# 定义不同数据类型及其对应的 dataset_seed
# simple spread 数据集用 0 0 0 2
# simple tag 4 0 4 0
# simple world 1 0 1 2

TASK="simple_world"   # simple xx
device=(4 5)
declare -A dataset_seeds=(
    ["expert"]=1
    ["medium"]=0
    ["medium-replay"]=1
    ["random"]=2
)

marltype="JAL"      # Can train JAL, IND    seq使用 jal critic
datatypes=("expert" "medium" "medium-replay" "random")
mix=False
# if train on mixed data: --mixed_data

# 计数器，用于轮流分配 GPU
counter=0

for types in "${datatypes[@]}"
do
    # 使用取模运算来选择GPU
    current_device=${device[$((counter % ${#device[@]}))]}
    
    seed=${dataset_seeds[$types]}
    echo "data types: $types, device: $current_device, data number: $seed, mix datasets: $mix"

    python /data/qiaodan/code/diffmarl/pretrain_critic.py \
        --env_id $TASK \
        --data_type $types \
        --dataset_num $seed \
        --device $current_device \
        --srpo_mode $marltype &
        
    # 增加计数器
    ((counter++))
done

wait