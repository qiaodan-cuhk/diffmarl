
import glob
import os
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt


# 定义读取和处理数据的函数
def read_and_process_data(keywords, tag, log_pattern):
    all_steps = []
    all_values = []
    
    for log_dir in glob.glob(log_pattern+"/*"):
        
        if all(keyword in log_dir for keyword in keywords):
            event_files = glob.glob(os.path.join(log_dir, '*.ubun'))  # 查找以 .ubun 结尾的文件
            # print(f"Processing directory: {log_dir}, found files: {event_files}")  # 调试信息
            for event_file in event_files:
                ea = event_accumulator.EventAccumulator(event_file)
                ea.Reload()  # 加载事件数据

                tags = ea.Tags()
                print(f"Available tags in {event_file}: {tags['scalars']}")  # 打印所有标量标签
                

                # 检查标签是否存在
                if tag not in tags['scalars']:
                    print(f"Warning: Tag '{tag}' not found in file: {event_file}. Skipping this file.")
                    continue


                data = ea.Scalars(tag)
                if not data:
                    print(f"No data found for tag '{tag}' in file: {event_file}")  # 调试信息
                    continue

                steps = [event.step for event in data]
                values = [event.value for event in data]

                if steps and values:  # 确保步骤和值不为空
                    all_steps.append(steps)
                    all_values.append(values)

            
    return all_steps, all_values

# 定义计算均值和方差的函数
def compute_mean_std(all_steps, all_values):
    max_steps = max(len(s) for s in all_steps)
    mean_values = np.zeros(max_steps)
    std_values = np.zeros(max_steps)

    for i in range(max_steps):
        values_at_step = [values[i] for values in all_values if i < len(values)]
        mean_values[i] = np.mean(values_at_step)
        std_values[i] = np.std(values_at_step)

    return mean_values, std_values

# 定义平滑和绘图的函数
def smooth_and_plot(mean_values, std_values, label_prefix, color):
    window_size = 5  # 设置窗口大小
    mean_values_smooth = np.convolve(mean_values, np.ones(window_size)/window_size, mode='valid')
    std_values_smooth = np.convolve(std_values, np.ones(window_size)/window_size, mode='valid')

    # 绘制带阴影带的图
    steps = [5000 * i for i in range(len(mean_values_smooth))]
    plt.plot(steps, mean_values_smooth, label=f'{label_prefix}', color=color)
    plt.fill_between(steps, 
                     mean_values_smooth - std_values_smooth, 
                     mean_values_smooth + std_values_smooth, 
                     color=color,
                     alpha=0.2)



dtset_list = ['expert', 'mid', 'mid_rep']

# 'random'

keywords_list = ['SEQ', 'IND', 'CTDE']

tag = 'eval_return'  # 替换为你想要读取的标签
log_pattern_meta = "/home/qiaodan/Code/diffmarl/results/plot/Abla1/"


def meta(dtset):
    log_pattern = log_pattern_meta + dtset  # 根据你的路径调整

    plt.figure(figsize=(9, 4), dpi=200)

    

    for keywords in keywords_list:
        all_steps, all_values = read_and_process_data(keywords, tag, log_pattern)
        mean_values, std_values = compute_mean_std(all_steps, all_values)

        # 初始化颜色和标签
        color_curr = 'black'  # 默认颜色
        label_curr = 'Unknown'  # 默认标签
        
        if keywords == "SEQ":
            color_curr = 'red'
            label_curr = 'OMSD (Ours)'
        elif keywords == "IND":
            color_curr = 'blue'
            label_curr = 'BRPO-IND'
        elif keywords == "CTDE":
            color_curr = 'green'
            label_curr = 'BRPO-FAC'

        smooth_and_plot(mean_values, std_values, label_curr, color_curr)  # 使用第一个关键词作为标签前缀

    if dtset == 'expert':
        name = 'Expert'
        absolute_reward = 3338.68
    elif dtset == 'mid_rep':
        name = 'Medium Replay'
        absolute_reward = 1568.86
    elif dtset == 'mid':
        name = 'Medium'
        absolute_reward = 423.48
    elif dtset == 'random':
        name = 'Random'
        absolute_reward = -282.89

    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)  
    plt.axhline(y=absolute_reward, color='black', linestyle='-.', linewidth=1)  #  label='Absolute Average Reward in Dataset'

    # 添加图例和标题
    plt.xlabel('Training Steps', fontsize=10)
    plt.ylabel('Evaluation Returns', fontsize=10)
    plt.title('Learning Curves of MAMujoco - {}'.format(name), fontsize=13)
    plt.legend(loc='lower right')
    plt.xlim(0, 1000000) 
    plt.ylim(bottom=0)  

    save_dir = '/home/qiaodan/Code/diffmarl/results/plot/Abla1'

    # 创建目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'Ablation_1_{}.png'.format(dtset)), dpi=200, bbox_inches='tight')
    plt.show()


for dt in dtset_list:
    meta(dt)