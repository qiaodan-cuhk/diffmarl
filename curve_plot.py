
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
def smooth_and_plot(mean_values, std_values, label_prefix):
    window_size = 5  # 设置窗口大小
    mean_values_smooth = np.convolve(mean_values, np.ones(window_size)/window_size, mode='valid')
    std_values_smooth = np.convolve(std_values, np.ones(window_size)/window_size, mode='valid')

    # 绘制带阴影带的图
    plt.plot(range(len(mean_values_smooth)), mean_values_smooth, label=f'{label_prefix}')
    plt.fill_between(range(len(std_values_smooth)), 
                     mean_values_smooth - std_values_smooth, 
                     mean_values_smooth + std_values_smooth, 
                     alpha=0.2)


def meta(dtset):
    log_pattern = log_pattern_meta + dtset  # 根据你的路径调整

    plt.figure(figsize=(9, 4), dpi=300)

    for keywords in keywords_list:
        all_steps, all_values = read_and_process_data(keywords, tag, log_pattern)
        mean_values, std_values = compute_mean_std(all_steps, all_values)
        smooth_and_plot(mean_values, std_values, keywords[0])  # 使用第一个关键词作为标签前缀

    # 添加图例和标题
    plt.xlabel('Step')
    plt.ylabel('Evaluation Returns')
    plt.title('Learning Curves of MAMujoco {}'.format(dtset))
    plt.legend()
    plt.savefig('Ablation_1_{}.png'.format(dtset))  # 替换为你想要的文件名
    plt.show()


dtset_list = ['expert', 'mid', 'mid_rep']

# , 'mid_rep', 'random'


keywords_list = [
    ['CTDE'],
    ['JAL'],
    ['IND'],
    ['SEQ']
]

tag = 'eval_return'  # 替换为你想要读取的标签
log_pattern_meta = "/home/qiaodan/Code/diffmarl/results/plot/Abla1/"

for dt in dtset_list:
    meta(dt)