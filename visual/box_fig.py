import numpy as np
import matplotlib.pyplot as plt
import os

def convert_max_min_to_mean_var(max_vals, min_vals):
    """
    将max和min转换为mean和variance
    mean = (max + min) / 2
    var = (max - min) / 2
    """
    means = (max_vals + min_vals) / 2
    variances = (max_vals - min_vals) / 2
    return means, variances

# 数据
algorithms = ['IQL', 'BRPO-IND', 'BRPO-IGO', 'OMSD']



# simple_spread
# max_expert = np.array([526.7, 527, 526, 532.1])  # 请替换为实际数据
# min_expert = np.array([524.3, 525, 524, 528.1])  # 请替换为实际数据

# max_medium = np.array([305.1, 366, 352.8, 415])  # 请替换为实际数据
# min_medium = np.array([301.4, 342, 330.8, 411])  # 请替换为实际数据

# max_random = np.array([240, 311, 344, 404])  # 请替换为实际数据
# min_random = np.array([234, 289, 301, 402])  # 请替换为实际数据


# simple_tag




# simple_world
# Expert数据
max_expert = np.array([120, 125, 132, 142])  # 请替换为实际数据
min_expert = np.array([114, 120, 122, 130])  # 请替换为实际数据
# Medium数据
max_medium = np.array([95.1, 100, 113, 135])  # 请替换为实际数据
min_medium = np.array([91.4, 92, 91, 130])  # 请替换为实际数据
# Random数据
max_random = np.array([42, 55, 59, 125])  # 请替换为实际数据
min_random = np.array([37, 42, 35, 110])  # 请替换为实际数据

# 计算均值和误差
rewards_expert, errors_expert = convert_max_min_to_mean_var(max_expert, min_expert)
rewards_medium, errors_medium = convert_max_min_to_mean_var(max_medium, min_medium)
rewards_random, errors_random = convert_max_min_to_mean_var(max_random, min_random)

# 计算每组的平均质量
expert_mean = np.mean(rewards_expert)
medium_mean = np.mean(rewards_medium)
random_mean = np.mean(rewards_random)
dataset_means = [expert_mean, medium_mean, random_mean]

# 设置图形大小和样式
plt.figure(figsize=(15, 8))
plt.style.use('default')

# 设置柱状图的位置
x = np.arange(3)  # 3组：expert, medium, random
width = 0.2  # 稍微调窄一点，因为现在是4个柱子

# 使用蓝色渐变色
colors = ['#08519c', '#3182bd', '#6baed6', '#bdd7e7']  # 从深蓝到浅蓝

# 绘制四组算法的柱状图
groups = ['Expert', 'Medium', 'Random']
rewards_data = [rewards_expert, rewards_medium, rewards_random]
errors_data = [errors_expert, errors_medium, errors_random]

# 为每个算法绘制柱状图
for i in range(len(algorithms)):
    # 收集当前算法在每个组的数据
    algorithm_rewards = [rewards[i] for rewards in rewards_data]
    algorithm_errors = [errors[i] for errors in errors_data]
    
    plt.bar(x + i*width, algorithm_rewards, width,
            yerr=algorithm_errors,
            color=colors[i],
            capsize=5,
            ecolor='black',
            label=algorithms[i])

# simple_spread
# dataset_means = [516.8, 246.7, 159.8]

dataset_means = [79.5, 24.7, -6.8]
# 为每组添加平均质量虚线
for i, mean_value in enumerate(dataset_means):
    # 计算每组的横线范围（覆盖该组所有柱状图的宽度）
    line_start = x[i] - width/2
    line_end = x[i] + width * 3.5  # 4个柱子的宽度
    plt.hlines(y=mean_value, xmin=line_start, xmax=line_end, 
              colors='#e377c2',  # 粉紫色
              linestyles='--',
              linewidth=2)  # 只在第一条线添加图例

# 设置x轴刻度和标签
plt.xticks(x + width * 1.5, groups, fontsize=24)
plt.yticks(fontsize=24)

# 添加标签和标题
plt.xlabel('Groups', fontsize=24)
plt.ylabel('Episode Reward', fontsize=24)
plt.title('World', fontsize=24)

# 添加图例
plt.legend(loc='upper right', fontsize=18)

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.3, axis='y')

# 调整布局
plt.tight_layout()

# 设置保存路径
exp_id = 'world'
plot_id = 'box_fig'
save_dir = os.path.join('/data/qiaodan/code/diffmarl/plots', plot_id)
save_path = os.path.join(save_dir, f'{exp_id}_{plot_id}.png')

# 保存图片
if save_path:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

plt.show()