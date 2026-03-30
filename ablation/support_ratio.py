# 用于加载ckpt，rollout数据，并与原始dataset算相似度和support ratio
# 评估 OMSD，BRPO-CTDE，和 MADiff 之间的差异，训练 MPE simple tag 上 expert medium random 就行了 

# 加载实际的训练datasets，加载保存的rollouts，然后构建KD-tree进行计算
