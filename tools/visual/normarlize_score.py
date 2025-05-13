


mpe_score = {"CN": [516.8, 159.8],
             "PP": [185.6, -4.1],
             "world": [79.5, -6.8]}

omar_score = {"2half": [3568.81, -283.97]}

ogmarl_score = {"2half": [0, 0],
                "2ant": [0, 0]}


# mean, var
dom2_cn = {"expert": [628.6, 17.2],
           "medium": [358.9, 25.2],
           "replay": [324.1, 38.6],
           "random": [337.8, 26.0]}

dom2_pp = {"expert": [259.1, 22.8],
           "medium": [155.8, 48.1],
           "replay": [150.5, 23.9],
           "random": [208.7, 57.3]}

dom2_world = {"expert": [99.5, 17.1],
              "medium": [84.5, 23.4],
              "replay": [65.9, 10.6],
              "random": [40.0, 14.3]}


def normalize_scores(score_list, expert_random_score):
    """
    Normalize scores using (score-random)/(expert-random) formula
    Args:
        score: [mean, var] from dom2
        expert_random_score: [expert_score, random_score] from mpe_score
    """
    expert_score, random_score = expert_random_score

    mean = score_list[0]
    var = score_list[1]
    max = mean+var
    min = mean-var

    normalized_max = (max - random_score) / (expert_score - random_score)
    normalized_min = (min - random_score) / (expert_score - random_score)

    norm_mean = (normalized_max+normalized_min)/2
    norm_var = norm_mean - normalized_min
    
    return [norm_mean*100, norm_var*100]

# CN环境的归一化
cn_normalized = {
    "expert": normalize_scores(dom2_cn["expert"], mpe_score["CN"]),
    "medium": normalize_scores(dom2_cn["medium"], mpe_score["CN"]),
    "replay": normalize_scores(dom2_cn["replay"], mpe_score["CN"]),
    "random": normalize_scores(dom2_cn["random"], mpe_score["CN"])
}

# PP环境的归一化
pp_normalized = {
    "expert": normalize_scores(dom2_pp["expert"], mpe_score["PP"]),
    "medium": normalize_scores(dom2_pp["medium"], mpe_score["PP"]),
    "replay": normalize_scores(dom2_pp["replay"], mpe_score["PP"]),
    "random": normalize_scores(dom2_pp["random"], mpe_score["PP"])
}

# World环境的归一化
world_normalized = {
    "expert": normalize_scores(dom2_world["expert"], mpe_score["world"]),
    "medium": normalize_scores(dom2_world["medium"], mpe_score["world"]),
    "replay": normalize_scores(dom2_world["replay"], mpe_score["world"]),
    "random": normalize_scores(dom2_world["random"], mpe_score["world"])
}

# 打印标准化后的结果
print("CN normalized scores:")
for k, v in cn_normalized.items():
    print(f"{k}: mean={v[0]:.3f}, var={v[1]:.3f}")

print("\nPP normalized scores:")
for k, v in pp_normalized.items():
    print(f"{k}: mean={v[0]:.3f}, var={v[1]:.3f}")

print("\nWorld normalized scores:")
for k, v in world_normalized.items():
    print(f"{k}: mean={v[0]:.3f}, var={v[1]:.3f}")



"""
DOM2

N normalized scores:
expert: mean=131.317, var=4.818
medium: mean=55.770, var=7.059
replay: mean=46.022, var=10.812
random: mean=49.860, var=7.283

PP normalized scores:
expert: mean=138.745, var=12.019
medium: mean=84.291, var=25.356
replay: mean=81.497, var=12.599
random: mean=112.177, var=30.206

World normalized scores:
expert: mean=123.175, var=19.815
medium: mean=105.794, var=27.115
replay: mean=84.241, var=12.283
random: mean=54.229, var=16.570
"""


