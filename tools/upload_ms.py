from modelscope.hub.api import HubApi
# from modelscope.hub import push_to_hub


token="f4dd363b-6c40-4f22-8795-c67174297fae"
# 初始化API（需要先获取ModelScope的token）
api = HubApi()
api.login(access_token=token)  # 从modelscope网站获取token


ms_dir="qiaodan/Offline_MARL_OMAR_datasets"
# paths=["/data/qiaodan/code/diffmarl/datasets/bandit",
#          "/data/qiaodan/code/diffmarl/datasets/HalfCheetah-v2",
#          "/data/qiaodan/code/diffmarl/datasets/simple_spread",
#          "/data/qiaodan/code/diffmarl/datasets/simple_tag",
#          "/data/qiaodan/code/diffmarl/datasets/simple_world",
#          "/data/qiaodan/code/diffmarl/datasets/mix_hc",
#          "/data/qiaodan/code/diffmarl/datasets/madiff",
#          ]

paths=["/data/qiaodan/code/diffmarl/datasets/HalfCheetah-v2",
        #  "/data/qiaodan/code/diffmarl/datasets/simple_spread",
        #  "/data/qiaodan/code/diffmarl/datasets/simple_tag",
        #  "/data/qiaodan/code/diffmarl/datasets/simple_world",
        ]


# 上传数据集
for path in paths:
    api.upload_folder(
    repo_id=ms_dir,            # 例如：username/my_dataset
    folder_path=path,  # 本地数据集路径
    repo_type='dataset',                      # 指定为数据集类型
    commit_message='上传MPE和mamujoco 200数据集,来自OMAR论文'                # 提交信息
)