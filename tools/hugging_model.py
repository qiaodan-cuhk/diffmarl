from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="/data/qiaodan/code/diffmarl/SRPO_premodels",
    repo_id="danqiao-cuhk/omsd",
    repo_type="model",
)
