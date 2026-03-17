from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

api.upload_folder(
    folder_path="predictive-maintenance/deployment",         # the local folder containing your files
    repo_id="ksricheenu/Predictive-Maintenance-Prediction",  # the target repo space in hugging face
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
