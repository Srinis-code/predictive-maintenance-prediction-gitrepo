#The below Libraries are been imporated, for bringing the necessary tools to work with Hugging Face Hub.
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

#Setting the Repository to define where the dataset should be stored.
repo_id = "ksricheenu/Predictive-Maintenance-Prediction"
repo_type = "dataset"

#Initialize Hugging Face API client
api = HfApi(token=os.getenv("HF_TOKEN"))

#Check if the dataset repo exists:
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

#Uploading the entire folder to the Hugging Face dataset space
api.upload_folder(
    folder_path="predictive-maintenance/data",
    repo_id=repo_id,
    repo_type=repo_type,
)

print("Dataset upload completed successfully.")
