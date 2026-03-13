from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

# Build deployment artifact
build_dir = "space_build"

if os.path.exists(build_dir):
    shutil.rmtree(build_dir)

os.makedirs(build_dir)

# copy streamlit app
shutil.copy(
    "predictive-maintenance/deployment/app.py",
    f"{build_dir}/streamlit_app.py"
)

# copy requirements
shutil.copy(
    "predictive-maintenance/deployment/requirements.txt",
    f"{build_dir}/requirements.txt"
)

# copy optional dockerfile if present
dockerfile_path = "predictive-maintenance/deployment/Dockerfile"
if os.path.exists(dockerfile_path):
    shutil.copy(dockerfile_path, f"{build_dir}/Dockerfile")


# Upload to HuggingFace Space
api.upload_folder(
    folder_path="predictive-maintenance/hosting",        # the local folder containing your files
    repo_id="ksricheenu/Predictive-Maintenance-Prediction", # the target repo space in huggingface
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
