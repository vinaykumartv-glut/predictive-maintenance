import os
import glob
from huggingface_hub import HfApi

from google.colab import userdata
access_key = userdata.get("MY_API_KEY")
#access_key = os.getenv("HF_TOKEN")

# Load token from environment
api = HfApi(token = access_key)

repo_id = "vinaykumartv/predictive-maintenance"
repo_type = "dataset"

# Ensure the repo exists
api.create_repo(repo_id = repo_id, repo_type = repo_type, exist_ok = True)

# Upload each file in the data folder individually
for file_path in glob.glob("predictive_maintenance/data/*"):
    print(f"Uploading {file_path}...")
    filename = os.path.basename(file_path)
    print(f"Uploading {filename}...")
    api.upload_file(
        path_or_fileobj = file_path,
        path_in_repo = filename,
        repo_id = repo_id,
        repo_type = repo_type,
        commit_message = f"Upload {filename}"
    )

print(f"✅ All files uploaded successfully to {repo_id} at {filename}")
