"""Upload the static project page to its Hugging Face Space (free static hosting).

    python space/build_page.py && python space/deploy.py      # needs `hf auth login` (write token)

live.json in the Space is written by `python app.py --share` and is left untouched here.
"""
from huggingface_hub import HfApi

REPO = "Arcyadav/steering-transformers"

api = HfApi()
api.create_repo(REPO, repo_type="space", space_sdk="static", exist_ok=True)
api.upload_folder(repo_id=REPO, repo_type="space", folder_path="space/site", commit_message="Update project page",
                  delete_patterns=["style.css"])
print(f"https://huggingface.co/spaces/{REPO}  ->  https://{REPO.lower().replace('/', '-')}.static.hf.space")
