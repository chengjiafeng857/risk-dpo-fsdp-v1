from huggingface_hub import HfApi

repo_id = "jackf857/llama32-8b-dpo-hh-beta0.5"
local_dir = "dpo_model"

api = HfApi()

api.create_repo(
    repo_id=repo_id,
    private=False,
    exist_ok=True,
)

api.upload_folder(
    folder_path=local_dir,
    repo_id=repo_id,
    repo_type="model",
    commit_message="DPO fine-tuned LLaMA-3 8B (HH, seed=42)"
)

print("Uploaded model to HF:", repo_id)