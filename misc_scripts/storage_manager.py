import os
import shutil

SCRATCH = os.path.expanduser("/var/scratch/ave303")  # replace with your actual username

# Directories commonly used for model caching
CACHE_DIRS = [
    "hf_cache",
    "hf_home",
    "torch_cache",
    "triton_cache"
]

# Optional cleanup: delete papermill output notebooks
# DELETE_NOTEBOOKS = True

def delete_dir_if_exists(path):
    if os.path.exists(path):
        print(f"🧹 Deleting: {path}")
        shutil.rmtree(path)
    else:
        print(f"✅ Already clean: {path}")

def clear_caches():
    for subdir in CACHE_DIRS:
        full_path = os.path.join(SCRATCH, subdir)
        delete_dir_if_exists(full_path)

# def clean_notebooks():
#     for root, _, files in os.walk(SCRATCH):
#         for f in files:
#             if f.endswith(".ipynb") and ("output" in f or "executed" in f):
#                 path = os.path.join(root, f)
#                 print(f"🧹 Removing notebook: {path}")
#                 os.remove(path)

def check_git_size():
    git_path = os.path.join(SCRATCH, "OP_bench", ".git")  # update "your_repo" to match your folder name
    if os.path.exists(git_path):
        size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, _, filenames in os.walk(git_path)
            for filename in filenames
        )
        print(f"📦 .git folder size: {round(size / (1024**2), 2)} MB")
    else:
        print("✅ No .git folder found in your repo path")

if __name__ == "__main__":
    print("🚀 Starting scratch cleanup...\n")
    clear_caches()
    # if DELETE_NOTEBOOKS:
    #     clean_notebooks()
    check_git_size()
    print("\n✅ Cleanup complete.")
