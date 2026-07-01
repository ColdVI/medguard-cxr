# MEDGUARD-CXR COLAB GIT-ONLY SETUP (NO KAGGLE DATA DOWNLOAD)

import os
import shutil
import subprocess
from pathlib import Path

REPO_ZIP_URL_MAIN = "https://github.com/ColdVI/medguard-cxr/archive/refs/heads/main.zip"
REPO_ZIP_URL_MASTER = "https://github.com/ColdVI/medguard-cxr/archive/refs/heads/master.zip"

REPO_DIR = Path("/content/MedImage")

def run(cmd, cwd=None, check=True):
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return result

print("=== 1. GPU / DISK CHECK ===")
run("nvidia-smi", check=False)
run("python --version", check=False)

print("\n=== 2. CLEAN OLD LOCAL TEMP ===")
for p in ["/content/MedImage"]:
    path = Path(p)
    if path.exists():
        print("Removing:", p)
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink()

print("\n=== 3. DOWNLOAD REPO ZIP ===")
repo_zip = Path("/content/medguard.zip")
ok = run(f'wget -q -O "{repo_zip}" "{REPO_ZIP_URL_MAIN}"', check=False)
extract_name = "medguard-cxr-main"

if ok.returncode != 0 or repo_zip.stat().st_size < 1000:
    print("main.zip failed, trying master.zip")
    ok = run(f'wget -q -O "{repo_zip}" "{REPO_ZIP_URL_MASTER}"', check=False)
    extract_name = "medguard-cxr-master"

if ok.returncode != 0 or repo_zip.stat().st_size < 1000:
    raise RuntimeError("Could not download repo zip.")

run(f'unzip -q "{repo_zip}" -d /content')

extracted = Path("/content") / extract_name
if not extracted.exists():
    candidates = list(Path("/content").glob("medguard-cxr-*"))
    if not candidates:
        raise RuntimeError("Repo extracted but not found.")
    extracted = candidates[0]

shutil.move(str(extracted), str(REPO_DIR))
os.chdir(REPO_DIR)
print("Repo ready:", Path.cwd())

print("\n=== 4. INSTALL PROJECT ===")
run("pip install -q --upgrade pip")
run('pip install -q -e ".[dev,vlm]"')

print("\n=== 5. REPO CHECKS ===")
run("make lint")
run("make test", check=False)

print("\n✅ GIT ONLY SETUP COMPLETE.")
