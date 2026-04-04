import subprocess
import sys

for pkg in ["gdown", "tqdm"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import gdown

folder_url = "https://drive.google.com/drive/folders/1Kh1mQ1KyDOiYqxTvupFyvwO_70o3__gt"

gdown.download_folder(folder_url, output=".", quiet=False)
