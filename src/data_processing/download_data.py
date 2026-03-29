import requests
from tqdm import tqdm

url = "https://www.dropbox.com/scl/fo/2aczdnx37hxvcfdo4rq4q/AOjRokSTaiKxXmgUyqdcI6k?rlkey=5bg7mxxbq46t7aujfch46dlvz&dl=1"

response = requests.get(url, stream=True)
total_size = int(response.headers.get("content-length", 0))

with open("/home/emogenai4e/emo/Hung_data/data.zip", "wb") as f, tqdm(
    desc="Đang tải",
    total=total_size,
    unit="B",
    unit_scale=True,
    unit_divisor=1024,
) as bar:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
        bar.update(len(chunk))

print("Tải xong!")