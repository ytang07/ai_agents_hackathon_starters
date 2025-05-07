import requests
from config import GMI_API_KEY, create_or_open_lancedb_table

url = "https://api.gmi-serving.com/v1/embeddings"

headers = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Authorization": f"Bearer {GMI_API_KEY}"
}

payload = {
    "model": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
}

def get_embedding(text_or_img):
    if text_or_img.startswith("data:JPEG;base64") or text_or_img.startswith("data:image/jpeg;base64"):
        payload["input"] = [{"image": text_or_img}]
    else:
        payload["input"] = text_or_img

    response = requests.post(url, headers=headers, json=payload)
    print(response.status_code)
    return response.json()["data"][0]["embedding"]


def retrieve(text_or_img, limit=30):
    embedding = get_embedding(text_or_img)
    table = create_or_open_lancedb_table()
    # list all columns in schema except vector
    columns = [col.name for col in table.schema if col.name != "vector"]
    rs = table.search(embedding).limit(limit).select(columns).to_pandas()
    return rs

if __name__ == "__main__":
    print(retrieve("sea facing bungalow"))
