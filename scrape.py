import os
import pickle
import requests
from bs4 import BeautifulSoup

def scrape(url="https://www.gutenberg.org/files/1342/1342-0.txt", out_file="data/data.pkl"):
    """Scrape text data and save as pickle."""
    os.makedirs("data", exist_ok=True)

    print(f"Downloading text from {url}...")
    response = requests.get(url)
    text = response.text

    text = text.replace("\r", "").replace("\n", " ")

    chars = sorted(list(set(text)))
    char_to_int = {c: i for i, c in enumerate(chars)}
    int_to_char = {i: c for i, c in enumerate(chars)}

    encoded = [char_to_int[c] for c in text]

    with open(out_file, "wb") as f:
        pickle.dump((encoded, char_to_int, int_to_char), f)

    print(f"Saved dataset to {out_file}, vocab size: {len(chars)}, text length: {len(encoded)}")

if __name__ == "__main__":
    scrape()
