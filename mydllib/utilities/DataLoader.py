import gzip
from typing import List, Tuple
import os
import urllib.request
import numpy as np
import struct


class DataLoader:
    def __init__(self, download_dir: str, urls: List[str]) -> None:
        self.download_dir = download_dir
        self.urls = urls

    def parse_file(self, path: str) -> np.ndarray | None:
        with gzip.open(path) as f:
            magic = struct.unpack('>I', f.read(4))[0]
            if magic == 2049:
                size = struct.unpack('>I', f.read(4))[0]
                return np.frombuffer(f.read(), dtype=np.uint8)
            elif magic == 2051:
                size, rows, cols = struct.unpack('>III', f.read(12))
                return np.frombuffer(f.read(), dtype=np.uint8).reshape(size, rows, cols)
            else:
                print(f"File {path} is corrupted or format is incorrect")

    def download_file(self, url: str, download_dir: str) -> str:
        path = download_dir + "/" + url.split('/')[-1]
        if not os.path.isfile(path):
            print(f"Downloading {path}")
            urllib.request.urlretrieve(url, path)
            print(f"Skipping to download {path}")
        return path

    def read_dataset(self) -> Tuple[np.ndarray]:
        data = []
        os.makedirs(self.download_dir, exist_ok=True)
        for url in self.urls:
            filepath = self.download_file(url, self.download_dir)
            chunk = self.parse_file(filepath)
            data.append(chunk)
        return tuple(data)
