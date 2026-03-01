"""
Stream individual files out of a remote HuggingFace zip using HTTP range requests.
Only the zip central directory + the requested parquet files are downloaded.
"""

import zipfile
import requests
from huggingface_hub import hf_hub_url, get_token

REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
FILENAME = "lidar/lidar_top_360fov/lidar_top_360fov.chunk_0000.zip"
N_FILES = 2


class RangeRequestFile:
    """Seekable file-like object backed by HTTP range requests."""

    def __init__(self, url: str, token: str | None = None):
        self.url = url
        self._pos = 0
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        resp = requests.head(url, headers=self._headers, allow_redirects=True)
        resp.raise_for_status()
        self.size = int(resp.headers["Content-Length"])

    def seek(self, pos: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = pos
        elif whence == 1:
            self._pos += pos
        elif whence == 2:
            self._pos = self.size + pos
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if n == -1:
            end = self.size - 1
        else:
            end = min(self._pos + n - 1, self.size - 1)
        if self._pos > self.size - 1:
            return b""
        resp = requests.get(self.url, headers={**self._headers, "Range": f"bytes={self._pos}-{end}"})
        resp.raise_for_status()
        data = resp.content
        self._pos += len(data)
        return data

    def readable(self) -> bool: return True
    def seekable(self) -> bool: return True
    def writable(self) -> bool: return False


url = hf_hub_url(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
token = get_token()
print(f"Remote zip URL:\n  {url}\n")

remote = RangeRequestFile(url, token=token)
print(f"Remote file size: {remote.size / 1e9:.2f} GB")

with zipfile.ZipFile(remote) as zf:
    names = zf.namelist()
    print(f"\nFiles in zip: {len(names)}")
    for n in names[:10]:
        info = zf.getinfo(n)
        print(f"  {n}  ({info.file_size / 1e6:.1f} MB  →  {info.compress_size / 1e6:.1f} MB compressed)")

    print(f"\nDownloading first {N_FILES} file(s)...")
    for name in names[:N_FILES]:
        data = zf.read(name)
        local_path = f"lidar_{name}"
        with open(local_path, "wb") as f:
            f.write(data)
        print(f"  Saved {local_path}  ({len(data) / 1e6:.1f} MB)")

print("\nDone.")
