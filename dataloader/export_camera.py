"""
Download camera MP4s and timestamps from HuggingFace, compute frame-to-egomotion
mapping, and export camera_index.json for synchronized playback in the frontend.
"""

import io
import json
import os
import shutil
import zipfile

import cv2
import numpy as np
import pyarrow.parquet as pq
import requests
from huggingface_hub import get_token, hf_hub_url

# ── config ────────────────────────────────────────────────────────────────────
LOG_ID  = "25cd4769-5dcf-4b53-a351-bf2c5deb6124"
REPO    = "nvidia/PhysicalAI-Autonomous-Vehicles"
CAMERAS = {
    "front": "camera_front_wide_120fov",
    "left":  "camera_cross_left_120fov",
    "right": "camera_cross_right_120fov",
}
EGOMOTION_PATH = "C:/Users/laure/Projects/physical-av-dataloader/frontend/public/egomotion.json"
OUT_DIR        = "C:/Users/laure/Projects/physical-av-dataloader/frontend/public"

# ── HTTP range-request helper ─────────────────────────────────────────────────
class RangeRequestFile:
    def __init__(self, url: str, token: str | None = None):
        self.url = url
        self._pos = 0
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        resp = requests.head(url, headers=self._headers, allow_redirects=True)
        resp.raise_for_status()
        self.size = int(resp.headers["Content-Length"])

    def seek(self, pos: int, whence: int = 0) -> int:
        if whence == 0:   self._pos = pos
        elif whence == 1: self._pos += pos
        elif whence == 2: self._pos = self.size + pos
        return self._pos

    def tell(self) -> int: return self._pos

    def read(self, n: int = -1) -> bytes:
        end = self.size - 1 if n == -1 else min(self._pos + n - 1, self.size - 1)
        if self._pos > self.size - 1: return b""
        resp = requests.get(self.url,
            headers={**self._headers, "Range": f"bytes={self._pos}-{end}"})
        resp.raise_for_status()
        data = resp.content
        self._pos += len(data)
        return data

    def readable(self) -> bool: return True
    def seekable(self) -> bool: return True
    def writable(self) -> bool: return False


# ── load egomotion timestamps ─────────────────────────────────────────────────
print("Loading egomotion.json...")
with open(EGOMOTION_PATH) as f:
    ego = json.load(f)
ego_ts = np.array([fr["timestamp"] for fr in ego["frames"]], dtype=np.int64)
print(f"  {len(ego_ts)} egomotion frames")
print(f"  ego ts range: [{ego_ts[0]}, {ego_ts[-1]}]")

token = get_token()
camera_index = {
    "log_id": LOG_ID,
    "cameras": {},
}

# ── process each camera ───────────────────────────────────────────────────────
for cam_name, cam_path in CAMERAS.items():
    print(f"\nProcessing {cam_name} ({cam_path})...")

    # Build HF URL and stream zip
    chunk = "chunk_0000"
    zip_file = f"camera/{cam_path}/{cam_path}.{chunk}.zip"
    url = hf_hub_url(REPO, zip_file, repo_type="dataset")
    remote = RangeRequestFile(url, token=token)
    print(f"  zip size: {remote.size/1e9:.2f} GB")

    # Extract timestamps and mp4 from zip
    with zipfile.ZipFile(remote) as zf:
        ts_bytes = zf.read(f"{LOG_ID}.{cam_path}.timestamps.parquet")
        # Stream MP4 to disk to avoid loading GB into RAM
        mp4_filename = f"camera_{cam_name}_wide_120fov.mp4"
        mp4_out_path = os.path.join(OUT_DIR, mp4_filename)
        with zf.open(f"{LOG_ID}.{cam_path}.mp4") as src, open(mp4_out_path, "wb") as dst:
            shutil.copyfileobj(src, dst, length=8*1024*1024)
        print(f"  saved: {mp4_filename}")

    # Extract timestamps and parse
    frame_ts = pq.read_table(io.BytesIO(ts_bytes)).to_pandas()["timestamp"].values.astype(np.int64)
    print(f"  {len(frame_ts)} camera frames")
    print(f"  cam ts range: [{frame_ts[0]}, {frame_ts[-1]}]")

    # Get FPS and frame count from video file
    cap = cv2.VideoCapture(mp4_out_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"  fps: {fps}, frame_count: {frame_count}")

    # Compute ego_to_frame: for each ego frame, find nearest camera frame index
    # Use same approach as colorize_lidar.py: direct timestamp comparison (no conversion)
    ego_interval = np.mean(np.diff(ego_ts))
    cam_interval = np.mean(np.diff(frame_ts))
    print(f"  ego interval: ~{ego_interval:.0f}, cam interval: ~{cam_interval:.0f}")

    ego_to_frame = np.argmin(np.abs(frame_ts[np.newaxis, :] - ego_ts[:, np.newaxis]), axis=1)

    # Clamp to valid frame range: videos are 0-indexed, so max valid index is frame_count-1
    ego_to_frame = np.clip(ego_to_frame, 0, max(0, frame_count - 1))
    print(f"  ego_to_frame: {len(ego_to_frame)} entries, range=[{ego_to_frame.min()}, {ego_to_frame.max()}], video has {frame_count} frames")

    # Store in output
    camera_index["cameras"][cam_name] = {
        "file": mp4_filename,
        "fps": float(fps),
        "ego_to_frame": ego_to_frame.tolist(),
    }

# ── write output ──────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "camera_index.json")
print(f"\nWriting {out_path}...")
with open(out_path, "w") as f:
    json.dump(camera_index, f, separators=(",", ":"))

size_kb = os.path.getsize(out_path) / 1e3
print(f"Done. {size_kb:.1f} KB")
