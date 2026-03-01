"""
Decode Draco lidar spins, downsample, and export synchronized JSON for the frontend.
Points are in vehicle frame — the frontend applies egomotion pose to place them on the map.
"""

import json
import numpy as np
import DracoPy
import pyarrow.parquet as pq

LIDAR_PATH = "C:/Users/laure/Projects/physical-av-dataloader/dataloader/lidar_25cd4769-5dcf-4b53-a351-bf2c5deb6124.lidar_top_360fov.parquet"
EGOMOTION_PATH = "C:/Users/laure/Projects/physical-av-dataloader/frontend/public/egomotion.json"
OUT_PATH = "C:/Users/laure/Projects/physical-av-dataloader/frontend/public/lidar.json"

SUBSAMPLE = 50       # keep every Nth point
MAX_RANGE = 60.0     # discard points farther than this (metres)
Z_MIN, Z_MAX = -4.0, 8.0  # discard ground-noise and sky

# --- load egomotion timestamps for sync ---
with open(EGOMOTION_PATH) as f:
    ego = json.load(f)
ego_ts = np.array([fr["timestamp"] for fr in ego["frames"]], dtype=np.int64)

# --- load lidar ---
table = pq.read_table(LIDAR_PATH)
timestamps = table.column("reference_timestamp").to_pylist()
blobs = table.column("draco_encoded_pointcloud").to_pylist()

spins = []
for i, (ts, blob) in enumerate(zip(timestamps, blobs)):
    pc = DracoPy.decode(bytes(blob))
    pts = pc.points  # (N, 3) float32  x, y, z in vehicle frame

    # filter by range and height
    dist = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    mask = (dist < MAX_RANGE) & (pts[:, 2] > Z_MIN) & (pts[:, 2] < Z_MAX)
    pts = pts[mask]

    # subsample
    pts = pts[::SUBSAMPLE]

    # round to 2 dp to keep JSON small
    pts = np.round(pts, 2)

    # nearest egomotion frame by timestamp
    ef = int(np.argmin(np.abs(ego_ts - ts)))

    spins.append({
        "t": ts,
        "ef": ef,
        "pts": pts.flatten().tolist(),
    })

    if (i + 1) % 20 == 0:
        print(f"  {i + 1}/{len(timestamps)}  pts after filter+subsample: {len(pts)}")

out = {"log_id": ego["log_id"], "spins": spins}
with open(OUT_PATH, "w") as f:
    json.dump(out, f, separators=(",", ":"))

size_mb = len(json.dumps(out, separators=(",", ":"))) / 1e6
print(f"\nSaved {len(spins)} spins → {OUT_PATH}  ({size_mb:.1f} MB)")
