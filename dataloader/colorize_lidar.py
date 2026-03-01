"""
Colorize LiDAR points with RGB from all wide cameras using fisheye projection.
Processes front, left, right, rear-left, rear-right cameras in priority order;
first valid projection per point wins.
"""

import io
import json
import os
import tempfile
import zipfile

import cv2
import numpy as np
import pyarrow.parquet as pq
import requests
from huggingface_hub import get_token, hf_hub_url

# ── config ────────────────────────────────────────────────────────────────────
LOG_ID  = "25cd4769-5dcf-4b53-a351-bf2c5deb6124"
REPO    = "nvidia/PhysicalAI-Autonomous-Vehicles"

# cameras in priority order (front first so it wins when overlap exists)
CAMERAS = [
    "camera_front_wide_120fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_rear_left_70fov",
    "camera_rear_right_70fov",
]

INTR_PATH  = "calibration/calibration/camera_intrinsics/camera_intrinsics.chunk_0000.parquet"
EXTR_PATH  = "calibration/calibration/sensor_extrinsics/sensor_extrinsics.chunk_0000.parquet"
LIDAR_JSON = "C:/Users/laure/Projects/physical-av-dataloader/frontend/public/lidar.json"
OUT_PATH   = LIDAR_JSON

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


# ── helpers ───────────────────────────────────────────────────────────────────
def quat_to_rot(qx, qy, qz, qw) -> np.ndarray:
    return np.array([
        [1-2*(qy*qy+qz*qz),  2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),    1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),    2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
    ])


def project_fisheye(pts_cam: np.ndarray, fw: list, cx: float, cy: float,
                    W: int, H: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """fw_poly: angle (rad) → pixel radius.  Returns u, v, valid_mask."""
    X, Y, Z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    rho = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(rho, Z)
    r = sum(b * (theta ** i) for i, b in enumerate(fw))
    safe_rho = np.where(rho > 1e-9, rho, 1.0)
    u = cx + r * (X / safe_rho)
    v = cy + r * (Y / safe_rho)
    valid = (Z > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, valid


# ── load calibration ──────────────────────────────────────────────────────────
print("Loading calibration...")
intr_df = pq.read_table(INTR_PATH).to_pandas()
extr_df = pq.read_table(EXTR_PATH).to_pandas()

token = get_token()

# per-camera data: calibration + downloaded frames
cam_data = {}   # cam_name → {fw, cx, cy, W, H, R_veh_to_cam, t_cam, frame_ts, cap, tmp_path}

for cam in CAMERAS:
    intr = intr_df.loc[(LOG_ID, cam)]
    extr = extr_df.loc[(LOG_ID, cam)]

    fw   = [intr[f"fw_poly_{i}"] for i in range(5)]
    cx, cy = float(intr["cx"]), float(intr["cy"])
    W, H   = int(intr["width"]), int(intr["height"])

    R_cam        = quat_to_rot(extr["qx"], extr["qy"], extr["qz"], extr["qw"])
    R_veh_to_cam = R_cam.T
    t_cam        = np.array([extr["x"], extr["y"], extr["z"]])

    cam_data[cam] = dict(fw=fw, cx=cx, cy=cy, W=W, H=H,
                         R_veh_to_cam=R_veh_to_cam, t_cam=t_cam)
    print(f"  {cam}: {W}x{H}  pos={t_cam.round(2)}")


# ── stream each camera zip ────────────────────────────────────────────────────
print("\nStreaming camera zips...")
tmp_paths = []

for cam in CAMERAS:
    chunk = "chunk_0000"
    zip_file = f"camera/{cam}/{cam}.{chunk}.zip"
    url = hf_hub_url(REPO, zip_file, repo_type="dataset")
    remote = RangeRequestFile(url, token=token)
    print(f"  {cam} zip: {remote.size/1e9:.2f} GB")

    with zipfile.ZipFile(remote) as zf:
        ts_bytes  = zf.read(f"{LOG_ID}.{cam}.timestamps.parquet")
        mp4_bytes = zf.read(f"{LOG_ID}.{cam}.mp4")

    frame_ts = pq.read_table(io.BytesIO(ts_bytes)).to_pandas()["timestamp"].values.astype(np.int64)

    tmp = tempfile.mktemp(suffix=".mp4")
    with open(tmp, "wb") as f:
        f.write(mp4_bytes)
    tmp_paths.append(tmp)

    cap = cv2.VideoCapture(tmp)
    cam_data[cam]["frame_ts"] = frame_ts
    cam_data[cam]["cap"]      = cap
    cam_data[cam]["prev_fi"]  = -1
    cam_data[cam]["frame"]    = None
    print(f"    {len(frame_ts)} frames, ts [{frame_ts[0]}, {frame_ts[-1]}]")


# ── load lidar.json ───────────────────────────────────────────────────────────
print("\nLoading lidar.json...")
with open(LIDAR_JSON) as f:
    lidar = json.load(f)
spins = lidar["spins"]
print(f"  {len(spins)} spins")


# ── colorize ──────────────────────────────────────────────────────────────────
print("\nColorizing spins...")

for si, spin in enumerate(spins):
    ts  = spin["t"]
    pts = np.array(spin["pts"], dtype=np.float32).reshape(-1, 3)
    n   = len(pts)

    colors  = np.zeros((n, 3), dtype=np.uint8)
    colored = np.zeros(n, dtype=bool)   # track which points are already assigned

    for cam, cd in cam_data.items():
        if colored.all():
            break   # every point already has a color

        # nearest frame for this camera
        fi = int(np.argmin(np.abs(cd["frame_ts"] - ts)))
        fi = min(fi, int(cd["cap"].get(cv2.CAP_PROP_FRAME_COUNT)) - 1)

        if fi != cd["prev_fi"]:
            cd["cap"].set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cd["cap"].read()
            cd["frame"]   = frame if ret else None
            cd["prev_fi"] = fi

        if cd["frame"] is None:
            continue

        # transform vehicle-frame → camera frame
        pts_cam = (cd["R_veh_to_cam"] @ (pts - cd["t_cam"]).T).T

        u, v, valid = project_fisheye(pts_cam, cd["fw"], cd["cx"], cd["cy"],
                                      cd["W"], cd["H"])

        # only fill uncolored points that project into this camera
        fill = valid & ~colored
        if not fill.any():
            continue

        ui = u[fill].astype(int)
        vi = v[fill].astype(int)
        bgr = cd["frame"][vi, ui]
        colors[fill, 0] = bgr[:, 2]   # R
        colors[fill, 1] = bgr[:, 1]   # G
        colors[fill, 2] = bgr[:, 0]   # B
        colored[fill] = True

    spin["rgb"] = colors.flatten().tolist()

    if (si + 1) % 20 == 0:
        print(f"  spin {si+1}/{len(spins)}  colored={colored.sum()}/{n} "
              f"({100*colored.sum()/n:.0f}%)")


# ── cleanup ───────────────────────────────────────────────────────────────────
for cam, cd in cam_data.items():
    cd["cap"].release()
for p in tmp_paths:
    if os.path.exists(p):
        os.unlink(p)


# ── write output ──────────────────────────────────────────────────────────────
print(f"\nWriting {OUT_PATH}...")
with open(OUT_PATH, "w") as f:
    json.dump(lidar, f, separators=(",", ":"))

size_mb = os.path.getsize(OUT_PATH) / 1e6
print(f"Done. {size_mb:.1f} MB")
