# Physical AV DataLoader

3D visualization of egomotion and LiDAR data with synchronized multi-camera views from the NVIDIA PhysicalAI Autonomous Vehicles dataset.

## Features

- **3D LiDAR Visualization**: Real-time point cloud rendering with egomotion pose alignment
- **Synchronized Multi-Camera Panels**: Front (120° FOV), left (120° FOV), and right (120° FOV) camera feeds synchronized with LiDAR playback
- **Interactive Controls**: Orbit camera, follow mode, point accumulation, display toggles
- **Height-based Coloring**: LiDAR points colored by elevation; camera RGB fallback available

## Setup

### Prerequisites

- Python 3.10+
- Node.js / pnpm
- ffmpeg (for MP4 optimization)
- HuggingFace API token (for dataset access)

### Installation

```bash
# Install Python dependencies
pip install -r dataloader/requirements.txt

# Install frontend dependencies
cd frontend
pnpm install
```

### Data Export

#### 1. Export LiDAR and Egomotion

```bash
cd dataloader
python export_lidar.py
```

Outputs:
- `frontend/public/lidar.json` — downsampled point clouds with spin indices
- `frontend/public/egomotion.json` — vehicle pose, velocity, acceleration

#### 2. Export and Index Camera Videos

```bash
python export_camera.py
```

Outputs:
- `frontend/public/camera_front_wide_120fov.mp4`
- `frontend/public/camera_left_wide_120fov.mp4`
- `frontend/public/camera_right_wide_120fov.mp4`
- `frontend/public/camera_index.json` — frame-to-egomotion mapping

#### 3. Optimize MP4s for Seeking (IMPORTANT)

The HuggingFace dataset MP4s have the `moov` atom at the end, which prevents instant seeking without buffering the entire file. Re-mux them to place the `moov` atom at the front:

```bash
cd frontend/public

# Front camera
ffmpeg -y -i camera_front_wide_120fov.mp4 -movflags faststart -c copy camera_front_reindex.mp4 && mv camera_front_reindex.mp4 camera_front_wide_120fov.mp4

# Left camera
ffmpeg -y -i camera_left_wide_120fov.mp4 -movflags faststart -c copy camera_left_reindex.mp4 && mv camera_left_reindex.mp4 camera_left_wide_120fov.mp4

# Right camera
ffmpeg -y -i camera_right_wide_120fov.mp4 -movflags faststart -c copy camera_right_reindex.mp4 && mv camera_right_reindex.mp4 camera_right_wide_120fov.mp4
```

**Why?** The `-movflags faststart` flag reorders the MP4 atoms so the `moov` (metadata) block comes first. This enables:
- Instant seeking without waiting for file download
- Smooth video scrubbing on slower connections
- Proper frame synchronization with LiDAR playback

**Note:** `-c copy` avoids re-encoding, so this is fast (~1 min per 2GB file).

### Running the Frontend

```bash
cd frontend
pnpm dev
```

Open `http://localhost:3000` in your browser.

## Controls

### Keyboard Shortcuts

- **P** — Play/Pause
- **D** — Toggle vehicle display (car model ↔ arrow)
- **R** — Toggle rotate-follow camera mode
- **K** — Toggle point accumulation (keep previous LiDAR frames)

### Mouse & UI

- **Drag** — Orbit camera
- **Scroll** — Zoom in/out
- **Right-drag** — Pan camera
- **Play/Pause** — Control playback
- **Slider** — Seek to frame (syncs all videos)
- **LiDAR button** — Toggle point cloud visibility
- **Keep button** — Toggle accumulation mode
- **Follow button** — Toggle automatic camera following

## File Structure

```
physical-av-dataloader/
├── dataloader/
│   ├── export_lidar.py         # LiDAR export (Draco decode, downsample, JSON)
│   ├── export_camera.py        # Camera export (HF download, frame mapping)
│   ├── colorize_lidar.py       # (Reference) LiDAR RGB projection
│   └── requirements.txt
├── frontend/
│   ├── app/
│   │   └── page.tsx            # Main React component
│   ├── public/
│   │   ├── lidar.json          # Generated: LiDAR spins
│   │   ├── egomotion.json      # Generated: vehicle poses
│   │   ├── camera_index.json   # Generated: frame sync mapping
│   │   ├── camera_*.mp4        # Generated: video files (after export + ffmpeg)
│   │   └── ...
│   └── package.json
└── README.md
```

## Troubleshooting

### Videos freeze after a few frames

**Cause:** MP4 files have `moov` atom at the end (not seekable without full download).

**Solution:** Run the ffmpeg re-muxing step above to move the `moov` atom to the front.

### Camera panels show "No camera data"

**Cause:** `camera_index.json` not found or failed to load.

**Solution:**
1. Verify `export_camera.py` completed successfully
2. Check `frontend/public/camera_index.json` exists
3. Check browser console for network errors

### Frame sync is off

**Cause:** Timestamp mismatch between ego and camera recordings.

**Solution:** This is expected if ego recording extends beyond camera availability. The mapping uses nearest-neighbor alignment and clamps to valid frame ranges.

### Slow seeking / buffering

**Cause:** Network is slow or browser is still buffering.

**Solution:**
1. Ensure ffmpeg re-muxing was applied (`-movflags faststart`)
2. Wait for initial video buffering (depends on file size and connection)
3. Check network tab in DevTools for HTTP 206 (range request) responses

## Browser Compatibility

- **Chrome/Edge**: ✅ Excellent (hardware acceleration, range requests)
- **Firefox**: ✅ Good
- **Safari**: ⚠️ May require additional codec support

## Development

### Debug Camera Index

In the browser console:
```javascript
debugCamera()
```

Shows:
- Loaded cameras and their properties
- Frame mapping lengths and ranges
- FPS values

### Build for Production

```bash
cd frontend
pnpm build
pnpm start
```

## References

- [NVIDIA PhysicalAI Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- [Three.js Documentation](https://threejs.org/docs/)
- [React Hooks Guide](https://react.dev/reference/react)
