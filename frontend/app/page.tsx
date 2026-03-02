"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// ── types ──────────────────────────────────────────────────────────────────
interface Frame {
  timestamp: number;
  x: number; y: number; z: number;
  qx: number; qy: number; qz: number; qw: number;
  vx: number; vy: number; vz: number;
  ax: number; ay: number; az: number;
  curvature: number;
}
interface EgomotionData { log_id: string; frames: Frame[]; }
interface LidarSpin { t: number; ef: number; pts: number[]; rgb?: number[]; }
interface LidarData { log_id: string; spins: LidarSpin[]; }
interface CameraEntry { file: string; fps: number; ego_to_frame: number[]; }
interface CameraIndex { log_id: string; cameras: { front: CameraEntry; left: CameraEntry; right: CameraEntry; }; }

// ── coordinate helpers ─────────────────────────────────────────────────────
// Data coords passed through directly; camera is oriented to taste.
function p3(x: number, y: number, z: number) { return new THREE.Vector3(x, y, z); }

// No frame conversion needed
const COORD_Q = new THREE.Quaternion(0, 0, 0, 1);
function toThreeQ(qx: number, qy: number, qz: number, qw: number) {
  return COORD_Q.clone().multiply(new THREE.Quaternion(qx, qy, qz, qw));
}

// height → HSL color (blue=low, yellow=high), returned as [r,g,b] 0-1
function heightRGB(z: number): [number, number, number] {
  const t = Math.max(0, Math.min(1, (z + 2) / 6));
  const c = new THREE.Color().setHSL((1 - t) * 0.56 + t * 0.17, 0.9, 0.55);
  return [c.r, c.g, c.b];
}

function speed(vx: number, vy: number, vz: number) {
  return Math.sqrt(vx * vx + vy * vy + vz * vz);
}

// ── diagnostic helper ─────────────────────────────────────────────────────
declare global { interface Window { debugCamera?: () => void; } }
if (typeof window !== "undefined") {
  window.debugCamera = () => {
    fetch("/camera_index.json").then(r => r.json()).then(d => {
      console.log("=== CAMERA INDEX ===");
      console.log("Log ID:", d.log_id);
      for (const [name, entry] of Object.entries(d.cameras)) {
        const e = entry as any;
        console.log(`${name}:`, {
          file: e.file,
          fps: e.fps,
          ego_to_frame_len: e.ego_to_frame.length,
          ego_to_frame_min: Math.min(...e.ego_to_frame),
          ego_to_frame_max: Math.max(...e.ego_to_frame),
        });
      }
    }).catch(err => console.error("Failed to fetch camera_index.json:", err));
  };
}

// ── component ──────────────────────────────────────────────────────────────
export default function Home() {
  const mountRef = useRef<HTMLDivElement>(null);
  const [ego, setEgo] = useState<EgomotionData | null>(null);
  const [lidar, setLidar] = useState<LidarData | null>(null);
  const [frameIdx, setFrameIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [showLidar, setShowLidar] = useState(true);
  const [keepPoints, setKeepPoints] = useState(false);
  const [rotateFollow, setRotateFollow] = useState(false);
  const [displayMode, setDisplayMode] = useState<"car" | "arrow">("car");

  // Camera state and refs
  const [camIdx, setCamIdx] = useState<CameraIndex | null>(null);
  const frontVideoRef = useRef<HTMLVideoElement>(null);
  const leftVideoRef  = useRef<HTMLVideoElement>(null);
  const rightVideoRef = useRef<HTMLVideoElement>(null);
  const camIdxRef = useRef<CameraIndex | null>(null);

  // Three.js object refs (set during scene init, used during frame updates)
  const carRef = useRef<THREE.Group | null>(null);
  const currentLidarRef = useRef<THREE.Points | null>(null);
  const accLidarRef = useRef<THREE.Points | null>(null);
  const accPositions = useRef<number[]>([]);
  const accColors = useRef<number[]>([]);
  const accSpins = useRef<Set<number>>(new Set());
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const rotateFollowRef = useRef(false);
  const orbitAngleRef = useRef(0);
  const arrowRef = useRef<THREE.Group | null>(null);
  const displayModeRef = useRef<"car" | "arrow">("car");

  // live state refs for the RAF loop
  const frameIdxRef = useRef(0);
  const playingRef = useRef(false);
  const showLidarRef = useRef(true);
  const keepRef = useRef(false);
  const egoRef = useRef<EgomotionData | null>(null);
  const lidarRef = useRef<LidarData | null>(null);

  useEffect(() => { frameIdxRef.current = frameIdx; }, [frameIdx]);
  useEffect(() => { playingRef.current = playing; }, [playing]);
  useEffect(() => { showLidarRef.current = showLidar; }, [showLidar]);
  useEffect(() => { keepRef.current = keepPoints; }, [keepPoints]);
  useEffect(() => { rotateFollowRef.current = rotateFollow; }, [rotateFollow]);
  useEffect(() => { displayModeRef.current = displayMode; }, [displayMode]);
  useEffect(() => { egoRef.current = ego; }, [ego]);
  useEffect(() => { lidarRef.current = lidar; }, [lidar]);
  useEffect(() => { camIdxRef.current = camIdx; }, [camIdx]);

  // load data
  useEffect(() => {
    fetch("/egomotion.json").then(r => r.json()).then(setEgo);
    fetch("/lidar.json").then(r => r.json()).then(setLidar);
    fetch("/camera_index.json")
      .then(r => { if (!r.ok) throw new Error(); return r.json(); })
      .then(setCamIdx).catch(() => {});   // optional — app works without it
  }, []);

  // attach video event listeners for debugging
  useEffect(() => {
    const videos = [
      [frontVideoRef, "front"],
      [leftVideoRef, "left"],
      [rightVideoRef, "right"],
    ];

    videos.forEach(([ref, label]) => {
      const el = ref.current;
      if (!el) return;

      const log = (evt: string) => {
        console.log(`${label} ${evt}: readyState=${el.readyState}, duration=${el.duration.toFixed(2)}s, buffered=${el.buffered.length > 0 ? `0-${el.buffered.end(el.buffered.length - 1).toFixed(1)}s` : 'none'}`);
      };

      el.addEventListener("loadstart", () => log("loadstart"));
      el.addEventListener("progress", () => log("progress"));
      el.addEventListener("canplay", () => log("canplay"));
      el.addEventListener("canplaythrough", () => log("canplaythrough"));
      el.addEventListener("loadeddata", () => log("loadeddata"));
      el.addEventListener("loadedmetadata", () => log("loadedmetadata"));
      el.addEventListener("seeking", () => log("seeking"));
      el.addEventListener("seeked", () => log("seeked"));
      el.addEventListener("stalled", () => log("stalled"));
      el.addEventListener("suspend", () => log("suspend"));
    });
  }, []);

  // synchronize camera video playback with frame index
  useEffect(() => {
    if (!camIdx) return;
    const pairs: [React.RefObject<HTMLVideoElement>, CameraEntry, string][] = [
      [frontVideoRef, camIdx.cameras.front, "front"],
      [leftVideoRef,  camIdx.cameras.left, "left"],
      [rightVideoRef, camIdx.cameras.right, "right"],
    ];

    // Log state every 100 frames for debugging
    const shouldLog = frameIdx % 100 === 0;
    if (shouldLog) {
      console.log(`[Frame ${frameIdx}]`);
    }

    for (const [ref, entry, label] of pairs) {
      const el = ref.current;
      if (!el) {
        if (shouldLog) console.log(`  ${label}: no ref`);
        continue;
      }

      // Bounds check: frameIdx should be within ego_to_frame length
      if (frameIdx >= entry.ego_to_frame.length) {
        if (shouldLog) console.log(`  ${label}: frameIdx ${frameIdx} >= ego_to_frame.length ${entry.ego_to_frame.length}`);
        continue;
      }

      const frameNum = entry.ego_to_frame[frameIdx];
      const t = frameNum / entry.fps;

      // Log video state
      if (shouldLog) {
        console.log(`  ${label}: readyState=${el.readyState}, duration=${el.duration.toFixed(2)}s, currentTime=${el.currentTime.toFixed(2)}s, targetTime=${t.toFixed(2)}s, buffered=[${el.buffered.length > 0 ? `0-${el.buffered.end(el.buffered.length - 1).toFixed(1)}s` : 'none'}]`);
      }

      // Only seek if video is ready and time is valid
      if (!isNaN(t) && isFinite(t) && el.readyState >= 2 && el.duration > 0) {
        // Clamp seek time to video duration to prevent seeking past EOF
        const clampedT = Math.min(t, el.duration - 0.01);
        const threshold = 0.5 / entry.fps;
        if (Math.abs(el.currentTime - clampedT) > threshold) {
          try {
            el.currentTime = clampedT;
            if (shouldLog) console.log(`    → seeked to ${clampedT.toFixed(2)}s`);
          } catch (e) {
            if (shouldLog) console.warn(`    → seek failed:`, e);
          }
        }
      } else if (shouldLog) {
        console.log(`    → not seeking (isNaN=${isNaN(t)}, isFinite=${isFinite(t)}, readyState=${el.readyState}, duration=${el.duration})`);
      }
    }
  }, [frameIdx, camIdx]);

  // keypress
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "p" || e.key === "P") {
        e.preventDefault();
        setPlaying(p => {
          if (!p && ego && frameIdx >= ego.frames.length - 1) setFrameIdx(0);
          return !p;
        });
      }
      if (e.key === "d" || e.key === "D") {
        setDisplayMode(prev => prev === "car" ? "arrow" : "car");
      }
      if (e.key === "r" || e.key === "R") {
        setRotateFollow(prev => {
          if (!prev) orbitAngleRef.current = 0;
          else if (controlsRef.current) controlsRef.current.enabled = true;
          return !prev;
        });
      }
      if (e.key === "k" || e.key === "K") {
        setKeepPoints(prev => {
          if (prev) {
            accPositions.current = [];
            accColors.current = [];
            accSpins.current.clear();
            if (accLidarRef.current) {
              accLidarRef.current.geometry.setAttribute("position",
                new THREE.BufferAttribute(new Float32Array(0), 3));
              accLidarRef.current.geometry.setAttribute("color",
                new THREE.BufferAttribute(new Float32Array(0), 3));
            }
          }
          return !prev;
        });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [ego, frameIdx]);

  // animation playback
  useEffect(() => {
    if (!playing || !ego) return;
    let raf: number;
    let count = 0;
    const tick = () => {
      if (!playingRef.current) return;
      if (++count % 3 === 0) {
        setFrameIdx(prev => {
          if (prev + 1 >= ego.frames.length) { setPlaying(false); return prev; }
          return prev + 1;
        });
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, ego]);

  // ── Three.js scene init (runs once ego is loaded) ──────────────────────
  useEffect(() => {
    if (!ego || !mountRef.current) return;
    const mount = mountRef.current;
    const W = mount.clientWidth || 720;
    const H = 520;

    // renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(W, H);
    mount.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f172a);
    scene.fog = new THREE.FogExp2(0x0f172a, 0.003);

    // camera
    const camera = new THREE.PerspectiveCamera(55, W / H, 0.1, 5000);
    camera.position.set(0, 80, 100);
    camera.lookAt(0, 0, 0);

    // controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controlsRef.current = controls;

    // grid — lies in XY plane (Z-up data frame)
    const grid = new THREE.GridHelper(600, 60, 0x1e293b, 0x1e293b);
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);

    // lights
    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const sun = new THREE.DirectionalLight(0xffffff, 0.8);
    sun.position.set(30, 60, 30);
    scene.add(sun);

    // ── egomotion path ────────────────────────────────────────────────────
    const pathPositions: number[] = [];
    const pathColors: number[] = [];
    ego.frames.forEach(f => {
      const v = p3(f.x, f.y, f.z);
      pathPositions.push(v.x, v.y, v.z);
      const norm = Math.min(speed(f.vx, f.vy, f.vz) / 20, 1);
      pathColors.push(norm * 0.86, 0.31, (1 - norm) * 0.78 + 0.16);
    });
    const pathGeo = new THREE.BufferGeometry();
    pathGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(pathPositions), 3));
    pathGeo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(pathColors), 3));
    scene.add(new THREE.Line(pathGeo, new THREE.LineBasicMaterial({ vertexColors: true })));

    // ── car ───────────────────────────────────────────────────────────────
    const car = new THREE.Group();
    // Align geometry to vehicle frame via two explicit rotations:
    // 1. Roll 90° CCW around geometry's forward axis (Z): Rz(+90°)
    // 2. Yaw  90° CW  around geometry's up axis (Y):     Ry(-90°)
    const pivot = new THREE.Group();
    const rollQ = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1),  Math.PI / 2);
    const yawQ  = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -Math.PI / 2);
    const flipQ = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1),  Math.PI); // correct upside-down
    pivot.quaternion.multiplyQuaternions(yawQ, flipQ.multiply(rollQ)); // roll, flip, then yaw
    car.add(pivot);

    const bodyMat  = new THREE.MeshStandardMaterial({ color: 0xfacc15, roughness: 0.35, metalness: 0.5 });
    const darkMat  = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.6, metalness: 0.2 });
    const glassMat = new THREE.MeshStandardMaterial({ color: 0x7dd3fc, roughness: 0.1, metalness: 0.1, transparent: true, opacity: 0.6 });
    const tireMat  = new THREE.MeshStandardMaterial({ color: 0x111827, roughness: 0.9 });
    const rimMat   = new THREE.MeshStandardMaterial({ color: 0xd1d5db, metalness: 0.8, roughness: 0.2 });

    // lower body — full length
    const lowerBody = new THREE.Mesh(new THREE.BoxGeometry(2.0, 0.7, 4.6), bodyMat);
    lowerBody.position.set(0, 0.55, 0);
    pivot.add(lowerBody);

    // bumpers (front & rear)
    for (const z of [-2.45, 2.45]) {
      const bumper = new THREE.Mesh(new THREE.BoxGeometry(2.0, 0.45, 0.2), darkMat);
      bumper.position.set(0, 0.35, z);
      pivot.add(bumper);
    }

    // cabin — sits on top of lower body, slightly narrower and shorter than full length
    const cabin = new THREE.Mesh(new THREE.BoxGeometry(1.75, 0.65, 2.4), bodyMat);
    cabin.position.set(0, 1.22, 0.15);
    pivot.add(cabin);

    // windshield (front glass)
    const windshield = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.58, 0.05), glassMat);
    windshield.rotation.x = Math.PI * 0.18;
    windshield.position.set(0, 1.18, -1.1);
    pivot.add(windshield);

    // rear window
    const rearWin = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.5, 0.05), glassMat);
    rearWin.rotation.x = -Math.PI * 0.15;
    rearWin.position.set(0, 1.18, 1.2);
    pivot.add(rearWin);

    // side windows (left & right)
    for (const [x, rx] of [[-0.88, 0], [0.88, 0]] as [number, number][]) {
      const sideWin = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.45, 1.9), glassMat);
      sideWin.position.set(x, 1.22, 0.1);
      pivot.add(sideWin);
    }

    // headlights
    for (const [x, col] of [[-0.65, 0xfef9c3], [0.65, 0xfef9c3]] as [number, number][]) {
      const hl = new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.18, 0.05),
        new THREE.MeshStandardMaterial({ color: col, emissive: col, emissiveIntensity: 1.5 }));
      hl.position.set(x, 0.62, -2.35);
      pivot.add(hl);
    }

    // tail lights
    for (const x of [-0.65, 0.65]) {
      const tl = new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.18, 0.05),
        new THREE.MeshStandardMaterial({ color: 0xef4444, emissive: 0xef4444, emissiveIntensity: 1.2 }));
      tl.position.set(x, 0.62, 2.35);
      pivot.add(tl);
    }

    // wheels — CylinderGeometry(rTop, rBot, height, segments), rotated to lie on XY plane
    const wheelPositions: [number, number, number][] = [
      [-1.05, 0.38, -1.4],  // front-left
      [ 1.05, 0.38, -1.4],  // front-right
      [-1.05, 0.38,  1.4],  // rear-left
      [ 1.05, 0.38,  1.4],  // rear-right
    ];
    for (const [wx, wy, wz] of wheelPositions) {
      const tire = new THREE.Mesh(new THREE.CylinderGeometry(0.38, 0.38, 0.28, 20), tireMat);
      tire.rotation.z = Math.PI / 2;
      tire.position.set(wx, wy, wz);
      pivot.add(tire);

      const rim = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 0.3, 10), rimMat);
      rim.rotation.z = Math.PI / 2;
      rim.position.set(wx, wy, wz);
      pivot.add(rim);
    }

    // glow point light
    const glow = new THREE.PointLight(0xfacc15, 2, 15);
    glow.position.set(0, 2, 0);
    car.add(glow);

    scene.add(car);
    carRef.current = car;

    // ── GPS arrow ─────────────────────────────────────────────────────────
    // Flat arrow in the XY (ground) plane, nose pointing in vehicle +X
    const arrowShape = new THREE.Shape();
    arrowShape.moveTo( 3.5,  0);     // nose
    arrowShape.lineTo(-1.5,  2.2);   // rear-left
    arrowShape.lineTo(-0.6,  0.9);   // notch-left
    arrowShape.lineTo(-0.6, -0.9);   // notch-right
    arrowShape.lineTo(-1.5, -2.2);   // rear-right
    arrowShape.closePath();

    const arrowMesh = new THREE.Mesh(
      new THREE.ShapeGeometry(arrowShape),
      new THREE.MeshStandardMaterial({
        color: 0x22d3ee, emissive: 0x22d3ee, emissiveIntensity: 0.6,
        side: THREE.DoubleSide,
      })
    );
    arrowMesh.position.z = 0.15; // slightly above ground
    const arrowGroup = new THREE.Group();
    arrowGroup.add(arrowMesh);
    arrowGroup.visible = false;
    scene.add(arrowGroup);
    arrowRef.current = arrowGroup;

    // ── current lidar points ──────────────────────────────────────────────
    const curGeo = new THREE.BufferGeometry();
    const curMat = new THREE.PointsMaterial({
      size: 0.35, vertexColors: true, sizeAttenuation: true,
    });
    const curPoints = new THREE.Points(curGeo, curMat);
    scene.add(curPoints);
    currentLidarRef.current = curPoints;

    // ── accumulated lidar points ──────────────────────────────────────────
    const accGeo = new THREE.BufferGeometry();
    const accMat = new THREE.PointsMaterial({
      size: 0.25, vertexColors: true, sizeAttenuation: true, transparent: true, opacity: 0.8,
    });
    const accPoints = new THREE.Points(accGeo, accMat);
    scene.add(accPoints);
    accLidarRef.current = accPoints;

    // ── position camera above start ───────────────────────────────────────
    const startPos = p3(ego.frames[0].x, ego.frames[0].y, ego.frames[0].z);
    camera.up.set(0, 0, 1);
    camera.position.set(startPos.x, startPos.y - 80, startPos.z + 80);
    controls.target.copy(startPos);
    controls.update();

    // ── RAF loop ──────────────────────────────────────────────────────────
    let rafId: number;
    const animate = () => {
      rafId = requestAnimationFrame(animate);

      const ego_ = egoRef.current;
      const lidar_ = lidarRef.current;
      const idx = frameIdxRef.current;

      if (ego_ && carRef.current) {
        const f = ego_.frames[idx];
        const pos = p3(f.x, f.y, f.z);
        const q   = toThreeQ(f.qx, f.qy, f.qz, f.qw);

        const isCar = displayModeRef.current === "car";
        carRef.current.position.copy(pos);
        carRef.current.quaternion.copy(q);
        carRef.current.visible = isCar;

        if (arrowRef.current) {
          arrowRef.current.position.copy(pos);
          arrowRef.current.quaternion.copy(q);
          arrowRef.current.visible = !isCar;
        }
      }

      // ── follow-cam ───────────────────────────────────────────────────────
      if (rotateFollowRef.current && carRef.current) {
        controls.enabled = false;
        const carPos = carRef.current.position;
        orbitAngleRef.current += 0.008;
        const a = orbitAngleRef.current;
        const radius = 20, height = 10;
        camera.position.set(
          carPos.x + radius * Math.cos(a),
          carPos.y + radius * Math.sin(a),
          carPos.z + height,
        );
        camera.up.set(0, 0, 1);
        camera.lookAt(carPos);
      } else {
        controls.enabled = true;
        controls.update();
      }

      if (lidar_ && ego_) {
        const f = ego_.frames[idx];
        const pose = p3(f.x, f.y, f.z);
        const poseQ = toThreeQ(f.qx, f.qy, f.qz, f.qw);

        // find nearest spin
        let bestSpin = lidar_.spins[0];
        let bestDist = Math.abs(lidar_.spins[0].ef - idx);
        for (const spin of lidar_.spins) {
          const d = Math.abs(spin.ef - idx);
          if (d < bestDist) { bestDist = d; bestSpin = spin; }
        }

        // transform spin points to world space
        const rawPts = bestSpin.pts;
        const n = rawPts.length / 3;
        const pos3 = new Float32Array(n * 3);
        const col3 = new Float32Array(n * 3);
        const tmp = new THREE.Vector3();
        const rawRgb = bestSpin.rgb;
        for (let i = 0; i < n; i++) {
          const vx = rawPts[i * 3], vy = rawPts[i * 3 + 1], vz = rawPts[i * 3 + 2];
          tmp.set(vx, vy, vz).applyQuaternion(poseQ).add(pose);
          pos3[i * 3] = tmp.x; pos3[i * 3 + 1] = tmp.y; pos3[i * 3 + 2] = tmp.z;
          // use camera RGB if available and non-zero, else fall back to height colour
          const ri = i * 3;
          if (rawRgb && (rawRgb[ri] | rawRgb[ri + 1] | rawRgb[ri + 2])) {
            col3[ri]     = rawRgb[ri]     / 255;
            col3[ri + 1] = rawRgb[ri + 1] / 255;
            col3[ri + 2] = rawRgb[ri + 2] / 255;
          } else {
            const [r, g, b] = heightRGB(vz);
            col3[ri] = r; col3[ri + 1] = g; col3[ri + 2] = b;
          }
        }

        // current-spin display
        if (showLidarRef.current && !keepRef.current && currentLidarRef.current) {
          const geo = currentLidarRef.current.geometry;
          geo.setAttribute("position", new THREE.BufferAttribute(pos3, 3));
          geo.setAttribute("color", new THREE.BufferAttribute(col3, 3));
          geo.computeBoundingSphere();
        } else if (currentLidarRef.current) {
          currentLidarRef.current.geometry.setAttribute("position",
            new THREE.BufferAttribute(new Float32Array(0), 3));
        }

        // accumulated display
        if (showLidarRef.current && keepRef.current && accLidarRef.current) {
          if (!accSpins.current.has(bestSpin.t)) {
            accSpins.current.add(bestSpin.t);
            for (let i = 0; i < n; i++) {
              accPositions.current.push(pos3[i * 3], pos3[i * 3 + 1], pos3[i * 3 + 2]);
              accColors.current.push(col3[i * 3], col3[i * 3 + 1], col3[i * 3 + 2]);
            }
            const geo = accLidarRef.current.geometry;
            geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(accPositions.current), 3));
            geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(accColors.current), 3));
            geo.computeBoundingSphere();
          }
        } else if (!keepRef.current && accLidarRef.current) {
          // hide when keep is off
          accLidarRef.current.geometry.setAttribute("position",
            new THREE.BufferAttribute(new Float32Array(0), 3));
        }
      }

      renderer.render(scene, camera);
    };
    animate();

    // resize
    const onResize = () => {
      const w = mount.clientWidth || 720;
      camera.aspect = w / H;
      camera.updateProjectionMatrix();
      renderer.setSize(w, H);
    };
    window.addEventListener("resize", onResize);

    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener("resize", onResize);
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, [ego]); // eslint-disable-line react-hooks/exhaustive-deps

  const cur = ego?.frames[frameIdx];

  return (
    <div className="flex flex-col items-center min-h-screen bg-slate-950 text-slate-100 p-6 gap-4">
      <h1 className="text-2xl font-bold tracking-tight">NVIDIA Physical AI AV Visualization</h1>

      <div className="flex flex-row gap-4 w-full max-w-[1400px] items-stretch">
        {/* 3D canvas */}
        <div
          ref={mountRef}
          className="flex-1 min-w-0 rounded-2xl border border-slate-800 shadow-2xl overflow-hidden"
          style={{ height: 520 }}
        />

        {/* camera column */}
        <div className="flex flex-col gap-2 w-[320px] shrink-0">
          {camIdx ? (
            [
              { ref: frontVideoRef, label: "Front", entry: camIdx.cameras.front  },
              { ref: leftVideoRef,  label: "Left",  entry: camIdx.cameras.left   },
              { ref: rightVideoRef, label: "Right", entry: camIdx.cameras.right  },
            ].map(({ ref, label, entry }) => (
              <div key={label} className="relative flex-1 rounded-lg overflow-hidden border border-slate-700 bg-slate-900">
                <video ref={ref} src={`/${entry.file}`} preload="metadata" muted playsInline
                       className="w-full h-full object-cover" crossOrigin="anonymous" />
                <span className="absolute top-1 left-2 text-xs text-slate-400 font-mono">{label}</span>
              </div>
            ))
          ) : (
            <div className="flex-1 rounded-lg border border-slate-800 bg-slate-900 flex items-center justify-center text-slate-600 text-xs font-mono">
              No camera data<br/>Run export_camera.py
            </div>
          )}
        </div>
      </div>

      {/* controls */}
      <div className="flex items-center gap-4 w-full max-w-[1400px]">
        <button
          onClick={() => {
            if (ego && frameIdx >= ego.frames.length - 1) setFrameIdx(0);
            setPlaying(p => !p);
          }}
          className="px-5 py-2 rounded-lg bg-yellow-400 text-slate-900 font-bold text-sm hover:bg-yellow-300 transition-colors shrink-0"
        >
          {playing ? "Pause" : "Play"}
        </button>

        <input
          type="range" min={0} max={(ego?.frames.length ?? 1) - 1} value={frameIdx}
          onChange={e => { setPlaying(false); setFrameIdx(Number(e.target.value)); }}
          className="flex-1 accent-yellow-400"
        />

        <button
          onClick={() => setRotateFollow(prev => {
            if (!prev) orbitAngleRef.current = 0;
            else if (controlsRef.current) controlsRef.current.enabled = true;
            return !prev;
          })}
          className={`px-3 py-2 rounded-lg text-sm font-semibold shrink-0 transition-colors ${rotateFollow ? "bg-violet-400 text-slate-900" : "bg-slate-700 text-slate-300"}`}
          title="Rotate follow cam (R)"
        >
          Follow [R]
        </button>

        <button
          onClick={() => setShowLidar(s => !s)}
          className={`px-3 py-2 rounded-lg text-sm font-semibold shrink-0 transition-colors ${showLidar ? "bg-cyan-500 text-slate-900" : "bg-slate-700 text-slate-300"}`}
        >
          LiDAR
        </button>

        <button
          onClick={() => setKeepPoints(prev => {
            if (prev) {
              accPositions.current = []; accColors.current = []; accSpins.current.clear();
              if (accLidarRef.current) {
                accLidarRef.current.geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(0), 3));
                accLidarRef.current.geometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(0), 3));
              }
            }
            return !prev;
          })}
          className={`px-3 py-2 rounded-lg text-sm font-semibold shrink-0 transition-colors ${keepPoints ? "bg-emerald-400 text-slate-900" : "bg-slate-700 text-slate-300"}`}
          title="Keep points (K)"
        >
          Keep [K]
        </button>
      </div>

      {/* stats */}
      {cur && (
        <div className="grid grid-cols-5 gap-2 w-full max-w-[1400px] text-xs font-mono">
          {(
            [
              ["Accel X", `${cur.ax.toFixed(3)} m/s\u00b2`],
              ["Accel Y", `${cur.ay.toFixed(3)} m/s\u00b2`],
              ["Accel Z", `${cur.az.toFixed(3)} m/s\u00b2`],
              ["Curvature", cur.curvature.toExponential(2)],
              ["Speed", `${speed(cur.vx, cur.vy, cur.vz).toFixed(2)} m/s`],
            ] as [string, string][]
          ).map(([label, val]) => (
            <div key={label} className="bg-slate-900 rounded-lg p-3 border border-slate-800">
              <div className="text-slate-500 mb-1">{label}</div>
              <div className="text-slate-100 font-bold">{val}</div>
            </div>
          ))}
        </div>
      )}

      <p className="text-xs text-slate-600 font-mono">drag to orbit · scroll to zoom · right-drag to pan · <kbd className="bg-slate-800 text-slate-400 px-1 rounded">P</kbd> play/pause · <kbd className="bg-slate-800 text-slate-400 px-1 rounded">D</kbd> toggle car / arrow · <kbd className="bg-slate-800 text-slate-400 px-1 rounded">R</kbd> follow cam · <kbd className="bg-slate-800 text-slate-400 px-1 rounded">K</kbd> keep points</p>
    </div>
  );
}
