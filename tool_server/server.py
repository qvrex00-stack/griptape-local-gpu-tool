"""
Griptape Tool Server - server.py
Depth Estimation + Pose Extraction + Pose Smoothing + Video Blending

Port: 8089
Usage: python server.py
"""

import base64
import io
import logging
import os
import threading
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tool_server")

_depth_pipeline = None
_depth_last_used = 0
_pose_estimator = None
_pose_last_used = 0
_cache_lock = threading.Lock()

UNLOAD_TIMEOUT = 300


def unload_idle_models():
    while True:
        time.sleep(60)
        now = time.time()
        with _cache_lock:
            global _depth_pipeline, _depth_last_used
            global _pose_estimator, _pose_last_used
            try:
                import torch
                if _depth_pipeline is not None and now - _depth_last_used > UNLOAD_TIMEOUT:
                    _depth_pipeline.model.to("cpu")
                    torch.cuda.empty_cache()
                    logger.info("[Auto-unload] Depth model offloaded to CPU")
            except Exception:
                pass
            try:
                import torch
                if _pose_estimator is not None and now - _pose_last_used > UNLOAD_TIMEOUT and _pose_last_used > 0:
                    torch.cuda.empty_cache()
                    logger.info("[Auto-unload] Pose model offloaded to CPU")
            except Exception:
                pass


# ── Request / Response Models ─────────────────────────────────────────────────

class DepthRequest(BaseModel):
    image_b64: str
    model_size: str = "small"
    colorize: bool = True

class PoseRequest(BaseModel):
    image_b64: str
    detector: str = "openpose"
    include_body: bool = True
    include_hand: bool = False
    include_face: bool = False

class DepthVideoRequest(BaseModel):
    video_path: str
    model_size: str = "small"
    colorize: bool = True
    fps: float = 0

class PoseVideoRequest(BaseModel):
    video_path: str
    include_body: bool = True
    include_hand: bool = False
    include_face: bool = False
    fps: float = 0

class PoseSmoothVideoRequest(BaseModel):
    video_path: str
    include_body: bool = True
    include_hand: bool = False
    include_face: bool = False
    fps: float = 0
    smooth_method: str = "gaussian"
    smooth_window: int = 5
    jump_threshold: float = 0.05
    confidence_threshold: float = 0.3

class PoseSmoothVideoResponse(BaseModel):
    video_path: str
    video_ext: str = "mp4"
    frame_count: int
    fps: float
    width: int
    height: int
    smoothed_keypoints_count: int = 0
    info: str = ""

class BlendVideoRequest(BaseModel):
    depth_video_path: str
    pose_video_path: str
    alpha: float = 0.5
    fps: float = 0

class BlendVideoResponse(BaseModel):
    video_path: str
    video_ext: str = "mp4"
    frame_count: int
    fps: float
    width: int
    height: int
    alpha: float
    info: str = ""

class ToolResponse(BaseModel):
    image_b64: str
    width: int
    height: int
    info: str = ""

class VideoResponse(BaseModel):
    video_path: str = ""
    video_ext: str = "mp4"
    frame_count: int
    fps: float
    width: int
    height: int
    info: str = ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def b64_to_pil(b64_str):
    from PIL import Image
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))

def pil_to_b64(pil_img, fmt="PNG"):
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()

def get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def extract_frames(video_path: str):
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames, fps

def frames_to_video(frames, output_path: str, fps: float):
    import cv2
    import numpy as np
    import subprocess
    import shutil

    if not frames:
        raise ValueError("No frames to encode")

    w, h = frames[0].size
    tmp_path = output_path + ".tmp.mp4"
    out = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        out.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    out.release()

    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path,
             "-vcodec", "libx264", "-pix_fmt", "yuv420p",
             "-preset", "fast", "-crf", "23", output_path],
            capture_output=True, timeout=300
        )
        os.remove(tmp_path)
        if result.returncode != 0:
            logger.warning(f"ffmpeg H264 failed: {result.stderr.decode()}")
            shutil.move(tmp_path, output_path)
    except Exception as e:
        logger.warning(f"ffmpeg unavailable ({e}), using mp4v")
        if os.path.exists(tmp_path):
            shutil.move(tmp_path, output_path)

def load_depth_model(model_size: str):
    global _depth_pipeline, _depth_last_used
    import torch
    from transformers import pipeline as hf_pipeline
    model_map = {
        "small": "depth-anything/Depth-Anything-V2-Small-hf",
        "base":  "depth-anything/Depth-Anything-V2-Base-hf",
        "large": "depth-anything/Depth-Anything-V2-Large-hf",
    }
    model_id = model_map.get(model_size, model_map["small"])
    device = get_device()
    with _cache_lock:
        if _depth_pipeline is None:
            logger.info(f"Loading Depth Anything V2 ({model_size}) on {device}...")
            _depth_pipeline = hf_pipeline(
                task="depth-estimation", model=model_id,
                device=0 if device == "cuda" else -1,
                torch_dtype=torch.float32,
            )
            logger.info("Depth model loaded.")
        else:
            try:
                if device == "cuda":
                    _depth_pipeline.model.to(device)
            except Exception:
                pass
        _depth_last_used = time.time()
    return _depth_pipeline

def load_pose_model():
    global _pose_estimator, _pose_last_used
    with _cache_lock:
        if _pose_estimator is None:
            logger.info("Loading OpenPose estimator...")
            from controlnet_aux import OpenposeDetector
            _pose_estimator = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
            logger.info("OpenPose loaded.")
        _pose_last_used = time.time()
    return _pose_estimator

def process_depth_frame(pil_image, pipeline, colorize: bool):
    import numpy as np
    import cv2
    from PIL import Image
    depth_map = pipeline(pil_image)["depth"]
    if colorize:
        depth_np = np.array(depth_map)
        depth_norm = ((depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8) * 255).astype(np.uint8)
        return Image.fromarray(cv2.cvtColor(cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)).resize(pil_image.size, Image.Resampling.LANCZOS)
    return depth_map.convert("L").resize(pil_image.size, Image.Resampling.LANCZOS)

def process_pose_frame(pil_image, estimator, include_body, include_hand, include_face):
    import numpy as np
    from PIL import Image
    output = estimator(pil_image, include_body=include_body, include_hand=include_hand, include_face=include_face)
    if not isinstance(output, Image.Image):
        output = Image.fromarray(np.array(output))
    return output.resize(pil_image.size, Image.Resampling.LANCZOS)

def smooth_keypoint_sequence(coords_seq, method="gaussian", window=5):
    import numpy as np
    n = len(coords_seq)
    arr = np.array(coords_seq, dtype=float)
    valid_mask = ~np.isnan(arr)
    if valid_mask.sum() < 2:
        arr = np.where(np.isnan(arr), 0.0, arr)
    else:
        indices = np.arange(n)
        arr = np.interp(indices, indices[valid_mask], arr[valid_mask])
    w = max(3, min(window if window % 2 == 1 else window + 1, n))
    try:
        if method == "savgol":
            from scipy.signal import savgol_filter
            return savgol_filter(arr, w, min(3, w - 1)).tolist()
        else:
            from scipy.ndimage import gaussian_filter1d
            return gaussian_filter1d(arr, sigma=w / 3.0).tolist()
    except ImportError:
        kernel = np.ones(w) / w
        return np.convolve(np.pad(arr, w // 2, mode="edge"), kernel, mode="valid")[:n].tolist()

def detect_jump_frames(coords_seq, frame_size, threshold):
    import numpy as np
    arr = np.array(coords_seq, dtype=float)
    limit = threshold * frame_size
    return {i for i in range(1, len(arr) - 1)
            if abs(arr[i] - arr[i-1]) > limit or abs(arr[i] - arr[i+1]) > limit}


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        import torch
        logger.info("=" * 50)
        logger.info("Griptape Tool Server Starting... (port 8089)")
        logger.info(f"torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Auto-unload timeout: {UNLOAD_TIMEOUT}s")
        logger.info("=" * 50)
    except Exception as e:
        logger.warning(f"torch info unavailable: {e}")
    threading.Thread(target=unload_idle_models, daemon=True).start()
    yield
    logger.info("Tool server shutting down...")


app = FastAPI(
    title="Griptape Tool Server",
    description="Depth Estimation + Pose Extraction + Pose Smoothing + Video Blending",
    version="1.3.0",
    lifespan=lifespan
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    try:
        import torch
        return {
            "status": "ok", "port": 8089,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "depth_loaded": _depth_pipeline is not None,
            "pose_loaded": _pose_estimator is not None,
            "unload_timeout_sec": UNLOAD_TIMEOUT,
        }
    except Exception:
        return {"status": "ok", "port": 8089, "torch_available": False}


@app.post("/depth", response_model=ToolResponse)
def depth(req: DepthRequest):
    pil_image = b64_to_pil(req.image_b64).convert("RGB")
    try:
        output = process_depth_frame(pil_image, load_depth_model(req.model_size), req.colorize)
        return ToolResponse(image_b64=pil_to_b64(output), width=output.width, height=output.height, info=f"model={req.model_size}")
    except Exception as e:
        logger.error(f"Depth failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pose", response_model=ToolResponse)
def pose(req: PoseRequest):
    pil_image = b64_to_pil(req.image_b64).convert("RGB")
    try:
        output = process_pose_frame(pil_image, load_pose_model(), req.include_body, req.include_hand, req.include_face)
        return ToolResponse(image_b64=pil_to_b64(output), width=output.width, height=output.height, info="detector=openpose")
    except Exception as e:
        logger.error(f"Pose failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/depth_video", response_model=VideoResponse)
def depth_video(req: DepthVideoRequest):
    try:
        if not os.path.exists(req.video_path):
            raise ValueError(f"Video not found: {req.video_path}")
        frames, orig_fps = extract_frames(req.video_path)
        fps = req.fps if req.fps > 0 else orig_fps
        logger.info(f"Depth video: {len(frames)} frames, {fps:.2f} fps")
        pipeline = load_depth_model(req.model_size)
        processed = []
        for i, frame in enumerate(frames):
            processed.append(process_depth_frame(frame.convert("RGB"), pipeline, req.colorize))
            if (i + 1) % 10 == 0:
                logger.info(f"  Depth: {i+1}/{len(frames)}")
        out_path = os.path.join(os.path.dirname(req.video_path),
                                os.path.splitext(os.path.basename(req.video_path))[0] + "_depth.mp4")
        frames_to_video(processed, out_path, fps)
        w, h = processed[0].size
        return VideoResponse(video_path=out_path, video_ext="mp4", frame_count=len(processed), fps=fps, width=w, height=h, info=f"model={req.model_size}")
    except Exception as e:
        logger.error(f"Depth video failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pose_video", response_model=VideoResponse)
def pose_video(req: PoseVideoRequest):
    try:
        if not os.path.exists(req.video_path):
            raise ValueError(f"Video not found: {req.video_path}")
        frames, orig_fps = extract_frames(req.video_path)
        fps = req.fps if req.fps > 0 else orig_fps
        logger.info(f"Pose video: {len(frames)} frames, {fps:.2f} fps")
        estimator = load_pose_model()
        processed = []
        for i, frame in enumerate(frames):
            processed.append(process_pose_frame(frame.convert("RGB"), estimator, req.include_body, req.include_hand, req.include_face))
            if (i + 1) % 10 == 0:
                logger.info(f"  Pose: {i+1}/{len(frames)}")
        out_path = os.path.join(os.path.dirname(req.video_path),
                                os.path.splitext(os.path.basename(req.video_path))[0] + "_pose.mp4")
        frames_to_video(processed, out_path, fps)
        w, h = processed[0].size
        return VideoResponse(video_path=out_path, video_ext="mp4", frame_count=len(processed), fps=fps, width=w, height=h)
    except Exception as e:
        logger.error(f"Pose video failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pose_smooth_video", response_model=PoseSmoothVideoResponse)
def pose_smooth_video(req: PoseSmoothVideoRequest):
    import numpy as np
    from controlnet_aux.open_pose import PoseResult, BodyResult, draw_poses
    from controlnet_aux.open_pose.body import Keypoint
    from PIL import Image

    try:
        if not os.path.exists(req.video_path):
            raise ValueError(f"Video not found: {req.video_path}")
        frames, orig_fps = extract_frames(req.video_path)
        fps = req.fps if req.fps > 0 else orig_fps
        n, w, h = len(frames), frames[0].size[0], frames[0].size[1]
        logger.info(f"Pose smooth: {n} frames, {fps:.2f} fps, {w}x{h}")
        estimator = load_pose_model()

        # Step 1: detect
        all_poses = []
        for i, frame in enumerate(frames):
            try:
                poses = estimator.detect_poses(np.array(frame.convert("RGB")),
                                               include_hand=req.include_hand, include_face=req.include_face)
            except Exception:
                poses = []
            all_poses.append(poses)
            if (i + 1) % 10 == 0:
                logger.info(f"  Detected: {i+1}/{n}")

        max_people = max((len(p) for p in all_poses), default=0)
        if max_people == 0:
            raise ValueError("No poses detected.")

        # Step 2: smooth
        N_KP = 18
        kp_x = np.full((n, max_people, N_KP), np.nan)
        kp_y = np.full((n, max_people, N_KP), np.nan)
        for fi, poses in enumerate(all_poses):
            for pi, pose in enumerate(poses[:max_people]):
                for ki, kp in enumerate(pose.body.keypoints[:N_KP]):
                    if kp is not None and kp.x > 1e-6 and kp.y > 1e-6:
                        kp_x[fi, pi, ki] = kp.x
                        kp_y[fi, pi, ki] = kp.y

        smoothed_x, smoothed_y = kp_x.copy(), kp_y.copy()
        smoothed_count = 0
        for pi in range(max_people):
            for ki in range(N_KP):
                jx = detect_jump_frames(kp_x[:, pi, ki].tolist(), 1.0, req.jump_threshold)
                jy = detect_jump_frames(kp_y[:, pi, ki].tolist(), 1.0, req.jump_threshold)
                for fi in jx | jy:
                    smoothed_x[fi, pi, ki] = np.nan
                    smoothed_y[fi, pi, ki] = np.nan
                smoothed_count += len(jx | jy)
                smoothed_x[:, pi, ki] = smooth_keypoint_sequence(smoothed_x[:, pi, ki].tolist(), req.smooth_method, req.smooth_window)
                smoothed_y[:, pi, ki] = smooth_keypoint_sequence(smoothed_y[:, pi, ki].tolist(), req.smooth_method, req.smooth_window)

        # Step 3: render
        rendered = []
        for fi in range(n):
            rebuilt = []
            for pi, orig in enumerate(all_poses[fi][:max_people]):
                kps = [None if (np.isnan(smoothed_x[fi, pi, ki]) or np.isnan(smoothed_y[fi, pi, ki]))
                       else Keypoint(x=float(smoothed_x[fi, pi, ki]), y=float(smoothed_y[fi, pi, ki]))
                       for ki in range(N_KP)]
                rebuilt.append(PoseResult(
                    body=BodyResult(keypoints=kps, total_score=orig.body.total_score, total_parts=orig.body.total_parts),
                    left_hand=orig.left_hand, right_hand=orig.right_hand, face=orig.face,
                ))
            rendered.append(Image.fromarray(draw_poses(rebuilt, h, w,
                                                        draw_body=req.include_body,
                                                        draw_hand=req.include_hand,
                                                        draw_face=req.include_face)))
            if (fi + 1) % 10 == 0:
                logger.info(f"  Rendered: {fi+1}/{n}")

        out_path = os.path.join(os.path.dirname(req.video_path),
                                os.path.splitext(os.path.basename(req.video_path))[0] + "_pose_smooth.mp4")
        frames_to_video(rendered, out_path, fps)
        logger.info(f"Pose smooth done: {n} frames → {out_path}")
        return PoseSmoothVideoResponse(
            video_path=out_path, video_ext="mp4", frame_count=n, fps=fps,
            width=w, height=h, smoothed_keypoints_count=smoothed_count,
            info=f"method={req.smooth_method}, window={req.smooth_window}, smoothed={smoothed_count}",
        )
    except Exception as e:
        logger.error(f"Pose smooth video failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/blend_video", response_model=BlendVideoResponse)
def blend_video(req: BlendVideoRequest):
    import numpy as np
    from PIL import Image

    try:
        if not os.path.exists(req.depth_video_path):
            raise ValueError(f"Depth video not found: {req.depth_video_path}")
        if not os.path.exists(req.pose_video_path):
            raise ValueError(f"Pose video not found: {req.pose_video_path}")

        alpha = max(0.0, min(1.0, req.alpha))
        depth_frames, depth_fps = extract_frames(req.depth_video_path)
        pose_frames, _          = extract_frames(req.pose_video_path)
        fps = req.fps if req.fps > 0 else depth_fps
        n = min(len(depth_frames), len(pose_frames))
        w, h = depth_frames[0].size
        logger.info(f"Blend video: {n} frames, alpha={alpha}, {w}x{h}")

        blended = []
        for i in range(n):
            d = np.array(depth_frames[i].convert("RGB"), dtype=np.float32)
            p = np.array(pose_frames[i].convert("RGB").resize((w, h), Image.Resampling.LANCZOS), dtype=np.float32)
            blended.append(Image.fromarray(np.clip(d * (1 - alpha) + p * alpha, 0, 255).astype(np.uint8)))
            if (i + 1) % 10 == 0:
                logger.info(f"  Blended: {i+1}/{n}")

        depth_name = os.path.splitext(os.path.basename(req.depth_video_path))[0]
        out_path = os.path.join(os.path.dirname(req.depth_video_path),
                                f"{depth_name}_blend{int(alpha*100)}.mp4")
        frames_to_video(blended, out_path, fps)
        logger.info(f"Blend done: {n} frames → {out_path}")
        return BlendVideoResponse(
            video_path=out_path, video_ext="mp4", frame_count=n,
            fps=fps, width=w, height=h, alpha=alpha,
            info=f"alpha={alpha}, depth={len(depth_frames)}, pose={len(pose_frames)}",
        )
    except Exception as e:
        logger.error(f"Blend video failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8089, log_level="info")
