"""Extract an optimized, lightweight demo GIF from an annotated video run.
"""

import os
from pathlib import Path
import cv2
from PIL import Image

def create_demo_gif(
    video_path: str = "runs/inference/video_1_annotated.mp4",
    output_path: str = "docs/assets/demo_inference.gif",
    target_width: int = 640,
    start_sec: float = 1.0,
    duration_sec: float = 4.0,
    fps: int = 10,
):
    video_file = Path(video_path)
    if not video_file.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_file))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    start_frame = int(start_sec * orig_fps)
    end_frame = min(int((start_sec + duration_sec) * orig_fps), total_frames)
    frame_step = max(1, int(round(orig_fps / fps)))

    print(f"Reading '{video_path}':")
    print(f"  Orig FPS: {orig_fps:.1f}, Frames: {total_frames}")
    print(f"  Extracting range: frames {start_frame} -> {end_frame} (step {frame_step})")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    current_idx = start_frame

    while current_idx < end_frame:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if (current_idx - start_frame) % frame_step == 0:
            h, w = frame.shape[:2]
            target_height = int(h * (target_width / w))
            resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            # Quantize with web/adaptive palette to reduce GIF file size
            pil_img = Image.fromarray(rgb).quantize(colors=64, method=Image.Quantize.MEDIANCUT)
            frames.append(pil_img)

        current_idx += 1

    cap.release()

    if not frames:
        raise RuntimeError("No frames extracted!")

    frame_duration_ms = int(1000 / fps)
    print(f"Saving {len(frames)} frames to '{output_path}' ({frame_duration_ms}ms/frame)...")
    
    frames[0].save(
        out_file,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=True,
    )

    size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"Done! Created '{output_path}' ({size_mb:.2f} MB)")

if __name__ == "__main__":
    create_demo_gif()
