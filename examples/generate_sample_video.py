#!/usr/bin/env python3
"""Generate a synthetic test video with moving rectangles simulating people.

Creates a 10-second video at 30fps with 2-4 "person" blobs walking across
the frame. No external data needed — runs purely with OpenCV + NumPy.

Usage:
    python examples/generate_sample_video.py
    python examples/generate_sample_video.py --output data/sample_videos/custom.mp4 --duration 20
"""

import os
import argparse
import numpy as np
import cv2


def generate_video(output_path: str, width: int = 640, height: int = 480,
                   fps: int = 30, duration: int = 10, num_people: int = 3):
    """Generate a synthetic video with moving person-like rectangles."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    total_frames = fps * duration

    # Initialize "people" with random positions and velocities
    rng = np.random.RandomState(42)
    people = []
    for i in range(num_people):
        pw, ph = rng.randint(40, 70), rng.randint(100, 180)
        x = rng.randint(0, width - pw)
        y = rng.randint(height // 3, height - ph)
        vx = rng.choice([-3, -2, -1, 1, 2, 3])
        vy = rng.choice([-1, 0, 0, 0, 1])
        color = tuple(int(c) for c in rng.randint(60, 200, size=3))
        # Stagger entry: some people appear later
        entry_frame = rng.randint(0, total_frames // 3)
        exit_frame = rng.randint(total_frames * 2 // 3, total_frames)
        people.append({
            'x': x, 'y': y, 'w': pw, 'h': ph,
            'vx': vx, 'vy': vy, 'color': color,
            'entry': entry_frame, 'exit': exit_frame,
        })

    for frame_idx in range(total_frames):
        # Background: store-like floor with gradient
        frame = np.full((height, width, 3), (220, 215, 210), dtype=np.uint8)
        # Floor line
        cv2.line(frame, (0, height * 2 // 3), (width, height * 2 // 3), (180, 175, 170), 1)

        for p in people:
            if frame_idx < p['entry'] or frame_idx > p['exit']:
                continue

            # Draw person rectangle with a "head" circle
            x, y, w, h = int(p['x']), int(p['y']), p['w'], p['h']
            # Body
            cv2.rectangle(frame, (x, y + h // 6), (x + w, y + h), p['color'], -1)
            # Head
            cx, cy = x + w // 2, y + h // 8
            cv2.circle(frame, (cx, cy), w // 3, p['color'], -1)

            # Move
            p['x'] += p['vx']
            p['y'] += p['vy']
            # Bounce off walls
            if p['x'] <= 0 or p['x'] + p['w'] >= width:
                p['vx'] *= -1
            if p['y'] <= 0 or p['y'] + p['h'] >= height:
                p['vy'] *= -1
            p['x'] = np.clip(p['x'], 0, width - p['w'])
            p['y'] = np.clip(p['y'], 0, height - p['h'])

        # Timestamp overlay
        t = frame_idx / fps
        cv2.putText(frame, f"t={t:.1f}s  frame={frame_idx}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)

        writer.write(frame)

    writer.release()
    print(f"Generated {total_frames} frames ({duration}s @ {fps}fps)")
    print(f"Saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic test video")
    parser.add_argument('--output', type=str, default='data/sample_videos/test_scene.mp4')
    parser.add_argument('--duration', type=int, default=10, help='Duration in seconds')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--num-people', type=int, default=3)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    args = parser.parse_args()

    generate_video(
        output_path=args.output,
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration=args.duration,
        num_people=args.num_people,
    )


if __name__ == '__main__':
    main()
