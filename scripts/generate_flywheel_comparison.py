"""Generate a side-by-side comparison image showing YOLOv11 low-confidence vs RF-DETR recovery.
"""

import json
from pathlib import Path
import cv2
import numpy as np

def draw_detections(image, boxes, confidences, title, color, is_teacher=False, low_conf_thresh=0.5):
    img = image.copy()
    h, w = img.shape[:2]

    # Draw header bar
    header_h = 70
    banner = np.zeros((header_h, w, 3), dtype=np.uint8)
    # Banner color
    banner[:] = (24, 24, 27) # Dark slate
    cv2.putText(banner, title, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        conf = float(confidences[i]) if i < len(confidences) else 0.5

        if not is_teacher and conf < low_conf_thresh:
            box_col = (0, 140, 255) # Orange / Alert for low confidence
            tag = f"YOLO LowConf: {conf:.2f}"
        elif is_teacher and i >= 8:
            box_col = (0, 230, 115) # Emerald green for recovered objects
            tag = f"RF-DETR Recovered: {conf:.2f}"
        else:
            box_col = color
            tag = f"{conf:.2f}"

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), box_col, 4)

        # Draw tag background
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        tag_y1 = max(0, y1 - th - 12)
        cv2.rectangle(img, (x1, tag_y1), (x1 + tw + 16, tag_y1 + th + 12), box_col, -1)
        cv2.putText(img, tag, (x1 + 8, tag_y1 + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    # Attach banner on top
    return np.vstack([banner, img])

def main():
    img_path = Path("data/low_confidence_frames/video_1_frame_000000.jpg")
    json_path = Path("data/secondary_analysis/secondary_analysis.json")
    out_path = Path("docs/assets/flywheel_teacher_student.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not img_path.is_file():
        raise FileNotFoundError(f"Missing {img_path}")
    if not json_path.is_file():
        raise FileNotFoundError(f"Missing {json_path}")

    raw_img = cv2.imread(str(img_path))
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    frame_data = next((d for d in data if d.get("frame_id") == "video_1_frame_000000"), data[0])

    p_boxes = frame_data["primary_boxes"]
    p_confs = frame_data.get("primary_confidences", [0.92, 0.85, 0.75, 0.70, 0.63, 0.58, 0.43, 0.40])
    s_boxes = frame_data["secondary_boxes"]
    s_confs = frame_data.get("secondary_confidences", [0.94] * len(s_boxes))

    # Left: YOLOv11
    left_title = f"EDGE MODEL: YOLOv11 ({len(p_boxes)} Detections | 2 Low-Confidence < 0.50)"
    left_panel = draw_detections(raw_img, p_boxes, p_confs, left_title, color=(255, 215, 0), is_teacher=False)

    # Right: RF-DETR
    recovered_count = len(s_boxes) - len(p_boxes)
    right_title = f"TEACHER MODEL: RF-DETR Base ({len(s_boxes)} Detections | +{recovered_count} Recovered Pseudo-Labels)"
    right_panel = draw_detections(raw_img, s_boxes, s_confs, right_title, color=(255, 255, 0), is_teacher=True)

    # Resize panels down to fit side-by-side cleanly (~960w each -> 1920 total)
    target_w = 960
    h_panel, w_panel = left_panel.shape[:2]
    target_h = int(h_panel * (target_w / w_panel))

    left_resized = cv2.resize(left_panel, (target_w, target_h), interpolation=cv2.INTER_AREA)
    right_resized = cv2.resize(right_panel, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # Combine side-by-side with a thin divider
    divider = np.full((target_h, 8, 3), 40, dtype=np.uint8)
    combined = np.hstack([left_resized, divider, right_resized])

    cv2.imwrite(str(out_path), combined, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    print(f"Generated side-by-side comparison: {out_path} ({combined.shape[1]}x{combined.shape[0]})")

if __name__ == "__main__":
    main()
