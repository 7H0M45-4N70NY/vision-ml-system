import os
import json
import glob
import tempfile
import zipfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import supervision as sv

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from ..logging import get_logger

logger = get_logger(__name__)


class AutoLabeler:
    """Collects low-confidence frames labeled by RF-DETR for active learning.

    Active learning approach: focus on frames where model struggles (low-confidence).
    These are saved by DualDetector and labeled by RF-DETR fallback detector.

    Single label source:
    - load_dual_detector_frames() — from DualDetector's saved low-confidence
      frames (data/low_confidence_frames/*.json), already labeled by RF-DETR
    """

    def __init__(self, config: dict):
        label_cfg = config.get('labeling', {})
        self.enabled = label_cfg.get('enabled', False)
        self.provider = label_cfg.get('provider', 'local')
        self.roboflow_api_key = label_cfg.get('roboflow_api_key') or os.environ.get('ROBOFLOW_API_KEY')
        self.roboflow_workspace = label_cfg.get('roboflow_workspace')
        self.roboflow_project = label_cfg.get('roboflow_project')
        self.pending_labels = []

    def collect(self, image, detections, image_id: str = None):
        """No-op: high-confidence collection disabled. Only low-confidence frames via load_dual_detector_frames()."""
        pass

    def load_dual_detector_frames(self, frame_dir: str = 'data/low_confidence_frames'):
        """Import pseudo-labels saved by DualDetector for offline training."""
        if not os.path.isdir(frame_dir):
            logger.warning(f"No saved frames at {frame_dir}")
            return 0
        label_files = sorted(glob.glob(os.path.join(frame_dir, '*.json')))
        for lf in label_files:
            with open(lf, 'r') as f:
                label = json.load(f)
            label['image_id'] = os.path.splitext(os.path.basename(lf))[0]
            label['image_path'] = lf.replace('.json', '.jpg')
            self.pending_labels.append(label)
        logger.info(f"Loaded {len(label_files)} pseudo-labels from {frame_dir}")
        return len(label_files)

    def flush(
        self,
        output_dir: str = 'data/auto_labeled',
        workers: int = 5,
        compress: bool = True,
        max_dim: int = 1280,
        use_zip: bool = True,
        batch_name: str = None,
    ):
        if self.provider == 'roboflow' and self.roboflow_api_key:
            self._upload_roboflow(
                workers=workers,
                compress=compress,
                max_dim=max_dim,
                use_zip=use_zip,
                batch_name=batch_name,
            )
        else:
            self._export_local(output_dir)
        self.pending_labels.clear()

    def _export_local(self, output_dir: str):
        if not self.pending_labels:
            logger.debug("No pending labels to export")
            return
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, 'auto_labels.json')
        with open(out_path, 'w') as f:
            json.dump(self.pending_labels, f, indent=2)
        logger.info(f"Exported {len(self.pending_labels)} labels to {out_path}")

    def export_zip(self, zip_path: str = 'data/auto_labeled/roboflow_batch.zip', compress: bool = True, max_dim: int = 1280):
        """Export frames and YOLO annotations to a single compressed zip file for instant Roboflow drag-and-drop web upload."""
        if not self.pending_labels:
            logger.warning("No pending labels to export to zip.")
            return None

        os.makedirs(os.path.dirname(os.path.abspath(zip_path)), exist_ok=True)
        total = len(self.pending_labels)
        logger.info(f"Packaging {total} frames into {zip_path}...")

        pbar = tqdm(total=total, desc="Creating Zip Archive", unit="frame") if tqdm else None
        start_time = time.time()
        exported = 0

        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf, tempfile.TemporaryDirectory() as tmp_dir:
            for idx, label in enumerate(self.pending_labels):
                img_path = label.get('image_path')
                if not img_path or not os.path.isfile(img_path):
                    if pbar:
                        pbar.update(1)
                    continue

                img_id = label.get('image_id', f"frame_{idx:06d}")

                # Prepare image (with optional optimization)
                if compress:
                    img = cv2.imread(img_path)
                    if img is not None:
                        h, w = img.shape[:2]
                        if max(h, w) > max_dim:
                            scale = max_dim / float(max(h, w))
                            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                        temp_img = os.path.join(tmp_dir, f"{img_id}.jpg")
                        cv2.imwrite(temp_img, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        zf.write(temp_img, arcname=f"{img_id}.jpg")
                    else:
                        zf.write(img_path, arcname=f"{img_id}.jpg")
                else:
                    zf.write(img_path, arcname=f"{img_id}.jpg")

                # Prepare YOLO txt annotation
                ann_path = os.path.join(tmp_dir, f"{img_id}.txt")
                self._write_yolo_annotation(label, ann_path)
                zf.write(ann_path, arcname=f"{img_id}.txt")

                exported += 1
                if pbar:
                    pbar.update(1)

        if pbar:
            pbar.close()

        elapsed = time.time() - start_time
        file_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        logger.info(f"Zip export complete: {exported} frames packaged into {zip_path} ({file_size_mb:.2f} MB in {elapsed:.1f}s)")
        return zip_path

    def _upload_roboflow(
        self,
        workers: int = 5,
        compress: bool = True,
        max_dim: int = 1280,
        use_zip: bool = True,
        batch_name: str = None,
    ):
        if not self.roboflow_api_key:
            logger.error("Roboflow API key not set. Set ROBOFLOW_API_KEY environment variable.")
            return
        if not self.roboflow_workspace or not self.roboflow_project:
            logger.error("Roboflow workspace or project not configured. Update config file.")
            logger.info("Falling back to local export...")
            self._export_local('data/auto_labeled')
            return
        try:
            from roboflow import Roboflow
            rf = Roboflow(api_key=self.roboflow_api_key)
            total = len(self.pending_labels)
            if total == 0:
                logger.info("No pending labels to upload.")
                return

            # IN-CODE ZIP UPLOAD FLOW: Upload single zip to signed GCS URL and unzip server-side
            if use_zip:
                try:
                    logger.info(f"Initiating in-code ZIP upload flow for {total} frames to Roboflow...")
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        temp_zip_path = os.path.join(tmp_dir, "roboflow_upload.zip")
                        self.export_zip(zip_path=temp_zip_path, compress=compress, max_dim=max_dim)

                        ws = rf.workspace(self.roboflow_workspace)
                        logger.info(f"Uploading zip to Roboflow signed storage (batch_name={batch_name})...")
                        res = ws.upload_dataset(
                            dataset_path=temp_zip_path,
                            project_name=self.roboflow_project,
                            batch_name=batch_name,
                            use_zip_upload=True,
                            wait=True,
                        )
                        logger.info(f"Roboflow cloud unzip & dataset import complete: {res}")
                        return
                except Exception as zip_err:
                    logger.warning(
                        f"In-code ZIP upload flow failed: {zip_err}. Falling back to concurrent per-image upload..."
                    )

            # FALLBACK / ALTERNATIVE: Concurrent per-image upload
            project = rf.workspace(self.roboflow_workspace).project(self.roboflow_project)
            logger.info(f"Connected to Roboflow project: {project.name}")
            logger.info(f"Starting parallel per-image upload of {total} frames (workers={workers}, compress={compress})...")
            start_time = time.time()

            pbar = tqdm(total=total, desc="Roboflow Upload", unit="frame") if tqdm else None
            success_count = 0
            error_count = 0

            with tempfile.TemporaryDirectory() as tmp_dir:
                def _upload_single_label(label_item):
                    img_path = label_item.get('image_path')
                    img_id = label_item.get('image_id', 'frame')
                    if not img_path or not os.path.isfile(img_path):
                        return False, "file_not_found"

                    # Optional image compression for fast transfer
                    upload_img_path = img_path
                    if compress:
                        img = cv2.imread(img_path)
                        if img is not None:
                            h, w = img.shape[:2]
                            if max(h, w) > max_dim:
                                scale = max_dim / float(max(h, w))
                                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                            compressed_path = os.path.join(tmp_dir, f"{img_id}_comp.jpg")
                            cv2.imwrite(compressed_path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            upload_img_path = compressed_path

                    # Build YOLO-format annotation file
                    annotation_path = os.path.join(tmp_dir, f"{img_id}.yolov8.txt")
                    self._write_yolo_annotation(label_item, annotation_path)

                    try:
                        project.upload(
                            image_path=upload_img_path,
                            annotation_path=annotation_path,
                        )
                        return True, None
                    except Exception as exc:
                        return False, str(exc)

                with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                    futures = [executor.submit(_upload_single_label, lbl) for lbl in self.pending_labels]
                    for fut in as_completed(futures):
                        ok, err = fut.result()
                        if ok:
                            success_count += 1
                        else:
                            error_count += 1
                            logger.warning(f"Upload error: {err}")

                        if pbar:
                            pbar.update(1)
                            pbar.set_postfix({'ok': success_count, 'err': error_count})

            if pbar:
                pbar.close()

            elapsed = time.time() - start_time
            rate = success_count / max(elapsed, 0.001)
            logger.info(
                f"Successfully uploaded {success_count}/{total} frames to Roboflow "
                f"({error_count} errors) in {elapsed:.1f}s ({rate:.1f} frames/sec)"
            )

        except ImportError:
            logger.error("roboflow package not installed. Run: pip install roboflow")
            logger.info("Falling back to local export...")
            self._export_local('data/auto_labeled')
        except Exception as e:
            logger.error(f"Roboflow upload failed: {e}")
            logger.info("Falling back to local export...")
            self._export_local('data/auto_labeled')

    @staticmethod
    def _write_yolo_annotation(label: dict, output_path: str):
        """Convert label dict (xyxy boxes) to YOLO format annotation file."""
        boxes = label.get('boxes', [])
        class_ids = label.get('class_ids', [])
        # Need image dimensions to normalize — get from image_path if available
        img_path = label.get('image_path')
        img_w, img_h = 640, 640  # fallback
        if img_path and os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                img_h, img_w = img.shape[:2]

        with open(output_path, 'w') as f:
            for i, box in enumerate(boxes):
                cls_id = class_ids[i] if i < len(class_ids) else 0
                x1, y1, x2, y2 = box
                cx = ((x1 + x2) / 2) / img_w
                cy = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

