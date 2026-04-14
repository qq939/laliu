import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


# ==========================================
# GLOBAL PARAMETERS (全局参数)
# ==========================================
# RTSP_URL 在本文件第 251 行作为参数传给 cv2.VideoCapture() 用于拉流
RTSP_URL = os.environ.get("LALIU_RTSP_URL", "rtsp://192.168.8.102:8554/ams/live")
# SAMPLE_INTERVAL_SEC 在本文件第 227 行用于控制采样节奏（默认 10 秒/帧）
SAMPLE_INTERVAL_SEC = float(os.environ.get("LALIU_SAMPLE_INTERVAL_SEC", "10"))
# OPENCV_FFMPEG_CAPTURE_OPTIONS 在本文件第 251 行创建 VideoCapture 前设置，使其在第 251 行生效（连接超时 5 秒，单位微秒）
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = os.environ.get(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS", "timeout;5000000"
)
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

# DUMMY_MODE 在本文件第 233 行决定是否使用假帧源（用于测试/无依赖运行）
DUMMY_MODE = os.environ.get("LALIU_DUMMY", "0") == "1"

# STREAMING_DIR 在本文件第 195 行输出 last.jpg 与 last-processed.jpg
STREAMING_DIR = os.environ.get("LALIU_STREAMING_DIR", "/Users/jimjiang/Downloads/laliu/streaming")
# OUTPUT_DIR 在本文件第 170 行输出 labels 与 ultralytics 输出（runs/stream）
OUTPUT_DIR = os.environ.get("LALIU_OUTPUT_DIR", "/Users/jimjiang/Downloads/laliu/runs/stream")

# DEFAULT_TEXTS 在本文件第 260 行作为 SAM3 text prompts
DEFAULT_TEXTS = ["pliers", "screwdriver"]
# DEFAULT_CONF 在本文件第 261 行作为 SAM3 置信度阈值
DEFAULT_CONF = 0.25
# TOPK 在本文件第 139 行控制后处理保留的目标数量
TOPK = 1


logging.getLogger("werkzeug").setLevel(logging.ERROR)

try:
    if hasattr(cv2, "LOG_LEVEL_ERROR") and hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
    elif hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(3)
    elif (
        hasattr(cv2, "utils")
        and hasattr(cv2.utils, "logging")
        and hasattr(cv2.utils.logging, "setLogLevel")
    ):
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass


@dataclass
class RuntimeConfig:
    texts: List[str]
    conf: float


def _last_jpg_path() -> str:
    return os.path.join(STREAMING_DIR, "last.jpg")


def _last_processed_jpg_path() -> str:
    return os.path.join(STREAMING_DIR, "last-processed.jpg")


def _labels_dir() -> str:
    return os.path.join(OUTPUT_DIR, "labels")


def _last_labels_json_path() -> str:
    return os.path.join(_labels_dir(), "last-labels.json")


def _last_labels_txt_path() -> str:
    return os.path.join(_labels_dir(), "last-labels.txt")


def _ultralytics_dir() -> str:
    return os.path.join(OUTPUT_DIR, "ultralytics")


def _ultralytics_predict_dir() -> str:
    return os.path.join(_ultralytics_dir(), "predict")


def _ultralytics_labels_dir() -> str:
    return os.path.join(_ultralytics_predict_dir(), "labels")


def _ultralytics_last_jpg_path() -> str:
    return os.path.join(_ultralytics_predict_dir(), "last.jpg")


def _ultralytics_last_txt_path() -> str:
    return os.path.join(_ultralytics_labels_dir(), "last.txt")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_jpg(path: str, frame_bgr) -> None:
    _ensure_dir(os.path.dirname(path))
    ok, buf = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(buf.tobytes())
    os.replace(tmp, path)


def _write_ultralytics_outputs(processed_frame_bgr, labels_txt: str) -> None:
    _write_jpg(_ultralytics_last_jpg_path(), processed_frame_bgr)
    _ensure_dir(_ultralytics_labels_dir())
    tmp_txt = _ultralytics_last_txt_path() + ".tmp"
    with open(tmp_txt, "w", encoding="utf-8") as f:
        f.write(labels_txt)
    os.replace(tmp_txt, _ultralytics_last_txt_path())


def _to_list(x):
    try:
        return x.detach().cpu().numpy().tolist()
    except Exception:
        try:
            return x.cpu().numpy().tolist()
        except Exception:
            try:
                return x.numpy().tolist()
            except Exception:
                return None


def _write_last_labels(cfg: RuntimeConfig, result) -> str:
    _ensure_dir(_labels_dir())

    boxes_out = []
    polygons_out = []

    if result is not None:
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes):
            xyxy = _to_list(getattr(boxes, "xyxy", None)) or []
            confs = _to_list(getattr(boxes, "conf", None)) or []
            clss = _to_list(getattr(boxes, "cls", None))
            if clss is None:
                clss = [0.0 for _ in range(len(xyxy))]
            for i in range(min(len(xyxy), len(confs), len(clss))):
                boxes_out.append(
                    {"xyxy": xyxy[i], "conf": float(confs[i]), "cls": float(clss[i])}
                )

        masks = getattr(result, "masks", None)
        xy = getattr(masks, "xy", None) if masks is not None else None
        if xy is not None:
            try:
                for poly in xy:
                    poly_list = _to_list(poly)
                    if poly_list is None and hasattr(poly, "tolist"):
                        poly_list = poly.tolist()
                    if poly_list is not None:
                        polygons_out.append(poly_list)
            except Exception:
                pass

    payload = {
        "texts": list(cfg.texts),
        "conf": float(cfg.conf),
        "boxes": boxes_out,
        "polygons": polygons_out,
    }
    tmp_json = _last_labels_json_path() + ".tmp"
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp_json, _last_labels_json_path())

    txt_lines = []
    for b in boxes_out:
        xyxy = b.get("xyxy") or []
        if len(xyxy) == 4:
            x1, y1, x2, y2 = xyxy
            txt_lines.append(f'{b["conf"]:.4f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}')
    labels_txt = "\n".join(txt_lines) + ("\n" if txt_lines else "")

    tmp_txt = _last_labels_txt_path() + ".tmp"
    with open(tmp_txt, "w", encoding="utf-8") as f:
        f.write(labels_txt)
    os.replace(tmp_txt, _last_labels_txt_path())

    return labels_txt


def _build_sam3_predictor(conf: float):
    from ultralytics.models.sam.predict import SAM3VideoSemanticPredictor

    try:
        from ultralytics.utils import LOGGER

        LOGGER.setLevel(logging.INFO)
    except Exception:
        pass

    overrides = dict(
        conf=conf,
        task="segment",
        mode="predict",
        imgsz=644,
        model="sam3.pt",
        half=False,
        verbose=True,
    )
    overrides.update(project=_ultralytics_dir(), name="predict", save_txt=True)

    predictor = SAM3VideoSemanticPredictor(overrides=overrides)
    _postprocess = predictor.postprocess

    def postprocess(preds, img, orig_imgs, *, _k=TOPK, _f=_postprocess):
        res = _f(preds, img, orig_imgs)
        out = []
        for r in res:
            if r.boxes is not None and len(r.boxes):
                idx = r.boxes.conf.argsort(descending=True)[:_k]
                r = r[idx]
            out.append(r)
        return out

    predictor.postprocess = postprocess
    return predictor


def _sam3_process_file(predictor, image_path: str, texts: List[str], conf: float):
    if hasattr(predictor, "args") and hasattr(predictor.args, "conf"):
        try:
            predictor.args.conf = conf
        except Exception:
            pass

    results = predictor(source=image_path, text=texts, stream=False, save=True)
    if isinstance(results, list) and results:
        r = results[0]
    else:
        r = next(iter(results))

    if hasattr(r, "plot"):
        plotted = r.plot()
        if plotted is not None:
            return plotted, r

    img = cv2.imread(image_path)
    return (img if img is not None else np.zeros((1, 1, 3), dtype=np.uint8)), r


def _dummy_frame(frame_id: int):
    xs = np.arange(640, dtype=np.uint16)
    ys = np.arange(480, dtype=np.uint16)[:, None]
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :, 0] = (xs % 256).astype(np.uint8)
    frame[:, :, 1] = (ys % 256).astype(np.uint8)
    frame[:, :, 2] = ((xs[None, :] + ys + frame_id) % 256).astype(np.uint8)
    return frame


def _read_one_frame() -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        return None
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return frame


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--texts", default="\n".join(DEFAULT_TEXTS))
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--window", default="SAM3")
    parser.add_argument("--no-ui", action="store_true")
    args = parser.parse_args(argv)

    cfg = RuntimeConfig(
        texts=[s.strip() for s in args.texts.splitlines() if s.strip()],
        conf=float(args.conf),
    )

    predictor = None
    if not DUMMY_MODE:
        try:
            predictor = _build_sam3_predictor(cfg.conf)
        except Exception:
            predictor = None

    last_ts = 0.0
    frame_id = 0

    while True:
        now = time.time()
        if now - last_ts >= SAMPLE_INTERVAL_SEC:
            last_ts = now
            frame_id += 1

            frame = _dummy_frame(frame_id) if DUMMY_MODE else _read_one_frame()
            if frame is None:
                time.sleep(0.2)
                continue

            _write_jpg(_last_jpg_path(), frame)

            if predictor is None:
                processed = frame
                res = None
            else:
                processed, res = _sam3_process_file(
                    predictor, _last_jpg_path(), cfg.texts, cfg.conf
                )

            _write_jpg(_last_processed_jpg_path(), processed)
            labels_txt = _write_last_labels(cfg, res)
            _write_ultralytics_outputs(processed, labels_txt)

        if args.no_ui or os.environ.get("LALIU_CV_DISABLE_UI", "0") == "1":
            time.sleep(0.05)
            continue

        img = cv2.imread(_last_processed_jpg_path())
        if img is not None:
            cv2.imshow(args.window, img)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

