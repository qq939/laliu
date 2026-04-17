from ultralytics import settings
settings.update({
    "runs_dir": "/Users/jimjiang/Downloads/laliu/runs"  # 强制锁死你的目录
}) # 强制切换到当前脚本目录
import argparse
import json
import os
import logging
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, redirect, request
from flask import cli


# ==========================================
# GLOBAL PARAMETERS (全局参数)
# ==========================================
# RTSP_URL 在本文件第 298 行作为参数传给 cv2.VideoCapture() 用于拉流
RTSP_URL = os.environ.get("LALIU_RTSP_URL", "rtsp://192.168.8.102:8554/ams/live")
# SAMPLE_INTERVAL_SEC 在本文件第 282 行用于控制采样节奏（默认 10 秒/帧）
SAMPLE_INTERVAL_SEC = float(os.environ.get("LALIU_SAMPLE_INTERVAL_SEC", "10"))
# OPENCV_FFMPEG_CAPTURE_OPTIONS 在本文件第 298 行创建 VideoCapture 前设置，使其在第 298 行生效（连接超时 5 秒，单位微秒）
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = os.environ.get(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS", "timeout;5000000"
)
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
# DUMMY_MODE 在本文件第 265 行决定是否加载 SAM3，并在第 282 行走假数据源路径
DUMMY_MODE = os.environ.get("LALIU_DUMMY", "0") == "1"
# STREAMING_DIR 在本文件第 310 行用于输出 last.jpg 与 last-processed.jpg
STREAMING_DIR = os.environ.get("LALIU_STREAMING_DIR", "/Users/jimjiang/Downloads/laliu/streaming")
# OUTPUT_DIR 在本文件第 171 行用于输出 labels / 推理辅助信息（runs/stream）
OUTPUT_DIR = os.environ.get("LALIU_OUTPUT_DIR", "/Users/jimjiang/Downloads/laliu/runs/stream")
# DEFAULT_TEXTS 在本文件第 45 行初始化 WebUI 的文本列表
DEFAULT_TEXTS = ["pliers", "screwdriver"]
# TOPK 在本文件第 92 行控制后处理保留的目标数量（与 test_video.py 一致）
TOPK = 3
# DEFAULT_CONF 在本文件第 121 行用于动态调整模型置信度阈值（从 WebUI 更新）
DEFAULT_CONF = 0.25


app = Flask(__name__)
cli.show_server_banner = lambda *args, **kwargs: None

logging.getLogger("werkzeug").setLevel(logging.ERROR)

try:
    if hasattr(cv2, "LOG_LEVEL_ERROR") and hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
    elif hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(3)
    elif hasattr(cv2, "utils") and hasattr(cv2.utils, "logging") and hasattr(cv2.utils.logging, "setLogLevel"):
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass


@dataclass
class SharedState:
    texts: List[str]
    conf: float
    rtsp_input: str
    rtsp_url: str
    sample_interval_sec: float
    model_loaded: bool = False
    last_infer_ms: float = 0.0
    last_infer_boxes: int = 0
    last_infer_polygons: int = 0
    last_processed_ts: float = 0.0
    last_saved_ts: float = 0.0
    frame_id: int = 0
    last_error: str = ""
    force_run: bool = False
    lock: threading.Lock = threading.Lock()


def _normalize_camera_input(s: str) -> str:
    raw = (s or "").strip()
    if not raw:
        return RTSP_URL
    if raw.startswith("rtsp://"):
        return raw
    if raw.startswith("gige://"):
        host = raw[len("gige://") :].strip()
        return f"rtsp://{host}:8554/ams/live"
    if raw.startswith("gige:"):
        host = raw[len("gige:") :].strip()
        return f"rtsp://{host}:8554/ams/live"
    if "://" in raw:
        return raw
    host = raw
    return f"rtsp://{host}:8554/ams/live"


STATE = SharedState(
    texts=list(DEFAULT_TEXTS),
    conf=float(DEFAULT_CONF),
    rtsp_input=str(RTSP_URL),
    rtsp_url=_normalize_camera_input(str(RTSP_URL)),
    sample_interval_sec=float(SAMPLE_INTERVAL_SEC),
    force_run=False,
)


def _last_jpg_path() -> str:
    return os.path.join(STREAMING_DIR, "last.jpg")


def _last_image_jpg_path() -> str:
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


def _set_texts_from_multiline(multiline: str) -> List[str]:
    items = []
    for line in multiline.splitlines():
        s = line.strip()
        if s:
            items.append(s)
    return items


def _dummy_process_frame(frame_bgr, texts: List[str]):
    out = frame_bgr.copy()
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    cv2.putText(out, now, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    with STATE.lock:
        fid = STATE.frame_id
        conf = STATE.conf
    cv2.putText(out, f"frame_id={fid} conf={conf:.2f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    y = 60
    for t in texts[:10]:
        cv2.putText(out, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        y += 30
    return out


def _build_sam3_predictor(conf: float):
    from ultralytics.models.sam.predict import SAM3SemanticPredictor

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

    predictor = SAM3SemanticPredictor(overrides=overrides)
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


def _sam3_process_frame(predictor, source, texts: List[str], conf: float):
    if hasattr(predictor, "args") and hasattr(predictor.args, "conf"):
        try:
            predictor.args.conf = conf
        except Exception:
            pass
    if isinstance(source, str):
        predictor.set_image(source)
        results = predictor(text=texts, save=True)
        r = results[0] if isinstance(results, list) and results else next(iter(results))
    else:
        results = predictor(source=source, text=texts, stream=False, save=True)
        r = results[0] if isinstance(results, list) and results else next(iter(results))
    if hasattr(r, "plot"):
        plotted = r.plot()
        if plotted is not None:
            return plotted, r
    if isinstance(source, str):
        img = cv2.imread(source)
        return (img if img is not None else np.zeros((1, 1, 3), dtype=np.uint8)), r
    return source, r


def _summarize_and_store_infer(res, infer_ms: float):
    boxes_n = 0
    polys_n = 0
    try:
        boxes = getattr(res, "boxes", None)
        if boxes is not None:
            boxes_n = int(len(boxes))
    except Exception:
        boxes_n = 0
    try:
        masks = getattr(res, "masks", None)
        xy = getattr(masks, "xy", None) if masks is not None else None
        if xy is not None:
            polys_n = int(len(xy))
    except Exception:
        polys_n = 0

    with STATE.lock:
        STATE.last_infer_ms = float(infer_ms)
        STATE.last_infer_boxes = boxes_n
        STATE.last_infer_polygons = polys_n


def _write_jpg(path: str, frame_bgr) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok, buf = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(buf.tobytes())
    os.replace(tmp, path)


def _write_last_jpg(frame_bgr) -> None:
    _write_jpg(_last_jpg_path(), frame_bgr)


def _write_last_image_jpg(frame_bgr) -> None:
    _write_jpg(_last_image_jpg_path(), frame_bgr)


def _write_ultralytics_outputs(processed_frame_bgr, labels_txt: str) -> None:
    _write_jpg(_ultralytics_last_jpg_path(), processed_frame_bgr)
    os.makedirs(_ultralytics_labels_dir(), exist_ok=True)
    tmp_txt = _ultralytics_last_txt_path() + ".tmp"
    with open(tmp_txt, "w", encoding="utf-8") as f:
        f.write(labels_txt)
    os.replace(tmp_txt, _ultralytics_last_txt_path())


def _write_last_labels(texts: List[str], conf: float, result) -> str:
    os.makedirs(_labels_dir(), exist_ok=True)

    boxes_out = []
    polygons_out = []

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

    if result is not None:
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes):
            xyxy = _to_list(getattr(boxes, "xyxy", None))
            confs = _to_list(getattr(boxes, "conf", None))
            clss = _to_list(getattr(boxes, "cls", None))
            if xyxy is None:
                xyxy = []
            if confs is None:
                confs = []
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
        "texts": list(texts),
        "conf": float(conf),
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


def _get_texts_snapshot() -> List[str]:
    with STATE.lock:
        return list(STATE.texts)


def _get_conf_snapshot() -> float:
    with STATE.lock:
        return float(STATE.conf)


def _get_rtsp_snapshot() -> str:
    with STATE.lock:
        return str(STATE.rtsp_url)


def _get_interval_snapshot() -> float:
    with STATE.lock:
        return float(STATE.sample_interval_sec)


def _update_status_ok():
    with STATE.lock:
        STATE.last_processed_ts = time.time()
        STATE.last_saved_ts = STATE.last_processed_ts
        STATE.frame_id += 1
        STATE.last_error = ""


def _update_status_err(msg: str):
    with STATE.lock:
        STATE.last_error = msg[:500]


def _processing_loop(stop_event: threading.Event):
    predictor = None
    if not DUMMY_MODE:
        try:
            predictor = _build_sam3_predictor(_get_conf_snapshot())
            with STATE.lock:
                STATE.model_loaded = True
        except Exception as e:
            _update_status_err(f"加载 SAM3 失败: {e}")
            predictor = None
            with STATE.lock:
                STATE.model_loaded = False

    last_ts = 0.0

    while not stop_event.is_set():
        now = time.time()
        with STATE.lock:
            forced = STATE.force_run
            if forced:
                STATE.force_run = False
        if not forced and now - last_ts < _get_interval_snapshot():
            time.sleep(0.05)
            continue

        try:
            last_ts = now
            if DUMMY_MODE:
                xs = np.arange(640, dtype=np.uint16)
                ys = np.arange(480, dtype=np.uint16)[:, None]
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame[:, :, 0] = (xs % 256).astype(np.uint8)
                frame[:, :, 1] = (ys % 256).astype(np.uint8)
                frame[:, :, 2] = ((xs[None, :] + ys) % 256).astype(np.uint8)
                texts = _get_texts_snapshot()
                conf = _get_conf_snapshot()
                _write_last_jpg(frame)
                out = _dummy_process_frame(frame, texts)
                _write_last_image_jpg(out)
                labels_txt = _write_last_labels(texts, conf, None)
                _write_ultralytics_outputs(out, labels_txt)
                _update_status_ok()
                continue

            cap = cv2.VideoCapture(_get_rtsp_snapshot(), cv2.CAP_FFMPEG)
            if not cap.isOpened():
                _update_status_err(f"无法打开 RTSP 流: {_get_rtsp_snapshot()}")
                cap.release()
                continue

            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                _update_status_err("读取帧失败")
                continue

            _write_last_jpg(frame)
            texts = _get_texts_snapshot()
            conf = _get_conf_snapshot()
            if predictor is None:
                out = _dummy_process_frame(frame, texts)
                res = None
            else:
                print(f"text={texts} conf={conf}")
                print(f"input={_last_jpg_path()}")
                print(f"processed={_last_image_jpg_path()}")
                print(f"results={_ultralytics_predict_dir()}")
                t0 = time.time()
                out, res = _sam3_process_frame(predictor, _last_jpg_path(), texts, conf)
                infer_ms = (time.time() - t0) * 1000.0
                _summarize_and_store_infer(res, infer_ms)
            _write_last_image_jpg(out)
            labels_txt = _write_last_labels(texts, conf, res)
            _write_ultralytics_outputs(out, labels_txt)
            _update_status_ok()

            try:
                h, w = frame.shape[:2]
            except Exception:
                h, w = 0, 0
            boxes_n = 0
            polys_n = 0
            with STATE.lock:
                boxes_n = STATE.last_infer_boxes
                polys_n = STATE.last_infer_polygons
            if predictor is not None:
                pass
        except Exception as e:
            _update_status_err(str(e))
            continue


@app.get("/")
def index():
    texts = _get_texts_snapshot()
    multiline = "\n".join(texts)
    conf = _get_conf_snapshot()
    html = f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>WebUI 工作台</title>
    <style>
      :root {{
        --bg: #0B1220;
        --bg-elevated: #0F1A2B;
        --bg-elevated-2: #0C1627;
        --border: #22324A;
        --border-soft: rgba(34, 50, 74, 0.6);
        --text: #C7D2E2;
        --text-muted: #94A3B8;
        --accent: #3B6FB6;
        --accent-hover: #4A7FC6;
        --success: #2F8F6B;
        --warning: #A27B2C;
        --danger: #A24343;
        --shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        --radius: 12px;
      }}

      html[data-theme="dark"] {{
        --bg: #060B14;
        --bg-elevated: #0A1220;
        --bg-elevated-2: #08101D;
        --border: #1C2A42;
        --border-soft: rgba(28, 42, 66, 0.6);
        --text: #D1DAEA;
        --text-muted: #9AA7BE;
        --accent: #3566AE;
        --accent-hover: #4477BE;
        --shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
      }}

      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", "Apple Color Emoji", "Segoe UI Emoji";
        background: radial-gradient(1200px 800px at 20% 0%, rgba(59, 111, 182, 0.12), transparent 55%),
                    radial-gradient(900px 700px at 80% 10%, rgba(148, 163, 184, 0.06), transparent 60%),
                    var(--bg);
        color: var(--text);
      }}

      a {{ color: var(--text); text-decoration: none; }}
      a:hover {{ color: var(--accent-hover); }}

      .topbar {{
        height: 56px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 16px;
        border-bottom: 1px solid var(--border);
        background: rgba(11, 18, 32, 0.65);
        backdrop-filter: blur(10px);
      }}

      html[data-theme="dark"] .topbar {{
        background: rgba(6, 11, 20, 0.65);
      }}

      .brand {{
        display: flex;
        align-items: baseline;
        gap: 10px;
      }}
      .brand h1 {{
        margin: 0;
        font-size: 16px;
        font-weight: 600;
        letter-spacing: 0.2px;
      }}
      .brand .sub {{
        font-size: 12px;
        color: var(--text-muted);
      }}

      .top-actions {{
        display: flex;
        align-items: center;
        gap: 10px;
      }}

      .chip {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 10px;
        border: 1px solid var(--border-soft);
        background: rgba(15, 26, 43, 0.65);
        border-radius: 999px;
        font-size: 12px;
        color: var(--text-muted);
      }}
      html[data-theme="dark"] .chip {{ background: rgba(10, 18, 32, 0.65); }}

      .toggle {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 10px;
        border: 1px solid var(--border-soft);
        background: rgba(15, 26, 43, 0.65);
        border-radius: 999px;
        cursor: pointer;
        user-select: none;
      }}
      html[data-theme="dark"] .toggle {{ background: rgba(10, 18, 32, 0.65); }}

      .switch {{
        width: 36px;
        height: 20px;
        background: rgba(148, 163, 184, 0.25);
        border: 1px solid var(--border-soft);
        border-radius: 999px;
        position: relative;
      }}
      .switch::after {{
        content: "";
        position: absolute;
        top: 2px;
        left: 2px;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: rgba(199, 210, 226, 0.85);
        transition: transform 160ms ease;
      }}
      html[data-theme="dark"] .switch {{ background: rgba(59, 111, 182, 0.25); }}
      html[data-theme="dark"] .switch::after {{ transform: translateX(16px); background: rgba(209, 218, 234, 0.9); }}

      .container {{
        padding: 16px;
      }}

      .grid {{
        display: grid;
        gap: 16px;
        grid-template-columns: 1fr;
        align-items: stretch;
      }}

      @media (min-width: 1200px) {{
        .grid {{
          grid-template-columns: 1fr 1fr 1fr;
        }}
      }}

      @media (min-width: 768px) and (max-width: 1199px) {{
        .grid {{
          grid-template-columns: 1fr 1fr;
        }}
        .panel.params {{ grid-column: 1 / -1; }}
      }}

      .panel {{
        background: linear-gradient(180deg, rgba(15, 26, 43, 0.85), rgba(12, 22, 39, 0.78));
        border: 1px solid var(--border);
        border-radius: var(--radius);
        box-shadow: var(--shadow);
        overflow: hidden;
        min-height: 520px;
        display: flex;
        flex-direction: column;
        height: 100%;
      }}

      html[data-theme="dark"] .panel {{
        background: linear-gradient(180deg, rgba(10, 18, 32, 0.9), rgba(8, 16, 29, 0.82));
      }}

      .panel-header {{
        padding: 12px 12px 10px 12px;
        border-bottom: 1px solid var(--border);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
      }}

      .panel-title {{
        display: flex;
        flex-direction: column;
        gap: 2px;
      }}
      .panel-title .name {{
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.2px;
      }}
      .panel-title .desc {{
        font-size: 12px;
        color: var(--text-muted);
      }}

      .panel-actions {{
        display: flex;
        align-items: center;
        gap: 8px;
      }}

      .btn {{
        height: 34px;
        padding: 0 12px;
        border-radius: 10px;
        border: 1px solid var(--border-soft);
        background: rgba(15, 26, 43, 0.7);
        color: var(--text);
        font-size: 12px;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        transition: background 140ms ease, border-color 140ms ease;
      }}
      .btn:hover {{ background: rgba(15, 26, 43, 0.92); border-color: rgba(59, 111, 182, 0.35); }}
      html[data-theme="dark"] .btn {{ background: rgba(10, 18, 32, 0.8); }}
      html[data-theme="dark"] .btn:hover {{ background: rgba(10, 18, 32, 0.95); border-color: rgba(59, 111, 182, 0.35); }}

      .btn.primary {{
        background: rgba(59, 111, 182, 0.2);
        border-color: rgba(59, 111, 182, 0.35);
      }}
      .btn.primary:hover {{ background: rgba(59, 111, 182, 0.3); }}
      .btn:disabled {{ opacity: 0.55; cursor: not-allowed; }}

      .panel-body {{
        padding: 12px;
        display: flex;
        flex-direction: column;
        gap: 12px;
        overflow: hidden;
        flex: 1;
      }}

      .img-shell {{
        flex: 1;
        min-height: 360px;
        border: 1px solid var(--border-soft);
        background: var(--bg-elevated-2);
        border-radius: 12px;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
      }}

      .img-shell img {{
        width: 100%;
        height: 100%;
        object-fit: contain;
        display: block;
      }}

      .img-placeholder {{
        position: absolute;
        inset: 0;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 6px;
        color: var(--text-muted);
        font-size: 12px;
        pointer-events: none;
      }}

      .img-toolbar {{
        position: absolute;
        top: 10px;
        right: 10px;
        display: flex;
        gap: 8px;
      }}

      .icon-btn {{
        width: 34px;
        height: 34px;
        border-radius: 10px;
        border: 1px solid var(--border-soft);
        background: rgba(15, 26, 43, 0.65);
        color: var(--text);
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        transition: background 140ms ease, border-color 140ms ease;
      }}
      .icon-btn:hover {{ background: rgba(15, 26, 43, 0.9); border-color: rgba(59, 111, 182, 0.35); }}
      html[data-theme="dark"] .icon-btn {{ background: rgba(10, 18, 32, 0.65); }}
      html[data-theme="dark"] .icon-btn:hover {{ background: rgba(10, 18, 32, 0.95); }}

      .kv {{
        display: grid;
        grid-template-columns: 120px 1fr;
        gap: 8px 10px;
        padding: 10px;
        border: 1px solid var(--border-soft);
        background: rgba(15, 26, 43, 0.45);
        border-radius: 12px;
      }}
      html[data-theme="dark"] .kv {{ background: rgba(10, 18, 32, 0.45); }}

      .kv .k {{ color: var(--text-muted); font-size: 12px; }}
      .kv .v {{ color: var(--text); font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}

      .badge-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
      }}
      .badge {{
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 12px;
        border: 1px solid var(--border-soft);
        background: rgba(15, 26, 43, 0.45);
        color: var(--text-muted);
      }}

      .badge.ok {{ color: rgba(199, 210, 226, 0.95); border-color: rgba(47, 143, 107, 0.35); background: rgba(47, 143, 107, 0.12); }}
      .badge.warn {{ color: rgba(199, 210, 226, 0.95); border-color: rgba(162, 123, 44, 0.35); background: rgba(162, 123, 44, 0.12); }}
      .badge.err {{ color: rgba(209, 218, 234, 0.95); border-color: rgba(162, 67, 67, 0.4); background: rgba(162, 67, 67, 0.12); }}

      .form {{
        display: flex;
        flex-direction: column;
        gap: 12px;
        overflow: hidden;
        flex: 1;
      }}

      .field {{
        display: flex;
        flex-direction: column;
        gap: 6px;
      }}

      .label {{
        font-size: 12px;
        color: var(--text-muted);
      }}

      textarea, input[type="number"] {{
        width: 100%;
        border-radius: 12px;
        border: 1px solid var(--border-soft);
        background: rgba(12, 22, 39, 0.65);
        color: var(--text);
        font-size: 13px;
        padding: 10px 10px;
        outline: none;
        transition: border-color 140ms ease;
      }}
      html[data-theme="dark"] textarea, html[data-theme="dark"] input[type="number"], html[data-theme="dark"] input[type="text"] {{ background: rgba(8, 16, 29, 0.65); }}

      textarea, input[type="number"], input[type="text"] {{
        width: 100%;
        border: 1px solid var(--border-soft);
        background: rgba(15, 26, 43, 0.45);
        border-radius: 12px;
        color: var(--text);
        outline: none;
        padding: 10px 10px;
        font-size: 14px;
        transition: border-color 140ms ease, background 140ms ease;
      }}

      textarea:hover, input[type="number"]:hover, input[type="text"]:hover {{ border-color: rgba(59, 111, 182, 0.28); }}
      textarea:focus, input[type="number"]:focus, input[type="text"]:focus {{ border-color: rgba(59, 111, 182, 0.5); background: rgba(15, 26, 43, 0.55); }}
      textarea:focus, input[type="number"]:focus {{ border-color: rgba(59, 111, 182, 0.45); }}

      textarea {{
        min-height: 160px;
        resize: vertical;
      }}

      .row {{ display: grid; grid-template-columns: 1fr 120px; gap: 10px; align-items: center; }}
      .range {{ width: 100%; }}
      input[type="range"] {{
        -webkit-appearance: none;
        appearance: none;
        width: 100%;
        height: 6px;
        border-radius: 999px;
        background: rgba(148, 163, 184, 0.18);
        outline: none;
      }}
      input[type="range"]::-webkit-slider-thumb {{
        -webkit-appearance: none;
        appearance: none;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: rgba(59, 111, 182, 0.95);
        border: 1px solid rgba(59, 111, 182, 0.45);
        box-shadow: 0 0 0 4px rgba(59, 111, 182, 0.12);
      }}

      .actions {{ display: flex; gap: 10px; justify-content: flex-end; }}

      .mono {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 12px;
      }}

      details {{
        border: 1px solid var(--border-soft);
        border-radius: 12px;
        background: rgba(15, 26, 43, 0.35);
        overflow: hidden;
      }}
      summary {{
        padding: 10px;
        cursor: pointer;
        color: var(--text-muted);
        font-size: 12px;
      }}
      pre {{
        margin: 0;
        padding: 10px;
        overflow: auto;
        max-height: 240px;
        color: rgba(199, 210, 226, 0.9);
      }}

      .footer-note {{
        margin-top: 10px;
        color: var(--text-muted);
        font-size: 12px;
      }}
    </style>
  </head>
  <body>
    <div class="topbar">
      <div class="brand">
        <h1>WebUI 工作台</h1>
        <div class="sub">视觉推理 · 极简低亮度</div>
      </div>
      <div class="top-actions">
        <div class="chip" id="chip_conn">连接中…</div>
        <div class="toggle" id="theme_toggle" role="button" tabindex="0" aria-label="切换主题">
          <span style="font-size:12px;color:var(--text-muted)">Dim/Dark</span>
          <span class="switch" aria-hidden="true"></span>
        </div>
      </div>
    </div>

    <div class="container">
      <div class="grid">
        <section class="panel preview">
          <div class="panel-header">
            <div class="panel-title">
              <div class="name">图片预览</div>
              <div class="desc">原始帧（last.jpg）</div>
            </div>
            <div class="panel-actions">
              <a class="btn" href="/last.jpg" target="_blank">打开</a>
              <button class="btn" id="btn_refresh_preview" type="button">刷新</button>
            </div>
          </div>
          <div class="panel-body">
            <div class="img-shell">
              <div class="img-toolbar">
                <a class="icon-btn" href="/last.jpg" target="_blank" title="新窗口打开">↗</a>
                <a class="icon-btn" href="/last.jpg" download title="下载">↓</a>
              </div>
              <div class="img-placeholder" id="ph_raw">
                <div class="mono">/last.jpg</div>
                <div>等待图像…</div>
              </div>
              <img id="img_raw" data-has-img="0" src="/last.jpg" alt="raw" onload="this.dataset.hasImg='1';document.getElementById('ph_raw').style.display='none';" />
            </div>
            <div class="kv">
              <div class="k">输入</div>
              <div class="v mono">{_last_jpg_path()}</div>
              <div class="k">刷新</div>
              <div class="v" id="txt_refresh">每 2 秒</div>
            </div>
          </div>
        </section>

        <section class="panel result">
          <div class="panel-header">
            <div class="panel-title">
              <div class="name">结果</div>
              <div class="desc">处理后帧（last-processed.jpg）</div>
            </div>
            <div class="panel-actions">
              <a class="btn" href="/last-processed.jpg" target="_blank">打开</a>
              <button class="btn" id="btn_refresh_result" type="button">刷新</button>
            </div>
          </div>
          <div class="panel-body">
            <div class="badge-row" id="badges">
              <span class="badge" id="b_frame">frame: -</span>
              <span class="badge" id="b_infer">infer: -</span>
              <span class="badge" id="b_obj">obj: -</span>
              <span class="badge" id="b_mode">mode: -</span>
            </div>
            <div class="img-shell">
              <div class="img-toolbar">
                <a class="icon-btn" href="/last-processed.jpg" target="_blank" title="新窗口打开">↗</a>
                <a class="icon-btn" href="/last-processed.jpg" download title="下载">↓</a>
              </div>
              <div class="img-placeholder" id="ph_res">
                <div class="mono">/last-processed.jpg</div>
                <div>等待结果…</div>
              </div>
              <img id="img" data-has-img="0" src="/last-processed.jpg" alt="processed" onload="this.dataset.hasImg='1';document.getElementById('ph_res').style.display='none';" />
            </div>
            <div class="kv">
              <div class="k">输出</div>
              <div class="v mono">{_last_image_jpg_path()}</div>
              <div class="k">生效参数</div>
              <div class="v">
                <div class="mono" id="txt_active_texts">-</div>
                <div style="margin-top:4px;color:var(--text-muted);font-size:12px;">
                  conf=<span class="mono" id="txt_active_conf">-</span> · interval=<span class="mono" id="txt_active_interval">-</span>s
                </div>
              </div>
              <div class="k">labels</div>
              <div class="v">
                <a class="mono" href="/last-labels.json" target="_blank">last-labels.json</a>
                <span style="color:var(--text-muted)"> · </span>
                <a class="mono" href="/last-labels.txt" target="_blank">last-labels.txt</a>
              </div>
              <div class="k">结果目录</div>
              <div class="v mono">{OUTPUT_DIR}</div>
              <div class="k">错误</div>
              <div class="v" id="txt_error" style="color:var(--text-muted)">-</div>
            </div>
            <details>
              <summary>查看原始状态 JSON</summary>
              <pre id="status" class="mono">loading…</pre>
            </details>
          </div>
        </section>

        <section class="panel params">
          <div class="panel-header">
            <div class="panel-title">
              <div class="name">参数</div>
              <div class="desc">文本 prompts 与阈值</div>
            </div>
            <div class="panel-actions">
              <button class="btn" id="btn_defaults" type="button">恢复默认</button>
            </div>
          </div>
          <div class="panel-body">
            <form id="cfgForm" class="form" method="post" action="/set_config">
              <div class="field">
                <div class="label">摄像机（RTSP / GigE）</div>
                <input id="inp_rtsp" name="rtsp_url" type="text" value="{STATE.rtsp_input}" />
                <div style="margin-top:6px;color:var(--text-muted);font-size:12px;">
                  支持 <span class="mono">rtsp://...</span>、<span class="mono">gige://IP</span> 或直接填 <span class="mono">IP</span>
                </div>
              </div>
              <div class="field">
                <div class="label">Text（每行一个）</div>
                <textarea id="inp_texts" name="texts" spellcheck="false">{multiline}</textarea>
              </div>
              <div class="field">
                <div class="label">Conf</div>
                <input id="inp_conf" name="conf" type="number" min="0" max="1" step="any" value="{conf:.2f}" />
              </div>
              <div class="field">
                <div class="label">采帧间隔（秒）</div>
                <input id="inp_interval" name="sample_interval_sec" type="number" min="0.05" max="3600" step="any" value="{STATE.sample_interval_sec:.1f}" />
              </div>
              <div class="field">
                <div class="label">自动刷新</div>
                <div class="actions" style="justify-content: space-between;">
                  <button class="btn" id="btn_toggle_refresh" type="button">暂停</button>
                  <div class="chip" id="chip_last">last: -</div>
                </div>
              </div>
              <div class="actions">
                <button class="btn" id="btn_sync" type="button">从服务同步</button>
                <button class="btn primary" id="btn_apply" type="submit">应用</button>
              </div>
              <div class="footer-note">
                保持接口不变：仍使用 <span class="mono">/set_config</span>、<span class="mono">/status</span>、<span class="mono">/last.jpg</span>、<span class="mono">/last-processed.jpg</span>。
              </div>
            </form>
          </div>
        </section>
      </div>
    </div>

    <script>
      const DEFAULT_TEXTS = {json.dumps(DEFAULT_TEXTS)};
      const DEFAULT_CONF = {DEFAULT_CONF};
      const DEFAULT_INTERVAL = {float(SAMPLE_INTERVAL_SEC)};

      function $(id) {{ return document.getElementById(id); }}

      function setTheme(theme) {{
        document.documentElement.setAttribute('data-theme', theme);
        try {{ localStorage.setItem('laliu_theme', theme); }} catch (e) {{}}
      }}

      function initTheme() {{
        let t = 'dim';
        try {{
          t = localStorage.getItem('laliu_theme') || 'dim';
        }} catch (e) {{}}
        if (t !== 'dim' && t !== 'dark') t = 'dim';
        setTheme(t);
      }}

      function toggleTheme() {{
        const t = document.documentElement.getAttribute('data-theme') === 'dark' ? 'dim' : 'dark';
        setTheme(t);
      }}

      function bindThemeToggle() {{
        const el = $('theme_toggle');
        el.addEventListener('click', toggleTheme);
        el.addEventListener('keydown', (e) => {{
          if (e.key === 'Enter' || e.key === ' ') {{ e.preventDefault(); toggleTheme(); }}
        }});
      }}

      function setImg(id, phId, url) {{
        const img = $(id);
        const ph = $(phId);
        const had = img.getAttribute('data-has-img') === '1';
        img.onload = () => {{
          img.style.display = 'block';
          img.setAttribute('data-has-img', '1');
          ph.style.display = 'none';
        }};
        img.onerror = () => {{
          const has = img.getAttribute('data-has-img') === '1';
          if (!has) {{
            img.style.display = 'none';
            ph.style.display = 'flex';
          }}
        }};
        img.src = url;
      }}

      let refreshTimer = null;
      let refreshEnabled = true;
      const REFRESH_MS = 2000;

      function setRefreshEnabled(on) {{
        refreshEnabled = on;
        $('btn_toggle_refresh').textContent = on ? '暂停' : '继续';
        $('txt_refresh').textContent = on ? '每 2 秒' : '已暂停';
        if (refreshTimer) {{ clearInterval(refreshTimer); refreshTimer = null; }}
        if (on) {{
          refreshTimer = setInterval(refresh, REFRESH_MS);
        }}
      }}

      function setConf(from) {{
        const v = Math.max(0, Math.min(1, Number(from)));
        const el = $('inp_conf');
        if (el) el.value = v.toFixed(2);
      }}

      function setIntervalSec(from) {{
        const v = Math.max(0.05, Math.min(3600, Number(from)));
        const el = $('inp_interval');
        if (el) el.value = v.toFixed(2);
      }}

      $('btn_defaults').addEventListener('click', () => {{
        $('inp_texts').value = DEFAULT_TEXTS.join('\n');
        setConf(DEFAULT_CONF);
        setIntervalSec(DEFAULT_INTERVAL);
      }});

      $('btn_toggle_refresh').addEventListener('click', () => setRefreshEnabled(!refreshEnabled));

      $('btn_refresh_preview').addEventListener('click', () => {{
        const ts = Date.now();
        setImg('img_raw', 'ph_raw', '/last.jpg?ts=' + ts);
      }});
      $('btn_refresh_result').addEventListener('click', () => {{
        const ts = Date.now();
        setImg('img', 'ph_res', '/last-processed.jpg?ts=' + ts);
      }});

      async function applyConfig() {{
        const btn = $('btn_apply');
        btn.disabled = true;
        try {{
          const texts = $('inp_texts').value;
          const conf = Number($('inp_conf').value);
          const rtsp_url = $('inp_rtsp').value;
          const sample_interval_sec = Number($('inp_interval').value);
          const r = await fetch('/set_config', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{ texts, conf, rtsp_url, sample_interval_sec }}),
          }});
          if (!r.ok) throw new Error('HTTP ' + r.status);
          const j = await r.json();
          $('inp_texts').value = (j.texts || []).join('\n');
          setConf(j.conf);
          if (j.sample_interval_sec != null) setIntervalSec(j.sample_interval_sec);
          await refresh();
        }} catch (e) {{
          $('txt_error').textContent = String(e);
        }} finally {{
          btn.disabled = false;
        }}
      }}

      $('cfgForm').addEventListener('submit', (e) => {{ e.preventDefault(); applyConfig(); }});

      $('btn_sync').addEventListener('click', async () => {{
        try {{
          const r = await fetch('/config');
          const j = await r.json();
          $('inp_rtsp').value = j.rtsp_input || '';
          $('inp_texts').value = (j.texts || []).join('\n');
          setConf(j.conf);
          if (j.sample_interval_sec != null) setIntervalSec(j.sample_interval_sec);
          await refresh();
        }} catch (e) {{
          $('txt_error').textContent = String(e);
        }}
      }});

      function setBadge(el, text, cls) {{
        el.className = 'badge' + (cls ? ' ' + cls : '');
        el.textContent = text;
      }}

      async function refresh() {{
        try {{
          const r = await fetch('/status');
          const j = await r.json();

          $('status').textContent = JSON.stringify(j, null, 2);

          const ts = Date.now();
          setImg('img_raw', 'ph_raw', '/last.jpg?ts=' + ts);
          setImg('img', 'ph_res', '/last-processed.jpg?ts=' + ts);

          const ok = !j.last_error;
          $('chip_conn').textContent = ok ? '在线' : '异常';
          $('chip_conn').style.color = ok ? 'rgba(199,210,226,0.95)' : 'rgba(209,218,234,0.95)';
          $('chip_conn').style.borderColor = ok ? 'rgba(47, 143, 107, 0.35)' : 'rgba(162, 67, 67, 0.4)';
          $('chip_conn').style.background = ok ? 'rgba(47, 143, 107, 0.12)' : 'rgba(162, 67, 67, 0.12)';

          $('txt_error').textContent = j.last_error ? j.last_error : '-';
          $('chip_last').textContent = 'last: ' + (j.last_processed_ts ? new Date(j.last_processed_ts * 1000).toLocaleTimeString() : '-');

          setText('txt_active_texts', (j.texts && j.texts.length) ? j.texts.join(', ') : '-');
          setText('txt_active_conf', (j.conf != null && Number(j.conf).toFixed) ? Number(j.conf).toFixed(2) : String(j.conf ?? '-'));
          setText('txt_active_interval', (j.sample_interval_sec != null && Number(j.sample_interval_sec).toFixed) ? Number(j.sample_interval_sec).toFixed(2) : String(j.sample_interval_sec ?? '-'));

          setBadge($('b_frame'), 'frame: ' + (j.frame_id ?? '-'), ok ? 'ok' : 'err');
          setBadge($('b_infer'), 'infer: ' + ((j.last_infer_ms ?? 0).toFixed ? j.last_infer_ms.toFixed(1) : j.last_infer_ms) + 'ms', j.model_loaded ? 'ok' : 'warn');
          setBadge($('b_obj'), 'obj: ' + (j.last_infer_boxes ?? '-') + '/' + (j.last_infer_polygons ?? '-'), '');
          setBadge($('b_mode'), j.dummy ? 'mode: dummy' : (j.model_loaded ? 'mode: sam3' : 'mode: init'), j.dummy ? 'warn' : (j.model_loaded ? 'ok' : 'warn'));
        }} catch (e) {{
          $('chip_conn').textContent = '离线';
          $('txt_error').textContent = String(e);
        }}
      }}

      initTheme();
      bindThemeToggle();
      setRefreshEnabled(true);
      refresh();
    </script>
  </body>
</html>
"""
    return Response(html, mimetype="text/html")


@app.post("/set_texts")
def set_texts():
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        multiline = payload.get("texts", "")
    else:
        multiline = request.form.get("texts", "")
    items = _set_texts_from_multiline(multiline)
    with STATE.lock:
        STATE.texts = items
    if request.is_json:
        return jsonify({"ok": True, "texts": items})
    return redirect("/")


@app.post("/set_config")
def set_config():
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        multiline = payload.get("texts", "")
        conf_raw = payload.get("conf", "")
        rtsp_input = payload.get("rtsp_url", "")
        interval_raw = payload.get("sample_interval_sec", "")
    else:
        multiline = request.form.get("texts", "")
        conf_raw = request.form.get("conf", "")
        rtsp_input = request.form.get("rtsp_url", "")
        interval_raw = request.form.get("sample_interval_sec", "")

    items = _set_texts_from_multiline(multiline)
    conf = DEFAULT_CONF
    if conf_raw is not None and str(conf_raw).strip() != "":
        try:
            conf = float(conf_raw)
        except Exception:
            conf = DEFAULT_CONF
    if conf < 0:
        conf = 0.0
    if conf > 1:
        conf = 1.0

    interval_sec = None
    if interval_raw is not None and str(interval_raw).strip() != "":
        try:
            interval_sec = float(interval_raw)
        except Exception:
            interval_sec = None
    if interval_sec is not None:
        if interval_sec < 0.05:
            interval_sec = 0.05
        if interval_sec > 3600:
            interval_sec = 3600.0

    with STATE.lock:
        STATE.texts = items
        STATE.conf = conf
        if rtsp_input is not None and str(rtsp_input).strip() != "":
            STATE.rtsp_input = str(rtsp_input).strip()
            STATE.rtsp_url = _normalize_camera_input(STATE.rtsp_input)
        if interval_sec is not None:
            STATE.sample_interval_sec = float(interval_sec)
        STATE.force_run = True

    if request.is_json:
        return jsonify(
            {
                "ok": True,
                "texts": items,
                "conf": conf,
                "rtsp_input": STATE.rtsp_input,
                "rtsp_url": STATE.rtsp_url,
                "sample_interval_sec": float(STATE.sample_interval_sec),
            }
        )
    return redirect("/")


@app.get("/texts")
def texts():
    return jsonify({"texts": _get_texts_snapshot()})


@app.get("/config")
def config():
    with STATE.lock:
        return jsonify(
            {
                "texts": list(STATE.texts),
                "conf": float(STATE.conf),
                "rtsp_input": str(STATE.rtsp_input),
                "rtsp_url": str(STATE.rtsp_url),
                "sample_interval_sec": float(STATE.sample_interval_sec),
            }
        )


@app.get("/status")
def status():
    with STATE.lock:
        paths = {
            "streaming_last_jpg": _last_jpg_path(),
            "streaming_last_processed_jpg": _last_image_jpg_path(),
            "runs_labels_json": _last_labels_json_path(),
            "runs_labels_txt": _last_labels_txt_path(),
            "runs_ultralytics_predict_last_jpg": _ultralytics_last_jpg_path(),
            "runs_ultralytics_predict_last_txt": _ultralytics_last_txt_path(),
        }
        return jsonify(
            {
                "dummy": DUMMY_MODE,
                "texts": list(STATE.texts),
                "rtsp_input": str(STATE.rtsp_input),
                "rtsp_url": str(STATE.rtsp_url),
                "sample_interval_sec": float(STATE.sample_interval_sec),
                "last_processed_ts": STATE.last_processed_ts,
                "last_saved_ts": STATE.last_saved_ts,
                "frame_id": STATE.frame_id,
                "conf": float(STATE.conf),
                "model_loaded": bool(STATE.model_loaded),
                "last_infer_ms": float(STATE.last_infer_ms),
                "last_infer_boxes": int(STATE.last_infer_boxes),
                "last_infer_polygons": int(STATE.last_infer_polygons),
                "last_error": STATE.last_error,
                "paths": paths,
            }
        )


@app.get("/last.jpg")
def last_jpg():
    path = _last_jpg_path()
    if not os.path.exists(path):
        return Response("no image", status=404)
    with open(path, "rb") as f:
        data = f.read()
    return Response(data, mimetype="image/jpeg")


@app.get("/last-processed.jpg")
def last_processed_jpg():
    path = _last_image_jpg_path()
    if not os.path.exists(path):
        return Response("no image", status=404)
    with open(path, "rb") as f:
        data = f.read()
    return Response(data, mimetype="image/jpeg")


@app.get("/last-image.jpg")
def last_image_jpg_alias():
    return last_processed_jpg()


@app.get("/last-labels.json")
def last_labels_json():
    path = _last_labels_json_path()
    if not os.path.exists(path):
        return Response("no labels", status=404)
    with open(path, "rb") as f:
        data = f.read()
    return Response(data, mimetype="application/json")


@app.get("/last-labels.txt")
def last_labels_txt():
    path = _last_labels_txt_path()
    if not os.path.exists(path):
        return Response("no labels", status=404)
    with open(path, "rb") as f:
        data = f.read()
    return Response(data, mimetype="text/plain")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8008, type=int)
    args = parser.parse_args(argv)

    stop_event = threading.Event()
    worker = threading.Thread(target=_processing_loop, args=(stop_event,), daemon=True)
    worker.start()

    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
        return 0
    finally:
        stop_event.set()
        worker.join(timeout=2.0)


if __name__ == "__main__":
    raise SystemExit(main())
