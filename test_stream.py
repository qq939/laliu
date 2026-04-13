import argparse
import os
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, redirect, request


# ==========================================
# GLOBAL PARAMETERS (全局参数)
# ==========================================
# RTSP_URL 在本文件第 216 行作为参数传给 cv2.VideoCapture() 用于拉流
RTSP_URL = os.environ.get("LALIU_RTSP_URL", "rtsp://192.168.8.102:8554/ams/live")
# SAMPLE_INTERVAL_SEC 在本文件第 193 行用于控制采样节奏（默认 10 秒/帧）
SAMPLE_INTERVAL_SEC = float(os.environ.get("LALIU_SAMPLE_INTERVAL_SEC", "10"))
# OPENCV_FFMPEG_CAPTURE_OPTIONS 在本文件第 216 行创建 VideoCapture 前设置，使其在第 216 行生效（连接超时 5 秒，单位微秒）
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = os.environ.get(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS", "timeout;5000000"
)
# DUMMY_MODE 在本文件第 148 行决定是否加载 SAM3（用于测试/无依赖运行）
DUMMY_MODE = os.environ.get("LALIU_DUMMY", "0") == "1"
# OUTPUT_DIR 在本文件第 49 行用于输出 latest.jpg 给 WebUI 展示
OUTPUT_DIR = os.environ.get("LALIU_OUTPUT_DIR", "run/stream")
# DEFAULT_TEXTS 在本文件第 45 行初始化 WebUI 的文本列表
DEFAULT_TEXTS = ["electric screwdriver"]
# TOPK 在本文件第 92 行控制后处理保留的目标数量（与 test_video.py 一致）
TOPK = 1
# DEFAULT_CONF 在本文件第 121 行用于动态调整模型置信度阈值（从 WebUI 更新）
DEFAULT_CONF = 0.25


app = Flask(__name__)


@dataclass
class SharedState:
    texts: List[str]
    conf: float
    last_processed_ts: float = 0.0
    last_saved_ts: float = 0.0
    frame_id: int = 0
    last_error: str = ""
    lock: threading.Lock = threading.Lock()


STATE = SharedState(texts=list(DEFAULT_TEXTS), conf=float(DEFAULT_CONF))


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _latest_jpg_path() -> str:
    return os.path.join(OUTPUT_DIR, "latest.jpg")


def _last_image_jpg_path() -> str:
    return os.path.join(OUTPUT_DIR, "last-image.jpg")


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
    from ultralytics.models.sam.predict import SAM3VideoSemanticPredictor

    overrides = dict(
        conf=conf,
        task="segment",
        mode="predict",
        imgsz=640,
        model="sam3.pt",
        half=False,
    )
    overrides.update(project="run/stream", save_txt=True)

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


def _sam3_process_frame(predictor, frame_bgr, texts: List[str], conf: float):
    if hasattr(predictor, "args") and hasattr(predictor.args, "conf"):
        try:
            predictor.args.conf = conf
        except Exception:
            pass
    results = predictor(source=frame_bgr, text=texts, stream=False, save=False)
    if isinstance(results, list) and results:
        r = results[0]
    else:
        r = next(iter(results))
    if hasattr(r, "plot"):
        plotted = r.plot()
        if plotted is not None:
            return plotted
    return frame_bgr


def _write_jpg(path: str, frame_bgr) -> None:
    _ensure_output_dir()
    ok, buf = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(buf.tobytes())
    os.replace(tmp, path)


def _write_latest_jpg(frame_bgr) -> None:
    _write_jpg(_latest_jpg_path(), frame_bgr)


def _write_last_image_jpg(frame_bgr) -> None:
    _write_jpg(_last_image_jpg_path(), frame_bgr)


def _get_texts_snapshot() -> List[str]:
    with STATE.lock:
        return list(STATE.texts)


def _get_conf_snapshot() -> float:
    with STATE.lock:
        return float(STATE.conf)


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
        except Exception as e:
            _update_status_err(f"加载 SAM3 失败: {e}")
            predictor = None

    cap = None
    last_ts = 0.0

    while not stop_event.is_set():
        now = time.time()
        if now - last_ts < SAMPLE_INTERVAL_SEC:
            time.sleep(0.05)
            continue

        try:
            if DUMMY_MODE:
                xs = np.arange(640, dtype=np.uint16)
                ys = np.arange(480, dtype=np.uint16)[:, None]
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame[:, :, 0] = (xs % 256).astype(np.uint8)
                frame[:, :, 1] = (ys % 256).astype(np.uint8)
                frame[:, :, 2] = ((xs[None, :] + ys) % 256).astype(np.uint8)
                texts = _get_texts_snapshot()
                _write_latest_jpg(frame)
                out = _dummy_process_frame(frame, texts)
                _write_last_image_jpg(out)
                _update_status_ok()
                last_ts = now
                continue

            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    _update_status_err(f"无法打开 RTSP 流: {RTSP_URL}")
                    time.sleep(1.0)
                    continue

            ok, frame = cap.read()
            if not ok or frame is None:
                _update_status_err("读取帧失败")
                cap.release()
                cap = None
                time.sleep(0.5)
                continue

            _write_latest_jpg(frame)
            texts = _get_texts_snapshot()
            conf = _get_conf_snapshot()
            if predictor is None:
                out = _dummy_process_frame(frame, texts)
            else:
                out = _sam3_process_frame(predictor, frame, texts, conf)
            _write_last_image_jpg(out)
            _update_status_ok()
            last_ts = now
        except Exception as e:
            _update_status_err(str(e))
            time.sleep(0.5)

    if cap is not None:
        cap.release()


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
    <title>RTSP Stream Processor</title>
  </head>
  <body>
    <h3>Texts (one per line)</h3>
    <form method="post" action="/set_config">
      <div>
        <label>Conf: </label>
        <input name="conf" type="number" step="0.01" min="0" max="1" value="{conf:.2f}" />
      </div>
      <textarea name="texts" rows="8" cols="60">{multiline}</textarea><br/>
      <button type="submit">Update</button>
    </form>
    <h3>Status</h3>
    <pre id="status">loading...</pre>
    <h3>Last Image</h3>
    <img id="img" src="/last-image.jpg" style="max-width: 95%; border: 1px solid #ddd;" />
    <script>
      async function refresh() {{
        const r = await fetch('/status');
        const j = await r.json();
        document.getElementById('status').textContent = JSON.stringify(j, null, 2);
        document.getElementById('img').src = '/last-image.jpg?ts=' + Date.now();
      }}
      setInterval(refresh, 2000);
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
    else:
        multiline = request.form.get("texts", "")
        conf_raw = request.form.get("conf", "")

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

    with STATE.lock:
        STATE.texts = items
        STATE.conf = conf

    if request.is_json:
        return jsonify({"ok": True, "texts": items, "conf": conf})
    return redirect("/")


@app.get("/texts")
def texts():
    return jsonify({"texts": _get_texts_snapshot()})


@app.get("/config")
def config():
    return jsonify({"texts": _get_texts_snapshot(), "conf": _get_conf_snapshot()})


@app.get("/status")
def status():
    with STATE.lock:
        return jsonify(
            {
                "dummy": DUMMY_MODE,
                "rtsp_url": RTSP_URL,
                "sample_interval_sec": SAMPLE_INTERVAL_SEC,
                "last_processed_ts": STATE.last_processed_ts,
                "last_saved_ts": STATE.last_saved_ts,
                "frame_id": STATE.frame_id,
                "conf": float(STATE.conf),
                "last_error": STATE.last_error,
            }
        )


@app.get("/latest.jpg")
def latest_jpg():
    path = _latest_jpg_path()
    if not os.path.exists(path):
        return Response("no image", status=404)
    with open(path, "rb") as f:
        data = f.read()
    return Response(data, mimetype="image/jpeg")


@app.get("/last-image.jpg")
def last_image_jpg():
    path = _last_image_jpg_path()
    if not os.path.exists(path):
        return Response("no image", status=404)
    with open(path, "rb") as f:
        data = f.read()
    return Response(data, mimetype="image/jpeg")


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
