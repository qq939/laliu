from ultralytics.models.sam.predict import SAM3VideoSemanticPredictor
overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=640, model="sam3.pt", half=False)
overrides.update(project="run/video", save_txt=True)
topk = 1
predictor = SAM3VideoSemanticPredictor (overrides=overrides)
_postprocess = predictor.postprocess

def postprocess(preds, img, orig_imgs, *, _k=topk, _f=_postprocess):
    res = _f(preds, img, orig_imgs)
    out = []
    for r in res:
        if r.boxes is not None and len(r.boxes):
            idx = r.boxes.conf.argsort(descending=True)[:_k]
            r = r[idx]
        out.append(r)
    return out

predictor.postprocess = postprocess
# Track concepts using text prompts
results = predictor (source="2miao_480x832.mp4", text=["electric screwdriver"], stream=True, save=True)
# Process results
for r in results:
    r.show()
# Display frame with tracked