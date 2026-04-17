import cv2
import numpy as np
from pathlib import Path

def test_sam3_video_predictor():
    """测试SAM3VideoPredictor功能"""
    print("测试SAM3视频分割预测器...")

    try:
        from ultralytics.models.sam import SAM3VideoPredictor

        test_video = Path("src/test_video.mp4")
        if not test_video.exists():
            print("⚠ 测试视频不存在，跳过测试")
            return

        overrides = dict(
            conf=0.25,
            task="segment",
            mode="predict",
            model="sam3.pt",
            half=False,
            save=False,
            verbose=False
        )

        print("正在加载SAM3视频预测器...")
        predictor = SAM3VideoPredictor(overrides=overrides)
        print("✓ SAM3视频预测器加载成功")

        bboxes = [[200, 150, 400, 350]]
        labels = [1]

        print(f"正在测试视频分割，跟踪bbox: {bboxes}")
        results = predictor(
            source=str(test_video),
            bboxes=bboxes,
            labels=labels,
            stream=True
        )

        frame_count = 0
        for r in results:
            if frame_count == 0:
                print(f"✓ 成功处理第1帧")
                print(f"  - 检测到 {len(r.masks) if r.masks is not None else 0} 个掩码")

            annotated = r.plot()
            print(f"  - 帧 {frame_count + 1} 分割完成")

            if frame_count >= 2:
                break
            frame_count += 1

        print(f"✓ SAM3视频分割测试通过 (处理了 {frame_count + 1} 帧)")

    except ImportError as e:
        print(f"✗ SAM3VideoPredictor导入失败: {e}")
        print("  提示: 确保已安装ultralytics>=8.3.237")
    except Exception as e:
        print(f"✗ SAM3视频分割测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sam3_video_predictor()
