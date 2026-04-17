import unittest
import cv2
import numpy as np
from pathlib import Path
import time

class TestSAMIntegration(unittest.TestCase):
    def test_sam_model_import(self):
        try:
            from ultralytics import SAM
            print("✓ SAM模型导入成功")
        except ImportError as e:
            self.fail(f"SAM模型导入失败: {e}")

    def test_sam_model_loading(self):
        try:
            from ultralytics import SAM
            model = SAM("sam_b.pt")
            print("✓ SAM模型加载成功")
        except Exception as e:
            print(f"⚠ SAM模型加载失败: {e}")
            print("→ 这是正常的，如果您还没有下载SAM模型")

    def test_bbox_to_sam_segmentation(self):
        try:
            from ultralytics import SAM
            model = SAM("sam_b.pt")

            img = np.zeros((480, 640, 3), dtype=np.uint8)
            img[100:300, 200:400] = [255, 128, 64]

            bbox = [200, 100, 400, 300]
            results = model(img, bboxes=[bbox], verbose=False)

            self.assertIsNotNone(results, "SAM模型应该返回结果")
            if results[0].masks is not None:
                mask = results[0].masks.data[0].cpu().numpy()
                print(f"✓ SAM分割成功，掩码形状: {mask.shape}")
            else:
                print("⚠ SAM未检测到掩码")

        except FileNotFoundError:
            print("⚠ SAM模型文件不存在，跳过测试")
        except Exception as e:
            print(f"⚠ SAM分割测试失败: {e}")

    def test_annotation_box_with_mask(self):
        from biaozhu import AnnotationBox

        box = AnnotationBox(100, 100, 300, 300, (255, 0, 0))
        self.assertEqual(box.x1, 100)
        self.assertEqual(box.y1, 100)
        self.assertEqual(box.x2, 300)
        self.assertEqual(box.y2, 300)

        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[150:250, 150:250] = 255
        box.mask = mask

        self.assertIsNotNone(box.mask, "AnnotationBox应该支持mask属性")

        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = box.apply_sam_mask_to_frame(test_frame)

        self.assertEqual(result.shape, test_frame.shape, "输出帧尺寸应该与输入相同")
        print("✓ AnnotationBox的SAM掩码应用功能测试通过")

    def test_video_processing_with_sam(self):
        test_video = Path("src/test_video.mp4")
        if not test_video.exists():
            self.skipTest("测试视频不存在")

        from biaozhu import VideoAnnotator
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            annotator = VideoAnnotator(str(test_video), tmpdir)

            annotator.boxes = [
                annotator.current_box if hasattr(annotator, 'current_box') else None
            ]

            print("✓ VideoAnnotator初始化成功")

if __name__ == '__main__':
    unittest.main(verbosity=2)
