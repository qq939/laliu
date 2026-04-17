import unittest
import cv2
import numpy as np
import os
import time
import threading
from pathlib import Path

class TestBiaozhu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src_dir = Path("src")
        cls.dst_dir = Path("dst")
        cls.src_dir.mkdir(exist_ok=True)
        cls.dst_dir.mkdir(exist_ok=True)

    def test_video_file_exists(self):
        video_files = list(self.src_dir.glob("*.mp4"))
        self.assertGreater(len(video_files), 0, "src目录下应该有视频文件")

    def test_cv2_import(self):
        import cv2
        self.assertIsNotNone(cv2)

    def test_numpy_import(self):
        import numpy as np
        self.assertIsNotNone(np)

    def test_video_capture_opens(self):
        video_files = list(self.src_dir.glob("*.mp4"))
        if len(video_files) == 0:
            self.skipTest("没有测试视频文件")

        cap = cv2.VideoCapture(str(video_files[0]))
        self.assertTrue(cap.isOpened(), "视频应该能正常打开")
        ret, frame = cap.read()
        self.assertTrue(ret, "应该能读取视频帧")
        self.assertIsNotNone(frame, "帧不应该为None")
        self.assertEqual(len(frame.shape), 3, "帧应该是彩色图像")
        cap.release()

    def test_video_writer_creation(self):
        test_output = self.dst_dir / "test_output.mp4"
        if test_output.exists():
            test_output.unlink()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(test_output), fourcc, 30.0, (640, 480))
        self.assertTrue(out.isOpened(), "VideoWriter应该能正常打开")

        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out.write(test_frame)
        out.release()

        self.assertTrue(test_output.exists(), "输出视频文件应该存在")
        test_output.unlink()

    def test_annotation_data_structure(self):
        from biaozhu import AnnotationBox
        box = AnnotationBox(100, 100, 200, 200, (0, 255, 0))
        self.assertEqual(box.x1, 100)
        self.assertEqual(box.y1, 100)
        self.assertEqual(box.x2, 200)
        self.assertEqual(box.y2, 200)
        self.assertEqual(box.color, (0, 255, 0))

    def test_timeout_mechanism(self):
        timeout_seconds = 5
        start_time = time.time()

        def long_running_task():
            time.sleep(10)

        thread = threading.Thread(target=long_running_task)
        thread.start()
        thread.join(timeout=timeout_seconds)

        elapsed = time.time() - start_time
        self.assertLessEqual(elapsed, timeout_seconds + 1, "任务应该在超时时间内完成")
        if thread.is_alive():
            thread.join(1)

    def test_multiple_boxes_different_colors(self):
        from biaozhu import AnnotationBox
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        boxes = [AnnotationBox(i*10, i*10, i*10+100, i*10+100, colors[i]) for i in range(4)]
        self.assertEqual(len(boxes), 4, "应该有4个不同的框")
        for i, box in enumerate(boxes):
            self.assertEqual(box.color, colors[i], f"第{i}个框的颜色应该正确")

if __name__ == '__main__':
    unittest.main(verbosity=2, timeout=10)
