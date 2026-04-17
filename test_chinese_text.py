import cv2
import numpy as np
from pathlib import Path
from biaozhu import put_chinese_text

def test_chinese_text_rendering():
    """测试中文文本渲染功能"""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (100, 100, 100)

    text = "视频标注工具"
    result = put_chinese_text(img, text, (100, 100), font_size=30, color=(255, 255, 255))

    assert result is not None, "中文文本渲染失败"
    assert result.shape == img.shape, "图像尺寸应该保持不变"

    Path("dst").mkdir(exist_ok=True)
    cv2.imwrite("dst/test_chinese_text.png", result)
    print("✅ 中文文本测试通过")
    print("测试图片已保存到: dst/test_chinese_text.png")

if __name__ == "__main__":
    test_chinese_text_rendering()
