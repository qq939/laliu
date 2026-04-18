import sys
import time
import tempfile
from pathlib import Path

def test_import():
    """测试导入"""
    print("测试导入...")
    try:
        import cv2
        import numpy as np
        import gradio as gr
        from PIL import Image
        print("✓ 所有依赖导入成功")
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_basic_functionality():
    """测试基本功能"""
    print("\n测试基本功能...")
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        
        from annotate_image import (
            AnnotationBox, ImageAnnotator, put_chinese_text,
            BOX_COLORS, DST_DIR
        )
        
        print("✓ 成功导入 annotate_image 模块")
        
        box = AnnotationBox(100, 100, 200, 200, BOX_COLORS[0])
        print(f"✓ 创建标注框成功: {box.x1}, {box.y1}, {box.x2}, {box.y2}")
        
        box.normalize()
        print(f"✓ 标准化坐标成功: {box.x1}, {box.y1}, {box.x2}, {box.y2}")
        
        center = box.get_center()
        print(f"✓ 计算中心点成功: {center}")
        
        area = box.get_area()
        print(f"✓ 计算面积成功: {area}")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gradio_interface():
    """测试 Gradio 界面创建"""
    print("\n测试 Gradio 界面创建...")
    try:
        from annotate_image import create_gradio_interface
        
        demo = create_gradio_interface()
        print("✓ Gradio 界面创建成功")
        print(f"✓ 界面标题: {demo.title}")
        
        return True
        
    except Exception as e:
        print(f"✗ Gradio 界面创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("图片标注工具 - Web UI 测试")
    print("=" * 60)
    
    tests = [
        ("导入测试", test_import),
        ("基本功能测试", test_basic_functionality),
        ("Gradio界面测试", test_gradio_interface),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"执行: {test_name}")
        print('='*60)
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"\n✓ {test_name} 通过")
            else:
                print(f"\n✗ {test_name} 失败")
                
        except Exception as e:
            print(f"\n✗ {test_name} 执行出错: {e}")
            results.append((test_name, False))
        
        time.sleep(0.5)
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status} - {test_name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！Web UI 已准备就绪")
        print("运行命令: python annotate_image.py")
        return True
    else:
        print("\n⚠ 部分测试失败，请检查错误信息")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ 测试执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
