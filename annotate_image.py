# global参数
FIND = []  # 第5行：文本提示词列表，运行时由用户输入，用于SAM3语义分割
SRC_DIR = "src"  # 第32行：图片源目录
DST_DIR = "dst"  # 第69行：输出图片目录
SAM_MODEL_PATH = "sam3.pt"  # SAM模型路径
BOX_COLORS = [  # 第56行：标注框颜色列表
    (255, 0, 0),      # 蓝色
    (0, 255, 0),      # 绿色
    (0, 0, 255),      # 红色
    (255, 255, 0),    # 青色
    (255, 0, 255),    # 紫色
    (0, 255, 255),    # 黄色
    (255, 128, 0),    # 橙色
    (128, 0, 255),    # 紫红色
]

import cv2
import numpy as np
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
import tempfile
import json
import base64
from datetime import datetime

def put_chinese_text(img, text, position, font_size=20, color=(255, 255, 255)):
    """在图像上绘制中文文本（使用UTF-8编码）"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", font_size)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/STHeiti Light.ttc", font_size)
        except:
            font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def upload_to_obs(file_path: str):
    """上传文件到OBS云存储"""
    obs_url = f"http://obs.dimond.top/{Path(file_path).name}"
    try:
        print(f"正在上传文件到OBS: {obs_url}")
        result = subprocess.run(
            ["curl", "--upload-file", file_path, obs_url],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            print(f"✓ 文件上传成功: {obs_url}")
            return True
        else:
            print(f"✗ 文件上传失败: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ 文件上传超时")
        return False
    except Exception as e:
        print(f"✗ 上传出错: {e}")
        return False

def get_output_filename(image_path: str, suffix: str = "_annotated") -> str:
    """生成输出文件名"""
    image_name = Path(image_path).stem
    image_ext = Path(image_path).suffix.lower()
    
    image_name = image_name.replace("..", "_")
    while "__" in image_name:
        image_name = image_name.replace("__", "_")
    if image_name.endswith("_"):
        image_name = image_name[:-1]
    
    return f"{image_name}{suffix}{image_ext}"

@dataclass
class AnnotationBox:
    x1: int
    y1: int
    x2: int
    y2: int
    color: Tuple[int, int, int]
    mask: Optional[np.ndarray] = None
    label: str = ""
    
    def normalize(self):
        """确保坐标正确（左上右下）"""
        if self.x1 > self.x2:
            self.x1, self.x2 = self.x2, self.x1
        if self.y1 > self.y2:
            self.y1, self.y2 = self.y2, self.y1
    
    def to_bbox_mask(self, height: int, width: int) -> np.ndarray:
        """将边界框转换为掩码"""
        mask = np.zeros((height, width), dtype=np.uint8)
        x1, y1 = max(0, self.x1), max(0, self.y1)
        x2, y2 = min(width, self.x2), min(height, self.y2)
        mask[y1:y2, x1:x2] = 255
        return mask
    
    def get_center(self) -> Tuple[int, int]:
        """获取边界框中心点"""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    def get_area(self) -> int:
        """获取边界框面积"""
        return abs(self.x2 - self.x1) * abs(self.y2 - self.y1)
    
    def apply_mask_to_frame(self, frame: np.ndarray, color: Tuple[int, int, int] = None) -> np.ndarray:
        """应用掩码到帧"""
        if self.mask is not None:
            mask = self.mask
        else:
            mask = self.to_bbox_mask(frame.shape[0], frame.shape[1])
        
        if color is None:
            color = self.color
        
        colored_mask = np.zeros_like(frame)
        colored_mask[:] = color
        frame_with_box = frame.copy()
        mask_bool = mask > 0
        frame_with_box[mask_bool] = cv2.addWeighted(
            frame[mask_bool], 0.3,
            colored_mask[mask_bool], 0.7, 0
        )
        return frame_with_box
    
    def apply_sam_mask_to_frame(self, frame: np.ndarray, color: Tuple[int, int, int] = None) -> np.ndarray:
        """应用SAM掩码到帧"""
        if self.mask is None:
            return self.apply_mask_to_frame(frame, color)
        
        if color is None:
            color = self.color
        
        mask = self.mask
        colored_mask = np.zeros_like(frame)
        colored_mask[:] = color
        frame_with_mask = frame.copy()
        mask_bool = mask > 0
        frame_with_mask[mask_bool] = cv2.addWeighted(
            frame[mask_bool], 0.3,
            colored_mask[mask_bool], 0.7, 0
        )
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame_with_mask, contours, -1, color, 2)
        
        return frame_with_mask

class ImageAnnotator:
    """图片标注器"""
    
    def __init__(self, image_path: str, output_dir: str):
        self.image_path = image_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        self.boxes: List[AnnotationBox] = []
        self.color_index = 0
        self.current_image = self.image.copy()
    
    def add_box(self, x1: int, y1: int, x2: int, y2: int, label: str = ""):
        """添加标注框"""
        color = BOX_COLORS[self.color_index % len(BOX_COLORS)]
        box = AnnotationBox(x1, y1, x2, y2, color, label=label)
        box.normalize()
        self.boxes.append(box)
        self.color_index += 1
        return box
    
    def remove_box(self, index: int):
        """移除标注框"""
        if 0 <= index < len(self.boxes):
            self.boxes.pop(index)
            return True
        return False
    
    def clear_boxes(self):
        """清空所有标注框"""
        self.boxes.clear()
        self.color_index = 0
    
    def get_next_color(self) -> Tuple[int, int, int]:
        """获取下一个颜色"""
        return BOX_COLORS[self.color_index % len(BOX_COLORS)]
    
    def render_boxes(self, image: np.ndarray) -> np.ndarray:
        """渲染所有标注框"""
        result = image.copy()
        
        for i, box in enumerate(self.boxes):
            cv2.rectangle(result,
                         (box.x1, box.y1),
                         (box.x2, box.y2),
                         box.color, 2)
            
            label = box.label if box.label else f"目标 {i + 1}"
            result = put_chinese_text(result, label,
                                    (box.x1, max(10, box.y1 - 10)),
                                    font_size=15, color=box.color)
        
        return result
    
    def apply_sam_segmentation(self, use_fallback: bool = False):
        """应用SAM分割"""
        if not self.boxes:
            return
        
        bboxes = [[box.x1, box.y1, box.x2, box.y2] for box in self.boxes]
        
        try:
            if not use_fallback:
                from ultralytics.models.sam import SAM3VideoSemanticPredictor
                print("正在加载SAM3视频分割模型...")
                
                overrides = dict(
                    conf=0.25,
                    task="segment",
                    mode="predict",
                    model=SAM_MODEL_PATH,
                    half=False,
                    save=False,
                    verbose=False
                )
                predictor = SAM3VideoSemanticPredictor(overrides=overrides)
                print(f"SAM3模型加载成功: {SAM_MODEL_PATH}")
                
                print(f"正在使用SAM3进行图片分割...")
                results = predictor(
                    source=self.image_path,
                    bboxes=bboxes,
                    text=FIND if FIND else None,
                    labels=[1] * len(bboxes),
                    stream=False
                )
                
                if results and len(results) > 0:
                    for i, r in enumerate(results):
                        if i < len(self.boxes) and r.masks is not None:
                            mask = r.masks.data[0].cpu().numpy()
                            mask = (mask * 255).astype(np.uint8)
                            self.boxes[i].mask = mask
                            print(f"✓ 目标 {i+1} 分割完成")
                
            else:
                from ultralytics import SAM
                print("正在加载SAM模型...")
                sam_model = SAM(SAM_MODEL_PATH)
                print(f"SAM模型加载成功: {SAM_MODEL_PATH}")
                
                for i, box in enumerate(self.boxes):
                    print(f"正在分割目标 {i+1}/{len(self.boxes)}...")
                    bbox = [box.x1, box.y1, box.x2, box.y2]
                    
                    try:
                        results = sam_model(self.image, bboxes=[bbox], verbose=False)
                        
                        if results and results[0].masks is not None:
                            mask = results[0].masks.data[0].cpu().numpy()
                            mask = (mask * 255).astype(np.uint8)
                            self.boxes[i].mask = mask
                            print(f"  ✓ 目标 {i+1} 分割完成")
                        else:
                            print(f"  ⚠ 目标 {i+1} SAM未检测到掩码")
                    except Exception as e:
                        print(f"  ✗ 目标 {i+1} 分割失败: {e}")
                        
        except ImportError as e:
            print(f"SAM3VideoSemanticPredictor导入失败: {e}")
            if not use_fallback:
                self.apply_sam_segmentation(use_fallback=True)
        except Exception as e:
            print(f"SAM分割失败: {e}")
            if not use_fallback:
                try:
                    self.apply_sam_segmentation(use_fallback=True)
                except:
                    pass
    
    def save_annotated_image(self, filename: str = None) -> str:
        """保存标注图片"""
        if filename is None:
            filename = get_output_filename(self.image_path)
        
        output_path = self.output_dir / filename
        
        annotated = self.image.copy()
        for box in self.boxes:
            if box.mask is not None:
                annotated = box.apply_sam_mask_to_frame(annotated)
            else:
                annotated = box.apply_mask_to_frame(annotated)
                cv2.rectangle(annotated,
                            (box.x1, box.y1),
                            (box.x2, box.y2),
                            box.color, 2)
            
            label = f"目标 {self.boxes.index(box) + 1}"
            annotated = put_chinese_text(annotated, label,
                                      (box.x1, max(10, box.y1 - 10)),
                                      font_size=15, color=box.color)
        
        cv2.imwrite(str(output_path), annotated)
        print(f"✓ 标注图片已保存到: {output_path}")
        
        return str(output_path)
    
    def save_masks(self, filename: str = None) -> List[str]:
        """保存分割掩码"""
        if filename is None:
            filename_base = get_output_filename(self.image_path, suffix="_mask")
        else:
            filename_base = Path(filename).stem
        
        mask_paths = []
        for i, box in enumerate(self.boxes):
            if box.mask is not None:
                mask_path = self.output_dir / f"{filename_base}_{i+1}.png"
                cv2.imwrite(str(mask_path), box.mask)
                mask_paths.append(str(mask_path))
                print(f"✓ 掩码 {i+1} 已保存到: {mask_path}")
        
        return mask_paths
    
    def save_annotations_json(self, filename: str = None) -> str:
        """保存标注数据为JSON"""
        if filename is None:
            filename = get_output_filename(self.image_path, suffix="_annotations") + ".json"
        
        output_path = self.output_dir / filename
        
        annotations = {
            "image_path": self.image_path,
            "timestamp": datetime.now().isoformat(),
            "boxes": [
                {
                    "label": box.label or f"目标 {i+1}",
                    "bbox": [box.x1, box.y1, box.x2, box.y2],
                    "color": box.color,
                    "center": box.get_center(),
                    "area": box.get_area(),
                    "has_mask": box.mask is not None
                }
                for i, box in enumerate(self.boxes)
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 标注数据已保存到: {output_path}")
        return str(output_path)

def create_gradio_interface():
    """创建Gradio界面"""
    
    with gr.Blocks(
        title="图片标注工具 - SAM3实例分割",
    ) as demo:
        gr.Markdown("""
        # 🖼️ 图片标注工具
        ### 基于 SAM3 的智能实例分割标注系统
        """)
        
        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("### 📸 图片预览")
                
                image_input = gr.Image(
                    label="上传或选择图片",
                    type="filepath",
                    height=500,
                )
                
                with gr.Row():
                    upload_btn = gr.Button("📁 上传图片", variant="primary")
                    clear_btn = gr.Button("🗑️ 清空标注", variant="secondary")
                
                gr.Markdown("""
                **使用说明：**
                1. 上传或拖拽图片到上方区域
                2. 使用画笔工具绘制标注区域（点或框）
                3. 点击"开始分割"按钮进行SAM3智能分割
                4. 可添加文本提示词进行语义分割
                5. 导出标注结果
                """)
            
            with gr.Column(scale=4):
                gr.Markdown("### ✨ 标注结果")
                
                image_output = gr.Image(
                    label="标注结果预览",
                    type="numpy",
                    height=500,
                )
                
                with gr.Row():
                    download_annotated = gr.Button("💾 下载标注图片", variant="primary")
                    download_masks = gr.Button("🎭 下载掩码", variant="secondary")
                    download_json = gr.Button("📋 下载JSON", variant="secondary")
                
                status_text = gr.Textbox(label="状态信息", lines=3, interactive=False)
            
            with gr.Column(scale=3):
                gr.Markdown("### ⚙️ 参数设置")
                
                text_prompt = gr.Textbox(
                    label="文本提示词",
                    placeholder="输入要分割的物体名称（如：人、汽车、狗）",
                    lines=2,
                )
                
                add_prompt_btn = gr.Button("➕ 添加提示词", variant="secondary")
                
                prompt_list = gr.HighlightedText(
                    label="已添加的提示词",
                    container=False,
                )
                
                clear_prompts_btn = gr.Button("🗑️ 清空提示词", variant="secondary")
                
                with gr.Accordion("📊 标注统计", open=False):
                    box_count = gr.Number(label="已标注目标数", value=0, interactive=False)
                    last_action = gr.Textbox(label="最近操作", value="等待上传图片", interactive=False)
        
        current_annotator = {"instance": None, "image_path": None}
        
        def load_image(image_file):
            """加载图片"""
            if image_file is None:
                return None, {"instance": None}, gr.update(value=0), "请先上传图片"
            
            try:
                annotator = ImageAnnotator(image_file, DST_DIR)
                current_annotator["instance"] = annotator
                current_annotator["image_path"] = image_file
                
                return image_file, {"instance": annotator}, gr.update(value=0), f"✓ 成功加载图片: {Path(image_file).name}"
            except Exception as e:
                return None, {"instance": None}, gr.update(value=0), f"✗ 加载失败: {str(e)}"
        
        def add_text_prompt(prompt, current_prompts):
            """添加文本提示词"""
            global FIND
            
            if prompt and prompt.strip():
                prompt = prompt.strip()
                if prompt not in FIND:
                    FIND.append(prompt)
                    return "", [{"text": p, "color": "#3B6FB6"} for p in FIND], f"✓ 已添加提示词: {prompt}"
                else:
                    return "", [{"text": p, "color": "#3B6FB6"} for p in FIND], f"⚠ 提示词已存在: {prompt}"
            return "", [{"text": p, "color": "#3B6FB6"} for p in FIND], "请输入有效的提示词"
        
        def clear_text_prompts():
            """清空文本提示词"""
            global FIND
            FIND.clear()
            return [], "✓ 已清空所有提示词"
        
        def process_segmentation(state):
            """处理分割"""
            if state.get("instance") is None:
                return None, "请先上传图片"
            
            annotator = state["instance"]
            
            if not annotator.boxes:
                return None, "请先绘制至少一个标注区域"
            
            try:
                annotator.apply_sam_segmentation()
                
                result_image = annotator.render_boxes(annotator.image)
                result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                
                box_count = len(annotator.boxes)
                has_mask_count = sum(1 for box in annotator.boxes if box.mask is not None)
                
                status = f"✓ 分割完成！\n"
                status += f"- 总目标数: {box_count}\n"
                status += f"- 已分割: {has_mask_count}\n"
                status += f"- 待处理: {box_count - has_mask_count}"
                
                return result_image_rgb, status
                
            except Exception as e:
                return None, f"✗ 分割失败: {str(e)}"
        
        def export_annotated_image(state):
            """导出标注图片"""
            if state.get("instance") is None:
                return None, "请先上传并处理图片"
            
            try:
                annotator = state["instance"]
                output_path = annotator.save_annotated_image()
                upload_to_obs(output_path)
                
                result_image = annotator.render_boxes(annotator.image)
                result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                
                return result_image_rgb, f"✓ 标注图片已保存并上传: {Path(output_path).name}"
            except Exception as e:
                return None, f"✗ 导出失败: {str(e)}"
        
        def export_masks(state):
            """导出掩码"""
            if state.get("instance") is None:
                return "请先上传并处理图片"
            
            try:
                annotator = state["instance"]
                mask_paths = annotator.save_masks()
                
                for path in mask_paths:
                    upload_to_obs(path)
                
                return f"✓ 已导出 {len(mask_paths)} 个掩码文件"
            except Exception as e:
                return f"✗ 导出掩码失败: {str(e)}"
        
        def export_json(state):
            """导出JSON"""
            if state.get("instance") is None:
                return "请先上传并处理图片"
            
            try:
                annotator = state["instance"]
                json_path = annotator.save_annotations_json()
                upload_to_obs(json_path)
                
                return f"✓ 标注数据已保存并上传: {Path(json_path).name}"
            except Exception as e:
                return f"✗ 导出JSON失败: {str(e)}"
        
        def clear_annotations(state):
            """清空标注"""
            if state.get("instance") is None:
                return None, state, gr.update(value=0), "没有可清空的内容"
            
            try:
                annotator = state["instance"]
                annotator.clear_boxes()
                
                return annotator.image_path, state, gr.update(value=0), "✓ 已清空所有标注"
            except Exception as e:
                return None, state, gr.update(value=0), f"✗ 清空失败: {str(e)}"
        
        def update_box_count(state):
            """更新标注数量"""
            if state.get("instance") is None:
                return 0
            return len(state["instance"].boxes)
        
        def on_image_change(image_file, state):
            """图片变化时的处理"""
            if image_file is None:
                return None, state, gr.update(value=0), "等待上传图片"
            return load_image(image_file)
        
        image_input.change(
            fn=on_image_change,
            inputs=[image_input, gr.State(current_annotator)],
            outputs=[image_output, gr.State(current_annotator), box_count, status_text]
        )
        
        upload_btn.click(
            fn=load_image,
            inputs=[image_input],
            outputs=[image_output, gr.State(current_annotator), box_count, status_text]
        )
        
        add_prompt_btn.click(
            fn=add_text_prompt,
            inputs=[text_prompt, prompt_list],
            outputs=[text_prompt, prompt_list, status_text]
        )
        
        clear_prompts_btn.click(
            fn=clear_text_prompts,
            outputs=[prompt_list, status_text]
        )
        
        clear_btn.click(
            fn=clear_annotations,
            inputs=[gr.State(current_annotator)],
            outputs=[image_output, gr.State(current_annotator), box_count, status_text]
        )
        
        download_annotated.click(
            fn=export_annotated_image,
            inputs=[gr.State(current_annotator)],
            outputs=[image_output, status_text]
        )
        
        download_masks.click(
            fn=export_masks,
            inputs=[gr.State(current_annotator)],
            outputs=[status_text]
        )
        
        download_json.click(
            fn=export_json,
            inputs=[gr.State(current_annotator)],
            outputs=[status_text]
        )
    
    return demo

def main():
    """主函数"""
    print("=" * 60)
    print("图片标注工具 - SAM3实例分割")
    print("=" * 60)
    print("\n配置信息：")
    print(f"- SAM模型路径: {SAM_MODEL_PATH}")
    print(f"- 图片源目录: {SRC_DIR}")
    print(f"- 输出目录: {DST_DIR}")
    print("\n" + "=" * 60)
    
    demo = create_gradio_interface()
    
    print("\n正在启动Web UI服务...")
    print("请在浏览器中打开: http://localhost:7860")
    print("按 Ctrl+C 停止服务\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
        )
    )

if __name__ == "__main__":
    main()
