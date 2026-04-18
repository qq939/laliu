# global参数
FIND = []  # 第1行：文本提示词列表，运行时由用户输入，用于SAM3语义分割
SRC_DIR = "src"  # 第31行：视频源目录
DST_DIR = "dst"  # 第67行：输出视频目录
WINDOW_NAME = "视频标注工具"  # 第37行：窗口名称
SAM_MODEL_PATH = "sam3.pt"  # SAM模型路径（可下载sam_b.pt或sam3.pt）
BOX_COLORS = [  # 第55行：标注框颜色列表
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
from dataclasses import dataclass
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont

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

def get_video_extension(video_path: str) -> str:
    """获取原视频的扩展名"""
    return Path(video_path).suffix.lower()

def get_output_filename(video_path: str) -> str:
    """生成输出文件名，避免双点"""
    video_name = Path(video_path).stem
    video_ext = Path(video_path).suffix.lower()

    video_name = video_name.replace("..", "_")

    while "__" in video_name:
        video_name = video_name.replace("__", "_")

    if video_name.endswith("_"):
        video_name = video_name[:-1]

    return f"{video_name}_annotated{video_ext}"

@dataclass
class AnnotationBox:
    x1: int
    y1: int
    x2: int
    y2: int
    color: Tuple[int, int, int]
    mask: np.ndarray = None

    def normalize(self):
        if self.x1 > self.x2:
            self.x1, self.x2 = self.x2, self.x1
        if self.y1 > self.y2:
            self.y1, self.y2 = self.y2, self.y1

    def to_bbox_mask(self, height: int, width: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        x1, y1 = max(0, self.x1), max(0, self.y1)
        x2, y2 = min(width, self.x2), min(height, self.y2)
        mask[y1:y2, x1:x2] = 255
        return mask

    def apply_mask_to_frame(self, frame: np.ndarray, color: Tuple[int, int, int] = None) -> np.ndarray:
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
            frame[mask_bool], 0.5,
            colored_mask[mask_bool], 0.5, 0
        )
        return frame_with_box

    def apply_sam_mask_to_frame(self, frame: np.ndarray, color: Tuple[int, int, int] = None) -> np.ndarray:
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

class VideoAnnotator:
    def __init__(self, video_path: str, output_dir: str):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        self.ret, self.frame = self.cap.read()
        if not self.ret:
            raise ValueError("无法读取视频帧")

        self.boxes: List[AnnotationBox] = []
        self.current_box: AnnotationBox = None
        self.drawing = False
        self.start_point = None
        self.color_index = 0
        self.button_clicked = False

        self.window_name = WINDOW_NAME
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback, self)

    def mouse_callback(self, event, x, y, flags, param):
        annotator = param

        if event == cv2.EVENT_LBUTTONDOWN:
            annotator.drawing = True
            annotator.start_point = (x, y)
            color = BOX_COLORS[annotator.color_index % len(BOX_COLORS)]
            annotator.current_box = AnnotationBox(x, y, x, y, color)

        elif event == cv2.EVENT_MOUSEMOVE:
            if annotator.drawing and annotator.current_box:
                annotator.current_box.x2 = x
                annotator.current_box.y2 = y

        elif event == cv2.EVENT_LBUTTONUP:
            button_x1 = annotator.frame.shape[1] - 150
            button_x2 = button_x1 + 130
            button_y1 = 30
            button_y2 = button_y1 + 40

            if button_x1 <= x <= button_x2 and button_y1 <= y <= button_y2:
                annotator.button_clicked = True
                print("点击了完成按钮")
            elif annotator.drawing and annotator.current_box:
                annotator.current_box.x2 = x
                annotator.current_box.y2 = y
                annotator.current_box.normalize()
                annotator.boxes.append(annotator.current_box)
                annotator.color_index += 1
                annotator.drawing = False
                annotator.current_box = None
            else:
                annotator.drawing = False
                annotator.current_box = None

    def draw_boxes(self, frame: np.ndarray) -> np.ndarray:
        display_frame = frame.copy()

        for box in self.boxes:
            cv2.rectangle(display_frame,
                         (box.x1, box.y1),
                         (box.x2, box.y2),
                         box.color, 2)
            label = f"目标 {self.boxes.index(box) + 1}"
            display_frame = put_chinese_text(display_frame, label,
                                            (box.x1, box.y1 - 10),
                                            font_size=15, color=box.color)

        if self.current_box and self.drawing:
            cv2.rectangle(display_frame,
                         (self.current_box.x1, self.current_box.y1),
                         (self.current_box.x2, self.current_box.y2),
                         self.current_box.color, 2)

        return display_frame

    def add_complete_button(self, frame: np.ndarray) -> np.ndarray:
        button_text = "完成标注"
        button_pos = (frame.shape[1] - 150, 30)
        button_size = (130, 40)

        cv2.rectangle(frame,
                     (button_pos[0], button_pos[1]),
                     (button_pos[0] + button_size[0], button_pos[1] + button_size[1]),
                     (0, 255, 0), -1)
        frame = put_chinese_text(frame, button_text,
                                (button_pos[0] + 15, button_pos[1] + 8),
                                font_size=18, color=(255, 255, 255))

        return frame

    def show_instructions(self, frame: np.ndarray) -> np.ndarray:
        instructions = [
            "操作说明:",
            "1. 鼠标左键框选目标",
            "2. 可框选多个目标",
            "3. 按 'c' 撤销最后一个框",
            "4. 按 'q' 退出",
            "5. 点击绿色按钮完成标注"
        ]

        for i, text in enumerate(instructions):
            frame = put_chinese_text(frame, text,
                                    (10, 30 + i * 25),
                                    font_size=16, color=(255, 255, 255))

        return frame

    def run(self):
        while True:
            display_frame = self.draw_boxes(self.frame)
            display_frame = self.add_complete_button(display_frame)
            display_frame = self.show_instructions(display_frame)

            cv2.imshow(self.window_name, display_frame)

            if self.button_clicked:
                self.process_video()
                break

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("用户退出")
                break
            elif key == ord('c'):
                if self.boxes:
                    removed = self.boxes.pop()
                    self.color_index = max(0, self.color_index - 1)
                    print(f"已撤销: {removed}")
                else:
                    print("没有可撤销的标注框")

        cv2.destroyAllWindows()
        self.cap.release()

    def process_video(self):
        if not self.boxes:
            print("没有标注框，不生成视频")
            return

        bboxes = [[box.x1, box.y1, box.x2, box.y2] for box in self.boxes]

        try:
            from ultralytics.models.sam import SAM3VideoSemanticPredictor
            print("正在加载SAM3视频分割模型...")

            overrides = dict(
                conf=0.25,
                task="segment",
                mode="predict",
                model=SAM_MODEL_PATH,
                half=False,
                save=True,
                verbose=False
            )
            predictor = SAM3VideoSemanticPredictor(overrides=overrides)
            print(f"SAM3视频模型加载成功: {SAM_MODEL_PATH}")

            print(f"正在使用SAM3进行视频实例分割跟踪...")
            if FIND:
                print(f"文本提示词: {FIND}")
            else:
                print("未提供文本提示词，将使用边界框进行分割")
            print(f"将跟踪 {len(self.boxes)} 个目标实例")

            output_filename = get_output_filename(self.video_path)
            output_path = self.output_dir / output_filename

            fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = ''.join([
                chr(fourcc_int & 0xFF),
                chr((fourcc_int >> 8) & 0xFF),
                chr((fourcc_int >> 16) & 0xFF),
                chr((fourcc_int >> 24) & 0xFF)
            ])
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            results = predictor(
                source=self.video_path,
                bboxes=bboxes,
                text=FIND,
                labels=[1] * len(bboxes),
                stream=True
            )

            frame_count = 0
            print("正在生成标注视频...")
            for r in results:
                annotated_frame = r.plot()
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

                for i, box in enumerate(self.boxes):
                    label = f"目标 {i + 1}"
                    annotated_frame_rgb = put_chinese_text(
                        annotated_frame_rgb,
                        label,
                        (box.x1, max(10, box.y1 - 10)),
                        font_size=15,
                        color=box.color
                    )

                out.write(annotated_frame_rgb)
                frame_count += 1

                if frame_count % 30 == 0:
                    print(f"已处理 {frame_count} 帧")

            out.release()
            print(f"✓ 标注视频已保存到: {output_path}")
            print(f"✓ 共处理 {frame_count} 帧")
            print(f"✓ 标注了 {len(self.boxes)} 个目标区域")
            upload_to_obs(str(output_path))

        except ImportError as e:
            print(f"SAM3VideoPredictor导入失败: {e}")
            print("正在回退到SAM图片分割模式...")

            try:
                from ultralytics import SAM
                print("正在加载SAM模型...")
                sam_model = SAM(SAM_MODEL_PATH)
                print(f"SAM模型加载成功: {SAM_MODEL_PATH}")

                print("正在使用SAM模型进行智能分割...")
                print("注意: SAM分割可能需要一些时间，请耐心等待...")

                for i, box in enumerate(self.boxes):
                    print(f"正在分割目标 {i+1}/{len(self.boxes)}...")
                    bbox = [box.x1, box.y1, box.x2, box.y2]

                    try:
                        results = sam_model(self.frame, bboxes=[bbox], verbose=False)

                        if results and results[0].masks is not None:
                            mask = results[0].masks.data[0].cpu().numpy()
                            mask = (mask * 255).astype(np.uint8)
                            box.mask = mask
                            print(f"  ✓ 目标 {i+1} 分割完成")
                        else:
                            print(f"  ⚠ 目标 {i+1} SAM未检测到掩码，使用矩形框")
                    except Exception as e:
                        print(f"  ✗ 目标 {i+1} 分割失败: {e}")
                        print(f"  → 使用矩形框替代")

                output_filename = get_output_filename(self.video_path)
                output_path = self.output_dir / output_filename

                fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
                fourcc_str = ''.join([
                    chr(fourcc_int & 0xFF),
                    chr((fourcc_int >> 8) & 0xFF),
                    chr((fourcc_int >> 16) & 0xFF),
                    chr((fourcc_int >> 24) & 0xFF)
                ])
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0

                print("正在生成标注视频...")
                while True:
                    ret, frame = self.cap.read()
                    if not ret:
                        break

                    annotated_frame = frame.copy()
                    for box in self.boxes:
                        if box.mask is not None:
                            annotated_frame = box.apply_sam_mask_to_frame(annotated_frame)
                        else:
                            annotated_frame = box.apply_mask_to_frame(annotated_frame)
                            cv2.rectangle(annotated_frame,
                                        (box.x1, box.y1),
                                        (box.x2, box.y2),
                                        box.color, 2)

                        label = f"目标 {self.boxes.index(box) + 1}"
                        annotated_frame = put_chinese_text(annotated_frame, label,
                                                        (box.x1, box.y1 - 10),
                                                        font_size=15, color=box.color)

                    out.write(annotated_frame)
                    frame_count += 1

                    if frame_count % 30 == 0:
                        print(f"已处理 {frame_count} 帧")

                out.release()
                print(f"✓ 标注视频已保存到: {output_path}")
                print(f"✓ 共处理 {frame_count} 帧")
                print(f"✓ 标注了 {len(self.boxes)} 个目标区域")
                upload_to_obs(str(output_path))

            except Exception as e:
                print(f"SAM模型加载失败: {e}")
                print("将使用简单的矩形框标注")

                output_filename = get_output_filename(self.video_path)
                output_path = self.output_dir / output_filename

                fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
                fourcc_str = ''.join([
                    chr(fourcc_int & 0xFF),
                    chr((fourcc_int >> 8) & 0xFF),
                    chr((fourcc_int >> 16) & 0xFF),
                    chr((fourcc_int >> 24) & 0xFF)
                ])
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0

                print("正在生成标注视频...")
                while True:
                    ret, frame = self.cap.read()
                    if not ret:
                        break

                    annotated_frame = frame.copy()
                    for box in self.boxes:
                        annotated_frame = box.apply_mask_to_frame(annotated_frame)
                        cv2.rectangle(annotated_frame,
                                    (box.x1, box.y1),
                                    (box.x2, box.y2),
                                    box.color, 2)

                        label = f"目标 {self.boxes.index(box) + 1}"
                        annotated_frame = put_chinese_text(annotated_frame, label,
                                                        (box.x1, box.y1 - 10),
                                                        font_size=15, color=box.color)

                    out.write(annotated_frame)
                    frame_count += 1

                    if frame_count % 30 == 0:
                        print(f"已处理 {frame_count} 帧")

                out.release()
                print(f"✓ 标注视频已保存到: {output_path}")
                print(f"✓ 共处理 {frame_count} 帧")
                print(f"✓ 标注了 {len(self.boxes)} 个目标区域")
                upload_to_obs(str(output_path))

def main():
    global FIND
    FIND = []

    print("=" * 50)
    print("视频标注工具 - SAM3实例分割")
    print("=" * 50)
    print("\n请输入要查找的物品名称（可输入多个）：")
    print("输入 'done' 表示完成输入")
    print("-" * 50)

    while True:
        item = input("物品名称: ").strip()
        if not item:
            print("物品名称不能为空，请重新输入")
            continue
        if item.lower() == 'done':
            if len(FIND) == 0:
                print("至少需要输入一个物品名称！")
                continue
            break
        if item not in FIND:
            FIND.append(item)
            print(f"  ✓ 已添加: {item}")
        else:
            print(f"  ⚠ 已存在: {item}")

    print("-" * 50)
    print(f"已添加 {len(FIND)} 个物品: {', '.join(FIND)}")
    print("=" * 50)

    video_files = list(Path(SRC_DIR).glob("*.mp4"))

    if not video_files:
        print(f"\n在 {SRC_DIR} 目录下没有找到视频文件")
        print("请将视频文件放入 src 目录")
        return

    print("\n找到以下视频文件:")
    for i, video_file in enumerate(video_files, 1):
        print(f"{i}. {video_file.name}")

    if len(video_files) == 1:
        video_path = str(video_files[0])
    else:
        choice = input("\n请选择要标注的视频编号: ")
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(video_files):
                video_path = str(video_files[idx])
            else:
                print("无效的选择")
                return
        except ValueError:
            print("请输入有效的数字")
            return

    print(f"\n开始标注: {video_path}")
    annotator = VideoAnnotator(video_path, DST_DIR)
    annotator.run()

if __name__ == "__main__":
    main()
