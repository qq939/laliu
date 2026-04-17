# SAM模型分割标注使用说明

## 功能概述

视频标注工具现在集成了**SAM3VideoPredictor**模型，可以将用户绘制的边界框作为正样本提示，对视频中的目标进行**实时实例分割跟踪**。

### 核心特性

- ✅ **视频级分割**: 使用SAM3VideoPredictor进行端到端视频处理
- ✅ **实例跟踪**: 自动跟踪多个目标实例，跨帧保持一致性
- ✅ **边界框提示**: 使用用户绘制的边界框作为正样本提示
- ✅ **多目标支持**: 同时跟踪多个目标，每个目标独立分割
- ✅ **智能回退**: 如果SAM3不可用，自动回退到SAM图片分割或简单矩形框

## 工作流程

### 1. 标注阶段
- 运行 `python biaozhu.py`
- 在视频第一帧上使用鼠标框选要分割的目标
- 可以框选多个目标（每个框颜色不同）
- 点击绿色"完成标注"按钮

### 2. SAM3视频分割阶段
程序会自动：
1. 加载SAM3VideoPredictor（首次运行会自动下载 `sam3.pt`）
2. 使用您绘制的边界框作为提示（bbox prompt）
3. 对整个视频进行实时实例分割跟踪
4. 在每一帧上生成精确的目标掩码（mask）

### 3. 视频生成阶段
- 对视频的每一帧应用分割掩码
- 标注区域半透明高亮显示
- 显示目标编号标签
- 输出到 `dst` 目录

## SAM3模型说明

### 为什么使用SAM3VideoPredictor？

SAM3VideoPredictor vs SAM图片分割：

| 特性 | SAM3VideoPredictor | SAM图片分割 |
|------|-------------------|------------|
| 处理方式 | 端到端视频处理 | 逐帧处理 |
| 实例跟踪 | ✅ 自动跟踪多实例 | ❌ 无跟踪 |
| 时序一致性 | ✅ 跨帧掩码一致 | ❌ 每帧独立 |
| 性能 | ⚡ 高度优化 | 🐢 较慢 |
| 内存效率 | ✅ 共享特征 | ❌ 重复计算 |

### 可用模型

1. **sam3.pt** (推荐)
   - 最新SAM3模型，支持视频实例分割
   - 自动下载大小: ~3.5GB
   - 包含文本提示和概念分割功能

2. **sam_b.pt** (备选)
   - 基础版本，适合简单场景
   - 自动下载大小: ~375MB
   - 如果SAM3不可用，会回退使用

### 模型配置

在 [biaozhu.py](file:///Users/jimjiang/Downloads/biaozhu/biaozhu.py) 第5行修改模型路径：

```python
SAM_MODEL_PATH = "sam3.pt"  # 默认使用SAM3
# 或使用: SAM_MODEL_PATH = "sam_b.pt"  # 备选模型
```

## 技术细节

### 分割原理

SAM3VideoPredictor接收边界框作为视觉提示：
- **输入**: 视频 + 边界框列表 + 标签
- **处理**: 端到端视频分割推理
- **输出**: 每帧的实例掩码，保持实例ID一致性

### 代码流程

```
用户绘制边界框
    ↓
[biaozhu.py:248-270] process_video()
    ↓
加载SAM3VideoPredictor
    ↓
调用: predictor(source=video, bboxes=bboxes, labels=labels, stream=True)
    ↓
实时处理视频流，生成每帧的分割掩码
    ↓
对每帧应用掩码渲染和标签
    ↓
生成标注视频
```

### 关键代码

#### SAM3视频分割调用 ([biaozhu.py:248-320](file:///Users/jimjiang/Downloads/biaozhu/biaozhu.py#L248-L320))

```python
from ultralytics.models.sam import SAM3VideoPredictor

# 创建视频预测器
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model="sam3.pt",
    half=False,
    save=False,
    verbose=False
)
predictor = SAM3VideoPredictor(overrides=overrides)

# 准备边界框
bboxes = [[box.x1, box.y1, box.x2, box.y2] for box in self.boxes]
labels = [1] * len(bboxes)  # 1表示正样本

# 进行视频实例分割跟踪
results = predictor(
    source=video_path,
    bboxes=bboxes,
    labels=labels,
    stream=True
)

# 处理每帧结果
for r in results:
    annotated_frame = r.plot()  # SAM自动渲染分割结果
    # 添加中文标签
    annotated_frame = put_chinese_text(...)
    out.write(annotated_frame)
```

#### 智能回退机制 ([biaozhu.py:325-430](file:///Users/jimjiang/Downloads/biaozhu/biaozhu.py#L325-L430))

```python
try:
    # 优先使用SAM3VideoPredictor
    from ultralytics.models.sam import SAM3VideoPredictor
    predictor = SAM3VideoPredictor(overrides=overrides)
    # 进行视频分割...
except ImportError:
    print("SAM3不可用，回退到SAM图片分割")
    # 回退逻辑...
except Exception:
    print("SAM不可用，回退到简单矩形框")
    # 简单矩形框...
```

### 实例跟踪说明

SAM3VideoPredictor的实例跟踪特性：
- **实例ID**: 每个目标分配唯一ID，跨帧保持
- **掩码一致性**: 同一实例在不同帧的掩码保持一致
- **遮挡处理**: 自动处理目标遮挡和重现
- **多目标**: 支持同时跟踪任意数量的目标实例

## 测试验证

### 1. SAM3视频预测器测试

```bash
python test_sam3_video.py
```

测试项目：
- ✅ SAM3VideoPredictor导入
- ✅ 视频分割推理
- ✅ 掩码生成
- ✅ 多帧处理

**测试结果示例**：
```
测试SAM3视频分割预测器...
正在加载SAM3视频预测器...
✓ SAM3视频预测器加载成功
正在测试视频分割，跟踪bbox: [[200, 150, 400, 350]]
✓ 成功处理第1帧
  - 检测到 1 个掩码
  - 帧 1 分割完成
✓ SAM3视频分割测试通过 (处理了 3 帧)
```

### 2. 完整标注流程测试

```bash
python biaozhu.py
```

预期输出：
```
正在加载SAM3视频分割模型...
SAM3视频模型加载成功: sam3.pt
正在使用SAM3进行视频实例分割跟踪...
将跟踪 2 个目标实例
正在生成标注视频...
已处理 30 帧
已处理 60 帧
...
✓ 标注视频已保存到: dst/video_annotated.mp4
✓ 共处理 90 帧
✓ 标注了 2 个目标区域
```

## 依赖说明

### 必需依赖

- `opencv-python`: 视频处理和图像显示
- `numpy`: 数值计算
- `Pillow`: 中文字体渲染
- `ultralytics>=8.3.237`: SAM3模型推理框架

### 安装命令

```bash
uv pip install opencv-python numpy Pillow ultralytics
```

### 版本要求

- **Python**: 3.8+
- **PyTorch**: 2.0+ (推荐2.7+)
- **Ultralytics**: 8.3.237+ (SAM3支持)
- **CUDA**: 可选，但推荐使用GPU加速

## 性能考虑

### ⚡ 性能优化

SAM3VideoPredictor已经高度优化：
- **批量处理**: 自动批处理帧
- **流式处理**: 使用stream=True减少内存占用
- **特征缓存**: 共享图像特征，避免重复计算
- **GPU加速**: 自动使用GPU（如果可用）

### 📊 性能基准

| 场景 | 帧率 | 说明 |
|------|------|------|
| GPU (H200) | ~30 FPS | 接近实时 |
| GPU (RTX) | ~15-20 FPS | 实时可用 |
| CPU | ~5-8 FPS | 演示可用 |

### ⚠️ 内存考虑

- SAM3模型约3.5GB
- 视频内存取决于分辨率
- 建议：处理大视频时使用stream=True

## 常见问题

### Q1: SAM3模型下载失败怎么办？

```bash
# 手动下载
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/sam3.pt

# 或使用镜像
wget https://huggingface.co/nick698/ultralytics-sam3/resolve/main/sam3.pt
```

### Q2: SAM3VideoPredictor导入失败？

确保安装正确版本的ultralytics：

```bash
pip install -U ultralytics
```

如果仍有导入问题，检查PyTorch版本：

```python
import torch
print(torch.__version__)  # 应该 >= 2.0
```

### Q3: 如何处理多个目标？

SAM3VideoPredictor原生支持多目标：

```python
# 多个边界框，每个框一个目标
bboxes = [
    [100, 100, 200, 200],  # 目标1
    [300, 150, 400, 300],  # 目标2
    [500, 200, 600, 350],  # 目标3
]
labels = [1, 1, 1]  # 全部为正样本

results = predictor(source=video, bboxes=bboxes, labels=labels, stream=True)
```

### Q4: 如何提高分割精度？

1. **调整边界框**: 尽量贴合目标边界
2. **调整置信度**: 修改conf参数（默认0.25）
3. **使用SAM3**: 推荐使用sam3.pt模型
4. **GPU加速**: 使用GPU可以获得更稳定的推理

## 参考资料

- [Ultralytics SAM3文档](https://docs.ultralytics.com/zh/models/sam-3/)
- [SAM3 GitHub仓库](https://github.com/RizwanMunawar/sam3-inference)
- [SAM3论文](https://arxiv.org/abs/2511.16719)
- [SAM2对比](https://docs.ultralytics.com/models/sam-2/)

## 更新日志

### v2.0 (当前版本)
- ✨ 添加SAM3VideoPredictor支持
- ✅ 视频级实例分割跟踪
- 🔄 智能回退机制（SAM3 → SAM → 矩形框）
- ⚡ 性能优化（流式处理）

### v1.0
- 初始版本
- SAM图片分割支持
- 简单矩形框标注
