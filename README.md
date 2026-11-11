医学类中医舌象舌诊数据集
：包含21种疾病类别，可用于舌诊的目标检测。其中，训练集包含5594张图像，验证集包含572张图像，测试集包含553张图像。
三种标注格式：coco/.txt/.xml，可通过配置文件用于相关目标检测算法的实验

数据集的构建采用了标准化的舌象采集系统和专家标注框架，每张图像平均具有2-3个经执业中医师验证的标签。

整个数据集1.1G

![以下是您提供的 **舌象舌诊数据集** 的详细信息表格：](https://i-blog.csdnimg.cn/direct/a78e89916dd1466cb14bc834834f7b81.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0637117d521b43e3ab57f6333ecb41d9.png)

---

### 📊 舌象舌诊数据集描述表

| **项目** | **详情** |
|---------|--------|
| **数据集名称** | 舌象舌诊数据集 |
| **总数据量** | 约 1.1 GB |
| **图像总数** | 6,719 张（训练集 + 验证集 + 测试集） |
| **训练集图像数量** | 5,594 张 |
| **验证集图像数量** | 572 张 |
| **测试集图像数量** | 553 张 |
| **疾病类别数量** | 21 种 |
| **标注目标类型** | 舌体区域或舌象特征区域（如：裂纹、齿痕、胖大、瘦小、瘀斑等）的目标检测框 |
| **每张图像平均标签数** | 2–3 个（经执业中医师验证） |
| **标注格式** | 支持三种格式：<br> - COCO (.json)<br> - YOLO (.txt)<br> - Pascal VOC (.xml) |
| **标注来源** | 由标准化舌象采集系统采集，经执业中医师人工标注与审核 |
| **适用任务** | 多类别目标检测（Object Detection）、疾病辅助诊断、中医智能化研究 |
| **典型应用场景** | - 中医智能辅助诊断系统<br> - 舌诊自动化分析<br> - 医疗AI研究与算法验证<br> - 跨模态医学图像理解 |

---
以下文字及代码仅供参考学习使用。
### 📁 数据集目录结构建议

```
tongue_diagnosis_dataset/
├── images/
│   ├── train/            # 5594 张
│   ├── val/              # 572 张
│   └── test/             # 553 张
├── annotations/
│   ├── train_coco.json   # COCO 格式（训练集）
│   ├── val_coco.json     # COCO 格式（验证集）
│   ├── test_coco.json    # COCO 格式（测试集）
│   ├── labels_train/     # YOLO .txt 格式
│   ├── labels_val/
│   ├── labels_test/
│   ├── voc_train.xml     # VOC 格式（可选合并或分文件）
│   ├── voc_val.xml
│   └── voc_test.xml
├── data.yaml             # 配置文件（用于 YOLO 等框架）
└── classes.txt           # 21 个疾病/舌象类别的名称列表
```

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b891fd82086a45f0bfd19fec63c8d1dd.png)

### 🧪 支持的算法实验（可通过配置文件切换）

| **算法框架** | **支持格式** | **配置方式** |
|------------|------------|------------|
| YOLOv5/v8/v10 | `.txt` (YOLO) | 使用 `data.yaml` 指定路径和类别 |
| Faster R-CNN / Mask R-CNN | `.xml` (VOC) 或 COCO | 通过数据加载器读取 |
| DETR / RT-DETR | COCO JSON | 直接加载 `train_coco.json` 等 |
| 自定义模型 | 任意格式 | 提供转换脚本支持互转 |

---

### 📌 数据集特点总结

| 特点 | 说明 |
|------|------|
| **高质量标注** | 每张图像标签均由执业中医师验证，保证医学准确性 |
| **多格式支持** | 一套数据支持主流目标检测框架，无需重复标注 |
| **标准化采集** | 使用统一光照、角度、设备采集，减少干扰因素 |
| **疾病导向** | 21 类疾病标签，可用于构建“舌-病”关联模型 |
| **科研友好** | 明确划分训练/验证/测试集，适合发表论文与算法对比 |

---

推动 **中医智能化、AI辅助诊断、数字中医** 发展的重要资源，适用于高校、医院、AI医疗企业等机构的研究与应用开发。

 **舌象舌诊数据集**（21类疾病、COCO/.txt/.xml 多格式标注）的 **完整 YOLOv8 训练代码**，支持从数据准备、配置文件生成到模型训练、验证与推理的全流程。

文字及代码仅供参考学习使用。
---

### ✅ 一、环境安装

```bash
# 安装 YOLOv8（Ultralytics）
pip install ultralytics opencv-python numpy tqdm
```

> 官方文档：[https://docs.ultralytics.com](https://docs.ultralytics.com)

---

### ✅ 二、数据集结构（推荐格式）

```
tongue_diagnosis_dataset/
├── images/
│   ├── train/          # 5594 张
│   ├── val/            # 572 张
│   └── test/           # 553 张
├── labels/
│   ├── train/          # YOLO 格式 .txt 文件
│   ├── val/
│   └── test/
├── classes.txt         # 21个类别名称，每行一个
└── data.yaml           # YOLO 配置文件
```

> ✅ 若已有 COCO 或 VOC 格式，可自动转换为 YOLO `.txt` 格式（见下文）。

---

### ✅ 三、若原始为 COCO 格式 → 转为 YOLO .txt 格式

```python
# convert_coco_to_yolo.py
import json
import os
from tqdm import tqdm

def coco_to_yolo(coco_json, output_dir, image_dir):
    with open(coco_json, 'r') as f:
        data = json.load(f)

    # 创建类别映射
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    num_classes = max(categories.keys()) + 1
    print(f"共 {num_classes} 个类别")

    # 图像 ID 到文件名映射
    images = {img['id']: img['file_name'] for img in data['images']}

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 按图像组织标注
    annotations = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations:
            annotations[image_id] = []
        annotations[image_id].append(ann)

    # 转换每张图像
    for image_id, anns in tqdm(annotations.items(), desc="Converting COCO to YOLO"):
        img_file = images[image_id]
        img_name = os.path.splitext(os.path.basename(img_file))[0]
        txt_path = os.path.join(output_dir, f"{img_name}.txt")

        yolo_lines = []
        for ann in anns:
            category_id = ann['category_id'] - 1  # COCO 通常从 1 开始，YOLO 从 0
            bbox = ann['bbox']  # [x, y, w, h] (COCO 是左上角 + 宽高)

            # 获取图像尺寸（若无，需从 images 字段读取）
            img_info = next(img for img in data['images'] if img['id'] == image_id)
            img_w, img_h = img_info['width'], img_info['height']

            # 归一化
            x_center = (bbox[0] + bbox[2] / 2) / img_w
            y_center = (bbox[1] + bbox[3] / 2) / img_h
            w = bbox[2] / img_w
            h = bbox[3] / img_h

            yolo_lines.append(f"{category_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        with open(txt_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

# 执行转换（示例）
coco_to_yolo(
    coco_json="tongue_diagnosis_dataset/annotations/train_coco.json",
    output_dir="tongue_diagnosis_dataset/labels/train",
    image_dir="tongue_diagnosis_dataset/images/train"
)

# 同样处理 val_coco.json → labels/val/
```

---

### ✅ 四、生成 `data.yaml` 配置文件

```yaml
# tongue_diagnosis_dataset/data.yaml
train: images/train
val: images/val
test: images/test

# 类别数量
nc: 21

# 类别名称（需与 classes.txt 一致）
names:
  - pale_tongue
  - red_tongue
  - dark_tongue
  - purple_tongue
  - cracked_tongue
  - tooth_marked
  - swollen
  - thin
  - petechiae
  - greasy_coating
  - peeled_coating
  - thick_coating
  - yellow_coating
  - white_coating
  - moist
  - dry
  - tremor
  - deviation
  - short_tongue
  - stiff_tongue
  - other_abnormality
```

> 📌 **注意**：请根据实际类别名称修改 `names` 列表。

---

### ✅ 五、YOLOv8 训练代码

```python
# train_tongue_yolo.py
from ultralytics import YOLO
import torch

# 检查 GPU
device = 0 if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 加载预训练模型（推荐使用 yolov8m 或 yolov8l 提升小目标检测能力）
model = YOLO("yolov8m.pt")  # 可选: yolov8n, yolov8s, yolov8l, yolov8x

# 开始训练
results = model.train(
    data="tongue_diagnosis_dataset/data.yaml",  # 配置文件路径
    epochs=150,                     # 训练轮数
    imgsz=640,                      # 输入图像大小（舌象通常居中，可缩放）
    batch=32,                       # 根据显存调整（建议16-64）
    optimizer='AdamW',              # 收敛更稳定
    lr0=0.001,                      # 初始学习率
    lrf=0.1,                        # 最终学习率 = lr0 * lrf
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    patience=20,                    # 早停：20轮无提升则停止
    project="runs/tongue_diagnosis",
    name="exp_yolov8m",             # 实验名称
    save=True,
    save_period=10,                 # 每10轮保存一次
    cache=False,                    # 若内存充足可设为 True 加速
    device=device,
    workers=4,                      # 数据加载线程数
    single_cls=False,               # 多类别检测
    rect=False,                     # 不使用矩形训练（保持原图比例）
    close_mosaic=10,                # 最后10轮关闭 Mosaic 增强
)

# 模型评估
print("开始验证...")
metrics = model.val()
print(f"mAP@0.5: {metrics.box.map:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map50_95:.4f}")
```

---

### ✅ 六、推理与可视化

```python
# predict.py
from ultralytics import YOLO

# 加载最佳模型
model = YOLO("runs/tongue_diagnosis/exp_yolov8m/weights/best.pt")

# 单图推理
results = model.predict(
    source="test_tongue.jpg",
    imgsz=640,
    conf=0.25,
    iou=0.45,
    save=True,
    project="inference",
    name="tongue_results",
    show_labels=True,
    show_conf=True
)

# 批量推理
results = model.predict(
    source="tongue_diagnosis_dataset/images/test",
    save=True,
    project="inference",
    name="test_batch"
)
```

---

### ✅ 七、高级技巧（提升性能）

#### 1. 使用 TTA（Test Time Augmentation）
```python
metrics = model.val(
    data="tongue_diagnosis_dataset/data.yaml",
    imgsz=640,
    batch=16,
    device=0,
    split='val',
    augment=True  # 启用 TTA
)
print("mAP with TTA:", metrics.box.map)
```

#### 2. 导出为 ONNX / TensorRT
```python
model.export(format="onnx", dynamic=True, opset=12)
model.export(format="engine", half=True)  # TensorRT（需安装 tensorrt）
```

#### 3. 自定义数据增强（在训练时）
```python
results = model.train(
    ...,
    hsv_h=0.015,  # 色调增强
    hsv_s=0.7,    # 饱和度增强（舌苔颜色重要）
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    shear=2.0,
    flipud=0.0,
    fliplr=0.5,
)
```

---

### ✅ 八、预期性能（参考）

| 模型 | mAP@0.5 | 显存占用 | 推理速度（640px） |
|------|---------|----------|-------------------|
| YOLOv8n | ~0.65 | < 4GB | ~8ms |
| YOLOv8s | ~0.70 | ~5GB | ~12ms |
| YOLOv8m | ~0.75 | ~7GB | ~20ms |
| YOLOv8l | ~0.78 | ~10GB | ~35ms |

> 📌 **提示**：舌象目标通常较小，建议：
> - 使用 **高分辨率输入**（如 `imgsz=800` 或 `1024`）
> - 在 `model.train()` 中设置 `mosaic=0` 减少小目标丢失
> - 使用 **FPN++ 或注意力机制** 的变体模型（如 YOLOv8-CA）

---

### ✅ 总结

该 YOLOv8 训练方案适用于：
- 中医智能辅助诊断系统
- 舌诊自动化分析平台
- 医疗 AI 算法研究与论文实验
- 数字中医、健康管理 APP 集成


