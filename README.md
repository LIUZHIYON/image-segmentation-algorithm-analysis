# 🎯 图像分割算法分析与应用

> **本科毕业设计** — 基于深度学习的主流图像分割算法对比分析与应用实现  
> 以**煤矿井下场景**为背景，对**人、掘进机、采煤机、巡检机器人**等多类目标进行分割检测

---

## 🧭 项目简介

图像分割是计算机视觉的核心任务之一。本毕业设计系统性地研究了当前主流的图像分割算法，涵盖 **语义分割** 与 **实例分割** 两大方向，在同一个自定义数据集上进行训练、评估和对比分析。

### 涉及的算法框架

| 框架 / 算法 | 类型 | 项目模块 |
|------------|------|---------|
| **YOLOv11-seg** (Ultralytics) | ⚡ 实时实例分割 | `yolo_bishe_seg/` |
| **Mask R-CNN** (Detectron2) | 🎯 高精度实例分割 | `yolov11_custom_segmentation/` |
| **DeepLabV3+** | 🧩 语义分割 | `yolov11_custom_segmentation/deeplab.py` |
| **PSPNet** | 🧩 语义分割 | `mmsegmentation-main/` |
| **SegFormer** | 🧩 语义分割 (Transformer) | `mmsegmentation-main/` |
| **SAM** (Meta) | 🪄 通用分割大模型 | `yolo_bishe_seg/sam_predict.py` |

### 目标类别

| 类别 | 说明 |
|------|------|
| `person` | 井下工作人员 |
| `roadheader` | 掘进机 |
| `shearer` | 采煤机 |
| `robot` | 巡检机器人 |

---

## 📁 项目结构

```
.
├── mmsegmentation-main/               # 📦 MMSegmentation — 语义分割实验
│   ├── main10.py                      #   QT 可视化界面（推理 + 结果展示）
│   ├── configs/                       #   模型配置文件（PSPNet / SegFormer 等）
│   ├── mmseg/                         #   MMSegmentation 核心代码
│   ├── tools/                         #   训练 / 测试 / 推理脚本
│   ├── demo/                          #   演示示例
│   └── outputs/                       #   训练输出
│
├── yolov11_custom_segmentation/       # 📦 Detectron2 — 实例分割实验
│   ├── main12.py                      #   QT 可视化界面（模型推理与可视化）
│   ├── detectron2/                    #   Detectron2 核心代码
│   ├── deeplab.py                     #   DeepLabV3+ 实现
│   ├── configs/                       #   Detectron2 配置文件
│   ├── custom_datasets.yaml           #   自定义数据集注册
│   ├── classes.txt                    #   类别定义文件
│   ├── predict.py                     #   预测脚本
│   └── train.py                       #   训练脚本
│
├── yolo_bishe_seg/                    # 📦 Ultralytics YOLOv11 — 实时分割
│   ├── train.py / train2.py           #   训练脚本
│   ├── predict.py / predict2.py       #   推理脚本
│   ├── sam_predict.py                 #   SAM 大模型推理
│   ├── Auto-Annotation.py             #   自动标注工具
│   ├── person.yaml                    #   数据集配置
│   ├── ultralytics/                   #   Ultralytics 核心代码
│   └── examples/                      #   部署示例（ONNX / OpenVINO / TFLite）
│
├── web_ui.py                          # 🌐 Web 统一界面（Gradio）
├── outputs/                          # 📂 分割结果保存目录
├── README.md                         # 📄 本文件
└── .gitignore
```

---

## 🚀 快速开始

### 环境要求

- **Python** 3.8 – 3.11
- **PyTorch** ≥ 1.10（推荐 2.0+）
- **CUDA** 11.3+（GPU 训练/推理，推荐）
- **操作系统** Windows / Linux

### 1️⃣ 克隆仓库

```bash
git clone https://github.com/LIUZHIYON/image-segmentation-algorithm-analysis.git
cd image-segmentation-algorithm-analysis
```

### 2️⃣ 安装依赖

根据你想使用的模块选择安装：

```bash
# ⚡ YOLOv11（推荐先试这个，最简单）
pip install ultralytics
# 或进入子目录
cd yolo_bishe_seg
pip install -r requirements.txt   # 如有
```

```bash
# 🎯 Detectron2（需要编译）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python pyqt5
cd yolov11_custom_segmentation
pip install -e detectron2/
```

```bash
# 🧩 MMSegmentation
pip install openmim
mim install mmengine
cd mmsegmentation-main
pip install -v -e .
```

### 3️⃣ 准备数据

- 准备自定义数据集（图片 + COCO/YOLO 格式标注）
- 根据使用的模块修改对应的配置文件：
  - `yolov11_custom_segmentation/custom_datasets.yaml`
  - `yolo_bishe_seg/person.yaml`
  - `mmsegmentation-main/configs/` 下对应的配置文件

### 4️⃣ 运行推理

```bash
# 🖼️ YOLOv11 单张图片推理
cd yolo_bishe_seg
python predict.py --source demo.jpg

# 🖥️ QT 可视化界面（Detectron2）
cd yolov11_custom_segmentation
python main12.py

# 🖥️ QT 可视化界面（MMSegmentation）
cd mmsegmentation-main
python main10.py

# 🪄 SAM 通用分割
cd yolo_bishe_seg
python sam_predict.py --image demo.jpg
```

### 5️⃣ 训练模型

```bash
# YOLOv11-seg 训练
cd yolo_bishe_seg
python train.py

# Mask R-CNN 训练
cd yolov11_custom_segmentation
python train.py
```

---

## 📊 实验结论

在煤矿井下自定义数据集上的对比结果：

| 算法 | 推理速度 | 精度 | 特点 |
|------|---------|------|------|
| **YOLOv11-seg** | ⚡⚡⚡⚡⚡ | 较高 | 实时性最佳，适合部署 |
| **Mask R-CNN** | ⚡⚡ | ⭐⭐⭐⭐⭐ | 精度最高，掩码更精细 |
| **DeepLabV3+** | ⚡⚡⚡ | ⭐⭐⭐⭐ | 语义分割边缘平滑 |
| **PSPNet** | ⚡⚡ | ⭐⭐⭐⭐ | 金字塔池化，场景理解好 |
| **SegFormer** | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | Transformer，光照鲁棒 |

> 详细指标和各模型对比请参考各子目录的 README。

---

## 🌐 Web 统一界面（推荐）

运行下面命令即可打开浏览器使用统一的 Web 界面：

```bash
python web_ui.py
```

访问 `http://localhost:7860`，支持：
- ✅ **模型选择** — 在 4 个模型间自由切换
- ✅ **图片分割** — 上传图片，一键分割
- ✅ **视频分割** — 上传视频，逐帧处理并导出
- ✅ **摄像头实时分割** — 打开摄像头实时预览
- ✅ **结果保存** — 分割图片可保存到 `outputs/` 目录
- ✅ **日志面板** — 记录每次处理的详细信息

## 🛠️ 核心功能亮点

- ✅ **多框架统一入口** — 一个项目中体验 YOLOv11 / Detectron2 / MMSegmentation
- ✅ **QT 可视化界面** — `main10.py` / `main12.py`，加载模型实时推理
- ✅ **Web 统一界面** — `web_ui.py`，Gradio 构建，浏览器即可使用
- ✅ **自动标注工具** — 利用预训练模型 + SAM 生成训练标注，减少人工成本
- ✅ **部署示例** — ONNX Runtime / OpenVINO / TFLite 等多平台推理
- ✅ **对比分析** — 同一数据集、统一指标，公平对比各算法

---

## 📝 注意事项

- 数据集（煤矿标注图片）和模型权重文件（`.pt` / `.pth` / `.pkl`）**未包含**在本仓库中，请自行准备
- `mmsegmentation-main/20230816_*` 为 MMSegmentation 教程 notebooks，已从仓库排除
- 每个子目录下有独立的 `README.md`，包含更详细的说明

---

## 📄 许可证

本项目仅用于**学术研究与毕业设计展示**，不作商业用途。

---

**👨‍🎓 作者：** 小夏  
**📅 时间：** 2025  
**🔗 GitHub：** [LIUZHIYON/image-segmentation-algorithm-analysis](https://github.com/LIUZHIYON/image-segmentation-algorithm-analysis)
