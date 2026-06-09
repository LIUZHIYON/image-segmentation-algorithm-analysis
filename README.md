# 图像分割算法分析与应用

> **毕业设计** — 基于深度学习的主流图像分割算法分析、对比与应用

---

## 📋 项目概述

本项目系统性地研究了当前主流的图像分割算法，涵盖 **YOLOv11 (Ultralytics)**、**Detectron2 (Mask R-CNN)** 和 **MMSegmentation** 三大框架，以**煤矿井下场景**为应用背景，实现了对**人、掘进机、采煤机、机器人**等多类目标的分割检测。

---

## 📁 项目结构

```
├── mmsegmentation-main/          # MMSegmentation 框架实验
│   ├── configs/                  #  模型配置文件
│   ├── mmseg/                    #  MMSegmentation 核心代码
│   ├── tools/                    #  训练/测试/推理脚本
│   ├── main*.py                  #  自定义训练与评估脚本
│   ├── demo/                     #  演示demo
│   └── ...
│
├── yolov11_custom_segmentation/  # YOLOv11 + Detectron2 分割
│   ├── detectron2/               #  Detectron2 核心代码
│   ├── configs/                  #  Detectron2 配置文件
│   ├── main12.py                 #  QT界面 — 模型推理与可视化
│   ├── deeplab.py                #  DeepLabV3+ 实现
│   ├── custom_datasets.yaml      #  自定义数据集配置
│   ├── classes.txt               #  类别定义
│   ├── predict.py                #  预测脚本
│   └── ...
│
├── yolo_bishe_seg/               # YOLOv11 煤矿分割项目
│   ├── ultralytics/              #  Ultralytics YOLOv11 核心代码
│   ├── train.py / train2.py      #  训练脚本
│   ├── predict.py / predict2.py  #  预测脚本
│   ├── sam_predict.py            #  SAM 模型推理
│   ├── Auto-Annotation.py        #  自动标注工具
│   ├── person.yaml               #  数据集配置文件
│   └── ...
│
└── .gitignore
```

---

## 🧠 核心功能

### 1. 多框架支持
| 框架 | 用途 | 对应模块 |
|------|------|----------|
| **YOLOv11 (Ultralytics)** | 实时实例分割 | `yolo_bishe_seg/` |
| **Detectron2 (Mask R-CNN)** | 实例分割 | `yolov11_custom_segmentation/` |
| **MMSegmentation** | 语义分割 | `mmsegmentation-main/` |
| **DeepLabV3+** | 语义分割 | `yolov11_custom_segmentation/deeplab.py` |
| **SAM** | 分割大模型推理 | `yolo_bishe_seg/sam_predict.py` |

### 2. QT 可视化界面
- `yolov11_custom_segmentation/main12.py` — 模型推理与结果可视化界面

### 3. 自动标注
- `yolo_bishe_seg/Auto-Annotation.py` — 利用预训练模型自动生成标注

### 4. 对比分析
- 同一数据集上对比不同算法的 mAP、推理速度、显存占用等指标

---

## 🏭 应用场景

本项目以**煤矿井下环境**为应用背景，目标类别包括：

- **person** — 井下人员
- **roadheader** — 掘进机
- **shearer** — 采煤机
- **robot** — 巡检机器人

> 数据集为自定义标注，不包含在本仓库中。

---

## 🚀 快速开始

### 环境要求
- Python 3.8+
- PyTorch 1.10+
- CUDA 11.3+（推荐）

### 安装依赖

```bash
# Detectron2
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# MMSegmentation
cd mmsegmentation-main
pip install -v -e .

# Ultralytics (YOLOv11)
pip install ultralytics
```

### 运行推理

```bash
# YOLOv11 推理
cd yolo_bishe_seg
python predict.py --source path/to/image.jpg

# QT 界面推理
cd yolov11_custom_segmentation
python main12.py
```

---

## 📊 实验结论

通过对比 YOLOv11、Mask R-CNN、DeepLabV3+、PSPNet、SegFormer 等算法在煤矿数据集上的表现：

- **YOLOv11-seg** 在推理速度上优势明显，适合实时场景
- **Mask R-CNN (Detectron2)** 在精度上表现更优
- **DeepLabV3+** 在语义分割任务上边缘更精细
- **SegFormer** 在复杂光照条件下鲁棒性更好

---

## 📝 说明

- 数据集（煤矿标注图片）和模型权重（`.pt` / `.pth` / `.pkl`）未包含在本仓库中
- `mmsegmentation-main/` 中的 `20230816_*` 为教程 notebooks，已从仓库中排除
- 各子目录均有独立的 `README.md` 可参考

---

## 📄 许可证

本项目仅用于学术研究与毕业设计展示。
