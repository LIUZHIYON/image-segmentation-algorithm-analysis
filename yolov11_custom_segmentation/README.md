# YOLOv11 + Detectron2 实例分割

> **Detectron2** 框架 + **YOLOv11** 对比，含自定义数据集训练与 QT 可视化

---

## 📋 说明

本目录使用 **Detectron2 (Mask R-CNN)** 和 **DeepLabV3+** 进行实例/语义分割实验，并提供 QT 可视化界面用于模型推理展示。

---

## 📁 主要文件

| 文件 | 说明 |
|------|------|
| `main12.py` | **QT可视化界面** — 加载模型进行推理与结果展示 |
| `deeplab.py` | DeepLabV3+ 语义分割实现 |
| `custom_datasets.yaml` | 自定义数据集注册配置 |
| `classes.txt` | 目标类别定义（person / roadheader / shearer / robot） |
| `predict.py` | 单张图片推理 |
| `train.py` | Mask R-CNN 训练脚本 |
| `mkdocs.yml` | 文档配置 |

## 📁 主要目录

| 目录 | 说明 |
|------|------|
| `detectron2/` | Detectron2 框架核心代码 |
| `configs/` | Detectron2 模型配置文件（Mask R-CNN、RetinaNet 等） |
| `examples/` | 示例代码（ONNX Runtime、OpenVINO 部署等） |
| `ultralytics/` | Ultralytics YOLOv11 核心代码 |
| `tests/` | 单元测试 |

---

## 🚀 使用

```bash
# QT 界面推理
python main12.py

# 训练
python train.py

# 预测
python predict.py --input path/to/image.jpg
```

> 模型权重文件（`.pth` / `.pkl` / `.pt`）未包含在仓库中。
