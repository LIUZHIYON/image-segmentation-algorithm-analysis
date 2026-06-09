# YOLOv11 煤矿场景分割

> **Ultralytics YOLOv11** 实现的实例分割，针对煤矿井下场景的自定义数据集训练

---

## 📋 说明

本目录使用 **Ultralytics YOLOv11** 框架，对煤矿井下场景进行**实例分割**，目标类别包括人、掘进机、采煤机、巡检机器人。

---

## 📁 主要文件

| 文件 | 说明 |
|------|------|
| `train.py` | YOLOv11-seg 模型训练 |
| `train2.py` | 优化参数训练（数据增强、超参调整） |
| `predict.py` | 单张图片 / 视频推理 |
| `predict2.py` | 批量推理与结果导出 |
| `sam_predict.py` | SAM（Segment Anything Model）推理对比 |
| `Auto-Annotation.py` | 自动标注工具（利用预训练模型生成标注） |
| `person.yaml` | 数据集配置文件（路径、类别） |
| `yolo_bishe_data.yaml` | 数据集配置（训练/验证划分） |

## 📁 主要目录

| 目录 | 说明 |
|------|------|
| `ultralytics/` | Ultralytics YOLOv11 核心代码 |
| `examples/` | 示例代码（ONNX、OpenVINO、TFLite 部署等） |
| `img/` | 测试图片 |
| `docs/` | 文档 |
| `tests/` | 单元测试 |

---

## 🚀 使用

```bash
# 训练
python train.py

# 推理
python predict.py --source path/to/image.jpg

# SAM 推理
python sam_predict.py --image path/to/image.jpg

# 自动标注
python Auto-Annotation.py --images path/to/images/
```

> 训练数据集（图片与标签）及模型权重文件（`.pt`）未包含在仓库中。
