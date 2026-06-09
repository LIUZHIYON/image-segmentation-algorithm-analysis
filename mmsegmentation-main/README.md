# MMSegmentation 图像分割实验

> **MMSegmentation** 框架下的语义分割算法实验与对比

---

## 📋 说明

本目录基于开源框架 [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) 进行二次开发，用于毕业设计中的**语义分割**算法实验。

---

## 📁 主要文件

| 文件 | 说明 |
|------|------|
| `main10.py` | ⭐ **QT可视化界面** — 加载模型进行推理与结果展示 |
| `test.py` | 模型测试脚本 |
| `pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py` | PSPNet 配置文件示例 |

## 📁 主要目录

| 目录 | 说明 |
|------|------|
| `configs/` | 各算法配置文件（PSPNet / DeepLabV3+ / SegFormer / UNet 等） |
| `mmseg/` | MMSegmentation 核心代码 |
| `tools/` | 训练、测试、推理工具脚本 |
| `demo/` | 演示示例 |
| `projects/` | 扩展项目 |

---

## 🚀 使用

```bash
# QT 可视化界面
python main10.py

# 测试
python test.py --config configs/xxx.py --checkpoint /path/to/ckpt
```

> 注：`20230816_*` 为教程 notebook，未包含在仓库中。
