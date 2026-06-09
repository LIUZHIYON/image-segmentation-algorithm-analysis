import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import torch
import numpy as np  # 添加这行
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode  # 添加ColorMode
from detectron2.data import MetadataCatalog

# 1. 自定义数据集配置（必须与训练时一致）
CUSTOM_CLASSES = ["person", "roadheader", "robot", "shearer"]  # 替换为你的实际类别
DATASET_NAME = "my_dataset"  # 与训练时注册的名称一致

# 2. 注册自定义数据集元数据
MetadataCatalog.get(DATASET_NAME).set(thing_classes=CUSTOM_CLASSES)
metadata = MetadataCatalog.get(DATASET_NAME)

# 3. 模型配置（需与训练配置匹配）
cfg = get_cfg()
cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # 使用你的配置文件路径
cfg.MODEL.WEIGHTS = "output/instance_segmentation/model_final.pth"  # 训练好的模型权重路径
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CUSTOM_CLASSES)  # 类别数量
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 检测阈值
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 4. 创建预测器
predictor = DefaultPredictor(cfg)

# 5. 单张图片预测函数
def predict_image(image_path, output_path="output.jpg"):
    # 读取图片（支持中文路径）
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

    # 执行预测
    outputs = predictor(image)

    # 可视化结果（带自定义颜色）
    v = Visualizer(image[:, :, ::-1],
                  metadata=metadata,
                  scale=1.0,
                  instance_mode=ColorMode.IMAGE_BW)  # 原始图像背景

    # 绘制预测结果
    instances = outputs["instances"].to("cpu")
    out = v.draw_instance_predictions(instances)

    # 保存结果（支持中文路径）
    cv2.imencode('.jpg', out.get_image()[:, :, ::-1])[1].tofile(output_path)
    print(f"预测结果已保存至: {output_path}")

# 6. 使用示例
predict_image("2.jpg", "预测结果.jpg")