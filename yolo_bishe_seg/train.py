import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from ultralytics import YOLO

# 加载模型
model = YOLO("yolo11m-seg.pt")

# 训练模型并增加数据增强功能
model.train(
    data="yolo_bishe_data.yaml",
    imgsz=640,
    device=0,
    batch=8,
    epochs=200,
    workers=0,
    # 数据增强参数
    augment=True,  # 启用数据增强
    hsv_h=0.015,   # 色调增强
    hsv_s=0.7,     # 饱和度增强
    hsv_v=0.4,     # 亮度增强
    degrees=10,    # 随机旋转角度范围
    translate=0.1, # 随机平移范围
    scale=0.5,     # 随机缩放范围
    shear=0.0,     # 随机剪切范围
    flipud=0.0,    # 上下翻转概率
    fliplr=0.5,    # 左右翻转概率
    mosaic=1.0,    # Mosaic数据增强概率
    mixup=0.0,     # Mixup数据增强概率
)