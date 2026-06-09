import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from ultralytics import YOLO

# Load a model
#model = YOLO("yolo11n.yaml")  # build a new model from YAML
model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)
#model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# 训练模型
results = model.train(
    data="ultralytics/cfg/datasets/coco8-seg.yaml",  # 数据集配置文件
    epochs=100,  # 训练轮数
    imgsz=640,  # 图片大小
    batch=16,  # 批量大小
    workers=0  # 禁用多进程加载
)