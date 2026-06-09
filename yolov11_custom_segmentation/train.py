import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")

model.train(data= "custom_datasets.yaml",imgsz = 640,device = 0,batch = 8,
            epochs = 500,workers = 0)

# # 训练模型
# results = model.train(
#     data="custom_datasets.yaml",  # 数据集配置文件
#     epochs=100,  # 训练轮数
#     imgsz=640,  # 图片大小
#     batch=16,  # 批量大小
#     workers=0  # 禁用多进程加载
# )