from ultralytics import YOLO

# 加载模型
#yolo = YOLO(model='yolo11n.pt',task='detect')
yolo = YOLO(model='runs/segment/train2/weights/best.pt', task='segment')
#result=yolo(source='screen',save=True)
# 设置更低的置信度阈值进行检测
#result = yolo(source='13.jpg', save=True)  # 置信度阈值设置为0.3
result = yolo(source='28.jpg', save=True, conf=0.02)  # 设置置信度阈值为 0.02