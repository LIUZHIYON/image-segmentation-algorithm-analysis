import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from PySide6 import QtWidgets, QtCore, QtGui
import cv2, os, time, numpy as np
from threading import Thread
from ultralytics import YOLO
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# 不然每次YOLO处理都会输出调试信息
os.environ['YOLO_VERBOSE'] = 'False'


class MWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        # 设置界面
        self.setupUI()

        self.videoBtn.clicked.connect(self.openVideoFile)
        self.camBtn.clicked.connect(self.startCamera)
        self.imageBtn.clicked.connect(self.openImageFile)
        self.stopBtn.clicked.connect(self.stop)

        # 定义定时器，用于控制显示视频的帧率
        self.timer_camera = QtCore.QTimer()
        self.timer_camera.timeout.connect(self.show_camera)

        # 加载 YOLO 模型
        self.yolo_model = YOLO('yolo11m-seg.pt')
        # self.yolo_model = YOLO('runs/segment/train4/weights/best.pt')  # 自制数据集的训练权值

        # 加载 Mask R-CNN 模型
        self.mask_rcnn_model = self.load_mask_rcnn_model()

        # 当前使用的模型，默认为 YOLO
        self.current_model = "YOLO"

        # 要处理的视频帧图片队列，目前就放1帧图片
        self.frameToAnalyze = []

        # 启动处理视频帧独立线程
        Thread(target=self.frameAnalyzeThreadFunc, daemon=True).start()

        # 记录输入图片的缩放比例
        self.image_scale = 1.0

    def setupUI(self):
        self.resize(1200, 800)
        self.setWindowTitle('煤矿井下场景 YOLO与Mask_RCNN -Qt 演示')

        # central Widget
        centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(centralWidget)

        # central Widget 里面的 主 layout
        mainLayout = QtWidgets.QVBoxLayout(centralWidget)

        # 界面的上半部分 : 图形展示部分
        topLayout = QtWidgets.QHBoxLayout()
        self.label_ori_video = QtWidgets.QLabel(self)
        self.label_treated = QtWidgets.QLabel(self)
        self.label_ori_video.setMinimumSize(520, 400)
        self.label_treated.setMinimumSize(520, 400)
        self.label_ori_video.setStyleSheet('border:1px solid #D7E2F9;')
        self.label_treated.setStyleSheet('border:1px solid #D7E2F9;')

        topLayout.addWidget(self.label_ori_video)
        topLayout.addWidget(self.label_treated)

        mainLayout.addLayout(topLayout)

        # 界面下半部分： 输出框 和 按钮
        groupBox = QtWidgets.QGroupBox(self)

        bottomLayout = QtWidgets.QHBoxLayout(groupBox)
        self.textLog = QtWidgets.QTextBrowser()
        bottomLayout.addWidget(self.textLog)

        mainLayout.addWidget(groupBox)

        # 添加算法选择下拉菜单
        self.modelComboBox = QtWidgets.QComboBox()
        self.modelComboBox.addItem("YOLO")
        self.modelComboBox.addItem("Mask R-CNN")
        self.modelComboBox.currentTextChanged.connect(self.change_model)

        btnLayout = QtWidgets.QVBoxLayout()
        self.videoBtn = QtWidgets.QPushButton('🎞️视频文件')
        self.camBtn = QtWidgets.QPushButton('📹摄像头')
        self.imageBtn = QtWidgets.QPushButton('🖼️加载图片')
        self.stopBtn = QtWidgets.QPushButton('🛑停止')
        btnLayout.addWidget(self.modelComboBox)
        btnLayout.addWidget(self.videoBtn)
        btnLayout.addWidget(self.camBtn)
        btnLayout.addWidget(self.imageBtn)
        btnLayout.addWidget(self.stopBtn)
        bottomLayout.addLayout(btnLayout)

    def load_mask_rcnn_model(self):
        """加载 Mask R-CNN 模型"""
        self.cfg = get_cfg()  # 将 cfg 保存为类的属性
        self.cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.cfg.MODEL.WEIGHTS = "mask_rcnn_R_50_FPN_3x.pkl"
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 置信度阈值
        self.cfg.MODEL.DEVICE = "cuda"  # 使用 CPU 或 "cuda" 使用 GPU
        return DefaultPredictor(self.cfg)

    def change_model(self, model_name):
        """切换模型"""
        self.current_model = model_name
        self.textLog.append(f"切换到模型: {model_name}")

    def openVideoFile(self):
        options = QtWidgets.QFileDialog.Options()
        videoPath, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择视频文件", "",
                                                             "视频文件 (*.mp4 *.avi *.mov);;所有文件 (*)",
                                                             options=options)
        if videoPath:
            self.cap = cv2.VideoCapture(videoPath)
            if not self.cap.isOpened():
                print("视频文件无法打开")
                return
            if not self.timer_camera.isActive():
                self.timer_camera.start(50)

    def startCamera(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("1号摄像头不能打开")
            return

        if not self.timer_camera.isActive():
            self.timer_camera.start(50)

    def openImageFile(self):
        options = QtWidgets.QFileDialog.Options()
        imagePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择图片文件", "",
                                                             "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)",
                                                             options=options)
        if imagePath:
            # 读取图片
            img = cv2.imread(imagePath)
            if img is None:
                print("无法读取图片文件")
                return

            # 缩放图片以适应UI界面
            max_width = 520  # 与label_ori_video的宽度一致
            max_height = 400  # 与label_ori_video的高度一致
            height, width = img.shape[:2]
            if width > max_width or height > max_height:
                self.image_scale = min(max_width / width, max_height / height)
                img = cv2.resize(img, (int(width * self.image_scale), int(height * self.image_scale)))
            else:
                self.image_scale = 1.0  # 如果图片不需要缩放，比例设为1.0

            # 转换为RGB格式
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 进行目标检测
            if self.current_model == "YOLO":
                results = self.yolo_model(img)[0]
                img_treated = results.plot(line_width=1)
            else:
                outputs = self.mask_rcnn_model(img)
                v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                               scale=1.0)  # 禁用缩放，确保输出尺寸与输入一致
                out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                img_treated = out.get_image()[:, :, ::-1]

            # 将输出图片缩放到与输入图片相同的尺寸
            img_treated = cv2.resize(img_treated, (int(width * self.image_scale), int(height * self.image_scale)))

            # 确保图像数据是 C 连续的
            img = np.ascontiguousarray(img)
            img_treated = np.ascontiguousarray(img_treated)

            # 将原图和处理后的图像显示在界面上
            self.displayImage(img, self.label_ori_video)
            self.displayImage(img_treated, self.label_treated)

            # 将识别结果输出到文本框中
            self.textLog.append(f"{self.current_model} 图片识别结果：")
            if self.current_model == "YOLO":
                for result in results:
                    self.textLog.append(
                        f"检测到: {result.names[result.boxes.cls[0].item()]} 置信度: {result.boxes.conf[0].item():.2f}")
            else:
                for i, score in enumerate(outputs["instances"].scores):
                    class_id = outputs["instances"].pred_classes[i].item()
                    class_name = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes[class_id]  # 使用 self.cfg
                    self.textLog.append(f"检测到: {class_name} 置信度: {score.item():.2f}")

    def displayImage(self, img, label):
        # 确保图像数据是 C 连续的
        img = np.ascontiguousarray(img)

        # 使用 QImage 的三个参数进行图像转换
        qImage = QtGui.QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QtGui.QImage.Format_RGB888)
        label.setPixmap(QtGui.QPixmap.fromImage(qImage))

    def show_camera(self):
        ret, frame = self.cap.read()  # 从视频流中读取
        if not ret:
            return

        # 视频色彩转换回RGB，OpenCV images as BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 缩放视频帧以适应UI界面
        max_width = 520  # 与label_ori_video的宽度一致
        max_height = 400  # 与label_ori_video的高度一致
        height, width = frame.shape[:2]
        if width > max_width or height > max_height:
            self.image_scale = min(max_width / width, max_height / height)
            frame = cv2.resize(frame, (int(width * self.image_scale), int(height * self.image_scale)))
        else:
            self.image_scale = 1.0  # 如果视频帧不需要缩放，比例设为1.0

        # 确保图像数据是 C 连续的
        frame = np.ascontiguousarray(frame)

        # 显示输入视频帧
        self.displayImage(frame, self.label_ori_video)

        # 如果当前没有处理任务
        if not self.frameToAnalyze:
            self.frameToAnalyze.append(frame)

    def frameAnalyzeThreadFunc(self):
        while True:
            if not self.frameToAnalyze:
                time.sleep(0.01)
                continue

            frame = self.frameToAnalyze.pop(0)

            if self.current_model == "YOLO":
                results = self.yolo_model(frame)[0]
                img = results.plot(line_width=1)
            else:
                outputs = self.mask_rcnn_model(frame)
                v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                               scale=1.0)  # 禁用缩放，确保输出尺寸与输入一致
                out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                img = out.get_image()[:, :, ::-1]

            # 将输出图片缩放到与输入图片相同的尺寸
            img = cv2.resize(img, (frame.shape[1], frame.shape[0]))

            # 确保图像数据是 C 连续的
            img = np.ascontiguousarray(img)

            # 显示输出视频帧
            self.displayImage(img, self.label_treated)

            # 将识别结果输出到文本框中
            self.textLog.append(f"{self.current_model} 视频帧识别结果：")
            if self.current_model == "YOLO":
                for result in results:
                    self.textLog.append(
                        f"检测到: {result.names[result.boxes.cls[0].item()]} 置信度: {result.boxes.conf[0].item():.2f}")
            else:
                for i, score in enumerate(outputs["instances"].scores):
                    class_id = outputs["instances"].pred_classes[i].item()
                    class_name = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes[class_id]  # 使用 self.cfg
                    self.textLog.append(f"检测到: {class_name} 置信度: {score.item():.2f}")

            time.sleep(0.5)

    def stop(self):
        self.timer_camera.stop()  # 关闭定时器
        self.cap.release()  # 释放视频流
        self.label_ori_video.clear()  # 清空视频显示区域
        self.label_treated.clear()  # 清空视频显示区域


app = QtWidgets.QApplication()
window = MWindow()
window.show()
app.exec()