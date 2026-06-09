import os
import cv2
import numpy as np
import time
from threading import Thread
from PySide6 import QtWidgets, QtCore, QtGui
from mmseg.apis import inference_model, init_model
from mmseg.utils import get_palette

# 设置环境变量
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['YOLO_VERBOSE'] = 'False'

class MWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUI()

        # 连接按钮事件
        self.videoBtn.clicked.connect(self.openVideoFile)
        self.camBtn.clicked.connect(self.startCamera)
        self.imageBtn.clicked.connect(self.openImageFile)
        self.stopBtn.clicked.connect(self.stop)

        # 定时器用于显示视频帧
        self.timer_camera = QtCore.QTimer()
        self.timer_camera.timeout.connect(self.show_camera)

        # 加载 MMSegmentation 模型
        self.mmseg_models = {
            "deeplabv3+": self.load_mmseg_model("deeplabv3+"),
            "pspnet": self.load_mmseg_model("pspnet")
        }
        self.current_model = "deeplabv3+"

        # 视频帧队列
        self.frameToAnalyze = []

        # 启动处理帧的线程
        Thread(target=self.frameAnalyzeThreadFunc, daemon=True).start()

    def setupUI(self):
        self.resize(1200, 800)
        self.setWindowTitle('MMSegmentation 演示')

        # 主布局
        centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(centralWidget)
        mainLayout = QtWidgets.QVBoxLayout(centralWidget)

        # 图像显示区域
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

        # 日志输出和按钮区域
        groupBox = QtWidgets.QGroupBox(self)
        bottomLayout = QtWidgets.QHBoxLayout(groupBox)
        self.textLog = QtWidgets.QTextBrowser()
        bottomLayout.addWidget(self.textLog)

        # 按钮布局
        btnLayout = QtWidgets.QVBoxLayout()
        self.modelComboBox = QtWidgets.QComboBox()
        self.modelComboBox.addItem("deeplabv3+")
        self.modelComboBox.addItem("pspnet")
        self.modelComboBox.currentTextChanged.connect(self.change_model)
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
        mainLayout.addWidget(groupBox)

    def load_mmseg_model(self, model_name):
        """加载 MMSegmentation 模型"""
        if model_name == "deeplabv3+":
            config_file = './configs/deeplabv3plus/deeplabv3plus_r18-d8_4xb2-80k_cityscapes-512x1024.py'
            checkpoint_file = 'checkpoint/deeplabv3plus_r18-d8_512x1024_80k_cityscapes_20201226_080942-cff257fe.pth'
        elif model_name == "pspnet":
            config_file = 'configs/pspnet/pspnet_r18-d8_4xb2-80k_cityscapes-512x1024.py'
            checkpoint_file = 'checkpoint/pspnet_r18-d8_512x1024_80k_cityscapes_20201225_021458-09ffa746.pth'
        else:
            raise ValueError(f"未知的模型名称: {model_name}")

        model = init_model(config_file, checkpoint_file, device='cpu')  # 使用 CPU 或 "cuda" 使用 GPU

        # 手动指定类别信息和颜色
        model.CLASSES = [
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
            'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
            'truck', 'bus', 'train', 'motorcycle', 'bicycle'
        ]
        model.PALETTE = [
            [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
            [0, 0, 230], [119, 11, 32]
        ]
        return model

    def change_model(self, model_name):
        """切换模型"""
        self.current_model = model_name
        self.textLog.append(f"切换到模型: {model_name}")

    def openVideoFile(self):
        """打开视频文件"""
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
        """启动摄像头"""
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("1号摄像头不能打开")
            return

        if not self.timer_camera.isActive():
            self.timer_camera.start(50)

    def openImageFile(self):
        """打开图片文件"""
        options = QtWidgets.QFileDialog.Options()
        imagePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择图片文件", "",
                                                             "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)",
                                                             options=options)
        if imagePath:
            img = cv2.imread(imagePath)
            if img is None:
                print("无法读取图片文件")
                return

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, (520, 400))

            # 进行图像分割
            result = inference_model(self.mmseg_models[self.current_model], img)
            img_treated = self.visualize_result(img, result)

            # 显示图像
            self.displayImage(img_resized, self.label_ori_video)
            self.displayImage(img_treated, self.label_treated)

            # 输出日志
            self.textLog.append(f"{self.current_model} 图片识别结果：")
            self.show_label_info(result)

    def visualize_result(self, img, result):
        """可视化分割结果"""
        model = self.mmseg_models[self.current_model]
        seg = result.pred_sem_seg.data[0].cpu().numpy()
        palette = model.PALETTE  # 获取颜色
        classes = model.CLASSES  # 获取类别名称

        # 创建彩色分割图
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        color_seg = color_seg[..., ::-1]  # BGR to RGB

        # 混合原图和分割结果
        img = img * 0.5 + color_seg * 0.5
        img = img.astype(np.uint8)

        # 在图像上绘制类别标签
        unique_labels = np.unique(seg)
        for label in unique_labels:
            if label >= len(classes):  # 确保标签在类别范围内
                continue
            mask = (seg == label).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 100:  # 过滤小区域
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                cv2.putText(img, classes[label], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return img

    def show_label_info(self, result):
        """显示识别到的类别信息"""
        seg = result.pred_sem_seg.data[0].cpu().numpy()
        unique_labels = np.unique(seg)
        model = self.mmseg_models[self.current_model]
        classes = model.CLASSES  # 获取类别名称

        for label in unique_labels:
            if label >= len(classes):  # 确保标签在类别范围内
                continue
            label_name = classes[label]
            pixel_count = np.sum(seg == label)
            self.textLog.append(f"类别: {label_name}, 像素数量: {pixel_count}")

    def displayImage(self, img, label):
        """显示图像"""
        img = np.ascontiguousarray(img)
        qImage = QtGui.QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QtGui.QImage.Format_RGB888)
        label.setPixmap(QtGui.QPixmap.fromImage(qImage))

    def show_camera(self):
        """显示摄像头帧"""
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.resize(frame, (520, 400))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.ascontiguousarray(frame)

        self.displayImage(frame, self.label_ori_video)

        if not self.frameToAnalyze:
            self.frameToAnalyze.append(frame)

    def frameAnalyzeThreadFunc(self):
        """处理帧的线程函数"""
        while True:
            if not self.frameToAnalyze:
                time.sleep(0.01)
                continue

            frame = self.frameToAnalyze.pop(0)

            # 进行图像分割
            result = inference_model(self.mmseg_models[self.current_model], frame)
            img = self.visualize_result(frame, result)

            # 显示处理后的图像
            self.displayImage(img, self.label_treated)

            # 输出日志
            self.textLog.append(f"{self.current_model} 视频帧识别结果：")
            self.show_label_info(result)

            time.sleep(0.5)

    def stop(self):
        """停止"""
        self.timer_camera.stop()
        self.cap.release()
        self.label_ori_video.clear()
        self.label_treated.clear()

# 启动应用程序
app = QtWidgets.QApplication([])
window = MWindow()
window.show()
app.exec()