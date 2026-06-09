import os
import cv2
import numpy as np
import time
from threading import Thread
from PySide6 import QtWidgets, QtCore, QtGui
from mmseg.apis import inference_model, init_model

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
            config_file = 'Zihao-Configs/ZihaoDataset_DeepLabV3plus_20230818.py'
            checkpoint_file = 'work_dirs/ZihaoDataset-DeepLabV3plus/best_mIoU_iter_3400.pth'
        elif model_name == "pspnet":
            config_file = 'Zihao-Configs/ZihaoDataset_PSPNet_20230818.py'
            checkpoint_file = 'work_dirs/ZihaoDataset-PSPNet/best_mIoU_iter_3800.pth'
        else:
            raise ValueError(f"未知的模型名称: {model_name}")

        model = init_model(config_file, checkpoint_file, device='cuda')  # 使用 CPU 或 "cuda" 使用 GPU

        # 手动指定类别信息和颜色
        model.CLASSES = ['background', 'person', 'roadheader', 'robot', 'shearer']
        model.PALETTE = [
            [127, 127, 127],  # 背景色：中等亮度的灰色
            [255, 0, 0],  # person：红色，更鲜艳
            [0, 200, 0],  # roadheader：浅绿色
            [0, 0, 255],  # robot：蓝色，
            [255, 215, 0]  # shearer：金黄色，保持明亮但不刺眼
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

            # 使用原图进行推理
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = inference_model(self.mmseg_models[self.current_model], img_rgb)
            img_treated = self.visualize_result(img_rgb, result)

            # 缩放图像用于显示
            img_resized = cv2.resize(img_rgb, (520, 400))
            img_treated_resized = cv2.resize(img_treated, (520, 400))

            # 显示图像
            self.displayImage(img_resized, self.label_ori_video)
            self.displayImage(img_treated_resized, self.label_treated)

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
        # color_seg = color_seg[..., ::-1]  # 不需要 BGR 转换

        # 混合原图和分割结果
        img = img * 0.5 + color_seg * 0.5
        img = img.astype(np.uint8)

        # 根据图像高度动态调整字体大小和粗细
        font_scale = img.shape[0] / 1000  # 根据图像高度调整字体大小
        font_thickness = max(1, int(font_scale * 3))  # 增加字体粗细

        # 在图像上绘制类别标签（每个类别只显示一个标签）
        unique_labels = np.unique(seg)
        for label in unique_labels:
            if label >= len(classes):  # 确保标签在类别范围内
                continue
            mask = (seg == label).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # 找到最大的区域
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) < 100:  # 过滤小区域
                    continue

                # 计算区域的中心点
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0

                # 在中心点绘制类别标签
                text = classes[label]
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                text_x = cX - text_width // 2
                text_y = cY + text_height // 2
                cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        return img

    def rgb_to_color_name(self, rgb):
        """将 RGB 颜色值映射到英文颜色名称"""
        color_map = {
            (127, 127, 127): "Gray",
            (255, 0, 0): "Red",
            (0, 200, 0): "Green",
            (0, 0, 255): "Blue",
            (255, 215, 0): "Gold"
        }
        return color_map.get(tuple(rgb), "Unknown")

    def show_label_info(self, result):
        """显示识别到的类别信息"""
        seg = result.pred_sem_seg.data[0].cpu().numpy()
        unique_labels = np.unique(seg)
        model = self.mmseg_models[self.current_model]
        classes = model.CLASSES  # 获取类别名称
        palette = model.PALETTE  # 获取颜色

        for label in unique_labels:
            if label >= len(classes):  # 确保标签在类别范围内
                continue
            label_name = classes[label]
            color_rgb = palette[label]
            color_name = self.rgb_to_color_name(color_rgb)
            self.textLog.append(f"类别: {label_name}, 颜色: {color_name}")

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

        # 使用原图进行推理
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not self.frameToAnalyze:
            self.frameToAnalyze.append(frame_rgb)

        # 缩放图像用于显示
        frame_resized = cv2.resize(frame_rgb, (520, 400))
        self.displayImage(frame_resized, self.label_ori_video)

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

            # 缩放图像用于显示
            img_resized = cv2.resize(img, (520, 400))

            # 显示处理后的图像
            self.displayImage(img_resized, self.label_treated)

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