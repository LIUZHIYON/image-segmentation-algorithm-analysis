from PySide6 import QtWidgets, QtCore, QtGui
import cv2, os, time
from threading import Thread
from ultralytics import YOLO

# 不然每次YOLO处理都会输出调试信息
os.environ['YOLO_VERBOSE'] = 'False'

class MWindow(QtWidgets.QMainWindow):

    def __init__(self):

        super().__init__()

        # 设置界面
        self.setupUI()

        self.videoBtn.clicked.connect(self.openVideoFile)
        self.camBtn.clicked.connect(self.startCamera)
        self.imageBtn.clicked.connect(self.openImageFile)  # 连接读取图片的按钮
        self.stopBtn.clicked.connect(self.stop)

        # 定义定时器，用于控制显示视频的帧率
        self.timer_camera = QtCore.QTimer()
        self.timer_camera.timeout.connect(self.show_camera)

        # 加载 YOLO nano 模型，第一次比较耗时，要20秒左右
        self.model = YOLO('mask_rcnn_R_50_FPN_3x.pkl')

        # 要处理的视频帧图片队列，目前就放1帧图片
        self.frameToAnalyze = []

        # 启动处理视频帧独立线程
        Thread(target=self.frameAnalyzeThreadFunc,daemon=True).start()

    def setupUI(self):

        self.resize(1200, 800)

        self.setWindowTitle('煤矿井下场景 YOLO-Qt 演示')

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

        btnLayout = QtWidgets.QVBoxLayout()
        self.videoBtn = QtWidgets.QPushButton('🎞️视频文件')
        self.camBtn = QtWidgets.QPushButton('📹摄像头')
        self.imageBtn = QtWidgets.QPushButton('🖼️加载图片')  # 新增按钮：加载图片
        self.stopBtn = QtWidgets.QPushButton('🛑停止')
        btnLayout.addWidget(self.videoBtn)
        btnLayout.addWidget(self.camBtn)
        btnLayout.addWidget(self.imageBtn)  # 添加到按钮布局中
        btnLayout.addWidget(self.stopBtn)
        bottomLayout.addLayout(btnLayout)

    def openVideoFile(self):
        options = QtWidgets.QFileDialog.Options()
        videoPath, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov);;所有文件 (*)", options=options)
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

    def openImageFile(self):  # 新增读取图片的功能
        options = QtWidgets.QFileDialog.Options()
        imagePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择图片文件", "", "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)", options=options)
        if imagePath:
            # 读取图片
            img = cv2.imread(imagePath)
            if img is None:
                print("无法读取图片文件")
                return

            # 转换为RGB格式
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 调整大小为界面显示大小
            img_resized = cv2.resize(img, (520, 400))

            # 进行YOLO目标检测
            results = self.model(img)[0]
            img_treated = results.plot(line_width=1)

            # 将原图和处理后的图像显示在界面上
            self.displayImage(img_resized, self.label_ori_video)
            self.displayImage(img_treated, self.label_treated)

            # 将识别结果输出到文本框中
            self.textLog.append("图片识别结果：")
            for result in results:
                self.textLog.append(f"检测到: {result.names[result.boxes.cls[0].item()]} 置信度: {result.boxes.conf[0].item():.2f}")

    def displayImage(self, img, label):
        # 使用 QImage 的三个参数进行图像转换
        qImage = QtGui.QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QtGui.QImage.Format_RGB888)
        label.setPixmap(QtGui.QPixmap.fromImage(qImage))

    def show_camera(self):
        ret, frame = self.cap.read()  # 从视频流中读取
        if not ret:
            return

        # 把读到的16:10帧的大小重新设置
        frame = cv2.resize(frame, (520, 400))
        # 视频色彩转换回RGB，OpenCV images as BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
            results = self.model(frame)[0]

            img = results.plot(line_width=1)

            self.displayImage(img, self.label_treated)

            # 将识别结果输出到文本框中
            self.textLog.append("视频帧识别结果：")
            for result in results:
                self.textLog.append(f"检测到: {result.names[result.boxes.cls[0].item()]} 置信度: {result.boxes.conf[0].item():.2f}")

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