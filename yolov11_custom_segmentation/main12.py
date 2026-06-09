import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['YOLO_VERBOSE'] = 'False'

from PySide6 import QtWidgets, QtCore, QtGui
import cv2, os, time, numpy as np
from threading import Thread
from ultralytics import YOLO
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog


class VideoProcessingThread(QtCore.QThread):
    """后台视频处理线程类"""
    progress_updated = QtCore.Signal(int)  # 进度更新信号
    processing_finished = QtCore.Signal(str)  # 处理完成信号

    def __init__(self, video_path, output_path, model_type, yolo_model=None, mask_rcnn_model=None, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.output_path = output_path
        self.model_type = model_type
        self.yolo_model = yolo_model
        self.mask_rcnn_model = mask_rcnn_model
        self._is_running = True  # 线程运行标志

    def process_frame(self, frame):
        """处理单帧图像"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.model_type == "YOLO":
            results = self.yolo_model(frame_rgb)[0]
            segmented_frame = results.plot(line_width=1)
        else:
            outputs = self.mask_rcnn_model(frame_rgb)
            v = Visualizer(frame_rgb[:, :, ::-1],
                          metadata=MetadataCatalog.get("custom_dataset"),
                          scale=0.8,
                          instance_mode=ColorMode.SEGMENTATION)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            segmented_frame = v.get_image()[:, :, ::-1]

        return cv2.cvtColor(segmented_frame, cv2.COLOR_RGB2BGR)

    def run(self):
        """线程主运行方法"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.processing_finished.emit("无法打开视频文件")
                return

            # 获取视频信息
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

            frame_count = 0
            while self._is_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 处理当前帧并写入输出视频
                processed_frame = self.process_frame(frame)
                out.write(processed_frame)

                frame_count += 1
                progress = int((frame_count / total_frames) * 100)
                self.progress_updated.emit(progress)

            # 释放资源
            cap.release()
            out.release()
            self.processing_finished.emit(f"视频处理完成，已保存到: {self.output_path}")

        except Exception as e:
            self.processing_finished.emit(f"处理出错: {str(e)}")

    def stop(self):
        """停止处理"""
        self._is_running = False


class MWindow(QtWidgets.QMainWindow):
    """主窗口类"""
    def __init__(self):
        super().__init__()
        self.setupUI()

        # 连接按钮信号
        self.videoBtn.clicked.connect(self.openVideoFile)
        self.camBtn.clicked.connect(self.startCamera)
        self.imageBtn.clicked.connect(self.openImageFile)
        self.stopBtn.clicked.connect(self.stop)
        self.saveImageBtn.clicked.connect(self.saveImageResult)
        self.saveVideoBtn.clicked.connect(self.saveVideoResult)

        # 初始化定时器
        self.timer_camera = QtCore.QTimer()
        self.timer_camera.timeout.connect(self.show_camera)

        # 加载模型
        self.yolo_model = YOLO('runs/segment/train6/weights/best.pt')
        self.mask_rcnn_model = self.load_mask_rcnn_model()
        self.current_model = "YOLO"

        # 初始化变量
        self.frameToAnalyze = []
        self.image_scale = 1.0
        self.current_video_path = ""
        self.current_image = None
        self.current_result = None
        self.video_processing_thread = None
        self.cap = None

        # 启动帧分析线程
        Thread(target=self.frameAnalyzeThreadFunc, daemon=True).start()

    def setupUI(self):
        """设置用户界面"""
        self.resize(1200, 800)
        self.setWindowTitle('煤矿井下场景实例分割系统')

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

        # 进度条
        self.progressBar = QtWidgets.QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        self.progressBar.setTextVisible(True)
        self.progressBar.setFormat("等待处理...")
        mainLayout.addWidget(self.progressBar)

        # 使用QSplitter实现可调节大小的界面
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # 输出区域
        self.textLog = QtWidgets.QTextBrowser()
        font = QtGui.QFont()
        font.setPointSize(12)
        self.textLog.setFont(font)

        # 按钮区域
        buttonContainer = QtWidgets.QWidget()
        btnLayout = QtWidgets.QVBoxLayout(buttonContainer)

        # 模型选择
        self.modelComboBox = QtWidgets.QComboBox()
        self.modelComboBox.addItem("YOLO")
        self.modelComboBox.addItem("Mask R-CNN")
        self.modelComboBox.currentTextChanged.connect(self.change_model)

        # 功能按钮
        self.videoBtn = QtWidgets.QPushButton('🎞️视频文件')
        self.camBtn = QtWidgets.QPushButton('📹摄像头')
        self.imageBtn = QtWidgets.QPushButton('🖼️加载图片')
        self.stopBtn = QtWidgets.QPushButton('🛑停止')
        self.saveImageBtn = QtWidgets.QPushButton('💾保存图片结果')
        self.saveVideoBtn = QtWidgets.QPushButton('🎥后台处理视频')

        # 布局按钮
        btnLayout.addWidget(self.modelComboBox)
        btnLayout.addWidget(self.videoBtn)
        btnLayout.addWidget(self.camBtn)
        btnLayout.addWidget(self.imageBtn)
        btnLayout.addWidget(self.stopBtn)
        btnLayout.addWidget(QtWidgets.QLabel("保存选项:"))
        btnLayout.addWidget(self.saveImageBtn)
        btnLayout.addWidget(self.saveVideoBtn)
        btnLayout.addStretch()

        # 添加部件到分割器
        splitter.addWidget(self.textLog)
        splitter.addWidget(buttonContainer)
        splitter.setStretchFactor(0, 3)  # 输出区域占3份
        splitter.setStretchFactor(1, 1)  # 按钮区域占1份

        mainLayout.addWidget(splitter)

    def load_mask_rcnn_model(self):
        """加载Mask R-CNN模型"""
        cfg = get_cfg()
        cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.WEIGHTS = "detectron_ckps/model_final.pth"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = "cuda"
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
        MetadataCatalog.get("custom_dataset").thing_classes = ["person", "roadheader", "robot", "shearer"]
        return DefaultPredictor(cfg)

    def change_model(self, model_name):
        """切换当前模型"""
        self.current_model = model_name
        self.textLog.append(f"切换到模型: {model_name}")

    def saveImageResult(self):
        """保存图片处理结果"""
        if self.current_image is None or self.current_result is None:
            self.textLog.append("没有可保存的图片结果！")
            return

        options = QtWidgets.QFileDialog.Options()
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存图片分割结果", "",
            "图片文件 (*.png *.jpg *.bmp);;所有文件 (*)",
            options=options)

        if save_path:
            if self.current_model == "YOLO":
                img = self.current_result.plot(line_width=1)
            else:
                v = Visualizer(self.current_image[:, :, ::-1],
                              metadata=MetadataCatalog.get("custom_dataset"),
                              scale=0.8,
                              instance_mode=ColorMode.SEGMENTATION)
                v = v.draw_instance_predictions(self.current_result["instances"].to("cpu"))
                img = v.get_image()[:, :, ::-1]

            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, img_bgr)
            self.textLog.append(f"图片分割结果已保存到: {save_path}")

    def saveVideoResult(self):
        """后台处理并保存视频结果"""
        if not self.current_video_path:
            self.textLog.append("请先选择要处理的视频文件！")
            return

        if self.video_processing_thread and self.video_processing_thread.isRunning():
            self.textLog.append("已有视频正在处理中，请等待完成！")
            return

        options = QtWidgets.QFileDialog.Options()
        output_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "选择保存位置", "",
            "视频文件 (*.mp4);;所有文件 (*)",
            options=options)

        if not output_path:
            return

        self.progressBar.setValue(0)
        self.progressBar.setFormat("正在后台处理视频...")

        # 创建并启动视频处理线程
        self.video_processing_thread = VideoProcessingThread(
            self.current_video_path,
            output_path,
            self.current_model,
            self.yolo_model,
            self.mask_rcnn_model,
            self
        )

        # 连接信号
        self.video_processing_thread.progress_updated.connect(self.update_progress)
        self.video_processing_thread.processing_finished.connect(self.video_processing_finished)

        # 启动线程
        self.video_processing_thread.start()
        self.textLog.append(f"开始后台处理视频: {self.current_video_path}")

    def update_progress(self, value):
        """更新进度条"""
        self.progressBar.setValue(value)
        self.progressBar.setFormat(f"后台处理进度: {value}%")

    def video_processing_finished(self, message):
        """视频处理完成回调"""
        self.textLog.append(message)
        self.progressBar.setFormat("后台处理完成")
        self.video_processing_thread = None

    def openVideoFile(self):
        """打开视频文件"""
        options = QtWidgets.QFileDialog.Options()
        videoPath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov);;所有文件 (*)",
            options=options)

        if videoPath:
            self.current_video_path = videoPath
            self.cap = cv2.VideoCapture(videoPath)

            if not self.cap.isOpened():
                self.textLog.append("视频文件无法打开")
                return

            if not self.timer_camera.isActive():
                self.timer_camera.start(50)

            self.textLog.append(f"已加载视频: {videoPath}")

    def startCamera(self):
        """启动摄像头"""
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.textLog.append("摄像头不能打开")
            return

        if not self.timer_camera.isActive():
            self.timer_camera.start(50)

    def openImageFile(self):
        """打开图片文件"""
        options = QtWidgets.QFileDialog.Options()
        imagePath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择图片文件", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)",
            options=options)

        if imagePath:
            img = cv2.imread(imagePath)
            if img is None:
                self.textLog.append("无法读取图片文件")
                return

            # 调整图片尺寸
            height, width = img.shape[:2]
            max_width, max_height = 520, 400
            if width > max_width or height > max_height:
                self.image_scale = min(max_width / width, max_height / height)
                img = cv2.resize(img, (int(width * self.image_scale), int(height * self.image_scale)))
            else:
                self.image_scale = 1.0

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.current_image = img

            # 处理图片
            if self.current_model == "YOLO":
                results = self.yolo_model(img)[0]
                img_treated = results.plot(line_width=1)
                self.current_result = results
            else:
                outputs = self.mask_rcnn_model(img)
                v = Visualizer(img[:, :, ::-1],
                              metadata=MetadataCatalog.get("custom_dataset"),
                              scale=0.8,
                              instance_mode=ColorMode.SEGMENTATION)
                v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                img_treated = v.get_image()[:, :, ::-1]
                self.current_result = outputs

            # 显示图片
            img_treated = cv2.resize(img_treated, (img.shape[1], img.shape[0]))
            img = np.ascontiguousarray(img)
            img_treated = np.ascontiguousarray(img_treated)

            self.displayImage(img, self.label_ori_video)
            self.displayImage(img_treated, self.label_treated)

            # 记录结果
            self.textLog.append(f"{self.current_model} 图片识别结果：")
            if self.current_model == "YOLO":
                for result in results:
                    self.textLog.append(
                        f"检测到: {result.names[result.boxes.cls[0].item()]} 置信度: {result.boxes.conf[0].item():.2f}")
            else:
                for i, score in enumerate(outputs["instances"].scores):
                    class_id = outputs["instances"].pred_classes[i].item()
                    class_name = MetadataCatalog.get("custom_dataset").thing_classes[class_id]
                    self.textLog.append(f"检测到: {class_name} 置信度: {score.item():.2f}")

    def displayImage(self, img, label):
        """在QLabel上显示图片"""
        img = np.ascontiguousarray(img)
        qImage = QtGui.QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QtGui.QImage.Format_RGB888)
        label.setPixmap(QtGui.QPixmap.fromImage(qImage))

    def show_camera(self):
        """显示摄像头或视频帧"""
        if hasattr(self, 'video_processing_thread') and self.video_processing_thread and self.video_processing_thread.isRunning():
            return  # 后台处理视频时不更新显示

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 调整尺寸
        height, width = frame.shape[:2]
        max_width, max_height = 520, 400
        if width > max_width or height > max_height:
            self.image_scale = min(max_width / width, max_height / height)
            frame = cv2.resize(frame, (int(width * self.image_scale), int(height * self.image_scale)))
        else:
            self.image_scale = 1.0

        frame = np.ascontiguousarray(frame)
        self.displayImage(frame, self.label_ori_video)

        # 添加到分析队列
        if not self.frameToAnalyze:
            self.frameToAnalyze.append(frame)

    def frameAnalyzeThreadFunc(self):
        """帧分析线程函数"""
        while True:
            if not self.frameToAnalyze:
                time.sleep(0.01)
                continue

            frame = self.frameToAnalyze.pop(0)

            # 处理帧
            if self.current_model == "YOLO":
                results = self.yolo_model(frame)[0]
                img = results.plot(line_width=1)
                self.current_result = results
            else:
                outputs = self.mask_rcnn_model(frame)
                v = Visualizer(frame[:, :, ::-1],
                              metadata=MetadataCatalog.get("custom_dataset"),
                              scale=0.8,
                              instance_mode=ColorMode.SEGMENTATION)
                v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                img = v.get_image()[:, :, ::-1]
                self.current_result = outputs

            # 显示处理结果
            img = cv2.resize(img, (frame.shape[1], frame.shape[0]))
            img = np.ascontiguousarray(img)
            self.displayImage(img, self.label_treated)

            # 记录结果
            self.textLog.append(f"{self.current_model} 视频帧识别结果：")
            if self.current_model == "YOLO":
                for result in results:
                    self.textLog.append(
                        f"检测到: {result.names[result.boxes.cls[0].item()]} 置信度: {result.boxes.conf[0].item():.2f}")
            else:
                for i, score in enumerate(outputs["instances"].scores):
                    class_id = outputs["instances"].pred_classes[i].item()
                    class_name = MetadataCatalog.get("custom_dataset").thing_classes[class_id]
                    self.textLog.append(f"检测到: {class_name} 置信度: {score.item():.2f}")

            time.sleep(0.5)

    def stop(self):
        """停止所有处理"""
        self.timer_camera.stop()

        # 释放视频捕获
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

        # 停止视频处理线程
        if hasattr(self, 'video_processing_thread') and self.video_processing_thread and self.video_processing_thread.isRunning():
            self.video_processing_thread.stop()
            self.video_processing_thread.wait()
            self.textLog.append("视频处理已停止")

        # 清空显示
        self.label_ori_video.clear()
        self.label_treated.clear()

    def closeEvent(self, event):
        """窗口关闭事件"""
        self.stop()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MWindow()
    window.show()
    app.exec()