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


class VideoProcessingThread(QtCore.QThread):
    progress_updated = QtCore.Signal(int)
    processing_finished = QtCore.Signal(str)

    def __init__(self, video_path, output_path, model, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.output_path = output_path
        self.model = model
        self._is_running = True

    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.processing_finished.emit("无法打开视频文件")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 创建输出视频文件
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

            frame_count = 0
            while self._is_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = inference_model(self.model, frame_rgb)
                segmented_frame = self.visualize_result(frame_rgb, result)
                segmented_bgr = cv2.cvtColor(segmented_frame, cv2.COLOR_RGB2BGR)
                out.write(segmented_bgr)

                frame_count += 1
                progress = int((frame_count / total_frames) * 100)
                self.progress_updated.emit(progress)

            cap.release()
            out.release()
            self.processing_finished.emit(f"视频处理完成，已保存到: {self.output_path}")

        except Exception as e:
            self.processing_finished.emit(f"处理出错: {str(e)}")

    def stop(self):
        self._is_running = False

    def visualize_result(self, img, result):
        seg = result.pred_sem_seg.data[0].cpu().numpy()
        palette = self.model.PALETTE
        classes = self.model.CLASSES

        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color

        img = img * 0.5 + color_seg * 0.5
        img = img.astype(np.uint8)
        return img


class MWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUI()

        # 连接按钮事件
        self.videoBtn.clicked.connect(self.openVideoFile)
        self.camBtn.clicked.connect(self.startCamera)
        self.imageBtn.clicked.connect(self.openImageFile)
        self.stopBtn.clicked.connect(self.stop)
        self.saveImageBtn.clicked.connect(self.saveImageResult)
        self.saveVideoBtn.clicked.connect(self.saveVideoResult)

        # 定时器用于显示视频帧
        self.timer_camera = QtCore.QTimer()
        self.timer_camera.timeout.connect(self.show_camera)

        # 视频处理线程
        self.video_processing_thread = None

        # 加载 MMSegmentation 模型
        self.mmseg_models = {
            "deeplabv3+": self.load_mmseg_model("deeplabv3+"),
            "pspnet": self.load_mmseg_model("pspnet")
        }
        self.current_model = "deeplabv3+"

        # 视频帧队列
        self.frameToAnalyze = []

        # 保存当前处理结果
        self.current_result = None
        self.current_image = None
        self.current_video_path = ""

        # 启动处理帧的线程
        Thread(target=self.frameAnalyzeThreadFunc, daemon=True).start()

    def setupUI(self):
        self.resize(1200, 800)
        self.setWindowTitle('煤矿井下场景语义分割系统')

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

        # 进度条
        self.progressBar = QtWidgets.QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        self.progressBar.setTextVisible(True)
        self.progressBar.setFormat("等待处理...")
        mainLayout.addWidget(self.progressBar)

        # 日志输出和按钮区域 - 使用QSplitter实现可调节大小
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # 输出区域
        self.textLog = QtWidgets.QTextBrowser()
        font = QtGui.QFont()
        font.setPointSize(15)
        self.textLog.setFont(font)

        # 按钮区域
        buttonContainer = QtWidgets.QWidget()
        btnLayout = QtWidgets.QVBoxLayout(buttonContainer)

        # 模型选择
        self.modelComboBox = QtWidgets.QComboBox()
        self.modelComboBox.addItem("deeplabv3+")
        self.modelComboBox.addItem("pspnet")
        self.modelComboBox.currentTextChanged.connect(self.change_model)

        # 功能按钮
        self.videoBtn = QtWidgets.QPushButton('🎞️视频文件')
        self.camBtn = QtWidgets.QPushButton('📹摄像头')
        self.imageBtn = QtWidgets.QPushButton('🖼️加载图片')
        self.stopBtn = QtWidgets.QPushButton('🛑停止')

        # 保存按钮
        self.saveImageBtn = QtWidgets.QPushButton('💾保存图片结果')
        self.saveVideoBtn = QtWidgets.QPushButton('🎥后台处理视频')

        # 添加到布局
        btnLayout.addWidget(self.modelComboBox)
        btnLayout.addWidget(self.videoBtn)
        btnLayout.addWidget(self.camBtn)
        btnLayout.addWidget(self.imageBtn)
        btnLayout.addWidget(self.stopBtn)
        btnLayout.addWidget(QtWidgets.QLabel("保存选项:"))
        btnLayout.addWidget(self.saveImageBtn)
        btnLayout.addWidget(self.saveVideoBtn)
        btnLayout.addStretch()

        # 添加到分割器
        splitter.addWidget(self.textLog)
        splitter.addWidget(buttonContainer)
        splitter.setStretchFactor(0, 3)  # 输出区域占3份
        splitter.setStretchFactor(1, 1)  # 按钮区域占1份

        mainLayout.addWidget(splitter)

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

        model = init_model(config_file, checkpoint_file, device='cuda')

        # 手动指定类别信息和颜色
        model.CLASSES = ['background', 'person', 'roadheader', 'robot', 'shearer']
        model.PALETTE = [
            [127, 127, 127],  # 背景色：中等亮度的灰色
            [255, 0, 0],  # person：红色
            [0, 200, 0],  # roadheader：浅绿色
            [0, 0, 255],  # robot：蓝色
            [255, 215, 0]  # shearer：金黄色
        ]

        return model

    def change_model(self, model_name):
        """切换模型"""
        self.current_model = model_name
        self.textLog.append(f"切换到模型: {model_name}")

    def saveImageResult(self):
        """保存图片分割结果"""
        if self.current_result is None:
            self.textLog.append("没有可保存的图片结果！")
            return

        options = QtWidgets.QFileDialog.Options()
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存图片分割结果", "",
            "图片文件 (*.png *.jpg *.bmp);;所有文件 (*)",
            options=options)

        if save_path:
            img = self.visualize_result(self.current_image, self.current_result)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, img_bgr)
            self.textLog.append(f"图片分割结果已保存到: {save_path}")

    def saveVideoResult(self):
        """后台处理并保存视频分割结果"""
        if not self.current_video_path:
            self.textLog.append("请先选择要处理的视频文件！")
            return

        if self.video_processing_thread and self.video_processing_thread.isRunning():
            self.textLog.append("已有视频正在处理中，请等待完成！")
            return

        # 选择保存位置和文件名
        options = QtWidgets.QFileDialog.Options()
        output_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "选择保存位置", "",
            "视频文件 (*.mp4);;所有文件 (*)",
            options=options)

        if not output_path:
            return

        self.progressBar.setValue(0)
        self.progressBar.setFormat("正在处理视频...")

        self.video_processing_thread = VideoProcessingThread(
            self.current_video_path,
            output_path,
            self.mmseg_models[self.current_model]
        )
        self.video_processing_thread.progress_updated.connect(self.update_progress)
        self.video_processing_thread.processing_finished.connect(self.video_processing_finished)
        self.video_processing_thread.start()

        self.textLog.append(f"开始后台处理视频: {self.current_video_path}")

    def update_progress(self, value):
        """更新进度条"""
        self.progressBar.setValue(value)
        self.progressBar.setFormat(f"处理进度: {value}%")

    def video_processing_finished(self, message):
        """视频处理完成"""
        self.textLog.append(message)
        self.progressBar.setFormat("处理完成")
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

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = inference_model(self.mmseg_models[self.current_model], img_rgb)

            self.current_result = result
            self.current_image = img_rgb

            img_treated = self.visualize_result(img_rgb, result)

            img_resized = self.resize_with_aspect_ratio(img_rgb, (520, 400))
            img_treated_resized = self.resize_with_aspect_ratio(img_treated, (520, 400))

            self.displayImage(img_resized, self.label_ori_video)
            self.displayImage(img_treated_resized, self.label_treated)

            self.textLog.append(f"{self.current_model} 图片识别结果：")
            self.show_label_info(result)

    def resize_with_aspect_ratio(self, image, target_size):
        """保持宽高比缩放图像"""
        (target_width, target_height) = target_size
        (height, width) = image.shape[:2]

        if width > height:
            scale = target_width / width
        else:
            scale = target_height / height

        new_size = (int(width * scale), int(height * scale))
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

        background = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        x_offset = (target_width - new_size[0]) // 2
        y_offset = (target_height - new_size[1]) // 2
        background[y_offset:y_offset + new_size[1], x_offset:x_offset + new_size[0]] = resized_image

        return background

    def visualize_result(self, img, result):
        """可视化分割结果"""
        model = self.mmseg_models[self.current_model]
        seg = result.pred_sem_seg.data[0].cpu().numpy()
        palette = model.PALETTE
        classes = model.CLASSES

        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color

        img = img * 0.5 + color_seg * 0.5
        img = img.astype(np.uint8)

        font_scale = img.shape[0] / 1000
        font_thickness = max(1, int(font_scale * 3))

        unique_labels = np.unique(seg)
        for label in unique_labels:
            if label >= len(classes):
                continue
            mask = (seg == label).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) < 100:
                    continue

                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0

                text = classes[label]
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                                               font_thickness)
                text_x = cX - text_width // 2
                text_y = cY + text_height // 2
                cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                            font_thickness)
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
        classes = model.CLASSES
        palette = model.PALETTE

        for label in unique_labels:
            if label >= len(classes):
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

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if not self.frameToAnalyze:
            self.frameToAnalyze.append(frame_rgb)

        frame_resized = self.resize_with_aspect_ratio(frame_rgb, (520, 400))
        self.displayImage(frame_resized, self.label_ori_video)

    def frameAnalyzeThreadFunc(self):
        """处理帧的线程函数"""
        while True:
            if not self.frameToAnalyze:
                time.sleep(0.01)
                continue

            frame = self.frameToAnalyze.pop(0)
            result = inference_model(self.mmseg_models[self.current_model], frame)

            self.current_result = result
            self.current_image = frame

            img = self.visualize_result(frame, result)
            img_resized = self.resize_with_aspect_ratio(img, (520, 400))

            self.displayImage(img_resized, self.label_treated)
            self.textLog.append(f"{self.current_model} 视频帧识别结果：")
            self.show_label_info(result)

            time.sleep(0.5)

    def stop(self):
        """停止"""
        self.timer_camera.stop()
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if self.video_processing_thread and self.video_processing_thread.isRunning():
            self.video_processing_thread.stop()
            self.video_processing_thread.wait()
            self.textLog.append("视频处理已停止")
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