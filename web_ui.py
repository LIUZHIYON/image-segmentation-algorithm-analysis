"""
图像分割算法分析与应用 — Web 统一界面
合并语义分割 (DeepLabV3+/PSPNet) 与实例分割 (YOLOv11/Mask R-CNN)
基于 Gradio 构建
"""
import os, sys, time, cv2, numpy as np
from datetime import datetime
from pathlib import Path
import threading
import gradio as gr

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["YOLO_VERBOSE"] = "False"

# ── 模型路径配置 ──────────────────────────────────────────
BASE_DIR = Path(__file__).parent

# MMSegmentation 语义分割模型
MMSEG_DIR = BASE_DIR / "mmsegmentation-main"
sys.path.insert(0, str(MMSEG_DIR))

# YOLOv11 实例分割模型
YOLO_DIR = BASE_DIR / "yolov11_custom_segmentation"

# Detectron2 实例分割模型
D2_DIR = YOLO_DIR

# ── 全局变量 ──────────────────────────────────────────────
models = {}           # 已加载的模型缓存
model_lock = threading.Lock()
log_history = []      # 日志历史

# ── 模型加载 ──────────────────────────────────────────────

def load_mmseg_model(config_rel, ckpt_rel, classes, palette):
    """加载 MMSegmentation 模型"""
    from mmseg.apis import init_model
    config_file = str(MMSEG_DIR / config_rel)
    ckpt_file = str(MMSEG_DIR / ckpt_rel)
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"配置文件不存在: {config_file}")
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"权重文件不存在: {ckpt_file}")
    model = init_model(config_file, ckpt_file, device="cuda" if __import__("torch").cuda.is_available() else "cpu")
    model.CLASSES = classes
    model.PALETTE = palette
    return model

def load_yolo_model():
    """加载 YOLOv11 模型"""
    from ultralytics import YOLO
    ckpt = str(YOLO_DIR / "runs/segment/train6/weights/best.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"YOLO权重文件不存在: {ckpt}")
    return YOLO(ckpt)

def load_mask_rcnn_model():
    """加载 Mask R-CNN 模型"""
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog

    cfg_file = str(D2_DIR / "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    ckpt_file = str(D2_DIR / "detectron_ckps/model_final.pth")
    if not os.path.exists(cfg_file):
        raise FileNotFoundError(f"Mask R-CNN 配置文件不存在: {cfg_file}")
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"Mask R-CNN 权重文件不存在: {ckpt_file}")

    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    cfg.MODEL.WEIGHTS = ckpt_file
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    MetadataCatalog.get("custom_dataset").thing_classes = ["person", "roadheader", "robot", "shearer"]
    return DefaultPredictor(cfg)

def get_model(model_key):
    """获取模型（带缓存）"""
    with model_lock:
        if model_key in models:
            return models[model_key]

        try:
            if model_key == "DeepLabV3+":
                models[model_key] = load_mmseg_model(
                    "Zihao-Configs/ZihaoDataset_DeepLabV3plus_20230818.py",
                    "work_dirs/ZihaoDataset-DeepLabV3plus/best_mIoU_iter_3400.pth",
                    ["background", "person", "roadheader", "robot", "shearer"],
                    [[127, 127, 127], [255, 0, 0], [0, 200, 0], [0, 0, 255], [255, 215, 0]]
                )
            elif model_key == "PSPNet":
                models[model_key] = load_mmseg_model(
                    "Zihao-Configs/ZihaoDataset_PSPNet_20230818.py",
                    "work_dirs/ZihaoDataset-PSPNet/best_mIoU_iter_3800.pth",
                    ["background", "person", "roadheader", "robot", "shearer"],
                    [[127, 127, 127], [255, 0, 0], [0, 200, 0], [0, 0, 255], [255, 215, 0]]
                )
            elif model_key == "YOLOv11":
                models[model_key] = load_yolo_model()
            elif model_key == "Mask R-CNN":
                models[model_key] = load_mask_rcnn_model()
        except Exception as e:
            models[model_key] = None
            raise
        return models[model_key]

# ── 处理函数 ──────────────────────────────────────────────

SEMANTIC_CLASSES = ["background", "person", "roadheader", "robot", "shearer"]
SEMANTIC_PALETTE = [[127,127,127], [255,0,0], [0,200,0], [0,0,255], [255,215,0]]
INSTANCE_CLASSES = ["person", "roadheader", "robot", "shearer"]

def add_log(log_box, msg):
    """添加日志"""
    global log_history
    ts = datetime.now().strftime("%H:%M:%S")
    formatted = f"[{ts}] {msg}"
    log_history.append(formatted)
    if len(log_history) > 200:
        log_history = log_history[-200:]
    return "\n".join(log_history[-50:])

def _mmseg_infer(model, img_rgb):
    """MMSegmentation 推理"""
    from mmseg.apis import inference_model
    return inference_model(model, img_rgb)

def _visualize_mmseg(model, img_rgb, result):
    """可视化 MMSeg 分割结果"""
    seg = result.pred_sem_seg.data[0].cpu().numpy()
    palette = model.PALETTE
    classes = model.CLASSES

    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color

    overlay = (img_rgb * 0.5 + color_seg * 0.5).astype(np.uint8)

    unique_labels = np.unique(seg)
    for label in unique_labels:
        if label >= len(classes) or label == 0:
            continue
        mask = (seg == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 100:
            continue
        M = cv2.moments(largest)
        if M["m00"] != 0:
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
        else:
            cx, cy = 0, 0
        cv2.putText(overlay, classes[label], (cx-20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return overlay

def process_image(model_key, input_img):
    """处理单张图片"""
    if input_img is None:
        return None, None, "请上传图片"

    log_box = ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        model = get_model(model_key)
        if model is None:
            return None, None, f"❌ {model_key} 模型加载失败，请检查路径"

        img_bgr = input_img
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        log_box = add_log(None, f"开始处理图片 — 模型: {model_key}")

        if model_key in ["DeepLabV3+", "PSPNet"]:
            # 语义分割
            result = _mmseg_infer(model, img_rgb)
            output = _visualize_mmseg(model, img_rgb, result)

            seg = result.pred_sem_seg.data[0].cpu().numpy()
            unique_labels = np.unique(seg)
            for label in unique_labels:
                if label >= len(SEMANTIC_CLASSES) or label == 0:
                    continue
                pixel_count = int((seg == label).sum())
                area_pct = pixel_count / seg.size * 100
                log_box = add_log(log_box, f"  🟢 {SEMANTIC_CLASSES[label]}: {area_pct:.1f}%")
            log_box = add_log(log_box, f"✅ 语义分割完成")

        elif model_key == "YOLOv11":
            results = model(img_rgb)[0]
            output = results.plot(line_width=2)
            for box, cls_id, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
                cls_id = int(cls_id.item())
                label = results.names[cls_id]
                conf = conf.item()
                log_box = add_log(log_box, f"  🔵 {label}: 置信度 {conf:.2%}")
            log_box = add_log(log_box, f"✅ YOLOv11 实例分割完成")

        else:  # Mask R-CNN
            outputs = model(img_rgb)
            from detectron2.utils.visualizer import Visualizer, ColorMode
            from detectron2.data import MetadataCatalog
            v = Visualizer(img_rgb[:, :, ::-1],
                          metadata=MetadataCatalog.get("custom_dataset"),
                          scale=0.8,
                          instance_mode=ColorMode.SEGMENTATION)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            output = v.get_image()[:, :, ::-1]
            for i, score in enumerate(outputs["instances"].scores):
                cls_id = outputs["instances"].pred_classes[i].item()
                label = INSTANCE_CLASSES[cls_id]
                log_box = add_log(log_box, f"  🔴 {label}: 置信度 {score.item():.2%}")
            log_box = add_log(log_box, f"✅ Mask R-CNN 实例分割完成")

        return img_rgb, output, log_box

    except Exception as e:
        err_msg = f"❌ 处理出错: {str(e)}"
        log_box = add_log(log_box or None, err_msg)
        import traceback
        log_box = add_log(log_box, traceback.format_exc())
        return input_img, None, log_box

def process_video(model_key, video_path, progress=gr.Progress()):
    """处理视频文件"""
    if video_path is None:
        return None, "请上传视频文件"

    log_box = ""
    try:
        model = get_model(model_key)
        if model is None:
            return None, f"❌ {model_key} 模型加载失败"

        log_box = add_log(None, f"开始处理视频 — 模型: {model_key}")

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total == 0:
            return None, "视频为空或无法读取"

        # 输出路径
        out_name = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        out_path = str(BASE_DIR / "outputs" / out_name)
        os.makedirs(str(BASE_DIR / "outputs"), exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        frame_idx = 0
        log_box = add_log(log_box, f"视频信息: {total}帧, {fps}fps, {w}x{h}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_bgr = frame

            if model_key in ["DeepLabV3+", "PSPNet"]:
                result = _mmseg_infer(model, frame_rgb)
                processed = _visualize_mmseg(model, frame_rgb, result)
                processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
            elif model_key == "YOLOv11":
                results = model(frame_rgb)[0]
                processed = results.plot(line_width=1)
                processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
            else:  # Mask R-CNN
                outputs = model(frame_rgb)
                from detectron2.utils.visualizer import Visualizer, ColorMode
                from detectron2.data import MetadataCatalog
                v = Visualizer(frame_rgb[:, :, ::-1],
                              metadata=MetadataCatalog.get("custom_dataset"),
                              scale=0.8,
                              instance_mode=ColorMode.SEGMENTATION)
                v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                processed = v.get_image()[:, :, ::-1]
                processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)

            writer.write(processed_bgr)
            frame_idx += 1
            progress(frame_idx / total, desc=f"处理中 {frame_idx}/{total} 帧")

        cap.release()
        writer.release()
        log_box = add_log(log_box, f"✅ 视频处理完成，共 {frame_idx} 帧")
        log_box = add_log(log_box, f"📁 保存到: {out_path}")

        return out_path, log_box

    except Exception as e:
        err_msg = f"❌ 视频处理出错: {str(e)}"
        log_box = add_log(log_box or None, err_msg)
        import traceback
        log_box = add_log(log_box, traceback.format_exc())
        return None, log_box

def process_webcam(model_key, img):
    """处理摄像头帧"""
    if img is None:
        return None, "等待摄像头..."

    log_box = ""
    try:
        model = get_model(model_key)
        if model is None:
            return None, f"❌ {model_key} 模型加载失败"

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if model_key in ["DeepLabV3+", "PSPNet"]:
            result = _mmseg_infer(model, img_rgb)
            output = _visualize_mmseg(model, img_rgb, result)
        elif model_key == "YOLOv11":
            results = model(img_rgb)[0]
            output = results.plot(line_width=2)
        else:
            outputs = model(img_rgb)
            from detectron2.utils.visualizer import Visualizer, ColorMode
            from detectron2.data import MetadataCatalog
            v = Visualizer(img_rgb[:, :, ::-1],
                          metadata=MetadataCatalog.get("custom_dataset"),
                          scale=0.8,
                          instance_mode=ColorMode.SEGMENTATION)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            output = v.get_image()[:, :, ::-1]

        return output, "实时分割中..."

    except Exception as e:
        return img, f"处理出错: {str(e)}"

# ── 保存结果 ──────────────────────────────────────────────

def save_result(img, model_key):
    """保存分割结果"""
    if img is None:
        return "没有可保存的结果"

    out_dir = BASE_DIR / "outputs"
    os.makedirs(str(out_dir), exist_ok=True)
    fname = f"{model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    path = str(out_dir / fname)

    if isinstance(img, np.ndarray):
        if len(img.shape) == 3 and img.shape[2] == 3:
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(path, img)
    return f"已保存: {path}"

# ── CSS 样式 ──────────────────────────────────────────────

CUSTOM_CSS = """
:root {
    --primary: #6366f1;
    --primary-light: #818cf8;
    --bg-dark: #0f0f1a;
    --bg-card: #1a1a2e;
    --bg-input: #16213e;
    --text: #e2e8f0;
    --text-dim: #94a3b8;
    --border: #2d2d4a;
    --success: #22c55e;
    --error: #ef4444;
    --info: #3b82f6;
}
body { background: var(--bg-dark); color: var(--text); }
.gradio-container { max-width: 1400px !important; margin: 0 auto; background: transparent; }
.tabs { border: none; }
.tab-nav { background: var(--bg-card) !important; border-radius: 12px 12px 0 0 !important; padding: 8px !important; gap: 4px; }
.tab-nav button { border-radius: 8px !important; padding: 8px 20px !important; font-weight: 600 !important; transition: all 0.2s; }
.tab-nav button.selected { background: var(--primary) !important; color: white !important; }
.gr-box { border-radius: 12px !important; background: var(--bg-card) !important; border: 1px solid var(--border) !important; overflow: hidden; }
.gr-input, .gr-select, input, select, textarea { background: var(--bg-input) !important; border: 1px solid var(--border) !important; color: var(--text) !important; border-radius: 8px !important; }
.gr-input:focus, select:focus { border-color: var(--primary) !important; box-shadow: 0 0 0 2px rgba(99,102,241,0.2) !important; }
button.gr-button, .gr-button { border-radius: 8px !important; font-weight: 600 !important; padding: 8px 20px !important; transition: all 0.2s !important; }
button.gr-button.gr-button-primary { background: var(--primary) !important; border: none !important; }
button.gr-button.gr-button-primary:hover { background: var(--primary-light) !important; transform: translateY(-1px); }
.gr-panel { border-radius: 12px !important; background: var(--bg-card) !important; border: 1px solid var(--border) !important; }
.gr-json { background: var(--bg-input) !important; }
label { color: var(--text) !important; font-weight: 500 !important; }
.gr-markdown h1, .gr-markdown h2, .gr-markdown h3 { color: var(--text) !important; }
footer { display: none !important; }
header { display: none !important; }
.video-container video { border-radius: 8px; }
.gr-panel.gr-compact { padding: 0 !important; }
.gr-gallery { background: transparent !important; }
.gr-accordion { border-radius: 8px !important; background: var(--bg-card) !important; border: 1px solid var(--border) !important; }
progress { accent-color: var(--primary); }
"""

HTML_HEADER = """
<div style="
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
    padding: 24px 32px;
    border-radius: 16px;
    margin-bottom: 20px;
    color: white;
    box-shadow: 0 8px 32px rgba(99,102,241,0.3);
">
    <h1 style="margin:0; font-size: 28px; font-weight: 700; letter-spacing: -0.5px;">
        🎯 图像分割算法分析系统
    </h1>
    <p style="margin:8px 0 0; opacity:0.9; font-size: 15px;">
        统一管理 4 种分割模型 · 支持图片 / 视频 / 摄像头实时分割
    </p>
</div>
"""

MODEL_DESCRIPTIONS = {
    "DeepLabV3+": "🧩 语义分割 · 空洞卷积 · 边缘精细",
    "PSPNet": "🏗️ 语义分割 · 金字塔池化 · 场景理解",
    "YOLOv11": "⚡ 实例分割 · 实时推理 · 速度优先",
    "Mask R-CNN": "🎯 实例分割 · 高精度 · 掩码精细",
}

# ── 构建界面 ──────────────────────────────────────────────

def build_app():
    with gr.Blocks(css=CUSTOM_CSS, title="图像分割算法分析系统", theme=gr.themes.Soft()) as demo:
        gr.HTML(HTML_HEADER)

        with gr.Row(equal_height=False):
            # ── 左侧：控制面板 ──
            with gr.Column(scale=1, min_width=280):
                with gr.Group(elem_classes=["gr-box"]):
                    gr.Markdown("### ⚙️ 模型选择")

                    model_selector = gr.Dropdown(
                        choices=[
                            ("🧩 DeepLabV3+ (语义分割)", "DeepLabV3+"),
                            ("🏗️ PSPNet (语义分割)", "PSPNet"),
                            ("⚡ YOLOv11 (实例分割)", "YOLOv11"),
                            ("🎯 Mask R-CNN (实例分割)", "Mask R-CNN"),
                        ],
                        value="DeepLabV3+",
                        label="模型",
                        interactive=True,
                    )

                    model_desc = gr.Markdown(MODEL_DESCRIPTIONS["DeepLabV3+"])

                    def update_model_desc(name):
                        return MODEL_DESCRIPTIONS.get(name, "")

                    model_selector.change(update_model_desc, model_selector, model_desc)

                    gr.Markdown("---")
                    gr.Markdown("### 📋 操作说明")
                    gr.Markdown(
                        "1. **选择模型** 👈 在下方选择要使用的分割模型\n"
                        "2. **图片分割** 📷 上传图片 → 自动分割 → 保存结果\n"
                        "3. **视频分割** 🎬 上传视频 → 逐帧处理 → 下载结果\n"
                        "4. **摄像头** 📹 打开摄像头 → 实时分割预览\n"
                        "5. **查看日志** 📜 右侧日志面板记录每次操作"
                    )

                    save_btn = gr.Button("💾 保存结果", variant="primary")
                    save_status = gr.Textbox(label="保存状态", interactive=False)

                with gr.Group(elem_classes=["gr-box"]):
                    gr.Markdown("### 📊 类别信息")
                    classes_info = gr.Markdown(
                        "**语义分割 (DeepLabV3+ / PSPNet):**\n"
                        "- 🟦 background · 🟥 person\n"
                        "- 🟩 roadheader · 🟦 robot\n"
                        "- 🟨 shearer\n\n"
                        "**实例分割 (YOLOv11 / Mask R-CNN):**\n"
                        "- person · roadheader · robot · shearer"
                    )

            # ── 右侧：主内容 ──
            with gr.Column(scale=3):
                with gr.Tabs(elem_classes=["tabs"]) as tabs:
                    # ===== Tab1: 图片 =====
                    with gr.TabItem("📷 图片分割", id="image"):
                        with gr.Row(equal_height=True):
                            with gr.Column():
                                gr.Markdown("##### 原始图片")
                                img_input = gr.Image(label="上传图片", type="numpy", height=400)
                            with gr.Column():
                                gr.Markdown("##### 分割结果")
                                img_output = gr.Image(label="结果", type="numpy", height=400)

                        with gr.Row():
                            img_process_btn = gr.Button("🚀 开始分割", variant="primary", size="lg", scale=2)
                            img_save_btn = gr.Button("💾 保存", scale=1)
                            img_clear_btn = gr.Button("🗑️ 清空", scale=1)

                    # ===== Tab2: 视频 =====
                    with gr.TabItem("🎬 视频分割", id="video"):
                        vid_input = gr.Video(label="上传视频", height=300)
                        vid_output = gr.Video(label="分割结果", height=300)
                        vid_process_btn = gr.Button("🚀 处理视频", variant="primary", size="lg")

                    # ===== Tab3: 摄像头 =====
                    with gr.TabItem("📹 摄像头实时分割", id="webcam"):
                        gr.Markdown("##### 选择模型后打开摄像头，实时查看分割效果")
                        webcam_input = gr.Image(sources=["webcam"], streaming=True, label="摄像头", height=400)
                        webcam_output = gr.Image(label="分割结果", height=400)

                # ── 日志面板 ──
                with gr.Group(elem_classes=["gr-box"]):
                    gr.Markdown("### 📜 处理日志")
                    log_output = gr.Textbox(
                        label="日志",
                        lines=8,
                        max_lines=15,
                        interactive=False,
                        value="[系统] 欢迎使用图像分割分析系统\n[系统] 请选择模型并上传数据开始处理"
                    )

        # ── 事件绑定 ──

        # 图片处理
        def img_process(model_key, img):
            orig, seg, log = process_image(model_key, img)
            if seg is not None and seg.shape[:2] != orig.shape[:2]:
                seg = cv2.resize(seg, (orig.shape[1], orig.shape[0]))
            return seg, log

        img_process_btn.click(
            fn=img_process,
            inputs=[model_selector, img_input],
            outputs=[img_output, log_output]
        )

        img_save_btn.click(
            fn=save_result,
            inputs=[img_output, model_selector],
            outputs=[save_status]
        )

        def img_clear():
            return None, None

        img_clear_btn.click(fn=img_clear, outputs=[img_input, img_output])

        save_btn.click(
            fn=save_result,
            inputs=[img_output, model_selector],
            outputs=[save_status]
        )

        # 视频处理
        vid_process_btn.click(
            fn=process_video,
            inputs=[model_selector, vid_input],
            outputs=[vid_output, log_output]
        )

        # 摄像头实时处理
        webcam_input.stream(
            fn=process_webcam,
            inputs=[model_selector, webcam_input],
            outputs=[webcam_output, log_output],
            time_limit=60,
            concurrency_limit=2
        )

    return demo

# ── 启动 ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  🎯 图像分割算法分析系统")
    print("  ===============================")
    print(f"  • 语义分割: DeepLabV3+ / PSPNet")
    print(f"  • 实例分割: YOLOv11 / Mask R-CNN")
    print(f"  • 输出目录: {BASE_DIR / 'outputs'}")
    print("=" * 60)
    print()

    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True
    )
