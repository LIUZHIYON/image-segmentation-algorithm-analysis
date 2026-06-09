"""
图像分割算法分析与应用 — Web 统一界面
合并语义分割 (DeepLabV3+/PSPNet/SegFormer) 与实例分割 (YOLOv11/Mask R-CNN)
基于 Gradio 构建，支持 5 个模型的图片/视频/摄像头实时分割

══════════════════════════════════════════════════════════════
启动方式:  E:\anaconda3\envs\openmmlab\python.exe web_ui.py
         或双击 run_web_ui.bat
══════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import os, sys, time, threading, traceback
from datetime import datetime
from pathlib import Path

import cv2, numpy as np, gradio as gr

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["YOLO_VERBOSE"] = "False"

# ═════════════════════════════════════════════════════════════
#  启动环境自检
# ═════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).parent.resolve()
MMSEG_DIR = BASE_DIR / "mmsegmentation-main"
YOLO_DIR  = BASE_DIR / "yolov11_custom_segmentation"
D2_DIR    = YOLO_DIR                               # Detectron2 也在同一目录下
OUTPUT_DIR = BASE_DIR / "outputs"

# 将 mmseg 和 detectron2 加入 sys.path
sys.path.insert(0, str(MMSEG_DIR))
sys.path.insert(0, str(YOLO_DIR))

def _startup_check():
    """启动前校验 Python 环境和模型文件"""
    # 1. 检查是否用的系统 Python（而非 openmmlab）
    exe = Path(sys.executable)
    if "openmmlab" not in str(exe).lower():
        print("[⚠ 警告] 当前 Python 不是 openmmlab 环境!")
        print(f"  当前: {exe}")
        print(f"  请用: E:\\anaconda3\\envs\\openmmlab\\python.exe")
        print()

    # 2. 检查 torch+cuda
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        print(f"[环境] Python {sys.version.split()[0]}  |  torch {torch.__version__}  |  CUDA: {'可用' if cuda_ok else '不可用(使用CPU)'}")
    except Exception:
        print("[⚠ 警告] torch 导入失败")

    # 3. 检查关键库
    for lib_name, pkg_name in [("mmcv", "mmcv"), ("mmseg", "mmsegmentation"),
                                 ("ultralytics", "ultralytics"), ("detectron2", "detectron2")]:
        try:
            __import__(lib_name)
        except ImportError:
            print(f"[❌ 环境缺失] 未找到 {pkg_name}，请: pip install {pkg_name}")

    # 4. 校验所有模型文件
    print("[文件校验]")
    checks = [
        ("DeepLabV3+",  MMSEG_DIR / "Zihao-Configs/ZihaoDataset_DeepLabV3plus_20230818.py"),
        ("DeepLabV3+",  MMSEG_DIR / "work_dirs/ZihaoDataset-DeepLabV3plus/best_mIoU_iter_3400.pth"),
        ("PSPNet",      MMSEG_DIR / "Zihao-Configs/ZihaoDataset_PSPNet_20230818.py"),
        ("PSPNet",      MMSEG_DIR / "work_dirs/ZihaoDataset-PSPNet/best_mIoU_iter_3800.pth"),
        ("SegFormer",   MMSEG_DIR / "Zihao-Configs/ZihaoDataset_Segformer_20230818.py"),
        ("SegFormer",   MMSEG_DIR / "work_dirs/ZihaoDataset-Segformer/best_mIoU_iter_1000.pth"),
        ("YOLOv11",     YOLO_DIR  / "runs/segment/train6/weights/best.pt"),
        ("Mask R-CNN",  YOLO_DIR  / "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"),
        ("Mask R-CNN",  YOLO_DIR  / "detectron_ckps/model_final.pth"),
    ]
    missing = []
    for name, path in checks:
        ok = path.exists()
        if not ok:
            print(f"  ❌ {name}: {path}")
            missing.append(str(path))
    if not missing:
        print(f"  ✅ 全部 {len(checks)} 个文件校验通过")
    else:
        print(f"  ⚠️  缺失 {len(missing)} 个文件，相关模型将无法加载")
    print()

_startup_check()


# ═════════════════════════════════════════════════════════════
#  模型配置
# ═════════════════════════════════════════════════════════════

CLASSES_NAMES = ["background", "person", "roadheader", "robot", "shearer"]
PALETTE = [
    [127, 127, 127],  # background 灰
    [255, 0, 0],      # person     红
    [0, 200, 0],      # roadheader 绿
    [0, 0, 255],      # robot      蓝
    [255, 215, 0],    # shearer    金
]

MODEL_CONFIGS = {
    "DeepLabV3+": {
        "type": "mmseg",
        "config": "Zihao-Configs/ZihaoDataset_DeepLabV3plus_20230818.py",
        "checkpoint": "work_dirs/ZihaoDataset-DeepLabV3plus/best_mIoU_iter_3400.pth",
        "desc": "🧩 语义分割 · 空洞卷积 · 边缘精细",
    },
    "PSPNet": {
        "type": "mmseg",
        "config": "Zihao-Configs/ZihaoDataset_PSPNet_20230818.py",
        "checkpoint": "work_dirs/ZihaoDataset-PSPNet/best_mIoU_iter_3800.pth",
        "desc": "🏗️ 语义分割 · 金字塔池化 · 场景理解",
    },
    "SegFormer": {
        "type": "mmseg",
        "config": "Zihao-Configs/ZihaoDataset_Segformer_20230818.py",
        "checkpoint": "work_dirs/ZihaoDataset-Segformer/best_mIoU_iter_1000.pth",
        "desc": "🤖 语义分割 · Transformer · SOTA精度",
    },
    "YOLOv11": {
        "type": "yolo",
        "checkpoint": "runs/segment/train6/weights/best.pt",
        "desc": "⚡ 实例分割 · 实时推理 · 速度优先",
    },
    "Mask R-CNN": {
        "type": "detectron2",
        "config": "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        "checkpoint": "detectron_ckps/model_final.pth",
        "desc": "🎯 实例分割 · 高精度 · 掩码精细",
    },
}


# ═════════════════════════════════════════════════════════════
#  全局状态
# ═════════════════════════════════════════════════════════════

models = {}
model_lock = threading.Lock()
log_history: list[str] = []


# ═════════════════════════════════════════════════════════════
#  日志
# ═════════════════════════════════════════════════════════════

def _log_append(msg: str):
    global log_history
    ts = datetime.now().strftime("%H:%M:%S")
    log_history.append(f"[{ts}] {msg}")
    if len(log_history) > 200:
        log_history = log_history[-200:]

def add_log(log_box, msg: str) -> str:
    _log_append(msg)
    return "\n".join(log_history[-50:])


# ═════════════════════════════════════════════════════════════
#  模型加载 (按需加载 + 缓存)
# ═════════════════════════════════════════════════════════════

def _load_mmseg_model(cfg_rel, ckpt_rel):
    from mmseg.apis import init_model

    cfg_path = MMSEG_DIR / cfg_rel
    ckpt_path = MMSEG_DIR / ckpt_rel
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"权重文件不存在: {ckpt_path}")

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model = init_model(str(cfg_path), str(ckpt_path), device=device)
    model.CLASSES = CLASSES_NAMES
    model.PALETTE = PALETTE
    return model


def _load_yolo_model():
    from ultralytics import YOLO
    ckpt = str(YOLO_DIR / MODEL_CONFIGS["YOLOv11"]["checkpoint"])
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"YOLO 权重不存在: {ckpt}")
    return YOLO(ckpt, verbose=False)


def _load_mask_rcnn_model():
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog

    cfg_path = YOLO_DIR / MODEL_CONFIGS["Mask R-CNN"]["config"]
    ckpt_path = YOLO_DIR / MODEL_CONFIGS["Mask R-CNN"]["checkpoint"]
    if not cfg_path.exists():
        raise FileNotFoundError(f"Detectron2 配置不存在: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Detectron2 权重不存在: {ckpt_path}")

    cfg = get_cfg()
    cfg.merge_from_file(str(cfg_path))
    cfg.MODEL.WEIGHTS = str(ckpt_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    MetadataCatalog.get("custom_dataset").thing_classes = ["person", "roadheader", "robot", "shearer"]
    return DefaultPredictor(cfg)


def get_model(model_key: str):
    """获取模型（线程安全，首次调用时加载并缓存）"""
    with model_lock:
        if model_key in models:
            return models[model_key]

        cfg = MODEL_CONFIGS.get(model_key)
        if cfg is None:
            raise ValueError(f"未知模型: {model_key}")

        _log_append(f"⏳ 正在加载 {model_key} 模型... (可能需要 10-30 秒)")
        try:
            t = cfg["type"]
            if t == "mmseg":
                models[model_key] = _load_mmseg_model(cfg["config"], cfg["checkpoint"])
            elif t == "yolo":
                models[model_key] = _load_yolo_model()
            elif t == "detectron2":
                models[model_key] = _load_mask_rcnn_model()
            else:
                raise ValueError(f"未知模型类型: {t}")
            _log_append(f"✅ {model_key} 模型加载成功")
        except Exception as e:
            models[model_key] = None
            _log_append(f"❌ {model_key} 模型加载失败: {e}")
            raise
        return models[model_key]


# ═════════════════════════════════════════════════════════════
#  MMSeg 语义分割推理 + 可视化
# ═════════════════════════════════════════════════════════════

def _mmseg_infer(model, img_rgb):
    from mmseg.apis import inference_model
    return inference_model(model, img_rgb)


def _visualize_mmseg(model, img_rgb, result):
    """
    MMSeg 可视化 — 半透明叠加 + 类别标签。
    输入输出均为 RGB。
    """
    seg = result.pred_sem_seg.data[0].cpu().numpy()
    palette = model.PALETTE
    classes = model.CLASSES

    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color

    overlay = (img_rgb.astype(np.float32) * 0.5 + color_seg.astype(np.float32) * 0.5)
    overlay = overlay.astype(np.uint8)

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
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
        (tw, th), _ = cv2.getTextSize(classes[label], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(overlay, (cx - tw//2 - 4, cy - th - 4),
                      (cx + tw//2 + 4, cy), (0, 0, 0), -1)
        cv2.putText(overlay, classes[label], (cx - tw//2, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return overlay


def _mmseg_log(model, result, log):
    seg = result.pred_sem_seg.data[0].cpu().numpy()
    for label in np.unique(seg):
        if label >= len(model.CLASSES) or label == 0:
            continue
        pct = (seg == label).sum() / seg.size * 100
        log = add_log(log, f"  🟢 {model.CLASSES[label]}: {pct:.1f}%")
    return log


# ═════════════════════════════════════════════════════════════
#  YOLOv11 实例分割推理 + 可视化
# ═════════════════════════════════════════════════════════════

def _yolo_infer_and_viz(model, img_rgb, line_width=2):
    """返回 (RGB 结果图, ultralytics Results)"""
    results = model(img_rgb, verbose=False)[0]
    plotted = results.plot(line_width=line_width)  # 跟随输入格式
    # 启发式 BGR→RGB 修正
    if plotted.shape == img_rgb.shape:
        diff_r = np.abs(plotted[:,:,0].astype(float) - img_rgb[:,:,0].astype(float)).mean()
        diff_b = np.abs(plotted[:,:,2].astype(float) - img_rgb[:,:,2].astype(float)).mean()
        if diff_b > diff_r * 3:
            plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
    return plotted, results


def _yolo_log(results, log):
    if results.boxes is not None and len(results.boxes) > 0:
        for cls_id, conf in zip(results.boxes.cls, results.boxes.conf):
            cls_id = int(cls_id.item())
            label = results.names.get(cls_id, f"class_{cls_id}")
            log = add_log(log, f"  🔵 {label}: 置信度 {conf.item():.2%}")
    else:
        log = add_log(log, "  ℹ️  未检测到目标")
    return log


# ═════════════════════════════════════════════════════════════
#  Mask R-CNN (Detectron2) 实例分割推理 + 可视化
# ═════════════════════════════════════════════════════════════

def _mask_rcnn_infer_and_viz(model, img_rgb):
    """返回 (RGB 结果图, Detectron2 outputs)"""
    from detectron2.utils.visualizer import Visualizer, ColorMode
    from detectron2.data import MetadataCatalog

    v = Visualizer(img_rgb,
                   metadata=MetadataCatalog.get("custom_dataset"),
                   scale=0.8,
                   instance_mode=ColorMode.SEGMENTATION)
    outputs = model(img_rgb)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return v.get_image(), outputs


def _detectron2_log(outputs, log):
    if len(outputs["instances"]) > 0:
        classes = ["person", "roadheader", "robot", "shearer"]
        for i in range(len(outputs["instances"])):
            score = outputs["instances"].scores[i].item()
            cls_id = outputs["instances"].pred_classes[i].item()
            log = add_log(log, f"  🔴 {classes[cls_id]}: 置信度 {score:.2%}")
    else:
        log = add_log(log, "  ℹ️  未检测到目标")
    return log


# ═════════════════════════════════════════════════════════════
#  主处理函数
# ═════════════════════════════════════════════════════════════

def process_image(model_key, input_img):
    """
    Gradio gr.Image(type="numpy") 提交时触发。
    input_img 是 RGB 格式 numpy 数组。
    """
    if input_img is None:
        return None, None, "⚠️ 请先上传一张图片"

    log = ""
    try:
        model = get_model(model_key)
        if model is None:
            return input_img, None, f"❌ {model_key} 加载失败"

        h, w = input_img.shape[:2]
        log = add_log(None, f"📷 开始处理图片 — 模型: {model_key} — 尺寸: {w}x{h}")

        img_rgb = input_img  # Gradio 直接给 RGB！
        mtype = MODEL_CONFIGS[model_key]["type"]

        if mtype == "mmseg":
            result = _mmseg_infer(model, img_rgb)
            output = _visualize_mmseg(model, img_rgb, result)
            log = _mmseg_log(model, result, log)
            log = add_log(log, f"✅ {model_key} 语义分割完成")
        elif mtype == "yolo":
            output, results = _yolo_infer_and_viz(model, img_rgb)
            log = _yolo_log(results, log)
            log = add_log(log, f"✅ YOLOv11 实例分割完成")
        elif mtype == "detectron2":
            output, outputs = _mask_rcnn_infer_and_viz(model, img_rgb)
            log = _detectron2_log(outputs, log)
            log = add_log(log, f"✅ Mask R-CNN 实例分割完成")
        else:
            return input_img, None, f"❌ 不支持的模型类型: {mtype}"

        return img_rgb, output, log

    except Exception as e:
        log = add_log(log or None, f"❌ 处理出错: {e}")
        log = add_log(log, traceback.format_exc())
        return input_img, None, log


def process_video(model_key, video_path, progress=gr.Progress()):
    """逐帧处理视频并生成新的 mp4"""
    if video_path is None:
        return None, "⚠️ 请先上传一个视频文件"

    log = ""
    try:
        model = get_model(model_key)
        if model is None:
            return None, f"❌ {model_key} 加载失败"

        log = add_log(None, f"🎬 开始处理视频 — 模型: {model_key}")

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if w == 0 or h == 0:
            cap.release()
            return None, "❌ 视频为空或无法读取"

        max_frames = min(total, 3000)
        log = add_log(log, f"📊 视频: {total}帧(处理上限{max_frames}), {fps}fps, {w}x{h}")

        out_name = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        out_path = str(OUTPUT_DIR / out_name)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        mtype = MODEL_CONFIGS[model_key]["type"]

        idx = 0
        while idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if mtype == "mmseg":
                r = _mmseg_infer(model, frame_rgb)
                out_rgb = _visualize_mmseg(model, frame_rgb, r)
            elif mtype == "yolo":
                out_rgb, _ = _yolo_infer_and_viz(model, frame_rgb, line_width=1)
            elif mtype == "detectron2":
                out_rgb, _ = _mask_rcnn_infer_and_viz(model, frame_rgb)
            else:
                cap.release(); writer.release()
                return None, f"❌ 不支持的模型类型: {mtype}"

            writer.write(cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))
            idx += 1
            progress(idx / max_frames, desc=f"处理中 {idx}/{max_frames} 帧")

        cap.release()
        writer.release()
        log = add_log(log, f"✅ 视频处理完成，共 {idx} 帧")
        log = add_log(log, f"📁 保存到: {out_path}")
        return out_path, log

    except Exception as e:
        log = add_log(log or None, f"❌ 视频处理出错: {e}")
        log = add_log(log, traceback.format_exc())
        return None, log


def process_webcam(model_key, img):
    """摄像头流式回调 (Gradio streaming) — 也是 RGB"""
    if img is None:
        return None, "⏳ 等待摄像头画面..."
    try:
        model = get_model(model_key)
        if model is None:
            return img, f"❌ {model_key} 加载失败"
        img_rgb = img
        mtype = MODEL_CONFIGS[model_key]["type"]

        if mtype == "mmseg":
            out = _visualize_mmseg(model, img_rgb, _mmseg_infer(model, img_rgb))
        elif mtype == "yolo":
            out, _ = _yolo_infer_and_viz(model, img_rgb)
        elif mtype == "detectron2":
            out, _ = _mask_rcnn_infer_and_viz(model, img_rgb)
        else:
            return img, f"❌ 不支持的模型类型: {mtype}"

        return out, "🟢 实时分割运行中..."

    except Exception as e:
        return img, f"❌ 处理出错: {e}"


# ═════════════════════════════════════════════════════════════
#  保存结果
# ═════════════════════════════════════════════════════════════

def save_result(img, model_key, fmt="png"):
    if img is None:
        return "⚠️ 没有可保存的结果，请先进行分割"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"{model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{fmt}"
    path = str(OUTPUT_DIR / fname)

    if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3:
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    elif isinstance(img, np.ndarray):
        cv2.imwrite(path, img)
    else:
        return "⚠️ 结果不是有效的图片格式"
    _log_append(f"💾 结果已保存: {fname}")
    return f"✅ 已保存到: outputs/{fname}"


# ═════════════════════════════════════════════════════════════
#  UI 样式
# ═════════════════════════════════════════════════════════════

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --primary: #6366f1;
  --primary-hover: #4f46e5;
  --bg: #0f172a;
  --bg-card: #1e293b;
  --bg-card-hover: #334155;
  --bg-input: #0f172a;
  --text: #f1f5f9;
  --text-dim: #94a3b8;
  --border: #334155;
  --radius: 12px;
  --radius-sm: 8px;
  --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.3);
}

body { background: var(--bg) !important; color: var(--text); font-family: 'Inter', system-ui, sans-serif; }

.gradio-container { max-width: 1440px !important; margin: 0 auto !important; background: transparent !important; }

.tabs { border: none !important; }
.tab-nav {
  background: var(--bg-card) !important;
  border-radius: var(--radius) var(--radius) 0 0 !important;
  padding: 6px 8px !important; gap: 4px;
  border-bottom: 1px solid var(--border) !important;
}
.tab-nav button {
  border-radius: var(--radius-sm) !important; padding: 10px 24px !important;
  font-weight: 600 !important; font-size: 14px !important;
  color: var(--text-dim) !important; transition: all 0.2s !important;
  border: none !important; background: transparent !important;
}
.tab-nav button:hover { color: var(--text) !important; background: var(--bg-card-hover) !important; }
.tab-nav button.selected {
  background: var(--primary) !important; color: white !important;
  box-shadow: 0 2px 8px rgb(99 102 241 / 0.4) !important;
}

.gr-box, .gr-form, .gr-group, .gr-panel {
  border-radius: var(--radius) !important;
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
}

.gr-input, .gr-select, input, select, textarea {
  background: var(--bg-input) !important; border: 1px solid var(--border) !important;
  color: var(--text) !important; border-radius: var(--radius-sm) !important;
}
.gr-input:focus, select:focus, textarea:focus {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 3px rgb(99 102 241 / 0.2) !important; outline: none !important;
}

button.gr-button, .gr-button {
  border-radius: var(--radius-sm) !important; font-weight: 600 !important;
  padding: 10px 24px !important; transition: all 0.2s !important; font-size: 14px !important;
}
button.gr-button-primary, .gr-button-primary { background: var(--primary) !important; border: none !important; color: white !important; }
button.gr-button-primary:hover { background: var(--primary-hover) !important; transform: translateY(-1px); box-shadow: var(--shadow) !important; }
button.gr-button-secondary { border: 1px solid var(--border) !important; }

label, .gr-form label, .gr-box label { color: var(--text) !important; font-weight: 500 !important; font-size: 13px !important; }

footer, header { display: none !important; }

.gr-markdown { color: var(--text) !important; }
.gr-markdown h1, .gr-markdown h2, .gr-markdown h3 { color: var(--text) !important; }
.gr-markdown code { background: var(--bg) !important; padding: 2px 6px !important; border-radius: 4px !important; }

.gr-textbox textarea {
  font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
  font-size: 12px !important; line-height: 1.6 !important; color: var(--text-dim) !important;
}

.gr-image { border-radius: var(--radius) !important; overflow: hidden !important; }

progress { accent-color: var(--primary); }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }

@keyframes fadeIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
.gr-box { animation: fadeIn 0.3s ease; }
"""

HTML_HEADER = """
<div style="
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 40%, #ec4899 100%);
  padding: 28px 36px; border-radius: 16px; margin-bottom: 24px;
  color: white; box-shadow: 0 12px 40px rgba(99,102,241,0.35);
  position: relative; overflow: hidden;
">
  <div style="position:absolute; top:-50%; right:-10%; width:300px; height:300px;
    background:rgba(255,255,255,0.06); border-radius:50%; pointer-events:none;"></div>
  <div style="position:relative; z-index:1;">
    <h1 style="margin:0; font-size:30px; font-weight:700; letter-spacing:-0.5px;">
      🎯 图像分割算法分析与应用
    </h1>
    <p style="margin:10px 0 0; opacity:0.9; font-size:15px; line-height:1.5;">
      统一管理 DeepLabV3+ · PSPNet · SegFormer · YOLOv11 · Mask R-CNN 五种分割模型
      <br/>支持 图片 / 视频 / 摄像头 实时分割 · 结果自动保存
    </p>
  </div>
</div>
"""

MODEL_DESCRIPTIONS = {k: v["desc"] for k, v in MODEL_CONFIGS.items()}


# ═════════════════════════════════════════════════════════════
#  构建 Gradio 界面
# ═════════════════════════════════════════════════════════════

def build_app():
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    )

    with gr.Blocks(css=CUSTOM_CSS, title="图像分割算法分析系统", theme=theme) as demo:
        gr.HTML(HTML_HEADER)

        with gr.Row(equal_height=False):
            # ── 左侧控制面板 ──
            with gr.Column(scale=1, min_width=300):
                with gr.Group(elem_classes=["gr-box"]):
                    gr.Markdown("### ⚙️ 模型配置")

                    model_choices = [
                        ("🧩 DeepLabV3+ (语义分割)", "DeepLabV3+"),
                        ("🏗️ PSPNet (语义分割)",   "PSPNet"),
                        ("🤖 SegFormer (语义分割)",  "SegFormer"),
                        ("⚡ YOLOv11 (实例分割)",    "YOLOv11"),
                        ("🎯 Mask R-CNN (实例分割)", "Mask R-CNN"),
                    ]
                    model_selector = gr.Dropdown(
                        choices=model_choices, value="DeepLabV3+",
                        label="选择分割模型", interactive=True,
                    )
                    model_desc = gr.Markdown(MODEL_DESCRIPTIONS["DeepLabV3+"])

                    def on_model_change(name):
                        return MODEL_DESCRIPTIONS.get(name, "")
                    model_selector.change(on_model_change, model_selector, model_desc)

                with gr.Group(elem_classes=["gr-box"]):
                    gr.Markdown("### 📋 使用说明")
                    gr.Markdown("""
                    1. 🎛️ 在上方选择分割模型 (支持 5 种)
                    2. 📷 切换到 **图片/视频/摄像头** 标签页
                    3. 🚀 上传数据 → 点击"开始分割"
                    4. 💾 保存分割结果到 `outputs/` 目录
                    """)

                with gr.Group(elem_classes=["gr-box"]):
                    gr.Markdown("### 📊 类别信息")
                    gr.Markdown("""
                    | 类别 | 颜色 | 说明 |
                    |------|------|------|
                    | 🟫 background | 灰色 | 背景区域 |
                    | 🟥 person | 红色 | 人员 |
                    | 🟩 roadheader | 绿色 | 掘进机 |
                    | 🟦 robot | 蓝色 | 机器人 |
                    | 🟨 shearer | 金色 | 采煤机 |
                    """)

                with gr.Group(elem_classes=["gr-box"]):
                    gr.Markdown("### 💾 保存选项")
                    save_btn = gr.Button("💾 保存当前结果", variant="primary", size="lg")
                    save_status = gr.Textbox(label="保存状态", interactive=False)

            # ── 右侧主内容区 ──
            with gr.Column(scale=3):
                with gr.Tabs(elem_classes=["tabs"]):
                    # Tab 1: 图片
                    with gr.TabItem("📷 图片分割", id="image"):
                        with gr.Row(equal_height=True):
                            with gr.Column():
                                gr.Markdown("##### 原始图片")
                                img_input = gr.Image(label="", type="numpy", height=420)
                            with gr.Column():
                                gr.Markdown("##### 分割结果")
                                img_output = gr.Image(label="", type="numpy", height=420)
                        with gr.Row():
                            img_process_btn = gr.Button("🚀 开始分割", variant="primary", size="lg", scale=3)
                            img_save_btn    = gr.Button("💾 保存", variant="secondary", scale=1)
                            img_clear_btn   = gr.Button("🗑️ 清空", variant="secondary", scale=1)

                    # Tab 2: 视频
                    with gr.TabItem("🎬 视频分割", id="video"):
                        with gr.Row(equal_height=True):
                            with gr.Column():
                                gr.Markdown("##### 原始视频")
                                vid_input = gr.Video(label="", height=340)
                            with gr.Column():
                                gr.Markdown("##### 分割结果")
                                vid_output = gr.Video(label="", height=340)
                        vid_process_btn = gr.Button("🚀 开始处理视频", variant="primary", size="lg")
                        gr.Markdown("⚠️ *大型视频自动限制 3000 帧*")

                    # Tab 3: 摄像头
                    with gr.TabItem("📹 摄像头实时分割", id="webcam"):
                        gr.Markdown("##### 开启摄像头后自动实时分割")
                        with gr.Row(equal_height=True):
                            with gr.Column():
                                webcam_input = gr.Image(
                                    sources=["webcam"], streaming=True,
                                    label="摄像头画面", height=400,
                                )
                            with gr.Column():
                                webcam_output = gr.Image(label="实时分割结果", height=400)

                # 日志面板
                with gr.Group(elem_classes=["gr-box"]):
                    gr.Markdown("### 📜 处理日志")
                    log_output = gr.Textbox(
                        label="", lines=10, max_lines=20, interactive=False,
                        value="[系统] 🎉 欢迎使用图像分割分析系统\n[系统] 📌 请选择模型并上传数据开始处理",
                    )

        # ── 事件绑定 ──

        def img_process(model_key, img):
            _, seg, log = process_image(model_key, img)
            return seg, log

        img_process_btn.click(img_process, [model_selector, img_input], [img_output, log_output])

        img_save_btn.click(lambda img, mk: save_result(img, mk),
                           [img_output, model_selector], [save_status])

        def img_clear():
            return None, None, ""
        img_clear_btn.click(img_clear, outputs=[img_input, img_output, log_output])

        save_btn.click(lambda img, mk: save_result(img, mk),
                       [img_output, model_selector], [save_status])

        vid_process_btn.click(process_video, [model_selector, vid_input],
                              [vid_output, log_output])

        # ⚠️ Gradio 4.44 不支持 time_limit 参数
        webcam_input.stream(process_webcam,
                            inputs=[model_selector, webcam_input],
                            outputs=[webcam_output, log_output])

    return demo


# ═════════════════════════════════════════════════════════════
#  启动
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  🎯 图像分割算法分析系统")
    print("  ===============================")
    print(f"  语义分割: DeepLabV3+ / PSPNet / SegFormer")
    print(f"  实例分割: YOLOv11 / Mask R-CNN")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("=" * 60)

    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True,
    )
