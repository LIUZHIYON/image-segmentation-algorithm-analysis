import os
import logging
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, DatasetMapper
from multiprocessing import freeze_support

# 配置日志输出
logging.basicConfig(level=logging.INFO)  # 设置日志级别为 INFO
logger = logging.getLogger(__name__)

def main():
    # 1. 注册数据集
    # 替换为你的数据集路径
    register_coco_instances("my_dataset_train", {}, "train.json", "train")
    register_coco_instances("my_dataset_val", {}, "test.json", "test")

    # 设置类别名称（替换为你的类别）
    classes = ["person", "roadheader", "robot", "shearer"]  # 你的类别
    MetadataCatalog.get("my_dataset_train").set(thing_classes=classes)
    MetadataCatalog.get("my_dataset_val").set(thing_classes=classes)

    # 2. 加载配置文件
    cfg = get_cfg()
    cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # 替换为你的配置文件路径

    # 设置模型类别数
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)  # 设置为你数据集的类别数

    # 3. 设置模型权重路径
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "instance_segmentation/model_final.pth")  # 替换为你的模型权重路径
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置检测阈值

    # 4. 创建预测器
    predictor = DefaultPredictor(cfg)

    # 5. 构建测试数据加载器（禁用多进程）
    mapper = DatasetMapper(cfg, is_train=False)
    val_loader = build_detection_test_loader(cfg, "my_dataset_val", mapper=mapper, num_workers=0)

    # 6. 进行检查
    logger.info("Checking the first batch of data...")
    for idx, batch in enumerate(val_loader):
        logger.info(f"Batch {idx}: {batch}")
        if idx >= 0:  # 只检查第一个批次
            break

    # 7. 进行评估
    logger.info("Starting evaluation...")
    evaluator = COCOEvaluator("my_dataset_val", cfg, False, output_dir="./output/")
    results = inference_on_dataset(predictor.model, val_loader, evaluator)
    logger.info("Evaluation completed! Results saved in ./output/")

    # 打印评估结果
    logger.info("Evaluation results:")
    for key, value in results.items():
        logger.info(f"{key}: {value}")

if __name__ == '__main__':
    freeze_support()  # Windows 系统需要调用 freeze_support()
    main()