from detectron2.data.datasets import register_coco_instances

# 注册数据集
register_coco_instances("my_dataset_train", {}, "train.json", "train")
register_coco_instances("my_dataset_val", {}, "test.json", "test")

import os
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

# 配置模型
cfg = get_cfg()
cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # 选择合适的配置文件

# 设置数据集和测试集
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)

# 配置数据加载器
cfg.DATALOADER.NUM_WORKERS = 0

# 使用本地训练好的权值文件
cfg.MODEL.WEIGHTS = "models/mask_rcnn_R_50_FPN_3x.pkl"  # 指定本地训练好的权值文件路径

# 配置训练相关的超参数
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # 类别数：1个目标类 + 背景

# 设置输出目录
cfg.OUTPUT_DIR = "./output"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# 创建训练器
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)

# 开始训练
trainer.train()
