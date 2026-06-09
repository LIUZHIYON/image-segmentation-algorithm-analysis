import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from detectron2.utils.logger import setup_logger

setup_logger()

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

import os
import pickle

from utils import *

#进行目标检测训练 练习
# 本地配置文件路径
#config_file_path = "configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"  # 替换为你的本地 YAML 路径
# 本地预训练权重路径
#checkpoint_path = "models/faster_rcnn_R_101_FPN_3x.pkl"
#目标检测  最终模型文件地址
#output_dir = "./output/object_detection"

#最终实现图像分割  mask_rcnn训练  修改文件地址
# 本地配置文件路径
config_file_path = "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"  # 替换为你的本地 YAML 路径
#本地预训练权重路径
checkpoint_path = "models/mask_rcnn_R_50_FPN_3x.pkl"
#修改最终模型文件地址
output_dir = "./output/instance_segmentation"

#最终实现图像分割  Cascade R-CNN 训练  修改文件地址 效果不太好，总的损失比mask_rnn高
# 本地配置文件路径
# config_file_path = "configs/quick_schedules/cascade_mask_rcnn_R_50_FPN_inference_acc_test.yaml"  # 替换为你的本地 YAML 路径
# # 本地预训练权重路径
# checkpoint_path = "models/cascade_mask_rcnn_R_50_FPN_3x.pkl"
# # 修改最终模型文件地址
# output_dir = "./output/instance_segmentation_cascade"

num_classes = 4

device = "cuda"

train_dataset_name = "LP_train"
train_images_path = "train"
train_json_annot_path = "train.json"

test_dataset_name = "LP_test"
test_images_path = "test"
test_json_annot_path = "test.json"

#这个也要修改
#cfg_save_path = "OD_cfg.pickle"

#实例分割
cfg_save_path = "IS_cfg.pickle"
#使用cascade实例分割
#cfg_save_path = "cascade_IS_cfg.pickle"



#################################

register_coco_instances(name=train_dataset_name, metadata={},
                        json_file=train_json_annot_path, image_root=train_images_path)

register_coco_instances(name=test_dataset_name, metadata={},
                        json_file=test_json_annot_path, image_root=test_images_path)

# plot_samples(dataset_name=train_dataset_name, n=2)

#################################

def main():
    cfg = get_train_cfg(config_file_path, checkpoint_path, train_dataset_name, test_dataset_name, num_classes, device, output_dir)

    with open(cfg_save_path, 'wb') as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    trainer.train()

if __name__ == '__main__':
    main()