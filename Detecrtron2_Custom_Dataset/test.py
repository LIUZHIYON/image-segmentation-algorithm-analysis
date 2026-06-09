import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from detectron2.engine import DefaultPredictor

import os
import pickle

from utils import *

#目标检测
#cfg_save_path ="OD_cfg.pickle"

#实例分割
cfg_save_path ="IS_cfg.pickle"

#实例分割
#cfg_save_path ="IS_cfg.pickle"

with open(cfg_save_path,'rb') as f:
    cfg = pickle.load(f)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR,"model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)

image_path = "1.png"
videoPath = "street.mp4"
on_image(image_path,predictor)

#实时分割
#on_video(videoPath,predictor)