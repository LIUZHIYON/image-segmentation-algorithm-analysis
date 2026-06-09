import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from ultralytics import YOLO

model = YOLO("runs/segment/train6/weights/best.pt")

model.predict(source= "4.jpg",show= True,save = True,
              conf = 0.5,line_width = 2,save_crop = True,save_txt = True,
              show_labels = True,show_conf = True)

# model.predict(source= "1.jpg",show= True,save = True,
#               conf = 0.7,line_width = 2,save_crop = True,save_txt = True,
#               show_labels = True,show_conf = True,classes = [0,1])

