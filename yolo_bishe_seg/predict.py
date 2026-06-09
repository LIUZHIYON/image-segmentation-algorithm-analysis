from ultralytics import YOLO

# model = YOLO("runs/segment/train3/weights/best.pt")

model = YOLO("runs/segment/train9/weights/best.pt")

model.predict(source= "4.png",show= True,save = True,
              conf = 0.5,line_width = 2,save_crop = True,save_txt = True,
              show_labels = True,show_conf = True)