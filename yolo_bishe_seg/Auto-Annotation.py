from ultralytics.data.annotator import auto_annotate

auto_annotate(data="datasets/data/images", det_model="yolo11x.pt", sam_model="models/sam2.1_b.pt")