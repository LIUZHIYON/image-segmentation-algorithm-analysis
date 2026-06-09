from ultralytics import SAM

# 加载模型
model = SAM("models/sam2_b.pt")

# 显示模型信息
model.info()

# 使用边界框提示进行推理
results = model("28.jpg", bboxes=[100, 100, 200, 200])

# 使用单点提示进行推理
#results = model("path/to/image.jpg", points=[900, 370], labels=[1])

# 使用多点提示进行推理
#results = model("path/to/image.jpg", points=[[400, 370], [900, 370]], labels=[1, 1])

# 使用多点提示（每个对象一组点）进行推理
#results = model("path/to/image.jpg", points=[[[400, 370], [900, 370]]], labels=[[1, 1]])

# 使用负点提示进行推理
#results = model("path/to/image.jpg", points=[[[400, 370], [900, 370]]], labels=[[1, 0]])

# 显示结果
#results.show()

# 保存结果
#results.save("output.jpg")