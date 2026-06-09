import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from pycocotools.coco import COCO
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 设置全局字体
rcParams.update({
    'font.size': 15,
})

# 创建输出目录
output_dir = "output/visualize_json"
os.makedirs(output_dir, exist_ok=True)

# 加载数据
with open("output/coco_instances_results.json", "r") as f:
    data = json.load(f)
df = pd.DataFrame(data)

# 特征工程
df["bbox_width"] = df["bbox"].apply(lambda x: x[2])
df["bbox_height"] = df["bbox"].apply(lambda x: x[3])
df["bbox_area"] = df["bbox_width"] * df["bbox_height"]

# 8. 优化后的混淆矩阵
coco = COCO("test.json")

# 准备数据
true_labels = []
pred_labels = []
for pred in data:
    anns = coco.loadAnns(coco.getAnnIds(imgIds=pred["image_id"]))
    true_labels.extend([ann["category_id"] for ann in anns])
    pred_labels.extend([pred["category_id"]] * len(anns))

# 计算混淆矩阵
cm = confusion_matrix(true_labels, pred_labels)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
category_names = [cat["name"] for cat in coco.loadCats(coco.getCatIds())]

# 创建图形
fig, ax = plt.subplots(figsize=(18, 16))

# 使用imshow保证精确控制
im = ax.imshow(cm_normalized, cmap="Blues", vmin=0, vmax=1)

# 创建与矩阵高度完全一致的色标
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=12)
cbar.set_label('Normalized Value', fontsize=20)

# 设置刻度（修改部分：x轴标签垂直显示）
ax.set_xticks(np.arange(len(category_names)))
ax.set_yticks(np.arange(len(category_names)))
ax.set_xticklabels(category_names, rotation=90, ha="center", va="top")  # 垂直显示
ax.set_yticklabels(category_names, ha="right", va="center")

# 标签和标题
ax.set_title("Normalized Confusion Matrix", pad=20, fontsize=20)
ax.set_xlabel("Predicted Label", labelpad=15, fontsize=20)
ax.set_ylabel("True Label", labelpad=15, fontsize=20)

# 调整布局并保存
plt.tight_layout(pad=3.0)
plt.savefig(os.path.join(output_dir, "8_confusion_matrix_vertical.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"所有可视化结果已保存至: {output_dir}")