import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def plot_pr_curves_from_results(results_json, annotations_json, output_dir="output/pr_curves"):
    """
    从coco_instances_results.json绘制PR曲线

    参数:
        results_json: 预测结果JSON文件路径
        annotations_json: COCO标注文件路径
        output_dir: 输出目录路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取预测结果JSON文件
    with open(results_json, "r") as f:
        data = json.load(f)
        print(f"已加载预测结果，共{len(data)}个检测实例")

    # 加载COCO标注
    coco_gt = COCO(annotations_json)
    coco_dt = coco_gt.loadRes(results_json)

    # 初始化评估对象
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()

    # 获取PR曲线数据
    precision = coco_eval.eval['precision']  # 形状: [iou, recall, cls, area, max_dets]
    recall = np.linspace(0, 1, 101)  # COCO标准的101点recall阈值

    # 选择代表性的IoU阈值
    iou_thresholds = [0.5, 0.75, 0.95]
    colors = ['blue', 'green', 'red']
    line_styles = ['-', '--', ':']

    # 1. 绘制边界框PR曲线
    plt.figure(figsize=(10, 6))
    for i, iou in enumerate(iou_thresholds):
        iou_idx = int((iou - 0.5) / 0.05)  # 计算对应的索引
        # 获取precision数据 (所有类别平均, area=all, max_dets=100)
        precisions = precision[iou_idx, :, :, 0, -1].mean(axis=1)
        plt.plot(recall, precisions,
                 color=colors[i], linestyle=line_styles[i],
                 linewidth=2, label=f'IoU={iou}')

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Bounding Box PR Curves (Average over all categories)', fontsize=14)
    plt.legend(fontsize=10, loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bbox_pr_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 检查是否有分割结果并绘制Mask PR曲线
    if 'segmentation' in data[0]:
        coco_eval_mask = COCOeval(coco_gt, coco_dt, 'segm')
        coco_eval_mask.evaluate()
        coco_eval_mask.accumulate()

        precision_mask = coco_eval_mask.eval['precision']

        plt.figure(figsize=(10, 6))
        for i, iou in enumerate(iou_thresholds):
            iou_idx = int((iou - 0.5) / 0.05)
            precisions = precision_mask[iou_idx, :, :, 0, -1].mean(axis=1)
            plt.plot(recall, precisions,
                     color=colors[i], linestyle=line_styles[i],
                     linewidth=2, label=f'IoU={iou}')

        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Instance Segmentation PR Curves (Average over all categories)', fontsize=14)
        plt.legend(fontsize=10, loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "mask_pr_curves.png"), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"PR曲线图已保存到: {output_dir}")


# 使用示例
plot_pr_curves_from_results(
    results_json="output/coco_instances_results.json",
    annotations_json="test.json",  # 替换为你的标注文件路径
    output_dir="output/pr_curves"
)