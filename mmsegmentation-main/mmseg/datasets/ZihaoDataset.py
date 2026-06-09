# 同济子豪兄 2023-6-25
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class ZihaoDataset(BaseSegDataset):
    # 类别和对应的 RGB配色
    METAINFO = {
        'classes': ['background', 'person', 'roadheader', 'robot', 'shearer'],
        'palette': [
            [127, 127, 127],  # 背景色：中等亮度的灰色
            [255, 0, 0],  # person：红色，更鲜艳
            [0, 200, 0],  # roadheader：浅绿色
            [0, 0, 255],  # robot：蓝色，与绿色有明显区别
            [255, 215, 0]  # shearer：金黄色，保持明亮但不刺眼
        ]
    }


    
    # 指定图像扩展名、标注扩展名
    def __init__(self,
                 seg_map_suffix='.png',   # 标注mask图像的格式
                 reduce_zero_label=False, # 类别ID为0的类别是否需要除去
                 **kwargs) -> None:
        super().__init__(
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)