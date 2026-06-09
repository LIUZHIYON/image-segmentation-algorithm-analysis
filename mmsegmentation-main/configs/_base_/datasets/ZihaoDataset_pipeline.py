# 数据处理 pipeline
# 同济子豪兄 2023-6-28

# 数据集路径
dataset_type ='ZihaoDataset' # 数据集类名
data_root = 'data/mmseg_custom_dataset/' # 数据集路径（相对于mmsegmentation主目录）

# 输入模型的图像裁剪尺寸，一般是 128 的倍数，越小显存开销越少
crop_size = (512, 512)

# 训练预处理  原始的代码
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

# 训练预处理  原始的代码
# train_pipeline = [
#     # 1. 加载图像
#     dict(type='LoadImageFromFile'),  # 从文件中加载图像
#     # 2. 加载标注
#     dict(type='LoadAnnotations'),    # 加载与图像对应的标注（如分割图）
#     # 3. 随机缩放
#     dict(
#         type='RandomResize',         # 随机调整图像和标注的大小
#         scale=(2048, 1024),          # 目标尺寸范围（宽, 高）
#         ratio_range=(0.5, 2.0),     # 缩放比例范围（最小比例, 最大比例）
#         keep_ratio=True              # 是否保持宽高比
#     ),
#     # 4. 随机裁剪
#     dict(
#         type='RandomCrop',          # 随机裁剪图像和标注
#         crop_size=crop_size,         # 裁剪的目标尺寸（宽, 高）
#         cat_max_ratio=0.75           # 单个类别的最大面积占比，避免裁剪后某个类别占比过大
#     ),
#     # 5. 随机翻转
#     dict(
#         type='RandomFlip',           # 随机水平或垂直翻转图像和标注
#         prob=0.5                     # 翻转的概率（0.5 表示 50% 的概率）
#     ),
#     # 6. 光度畸变
#     dict(
#         type='PhotoMetricDistortion',  # 对图像进行光度畸变增强
#         # 包括亮度、对比度、饱和度、色调的随机变化
#         # 具体参数可以自定义，这里使用默认值
#     ),
#     # 7. 打包数据
#     dict(type='PackSegInputs')       # 将图像和标注打包为模型输入格式
# ]



# 测试预处理
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# TTA后处理
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]

# 训练 Dataloader
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=train_pipeline))

# 验证 Dataloader
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))

# 测试 Dataloader
test_dataloader = val_dataloader

# 验证 Evaluator
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])

# 测试 Evaluator
test_evaluator = val_evaluator