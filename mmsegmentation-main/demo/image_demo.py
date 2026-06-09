# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmengine.model import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--with-labels',
        action='store_true',
        default=False,
        help='Whether to display the class labels.')
    parser.add_argument(
        '--title', default='result', help='The image identifier.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)
    # test a single image
    result = inference_model(model, args.img)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result,
        title=args.title,
        opacity=args.opacity,
        with_labels=args.with_labels,
        draw_gt=False,
        show=False if args.out_file is not None else True,
        out_file=args.out_file)


if __name__ == '__main__':
    main()

#下面是代码修改参数  不用命令
# Copyright (c) OpenMMLab. All rights reserved.
# from mmengine.model import revert_sync_batchnorm
# from mmseg.apis import inference_model, init_model, show_result_pyplot
#
#
# def main():
#     # 手动在代码中指定参数
#     img = "../2.jpg"  # 图像文件路径
#     config = "../pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py"  # 配置文件路径
#     checkpoint = "../pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth"  # 模型权重文件路径
#     out_file = None
#     #out_file = "path/to/your/output.png"  # 输出文件路径（可选，设置为 None 则不保存）
#     device = "cuda:0"  # 推理设备，可选 "cpu" 或 "cuda:0"
#     opacity = 0.5  # 分割图的透明度，范围在 (0, 1]
#     with_labels = True  # 是否显示类别标签
#     title = "result"  # 图像标题
#
#     # 根据配置文件和权重文件初始化模型
#     model = init_model(config, checkpoint, device=device)
#     if device == 'cpu':
#         model = revert_sync_batchnorm(model)
#
#     # 对单张图像进行推理
#     result = inference_model(model, img)
#
#     # 显示或保存结果
#     show_result_pyplot(
#         model,
#         img,
#         result,
#         title=title,
#         opacity=opacity,
#         with_labels=with_labels,
#         draw_gt=False,
#         show=False if out_file else True,  # 如果指定了 out_file，则不显示图像
#         out_file=out_file if out_file else None  # 如果 out_file 为 None，则不保存图像
#     )
#
#
# if __name__ == '__main__':
#     main()
