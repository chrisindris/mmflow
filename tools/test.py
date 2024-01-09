# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmcv.utils.logging import print_log

from mmflow import digit_version
from mmflow.apis import multi_gpu_test, single_gpu_test
from mmflow.core import online_evaluation
from mmflow.datasets import build_dataloader, build_dataset
from mmflow.datasets.utils.flow_io import write_flow, write_flow_kitti, visualize_flow
from mmflow.models import build_flow_estimator
from mmflow.utils import get_root_logger, setup_multi_processes


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Test (and eval)" " a flow estimator")
    )
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--out-dir", help="directory to save the flow file")
    parser.add_argument(
        "--sparse-flow",
        action="store_true",
        help="Whether The evaluation dataset is a sparse optical flow dataset",
    )
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )
    parser.add_argument(
        "--eval", type=str, nargs="+", help='evaluation metrics, e.g., "EPE"'
    )
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument(
        "--show-dir", help="directory where visual flow maps will be saved"
    )
    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="id of gpu to use " "(only applicable to non-distributed testing)",
    )
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu-collect is not specified",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    assert args.out_dir or args.eval or args.show_dir, (
        "Please specify at least one operation (save/eval/show the "
        'results / save the results) with the argument "--out-dir", "--eval"'
        ', "--show" or "--show-dir"'
    )

    if args.out_dir is not None:
        mmcv.mkdir_or_exist(args.out_dir)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set multi-process settings
    setup_multi_processes(cfg)

    # build the model and load checkpoint
    model = build_flow_estimator(cfg.model)
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if not distributed:
        warnings.warn(
            "SyncBN is only supported with DDP. To be compatible with DP, "
            "we convert SyncBN to BN. Please use dist_train.sh which can "
            "avoid this error."
        )
        model = revert_sync_batchnorm(model)
        if not torch.cuda.is_available():
            assert digit_version(mmcv.__version__) >= digit_version(
                "1.4.4"
            ), "Please use MMCV >= 1.4.4 for CPU training!"
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)

    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )

    rank, _ = get_dist_info()

    videos = sorted(os.listdir(cfg.data_root))[326:356]

    cfg.data.test.datasets = [
        dict(
            type=cfg.dataset_type,
            pipeline=cfg.test_pipeline,
            data_root=os.path.join(cfg.data_root, video_subdir),
            test_mode=True,
            out_dir=os.path.join(args.out_dir, "out_flow" + video_subdir[5:]),
            show_dir=os.path.join(args.show_dir, "out_flowmap" + video_subdir[5:]),
        )
        for video_subdir in videos
    ]

    # cfg.data.test.data_root = os.path.join(data_root, video_subdir)
    # args.out_dir = os.path.join(out_dir, "out_flow" + video_subdir[5:])
    # args.show_dir = os.path.join(show_dir, "out_flowmap" + video_subdir[5:])

    # The overall dataloader settings
    loader_cfg = {
        k: v
        for k, v in cfg.data.items()
        if k
        not in [
            "train",
            "val",
            "test",
            "train_dataloader",
            "val_dataloader",
            "test_dataloader",
        ]
    }
    # The specific training dataloader settings
    test_loader_cfg = {**loader_cfg, **cfg.data.get("test_dataloader", {})}

    # build the dataloader
    separate_eval = cfg.data.test.get("separate_eval", False)
    if separate_eval:
        # multi-datasets will be built as themselves.
        # dataset = [build_dataset(dataset) for dataset in cfg.data.test.datasets]
        dataset = []
        for _dataset in cfg.data.test.datasets:
            built_dataset = build_dataset(_dataset)

            if os.path.exists(built_dataset.data_root) and (
                os.listdir(built_dataset.data_root) == len(built_dataset)
            ):
                continue
            else:
                dataset.append(built_dataset)
                os.makedirs(built_dataset.out_dir, exist_ok=True)
                os.makedirs(built_dataset.show_dir, exist_ok=True)

    else:
        # multi-datasets will be concatenated as one dataset.
        dataset = [build_dataset(cfg.data.test)]

    data_loader = [
        build_dataloader(
            _dataset,
            **test_loader_cfg,
            dist=distributed,
        )
        for _dataset in dataset
    ]

    # perform the experiment
    for i, i_data_loader in enumerate(data_loader):
        print("i_data_loader.dataset.data_root = ", i_data_loader.dataset.data_root)
        if args.out_dir:
            if not distributed:
                outputs = single_gpu_test(
                    model,
                    i_data_loader,
                    out_dir=args.out_dir,
                    show_dir=args.show_dir,
                )
            else:
                outputs = multi_gpu_test(
                    model, i_data_loader, args.tmpdir, args.gpu_collect
                )
                if rank == 0:
                    print(f"\nwriting results to {i_data_loader.dataset.out_dir}")
                    for i, output in enumerate(outputs):
                        if args.sparse_flow:
                            write_flow_kitti(output, f"flow_{i}.png")
                        else:
                            write_flow(
                                output,
                                os.path.join(
                                    i_data_loader.dataset.out_dir, f"flow_{i}.flo"
                                ),
                            )
                            visualize_flow(
                                output,
                                os.path.join(
                                    i_data_loader.dataset.show_dir, f"flowmap_{i}.png"
                                ),
                            )

        if args.eval:
            dataset_name = dataset[i].__class__.__name__
            if hasattr(dataset[i], "pass_style"):
                dataset_name += f" {dataset[i].pass_style}"
            print_log(
                f"In {dataset_name} "
                f"{online_evaluation(model, i_data_loader, metric=args.eval)}"
                "\n",
                logger=get_root_logger(),
            )


if __name__ == "__main__":
    main()
