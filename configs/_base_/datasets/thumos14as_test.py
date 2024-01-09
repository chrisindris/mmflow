dataset_type = "Thumos14as"
# data_root = "/data/i5O/UCF101-THUMOS/THUMOS14/actionformer_subset_i3d_frames_all/"
data_root = "/data/i5O/THUMOS14/actionformer_subset_i3d_frames_all/"

img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], to_rgb=False)

thumos14as_test_pipeline = [
    dict(type="LoadImageFromZip", data_root=data_root),
    # dict(type="LoadAnnotations", sparse=True),
    dict(type="InputPad", exponent=3),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="TestFormatBundle"),
    dict(
        type="Collect",
        keys=["imgs"],
        meta_keys=[
            # "flow_gt",
            # "valid",
            "filename1",
            "filename2",
            "ori_filename1",
            "ori_filename2",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "scale_factor",
            "pad_shape",
            "pad",
        ],
    ),
]

test_pipeline = thumos14as_test_pipeline

thumos14as_val_test = dict(
    type=dataset_type,
    data_root=data_root,
    pipeline=thumos14as_test_pipeline,
    test_mode=True,
)

data = dict(
    train_dataloader=dict(
        samples_per_gpu=0, workers_per_gpu=0, drop_last=True, persistent_workers=True
    ),
    test_dataloader=dict(samples_per_gpu=16, workers_per_gpu=4, shuffle=False),
    test=dict(
        type="ConcatDataset",
        datasets=None,
        separate_eval=True,
    ),
)
