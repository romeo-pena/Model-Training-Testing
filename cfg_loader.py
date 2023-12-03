from mmcv import Config
from mmseg.apis import set_random_seed
from mmseg.utils import get_device


def getConfig_base_train(file, data_dir, load_pth, working, iterations=200, mode='Train'):
    cfg = Config.fromfile(file)
    img_dir = 'images'
    ann_dir = 'labels'

    val_img = "cimages"
    val_ann = "clabels"
    val_splits = 'csplits/val.txt'

    # Since we use only one GPU, BN is used instead of SyncBN
    cfg.norm_cfg = dict(type='BN', requires_grad=True)
    cfg.model.backbone.norm_cfg = cfg.norm_cfg
    cfg.model.decode_head.norm_cfg = cfg.norm_cfg
    cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg

    # modify num classes of the model in decode/auxiliary head
    cfg.model.decode_head.num_classes = 6
    cfg.model.auxiliary_head.num_classes = 6

    # Modify dataset type and path
    cfg.dataset_type = 'CustomExperimentDataset'
    cfg.data_root = data_dir

    cfg.data.samples_per_gpu = 10
    cfg.data.workers_per_gpu = 2

    cfg.img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    cfg.crop_size = (256, 256)
    cfg.train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(320, 240), ratio_range=(0.5, 2.0)),
        dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='Normalize', **cfg.img_norm_cfg),
        dict(type='Pad', size=cfg.crop_size, pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    ]

    cfg.test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(320, 240),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **cfg.img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]

    cfg.data.train.type = cfg.dataset_type
    cfg.data.train.data_root = cfg.data_root
    cfg.data.train.img_dir = img_dir
    cfg.data.train.ann_dir = ann_dir
    cfg.data.train.pipeline = cfg.train_pipeline
    cfg.data.train.split = 'splits/train.txt'

    cfg.data.val.type = cfg.dataset_type
    cfg.data.val.data_root = cfg.data_root
    cfg.data.val.img_dir = val_img
    cfg.data.val.ann_dir = val_ann
    cfg.data.val.pipeline = cfg.test_pipeline
    cfg.data.val.split = val_splits

    cfg.data.test.type = cfg.dataset_type
    cfg.data.test.data_root = cfg.data_root
    cfg.data.test.img_dir = val_img
    cfg.data.test.ann_dir = val_ann
    cfg.data.test.pipeline = cfg.test_pipeline
    cfg.data.test.split = val_splits

    cfg.work_dir = working

    if mode == 'Train':
        if load_pth is not None:
            cfg.load_from = load_pth
            cfg.resume_from = load_pth
        else:
            cfg.load_from = None
            cfg.resume_from = None
        cfg.runner.max_iters = iterations
        cfg.log_config.interval = 100
        cfg.evaluation.interval = 1000
        cfg.checkpoint_config.interval = 1000
    elif mode == 'Eval':
        cfg.resume_from = load_pth
        cfg.runner.max_iters = iterations
        cfg.log_config.interval = 100
        cfg.evaluation.interval = 1
        cfg.checkpoint_config.interval = 1001

    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)
    cfg.device = get_device()

    return cfg


def getConfig_base(file, data_dir, load_pth, working, iterations=200, mode='Train'):
    cfg = Config.fromfile(file)
    img_dir = 'images'
    ann_dir = 'labels'

    # Since we use only one GPU, BN is used instead of SyncBN
    cfg.norm_cfg = dict(type='BN', requires_grad=True)
    cfg.model.backbone.norm_cfg = cfg.norm_cfg
    cfg.model.decode_head.norm_cfg = cfg.norm_cfg
    cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg

    # modify num classes of the model in decode/auxiliary head
    cfg.model.decode_head.num_classes = 6
    cfg.model.auxiliary_head.num_classes = 6

    # Modify dataset type and path
    cfg.dataset_type = 'CustomExperimentDataset'
    cfg.data_root = data_dir

    cfg.data.samples_per_gpu = 10
    cfg.data.workers_per_gpu = 4

    cfg.img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    cfg.crop_size = (256, 256)
    cfg.train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(320, 240), ratio_range=(0.5, 2.0)),
        dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='Normalize', **cfg.img_norm_cfg),
        dict(type='Pad', size=cfg.crop_size, pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    ]

    cfg.test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(320, 240),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **cfg.img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]

    cfg.data.train.type = cfg.dataset_type
    cfg.data.train.data_root = cfg.data_root
    cfg.data.train.img_dir = img_dir
    cfg.data.train.ann_dir = ann_dir
    cfg.data.train.pipeline = cfg.train_pipeline
    cfg.data.train.split = 'splits/train.txt'

    cfg.data.val.type = cfg.dataset_type
    cfg.data.val.data_root = cfg.data_root
    cfg.data.val.img_dir = img_dir
    cfg.data.val.ann_dir = ann_dir
    cfg.data.val.pipeline = cfg.test_pipeline
    cfg.data.val.split = 'splits/val.txt'

    cfg.data.test.type = cfg.dataset_type
    cfg.data.test.data_root = cfg.data_root
    cfg.data.test.img_dir = img_dir
    cfg.data.test.ann_dir = ann_dir
    cfg.data.test.pipeline = cfg.test_pipeline
    cfg.data.test.split = 'splits/val.txt'

    cfg.work_dir = working

    if mode == 'Train':
        if load_pth is not None:
            cfg.load_from = load_pth
            cfg.resume_from = load_pth
        else:
            cfg.load_from = None
            cfg.resume_from = None
        cfg.runner.max_iters = iterations
        cfg.log_config.interval = 1000
        cfg.evaluation.interval = 1000
        cfg.checkpoint_config.interval = 1000
    elif mode == 'Eval':
        cfg.load_from = load_pth
        cfg.resume_from = load_pth
        cfg.runner.max_iters = iterations
        cfg.log_config.interval = 100
        cfg.evaluation.interval = 1
        cfg.checkpoint_config.interval = 1001

    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)
    cfg.device = get_device()

    return cfg