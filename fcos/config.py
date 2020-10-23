_config_dict = dict(
    MODEL=dict(
        # META_ARCHITECTURE="RetinaNet",
        RESNETS=dict(OUT_FEATURES=["res3", "res4", "res5"]),
        FPN=dict(IN_FEATURES=["res3", "res4", "res5"]),
        SHIFT_GENERATOR=dict(
            NUM_SHIFTS=1,
            # Relative offset between the center of the first shift and the top-left corner of img
            # Units: fraction of feature map stride (e.g., 0.5 means half stride)
            # Allowed values are floats in [0, 1) range inclusive.
            # Recommended value is 0.5, although it is not expected to affect model accuracy.
            OFFSET=0.0,
        ),
        FCOS=dict(
            NUM_CLASSES=80,
            IN_FEATURES=["p3", "p4", "p5", "p6", "p7"],
            NUM_CONVS=4,
            FPN_STRIDES=[8, 16, 32, 64, 128],
            PRIOR_PROB=0.01,
            CENTERNESS_ON_REG=False,
            NORM_REG_TARGETS=False,
            SCORE_THRESH_TEST=0.05,
            TOPK_CANDIDATES_TEST=1000,
            NMS_THRESH_TEST=0.6,
            BBOX_REG_WEIGHTS=(1.0, 1.0, 1.0, 1.0),
            FOCAL_LOSS_GAMMA=2.0,
            FOCAL_LOSS_ALPHA=0.25,
            IOU_LOSS_TYPE="iou",
            CENTER_SAMPLING_RADIUS=0.0,
            OBJECT_SIZES_OF_INTEREST=[
                [-1, 64],
                [64, 128],
                [128, 256],
                [256, 512],
                [512, float("inf")],
            ],
            NORM_SYNC=True,
        ),
    ),
)
#====
import os.path as osp

from cvpods.configs.fcos_config import FCOSConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-50.pkl",
        RESNETS=dict(DEPTH=50),
        FCOS=dict(
            NUM_CONVS=16,
            CENTERNESS_ON_REG=True,
            NORM_REG_TARGETS=True,
            NMS_THRESH_TEST=0.6,
            BBOX_REG_WEIGHTS=(1.0, 1.0, 1.0, 1.0),
            FOCAL_LOSS_GAMMA=2.0,
            FOCAL_LOSS_ALPHA=0.25,
            IOU_LOSS_TYPE="giou",
            CENTER_SAMPLING_RADIUS=1.5,
            OBJECT_SIZES_OF_INTEREST=[
                [-1, 64],
                [64, 128],
                [128, 256],
                [256, 512],
                [512, float("inf")],
            ],
            NORM_SYNC=True,
            NUM_GROUPS=1,
            GATE_ACTIVATION="GeReTanH",
            GATE_ACTIVATION_KARGS=dict(tau=1.5),
            RESIZE_METHOD="bilinear",
            BUDGET_LOSS_LAMBDA=0.0,
        ),
    ),
    DATASETS=dict(
        TRAIN=("coco_2017_train",),
        TEST=("coco_2017_val",),
    ),
    SOLVER=dict(
        IMS_PER_BATCH=16,
        BASE_LR=0.01,
        STEPS=(60000, 80000),
        MAX_ITER=90000,
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=(800,), max_size=1333, sample_style="choice")),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=800, max_size=1333, sample_style="choice")),
            ],
        )
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]
    ),
)

