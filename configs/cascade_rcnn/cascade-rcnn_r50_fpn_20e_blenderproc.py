_base_ = [
    '../_base_/models/cascade-rcnn_r50_fpn.py',
    '../_base_/datasets/blenderproc_cubes.py', '../_base_/default_runtime.py'
]

# training schedule for 20e
max_epochs = 200
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=50)
# val_cfg = dict(type='ValLoop')
# test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0,
        begin=1,
        T_max=299,
        end=300,
        by_epoch=True,
        convert_to_iter_based=True)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    # optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
    optimizer=dict(type='Adam', lr=0.003, weight_decay=0.0001)
    )

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
