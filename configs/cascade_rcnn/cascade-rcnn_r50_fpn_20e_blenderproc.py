_base_ = [
    '../_base_/models/cascade-rcnn_r50_fpn.py',
    '../_base_/datasets/blenderproc_cubes.py', '../_base_/default_runtime.py'
]

# training schedule for 20e
max_epochs = 100
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',  # Use linear learning rate warmup
        start_factor=0.001, # Coefficient for learning rate warmup
        by_epoch=False,  # Update the learning rate during warmup at each iteration
        begin=0,  # Starting from the first iteration
        end=2000),  # End at the 500th iteration
    dict(
        type='MultiStepLR',  # Use multi-step learning rate strategy during training
        by_epoch=True,  # Update the learning rate at each epoch
        begin=0,   # Starting from the first epoch
        end=12,  # Ending at the 12th epoch
        milestones=[8, 11],  # Learning rate decay at which epochs
        gamma=0.1)  # Learning rate decay coefficient
]

# optimizer
optim_wrapper = dict(  # Configuration for the optimizer wrapper
    type='AmpOptimWrapper',  # Type of optimizer wrapper, you can switch to AmpOptimWrapper to enable mixed precision training
    optimizer=dict(  # Optimizer configuration, supports various PyTorch optimizers, please refer to https://pytorch.org/docs/stable/optim.html#algorithms
        type='SGD',  # SGD
        lr=0.005,  # Base learning rate
        momentum=0.9,  # SGD with momentum
        weight_decay=0.0001),  # Weight decay
    clip_grad=dict(max_norm=35, norm_type=2),  # Configuration for gradient clipping, set to None to disable. For usage, please see https://mmengine.readthedocs.io/en/latest/tutorials/optimizer.html
    )

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
