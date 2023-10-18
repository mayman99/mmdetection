# training schedule for 20e
max_epochs = 1000
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=8)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',  # Use linear learning rate warmup
        start_factor=0.001, # Coefficient for learning rate warmup
        by_epoch=False,  # Update the learning rate during warmup at each iteration
        begin=0,  # Starting from the first iteration
        end=1000),  # End at the 500th iteration -> the bigger the less likely to go to nan
    dict(
        type='MultiStepLR',  # Use multi-step learning rate strategy during training
        by_epoch=True,  # Update the learning rate at each epoch
        begin=0,   # Starting from the first epoch
        end=max_epochs,  # Ending at the 12th epoch
        milestones=[max_epochs],  # Learning rate decay at which epochs
        gamma=0.1)  # Learning rate decay coefficient
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    # optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
    optimizer=dict(type='Adam', lr=0.00005, weight_decay=0.0001)
    # optimizer=dict(type='Adam', lr=0.0005, weight_decay=0.0001) worked for 27 images after epoch 60, warmup of 2000 its
    )

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
