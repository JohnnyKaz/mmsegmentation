_base_ = './segmenter_vit-t_mask_8x1_512x512_160k_ade20k.py'

model = dict(
    decode_head=dict(
        _delete_=True,
        type='MeanshifterLinearHead',
        in_channels=192,
        channels=192,
        meanshift_channels=192,
        meanshift_layers=2,
        num_convs=0,
        dropout_ratio=0.0,
        concat_input=False,
        num_classes=150,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))
