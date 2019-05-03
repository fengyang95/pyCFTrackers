class CSRDCFLPConfig:
    # filter params
    padding=3
    interp_factor=0.02
    y_sigma=1
    channels_weight_lr=interp_factor
    use_channel_weights=True

    # segmentation params
    hist_lr=0.04
    nbins=16
    seg_colorspace='hsv' # 'bgr' or 'hsv'
    use_segmentation=True

    sc=1.



