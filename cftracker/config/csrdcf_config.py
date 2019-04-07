class CSRDCFConfig:
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
    mask_diletation_type='disk' # for function strel (square, disk , ...)
    mask_diletation_sz=0

    # scale adaptation parameters
    current_scale_factor=1.
    n_scales=33
    scale_model_factor=1.
    scale_sigma_factor=1/4
    scale_step=1.02
    scale_model_max_area=32*16
    scale_lr=0.025



