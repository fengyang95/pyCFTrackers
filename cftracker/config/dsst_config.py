class DSSTConfig:
    interp_factor = 0.025
    sigma = 0.2
    lambda_ = 0.01
    output_sigma_factor = 1. / 16
    padding = 1
    scale_type='normal'
    class ScaleConfig:
        scale_sigma_factor = 1 / 16.  # scale label function sigma
        scale_learning_rate = 0.025  # scale filter learning rate
        number_of_scales_filter = 17  # number of scales
        number_of_interp_scales = 33  # number of interpolated scales
        scale_model_factor = 1.0  # scaling of the scale model
        scale_step_filter = 1.02  # the scale factor of the scale sample patch
        scale_model_max_area = 32 * 16  # maximume area for the scale sample patch
        scale_feature = 'HOG4'  # features for the scale filter (only HOG4 supported)
        s_num_compressed_dim = 'MAX'  # number of compressed feature dimensions in the scale filter
        lamBda = 1e-2  # scale filter regularization
        do_poly_interp = False

    scale_config=ScaleConfig()


class DSSTLPConfig:
    interp_factor = 0.025
    sigma = 0.2
    lambda_ = 0.01
    output_sigma_factor = 1. / 16
    padding = 1
    use_scale_filter=True
    scale_type='LP'
    class ScaleConfig:
        learning_rate_scale = 0.015
        scale_sz_window = (128, 128)

    scale_config=ScaleConfig()
