import numpy as np

class MKCFupConfig:
    gap=6
    lr_cn_color=0.0174
    lr_cn_gray=0.0175
    lr_hog_color=0.0173
    lr_hog_gray=0.018
    num_compressed_dim_cn=4
    num_compressed_dim_hog=4

    padding=1.5
    output_sigma_factor=1/16
    lambda_=1e-2
    interp_factor=0.025

    cn_sigma_color=0.515
    hog_sigma_color=0.6

    cn_sigma_gray=0.3
    hog_sigma_gray=0.4
    refinement_iterations=1
    translation_model_max_area=np.inf

    interpolate_response=1

    scale_type = 'normal'

    class ScaleConfig:
        scale_sigma_factor = 1 / 16.  # scale label function sigma
        scale_learning_rate = 0.025  # scale filter learning rate
        number_of_scales_filter = 20  # number of scales
        number_of_interp_scales = 39  # number of interpolated scales
        scale_model_factor = 1.0  # scaling of the scale model
        scale_step_filter = 1.02  # the scale factor of the scale sample patch
        scale_model_max_area = 32 * 16  # maximume area for the scale sample patch
        scale_feature = 'HOG4'  # features for the scale filter (only HOG4 supported)
        s_num_compressed_dim = 'MAX'  # number of compressed feature dimensions in the scale filter
        lamBda = 1e-2  # scale filter regularization
        do_poly_interp = False

    scale_config = ScaleConfig()


class MKCFupLPConfig:
    gap = 6
    lr_cn_color = 0.0174
    lr_cn_gray = 0.0175
    lr_hog_color = 0.0173
    lr_hog_gray = 0.018
    num_compressed_dim_cn = 4
    num_compressed_dim_hog = 4

    padding = 1.5
    output_sigma_factor = 1 / 16
    lambda_ = 1e-2
    interp_factor = 0.025

    cn_sigma_color = 0.515
    hog_sigma_color = 0.6

    cn_sigma_gray = 0.3
    hog_sigma_gray = 0.4
    refinement_iterations = 1
    translation_model_max_area = np.inf

    interpolate_response = 1

    scale_type = 'LP'

    class ScaleConfig:
        learning_rate_scale = 0.015
        scale_sz_window = (128, 128)

    scale_config=ScaleConfig()











