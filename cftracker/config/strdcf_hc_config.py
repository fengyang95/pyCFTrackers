class STRDCFHCConfig:
    hog_cell_size=4
    hog_compressed_dim=10
    hog_n_dim=31

    gray_cell_size=4
    cn_use_for_gray=False
    cn_cell_size=4
    cn_n_dim=10

    search_area_shape = 'square'        # the shape of the samples
    search_area_scale = 5.0             # the scaling of the target size to get the search area
    min_image_sample_size = 150 ** 2    # minimum area of image samples
    max_image_sample_size = 200 ** 2    # maximum area of image samples

    feature_downsample_ratio=4
    reg_window_max=1e5
    reg_window_min=1e-3
    alpha=1000
    beta=0.4
    p=2



    # detection parameters
    refinement_iterations = 1        # number of iterations used to refine the resulting position in a frame
    newton_iterations = 5              # the number of Netwon iterations used for optimizing the detection score
    clamp_position = False              # clamp the target position to be inside the image

    # learning parameters
    output_sigma_factor = 1 / 16.       # label function sigma
    temporal_regularization_factor=15

    # ADMM params
    max_iterations=2
    init_penalty_factor=1
    max_penalty_factor=0.1
    penalty_scale_step=10

    # scale parameters
    number_of_scales = 1      # number of scales to run the detector
    scale_step = 1.01               # the scale factor
    use_scale_filter = True             # use the fDSST scale filter or not


    scale_type='LP'
    class ScaleConfig:
        learning_rate_scale = 0.015
        scale_sz_window = (64, 64)

    scale_config=ScaleConfig()


    """
    scale_type = 'normal'

    class ScaleConfig:
        scale_sigma_factor = 1 / 16.  # scale label function sigma
        scale_learning_rate = 0.025  # scale filter learning rate
        number_of_scales_filter = 11  # number of scales
        number_of_interp_scales = 11  # number of interpolated scales
        scale_model_factor = 1.0  # scaling of the scale model
        scale_step_filter = 1.02  # the scale factor of the scale sample patch
        scale_model_max_area = 32 * 16  # maximume area for the scale sample patch
        scale_feature = 'HOG4'  # features for the scale filter (only HOG4 supported)
        s_num_compressed_dim = 'MAX'  # number of compressed feature dimensions in the scale filter
        lamBda = 1e-2  # scale filter regularization
        do_poly_interp = False

    scale_config = ScaleConfig()
    """

    normalize_power = 2  # Lp normalization with this p
    normalize_size = True  # also normalize with respect to the spatial size of the feature
    normalize_dim = True  # also normalize with respect to the dimensionality of the feature
    square_root_normalization = False


