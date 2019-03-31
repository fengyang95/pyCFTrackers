class DemoConfig:
    fhog_params = {'fname': 'fhog',
                   'num_orients': 9,
                   'cell_size': 6,
                   'compressed_dim': 10,
                   # 'nDim': 9 * 3 + 5 -1
                   }

    cn_params = {"fname": 'cn',
                 "table_name": "CNnorm",
                 "use_for_color": True,
                 "cell_size": 4,
                 "compressed_dim": 3,
                 # "nDim": 10
                 }

    ic_params = {'fname': 'ic',
                 "table_name": "intensityChannelNorm6",
                 "use_for_color": False,
                 "cell_size": 4,
                 "compressed_dim": 3,
                 # "nDim": 10
                 }

    features = [fhog_params, cn_params, ic_params]

    # feature parameters
    normalize_power = 2
    normalize_size = True
    normalize_dim = True
    square_root_normalization = False

    # image sample parameters
    search_area_shape = 'square'
    search_area_scale = 4.0
    min_image_sample_size = 150 ** 2
    max_image_sample_size = 200 ** 2

    # detection parameters
    refinement_iterations = 1           # number of iterations used to refine the resulting position in a frame
    newton_iterations = 5
    clamp_position = False              # clamp the target position to be inside the image

    # learning parameters
    output_sigma_factor = 1 / 16.       # label function sigma
    learning_rate = 0.009
    num_samples = 30
    sample_replace_startegy = 'lowest_prior'
    lt_size = 0
    train_gap = 5
    skip_after_frame = 10
    use_detection_sample = True

    # factorized convolution parameters
    use_projection_matrix = True
    update_projection_matrix = True
    proj_init_method = 'pca'
    projection_reg = 1e-7

    # generative sample space model parameters
    use_sample_merge = True
    sample_merge_type = 'merge'
    distance_matrix_update_type = 'exact'

    # CG paramters
    CG_iter = 5
    init_CG_iter = 10 * 15
    init_GN_iter = 10
    CG_use_FR = False
    CG_standard_alpha = True
    CG_forgetting_rate = 50
    precond_data_param = 0.75
    precond_reg_param = 0.25
    precond_proj_param = 40

    # regularization window paramters
    use_reg_window = True
    reg_window_min = 1e-4
    reg_window_edge = 10e-3
    reg_window_power = 2
    reg_sparsity_threshold = 0.05

    # interpolation parameters
    interp_method = 'bicubic'
    interp_bicubic_a = -0.75
    interp_centering = True
    interp_windowing = False

    # scale parameters
    number_of_scales = 7
    scale_step = 1.01
    use_scale_filter = True

    # only used if use_scale_filter == true
    scale_sigma_factor = 1 / 16.
    scale_learning_rate = 0.025
    number_of_scales_filter = 17
    number_of_interp_scales = 33
    scale_model_factor = 1.0
    scale_step_filter = 1.02
    scale_model_max_area = 32 * 16
    scale_feature = 'HOG4'
    s_num_compressed_dim = 'MAX'
    lamBda = 1e-2
    do_poly_interp = True

    # visualization
    visualization = 1
    debug = 0

    # gpu
    use_gpu = False
    gpu_id = None
