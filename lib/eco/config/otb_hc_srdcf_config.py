# https://github.com/martin-danelljan/ECO/blob/master/runfiles/SRDCF_settings.m
class OTBHCSRDCFConfig:
    fhog_params = {'fname': 'fhog',
                   'num_orients': 9,
                   'cell_size': 6,
                   'compressed_dim': 10,
                   }

    cn_params = {"fname": 'cn',
                 "table_name": "CNnorm",
                 "use_for_color": True,
                 "cell_size": 4,
                 "compressed_dim": 3,
                 }

    ic_params = {'fname': 'ic',
                 "table_name": "intensityChannelNorm6",
                 "use_for_color": False,
                 "cell_size": 4,
                 "compressed_dim": 3,
                 }

    features = [fhog_params,cn_params, ic_params]

    # feature parameters
    normalize_power = 2                 # Lp normalization with this p
    normalize_size = True               # also normalize with respect to the spatial size of the feature
    normalize_dim = True                # also normalize with respect to the dimensionality of the feature
    square_root_normalization = False   #

    # image sample parameters
    search_area_shape = 'square'        # the shape of the samples
    search_area_scale = 4.0             # the scaling of the target size to get the search area
    min_image_sample_size = 150 ** 2    # minimum area of image samples
    max_image_sample_size = 200 ** 2    # maximum area of image samples

    # detection parameters
    refinement_iterations = 1           # number of iterations used to refine the resulting position in a frame
    newton_iterations = 5               # the number of Netwon iterations used for optimizing the detection score
    clamp_position = False              # clamp the target position to be inside the image

    # learning parameters
    output_sigma_factor = 1 / 16.       # label function sigma
    learning_rate = 0.010               # learning rate
    num_samples = 300                  # maximum number of stored training samples
    sample_replace_startegy = 'lowest_prior' # which sample to replace when the memory is full
    lt_size = 0                         # the size of the long-term memory (where all samples have equal weight)
    train_gap = 0                      # the number of intermediate frames with no training (0 corresponds to the training every frame)
    skip_after_frame = 0               # after which frame number the sparse update scheme should start (1 is directly)
    use_detection_sample = False         # use the sample that was extracted at the detection stage also for learning

    # factorized convolution parameters
    use_projection_matrix = False        # use projection matrix, i.e. use the factorized convolution formulation
    update_projection_matrix = True     # whether the projection matrix should be optimized or not
    proj_init_method = 'pca'            # method for initializing the projection matrix
    projection_reg = 1e-7               # regularization parameter of the projection matrix

    # generative sample space model parameters
    use_sample_merge = False             # use the generative sample space model to merge samples
    sample_merge_type = 'merge'         # strategy for updating the samples
    distance_matrix_update_type = 'exact' # strategy for updating the distance matrix

    # CG paramters
    CG_iter = 5                         # the number of Conjugate Gradient iterations in each update after the first time
    init_CG_iter = 50              # the total number of Conjugate Gradient iterations used in the first time
    init_GN_iter = 5                   # the number of Gauss-Netwon iterations used in the first frame (only if the projection matrix is updated)
    CG_use_FR = False                   # use the Fletcher-Reeves or Polak-Ribiere formula in the Conjugate Gradient
    CG_standard_alpha = True            # use the standard formula for computing the step length in Conjugate Gradient
    CG_forgetting_rate = 50             # forgetting rate of the last conjugate direction
    precond_data_param = 0.75           # weight of the data term in the preconditioner
    precond_reg_param = 0.25            # weight of the regularization term in the preconditioner
    #precond_proj_param = 40             # weight of the projection matrix part in the preconditioner

    # regularization window paramters
    use_reg_window = True               # use spatial regularizaiton or not
    reg_window_min = 1e-4               # the minimum value of the regularization window
    reg_window_edge = 10e-3             # the impace of the spatial regularization
    reg_window_power = 2                # the degree of the polynomial to use (e.g. 2 is q quadratic window)
    reg_sparsity_threshold = 0.05       # a relative threshold of which DFT coefficients of the kernel

    # interpolation parameters
    interp_method = 'ideal'           # the kind of interpolation kernel
    #interp_bicubic_a = -0.75            # the parameter for the bicubic interpolation kernel
    interp_centering = False             # center the kernel at the feature sample
    interp_windowing = False            # do additional windowing on the Fourier coefficients of the kernel

    # scale parameters
    number_of_scales = 7                # number of scales to run the detector
    scale_step = 1.01                   # the scale factor
    use_scale_filter = False            # use the fDSST scale filter or not

    # only used if use_scale_filter == true
    #scale_sigma_factor = 1 / 16.        # scale label function sigma
    #scale_learning_rate = 0.025         # scale filter learning rate
    #number_of_scales_filter = 17        # number of scales
    #number_of_interp_scales = 33        # number of interpolated scales
    #scale_model_factor = 1.0            # scaling of the scale model
    #scale_step_filter = 1.02            # the scale factor of the scale sample patch
    #scale_model_max_area = 32 * 16      # maximume area for the scale sample patch
    #scale_feature = 'HOG4'              # features for the scale filter (only HOG4 supported)
    #s_num_compressed_dim = 'MAX'        # number of compressed feature dimensions in the scale filter
    #lamBda = 1e-2                       # scale filter regularization
    #do_poly_interp = False               # do 2nd order polynomial interpolation to obtain more accurate scale

    # gpu
    use_gpu = False                     # enable gpu or not, only use if use deep feature
    gpu_id = None                       # set the gpu id
