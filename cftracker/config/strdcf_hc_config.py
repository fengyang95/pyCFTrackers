class STRDCFHCConfig:
    hog_cell_size=4
    hog_compressed_dim=10
    hog_n_dim=31

    gray_cell_size=4
    cn_use_for_gray=False
    cn_cell_size=4
    cn_n_dim=10


    search_area_shape = 'square'        # the shape of the samples
    search_area_scale = 4.0             # the scaling of the target size to get the search area
    min_image_sample_size = 150 ** 2    # minimum area of image samples
    max_image_sample_size = 200 ** 2    # maximum area of image samples

    feature_downsample_ratio=4
    reg_window_max=1e5
    reg_window_min=1e-3
    alpha=1000
    beta=0.4
    p=2


    # detection parameters
    refinement_iterations = 1          # number of iterations used to refine the resulting position in a frame
    newton_iterations = 5               # the number of Netwon iterations used for optimizing the detection score
    clamp_position = False              # clamp the target position to be inside the image

    # learning parameters
    output_sigma_factor = 1 / 16.       # label function sigma
    temporal_regularization_factor=15

    # ADMM params
    max_iterations=2
    init_penalty_factor=1
    max_penalty_factor=0.1
    penalty_scale_step=10


    # factorized convolution parameters
    use_projection_matrix = True        # use projection matrix, i.e. use the factorized convolution formulation
    update_projection_matrix = True     # whether the projection matrix should be optimized or not
    proj_init_method = 'pca'            # method for initializing the projection matrix
    projection_reg = 1e-7               # regularization parameter of the projection matrix



    # scale parameters
    number_of_scales = 1              # number of scales to run the detector
    scale_step = 1.01                 # the scale factor
    use_scale_filter = True             # use the fDSST scale filter or not

    class ScaleConfig:
        learning_rate_scale = 0.015
        scale_sz_window = (128, 128)

    scale_config=ScaleConfig()


