class ASRCFHCConfig:
    cell_size=4
    cell_selection_thresh=0.75**2
    search_area_shape='square'
    search_area_scale=5
    filter_max_area=50**2
    interp_factor=0.0186
    output_sigma_factor=1./16
    interpolate_response=4
    newton_iterations=5
    number_of_scales=1
    scale_step=1.01
    admm_iterations=1
    admm_lambda1=0.2
    admm_lambda2=1e-3

    reg_window_min = 1e-4# the minimum value of the regularization window
    reg_window_edge=1e-4

    class ScaleConfig:
        learning_rate_scale = 0.015
        scale_sz_window = (128, 128)

    scale_config=ScaleConfig()


