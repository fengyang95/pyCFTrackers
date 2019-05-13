class BACFConfig:
    cell_size=4
    cell_selection_thresh=0.75**2
    search_area_shape='square'
    search_area_scale=5
    filter_max_area=50**2
    interp_factor=0.015
    output_sigma_factor=1./16
    interpolate_response=4
    newton_iterations=5
    number_of_scales=1
    scale_step=1.01
    admm_iterations=2
    admm_lambda=0.01

    class ScaleConfig:
        learning_rate_scale = 0.015
        scale_sz_window = (128, 128)

    scale_config=ScaleConfig()


