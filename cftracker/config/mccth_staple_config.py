class MCCTHOTBConfig:
    hog_cell_size = 4
    fixed_area = 150 ** 2
    n_bins = 2 ** 5
    interp_factor_pwp = 0.04
    inner_padding = 0.2
    output_sigma_factor = 1. / 16
    lambda_ = 1e-3
    interp_factor_cf = 0.01
    merge_factor = 0.3
    padding=1
    den_per_channel = False

    scale_adaptation=True
    class ScaleConfig:
        learning_rate_scale = 0.015
        scale_sz_window = (128, 128)

    scale_config=ScaleConfig()
    use_ca=False
    period=5
    update_thresh=0.6
    expert_num=7

class MCCTHVOTConfig:
    hog_cell_size = 4
    fixed_area = 22453
    n_bins = 16
    interp_factor_pwp = 0.023
    inner_padding = 0.0149
    output_sigma_factor =0.0892
    lambda_ = 1e-3
    interp_factor_cf = 0.0153
    merge_factor = 0.567
    den_per_channel = False
    padding=1

    scale_adaptation = True
    class ScaleConfig:
        learning_rate_scale = 0.015
        scale_sz_window = (128, 128)

    scale_config = ScaleConfig()
    use_ca = False
    period = 5
    update_thresh = 0.6
    expert_num = 7



