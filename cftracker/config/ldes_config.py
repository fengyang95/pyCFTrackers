class LDESOTBLinearConfig:
    kernel_type='linear'
    padding=1.5
    lambda_=1e-4
    output_sigma_factor=0.1
    interp_factor=0.01
    cell_size=4

    min_image_sample_size=100**2
    max_image_sample_size=350**2

    fixed_model_sz=(224,224)
    is_rotation=True
    is_BGD=True
    is_subpixel=True
    interp_n=0.85

    learning_rate_scale=0.015
    scale_sz_window=(128,128)

    # color histogram
    inter_patch_rate=0.3
    nbin=10
    color_update_rate=0.01
    merge_factor=0.4
    use_color_hist=True
    polygon=False

    sigma=None

class LDESOTBLinearConfig_AAAI_Paper:
    kernel_type='linear'
    padding=1.5
    lambda_=1e-4
    output_sigma_factor=0.1
    interp_factor=0.15
    cell_size=4

    min_image_sample_size=100**2
    max_image_sample_size=350**2

    fixed_model_sz=(224,224)
    is_rotation=True
    is_BGD=True
    is_subpixel=True
    interp_n=0.85

    learning_rate_scale=0.015
    scale_sz_window=(128,128)

    # color histogram
    inter_patch_rate=0.3
    nbin=10
    color_update_rate=0.01
    merge_factor=0.4
    use_color_hist=True
    polygon=False

    sigma=None


class LDESVOTLinearConfig:
    kernel_type = 'linear'
    padding = 1.5
    lambda_ = 1e-4
    output_sigma_factor = 0.1
    interp_factor = 0.01
    cell_size = 4

    min_image_sample_size = 100 ** 2
    max_image_sample_size = 350 ** 2

    fixed_model_sz = (224, 224)
    is_rotation = True
    is_BGD = is_rotation
    is_subpixel = True
    interp_n = 0.85

    learning_rate_scale = 0.015
    scale_sz_window = (128, 128)

    # color histogram
    inter_patch_rate = 0.3
    nbin = 10
    color_update_rate = 0.01
    merge_factor = 0.4
    use_color_hist=True
    sigma=None
    polygon = True



