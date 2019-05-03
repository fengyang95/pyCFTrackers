import numpy as np
class MKCFupOTB50Config:
    gap=6
    lr_cn_color=0.0174
    lr_cn_gray=0.0175
    lr_hog_color=0.0173
    lr_hog_gray=0.018
    num_compressed_dim_cn=4
    num_compressed_dim_hog=4

    padding=1.5
    output_sigma_factor=1/16
    scale_sigma_factor=1/16
    lambda_=1e-2
    interp_factor=0.025

    cn_sigma_color=0.515
    hog_sigma_color=0.6

    cn_sigma_gray=0.3
    hog_sigma_gray=0.4
    refinement_iterations=1
    translation_model_max_area=np.inf
    interpolate_response=1
    num_of_scales=20
    num_of_interp_scales=39
    scale_model_factor=1.
    scale_step=1.02
    scale_model_max_area=512
    s_num_compressed_dim='MAX'








