import numpy as np
class DATConfig:
    img_scale_target_diagonal=75
    search_win_padding=2
    surr_win_factor=1.9
    color_space='rgb'
    num_bins=16
    prob_lut_update_rate=0.05
    distractor_aware=True
    adapt_thresh_prob_bins=np.arange(0,1.001,0.05)
    motion_estimation_history_size=5
    nms_scale=1
    nms_overlap=0.9
    nms_score_factor=0.5
    nms_include_center_vote=True