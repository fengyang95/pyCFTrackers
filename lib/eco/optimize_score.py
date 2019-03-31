from .fourier_tools import sample_fs
from .config import gpu_config

import numpy as np
if gpu_config.use_gpu:
    import cupy as cp

"""
    code no problem
"""

def optimize_score(scores_fs, iterations):
    """
        Maximizes the continuous convolution response (classification scores)
    """
    if gpu_config.use_gpu:
        xp = cp.get_array_module(scores_fs)
    else:
        xp = np
    if len(scores_fs.shape) == 2:
        scores_fs = scores_fs[:, :, xp.newaxis]
    output_sz = scores_fs.shape[:2]

    # do the grid search step by finding the maximum in the sampled response for each scale
    sampled_scores = sample_fs(scores_fs)
    init_max_score = xp.max(sampled_scores, axis=(0, 1))
    max_idx = xp.reshape(sampled_scores, (-1, sampled_scores.shape[2])).argmax(axis=0)
    max_pos = xp.column_stack(xp.unravel_index(max_idx, sampled_scores[:,:,0].shape))
    row = max_pos[:, 0:1]
    col = max_pos[:, 1:2]

    # shift and rescale the coordinate system to [-pi, -pi]
    trans_row = (row + np.floor((output_sz[0] - 1)/2)) % output_sz[0] - np.floor((output_sz[1]-1)/2)
    trans_col = (col + np.floor((output_sz[1] - 1)/2)) % output_sz[1] - np.floor((output_sz[1]-1)/2)
    init_pos_y = 2 * np.pi * trans_row / output_sz[0]
    init_pos_x = 2 * np.pi * trans_col / output_sz[1]

    max_pos_y = init_pos_y
    max_pos_x = init_pos_x

    # construct grid
    ky = xp.arange(- np.ceil((output_sz[0] - 1)/2), np.floor(output_sz[0]-1)/2 + 1).reshape(1, -1)
    kx = xp.arange(- np.ceil((output_sz[1] - 1)/2), np.floor(output_sz[1]-1)/2 + 1).reshape(-1, 1)

    exp_iky = xp.exp(1j * max_pos_y * ky)[:, xp.newaxis, :].astype(xp.complex64)
    exp_ikx = xp.exp(1j * kx * max_pos_x.T).transpose()[:, :, xp.newaxis].astype(xp.complex64)

    ky2 = ky * ky
    kx2 = kx * kx

    max_pos_y = max_pos_y[:, :, xp.newaxis]
    max_pos_x = max_pos_x[:, :, xp.newaxis]

    init_pos_y = init_pos_y[:, :, xp.newaxis]
    init_pos_x = init_pos_x[:, :, xp.newaxis]

    scores_fs = scores_fs.transpose(2, 0, 1)
    for _ in range(iterations):
        # compute gradient
        ky_exp_ky = ky * exp_iky
        kx_exp_kx = kx * exp_ikx
        y_resp = xp.matmul(exp_iky, scores_fs)
        resp_x = xp.matmul(scores_fs, exp_ikx)
        grad_y = -xp.imag(xp.matmul(ky_exp_ky, resp_x))
        grad_x = -xp.imag(xp.matmul(y_resp, kx_exp_kx))

        # compute hessian
        ival = 1j * xp.matmul(exp_iky, resp_x)
        H_yy = xp.real(-xp.matmul(ky2 * exp_iky, resp_x) + ival)
        H_xx = xp.real(-xp.matmul(y_resp, kx2 * exp_ikx) + ival)
        H_xy = xp.real(-xp.matmul(ky_exp_ky, xp.matmul(scores_fs, kx_exp_kx)))
        det_H = H_yy * H_xx - H_xy * H_xy

        # compute new position using newtons method
        max_pos_y = max_pos_y - (H_xx * grad_y - H_xy * grad_x) / det_H
        max_pos_x = max_pos_x - (H_yy * grad_x - H_xy * grad_y) / det_H

        # evaluate maximum
        exp_iky = xp.exp(1j * ky * max_pos_y).astype(xp.complex64)
        exp_ikx = xp.exp(1j * kx * max_pos_x).astype(xp.complex64)

    max_score = xp.real(xp.matmul(xp.matmul(exp_iky, scores_fs), exp_ikx)).flatten()
    # check for scales that have not increased in score
    idx = max_score < init_max_score
    max_score[idx] = init_max_score[idx]
    max_pos_y[idx] = init_pos_y[idx]
    max_pos_x[idx] = init_pos_x[idx]
    scale_idx = xp.argmax(max_score)
    max_scale_response = max_score[scale_idx]
    disp_row = ((max_pos_y[scale_idx][0][0] + np.pi) % (2 * np.pi) - np.pi) / (2 * np.pi) * output_sz[0]
    disp_col = ((max_pos_x[scale_idx][0][0] + np.pi) % (2 * np.pi) - np.pi) / (2 * np.pi) * output_sz[1]

    if xp is np:
        return disp_row, disp_col, scale_idx
    else:
        return disp_row.get(), disp_col.get(), scale_idx.get()
