import numpy as np
import cv2


def resize_dft2(input_dft, desired_sz):
    h, w, num_imgs = input_dft.shape
    if desired_sz[0] != w or desired_sz[1] != h:
        minsz = (int(min(w, desired_sz[0])), int(min(h, desired_sz[1])))
        scaling = (desired_sz[0] * desired_sz[1] / (h * w))
        resized_dfft = np.zeros((desired_sz[1], desired_sz[0], num_imgs), dtype=np.complex64)
        mids = (int(np.ceil(minsz[0] / 2)), int(np.ceil(minsz[1] / 2)))
        mide = (int(np.floor((minsz[0] - 1) / 2)) - 1, int(np.floor((minsz[1] - 1) / 2)) - 1)
        resized_dfft[:mids[1], :mids[0], :] = scaling * input_dft[:mids[1], :mids[0], :]
        resized_dfft[:mids[1], -1 - mide[0]:, :] = scaling * input_dft[:mids[1], -1 - mide[0]:, :]
        resized_dfft[-1 - mide[1]:, :mids[0], :] = scaling * input_dft[-1 - mide[1]:, :mids[0], :]
        resized_dfft[-1 - mide[1]:, -1 - mide[0]:, :] = scaling * input_dft[-1 - mide[1]:, -1 - mide[0]:, :]
        return resized_dfft
    else:
        return input_dft


def mex_resize(img, sz, method='auto'):
    sz = (int(sz[0]), int(sz[1]))
    src_sz = (img.shape[1], img.shape[0])
    if method == 'antialias':
        interpolation = cv2.INTER_AREA
    elif method == 'linear':
        interpolation = cv2.INTER_LINEAR
    else:
        if sz[1] > src_sz[1]:
            interpolation = cv2.INTER_LINEAR
        else:
            interpolation = cv2.INTER_AREA
    img = cv2.resize(img, sz, interpolation=interpolation)
    return img


"""
I didn't know how to convert matlab function mtimesx to numpy
Just finetune from 4kubo's implementation
https://github.com/4kubo/bacf_python/blob/master/special_operation/resp_newton.py
"""
def resp_newton(response, responsef, iterations, ky, kx, use_sz):
    n_scale = response.shape[2]
    index_max_in_row = np.argmax(response, 0)
    max_resp_in_row = np.max(response, 0)
    index_max_in_col = np.argmax(max_resp_in_row, 0)
    init_max_response = np.max(max_resp_in_row, 0)
    col = index_max_in_col.flatten(order="F")

    max_row_perm = index_max_in_row
    row = max_row_perm[col, np.arange(n_scale)]

    trans_row = (row - 1 + np.floor((use_sz[1] - 1) / 2)) % use_sz[1] \
                - np.floor((use_sz[1] - 1) / 2) + 1
    trans_col = (col - 1 + np.floor((use_sz[0] - 1) / 2)) % use_sz[0] \
                - np.floor((use_sz[0] - 1) / 2) + 1
    init_pos_y = np.reshape(2 * np.pi * trans_row / use_sz[1], (1, 1, n_scale))
    init_pos_x = np.reshape(2 * np.pi * trans_col / use_sz[0], (1, 1, n_scale))
    max_pos_y = init_pos_y
    max_pos_x = init_pos_x

    # pre-compute complex exponential
    iky = 1j * ky
    exp_iky = np.tile(iky[np.newaxis, :, np.newaxis], (1, 1, n_scale)) * \
              np.tile(max_pos_y, (1, ky.shape[0], 1))
    exp_iky = np.exp(exp_iky)

    ikx = 1j * kx
    exp_ikx = np.tile(ikx[:, np.newaxis, np.newaxis], (1, 1, n_scale)) * \
              np.tile(max_pos_x, (kx.shape[0], 1, 1))
    exp_ikx = np.exp(exp_ikx)

    # gradient_step_size = gradient_step_size / prod(use_sz)

    ky2 = ky * ky
    kx2 = kx * kx

    iter = 1
    while iter <= iterations:
        # Compute gradient
        ky_exp_ky = np.tile(ky[np.newaxis, :, np.newaxis], (1, 1, exp_iky.shape[2])) * exp_iky
        kx_exp_kx = np.tile(kx[:, np.newaxis, np.newaxis], (1, 1, exp_ikx.shape[2])) * exp_ikx
        y_resp = np.einsum('ilk,ljk->ijk', exp_iky, responsef)
        resp_x = np.einsum('ilk,ljk->ijk', responsef, exp_ikx)
        grad_y = -np.imag(np.einsum('ilk,ljk->ijk', ky_exp_ky, resp_x))
        grad_x = -np.imag(np.einsum('ilk,ljk->ijk', y_resp, kx_exp_kx))
        ival = 1j * np.einsum('ilk,ljk->ijk', exp_iky, resp_x)
        H_yy = np.tile(ky2[np.newaxis, :, np.newaxis], (1, 1, n_scale)) * exp_iky
        H_yy = np.real(-np.einsum('ilk,ljk->ijk', H_yy, resp_x) + ival)

        H_xx = np.tile(kx2[:, np.newaxis, np.newaxis], (1, 1, n_scale)) * exp_ikx
        H_xx = np.real(-np.einsum('ilk,ljk->ijk', y_resp, H_xx) + ival)
        H_xy = np.real(-np.einsum('ilk,ljk->ijk', ky_exp_ky, np.einsum('ilk,ljk->ijk', responsef, kx_exp_kx)))
        det_H = H_yy * H_xx - H_xy * H_xy

        # Compute new position using newtons method
        diff_y = (H_xx * grad_y - H_xy * grad_x) / det_H
        diff_x = (H_yy * grad_x - H_xy * grad_y) / det_H
        max_pos_y = max_pos_y - diff_y
        max_pos_x = max_pos_x - diff_x

        # Evaluate maximum
        exp_iky = np.tile(iky[np.newaxis, :, np.newaxis], (1, 1, n_scale)) * \
                  np.tile(max_pos_y, (1, ky.shape[0], 1))
        exp_iky = np.exp(exp_iky)

        exp_ikx = np.tile(ikx[:, np.newaxis, np.newaxis], (1, 1, n_scale)) * \
                  np.tile(max_pos_x, (kx.shape[0], 1, 1))
        exp_ikx = np.exp(exp_ikx)

        iter = iter + 1

    max_response = 1 / np.prod(use_sz) * \
                   np.real(np.einsum('ilk,ljk->ijk',
                                     np.einsum('ilk,ljk->ijk', exp_iky, responsef),
                                     exp_ikx))

    # check for scales that have not increased in score
    ind = max_response < init_max_response
    max_response[0, 0, ind.flatten()] = init_max_response[ind.flatten()]
    max_pos_y[0, 0, ind.flatten()] = init_pos_y[0, 0, ind.flatten()]
    max_pos_x[0, 0, ind.flatten()] = init_pos_x[0, 0, ind.flatten()]

    sind = int(np.nanargmax(max_response, 2))
    disp_row = (np.mod(max_pos_y[0, 0, sind] + np.pi, 2 * np.pi) - np.pi) / (2 * np.pi) * use_sz[1]
    disp_col = (np.mod(max_pos_x[0, 0, sind] + np.pi, 2 * np.pi) - np.pi) / (2 * np.pi) * use_sz[0]

    return disp_row, disp_col, sind
