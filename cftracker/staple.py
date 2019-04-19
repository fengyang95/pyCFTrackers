"""
Python re-implementation of "Staple: Complementary Learners for Real-Time Tracking"
@inproceedings{Bertinetto2016Staple,
  title={Staple: Complementary Learners for Real-Time Tracking},
  author={Bertinetto, Luca and Valmadre, Jack and Golodetz, Stuart and Miksik, Ondrej and Torr, Philip},
  booktitle={Computer Vision & Pattern Recognition},
  year={2016},
}
"""
import numpy as np
import cv2
from .base import BaseCF
from .feature import extract_hog_feature
from lib.utils import cos_window
from lib.fft_tools import fft2, ifft2


def mod_one(a, b):
    y = np.mod(a - 1, b) + 1
    return y


# max val at the bottom right loc
def gaussian2d_rolled_labels_staple(sz, sigma):
    halfx, halfy = int(np.floor((sz[0] - 1) / 2)), int(np.floor((sz[1] - 1) / 2))
    x_range = np.arange(-halfx, halfx + 1)
    y_range = np.arange(-halfy, halfy + 1)
    i, j = np.meshgrid(y_range, x_range)
    i_mod_range = mod_one(i, sz[1])
    j_mod_range = mod_one(j, sz[0])
    labels = np.zeros((sz[1], sz[0]))
    labels[i_mod_range - 1, j_mod_range - 1] = np.exp(-(i ** 2 + j ** 2) / (2 * sigma ** 2))
    return labels


def crop_filter_response(response_cf, response_sz):
    h, w = response_cf.shape[:2]
    half_width = int(np.floor(response_sz[0] / 2))
    half_height = int(np.floor(response_sz[1] / 2))
    range_i, range_j = np.arange(-half_height, half_height + 1), np.arange(-half_width, half_width + 1)
    i, j = np.meshgrid(mod_one(range_i, h), mod_one(range_j, w))
    new_responses = response_cf[i - 1, j - 1]
    return new_responses.T


def get_center_likelihood(likelihood_map, m):
    h, w = likelihood_map.shape[:2]
    n1 = h - m[1] + 1
    n2 = w - m[0] + 1
    sat = cv2.integral(likelihood_map)
    i, j = np.arange(n1), np.arange(n2)
    i, j = np.meshgrid(i, j)
    sat1 = sat[i, j]
    sat2 = np.roll(sat, -m[1], axis=0)
    sat2 = np.roll(sat2, -m[0], axis=1)
    sat2 = sat2[i, j]
    sat3 = np.roll(sat, -m[1], axis=0)
    sat3 = sat3[i, j]
    sat4 = np.roll(sat, -m[0], axis=1)
    sat4 = sat4[i, j]
    center_likelihood = (sat1 + sat2 - sat3 - sat4) / (m[0] * m[1])
    return center_likelihood.T


class Staple(BaseCF):
    def __init__(self, config):
        super(Staple).__init__()
        self.hog_cell_size = config.hog_cell_size
        self.fixed_area = config.fixed_area
        self.n_bins = config.n_bins
        self.interp_factor_pwp = config.interp_factor_pwp
        self.inner_padding = config.inner_padding
        self.output_sigma_factor = config.output_sigma_factor
        self.lambda_ = config.lambda_
        self.interp_factor_cf = config.interp_factor_cf
        self.merge_factor = config.merge_factor
        self.den_per_channel = config.den_per_channel

        self.scale_adaptation = config.scale_adaptation
        self.hog_scale_cell_size = config.hog_scale_cell_size
        self.interp_factor_scale = config.interp_factor_scale
        self.scale_sigma_factor = config.scale_sigma_factor
        self.num_scales = config.num_scales
        self.scale_model_factor = config.scale_model_factor
        self.scale_step = config.scale_step
        self.scale_model_max_area = config.scale_model_max_area
        self.padding = config.padding
        self.use_ca = config.use_ca
        if self.use_ca is True:
            self.lambda_2 = config.lambda_2

    def init(self, first_frame, bbox):
        first_frame = first_frame.astype(np.float32)
        bbox = np.array(bbox).astype(np.int64)
        x, y, w, h = tuple(bbox)
        self._center = (x + w / 2, y + h / 2)
        self.w, self.h = w, h
        self.crop_size = (int(w * (1 + self.padding)), int(h * (1 + self.padding)))
        self.target_sz = (self.w, self.h)

        self.bin_mapping = self.get_bin_mapping(self.n_bins)
        avg_dim = (w + h) / 2
        self.bg_area = (round(w + avg_dim), round(h + avg_dim))
        self.fg_area = (int(round(w - avg_dim * self.inner_padding)), int(round(h - avg_dim * self.inner_padding)))
        self.bg_area = (
        int(min(self.bg_area[0], first_frame.shape[1] - 1)), int(min(self.bg_area[1], first_frame.shape[0] - 1)))

        self.bg_area = (self.bg_area[0] - (self.bg_area[0] - self.target_sz[0]) % 2,
                        self.bg_area[1] - (self.bg_area[1] - self.target_sz[1]) % 2)
        self.fg_area = (self.fg_area[0] + (self.bg_area[0] - self.fg_area[0]) % 2,
                        self.fg_area[1] + (self.bg_area[1] - self.fg_area[1]) % 2)
        self.area_resize_factor = np.sqrt(self.fixed_area / (self.bg_area[0] * self.bg_area[1]))
        self.norm_bg_area = (
        round(self.bg_area[0] * self.area_resize_factor), round(self.bg_area[1] * self.area_resize_factor))

        self.cf_response_size = (int(np.floor(self.norm_bg_area[0] / self.hog_cell_size)),
                                 int(np.floor(self.norm_bg_area[1] / self.hog_cell_size)))
        norm_target_sz_w = 0.75 * self.norm_bg_area[0] - 0.25 * self.norm_bg_area[1]
        norm_target_sz_h = 0.75 * self.norm_bg_area[1] - 0.25 * self.norm_bg_area[0]
        self.norm_target_sz = (round(norm_target_sz_w), round(norm_target_sz_h))
        norm_pad = (int(np.floor((self.norm_bg_area[0] - norm_target_sz_w) / 2)),
                    int(np.floor((self.norm_bg_area[1] - norm_target_sz_h) / 2)))
        radius = min(norm_pad[0], norm_pad[1])
        self.norm_delta_area = (2 * radius + 1, 2 * radius + 1)
        self.norm_pwp_search_area = (self.norm_target_sz[0] + self.norm_delta_area[0] - 1,
                                     self.norm_target_sz[1] + self.norm_delta_area[1] - 1)

        patch_padded = self.get_sub_window(first_frame, self._center, self.norm_bg_area, self.bg_area)
        self.new_pwp_model = True
        self.bg_hist, self.fg_hist = self.update_hist_model(self.new_pwp_model, patch_padded, self.bg_area,
                                                            self.fg_area,
                                                            self.target_sz, self.norm_bg_area, self.n_bins,
                                                            )
        self.new_pwp_model = False
        self._window = cos_window(self.cf_response_size)
        output_sigma = np.sqrt(
            self.norm_target_sz[0] * self.norm_target_sz[1]) * self.output_sigma_factor / self.hog_cell_size
        self.y = gaussian2d_rolled_labels_staple(self.cf_response_size, output_sigma)
        self._init_response_center = np.unravel_index(np.argmax(self.y, axis=None), self.y.shape)
        self.yf = fft2(self.y)

        if self.use_ca:
            # w,h format
            self.offset = [[0, -self.target_sz[1]], [-self.target_sz[0], 0],
                           [0, self.target_sz[1]], [self.target_sz[0], 0]]

        if self.scale_adaptation is True:
            self.scale_factor = 1
            self.base_target_sz = self.target_sz
            self.scale_sigma = np.sqrt(self.num_scales) * self.scale_sigma_factor
            ss = np.arange(1, self.num_scales + 1) - np.ceil(self.num_scales / 2)
            ys = np.exp(-0.5 * (ss ** 2) / (self.scale_sigma ** 2))
            self.ysf = np.fft.fft(ys)
            if self.num_scales % 2 == 0:
                scale_window = np.hanning(self.num_scales + 1)
                self.scale_window = scale_window[1:]
            else:
                self.scale_window = np.hanning(self.num_scales)
            ss = np.arange(1, self.num_scales + 1)
            self.scale_factors = self.scale_step ** (np.ceil(self.num_scales / 2) - ss)

            self.scale_model_factor = 1.
            if (self.w * self.h) > self.scale_model_max_area:
                self.scale_model_factor = np.sqrt(self.scale_model_max_area / (self.w * self.h))

            self.scale_model_sz = (
                int(np.floor(self.w * self.scale_model_factor)), int(np.floor(self.h * self.scale_model_factor)))

            self.current_scale_factor = 1.

            self.min_scale_factor = self.scale_step ** (
                int(np.ceil(np.log(max(5 / self.crop_size[0], 5 / self.crop_size[1])) /
                            np.log(self.scale_step))))
            self.max_scale_factor = self.scale_step ** (
                int(np.floor((np.log(min(first_frame.shape[1] / self.w, first_frame.shape[0] / self.h)) /
                              np.log(self.scale_step)))))
        im_patch_bg = self.get_sub_window(first_frame, self._center, self.norm_bg_area, self.bg_area)
        xt = self.get_feature_map(im_patch_bg, self.hog_cell_size)
        xt = self._window[:, :, None] * xt
        xtf = fft2(xt)
        if self.use_ca:
            sum_kfn=np.zeros_like(xtf)
            for j in range(len(self.offset)):
                im_patch_bgn = self.get_sub_window(first_frame, (
                self._center[0] + self.offset[j][0], self._center[1] + self.offset[j][1]),
                                                   self.norm_bg_area, self.bg_area)
                xtn = self.get_feature_map(im_patch_bgn, self.hog_cell_size)
                xtn = self._window[:, :, None] * xtn
                xtfn = fft2(xtn)
                sum_kfn+=np.conj(xtfn)*xtfn
            self.hf_num = self.yf[:, :, None] * np.conj(xtf)
            self.hf_den = np.conj(xtf) * xtf + self.lambda_ + self.lambda_2 * sum_kfn

        else:
            self.hf_num = np.conj(self.yf)[:, :, None] * xtf / (self.cf_response_size[0] * self.cf_response_size[1])
            self.hf_den = np.conj(xtf) * xtf / (self.cf_response_size[0] * self.cf_response_size[1])

        if self.scale_adaptation is True:
            im_patch_scale = self.get_scale_subwindow(first_frame, self._center,
                                                      self.base_target_sz, self.scale_factor * self.scale_factors,
                                                      self.scale_window, self.scale_model_sz,
                                                      self.hog_scale_cell_size)
            xsf = np.fft.fft(im_patch_scale, axis=1)
            self.sf_den = np.sum(xsf * np.conj(xsf), axis=0)
            self.sf_num = self.ysf * np.conj(xsf)
        self.rect_position_padded = None

    def update(self, current_frame, vis=False):
        im_patch_cf = self.get_sub_window(current_frame, self._center, self.norm_bg_area, self.bg_area)
        pwp_search_area = (round(self.norm_pwp_search_area[0] / self.area_resize_factor),
                           round(self.norm_pwp_search_area[1] / self.area_resize_factor))
        im_patch_pwp = self.get_sub_window(current_frame, self._center, self.norm_pwp_search_area, pwp_search_area)

        xt = self.get_feature_map(im_patch_cf, self.hog_cell_size)
        xt_windowed = self._window[:, :, None] * xt
        xtf = fft2(xt_windowed)
        if self.use_ca is False:
            if self.den_per_channel:
                hf = self.hf_num / (self.hf_den + self.lambda_)
            else:
                hf = self.hf_num / (np.sum(self.hf_den, axis=2) + self.lambda_)[:, :, None]
        else:
            if self.den_per_channel:
                hf = self.hf_num / self.hf_den
            else:
                hf = self.hf_num / (np.sum(self.hf_den, axis=2)[:, :, None])

        if self.use_ca is False:
            response_cf = np.real(ifft2(np.sum(np.conj(hf) * xtf, axis=2)))
        else:
            response_cf = np.real(ifft2(np.sum(hf * xtf, axis=2)))

        response_sz = (self.floor_odd(self.norm_delta_area[0] / self.hog_cell_size),
                       self.floor_odd(self.norm_delta_area[1] / self.hog_cell_size))
        response_cf = crop_filter_response(response_cf, response_sz)
        if self.hog_cell_size > 1:
            if self.use_ca is True:
                #response_cf = self.mex_resize(response_cf, self.norm_delta_area)
                response_cf = cv2.resize(response_cf, self.norm_delta_area, cv2.INTER_NEAREST)
            else:
                response_cf = cv2.resize(response_cf, self.norm_delta_area, cv2.INTER_NEAREST)
        likelihood_map = self.get_colour_map(im_patch_pwp, self.bg_hist, self.fg_hist, self.bin_mapping)

        likelihood_map[np.isnan(likelihood_map)] = 0.
        response_cf[np.isnan(response_cf)] = 0.
        self.norm_target_sz = (int(self.norm_target_sz[0]), int(self.norm_target_sz[1]))
        response_pwp = get_center_likelihood(likelihood_map, self.norm_target_sz)

        response = (1 - self.merge_factor) * response_cf + self.merge_factor * response_pwp
        if vis is True:
            self.score = response
        curr = np.unravel_index(np.argmax(response, axis=None), response.shape)
        center = ((self.norm_delta_area[0] - 1) / 2, (self.norm_delta_area[1] - 1) / 2)
        dy = (curr[0] - center[1]) / self.area_resize_factor
        dx = (curr[1] - center[0]) / self.area_resize_factor
        x_c, y_c = self._center
        x_c += dx
        y_c += dy
        self._center = (x_c, y_c)

        if self.scale_adaptation:
            im_patch_scale = self.get_scale_subwindow(current_frame, self._center, self.base_target_sz,
                                                      self.scale_factor * self.scale_factors, self.scale_window,
                                                      self.scale_model_sz, self.hog_scale_cell_size)
            xsf = np.fft.fft(im_patch_scale, axis=1)
            scale_response = np.real(np.fft.ifft(np.sum(self.sf_num * xsf, axis=0) / (self.sf_den + self.lambda_)))
            recovered_scale = np.argmax(scale_response)
            self.scale_factor = self.scale_factor * self.scale_factors[recovered_scale]
            self.scale_factor = np.clip(self.scale_factor, a_min=self.min_scale_factor, a_max=self.max_scale_factor)
            self.target_sz = (
            round(self.base_target_sz[0] * self.scale_factor), round(self.base_target_sz[1] * self.scale_factor))
            avg_dim = (self.target_sz[0] + self.target_sz[1]) / 2
            bg_area = (round(self.target_sz[0] + avg_dim), round(self.target_sz[1] + avg_dim))
            fg_area = (round(self.target_sz[0] - avg_dim * self.inner_padding),
                       round(self.target_sz[1] - avg_dim * self.inner_padding))
            bg_area = (min(bg_area[0], current_frame.shape[1] - 1), min(bg_area[1], current_frame.shape[0] - 1))
            self.bg_area = (
            bg_area[0] - (bg_area[0] - self.target_sz[0]) % 2, bg_area[1] - (bg_area[1] - self.target_sz[1]) % 2)
            self.fg_area = (
            fg_area[0] + (self.bg_area[0] - fg_area[0]) % 2, fg_area[1] + (self.bg_area[1] - fg_area[1]) % 2)
            self.area_resize_factor = np.sqrt(self.fixed_area / (self.bg_area[0] * self.bg_area[1]))

        im_patch_bg = self.get_sub_window(current_frame, self._center, self.norm_bg_area, self.bg_area)
        xt = self.get_feature_map(im_patch_bg, self.hog_cell_size)
        xt = self._window[:, :, None] * xt
        xtf = fft2(xt)

        if self.use_ca:
            sum_kfn=np.zeros_like(xtf)
            for j in range(len(self.offset)):
                im_patch_bgn = self.get_sub_window(current_frame, (
                    self._center[0] + self.offset[j][0], self._center[1] + self.offset[j][1]),
                                                   self.norm_bg_area, self.bg_area)
                xtn = self.get_feature_map(im_patch_bgn, self.hog_cell_size)
                xtn = self._window[:, :, None] * xtn
                xtfn = fft2(xtn)
                sum_kfn+=np.conj(xtfn)*xtfn
            new_hf_num = self.yf[:, :, None] * np.conj(xtf)
            new_hf_den = np.conj(xtf) * xtf + self.lambda_ + self.lambda_2 * sum_kfn
        else:
            new_hf_num = np.conj(self.yf)[:, :, None] * xtf / (self.cf_response_size[0] * self.cf_response_size[1])
            new_hf_den = (np.conj(xtf) * xtf) / (self.cf_response_size[0] * self.cf_response_size[1])


        self.hf_den = (1 - self.interp_factor_cf) * self.hf_den + self.interp_factor_cf * new_hf_den
        self.hf_num = (1 - self.interp_factor_cf) * self.hf_num + self.interp_factor_cf * new_hf_num
        self.bg_hist, self.fg_hist = self.update_hist_model(self.new_pwp_model,
                                                            im_patch_bg, self.bg_area, self.fg_area, self.target_sz,
                                                            self.norm_bg_area, self.n_bins)
        if self.scale_adaptation:
            im_patch_scale = self.get_scale_subwindow(current_frame, self._center, self.base_target_sz,
                                                      self.scale_factor * self.scale_factors,
                                                      self.scale_window, self.scale_model_sz, self.hog_scale_cell_size)
            xsf = np.fft.fft(im_patch_scale, axis=1)
            new_sf_num = self.ysf * np.conj(xsf)
            new_sf_den = np.sum(xsf * np.conj(xsf), axis=0)
            self.sf_den = (1 - self.interp_factor_scale) * self.sf_den + self.interp_factor_scale * new_sf_den
            self.sf_num = (1 - self.interp_factor_scale) * self.sf_num + self.interp_factor_scale * new_sf_num

        return [self._center[0] - self.target_sz[0] / 2, self._center[1] - self.target_sz[1] / 2,
                self.target_sz[0], self.target_sz[1]]

    def floor_odd(self, x):
        return 2 * int(np.floor((x - 1) / 2)) + 1

    def get_scale_subwindow(self, im, center, base_target_sz, scale_factors, scale_window, scale_model_sz,
                            hog_scale_cell_sz):
        n_scales = len(self.scale_factors)
        out = None
        for s in range(n_scales):
            patch_sz = (int(base_target_sz[0] * scale_factors[s]),
                        int(base_target_sz[1] * scale_factors[s]))
            patch_sz = (max(2, patch_sz[0]), max(2, patch_sz[1]))
            im_patch = cv2.getRectSubPix(im, patch_sz, center)
            im_patch_resized = self.mex_resize(im_patch, scale_model_sz).astype(np.uint8)
            tmp = extract_hog_feature(im_patch_resized, cell_size=hog_scale_cell_sz)
            if out is None:
                out = tmp.flatten() * scale_window[s]
            else:
                out = np.c_[out, tmp.flatten() * scale_window[s]]
        return out

    def get_feature_map(self, im_patch, hog_cell_sz):
        hog_feature = extract_hog_feature(im_patch, cell_size=hog_cell_sz)[:, :, :27]
        if hog_cell_sz > 1:
            im_patch = self.mex_resize(im_patch, (self._window.shape[1], self._window.shape[0])).astype(np.uint8)
        gray = cv2.cvtColor(im_patch, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis] / 255 - 0.5
        return np.concatenate((gray, hog_feature), axis=2)

    def update_hist_model(self, new_model, patch, bg_area, fg_area, target_sz, norm_area,
                          n_bins):
        pad_offset1 = ((bg_area[0] - target_sz[0]) / 2, (bg_area[1] - target_sz[1]) / 2)
        assert pad_offset1[0] == round(pad_offset1[0]) and pad_offset1[1] == round(pad_offset1[1])
        bg_mask = np.ones((int(bg_area[1]), int(bg_area[0])))
        pad_offset1 = (int(max(1, pad_offset1[0])), int(max(1, pad_offset1[1])))
        bg_mask[pad_offset1[1]:-pad_offset1[1], pad_offset1[0]:-pad_offset1[0]] = 0.

        pad_offset2 = ((bg_area[0] - fg_area[0]) / 2, (bg_area[1] - fg_area[1]) / 2)
        assert pad_offset2[0] == round(pad_offset2[0]) and pad_offset2[1] == round(pad_offset2[1])
        fg_mask = np.zeros((int(bg_area[1]), int(bg_area[0])))
        pad_offset2 = (int(max(1, pad_offset2[0])), int(max(1, pad_offset2[1])))
        fg_mask[pad_offset2[1]:-pad_offset2[1], pad_offset2[0]:-pad_offset2[0]] = 1.
        fg_mask = self.mex_resize(fg_mask, norm_area)
        bg_mask = self.mex_resize(bg_mask, norm_area)
        bg_hist_new = self.compute_histogram(patch, bg_mask, n_bins)
        fg_hist_new = self.compute_histogram(patch, fg_mask, n_bins)

        if new_model is not True:
            bg_hist_new = (1 - self.interp_factor_pwp) * self.bg_hist + self.interp_factor_pwp * bg_hist_new
            fg_hist_new = (1 - self.interp_factor_pwp) * self.fg_hist + self.interp_factor_pwp * fg_hist_new
        return bg_hist_new, fg_hist_new

    def get_sub_window(self, img, center, model_sz, scaled_sz=None):
        model_sz = (int(model_sz[0]), int(model_sz[1]))
        if scaled_sz is None:
            sz = model_sz
        else:
            sz = scaled_sz
        sz = (max(int(sz[0]), 2), max(int(sz[1]), 2))
        im_patch = cv2.getRectSubPix(img, sz, center)
        if scaled_sz is not None:
            im_patch = self.mex_resize(im_patch, model_sz).astype(np.uint8)
        return im_patch

    def mex_resize(self, img, sz,method='auto'):
        sz = (int(sz[0]), int(sz[1]))
        src_sz = (img.shape[1], img.shape[0])
        if method=='antialias':
            interpolation=cv2.INTER_AREA
        elif method=='linear':
            interpolation=cv2.INTER_LINEAR
        else:
            if sz[1] > src_sz[1]:
                interpolation = cv2.INTER_LINEAR
            else:
                interpolation = cv2.INTER_AREA
        img = cv2.resize(img, sz, interpolation=interpolation)
        return img

    def compute_histogram(self, patch, mask, n_bins):
        h, w, d = patch.shape
        assert h == mask.shape[0] and w == mask.shape[1]
        mask = mask.astype(np.uint8)
        histogram = cv2.calcHist([patch], [0, 1, 2], mask, [n_bins, n_bins, n_bins],
                                 [0, 256, 0, 256, 0, 256]) / np.count_nonzero(mask)
        return histogram

    def get_colour_map(self, patch, bg_hist, fg_hist, bin_mapping):
        frame_bin = cv2.LUT(patch, bin_mapping).astype(np.int64)
        P_fg = fg_hist[frame_bin[:, :, 0], frame_bin[:, :, 1], frame_bin[:, :, 2]]
        P_bg = bg_hist[frame_bin[:, :, 0], frame_bin[:, :, 1], frame_bin[:, :, 2]]
        P_O = P_fg / (P_fg + P_bg)
        return P_O

    def get_bin_mapping(self, num_bins):
        bin_mapping = np.zeros((256,))
        for i in range(bin_mapping.shape[0]):
            bin_mapping[i] = (np.floor(i / (256 / num_bins)))
        return bin_mapping.astype(np.uint8)
